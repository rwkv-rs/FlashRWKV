#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness-gated benchmark for every canonical FlashRWKV kernel identity.

The B/T matrix and public row schema follow BlinkDL/Albatross
``faster3a_2607``. Unlike the model benchmark, CUDA events here enclose only
one named logical operator. The KDA-derived operator is intentionally measured
as the consecutive K1 and K2 launches.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from flash_rwkv import (
    infer_chunk_bf16_forward,
    infer_chunk_bf16_forward_varlen,
    infer_recurrent_fp16_forward_varlen,
    infer_recurrent_fp32io16_forward_varlen,
    pretrain_recurrent_fp32io16_forward,
)
from flash_rwkv import _extension
from flash_rwkv.benchmark_contract import (
    ALBATROSS_BT_MATRIX,
    ALBATROSS_ROW_FIELDS,
    summarize_samples,
)
from flash_rwkv.providers.fla import (
    infer_recurrent_fp32io16_forward_varlen as fla_infer_recurrent,
)
from flash_rwkv.providers.fla import (
    pretrain_chunk_fp32io16_backward as fla_pretrain_backward,
)
from flash_rwkv.providers.fla import (
    pretrain_chunk_fp32io16_forward as fla_pretrain_forward,
)
from flash_rwkv.registry import KernelSpec, kernel_specs


HEAD_SIZE = 64
CHUNK_SIZE = 16
SOURCE_ROOT = Path(__file__).resolve().parents[1]
TOLERANCE_PATH = SOURCE_ROOT / "tests/fixtures/tolerances-v1.json"
ALBATROSS_SOURCE = {
    "repository": "https://github.com/BlinkDL/Albatross",
    "revision": "63c53f4abf2cd891dd3a18c8f44f5b2cccc8c64b",
    "implementation": "faster3a_2607",
    "adaptation": (
        "21-case B/T matrix and p10/p50/p90 plus p50 token-throughput "
        "statistics; timing narrowed from model execution to one logical "
        "kernel operator"
    ),
}
IDENTITY_LABELS = tuple(
    f"{spec.provider}/{spec.name}" for spec in kernel_specs()
)
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    hidden_size: int
    identities: tuple[str, ...]
    cases: tuple[tuple[int, int], ...]
    fp32io16_io_dtype: str
    warmup_iters: int
    iters: int
    seed: int
    output: Path
    measure: bool


@dataclass(frozen=True, slots=True)
class Inputs:
    batch_size: int
    token_count: int
    num_heads: int
    packed: bool
    numerical_mode: str
    token_dtype_name: str
    state_dtype_name: str
    tensors: tuple[torch.Tensor, ...]
    flat_tensors: tuple[torch.Tensor, ...]
    initial_state: torch.Tensor
    cu_seqlens_cpu: torch.Tensor | None
    cu_seqlens_cuda: torch.Tensor | None
    query_start_loc: torch.Tensor
    state_indices: torch.Tensor
    sequence_chunk_offsets: torch.Tensor
    chunk_token_starts: torch.Tensor
    chunk_token_ends: torch.Tensor

    @property
    def total_tokens(self) -> int:
        return self.batch_size * self.token_count

    @property
    def sequence_lengths(self) -> tuple[int, ...]:
        return (self.token_count,) * self.batch_size


@dataclass(slots=True)
class PreparedOperator:
    """One preallocated operator whose reset is outside the event interval."""

    boundary: str
    configuration: dict[str, object]
    workspace_bytes: int
    reset: Callable[[], None]
    launch: Callable[[], None]
    artifacts: dict[str, object]


def _identity(spec: KernelSpec) -> str:
    return f"{spec.provider}/{spec.name}"


def _parse_cases(values: Sequence[str]) -> tuple[tuple[int, int], ...]:
    known = {f"{batch}x{tokens}": (batch, tokens) for batch, tokens in ALBATROSS_BT_MATRIX}
    cases: list[tuple[int, int]] = []
    for value in values:
        normalized = value.lower().replace("×", "x")
        try:
            case = known[normalized]
        except KeyError as error:
            choices = ", ".join(known)
            raise argparse.ArgumentTypeError(
                f"unknown Albatross case {value!r}; choose from: {choices}"
            ) from error
        if case not in cases:
            cases.append(case)
    return tuple(cases)


def _numerical_mode(spec: KernelSpec) -> str:
    if "_fp32io16_" in spec.name:
        return "fp32io16"
    if "_fp16_" in spec.name:
        return "fp16"
    if "_bf16_" in spec.name:
        return "bf16"
    raise ValueError(f"kernel name has no supported numerical mode: {spec.name}")


def _dtype_policy(
    spec: KernelSpec,
    fp32io16_io_dtype: str,
) -> tuple[str, torch.dtype, str, torch.dtype]:
    mode = _numerical_mode(spec)
    if mode == "fp32io16":
        token_name = fp32io16_io_dtype
        return token_name, DTYPES[token_name], "float32", torch.float32
    if mode == "fp16":
        return "float16", torch.float16, "float16", torch.float16
    return "bfloat16", torch.bfloat16, "bfloat16", torch.bfloat16


def _make_chunk_metadata(
    *,
    batch_size: int,
    token_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_chunk_offsets = [0]
    chunk_token_starts: list[int] = []
    chunk_token_ends: list[int] = []
    for sequence_index in range(batch_size):
        sequence_start = sequence_index * token_count
        sequence_end = sequence_start + token_count
        for chunk_start in range(sequence_start, sequence_end, CHUNK_SIZE):
            chunk_token_starts.append(chunk_start)
            chunk_token_ends.append(min(chunk_start + CHUNK_SIZE, sequence_end))
        sequence_chunk_offsets.append(len(chunk_token_starts))
    return (
        torch.tensor(sequence_chunk_offsets, dtype=torch.int32, device=device),
        torch.tensor(chunk_token_starts, dtype=torch.int32, device=device),
        torch.tensor(chunk_token_ends, dtype=torch.int32, device=device),
    )


def _make_inputs(
    spec: KernelSpec,
    *,
    batch_size: int,
    token_count: int,
    hidden_size: int,
    fp32io16_io_dtype: str,
    seed: int,
) -> Inputs:
    num_heads = hidden_size // HEAD_SIZE
    packed = spec.layouts == ("packed",)
    token_dtype_name, token_dtype, state_dtype_name, state_dtype = _dtype_policy(
        spec,
        fp32io16_io_dtype,
    )
    total_tokens = batch_size * token_count
    flat_shape = (total_tokens, num_heads, HEAD_SIZE)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def normal(scale: float) -> torch.Tensor:
        return scale * torch.randn(
            flat_shape,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )

    direction = functional.normalize(normal(1.0), dim=-1)
    strength = 0.1 * torch.rand(
        flat_shape,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    flat_tensors = (
        normal(0.05).to(token_dtype),
        (
            -0.05
            - 0.15
            * torch.rand(
                flat_shape,
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
        ).to(token_dtype),
        normal(0.05).to(token_dtype),
        normal(0.05).to(token_dtype),
        (-direction).to(token_dtype),
        (direction * strength).to(token_dtype),
    )
    if packed:
        surface_shape = (1, total_tokens, num_heads, HEAD_SIZE)
        offsets = torch.arange(
            0,
            total_tokens + 1,
            token_count,
            dtype=torch.int64,
        )
        cu_seqlens_cpu = offsets
        cu_seqlens_cuda = offsets.to(device="cuda")
    else:
        surface_shape = (batch_size, token_count, num_heads, HEAD_SIZE)
        cu_seqlens_cpu = None
        cu_seqlens_cuda = None
    tensors = tuple(tensor.reshape(surface_shape) for tensor in flat_tensors)

    initial_state = (
        0.02
        * torch.randn(
            batch_size,
            num_heads,
            HEAD_SIZE,
            HEAD_SIZE,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
    ).to(state_dtype)
    query_start_loc = torch.arange(
        0,
        total_tokens + 1,
        token_count,
        dtype=torch.int32,
        device="cuda",
    )
    state_indices = torch.arange(
        batch_size,
        dtype=torch.int32,
        device="cuda",
    )
    chunk_metadata = _make_chunk_metadata(
        batch_size=batch_size,
        token_count=token_count,
        device=torch.device("cuda"),
    )
    return Inputs(
        batch_size=batch_size,
        token_count=token_count,
        num_heads=num_heads,
        packed=packed,
        numerical_mode=_numerical_mode(spec),
        token_dtype_name=token_dtype_name,
        state_dtype_name=state_dtype_name,
        tensors=tensors,
        flat_tensors=flat_tensors,
        initial_state=initial_state,
        cu_seqlens_cpu=cu_seqlens_cpu,
        cu_seqlens_cuda=cu_seqlens_cuda,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        sequence_chunk_offsets=chunk_metadata[0],
        chunk_token_starts=chunk_metadata[1],
        chunk_token_ends=chunk_metadata[2],
    )


def _reference_equal_sequences(
    inputs: Sequence[torch.Tensor],
    initial_state: torch.Tensor,
    *,
    batch_size: int,
    token_count: int,
    packed: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_heads = inputs[0].shape[-2]
    shaped = tuple(
        tensor.reshape(batch_size, token_count, num_heads, HEAD_SIZE).float()
        for tensor in inputs
    )
    r, decay_logits, k, v, a, b = shaped
    log_decay = -0.6065306597126334 * torch.sigmoid(decay_logits)
    state = initial_state.float()
    outputs: list[torch.Tensor] = []
    for token_index in range(token_count):
        previous_state = state
        state_dot_a = torch.einsum(
            "bhk,bhkv->bhv",
            a[:, token_index],
            previous_state,
        )
        state = (
            torch.exp(log_decay[:, token_index]).unsqueeze(-1) * previous_state
            + b[:, token_index].unsqueeze(-1) * state_dot_a.unsqueeze(-2)
            + k[:, token_index].unsqueeze(-1) * v[:, token_index].unsqueeze(-2)
        )
        outputs.append(
            float(scale)
            * torch.einsum("bhk,bhkv->bhv", r[:, token_index], state)
        )
    output = torch.stack(outputs, dim=1)
    if packed:
        output = output.reshape(1, batch_size * token_count, num_heads, HEAD_SIZE)
    return output, state


def _call_public_forward(
    spec: KernelSpec,
    tensors: Sequence[torch.Tensor],
    initial_state: torch.Tensor,
    inputs: Inputs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r, decay_logits, k, v, a, b = tensors
    identity = spec.identity
    if identity in {
        ("rwkv-lm", "pretrain_recurrent_fp32io16_forward"),
        ("rwkv-lm", "pretrain_recurrent_fp32io16_backward"),
    }:
        return pretrain_recurrent_fp32io16_forward(
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            initial_state=initial_state,
            output_final_state=True,
        )
    if identity == ("vllm-rwkv", "infer_recurrent_fp32io16_forward_varlen"):
        assert inputs.cu_seqlens_cpu is not None
        return infer_recurrent_fp32io16_forward_varlen(
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            initial_state=initial_state,
            cu_seqlens=inputs.cu_seqlens_cpu,
        )
    if identity == ("vllm-rwkv", "infer_recurrent_fp16_forward_varlen"):
        assert inputs.cu_seqlens_cpu is not None
        return infer_recurrent_fp16_forward_varlen(
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            initial_state=initial_state,
            cu_seqlens=inputs.cu_seqlens_cpu,
        )
    if identity == ("flashkda-derived", "infer_chunk_bf16_forward"):
        return infer_chunk_bf16_forward(
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            initial_state=initial_state,
        )
    if identity == (
        "flashkda-derived",
        "infer_chunk_bf16_forward_varlen",
    ):
        assert inputs.cu_seqlens_cpu is not None
        return infer_chunk_bf16_forward_varlen(
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            initial_state=initial_state,
            cu_seqlens=inputs.cu_seqlens_cpu,
        )
    if identity in {
        ("fla", "pretrain_chunk_fp32io16_forward"),
        ("fla", "pretrain_chunk_fp32io16_backward"),
    }:
        return fla_pretrain_forward(
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            initial_state=initial_state,
            output_final_state=True,
            safe_gate=True,
        )
    if identity == ("fla", "infer_recurrent_fp32io16_forward_varlen"):
        assert inputs.cu_seqlens_cuda is not None
        return fla_infer_recurrent(
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            initial_state=initial_state,
            cu_seqlens=inputs.cu_seqlens_cuda,
        )
    raise KeyError(f"no public forward adapter for {_identity(spec)}")


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    rmse = difference.square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return float((rmse / baseline).item())


def _max_absolute_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _tolerance_key(spec: KernelSpec) -> str:
    if spec.provider == "rwkv-lm":
        return "fp32io16_pretrain_recurrent"
    if spec.provider == "flashkda-derived":
        return "bf16_kda_chunk"
    if spec.provider == "fla" and spec.name.startswith("pretrain_chunk"):
        return "fp32io16_chunk"
    if "_fp16_" in spec.name:
        return "fp16"
    return "fp32io16_recurrent"


def _forward_error_summary(
    actual_output: torch.Tensor,
    actual_state: torch.Tensor,
    expected_output: torch.Tensor,
    expected_state: torch.Tensor,
) -> dict[str, float | bool]:
    return {
        "output_relative_rmse": _relative_rmse(
            actual_output,
            expected_output,
        ),
        "output_max_absolute_error": _max_absolute_error(
            actual_output,
            expected_output,
        ),
        "final_state_relative_rmse": _relative_rmse(
            actual_state,
            expected_state,
        ),
        "final_state_max_absolute_error": _max_absolute_error(
            actual_state,
            expected_state,
        ),
        "finite": bool(
            torch.isfinite(actual_output).all().item()
            and torch.isfinite(actual_state).all().item()
        ),
    }


def _backward_correctness(
    spec: KernelSpec,
    inputs: Inputs,
    limits: dict[str, float],
    *,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    actual_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in inputs.tensors
    )
    actual_initial_state = (
        inputs.initial_state.detach().clone().requires_grad_(True)
    )
    initial_snapshot = actual_initial_state.detach().clone()
    actual_output, actual_state = _call_public_forward(
        spec,
        actual_inputs,
        actual_initial_state,
        inputs,
    )
    if actual_state is None:
        raise RuntimeError("training forward did not return final state")
    grad_output = torch.randn(
        actual_output.shape,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).to(actual_output.dtype)
    grad_final_state = torch.randn(
        actual_state.shape,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    actual_gradient_inputs = (*actual_inputs, actual_initial_state)
    if spec.provider == "fla":
        actual_gradients = fla_pretrain_backward(
            actual_output,
            actual_state,
            actual_gradient_inputs,
            grad_output=grad_output,
            grad_final_state=grad_final_state,
        )
    else:
        actual_gradients = torch.autograd.grad(
            outputs=(actual_output, actual_state),
            inputs=actual_gradient_inputs,
            grad_outputs=(grad_output, grad_final_state),
        )

    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in inputs.tensors
    )
    reference_initial_state = (
        inputs.initial_state.detach().float().clone().requires_grad_(True)
    )
    expected_output, expected_state = _reference_equal_sequences(
        reference_inputs,
        reference_initial_state,
        batch_size=inputs.batch_size,
        token_count=inputs.token_count,
        packed=inputs.packed,
        scale=1.0,
    )
    expected_gradients = torch.autograd.grad(
        outputs=(expected_output, expected_state),
        inputs=(*reference_inputs, reference_initial_state),
        grad_outputs=(grad_output.float(), grad_final_state.float()),
    )
    torch.cuda.synchronize()

    forward_errors = _forward_error_summary(
        actual_output,
        actual_state,
        expected_output,
        expected_state,
    )
    gradient_names = (
        "r",
        "decay_logits",
        "k",
        "v",
        "a",
        "b",
        "initial_state",
    )
    gradient_errors = {
        name: {
            "relative_rmse": _relative_rmse(actual, expected),
            "max_absolute_error": _max_absolute_error(actual, expected),
            "finite": bool(torch.isfinite(actual).all().item()),
        }
        for name, actual, expected in zip(
            gradient_names,
            actual_gradients,
            expected_gradients,
            strict=True,
        )
    }
    max_gradient_error = max(
        float(error["relative_rmse"]) for error in gradient_errors.values()
    )
    initial_state_preserved = bool(
        torch.equal(actual_initial_state.detach(), initial_snapshot)
    )
    passed = bool(
        forward_errors["finite"]
        and initial_state_preserved
        and float(forward_errors["output_relative_rmse"])
        <= limits["output_relative_rmse"]
        and float(forward_errors["final_state_relative_rmse"])
        <= limits["state_relative_rmse"]
        and max_gradient_error <= limits["gradient_relative_rmse"]
        and all(bool(error["finite"]) for error in gradient_errors.values())
    )
    return {
        "passed": passed,
        **forward_errors,
        "gradient_relative_rmse_max": max_gradient_error,
        "gradient_relative_rmse_limit": limits["gradient_relative_rmse"],
        "gradient_errors": gradient_errors,
        "output_relative_rmse_limit": limits["output_relative_rmse"],
        "final_state_relative_rmse_limit": limits["state_relative_rmse"],
        "initial_state_preserved": initial_state_preserved,
    }


def _forward_correctness(
    spec: KernelSpec,
    inputs: Inputs,
    limits: dict[str, float],
) -> dict[str, object]:
    initial_snapshot = inputs.initial_state.clone()
    with torch.no_grad():
        actual_output, actual_state = _call_public_forward(
            spec,
            inputs.tensors,
            inputs.initial_state,
            inputs,
        )
        expected_output, expected_state = _reference_equal_sequences(
            inputs.tensors,
            inputs.initial_state,
            batch_size=inputs.batch_size,
            token_count=inputs.token_count,
            packed=inputs.packed,
            scale=1.0,
        )
    torch.cuda.synchronize()
    if actual_state is None:
        raise RuntimeError("forward did not return final state")
    errors = _forward_error_summary(
        actual_output,
        actual_state,
        expected_output,
        expected_state,
    )
    initial_state_preserved = bool(
        torch.equal(inputs.initial_state, initial_snapshot)
    )
    passed = bool(
        errors["finite"]
        and initial_state_preserved
        and float(errors["output_relative_rmse"])
        <= limits["output_relative_rmse"]
        and float(errors["final_state_relative_rmse"])
        <= limits["state_relative_rmse"]
    )
    return {
        "passed": passed,
        **errors,
        "output_relative_rmse_limit": limits["output_relative_rmse"],
        "final_state_relative_rmse_limit": limits["state_relative_rmse"],
        "initial_state_preserved": initial_state_preserved,
    }


def _correctness(
    spec: KernelSpec,
    inputs: Inputs,
    tolerances: dict[str, object],
    *,
    seed: int,
) -> dict[str, object]:
    key = _tolerance_key(spec)
    limits = tolerances[key]
    if not isinstance(limits, dict):
        raise TypeError(f"invalid tolerance entry: {key}")
    if spec.name.endswith("_backward"):
        return _backward_correctness(spec, inputs, limits, seed=seed)
    return _forward_correctness(spec, inputs, limits)


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _prepare_rwkv_pretrain_forward(inputs: Inputs) -> PreparedOperator:
    state = torch.empty_like(inputs.initial_state, dtype=torch.float32)
    output = torch.empty_like(inputs.flat_tensors[3])
    num_chunks = inputs.chunk_token_starts.numel()
    boundary = torch.empty(
        num_chunks,
        inputs.num_heads,
        HEAD_SIZE,
        HEAD_SIZE,
        dtype=torch.float32,
        device="cuda",
    )
    state_dot_a = torch.empty(
        inputs.total_tokens,
        inputs.num_heads,
        HEAD_SIZE,
        dtype=torch.float32,
        device="cuda",
    )

    def reset() -> None:
        state.copy_(inputs.initial_state)

    def launch() -> None:
        _extension.pretrain_recurrent_fp32io16_from_decay_logits_forward(
            inputs.sequence_chunk_offsets,
            inputs.chunk_token_starts,
            inputs.chunk_token_ends,
            state,
            *inputs.flat_tensors,
            output,
            boundary,
            state_dot_a,
            1.0,
        )

    return PreparedOperator(
        boundary=(
            "one native pretrain_recurrent_fp32io16_forward launch; "
            "FP32 state reset is synchronized before the start event"
        ),
        configuration={
            "chunk_size": CHUNK_SIZE,
            "state_accumulation": "float32",
            "output_dtype": inputs.token_dtype_name,
        },
        workspace_bytes=_tensor_bytes(state, output, boundary, state_dot_a),
        reset=reset,
        launch=launch,
        artifacts={
            "state": state,
            "output": output,
            "boundary": boundary,
            "state_dot_a": state_dot_a,
        },
    )


def _prepare_rwkv_pretrain_backward(inputs: Inputs) -> PreparedOperator:
    forward = _prepare_rwkv_pretrain_forward(inputs)
    forward.reset()
    torch.cuda.synchronize()
    forward.launch()
    torch.cuda.synchronize()
    generator = torch.Generator(device="cuda").manual_seed(
        inputs.batch_size * 1009 + inputs.token_count * 9173
    )
    grad_output = torch.randn(
        inputs.flat_tensors[3].shape,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).to(inputs.flat_tensors[3].dtype)
    grad_final_state = torch.randn(
        inputs.initial_state.shape,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    gradients = tuple(torch.empty_like(tensor) for tensor in inputs.flat_tensors)
    grad_initial_state = torch.empty_like(inputs.initial_state, dtype=torch.float32)

    def launch() -> None:
        _extension.pretrain_recurrent_fp32io16_from_decay_logits_backward(
            inputs.sequence_chunk_offsets,
            inputs.chunk_token_starts,
            inputs.chunk_token_ends,
            forward.artifacts["state"],
            *inputs.flat_tensors,
            forward.artifacts["state_dot_a"],
            grad_output,
            grad_final_state,
            forward.artifacts["boundary"],
            *gradients,
            grad_initial_state,
            1.0,
        )

    return PreparedOperator(
        boundary=(
            "one native pretrain_recurrent_fp32io16_backward launch; "
            "forward context and upstream gradients are prepared outside events"
        ),
        configuration={
            "chunk_size": CHUNK_SIZE,
            "forward_context": "precomputed once outside timed samples",
            "gradient_accumulation": "float32 registers",
            "gradient_output_dtype": inputs.token_dtype_name,
        },
        workspace_bytes=(
            forward.workspace_bytes
            + _tensor_bytes(
                grad_output,
                grad_final_state,
                *gradients,
                grad_initial_state,
            )
        ),
        reset=lambda: None,
        launch=launch,
        artifacts={
            "gradients": gradients,
            "grad_initial_state": grad_initial_state,
            "forward": forward,
        },
    )


def _prepare_vllm_recurrent(
    inputs: Inputs,
    *,
    fp16_state: bool,
) -> PreparedOperator:
    state_dtype = torch.float16 if fp16_state else torch.float32
    state = torch.empty_like(inputs.initial_state, dtype=state_dtype)
    output = torch.empty_like(inputs.flat_tensors[3])
    extension_op = (
        _extension.recurrent_fp16_from_decay_logits
        if fp16_state
        else _extension.recurrent_fp32_from_decay_logits
    )
    validated_metadata = _extension.prepare_recurrent_metadata(
        inputs.query_start_loc,
        inputs.state_indices,
        total_tokens=inputs.total_tokens,
        state_pool_size=inputs.initial_state.shape[0],
    )

    def reset() -> None:
        state.copy_(inputs.initial_state)

    def launch() -> None:
        extension_op(
            inputs.query_start_loc,
            inputs.state_indices,
            state,
            *inputs.flat_tensors,
            output,
            1.0,
            validated_metadata=validated_metadata,
        )

    return PreparedOperator(
        boundary=(
            "one native fused raw decay-logit recurrent "
            "launch; packed metadata is prebuilt and state reset is synchronized "
            "before the start event"
        ),
        configuration={
            "packed_equal_sequences": True,
            "state_dtype": "float16" if fp16_state else "float32",
            "accumulation": "float16" if fp16_state else "float32",
        },
        workspace_bytes=_tensor_bytes(state, output),
        reset=reset,
        launch=launch,
        artifacts={"state": state, "output": output},
    )


def _prepare_kda_chunk(inputs: Inputs) -> PreparedOperator:
    state = torch.empty_like(inputs.initial_state, dtype=torch.bfloat16)
    output = torch.empty_like(inputs.flat_tensors[3])
    chunk_shape = (
        inputs.chunk_token_starts.numel(),
        inputs.num_heads,
        HEAD_SIZE,
        HEAD_SIZE,
    )
    chunk_transform = torch.empty(
        chunk_shape,
        dtype=torch.float32,
        device="cuda",
    )
    chunk_bias = torch.empty_like(chunk_transform)
    token_transform = torch.empty(
        inputs.total_tokens,
        inputs.num_heads,
        HEAD_SIZE,
        dtype=torch.float32,
        device="cuda",
    )
    token_bias = torch.empty_like(token_transform)

    def reset() -> None:
        state.copy_(inputs.initial_state)

    def launch() -> None:
        _extension.infer_chunk_bf16_forward_k1_prepare_from_decay_logits(
            inputs.chunk_token_starts,
            inputs.chunk_token_ends,
            *inputs.flat_tensors,
            chunk_transform,
            chunk_bias,
            token_transform,
            token_bias,
            1.0,
        )
        _extension.infer_chunk_bf16_forward_k2_recurrence(
            inputs.sequence_chunk_offsets,
            inputs.chunk_token_starts,
            inputs.chunk_token_ends,
            state,
            output,
            chunk_transform,
            chunk_bias,
            token_transform,
            token_bias,
        )

    return PreparedOperator(
        boundary=(
            "one logical raw-decay infer_chunk_bf16_forward operator: "
            "consecutive K1 prepare and K2 recurrence native launches; BF16 "
            "state reset is synchronized before the start event"
        ),
        configuration={
            "chunk_size": CHUNK_SIZE,
            "stages": ["K1 fused raw-decay prepare", "K2 recurrence"],
            "global_state_dtype": "bfloat16",
            "workspace_dtype": "float32",
            "accumulation": "float32",
        },
        workspace_bytes=_tensor_bytes(
            state,
            output,
            chunk_transform,
            chunk_bias,
            token_transform,
            token_bias,
        ),
        reset=reset,
        launch=launch,
        artifacts={"state": state, "output": output},
    )


def _prepare_fla_forward(inputs: Inputs) -> PreparedOperator:
    token_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in inputs.tensors
    )
    initial_state = inputs.initial_state.detach().clone().requires_grad_(True)
    result: dict[str, object] = {}

    def reset() -> None:
        result.clear()

    def launch() -> None:
        result["value"] = fla_pretrain_forward(
            *token_inputs,
            initial_state=initial_state,
            output_final_state=True,
            safe_gate=True,
        )

    return PreparedOperator(
        boundary=(
            "one FLA raw-decay chunk_rwkv7 forward logical operator call; "
            "input graph leaves and state are prepared outside events"
        ),
        configuration={
            "implementation": "fla.ops.rwkv7.chunk_rwkv7",
            "decay_input": "raw logits",
            "chunk_size": CHUNK_SIZE,
            "safe_gate": True,
            "autograd_enabled": True,
            "state_dtype": "float32",
        },
        workspace_bytes=_tensor_bytes(initial_state, *token_inputs),
        reset=reset,
        launch=launch,
        artifacts={"result": result},
    )


def _prepare_fla_backward(inputs: Inputs) -> PreparedOperator:
    token_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in inputs.tensors
    )
    initial_state = inputs.initial_state.detach().clone().requires_grad_(True)
    output, final_state = fla_pretrain_forward(
        *token_inputs,
        initial_state=initial_state,
        output_final_state=True,
        safe_gate=True,
    )
    if final_state is None:
        raise RuntimeError("FLA forward did not produce final state")
    generator = torch.Generator(device="cuda").manual_seed(
        inputs.batch_size * 1009 + inputs.token_count * 9173
    )
    grad_output = torch.randn(
        output.shape,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).to(output.dtype)
    grad_final_state = torch.randn(
        final_state.shape,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    result: dict[str, object] = {}

    def reset() -> None:
        result.clear()

    def launch() -> None:
        result["value"] = fla_pretrain_backward(
            output,
            final_state,
            (*token_inputs, initial_state),
            grad_output=grad_output,
            grad_final_state=grad_final_state,
            retain_graph=True,
        )

    return PreparedOperator(
        boundary=(
            "one FLA raw-decay chunk_rwkv7 autograd backward logical operator "
            "call; forward graph and upstream gradients are prepared outside "
            "events"
        ),
        configuration={
            "implementation": "torch.autograd.grad over FLA chunk_rwkv7",
            "decay_input": "raw logits",
            "chunk_size": CHUNK_SIZE,
            "safe_gate": True,
            "forward_context": "precomputed once outside timed samples",
            "retain_graph": True,
        },
        workspace_bytes=_tensor_bytes(
            initial_state,
            output,
            final_state,
            grad_output,
            grad_final_state,
            *token_inputs,
        ),
        reset=reset,
        launch=launch,
        artifacts={"result": result},
    )


def _prepare_fla_recurrent(inputs: Inputs) -> PreparedOperator:
    if inputs.cu_seqlens_cuda is None:
        raise ValueError("FLA recurrent benchmark requires packed metadata")
    result: dict[str, object] = {}

    def reset() -> None:
        result.clear()

    def launch() -> None:
        result["value"] = fla_infer_recurrent(
            *inputs.tensors,
            initial_state=inputs.initial_state,
            cu_seqlens=inputs.cu_seqlens_cuda,
        )

    return PreparedOperator(
        boundary=(
            "one FLA recurrent_rwkv7 raw-decay logical operator call; packed "
            "metadata and functional initial state are prepared outside events"
        ),
        configuration={
            "implementation": "fla.ops.rwkv7.recurrent_rwkv7",
            "decay_input": "raw logits",
            "packed_equal_sequences": True,
            "state_dtype": "float32",
            "accumulation": "float32",
        },
        workspace_bytes=_tensor_bytes(inputs.initial_state, *inputs.tensors),
        reset=reset,
        launch=launch,
        artifacts={"result": result},
    )


def _prepare_operator(spec: KernelSpec, inputs: Inputs) -> PreparedOperator:
    identity = spec.identity
    if identity == ("rwkv-lm", "pretrain_recurrent_fp32io16_forward"):
        return _prepare_rwkv_pretrain_forward(inputs)
    if identity == ("rwkv-lm", "pretrain_recurrent_fp32io16_backward"):
        return _prepare_rwkv_pretrain_backward(inputs)
    if identity == ("vllm-rwkv", "infer_recurrent_fp32io16_forward_varlen"):
        return _prepare_vllm_recurrent(inputs, fp16_state=False)
    if identity == ("vllm-rwkv", "infer_recurrent_fp16_forward_varlen"):
        return _prepare_vllm_recurrent(inputs, fp16_state=True)
    if spec.provider == "flashkda-derived":
        return _prepare_kda_chunk(inputs)
    if identity == ("fla", "pretrain_chunk_fp32io16_forward"):
        return _prepare_fla_forward(inputs)
    if identity == ("fla", "pretrain_chunk_fp32io16_backward"):
        return _prepare_fla_backward(inputs)
    if identity == ("fla", "infer_recurrent_fp32io16_forward_varlen"):
        return _prepare_fla_recurrent(inputs)
    raise KeyError(f"no benchmark adapter for {_identity(spec)}")


def _measure(
    operator: PreparedOperator,
    *,
    warmup_iters: int,
    iters: int,
) -> list[float]:
    for _ in range(warmup_iters):
        operator.reset()
        torch.cuda.synchronize()
        operator.launch()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        operator.reset()
        torch.cuda.synchronize()
        start.record()
        operator.launch()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _run_case(
    spec: KernelSpec,
    *,
    batch_size: int,
    token_count: int,
    config: BenchmarkConfig,
    tolerances: dict[str, object],
    seed: int,
) -> dict[str, object]:
    label = f"B{batch_size}T{token_count}"
    inputs = _make_inputs(
        spec,
        batch_size=batch_size,
        token_count=token_count,
        hidden_size=config.hidden_size,
        fp32io16_io_dtype=config.fp32io16_io_dtype,
        seed=seed,
    )
    case: dict[str, object] = {
        "label": label,
        "B": batch_size,
        "T": token_count,
        "layout": "packed" if inputs.packed else "fixed",
        "sequence_lengths": list(inputs.sequence_lengths),
        "total_tokens": inputs.total_tokens,
        "shape": list(inputs.tensors[0].shape),
        "precision": {
            "numerical_mode": inputs.numerical_mode,
            "token_dtype": inputs.token_dtype_name,
            "state_dtype": inputs.state_dtype_name,
        },
    }
    try:
        correctness = _correctness(
            spec,
            inputs,
            tolerances,
            seed=seed + 1_000_003,
        )
        case["correctness"] = correctness
        if not correctness["passed"]:
            case["measurement"] = {
                "valid": False,
                "reason": "correctness gate failed",
                "row": None,
                "raw_samples_ms": [],
            }
            return case
        operator = _prepare_operator(spec, inputs)
        case["measurement_boundary"] = operator.boundary
        case["operator_configuration"] = operator.configuration
        case["workspace_bytes"] = operator.workspace_bytes
        if not config.measure:
            case["measurement"] = {
                "valid": True,
                "reason": "correctness-only run",
                "row": None,
                "raw_samples_ms": [],
            }
            return case
        samples = _measure(
            operator,
            warmup_iters=config.warmup_iters,
            iters=config.iters,
        )
        row = summarize_samples(
            label=label,
            batch_size=batch_size,
            token_count=token_count,
            samples_ms=samples,
        )
        case["measurement"] = {
            "valid": True,
            "reason": None,
            "row": row.as_dict(),
            "raw_samples_ms": samples,
        }
    except Exception as error:
        case["measurement"] = {
            "valid": False,
            "reason": f"{type(error).__name__}: {error}",
            "row": None,
            "raw_samples_ms": [],
        }
    return case


def _git_output(repository: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _repository_metadata(repository: Path) -> dict[str, object]:
    status = _git_output(
        repository,
        "status",
        "--short",
        "--ignore-submodules=none",
    )
    return {
        "path": str(repository),
        "revision": _git_output(repository, "rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_manifest() -> dict[str, object]:
    candidates = [
        *SOURCE_ROOT.joinpath("flash_rwkv").rglob("*.py"),
        *SOURCE_ROOT.joinpath("csrc").rglob("*.cpp"),
        *SOURCE_ROOT.joinpath("csrc").rglob("*.cu"),
        *SOURCE_ROOT.joinpath("csrc").rglob("*.h"),
        *SOURCE_ROOT.joinpath("benchmarks").glob("*.py"),
        SOURCE_ROOT / "setup.py",
        SOURCE_ROOT / "pyproject.toml",
        SOURCE_ROOT / "README.md",
        SOURCE_ROOT / "NOTICE",
        TOLERANCE_PATH,
    ]
    paths = sorted(
        {
            path.resolve()
            for path in candidates
            if path.is_file()
        },
        key=lambda path: path.relative_to(SOURCE_ROOT).as_posix(),
    )
    entries = [
        {
            "path": path.relative_to(SOURCE_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry["path"]).encode())
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode())
        digest.update(b"\0")
    return {
        "algorithm": "sha256(path NUL file_sha256 NUL)",
        "file_count": len(entries),
        "sha256": digest.hexdigest(),
        "files": entries,
    }


def _source_metadata() -> dict[str, object]:
    extension = _extension._load_extension()
    extension_path = Path(extension.__file__).resolve()
    workspace = SOURCE_ROOT.parents[2]
    manifest_path = workspace / ".helicopter-dev/source-revisions.json"
    manifest: object = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fla_root = SOURCE_ROOT.parent / "fla-rwkv"
    try:
        fla_version = importlib.metadata.version("flash-linear-attention")
    except importlib.metadata.PackageNotFoundError:
        fla_version = None
    return {
        "flash_rwkv": _repository_metadata(SOURCE_ROOT),
        "fla": {
            **_repository_metadata(fla_root),
            "package_version": fla_version,
        },
        "synchronized_source_manifest": {
            "path": str(manifest_path),
            "available": manifest_path.is_file(),
            "payload": manifest,
        },
        "extension": {
            "path": str(extension_path),
            "sha256": _sha256(extension_path),
        },
        "flash_rwkv_source_tree": _source_tree_manifest(),
        "tolerance_fixture": {
            "path": str(TOLERANCE_PATH),
            "sha256": _sha256(TOLERANCE_PATH),
        },
    }


def _hardware_metadata() -> dict[str, object]:
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "device_index": device_index,
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _safe_filename(identity: str) -> str:
    return identity.replace("/", "__").replace("-", "_")


def _write_csv_files(
    payload: dict[str, object],
    output_path: Path,
) -> list[str]:
    csv_directory = output_path.with_suffix("").with_name(
        f"{output_path.stem}-csv"
    )
    csv_directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for kernel in payload["kernels"]:
        identity = str(kernel["identity"])
        path = csv_directory / f"{_safe_filename(identity)}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=ALBATROSS_ROW_FIELDS)
            writer.writeheader()
            for case in kernel["cases"]:
                row = case["measurement"]["row"]
                if row is not None:
                    writer.writerow(row)
        paths.append(str(path))
    return paths


def _write_markdown_report(
    payload: dict[str, object],
    output_path: Path,
) -> str:
    report_path = output_path.with_suffix(".md")
    hardware = payload["hardware"]
    lines = [
        "# FlashRWKV canonical kernel benchmark",
        "",
        (
            f"- GPU: `{hardware['device_name']}` "
            f"(compute capability {hardware['compute_capability']})"
        ),
        (
            f"- Hidden size: `{payload['hidden_size']}`; "
            f"warmup: `{payload['warmup_iters']}`; "
            f"iters: `{payload['iters']}`"
        ),
        (
            "- Throughput: "
            "`tok_s_p50 = B * T * 1000 / p50_ms`"
        ),
        "",
    ]
    for kernel in payload["kernels"]:
        lines.extend(
            [
                f"## `{kernel['identity']}`",
                "",
                "| label | B | T | iters | p10_ms | p50_ms | p90_ms | tok_s_p50 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for case in kernel["cases"]:
            measurement = case["measurement"]
            row = measurement["row"]
            if row is None:
                lines.append(
                    f"| {case['label']} | {case['B']} | {case['T']} | "
                    f"0 | — | — | — | invalid: {measurement['reason']} |"
                )
                continue
            lines.append(
                f"| {row['label']} | {row['B']} | {row['T']} | "
                f"{row['iters']} | {row['p10_ms']:.6f} | "
                f"{row['p50_ms']:.6f} | {row['p90_ms']:.6f} | "
                f"{row['tok_s_p50']:.3f} |"
            )
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def run(config: BenchmarkConfig) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if config.hidden_size <= 0 or config.hidden_size % HEAD_SIZE:
        raise ValueError("hidden_size must be a positive multiple of 64")
    if config.warmup_iters < 0 or config.iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive")
    if not config.cases:
        raise ValueError("at least one B/T case is required")

    spec_by_identity = {_identity(spec): spec for spec in kernel_specs()}
    tolerances = json.loads(TOLERANCE_PATH.read_text(encoding="utf-8"))
    kernels: list[dict[str, object]] = []
    for identity_index, identity in enumerate(config.identities):
        spec = spec_by_identity[identity]
        cases: list[dict[str, object]] = []
        for case_index, (batch_size, token_count) in enumerate(config.cases):
            case_seed = (
                config.seed
                + batch_size * 1_000_003
                + token_count * 10_007
            )
            cases.append(
                _run_case(
                    spec,
                    batch_size=batch_size,
                    token_count=token_count,
                    config=config,
                    tolerances=tolerances,
                    seed=case_seed,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
        valid_measurements = sum(
            bool(case["measurement"]["valid"]) for case in cases
        )
        measured_rows = sum(
            case["measurement"]["row"] is not None for case in cases
        )
        kernels.append(
            {
                "identity": identity,
                "registry": asdict(spec),
                "case_count": len(cases),
                "valid_measurement_count": valid_measurements,
                "measured_row_count": measured_rows,
                "matrix_complete": tuple(config.cases) == ALBATROSS_BT_MATRIX,
                "cases": cases,
            }
        )

    all_valid = all(
        case["measurement"]["valid"]
        for kernel in kernels
        for case in kernel["cases"]
    )
    all_measured = config.measure and all(
        case["measurement"]["row"] is not None
        for kernel in kernels
        for case in kernel["cases"]
    )
    payload: dict[str, object] = {
        "schema_version": 2,
        "benchmark": "flash_rwkv_canonical_kernels",
        "baseline_source": ALBATROSS_SOURCE,
        "row_fields": list(ALBATROSS_ROW_FIELDS),
        "bt_matrix": [
            {"B": batch_size, "T": token_count}
            for batch_size, token_count in ALBATROSS_BT_MATRIX
        ],
        "selected_cases": [
            {"B": batch_size, "T": token_count}
            for batch_size, token_count in config.cases
        ],
        "matrix_complete": tuple(config.cases) == ALBATROSS_BT_MATRIX,
        "throughput_definition": "B * T * 1000 / p50_ms",
        "measurement_boundary": (
            "CUDA events enclose exactly one named logical operator; input "
            "generation, compilation/autotune, correctness, metadata, state "
            "clone/reset, logging, and serialization are outside events; KDA "
            "logical operators include consecutive K1 and K2 launches"
        ),
        "synchronization_policy": (
            "each reset is followed by explicit CUDA synchronization before "
            "the start event; every sample synchronizes its end event"
        ),
        "hidden_size": config.hidden_size,
        "head_size": HEAD_SIZE,
        "fp32io16_io_dtype": config.fp32io16_io_dtype,
        "warmup_iters": config.warmup_iters,
        "iters": config.iters,
        "seed": config.seed,
        "measurement_enabled": config.measure,
        "hardware": _hardware_metadata(),
        "runtime": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
        },
        "source": _source_metadata(),
        "kernel_count": len(kernels),
        "case_count": sum(len(kernel["cases"]) for kernel in kernels),
        "all_cases_valid": all_valid,
        "all_cases_measured": all_measured,
        "kernels": kernels,
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["csv_files"] = _write_csv_files(payload, config.output)
    payload["markdown_report"] = _write_markdown_report(payload, config.output)
    config.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _default_output() -> Path:
    run_log_directory = os.environ.get("REMOTE_RUN_LOG_DIR")
    if run_log_directory:
        return Path(run_log_directory) / "flash-rwkv-kernels.json"
    return Path("flash-rwkv-kernels.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--identities",
        nargs="+",
        choices=IDENTITY_LABELS,
        default=list(IDENTITY_LABELS),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            f"{batch_size}x{token_count}"
            for batch_size, token_count in ALBATROSS_BT_MATRIX
        ],
    )
    parser.add_argument(
        "--fp32io16-io-dtype",
        choices=tuple(DTYPES),
        default="bfloat16",
    )
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, default=_default_output())
    parser.add_argument("--correctness-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    try:
        cases = _parse_cases(arguments.cases)
    except argparse.ArgumentTypeError as error:
        raise SystemExit(str(error)) from error
    config = BenchmarkConfig(
        hidden_size=arguments.hidden_size,
        identities=tuple(arguments.identities),
        cases=cases,
        fp32io16_io_dtype=arguments.fp32io16_io_dtype,
        warmup_iters=arguments.warmup_iters,
        iters=arguments.iters,
        seed=arguments.seed,
        output=arguments.output,
        measure=not arguments.correctness_only,
    )
    payload = run(config)
    print(
        json.dumps(
            {
                "output": str(config.output),
                "markdown_report": payload["markdown_report"],
                "kernel_count": payload["kernel_count"],
                "case_count": payload["case_count"],
                "matrix_complete": payload["matrix_complete"],
                "all_cases_valid": payload["all_cases_valid"],
                "all_cases_measured": payload["all_cases_measured"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not payload["all_cases_valid"]:
        raise SystemExit(1)
    if config.measure and not payload["all_cases_measured"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
