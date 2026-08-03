# SPDX-License-Identifier: MIT

"""FLA adapters named by the canonical FlashRWKV execution contract."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Sequence
from functools import cache

import torch


@cache
def _raw_decay_operator(
    name: str,
    *,
    required_parameters: tuple[str, ...],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
    """Load an FLA operator only when its producer boundary is raw decay."""

    try:
        module = importlib.import_module("fla.ops.rwkv7")
        operator = getattr(module, name)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "FlashRWKV requires a matching FLA raw-decay API; "
            f"fla.ops.rwkv7.{name} is unavailable"
        ) from error
    try:
        parameters = tuple(inspect.signature(operator).parameters)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "FlashRWKV cannot verify the installed FLA raw-decay ABI for "
            f"fla.ops.rwkv7.{name}"
        ) from error
    missing = tuple(
        parameter
        for parameter in required_parameters
        if parameter not in parameters
    )
    if len(parameters) < 2 or parameters[1] != "decay_logits" or missing:
        details = (
            f"second parameter is {parameters[1]!r}"
            if len(parameters) >= 2
            else "operator has fewer than two parameters"
        )
        if missing:
            details += f"; missing parameters: {', '.join(missing)}"
        raise RuntimeError(
            "installed FLA is incompatible with FlashRWKV's raw-decay ABI: "
            f"fla.ops.rwkv7.{name} {details}"
        )
    return operator


def pretrain_chunk_fp32io16_forward(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    safe_gate: bool = True,
    decay_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run FLA's raw-decay fixed-length chunk/autograd implementation."""

    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError(
            "FLA pretrain_chunk_fp32io16_forward requires FP32 initial_state"
        )
    chunk_rwkv7 = _raw_decay_operator(
        "chunk_rwkv7",
        required_parameters=("decay_bias",),
    )
    return chunk_rwkv7(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        safe_gate=safe_gate,
        chunk_size=16,
        disable_recompute=False,
        cp_context=None,
        decay_bias=decay_bias,
    )


def pretrain_chunk_fp32io16_backward(
    output: torch.Tensor,
    final_state: torch.Tensor,
    inputs: Sequence[torch.Tensor],
    *,
    grad_output: torch.Tensor,
    grad_final_state: torch.Tensor,
    retain_graph: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    """Run the FLA autograd backward for a prepared forward graph."""

    if len(inputs) != 7:
        raise ValueError("FLA backward requires six token inputs and initial_state")
    return torch.autograd.grad(
        outputs=(output, final_state),
        inputs=tuple(inputs),
        grad_outputs=(grad_output, grad_final_state),
        retain_graph=retain_graph,
        allow_unused=False,
    )


def infer_recurrent_fp32io16_forward_varlen(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    scale: float = 1.0,
    output_final_state: bool = True,
    state_indices: torch.Tensor | None = None,
    decay_bias: torch.Tensor | None = None,
    validated_metadata: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run FLA's packed raw-decay recurrent forward with FP32 state."""

    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError(
            "FLA infer_recurrent_fp32io16_forward_varlen requires "
            "FP32 initial_state"
        )
    recurrent_rwkv7 = _raw_decay_operator(
        "recurrent_rwkv7",
        required_parameters=(
            "state_indices",
            "mode",
            "decay_bias",
            "elapsed_t",
            "validated_metadata",
        ),
    )
    return recurrent_rwkv7(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=None,
        state_indices=state_indices,
        mode="fp32io16",
        safe_gate=False,
        chunk_size=None,
        disable_recompute=False,
        cp_context=None,
        decay_bias=decay_bias,
        elapsed_t=None,
        validated_metadata=validated_metadata,
    )
