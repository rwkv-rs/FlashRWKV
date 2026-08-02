#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Correctness-gated benchmark for imported fused inference block operators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from flash_rwkv import (
    infer_cmix_mix_fp16,
    infer_tmix_kk_a_gate_fp16,
    infer_tmix_lnx_rkvres_xg_fp16,
    infer_tmix_mix6_fp16,
    infer_tmix_vres_gate_fp16,
)
from flash_rwkv.benchmark_contract import ALBATROSS_BT_MATRIX, summarize_samples
from flash_rwkv.registry import INFERENCE_OPERATOR_SPECS

SOURCE_ROOT = Path(__file__).resolve().parents[1]
IDENTITIES = tuple(f"{spec.provider}/{spec.name}" for spec in INFERENCE_OPERATOR_SPECS)


@dataclass(slots=True)
class PreparedCase:
    reset: Callable[[], None]
    launch: Callable[[], None]
    actual: Callable[[], tuple[torch.Tensor, ...]]
    expected: tuple[torch.Tensor, ...]


def _random(shape: tuple[int, ...], *, scale: float = 0.2) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=torch.float16).mul_(scale)


def _prepare_mix6(batch_size: int, token_count: int, channels: int) -> PreparedCase:
    x = _random((batch_size, token_count, channels))
    initial_state = _random((batch_size, channels))
    shift_state = initial_state.clone()
    mixes = tuple(_random((channels,)) for _ in range(6))
    result: tuple[torch.Tensor, ...] = ()
    previous = torch.cat((initial_state[:, None], x[:, :-1]), dim=1)
    delta = previous.float() - x.float()
    expected = tuple((x.float() + delta * mix.float()).half() for mix in mixes)

    def reset() -> None:
        shift_state.copy_(initial_state)

    def launch() -> None:
        nonlocal result
        result = (*infer_tmix_mix6_fp16(x, shift_state, mixes), shift_state.clone())

    return PreparedCase(reset, launch, lambda: result, (*expected, x[:, -1].clone()))


def _prepare_kk_gate(batch_size: int, token_count: int, channels: int) -> PreparedCase:
    key = _random((batch_size, token_count, channels))
    key_scale = _random((channels,))
    gate_bias = _random((channels,))
    gate_delta = _random((batch_size, token_count, channels))
    key_gate_scale = _random((channels,))
    result: tuple[torch.Tensor, ...] = ()
    gate = torch.sigmoid(gate_bias.float() + gate_delta.float())
    direction_input = (key.float() * key_scale.float()).view(
        batch_size, token_count, channels // 64, 64
    )
    direction = direction_input / direction_input.square().sum(
        -1, keepdim=True
    ).sqrt().clamp_min(1e-12)
    direction = direction.view_as(key)
    expected = (
        (key.float() * (1.0 + key_gate_scale.float() * (gate - 1.0))).half(),
        -direction.half(),
        (direction * gate).half(),
    )

    def launch() -> None:
        nonlocal result
        result = infer_tmix_kk_a_gate_fp16(
            key, key_scale, gate_bias, gate_delta, key_gate_scale
        )

    return PreparedCase(lambda: None, launch, lambda: result, expected)


def _prepare_lnx(batch_size: int, token_count: int, channels: int) -> PreparedCase:
    shape = (batch_size, token_count, channels)
    recurrent_output, receptance, key, value, gate = tuple(
        _random(shape) for _ in range(5)
    )
    residual_scale = _random((channels,))
    norm_weight = _random((channels,))
    norm_bias = _random((channels,))
    result: tuple[torch.Tensor, ...] = ()
    heads = channels // 64
    grouped = recurrent_output.float().view(batch_size, token_count, heads, 64)
    mean = grouped.mean(dim=-1, keepdim=True)
    reciprocal_std = (grouped.var(dim=-1, correction=0, keepdim=True) + 64e-5).rsqrt()
    normalized = (grouped - mean) * reciprocal_std
    normalized = normalized * norm_weight.float().view(1, 1, heads, 64)
    normalized = normalized + norm_bias.float().view(1, 1, heads, 64)
    residual = (
        receptance.float().view(batch_size, token_count, heads, 64)
        * key.float().view(batch_size, token_count, heads, 64)
        * residual_scale.float().view(1, 1, heads, 64)
    ).sum(dim=-1, keepdim=True) * value.float().view(batch_size, token_count, heads, 64)
    expected = (
        ((normalized + residual) * gate.float().view_as(grouped)).half().view(shape),
    )

    def launch() -> None:
        nonlocal result
        result = (
            infer_tmix_lnx_rkvres_xg_fp16(
                recurrent_output,
                receptance,
                key,
                value,
                residual_scale,
                norm_weight,
                norm_bias,
                gate,
            ),
        )

    return PreparedCase(lambda: None, launch, lambda: result, expected)


def _prepare_vres(batch_size: int, token_count: int, channels: int) -> PreparedCase:
    shape = (batch_size, token_count, channels)
    value = _random(shape)
    first_value = _random(shape)
    gate_bias = _random((channels,))
    gate_delta = _random(shape)
    result: tuple[torch.Tensor, ...] = ()
    gate = torch.sigmoid(gate_bias.float() + gate_delta.float())
    expected = ((value.float() + (first_value.float() - value.float()) * gate).half(),)

    def launch() -> None:
        nonlocal result
        result = (infer_tmix_vres_gate_fp16(value, first_value, gate_bias, gate_delta),)

    return PreparedCase(lambda: None, launch, lambda: result, expected)


def _prepare_cmix(batch_size: int, token_count: int, channels: int) -> PreparedCase:
    x = _random((batch_size, token_count, channels))
    initial_state = _random((batch_size, channels))
    shift_state = initial_state.clone()
    mix = _random((channels,))
    result: tuple[torch.Tensor, ...] = ()
    previous = torch.cat((initial_state[:, None], x[:, :-1]), dim=1)
    expected = (
        (x.float() + (previous.float() - x.float()) * mix.float()).half(),
        x[:, -1].clone(),
    )

    def reset() -> None:
        shift_state.copy_(initial_state)

    def launch() -> None:
        nonlocal result
        result = (infer_cmix_mix_fp16(x, shift_state, mix), shift_state.clone())

    return PreparedCase(reset, launch, lambda: result, expected)


BUILDERS: dict[str, Callable[[int, int, int], PreparedCase]] = {
    "albatross/infer_tmix_mix6_fp16_forward": _prepare_mix6,
    "albatross/infer_tmix_kk_a_gate_fp16_forward": _prepare_kk_gate,
    "albatross/infer_tmix_lnx_rkvres_xg_fp16_forward": _prepare_lnx,
    "albatross/infer_tmix_vres_gate_fp16_forward": _prepare_vres,
    "albatross/infer_cmix_mix_fp16_forward": _prepare_cmix,
}


def _error_summary(prepared: PreparedCase) -> dict[str, object]:
    prepared.reset()
    prepared.launch()
    torch.cuda.synchronize()
    actual = prepared.actual()
    if len(actual) != len(prepared.expected):
        raise RuntimeError("operator output count changed during correctness gate")
    max_absolute_error = max(
        (left.float() - right.float()).abs().max().item()
        for left, right in zip(actual, prepared.expected, strict=True)
    )
    all_finite = all(torch.isfinite(tensor).all().item() for tensor in actual)
    return {
        "passed": all_finite and max_absolute_error <= 0.02,
        "all_finite": all_finite,
        "max_absolute_error": max_absolute_error,
        "absolute_tolerance": 0.02,
    }


def _measure(prepared: PreparedCase, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        prepared.reset()
        prepared.launch()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        prepared.reset()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        prepared.launch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _git_revision() -> str | None:
    result = subprocess.run(
        ("git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (
            SOURCE_ROOT
            / "csrc/infer/wkv7/infer_smxx_fused_fp16_forward_registration.cpp",
            SOURCE_ROOT / "csrc/infer/wkv7/infer_smxx_fused_fp16_forward.cu",
            SOURCE_ROOT / "flash_rwkv/inference_blocks.py",
            Path(__file__),
        )
    ):
        digest.update(path.relative_to(SOURCE_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run(
    *,
    identities: Sequence[str],
    cases: Sequence[tuple[int, int]],
    channels: int,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if channels <= 0 or channels % 64:
        raise ValueError("channels must be a positive multiple of 64")
    if warmup < 0 or iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive")
    specs = {f"{spec.provider}/{spec.name}": spec for spec in INFERENCE_OPERATOR_SPECS}
    rows: list[dict[str, object]] = []
    for identity_index, identity in enumerate(identities):
        for case_index, (batch_size, token_count) in enumerate(cases):
            torch.manual_seed(seed + identity_index * 1009 + case_index * 9173)
            prepared = BUILDERS[identity](batch_size, token_count, channels)
            correctness = _error_summary(prepared)
            if not correctness["passed"]:
                raise RuntimeError(
                    f"correctness gate failed for {identity} B{batch_size}T{token_count}"
                )
            samples = _measure(prepared, warmup=warmup, iters=iters)
            row = summarize_samples(
                label=f"B{batch_size}T{token_count}",
                batch_size=batch_size,
                token_count=token_count,
                samples_ms=samples,
            )
            rows.append(
                {
                    "provider": specs[identity].provider,
                    "name": specs[identity].name,
                    "source_revision": specs[identity].source_revision,
                    "registry": asdict(specs[identity]),
                    **row.as_dict(),
                    "correctness": correctness,
                    "raw_samples_ms": samples,
                }
            )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema_version": 1,
        "benchmark": "flash_rwkv_fused_inference_blocks",
        "revision": _git_revision(),
        "source_digest": _source_digest(),
        "hardware": {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "dtype": "float16",
        "channels": channels,
        "warmup": warmup,
        "iters": iters,
        "rows": rows,
    }


def _parse_cases(values: Sequence[str]) -> tuple[tuple[int, int], ...]:
    known = {
        f"{batch}x{tokens}": (batch, tokens) for batch, tokens in ALBATROSS_BT_MATRIX
    }
    try:
        return tuple(dict.fromkeys(known[value.lower()] for value in values))
    except KeyError as error:
        raise SystemExit(f"unknown B/T case {error.args[0]!r}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--identities", nargs="+", choices=IDENTITIES, default=IDENTITIES
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[f"{batch}x{tokens}" for batch, tokens in ALBATROSS_BT_MATRIX],
    )
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--output", type=Path, default=Path("flash-rwkv-operators.json")
    )
    arguments = parser.parse_args()
    payload = run(
        identities=arguments.identities,
        cases=_parse_cases(arguments.cases),
        channels=arguments.channels,
        warmup=arguments.warmup,
        iters=arguments.iters,
        seed=arguments.seed,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(f"{arguments.output.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(arguments.output)
    print(
        json.dumps(
            {"output": str(arguments.output), "rows": len(payload["rows"])},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
