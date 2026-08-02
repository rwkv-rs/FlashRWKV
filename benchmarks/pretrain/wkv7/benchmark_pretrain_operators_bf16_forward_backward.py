#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Forward/backward benchmark for source-pinned non-recurrence training operators."""

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
from torch.nn import functional

from flash_rwkv import (
    pretrain_cmix_bf16,
    pretrain_head_l2wrap_ce_bf16,
    pretrain_l2wrap_ce_bf16,
    pretrain_tmix_a_gate_bf16,
    pretrain_tmix_kk_pre_bf16,
    pretrain_tmix_lnx_rkvres_xg_bf16,
    pretrain_tmix_mix6_bf16,
    pretrain_tmix_vres_gate_bf16,
)
from flash_rwkv.benchmark_contract import (
    ALBATROSS_BT_MATRIX,
    format_result,
    summarize_samples,
)
from flash_rwkv.provenance import imported_source_family
from flash_rwkv.registry import TRAINING_OPERATOR_SPECS

SOURCE_ROOT = Path(__file__).resolve().parents[3]
IDENTITIES = tuple(f"{spec.provider}/{spec.name}" for spec in TRAINING_OPERATOR_SPECS)


@dataclass(slots=True)
class PreparedCase:
    inputs: tuple[torch.Tensor, ...]
    launch: Callable[[], tuple[torch.Tensor, ...]]
    expected: tuple[torch.Tensor, ...]

    def reset_gradients(self) -> None:
        for tensor in self.inputs:
            tensor.grad = None


def _random(shape: tuple[int, ...], *, scale: float = 0.2) -> torch.Tensor:
    return (
        torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        .mul_(scale)
        .requires_grad_(True)
    )


def _with_backward(
    forward: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
) -> Callable[[], tuple[torch.Tensor, ...]]:
    def launch() -> tuple[torch.Tensor, ...]:
        result = forward()
        outputs = (result,) if isinstance(result, torch.Tensor) else result
        sum(output.float().square().mean() for output in outputs).backward()
        return outputs

    return launch


def _prepare_a_gate(batch_size: int, token_count: int, channels: int) -> PreparedCase:
    bias = _random((channels,))
    delta = _random((batch_size, token_count, channels))
    expected = (
        torch.sigmoid(bias.detach().float() + delta.detach().float()).bfloat16(),
    )
    return PreparedCase(
        (bias, delta),
        _with_backward(lambda: pretrain_tmix_a_gate_bf16(bias, delta)),
        expected,
    )


def _prepare_mix6(batch_size: int, token_count: int, channels: int) -> PreparedCase:
    x = _random((batch_size, token_count, channels))
    mixes = tuple(_random((channels,)) for _ in range(6))
    previous = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
    delta = previous.detach().float() - x.detach().float()
    expected = tuple(
        (x.detach().float() + delta * mix.detach().float()).bfloat16() for mix in mixes
    )
    return PreparedCase(
        (x, *mixes),
        _with_backward(lambda: pretrain_tmix_mix6_bf16(x, *mixes)),
        expected,
    )


def _prepare_kk(batch_size: int, token_count: int, channels: int) -> PreparedCase:
    key = _random((batch_size, token_count, channels))
    key_scale = _random((channels,))
    learning_rate = _random((batch_size, token_count, channels))
    learning_rate_scale = _random((channels,))
    direction = key.detach().float() * key_scale.detach().float()
    direction = functional.normalize(
        direction.view(batch_size, token_count, channels // 64, 64),
        dim=-1,
        eps=1e-12,
    ).view_as(key)
    expected = (
        (
            key.detach().float()
            * (
                1
                + (learning_rate.detach().float() - 1)
                * learning_rate_scale.detach().float()
            )
        ).bfloat16(),
        -direction.bfloat16(),
        (direction * learning_rate.detach().float()).bfloat16(),
    )
    inputs = (key, key_scale, learning_rate, learning_rate_scale)
    return PreparedCase(
        inputs,
        _with_backward(lambda: pretrain_tmix_kk_pre_bf16(*inputs)),
        expected,
    )


def _prepare_vres(batch_size: int, token_count: int, channels: int) -> PreparedCase:
    shape = (batch_size, token_count, channels)
    value = _random(shape)
    first_value = _random(shape)
    bias = _random((channels,))
    delta = _random(shape)
    gate = torch.sigmoid(bias.detach().float() + delta.detach().float())
    expected = (
        (
            value.detach().float()
            + (first_value.detach().float() - value.detach().float()) * gate
        ).bfloat16(),
    )
    inputs = (value, first_value, bias, delta)
    return PreparedCase(
        inputs,
        _with_backward(lambda: pretrain_tmix_vres_gate_bf16(*inputs)),
        expected,
    )


def _prepare_lnx(batch_size: int, token_count: int, channels: int) -> PreparedCase:
    shape = (batch_size, token_count, channels)
    recurrent_output, receptance, key, value = tuple(_random(shape) for _ in range(4))
    residual_scale = _random((channels // 64, 64))
    weight = _random((channels,))
    bias = _random((channels,))
    gate = _random(shape)
    grouped_output = recurrent_output.detach().float().view(
        batch_size, token_count, channels // 64, 64
    )
    mean = grouped_output.mean(dim=-1, keepdim=True)
    reciprocal_std = (
        grouped_output.var(dim=-1, correction=0, keepdim=True) + 64e-5
    ).rsqrt()
    normalized = (
        (grouped_output - mean)
        * reciprocal_std
        * weight.detach().float().view(1, 1, channels // 64, 64)
        + bias.detach().float().view(1, 1, channels // 64, 64)
    ).view_as(recurrent_output)
    residual = (
        receptance.detach().float().view(batch_size, token_count, channels // 64, 64)
        * key.detach().float().view(batch_size, token_count, channels // 64, 64)
        * residual_scale.detach().float()
    ).sum(dim=-1, keepdim=True) * value.detach().float().view(
        batch_size, token_count, channels // 64, 64
    )
    expected = (
        (
            (normalized + residual.view_as(normalized)) * gate.detach().float()
        ).bfloat16(),
    )
    inputs = (
        recurrent_output,
        receptance,
        key,
        value,
        residual_scale,
        weight,
        bias,
        gate,
    )
    return PreparedCase(
        inputs,
        _with_backward(lambda: pretrain_tmix_lnx_rkvres_xg_bf16(*inputs)),
        expected,
    )


def _prepare_cmix(batch_size: int, token_count: int, channels: int) -> PreparedCase:
    x = _random((batch_size, token_count, channels))
    mix = _random((channels,))
    key_weight = _random((4 * channels, channels), scale=0.05)
    value_weight = _random((channels, 4 * channels), scale=0.05)
    previous = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
    mixed = (
        x.detach().float()
        + (previous.detach().float() - x.detach().float()) * mix.detach().float()
    )
    activation = torch.relu(mixed @ key_weight.detach().float().T).square()
    expected = ((activation @ value_weight.detach().float().T).bfloat16(),)
    inputs = (x, mix, key_weight, value_weight)
    return PreparedCase(
        inputs,
        _with_backward(lambda: pretrain_cmix_bf16(*inputs)),
        expected,
    )


def _prepare_l2wrap(batch_size: int, token_count: int, channels: int) -> PreparedCase:
    vocab_size = max(256, channels)
    logits = _random((batch_size, token_count, vocab_size))
    targets = (
        torch.arange(batch_size * token_count, device="cuda", dtype=torch.int64).view(
            batch_size, token_count
        )
        % vocab_size
    )
    expected = (
        functional.cross_entropy(
            logits.detach().float().reshape(-1, vocab_size), targets.reshape(-1)
        ).reshape(()),
    )
    return PreparedCase(
        (logits,),
        _with_backward(lambda: pretrain_l2wrap_ce_bf16(logits, targets)),
        expected,
    )


def _prepare_head(batch_size: int, token_count: int, channels: int) -> PreparedCase:
    hidden = _random((batch_size, token_count, channels))
    weight = _random((65_536, channels), scale=0.02)
    targets = (
        torch.arange(batch_size * token_count, device="cuda", dtype=torch.int64).view(
            batch_size, token_count
        )
        % 65_536
    )
    expected = (
        functional.cross_entropy(
            hidden.detach().float().reshape(-1, channels) @ weight.detach().float().T,
            targets.reshape(-1),
        ).reshape(()),
    )
    return PreparedCase(
        (hidden, weight),
        _with_backward(
            lambda: pretrain_head_l2wrap_ce_bf16(
                hidden, weight, targets, chunk_rows=min(4096, batch_size * token_count)
            )
        ),
        expected,
    )


BUILDERS: dict[str, Callable[[int, int, int], PreparedCase]] = {
    "rwkv-lm/pretrain_tmix_a_gate_bf16": _prepare_a_gate,
    "rwkv-lm/pretrain_tmix_mix6_bf16": _prepare_mix6,
    "rwkv-lm/pretrain_tmix_kk_pre_bf16": _prepare_kk,
    "rwkv-lm/pretrain_tmix_vres_gate_bf16": _prepare_vres,
    "rwkv-lm/pretrain_tmix_lnx_rkvres_xg_bf16": _prepare_lnx,
    "rwkv-lm/pretrain_cmix_bf16": _prepare_cmix,
    "rwkv-lm/pretrain_l2wrap_ce_bf16": _prepare_l2wrap,
    "rwkv-lm/pretrain_head_l2wrap_ce_bf16": _prepare_head,
}


def _correctness(prepared: PreparedCase) -> dict[str, object]:
    prepared.reset_gradients()
    outputs = prepared.launch()
    torch.cuda.synchronize()
    max_error = max(
        (actual.float() - expected.float()).abs().max().item()
        for actual, expected in zip(outputs, prepared.expected, strict=True)
    )
    gradients = tuple(tensor.grad for tensor in prepared.inputs)
    all_gradients_present = all(gradient is not None for gradient in gradients)
    finite_gradients = all(
        gradient is not None and torch.isfinite(gradient).all().item()
        for gradient in gradients
    )
    nonzero_gradients = all(
        gradient is not None and torch.count_nonzero(gradient).item() > 0
        for gradient in gradients
    )
    return {
        "passed": max_error <= 0.02
        and all_gradients_present
        and finite_gradients
        and nonzero_gradients,
        "max_absolute_output_error": max_error,
        "absolute_tolerance": 0.02,
        "all_gradients_present": all_gradients_present,
        "finite_gradients": finite_gradients,
        "nonzero_gradients": nonzero_gradients,
    }


def _measure(prepared: PreparedCase, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        prepared.reset_gradients()
        prepared.launch()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        prepared.reset_gradients()
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
            SOURCE_ROOT / "flash_rwkv/channel_mix.py",
            SOURCE_ROOT / "flash_rwkv/head_l2wrap_ce.py",
            SOURCE_ROOT / "flash_rwkv/l2wrap_ce.py",
            SOURCE_ROOT / "flash_rwkv/time_mix.py",
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
    specs = {f"{spec.provider}/{spec.name}": spec for spec in TRAINING_OPERATOR_SPECS}
    rows: list[dict[str, object]] = []
    for identity_index, identity in enumerate(identities):
        for case_index, (batch_size, token_count) in enumerate(cases):
            torch.manual_seed(seed + identity_index * 1009 + case_index * 9173)
            prepared = BUILDERS[identity](batch_size, token_count, channels)
            correctness = _correctness(prepared)
            if not correctness["passed"]:
                raise RuntimeError(
                    f"correctness gate failed for {identity} B{batch_size}T{token_count}: {correctness}"
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
        "benchmark": "flash_rwkv_training_operators_forward_backward",
        "source_provenance": asdict(imported_source_family("rwkv-lm-pretrain")),
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
        "measurement_boundary": "one public operator forward plus paired backward; gradient reset and synchronization are outside CUDA events",
        "dtype": "bfloat16",
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
    parser.add_argument("--cases", nargs="+", default=["1x16", "8x8", "16x16"])
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--output", type=Path, default=Path("flash-rwkv-training-operators.json")
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
    for row in payload["rows"]:
        print(
            f"{format_result(row)} provider={row['provider']} "
            f"name={row['name']} source_revision={row['source_revision']}"
        )
    print(
        json.dumps(
            {"output": str(arguments.output), "rows": len(payload["rows"])},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
