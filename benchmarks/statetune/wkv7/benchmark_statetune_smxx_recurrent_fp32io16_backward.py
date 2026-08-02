#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""Correctness-gated nonzero-state RWKV-7 StateTune benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional

from flash_rwkv import pretrain_recurrent_fp32io16_forward, rwkv7_reference
from flash_rwkv.benchmark_contract import format_result, summarize_samples
from flash_rwkv.provenance import imported_source_family

SOURCE_ROOT = Path(__file__).resolve().parents[3]
HEAD_SIZE = 64
HEAD_COUNT = 2


def _git_revision() -> str | None:
    result = subprocess.run(
        ("git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _inputs(
    batch_size: int,
    token_count: int,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, token_count, HEAD_COUNT, HEAD_SIZE)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def normal(scale: float) -> torch.Tensor:
        return scale * torch.randn(
            shape,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )

    direction = functional.normalize(normal(1.0), dim=-1)
    strength = 0.1 * torch.rand(
        shape,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    tensors = (
        normal(0.05).to(torch.bfloat16),
        (
            -0.05
            - 0.15
            * torch.rand(
                shape,
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            )
        ).to(torch.bfloat16),
        normal(0.05).to(torch.bfloat16),
        normal(0.05).to(torch.bfloat16),
        (-direction).to(torch.bfloat16),
        (direction * strength).to(torch.bfloat16),
    )
    initial_state = 0.02 * torch.randn(
        batch_size,
        HEAD_COUNT,
        HEAD_SIZE,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    grad_output = 0.05 * torch.randn(
        shape,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    grad_final_state = 0.02 * torch.randn(
        initial_state.shape,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    return tensors, initial_state, grad_output, grad_final_state


def _run(
    implementation: str,
    base_inputs: tuple[torch.Tensor, ...],
    base_initial_state: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in base_inputs)
    initial_state = base_initial_state.detach().clone().requires_grad_(True)
    operation = (
        rwkv7_reference
        if implementation == "torch"
        else pretrain_recurrent_fp32io16_forward
    )
    output, final_state = operation(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        scale=0.125,
    )
    assert final_state is not None
    loss = (output.float() * grad_output).sum()
    loss = loss + (final_state * grad_final_state).sum()
    loss.backward()
    gradients = tuple(tensor.grad for tensor in (*inputs, initial_state))
    assert all(gradient is not None for gradient in gradients)
    return output, final_state, tuple(gradient for gradient in gradients if gradient is not None)


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return float((difference.square().mean().sqrt() / baseline).item())


def _correctness(
    inputs: tuple[torch.Tensor, ...],
    initial_state: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final_state: torch.Tensor,
) -> dict[str, object]:
    expected = _run("torch", inputs, initial_state, grad_output, grad_final_state)
    actual = _run("flash", inputs, initial_state, grad_output, grad_final_state)
    output_error = _relative_rmse(actual[0], expected[0])
    state_error = _relative_rmse(actual[1], expected[1])
    gradient_errors = tuple(
        _relative_rmse(value, reference)
        for value, reference in zip(actual[2], expected[2], strict=True)
    )
    passed = (
        output_error <= 0.02
        and state_error <= 0.02
        and max(gradient_errors) <= 0.08
        and all(torch.isfinite(gradient).all().item() for gradient in actual[2])
        and all(torch.count_nonzero(gradient).item() > 0 for gradient in actual[2])
    )
    return {
        "passed": passed,
        "output_relative_rmse": output_error,
        "final_state_relative_rmse": state_error,
        "gradient_relative_rmse": gradient_errors,
        "all_gradients_finite": all(
            torch.isfinite(gradient).all().item() for gradient in actual[2]
        ),
        "all_gradients_nonzero": all(
            torch.count_nonzero(gradient).item() > 0 for gradient in actual[2]
        ),
    }


def _measure(
    inputs: tuple[torch.Tensor, ...],
    initial_state: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final_state: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> list[float]:
    leaves = tuple(tensor.detach().requires_grad_(True) for tensor in inputs)
    state_leaf = initial_state.detach().requires_grad_(True)

    def launch() -> None:
        for tensor in (*leaves, state_leaf):
            tensor.grad = None
        output, final_state = pretrain_recurrent_fp32io16_forward(
            *leaves,
            initial_state=state_leaf,
            output_final_state=True,
            scale=0.125,
        )
        assert final_state is not None
        loss = (output.float() * grad_output).sum()
        loss = loss + (final_state * grad_final_state).sum()
        loss.backward()

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _parse_case(value: str) -> tuple[int, int]:
    try:
        batch_text, token_text = value.lower().split("x", maxsplit=1)
        batch_size, token_count = int(batch_text), int(token_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("case must use BxT") from error
    if batch_size <= 0 or token_count <= 0:
        raise argparse.ArgumentTypeError("B and T must be positive")
    return batch_size, token_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["1x17", "2x33", "4x65"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--output", type=Path, default=Path("flash-rwkv-statetune.json")
    )
    arguments = parser.parse_args()
    cases = tuple(_parse_case(value) for value in arguments.cases)
    rows = []
    for index, (batch_size, token_count) in enumerate(cases):
        inputs, initial_state, grad_output, grad_final_state = _inputs(
            batch_size,
            token_count,
            arguments.seed + index * 1009,
        )
        correctness = _correctness(
            inputs,
            initial_state,
            grad_output,
            grad_final_state,
        )
        if not correctness["passed"]:
            raise RuntimeError(f"StateTune correctness gate failed: {correctness}")
        samples = _measure(
            inputs,
            initial_state,
            grad_output,
            grad_final_state,
            warmup=arguments.warmup,
            iters=arguments.iters,
        )
        row = summarize_samples(
            label=f"statetune-B{batch_size}T{token_count}",
            batch_size=batch_size,
            token_count=token_count,
            samples_ms=samples,
        ).as_dict()
        rows.append({**row, "correctness": correctness, "raw_samples_ms": samples})
        print(format_result(row))
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = {
        "schema_version": 1,
        "benchmark": "flash_rwkv_statetune_nonzero_state_forward_backward",
        "revision": _git_revision(),
        "source_provenance": asdict(imported_source_family("rwkv-lm-pretrain")),
        "wkv_mode": "fp32io16",
        "dtype": "bfloat16",
        "warmup": arguments.warmup,
        "iters": arguments.iters,
        "measurement_boundary": (
            "public recurrent forward, final-state objective, and backward; "
            "input construction and correctness oracle are excluded"
        ),
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
        "rows": rows,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(f"{arguments.output.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(arguments.output)


if __name__ == "__main__":
    main()
