#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""Correctness-gated nonzero-state RWKV-7 StateTune benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional
from torch.profiler import ProfilerActivity, profile

from flash_rwkv import statetune_recurrent_fp32io16_forward
from flash_rwkv.benchmark_contract import format_result, summarize_samples
from flash_rwkv.ops import _canonical_statetune_recurrent_fp32io16
from flash_rwkv.provenance import imported_source_family
from flash_rwkv.reference import rwkv7_decay_logits_reference
from flash_rwkv.registry import get_kernel_spec

SOURCE_ROOT = Path(__file__).resolve().parents[3]
HEAD_SIZE = 64
HEAD_COUNT = 2
OPERATOR_SPEC = get_kernel_spec(
    "statetune_recurrent_fp32io16_forward_backward",
    provider="flash_rwkv",
)
MODE = "fp32io16"
CORRECTNESS_LIMITS = {
    "output_relative_rmse": 0.007,
    "final_state_relative_rmse": 0.008,
    "gradient_relative_rmse": 0.008,
}


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
        normal(1.0).to(torch.bfloat16),
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


def _unfused_correct_statetune(
    inputs: tuple[torch.Tensor, ...],
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r, decay_logits, k, v, a, b = inputs
    log_decay = -0.6065306597126334 * torch.sigmoid(decay_logits)
    return _canonical_statetune_recurrent_fp32io16(
        r,
        log_decay,
        k,
        v,
        a,
        b,
        initial_state=initial_state,
        output_final_state=True,
        scale=0.125,
    )


def _run(
    implementation: str,
    base_inputs: tuple[torch.Tensor, ...],
    base_initial_state: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in base_inputs)
    initial_state = base_initial_state.detach().clone().requires_grad_(True)
    if implementation == "torch":
        output, final_state = rwkv7_decay_logits_reference(
            *inputs,
            initial_state=initial_state,
            output_final_state=True,
            scale=0.125,
        )
    elif implementation == "unfused":
        output, final_state = _unfused_correct_statetune(
            inputs,
            initial_state,
        )
    elif implementation == "fused":
        output, final_state = statetune_recurrent_fp32io16_forward(
            *inputs,
            initial_state=initial_state,
            output_final_state=True,
            scale=0.125,
        )
    else:
        raise ValueError(f"unknown implementation: {implementation}")
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
    implementations = {
        name: _run(name, inputs, initial_state, grad_output, grad_final_state)
        for name in ("unfused", "fused")
    }
    errors: dict[str, dict[str, object]] = {}
    passed = True
    for name, actual in implementations.items():
        output_error = _relative_rmse(actual[0], expected[0])
        state_error = _relative_rmse(actual[1], expected[1])
        gradient_errors = tuple(
            _relative_rmse(value, reference)
            for value, reference in zip(actual[2], expected[2], strict=True)
        )
        finite = all(
            torch.isfinite(gradient).all().item() for gradient in actual[2]
        )
        nonzero = all(
            torch.count_nonzero(gradient).item() > 0 for gradient in actual[2]
        )
        implementation_passed = (
            output_error <= CORRECTNESS_LIMITS["output_relative_rmse"]
            and state_error <= CORRECTNESS_LIMITS[
                "final_state_relative_rmse"
            ]
            and max(gradient_errors)
            <= CORRECTNESS_LIMITS["gradient_relative_rmse"]
            and finite
            and nonzero
        )
        passed = passed and implementation_passed
        errors[name] = {
            "passed": implementation_passed,
            "output_relative_rmse": output_error,
            "final_state_relative_rmse": state_error,
            "gradient_relative_rmse": gradient_errors,
            "all_gradients_finite": finite,
            "all_gradients_nonzero": nonzero,
        }
    unfused = implementations["unfused"]
    fused = implementations["fused"]
    cross_errors = {
        "output_relative_rmse": _relative_rmse(unfused[0], fused[0]),
        "final_state_relative_rmse": _relative_rmse(unfused[1], fused[1]),
        "gradient_relative_rmse": tuple(
            _relative_rmse(left, right)
            for left, right in zip(unfused[2], fused[2], strict=True)
        ),
    }
    cross_passed = (
        cross_errors["output_relative_rmse"]
        <= CORRECTNESS_LIMITS["output_relative_rmse"]
        and cross_errors["final_state_relative_rmse"]
        <= CORRECTNESS_LIMITS["final_state_relative_rmse"]
        and max(cross_errors["gradient_relative_rmse"])
        <= CORRECTNESS_LIMITS["gradient_relative_rmse"]
    )
    return {
        "passed": passed and cross_passed,
        "oracle": "independent raw decay producer plus FP32 torch recurrence",
        "limits": CORRECTNESS_LIMITS,
        "implementations": errors,
        "unfused_vs_fused": {**cross_errors, "passed": cross_passed},
    }


def _training_launch(
    implementation: str,
    inputs: tuple[torch.Tensor, ...],
    initial_state: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final_state: torch.Tensor,
) -> tuple[Callable[[], None], tuple[torch.Tensor, ...]]:
    leaves = tuple(tensor.detach().requires_grad_(True) for tensor in inputs)
    state_leaf = initial_state.detach().requires_grad_(True)

    def launch() -> None:
        for tensor in (*leaves, state_leaf):
            tensor.grad = None
        if implementation == "unfused":
            output, final_state = _unfused_correct_statetune(
                leaves,
                state_leaf,
            )
        elif implementation == "fused":
            output, final_state = statetune_recurrent_fp32io16_forward(
                *leaves,
                initial_state=state_leaf,
                output_final_state=True,
                scale=0.125,
            )
        else:
            raise ValueError(f"unknown implementation: {implementation}")
        assert final_state is not None
        loss = (output.float() * grad_output).sum()
        loss = loss + (final_state * grad_final_state).sum()
        loss.backward()

    return launch, (*leaves, state_leaf)


def _measure(
    implementation: str,
    inputs: tuple[torch.Tensor, ...],
    initial_state: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final_state: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> list[float]:
    launch, _ = _training_launch(
        implementation,
        inputs,
        initial_state,
        grad_output,
        grad_final_state,
    )

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


def _launch_trace(
    implementation: str,
    inputs: tuple[torch.Tensor, ...],
    initial_state: torch.Tensor,
    grad_output: torch.Tensor,
    grad_final_state: torch.Tensor,
) -> dict[str, object]:
    launch, _ = _training_launch(
        implementation,
        inputs,
        initial_state,
        grad_output,
        grad_final_state,
    )
    launch()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    ) as trace:
        launch()
        torch.cuda.synchronize()
    cuda_kernel_names = [
        event.name
        for event in trace.events()
        if "cuda" in str(event.device_type).lower()
    ]
    return {
        "cuda_kernel_count": len(cuda_kernel_names),
        "cuda_kernel_names": cuda_kernel_names,
        "forward_recurrent_kernel_count": sum(
            "recurrent_common_fp32io16_forward_kernel" in name
            for name in cuda_kernel_names
        ),
        "backward_recurrent_kernel_count": sum(
            "recurrent_common_fp32io16_backward_kernel" in name
            for name in cuda_kernel_names
        ),
    }


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
    source_revision = _git_revision()
    if source_revision is None or len(source_revision) != 40:
        raise RuntimeError("StateTune benchmark requires a full Git source revision")
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
        samples_by_implementation = {
            implementation: _measure(
                implementation,
                inputs,
                initial_state,
                grad_output,
                grad_final_state,
                warmup=arguments.warmup,
                iters=arguments.iters,
            )
            for implementation in ("unfused", "fused")
        }
        launch_traces = {
            implementation: _launch_trace(
                implementation,
                inputs,
                initial_state,
                grad_output,
                grad_final_state,
            )
            for implementation in ("unfused", "fused")
        }
        if (
            launch_traces["unfused"]["cuda_kernel_count"]
            <= launch_traces["fused"]["cuda_kernel_count"]
        ):
            raise RuntimeError(
                "StateTune fusion did not eliminate transform launches: "
                f"{launch_traces}"
            )
        samples = samples_by_implementation["fused"]
        unfused_p50 = summarize_samples(
            label=f"statetune-unfused-B{batch_size}T{token_count}",
            batch_size=batch_size,
            token_count=token_count,
            samples_ms=samples_by_implementation["unfused"],
        ).p50_ms
        row = summarize_samples(
            label=f"statetune-B{batch_size}T{token_count}",
            batch_size=batch_size,
            token_count=token_count,
            samples_ms=samples,
        ).as_dict()
        row.update(
            provider=OPERATOR_SPEC.provider,
            name=OPERATOR_SPEC.name,
            source_revision=source_revision,
            mode=MODE,
        )
        rows.append(
            {
                **row,
                "correctness": correctness,
                "raw_samples_ms": samples,
                "training_ab": {
                    "A": "unfused_correct_product",
                    "B": "public_raw_fused_recurrent",
                    "same_inputs_initial_state_loss_upstream_gradients": True,
                    "metadata_boundary": (
                        "both private canonical and public raw wrappers build "
                        "identical fixed-length chunk metadata inside timing"
                    ),
                    "unfused_raw_samples_ms": samples_by_implementation[
                        "unfused"
                    ],
                    "fused_raw_samples_ms": samples,
                    "unfused_p50_ms": unfused_p50,
                    "fused_p50_ms": row["p50_ms"],
                    "fused_speedup_over_unfused": (
                        unfused_p50 / float(row["p50_ms"])
                    ),
                    "timed_transform_materialization_bytes": {
                        "unfused": (
                            inputs[1].numel() * inputs[1].element_size()
                        ),
                        "fused": 0,
                    },
                    "launch_trace": launch_traces,
                },
            }
        )
        print(
            f"{format_result(row)} provider={row['provider']} "
            f"name={row['name']} source_revision={row['source_revision']} "
            f"mode={row['mode']}"
        )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = {
        "schema_version": 1,
        "benchmark": "flash_rwkv_statetune_nonzero_state_forward_backward",
        "revision": source_revision,
        "source_provenance": asdict(imported_source_family("rwkv-lm-pretrain")),
        "wkv_mode": "fp32io16",
        "dtype": "bfloat16",
        "warmup": arguments.warmup,
        "iters": arguments.iters,
        "measurement_boundary": (
            "A includes raw-logit to canonical-log-decay producer, private "
            "canonical recurrent forward, final-state objective, and backward; "
            "B includes public raw fused recurrent forward, the identical "
            "objective, and backward; input construction and correctness "
            "oracle are excluded"
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
