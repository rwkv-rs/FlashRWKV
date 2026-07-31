# SPDX-License-Identifier: MIT
"""Correctness-gated fixed-length RWKV-7 forward/backward benchmark."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional

from benchmark_rwkv7 import (
    DTYPES,
    HEAD_SIZE,
    TOLERANCE_PATH,
    _hardware_metadata,
    _latency_summary,
    _sha256,
    _source_metadata,
)
from flash_rwkv import (
    decay_logits_to_log_decay,
    rwkv7_from_decay_logits,
    rwkv7_reference,
)
from flash_rwkv.config import chunk_tuning_key, select_chunk_config


INPUT_NAMES = ("r", "decay_logits", "k", "v", "a", "b", "initial_state")
PROVIDER_NAMES = ("flash_chunk", "fla_chunk")
PROFILES: dict[str, tuple[int, int]] = {
    "tail17_b1": (1, 17),
    "multi_chunk64_b4": (4, 64),
    "tail257_b1": (1, 257),
}
TrainingResult = tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, ...],
]


@dataclass(frozen=True)
class BenchmarkConfig:
    hidden_size: int
    profiles: tuple[str, ...]
    dtypes: tuple[str, ...]
    providers: tuple[str, ...]
    warmup_iters: int
    samples: int
    seed: int
    output: Path | None
    measure: bool
    profile_provider: str | None
    profile_case: str | None
    profile_dtype: str
    profile_iterations: int


@dataclass(frozen=True)
class Inputs:
    tensors: tuple[torch.Tensor, ...]
    initial_state: torch.Tensor
    grad_output: torch.Tensor
    grad_final_state: torch.Tensor


@dataclass(frozen=True)
class Provider:
    name: str
    configuration: dict[str, object]
    call: Callable[[], TrainingResult]


def _make_inputs(
    batch_size: int,
    sequence_length: int,
    *,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
) -> Inputs:
    num_heads = hidden_size // HEAD_SIZE
    shape = (batch_size, sequence_length, num_heads, HEAD_SIZE)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def normal(scale: float) -> torch.Tensor:
        return scale * torch.randn(
            shape,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )

    direction = functional.normalize(normal(1.0), dim=-1)
    strength = 0.1 * torch.rand(
        shape,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    tensors = (
        normal(0.05).to(dtype),
        normal(1.0).to(dtype),
        normal(0.05).to(dtype),
        normal(0.05).to(dtype),
        (-direction).to(dtype),
        (direction * strength).to(dtype),
    )
    initial_state = 0.02 * torch.randn(
        (
            batch_size,
            num_heads,
            HEAD_SIZE,
            HEAD_SIZE,
        ),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    grad_output = (
        0.05
        * torch.randn(
            shape,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
    ).to(dtype).float()
    grad_final_state = 0.02 * torch.randn(
        initial_state.shape,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    return Inputs(
        tensors=tensors,
        initial_state=initial_state,
        grad_output=grad_output,
        grad_final_state=grad_final_state,
    )


def _leaf_inputs(inputs: Inputs) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    tensors = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in inputs.tensors
    )
    initial_state = (
        inputs.initial_state.detach().clone().requires_grad_(True)
    )
    return tensors, initial_state


def _loss_and_gradients(
    output: torch.Tensor,
    final_state: torch.Tensor | None,
    inputs: tuple[torch.Tensor, ...],
    initial_state: torch.Tensor,
    upstream: Inputs,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    if final_state is None:
        raise RuntimeError("training provider did not return final state")
    loss = (
        output.float() * upstream.grad_output
    ).sum() + (
        final_state * upstream.grad_final_state
    ).sum()
    gradients = torch.autograd.grad(
        loss,
        (*inputs, initial_state),
    )
    return output, final_state, gradients


def _providers(
    inputs: Inputs,
    names: tuple[str, ...],
) -> tuple[
    Provider,
    dict[str, Provider],
]:
    reference_inputs, reference_state = _leaf_inputs(inputs)

    def reference_call() -> TrainingResult:
        r, decay_logits, k, v, a, b = reference_inputs
        output, final_state = rwkv7_reference(
            r,
            decay_logits_to_log_decay(decay_logits),
            k,
            v,
            a,
            b,
            initial_state=reference_state,
            output_final_state=True,
        )
        return _loss_and_gradients(
            output,
            final_state,
            reference_inputs,
            reference_state,
            inputs,
        )

    sample = inputs.tensors[0]
    key = chunk_tuning_key(
        sample,
        mode="fp32io16",
        packed=False,
        max_sequence_length=sample.shape[1],
    )
    selected = select_chunk_config(key)
    flash_inputs, flash_state = _leaf_inputs(inputs)

    def flash_call() -> TrainingResult:
        output, final_state = rwkv7_from_decay_logits(
            *flash_inputs,
            initial_state=flash_state,
            output_final_state=True,
            mode="fp32io16",
            algorithm="chunk",
            chunk_config=selected.config,
        )
        return _loss_and_gradients(
            output,
            final_state,
            flash_inputs,
            flash_state,
            inputs,
        )

    fla_inputs, fla_state = _leaf_inputs(inputs)

    def fla_call() -> TrainingResult:
        from fla.ops.rwkv7 import chunk_rwkv7

        r, decay_logits, k, v, a, b = fla_inputs
        output, final_state = chunk_rwkv7(
            r,
            decay_logits_to_log_decay(decay_logits),
            k,
            v,
            a,
            b,
            initial_state=fla_state,
            output_final_state=True,
            safe_gate=True,
            chunk_size=16,
        )
        return _loss_and_gradients(
            output,
            final_state,
            fla_inputs,
            fla_state,
            inputs,
        )

    reference = Provider(
        name="torch_reference",
        configuration={
            "implementation": (
                "flash_rwkv.rwkv7_reference with differentiable adapter"
            ),
            "state_dtype": "float32",
        },
        call=reference_call,
    )
    available = {
        "flash_chunk": Provider(
            name="flash_chunk",
            configuration={
                "implementation": (
                    "flash_rwkv.rwkv7_from_decay_logits(chunk)"
                ),
                "mode": "fp32io16",
                "tuning_key": key.identifier,
                "selection_source": selected.source,
                "chunk_config": asdict(selected.config),
            },
            call=flash_call,
        ),
        "fla_chunk": Provider(
            name="fla_chunk",
            configuration={
                "implementation": "fla.ops.rwkv7.chunk_rwkv7",
                "state_dtype": "float32",
                "safe_gate": True,
                "chunk_size": 16,
            },
            call=fla_call,
        ),
    }
    return reference, {name: available[name] for name in names}


def _relative_rmse(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    difference = actual.float() - expected.float()
    rmse = difference.square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return float((rmse / baseline).item())


def _correctness(
    provider: Provider,
    expected: tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
    ],
    *,
    limits: dict[str, float],
) -> dict[str, object]:
    output, final_state, gradients = provider.call()
    torch.cuda.synchronize()
    expected_output, expected_state, expected_gradients = expected
    output_error = _relative_rmse(output, expected_output)
    state_error = _relative_rmse(final_state, expected_state)
    gradient_errors = {
        name: _relative_rmse(gradient, expected_gradient)
        for name, gradient, expected_gradient in zip(
            INPUT_NAMES,
            gradients,
            expected_gradients,
            strict=True,
        )
    }
    finite = bool(
        torch.isfinite(output).all().item()
        and torch.isfinite(final_state).all().item()
        and all(
            torch.isfinite(gradient).all().item()
            for gradient in gradients
        )
    )
    passed = bool(
        finite
        and output_error <= limits["output_relative_rmse"]
        and state_error <= limits["state_relative_rmse"]
        and all(
            error <= limits["gradient_relative_rmse"]
            for error in gradient_errors.values()
        )
    )
    return {
        "passed": passed,
        "output_relative_rmse": output_error,
        "final_state_relative_rmse": state_error,
        "gradient_relative_rmse": gradient_errors,
        "maximum_gradient_relative_rmse": max(
            gradient_errors.values()
        ),
        "output_relative_rmse_limit": limits[
            "output_relative_rmse"
        ],
        "final_state_relative_rmse_limit": limits[
            "state_relative_rmse"
        ],
        "gradient_relative_rmse_limit": limits[
            "gradient_relative_rmse"
        ],
        "finite": finite,
    }


def _measure(
    providers: dict[str, Provider],
    correctness: dict[str, dict[str, object]],
    *,
    total_tokens: int,
    warmup_iters: int,
    samples: int,
    seed: int,
) -> dict[str, dict[str, object] | None]:
    correct = {
        name: provider
        for name, provider in providers.items()
        if correctness[name]["passed"]
    }
    for provider in correct.values():
        for _ in range(warmup_iters):
            provider.call()
    torch.cuda.synchronize()

    raw_samples = {name: [] for name in correct}
    ordering = list(correct)
    generator = random.Random(seed)
    for _ in range(samples):
        generator.shuffle(ordering)
        for name in ordering:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            correct[name].call()
            end.record()
            end.synchronize()
            raw_samples[name].append(start.elapsed_time(end))
    return {
        name: (
            _latency_summary(
                raw_samples[name],
                total_tokens=total_tokens,
            )
            if name in raw_samples
            else None
        )
        for name in providers
    }


def _profile_provider(
    provider: Provider,
    *,
    range_name: str,
    iterations: int,
) -> dict[str, object]:
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        torch.cuda.nvtx.range_push(range_name)
        try:
            for _ in range(iterations):
                provider.call()
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
    finally:
        cudart.cudaProfilerStop()
    return {"name": range_name, "iterations": iterations}


def _run_case(
    profile_name: str,
    dtype_name: str,
    config: BenchmarkConfig,
    limits: dict[str, float],
    *,
    seed: int,
) -> dict[str, object]:
    batch_size, sequence_length = PROFILES[profile_name]
    inputs = _make_inputs(
        batch_size,
        sequence_length,
        hidden_size=config.hidden_size,
        dtype=DTYPES[dtype_name],
        seed=seed,
    )
    reference, providers = _providers(inputs, config.providers)
    expected = reference.call()
    torch.cuda.synchronize()
    correctness = {
        name: _correctness(provider, expected, limits=limits)
        for name, provider in providers.items()
    }
    measurements = (
        _measure(
            providers,
            correctness,
            total_tokens=batch_size * sequence_length,
            warmup_iters=config.warmup_iters,
            samples=config.samples,
            seed=seed + 7919,
        )
        if config.measure
        else {name: None for name in providers}
    )

    profile_range = None
    if (
        config.profile_provider is not None
        and config.profile_case == profile_name
        and config.profile_dtype == dtype_name
    ):
        if not correctness[config.profile_provider]["passed"]:
            raise RuntimeError(
                "profiler provider failed correctness for the exact case"
            )
        profile_range = _profile_provider(
            providers[config.profile_provider],
            range_name=(
                f"flash_rwkv.training.{config.profile_provider}."
                f"{profile_name}.{dtype_name}"
            ),
            iterations=config.profile_iterations,
        )

    result = {
        "profile": profile_name,
        "layout": "fixed",
        "dtype": dtype_name,
        "mode": "fp32io16",
        "input_semantics": "RWKV-LM decay logits with differentiable adapter",
        "shape": [
            batch_size,
            sequence_length,
            config.hidden_size // HEAD_SIZE,
            HEAD_SIZE,
        ],
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "total_tokens": batch_size * sequence_length,
        "hidden_size": config.hidden_size,
        "head_size": HEAD_SIZE,
        "head_count": config.hidden_size // HEAD_SIZE,
        "loss_boundary": "output plus FP32 final-state upstream gradient",
        "seed": seed,
        "providers": {
            name: {
                "configuration": provider.configuration,
                "correctness": correctness[name],
                "measurement_valid": measurements[name] is not None,
                "measurements": measurements[name],
            }
            for name, provider in providers.items()
        },
        "profile_range": profile_range,
    }
    del inputs, expected, reference, providers
    torch.cuda.empty_cache()
    return result


def run(config: BenchmarkConfig) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if config.hidden_size <= 0 or config.hidden_size % HEAD_SIZE:
        raise ValueError("hidden_size must be a positive multiple of 64")
    if config.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if config.samples <= 0 or config.profile_iterations <= 0:
        raise ValueError("sample counts must be positive")
    if (
        config.profile_provider is not None
        and config.profile_provider not in config.providers
    ):
        raise ValueError("profile provider must be enabled")
    if (
        config.profile_provider is not None
        and config.profile_case not in config.profiles
    ):
        raise ValueError("profile case must be enabled")
    if (
        config.profile_provider is not None
        and config.profile_dtype not in config.dtypes
    ):
        raise ValueError("profile dtype must be enabled")

    fixture = json.loads(TOLERANCE_PATH.read_text(encoding="utf-8"))
    limits = fixture["fp32io16_chunk"]
    results: list[dict[str, object]] = []
    counter = 0
    for profile_name in config.profiles:
        for dtype_name in config.dtypes:
            results.append(
                _run_case(
                    profile_name,
                    dtype_name,
                    config,
                    limits,
                    seed=config.seed + 1009 * counter,
                )
            )
            counter += 1

    provider_cases = [
        provider
        for result in results
        for provider in result["providers"].values()
    ]
    all_correct = all(
        provider["correctness"]["passed"]
        for provider in provider_cases
    )
    source = _source_metadata()
    source["training_benchmark_script_sha256"] = _sha256(
        Path(__file__).resolve()
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "flash_rwkv_fixed_training",
        "operator_boundary": (
            "public provider forward, output plus final-state loss, and "
            "torch.autograd.grad for all six token inputs and initial state"
        ),
        "synchronization_policy": (
            "CUDA start/end events around each complete forward/backward "
            "provider call and end-event synchronization for every sample"
        ),
        "warmup_iters": config.warmup_iters,
        "samples": config.samples,
        "measurement_enabled": config.measure,
        "seed": config.seed,
        "hardware": _hardware_metadata(),
        "source": source,
        "result_count": len(results),
        "provider_case_count": len(provider_cases),
        "all_cases_correct": all_correct,
        "results": results,
    }
    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _default_output() -> Path | None:
    run_log_dir = os.environ.get("REMOTE_RUN_LOG_DIR")
    return (
        Path(run_log_dir) / "flash-rwkv-training.json"
        if run_log_dir
        else None
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILES),
        default=list(PROFILES),
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=tuple(DTYPES),
        default=list(DTYPES),
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDER_NAMES,
        default=list(PROVIDER_NAMES),
    )
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, default=_default_output())
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument(
        "--profile-provider",
        choices=PROVIDER_NAMES,
    )
    parser.add_argument("--profile-case", choices=tuple(PROFILES))
    parser.add_argument(
        "--profile-dtype",
        choices=tuple(DTYPES),
        default="float16",
    )
    parser.add_argument("--profile-iterations", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if (arguments.profile_provider is None) != (
        arguments.profile_case is None
    ):
        raise SystemExit(
            "--profile-provider and --profile-case must be supplied together"
        )
    config = BenchmarkConfig(
        hidden_size=arguments.hidden_size,
        profiles=tuple(arguments.profiles),
        dtypes=tuple(arguments.dtypes),
        providers=tuple(arguments.providers),
        warmup_iters=arguments.warmup_iters,
        samples=arguments.samples,
        seed=arguments.seed,
        output=arguments.output,
        measure=not arguments.correctness_only,
        profile_provider=arguments.profile_provider,
        profile_case=arguments.profile_case,
        profile_dtype=arguments.profile_dtype,
        profile_iterations=arguments.profile_iterations,
    )
    payload = run(config)
    print(
        json.dumps(
            {
                "output": (
                    None if config.output is None else str(config.output)
                ),
                "result_count": payload["result_count"],
                "provider_case_count": payload["provider_case_count"],
                "all_cases_correct": payload["all_cases_correct"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not payload["all_cases_correct"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
