# SPDX-License-Identifier: MIT
"""Correctness-gated materialized versus factor-recompute chunk benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional

from flash_rwkv import _C
from flash_rwkv.config import (
    ChunkConfig,
    chunk_tuning_key,
    select_chunk_config,
)

HEAD_SIZE = 64
SOURCE_ROOT = Path(__file__).resolve().parents[3]
TOLERANCE_PATH = SOURCE_ROOT / "tests/fixtures/tolerances-v1.json"
PROFILES: dict[str, tuple[str, tuple[int, ...]]] = {
    "fixed_medium": ("fixed", (64,)),
    "fixed_long": ("fixed", (256,)),
    "fixed_very_long": ("fixed", (1024,)),
    "packed_medium": ("packed", (17, 32, 64)),
    "packed_long": ("packed", (65, 128, 256)),
    "packed_very_long": ("packed", (257, 512, 1024)),
}
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
SOURCE_PATHS = (
    Path("setup.py"),
    Path("csrc/bindings.cpp"),
    Path("csrc/rl_infctx/wkv7/rl_infctx_smxx_chunk_fp32io16_forward_materialized.cu"),
    Path("csrc/rl_infctx/wkv7/rl_infctx_smxx_chunk_fp32io16_forward_recompute.cu"),
    Path("csrc/rl_infctx/wkv7/rl_infctx_smxx_chunk_fp32io16_backward_replay.cuh"),
    Path("csrc/rl_infctx/wkv7/rl_infctx_smxx_chunk_fp32io16_backward_replay.cu"),
    Path("flash_rwkv/config.py"),
)


@dataclass(frozen=True)
class BenchmarkConfig:
    hidden_size: int
    profiles: tuple[str, ...]
    dtypes: tuple[str, ...]
    warmup_iters: int
    samples: int
    tuning_samples: int
    seed: int
    output: Path | None
    profile_strategy: str | None
    profile_iterations: int


@dataclass(frozen=True)
class Inputs:
    seq_lens: tuple[int, ...]
    offsets: torch.Tensor
    state_indices: torch.Tensor
    initial_state: torch.Tensor
    r: torch.Tensor
    log_decay: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor

    @property
    def num_heads(self) -> int:
        return self.r.shape[1]

    @property
    def total_tokens(self) -> int:
        return self.r.shape[0]


@dataclass
class Strategy:
    name: str
    state: torch.Tensor
    output: torch.Tensor
    workspace_bytes: int
    config: dict[str, int] | None
    call: Callable[[], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repository: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_metadata() -> dict[str, object]:
    hashes = {
        str(path): _sha256(SOURCE_ROOT / path)
        for path in SOURCE_PATHS
    }
    extension_path = Path(_C.__file__).resolve()
    leaf_status = _git_output(SOURCE_ROOT, "status", "--short")
    return {
        "leaf_revision": _git_output(SOURCE_ROOT, "rev-parse", "HEAD"),
        "leaf_dirty": None if leaf_status is None else bool(leaf_status),
        "benchmark_sha256": _sha256(Path(__file__).resolve()),
        "source_sha256": hashes,
        "source_set_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "extension_path": str(extension_path),
        "extension_sha256": _sha256(extension_path),
        "tolerance_fixture_sha256": _sha256(TOLERANCE_PATH),
    }


def _hardware_metadata() -> dict[str, object]:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device_index": index,
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _make_inputs(
    seq_lens: tuple[int, ...],
    *,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
) -> Inputs:
    total_tokens = sum(seq_lens)
    num_heads = hidden_size // HEAD_SIZE
    token_shape = (total_tokens, num_heads, HEAD_SIZE)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def normal(scale: float) -> torch.Tensor:
        return scale * torch.randn(
            token_shape,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )

    direction = functional.normalize(normal(1.0), dim=-1)
    strength = 0.1 * torch.rand(
        token_shape,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    log_decay = -0.05 - 0.15 * torch.rand(
        token_shape,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    return Inputs(
        seq_lens=seq_lens,
        offsets=torch.tensor(offsets, device="cuda", dtype=torch.int32),
        state_indices=torch.arange(
            len(seq_lens),
            device="cuda",
            dtype=torch.int32,
        ),
        initial_state=0.02
        * torch.randn(
            (
                len(seq_lens),
                num_heads,
                HEAD_SIZE,
                HEAD_SIZE,
            ),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        ),
        r=normal(0.05).to(dtype),
        log_decay=log_decay.to(dtype),
        k=normal(0.05).to(dtype),
        v=normal(0.05).to(dtype),
        a=(-direction).to(dtype),
        b=(direction * strength).to(dtype),
    )


def _chunk_metadata(
    seq_lens: tuple[int, ...],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_chunk_offsets = [0]
    chunk_token_starts: list[int] = []
    chunk_token_ends: list[int] = []
    token_start = 0
    for length in seq_lens:
        token_end = token_start + length
        for chunk_start in range(token_start, token_end, chunk_size):
            chunk_token_starts.append(chunk_start)
            chunk_token_ends.append(min(chunk_start + chunk_size, token_end))
        sequence_chunk_offsets.append(len(chunk_token_starts))
        token_start = token_end
    return (
        torch.tensor(
            sequence_chunk_offsets,
            device="cuda",
            dtype=torch.int32,
        ),
        torch.tensor(
            chunk_token_starts,
            device="cuda",
            dtype=torch.int32,
        ),
        torch.tensor(
            chunk_token_ends,
            device="cuda",
            dtype=torch.int32,
        ),
    )


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    rmse = difference.square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return float((rmse / baseline).item())


def _recurrent_expected(inputs: Inputs) -> tuple[torch.Tensor, torch.Tensor]:
    state = inputs.initial_state.clone()
    output = torch.empty_like(inputs.v)
    _C.recurrent_fp32(
        inputs.offsets,
        inputs.state_indices,
        state,
        inputs.r,
        inputs.log_decay,
        inputs.k,
        inputs.v,
        inputs.a,
        inputs.b,
        output,
        1.0,
    )
    torch.cuda.synchronize()
    return output, state


def _materialized_strategy(
    inputs: Inputs,
    config: ChunkConfig,
) -> Strategy:
    sequence_chunk_offsets, chunk_starts, chunk_ends = _chunk_metadata(
        inputs.seq_lens,
        config.chunk_size,
    )
    shape = (
        chunk_starts.numel(),
        inputs.num_heads,
        HEAD_SIZE,
        HEAD_SIZE,
    )
    transform = torch.empty(shape, dtype=torch.float32, device="cuda")
    bias = torch.empty_like(transform)
    boundary = torch.empty_like(transform)
    state = inputs.initial_state.clone()
    output = torch.empty_like(inputs.v)

    def call() -> None:
        _C.materialized_chunk_fp32(
            sequence_chunk_offsets,
            chunk_starts,
            chunk_ends,
            inputs.state_indices,
            state,
            inputs.r,
            inputs.log_decay,
            inputs.k,
            inputs.v,
            inputs.a,
            inputs.b,
            output,
            transform,
            bias,
            boundary,
            config.build_warps,
            config.stages,
            config.state_tile,
            1.0,
        )

    return Strategy(
        name="materialized",
        state=state,
        output=output,
        workspace_bytes=(
            transform.numel() + bias.numel() + boundary.numel()
        )
        * transform.element_size(),
        config=asdict(config),
        call=call,
    )


def _recompute_strategy(inputs: Inputs, chunk_size: int) -> Strategy:
    sequence_chunk_offsets, chunk_starts, chunk_ends = _chunk_metadata(
        inputs.seq_lens,
        chunk_size,
    )
    boundary = torch.empty(
        (
            chunk_starts.numel(),
            inputs.num_heads,
            HEAD_SIZE,
            HEAD_SIZE,
        ),
        dtype=torch.float32,
        device="cuda",
    )
    state = inputs.initial_state.clone()
    output = torch.empty_like(inputs.v)

    def call() -> None:
        _C.recompute_chunk_fp32(
            sequence_chunk_offsets,
            chunk_starts,
            chunk_ends,
            inputs.state_indices,
            state,
            inputs.r,
            inputs.log_decay,
            inputs.k,
            inputs.v,
            inputs.a,
            inputs.b,
            output,
            boundary,
            1.0,
        )

    return Strategy(
        name="factor_recompute",
        state=state,
        output=output,
        workspace_bytes=boundary.numel() * boundary.element_size(),
        config={"chunk_size": chunk_size},
        call=call,
    )


def _recurrent_strategy(inputs: Inputs) -> Strategy:
    state = inputs.initial_state.clone()
    output = torch.empty_like(inputs.v)

    def call() -> None:
        _C.recurrent_fp32(
            inputs.offsets,
            inputs.state_indices,
            state,
            inputs.r,
            inputs.log_decay,
            inputs.k,
            inputs.v,
            inputs.a,
            inputs.b,
            output,
            1.0,
        )

    return Strategy(
        name="recurrent",
        state=state,
        output=output,
        workspace_bytes=0,
        config=None,
        call=call,
    )


def _check_strategy(
    strategy: Strategy,
    inputs: Inputs,
    expected_output: torch.Tensor,
    expected_state: torch.Tensor,
    limits: dict[str, float],
) -> dict[str, object]:
    strategy.state.copy_(inputs.initial_state)
    strategy.call()
    torch.cuda.synchronize()
    output_error = _relative_rmse(strategy.output, expected_output)
    state_error = _relative_rmse(strategy.state, expected_state)
    passed = bool(
        torch.isfinite(strategy.output).all().item()
        and torch.isfinite(strategy.state).all().item()
        and output_error <= limits["output_relative_rmse"]
        and state_error <= limits["state_relative_rmse"]
    )
    return {
        "passed": passed,
        "output_relative_rmse": output_error,
        "state_relative_rmse": state_error,
        "output_relative_rmse_limit": limits["output_relative_rmse"],
        "state_relative_rmse_limit": limits["state_relative_rmse"],
    }


def _measure(
    strategy: Strategy,
    inputs: Inputs,
    *,
    warmup_iters: int,
    samples: int,
) -> list[float]:
    for _ in range(warmup_iters):
        strategy.state.copy_(inputs.initial_state)
        strategy.call()
    torch.cuda.synchronize()
    measurements: list[float] = []
    for _ in range(samples):
        strategy.state.copy_(inputs.initial_state)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        strategy.call()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end))
    return measurements


def _select_recompute(
    inputs: Inputs,
    expected_output: torch.Tensor,
    expected_state: torch.Tensor,
    limits: dict[str, float],
    config: BenchmarkConfig,
) -> tuple[Strategy, list[dict[str, object]]]:
    candidates: list[tuple[Strategy, dict[str, object]]] = []
    for chunk_size in (16, 32, 64):
        strategy = _recompute_strategy(inputs, chunk_size)
        correctness = _check_strategy(
            strategy,
            inputs,
            expected_output,
            expected_state,
            limits,
        )
        samples = (
            _measure(
                strategy,
                inputs,
                warmup_iters=config.warmup_iters,
                samples=config.tuning_samples,
            )
            if correctness["passed"]
            else []
        )
        candidates.append(
            (
                strategy,
                {
                    "config": strategy.config,
                    "workspace_bytes": strategy.workspace_bytes,
                    "correctness": correctness,
                    "raw_tuning_samples_ms": samples,
                    "p50_tuning_ms": (
                        statistics.median(samples) if samples else None
                    ),
                },
            )
        )
    correct = [
        pair
        for pair in candidates
        if pair[1]["correctness"]["passed"]
        and pair[1]["p50_tuning_ms"] is not None
    ]
    if not correct:
        raise RuntimeError("no correct factor-recompute candidate")
    winner, _ = min(
        correct,
        key=lambda pair: float(pair[1]["p50_tuning_ms"]),
    )
    return winner, [result for _, result in candidates]


def _run_profile(
    name: str,
    layout: str,
    seq_lens: tuple[int, ...],
    dtype_name: str,
    config: BenchmarkConfig,
    limits: dict[str, float],
    seed: int,
) -> dict[str, object]:
    inputs = _make_inputs(
        seq_lens,
        hidden_size=config.hidden_size,
        dtype=DTYPES[dtype_name],
        seed=seed,
    )
    expected_output, expected_state = _recurrent_expected(inputs)
    key = chunk_tuning_key(
        inputs.r,
        mode="fp32io16",
        packed=layout == "packed",
        max_sequence_length=max(seq_lens),
    )
    materialized_config = select_chunk_config(key).config
    materialized = _materialized_strategy(inputs, materialized_config)
    recompute, recompute_candidates = _select_recompute(
        inputs,
        expected_output,
        expected_state,
        limits,
        config,
    )
    recurrent = _recurrent_strategy(inputs)
    strategies = {
        strategy.name: strategy
        for strategy in (materialized, recompute, recurrent)
    }
    correctness = {
        strategy.name: _check_strategy(
            strategy,
            inputs,
            expected_output,
            expected_state,
            limits,
        )
        for strategy in strategies.values()
    }
    if not all(result["passed"] for result in correctness.values()):
        raise RuntimeError(f"correctness failed for {name}/{dtype_name}")

    for strategy in strategies.values():
        for _ in range(config.warmup_iters):
            strategy.state.copy_(inputs.initial_state)
            strategy.call()
    torch.cuda.synchronize()

    samples = {name: [] for name in strategies}
    ordering = list(strategies)
    generator = random.Random(seed + 7919)
    for _ in range(config.samples):
        generator.shuffle(ordering)
        for strategy_name in ordering:
            strategy = strategies[strategy_name]
            strategy.state.copy_(inputs.initial_state)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            strategy.call()
            end.record()
            end.synchronize()
            samples[strategy_name].append(start.elapsed_time(end))

    measurements = {
        strategy_name: {
            "config": strategies[strategy_name].config,
            "workspace_bytes": strategies[strategy_name].workspace_bytes,
            "workspace_bytes_per_token": (
                strategies[strategy_name].workspace_bytes
                / inputs.total_tokens
            ),
            "correctness": correctness[strategy_name],
            "raw_samples_ms": values,
            "p50_ms": statistics.median(values),
        }
        for strategy_name, values in samples.items()
    }
    materialized_p50 = float(measurements["materialized"]["p50_ms"])
    recompute_p50 = float(measurements["factor_recompute"]["p50_ms"])
    recurrent_p50 = float(measurements["recurrent"]["p50_ms"])
    materialized_workspace = int(
        measurements["materialized"]["workspace_bytes"]
    )
    recompute_workspace = int(
        measurements["factor_recompute"]["workspace_bytes"]
    )

    profile_range = None
    if config.profile_strategy is not None:
        strategy = strategies[config.profile_strategy]
        range_name = (
            f"flash_rwkv.{config.profile_strategy}."
            f"{name}.{dtype_name}"
        )
        cudart = torch.cuda.cudart()
        cudart.cudaProfilerStart()
        try:
            torch.cuda.nvtx.range_push(range_name)
            try:
                for _ in range(config.profile_iterations):
                    strategy.state.copy_(inputs.initial_state)
                    strategy.call()
                torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
        finally:
            cudart.cudaProfilerStop()
        profile_range = {
            "name": range_name,
            "iterations": config.profile_iterations,
        }

    result = {
        "profile": name,
        "layout": layout,
        "dtype": dtype_name,
        "seq_lens": list(seq_lens),
        "total_tokens": inputs.total_tokens,
        "hidden_size": config.hidden_size,
        "head_count": inputs.num_heads,
        "tuning_key": key.identifier,
        "recompute_candidates": recompute_candidates,
        "measurements": measurements,
        "comparison": {
            "materialized_speedup_vs_recompute": (
                recompute_p50 / materialized_p50
            ),
            "materialized_speedup_vs_recurrent": (
                recurrent_p50 / materialized_p50
            ),
            "recompute_speedup_vs_recurrent": (
                recurrent_p50 / recompute_p50
            ),
            "recompute_workspace_fraction": (
                recompute_workspace / materialized_workspace
            ),
            "workspace_bytes_saved_by_recompute": (
                materialized_workspace - recompute_workspace
            ),
        },
        "profile_range": profile_range,
    }
    del inputs, expected_output, expected_state, strategies
    torch.cuda.empty_cache()
    return result


def run(config: BenchmarkConfig) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if config.hidden_size <= 0 or config.hidden_size % HEAD_SIZE:
        raise ValueError("hidden_size must be a positive multiple of 64")
    if config.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if config.samples <= 0 or config.tuning_samples <= 0:
        raise ValueError("sample counts must be positive")
    if config.profile_iterations <= 0:
        raise ValueError("profile_iterations must be positive")

    fixture = json.loads(TOLERANCE_PATH.read_text(encoding="utf-8"))
    limits = fixture["fp32io16_chunk"]
    results: list[dict[str, object]] = []
    counter = 0
    for profile_name in config.profiles:
        layout, seq_lens = PROFILES[profile_name]
        for dtype_name in config.dtypes:
            results.append(
                _run_profile(
                    profile_name,
                    layout,
                    seq_lens,
                    dtype_name,
                    config,
                    limits,
                    config.seed + 1009 * counter,
                )
            )
            counter += 1

    payload: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "flash_rwkv_chunk_strategy_comparison",
        "measurement_boundary": (
            "preallocated low-level operator; state reset is ordered before "
            "the CUDA start event; strategy order is shuffled per sample"
        ),
        "correctness_baseline": "validated recurrent_fp32 low-level operator",
        "warmup_iters": config.warmup_iters,
        "samples": config.samples,
        "recompute_tuning_samples": config.tuning_samples,
        "seed": config.seed,
        "hardware": _hardware_metadata(),
        "source": _source_metadata(),
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
        Path(run_log_dir) / "flash-rwkv-chunk-strategies.json"
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
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--tuning-samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, default=_default_output())
    parser.add_argument(
        "--profile-strategy",
        choices=("materialized", "factor_recompute", "recurrent"),
    )
    parser.add_argument("--profile-iterations", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(
        BenchmarkConfig(
            hidden_size=args.hidden_size,
            profiles=tuple(args.profiles),
            dtypes=tuple(args.dtypes),
            warmup_iters=args.warmup_iters,
            samples=args.samples,
            tuning_samples=args.tuning_samples,
            seed=args.seed,
            output=args.output,
            profile_strategy=args.profile_strategy,
            profile_iterations=args.profile_iterations,
        )
    )
    print(
        json.dumps(
            {
                "benchmark": payload["benchmark"],
                "profiles": len(payload["results"]),
                "output": str(args.output) if args.output else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
