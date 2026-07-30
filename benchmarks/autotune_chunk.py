# SPDX-License-Identifier: MIT
"""Offline correctness-gated autotuner for materialized chunk variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional

from flash_rwkv import _C
from flash_rwkv.config import (
    ChunkConfig,
    chunk_tuning_key,
    config_as_dict,
    enumerate_chunk_configs,
)


HEAD_SIZE = 64
SOURCE_ROOT = Path(__file__).resolve().parents[1]
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
NATIVE_SOURCE_PATHS = (
    Path("setup.py"),
    Path("csrc/bindings.cpp"),
    Path("csrc/chunk/materialized_fp32.cu"),
    Path("csrc/chunk/replay.h"),
    Path("csrc/chunk/replay_fp32.cu"),
    Path("flash_rwkv/config.py"),
    Path("flash_rwkv/ops.py"),
)


@dataclass(frozen=True)
class TuningConfig:
    hidden_size: int
    profiles: tuple[str, ...]
    dtypes: tuple[str, ...]
    warmup_iters: int
    samples: int
    seed: int
    output: Path | None


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_metadata() -> dict[str, object]:
    hashes = {
        str(path): _sha256(SOURCE_ROOT / path)
        for path in NATIVE_SOURCE_PATHS
    }
    extension_path = Path(_C.__file__).resolve()
    return {
        "autotuner_sha256": _sha256(Path(__file__).resolve()),
        "source_sha256": hashes,
        "source_set_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
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
    shape = (total_tokens, num_heads, HEAD_SIZE)
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
    log_decay = -0.05 - 0.15 * torch.rand(
        shape,
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


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    rmse = difference.square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return float((rmse / baseline).item())


def _run_candidate(
    inputs: Inputs,
    config: ChunkConfig,
    expected_output: torch.Tensor,
    expected_state: torch.Tensor,
    tuning: TuningConfig,
    limits: dict[str, float],
) -> dict[str, object]:
    sequence_chunk_offsets, chunk_starts, chunk_ends = _chunk_metadata(
        inputs.seq_lens,
        config.chunk_size,
    )
    num_chunks = chunk_starts.numel()
    workspace_shape = (
        num_chunks,
        inputs.num_heads,
        HEAD_SIZE,
        HEAD_SIZE,
    )
    transform = torch.empty(
        workspace_shape,
        dtype=torch.float32,
        device="cuda",
    )
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

    state.copy_(inputs.initial_state)
    call()
    torch.cuda.synchronize()
    output_error = _relative_rmse(output, expected_output)
    state_error = _relative_rmse(state, expected_state)
    passed = bool(
        torch.isfinite(output).all().item()
        and torch.isfinite(state).all().item()
        and output_error <= limits["output_relative_rmse"]
        and state_error <= limits["state_relative_rmse"]
    )
    result: dict[str, object] = {
        "config": config_as_dict(config),
        "identifier": config.identifier,
        "workspace_bytes": (
            transform.numel()
            + bias.numel()
            + boundary.numel()
        )
        * transform.element_size(),
        "correctness": {
            "passed": passed,
            "output_relative_rmse": output_error,
            "state_relative_rmse": state_error,
            "output_relative_rmse_limit": limits[
                "output_relative_rmse"
            ],
            "state_relative_rmse_limit": limits[
                "state_relative_rmse"
            ],
        },
        "raw_samples_ms": None,
        "p50_ms": None,
    }
    if not passed:
        return result

    for _ in range(tuning.warmup_iters):
        state.copy_(inputs.initial_state)
        call()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(tuning.samples):
        state.copy_(inputs.initial_state)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    result["raw_samples_ms"] = samples
    result["p50_ms"] = statistics.median(samples)
    return result


def _run_profile(
    name: str,
    layout: str,
    seq_lens: tuple[int, ...],
    dtype_name: str,
    tuning: TuningConfig,
    limits: dict[str, float],
    seed: int,
) -> tuple[dict[str, object], str, dict[str, object]]:
    dtype = DTYPES[dtype_name]
    inputs = _make_inputs(
        seq_lens,
        hidden_size=tuning.hidden_size,
        dtype=dtype,
        seed=seed,
    )
    expected_output, expected_state = _recurrent_expected(inputs)
    candidates = list(enumerate_chunk_configs())
    random.Random(seed).shuffle(candidates)
    results = [
        _run_candidate(
            inputs,
            candidate,
            expected_output,
            expected_state,
            tuning,
            limits,
        )
        for candidate in candidates
    ]
    passed = [
        result
        for result in results
        if result["correctness"]["passed"]
        and result["p50_ms"] is not None
    ]
    if not passed:
        raise RuntimeError(f"no correct chunk candidate for {name}/{dtype_name}")
    winner = min(passed, key=lambda result: float(result["p50_ms"]))
    key = chunk_tuning_key(
        inputs.r,
        mode="fp32io16",
        packed=layout == "packed",
        max_sequence_length=max(seq_lens),
    )
    cache_entry = {
        "config": winner["config"],
        "profile": name,
        "p50_ms": winner["p50_ms"],
        "correct_candidates": len(passed),
    }
    profile_result = {
        "profile": name,
        "layout": layout,
        "dtype": dtype_name,
        "seq_lens": list(seq_lens),
        "total_tokens": sum(seq_lens),
        "hidden_size": tuning.hidden_size,
        "head_count": tuning.hidden_size // HEAD_SIZE,
        "tuning_key": key.identifier,
        "winner": winner,
        "candidate_count": len(results),
        "correct_candidate_count": len(passed),
        "candidates": results,
    }
    del inputs, expected_output, expected_state
    torch.cuda.empty_cache()
    return profile_result, key.identifier, cache_entry


def run(tuning: TuningConfig) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if tuning.hidden_size <= 0 or tuning.hidden_size % HEAD_SIZE:
        raise ValueError("hidden_size must be a positive multiple of 64")
    if tuning.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if tuning.samples <= 0:
        raise ValueError("samples must be positive")

    fixture = json.loads(TOLERANCE_PATH.read_text(encoding="utf-8"))
    limits = fixture["fp32io16_chunk"]
    results: list[dict[str, object]] = []
    entries: dict[str, object] = {}
    profile_counter = 0
    for profile_name in tuning.profiles:
        layout, seq_lens = PROFILES[profile_name]
        for dtype_name in tuning.dtypes:
            result, key, entry = _run_profile(
                profile_name,
                layout,
                seq_lens,
                dtype_name,
                tuning,
                limits,
                tuning.seed + 1009 * profile_counter,
            )
            profile_counter += 1
            if key in entries:
                raise RuntimeError(f"duplicate tuning key: {key}")
            entries[key] = entry
            results.append(result)

    payload: dict[str, object] = {
        "schema_version": 1,
        "autotuner": "flash_rwkv_materialized_chunk",
        "measurement_boundary": (
            "preallocated low-level three-kernel operator; state reset is "
            "ordered before the start event"
        ),
        "correctness_baseline": "validated recurrent_fp32 low-level operator",
        "candidate_space": [
            config_as_dict(config) for config in enumerate_chunk_configs()
        ],
        "warmup_iters": tuning.warmup_iters,
        "samples": tuning.samples,
        "seed": tuning.seed,
        "hardware": _hardware_metadata(),
        "source": _source_metadata(),
        "cache": {
            "schema_version": 1,
            "source": "FlashRWKV canonical-device offline autotuning",
            "entries": entries,
        },
        "results": results,
    }
    if tuning.output is not None:
        tuning.output.parent.mkdir(parents=True, exist_ok=True)
        tuning.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _default_output() -> Path | None:
    run_log_dir = os.environ.get("REMOTE_RUN_LOG_DIR")
    if not run_log_dir:
        return None
    return Path(run_log_dir) / "flash-rwkv-chunk-autotune.json"


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
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, default=_default_output())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(
        TuningConfig(
            hidden_size=args.hidden_size,
            profiles=tuple(args.profiles),
            dtypes=tuple(args.dtypes),
            warmup_iters=args.warmup_iters,
            samples=args.samples,
            seed=args.seed,
            output=args.output,
        )
    )
    print(
        json.dumps(
            {
                "autotuner": payload["autotuner"],
                "keys": len(payload["cache"]["entries"]),
                "output": str(args.output) if args.output else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
