# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-gated JSON benchmark for FlashRWKV recurrent kernels.

The profile set is adapted from vllm-rwkv's canonical packed-varlen RWKV-7
benchmark at commit 6d683f9e49a2997e405c47edc147872c8609513b. FlashRWKV
uses explicit log-decay, canonical [K,V] state, and separate functional and
stateful measurement boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from flash_rwkv import _C, rwkv7_recurrent_stateful

HEAD_SIZE = 64
PROFILE_LENGTHS: dict[str, tuple[int, ...]] = {
    "decode_b1": (1,),
    "decode_b16": (1,) * 16,
    "decode_b32": (1,) * 32,
    "decode_b64": (1,) * 64,
    "decode_b128": (1,) * 128,
    "decode_b320": (1,) * 320,
    "decode_b2048": (1,) * 2048,
    "equal_chunk16_b320": (16,) * 320,
    "ragged_chunk16_b320": tuple(range(1, 17)) * 20,
    "ragged_long_b32": (1, 4, 8, 16, 32, 64, 96, 128) * 4,
    "ragged_skew_b32": (128,) + (1,) * 31,
}
MODES = ("fp32io16", "fp16")
SOURCE_ROOT = Path(__file__).resolve().parents[3]
TOLERANCE_PATH = SOURCE_ROOT / "tests/fixtures/tolerances-v1.json"
NATIVE_SOURCE_PATHS = (
    Path(".github/workflows/pro6000-gpu.yml"),
    Path("pyproject.toml"),
    Path("setup.py"),
    Path("csrc/bindings.cpp"),
    Path("csrc/validation/recurrent_metadata.cu"),
    Path("csrc/infer/wkv7/infer_common_recurrent_varlen_bindings.cpp"),
    Path("csrc/infer/wkv7/infer_common_recurrent_fp32io16_forward_varlen.cu"),
    Path("csrc/infer/wkv7/infer_sm80_recurrent_fp16_forward_varlen.cu"),
    Path("flash_rwkv/__init__.py"),
    Path("flash_rwkv/_extension.py"),
    Path("flash_rwkv/architecture.py"),
    Path("flash_rwkv/ops.py"),
    Path("flash_rwkv/validation.py"),
)
CONTROLLED_GENERATED_PATH_PATTERNS = (
    ".venv/**",
    "artifacts/**",
    "build/**",
    "*.egg-info/**",
    "*.so",
    ".cache/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    "__pycache__/**",
    "**/__pycache__/**",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    hidden_size: int
    profiles: tuple[str, ...]
    modes: tuple[str, ...]
    warmup_iters: int
    samples: int
    stateful_sample_iters: int
    seed: int
    output: Path | None
    measure: bool


@dataclass(frozen=True)
class ProfilePayload:
    seq_lens: tuple[int, ...]
    query_start_loc: torch.Tensor
    state_indices: torch.Tensor
    initial_state: torch.Tensor
    r: torch.Tensor
    log_decay: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    output: torch.Tensor

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)


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
    if result.returncode:
        return None
    return result.stdout.strip()


def _revision_metadata() -> dict[str, object]:
    parent_root = SOURCE_ROOT.parents[2]
    leaf_status = _git_output(
        SOURCE_ROOT,
        "status",
        "--short",
        "--untracked-files=no",
    )
    parent_status = _git_output(
        parent_root,
        "status",
        "--short",
        "--ignore-submodules=none",
    )
    return {
        "leaf_revision": _git_output(SOURCE_ROOT, "rev-parse", "HEAD"),
        "leaf_dirty": None if leaf_status is None else bool(leaf_status),
        "tracked_source_dirty": None if leaf_status is None else bool(leaf_status),
        "tracked_status_entries": None if leaf_status is None else leaf_status.splitlines(),
        "controlled_generated_path_patterns": CONTROLLED_GENERATED_PATH_PATTERNS,
        "parent_revision": _git_output(parent_root, "rev-parse", "HEAD"),
        "parent_dirty": None if parent_status is None else bool(parent_status),
    }


def _source_metadata() -> dict[str, object]:
    source_hashes = {
        str(relative): _sha256(SOURCE_ROOT / relative)
        for relative in NATIVE_SOURCE_PATHS
    }
    extension_path = Path(_C.__file__).resolve()
    return {
        **_revision_metadata(),
        "benchmark_script_sha256": _sha256(Path(__file__).resolve()),
        "source_sha256": source_hashes,
        "source_set_sha256": hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True).encode()
        ).hexdigest(),
        "tolerance_fixture": str(TOLERANCE_PATH.relative_to(SOURCE_ROOT)),
        "tolerance_fixture_sha256": _sha256(TOLERANCE_PATH),
        "extension_path": str(extension_path),
        "extension_sha256": _sha256(extension_path),
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


def _load_tolerances() -> dict[str, Any]:
    return json.loads(TOLERANCE_PATH.read_text(encoding="utf-8"))


def _mode_tolerances(
    tolerances: dict[str, Any],
    mode: str,
) -> tuple[float, float] | None:
    entry = tolerances.get(
        "fp32io16_recurrent" if mode == "fp32io16" else "fp16"
    )
    if not isinstance(entry, dict):
        return None
    output_limit = entry.get("output_relative_rmse")
    state_limit = entry.get("state_relative_rmse")
    if not isinstance(output_limit, (int, float)) or not isinstance(
        state_limit, (int, float)
    ):
        return None
    return float(output_limit), float(state_limit)


def _profile_seed(seed: int, profile_index: int) -> int:
    return seed + 1009 * profile_index


def _make_payload(
    seq_lens: tuple[int, ...],
    *,
    hidden_size: int,
    mode: str,
    seed: int,
) -> ProfilePayload:
    total_tokens = sum(seq_lens)
    num_heads = hidden_size // HEAD_SIZE
    num_sequences = len(seq_lens)
    num_slots = num_sequences + 7
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def normal(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return scale * torch.randn(
            shape,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )

    token_shape = (total_tokens, num_heads, HEAD_SIZE)
    r = normal(token_shape, 0.02).to(torch.float16)
    log_decay = (
        -0.05
        - 0.15
        * torch.rand(
            token_shape,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
    ).to(torch.float16)
    k = normal(token_shape, 0.02).to(torch.float16)
    v = normal(token_shape, 0.02).to(torch.float16)
    a = normal(token_shape, 0.02).to(torch.float16)
    b = normal(token_shape, 0.02).to(torch.float16)

    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    query_start_loc = torch.tensor(
        offsets,
        dtype=torch.int32,
        device="cuda",
    )
    state_indices = torch.arange(
        num_sequences - 1,
        -1,
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
    initial_state = normal(
        (num_slots, num_heads, HEAD_SIZE, HEAD_SIZE),
        0.02,
    ).to(state_dtype)
    return ProfilePayload(
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        initial_state=initial_state,
        r=r,
        log_decay=log_decay,
        k=k,
        v=v,
        a=a,
        b=b,
        output=torch.empty_like(v),
    )


def _reference(
    payload: ProfilePayload,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_pool = payload.initial_state.float().clone()
    output = torch.empty_like(payload.v, dtype=torch.float32)
    offsets = payload.query_start_loc.cpu().tolist()
    state_indices = payload.state_indices.long()
    max_length = max(payload.seq_lens)

    for token_offset in range(max_length):
        active_sequences = [
            index
            for index, length in enumerate(payload.seq_lens)
            if token_offset < length
        ]
        sequence_indices = torch.tensor(
            active_sequences,
            dtype=torch.long,
            device="cuda",
        )
        token_indices = torch.tensor(
            [offsets[index] + token_offset for index in active_sequences],
            dtype=torch.long,
            device="cuda",
        )
        slots = state_indices.index_select(0, sequence_indices)
        previous_state = state_pool.index_select(0, slots)
        r = payload.r.index_select(0, token_indices).float()
        log_decay = payload.log_decay.index_select(0, token_indices).float()
        k = payload.k.index_select(0, token_indices).float()
        v = payload.v.index_select(0, token_indices).float()
        a = payload.a.index_select(0, token_indices).float()
        b = payload.b.index_select(0, token_indices).float()

        a_state = torch.einsum("nhk,nhkv->nhv", a, previous_state)
        updated_state = (
            torch.exp(log_decay).unsqueeze(-1) * previous_state
            + b.unsqueeze(-1) * a_state.unsqueeze(-2)
            + k.unsqueeze(-1) * v.unsqueeze(-2)
        )
        token_output = float(scale) * torch.einsum(
            "nhk,nhkv->nhv",
            r,
            updated_state,
        )
        state_pool.index_copy_(0, slots, updated_state)
        output.index_copy_(0, token_indices, token_output)

    return output, state_pool


def _launch(
    payload: ProfilePayload,
    state: torch.Tensor,
    *,
    mode: str,
    scale: float,
) -> torch.Tensor:
    return rwkv7_recurrent_stateful(
        *(tensor.unsqueeze(0) for tensor in (
            payload.r,
            payload.log_decay,
            payload.k,
            payload.v,
            payload.a,
            payload.b,
        )),
        state_pool=state,
        cu_seqlens=payload.query_start_loc,
        state_indices=payload.state_indices,
        scale=scale,
        mode=mode,
    )


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    rmse = difference.square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return float((rmse / baseline).item())


def _max_absolute_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _correctness(
    payload: ProfilePayload,
    *,
    mode: str,
    scale: float,
    limits: tuple[float, float] | None,
) -> dict[str, object]:
    expected_output, expected_state = _reference(payload, scale=scale)
    first_state = payload.initial_state.clone()
    second_state = payload.initial_state.clone()
    first_output = _launch(payload, first_state, mode=mode, scale=scale)[0]
    second_output = _launch(payload, second_state, mode=mode, scale=scale)[0]
    torch.cuda.synchronize()

    active_slots = payload.state_indices.long()
    actual_active_state = first_state.index_select(0, active_slots)
    expected_active_state = expected_state.index_select(0, active_slots)
    untouched = torch.ones(
        payload.initial_state.shape[0],
        dtype=torch.bool,
        device="cuda",
    )
    untouched[active_slots] = False
    output_error = _relative_rmse(first_output, expected_output)
    state_error = _relative_rmse(actual_active_state, expected_active_state)
    finite = bool(
        torch.isfinite(first_output).all().item()
        and torch.isfinite(actual_active_state).all().item()
    )
    deterministic = bool(
        torch.equal(first_output, second_output)
        and torch.equal(first_state, second_state)
    )
    untouched_preserved = bool(
        torch.equal(
            first_state[untouched],
            payload.initial_state[untouched],
        )
    )
    if limits is None:
        status = "report_only"
        passed = False
        output_limit = None
        state_limit = None
    else:
        output_limit, state_limit = limits
        passed = bool(
            finite
            and deterministic
            and untouched_preserved
            and output_error <= output_limit
            and state_error <= state_limit
        )
        status = "passed" if passed else "failed"
    return {
        "status": status,
        "passed": passed,
        "output_relative_rmse": output_error,
        "output_max_absolute_error": _max_absolute_error(
            first_output,
            expected_output,
        ),
        "final_state_relative_rmse": state_error,
        "final_state_max_absolute_error": _max_absolute_error(
            actual_active_state,
            expected_active_state,
        ),
        "output_relative_rmse_limit": output_limit,
        "final_state_relative_rmse_limit": state_limit,
        "finite": finite,
        "deterministic": deterministic,
        "untouched_slots_preserved": untouched_preserved,
    }


def _percentile(samples: Sequence[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _latency_summary(
    samples: list[float],
    *,
    total_tokens: int,
) -> dict[str, object]:
    median_ms = statistics.median(samples)
    return {
        "min": min(samples),
        "mean": statistics.fmean(samples),
        "p10": _percentile(samples, 0.10),
        "p50": median_ms,
        "p90": _percentile(samples, 0.90),
        "useful_tokens_per_s_p50": total_tokens * 1000.0 / median_ms,
        "raw_samples_ms": samples,
    }


def _measure_functional(
    payload: ProfilePayload,
    *,
    mode: str,
    scale: float,
    warmup_iters: int,
    samples: int,
) -> tuple[list[float], bool]:
    state = payload.initial_state.clone()
    output = payload.output
    for _ in range(warmup_iters):
        state.copy_(payload.initial_state)
        output = _launch(payload, state, mode=mode, scale=scale)[0]
    torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(samples):
        state.copy_(payload.initial_state)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = _launch(payload, state, mode=mode, scale=scale)[0]
        end.record()
        end.synchronize()
        latencies_ms.append(start.elapsed_time(end))
    return latencies_ms, bool(
        torch.isfinite(output).all().item()
        and torch.isfinite(
            state.index_select(0, payload.state_indices.long())
        )
        .all()
        .item()
    )


def _measure_stateful(
    payload: ProfilePayload,
    *,
    mode: str,
    scale: float,
    warmup_iters: int,
    samples: int,
    sample_iters: int,
) -> tuple[list[float], bool]:
    warmup_state = payload.initial_state.clone()
    output = payload.output
    for _ in range(warmup_iters):
        output = _launch(payload, warmup_state, mode=mode, scale=scale)[0]
    torch.cuda.synchronize()

    state = payload.initial_state.clone()
    latencies_ms: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(sample_iters):
            output = _launch(payload, state, mode=mode, scale=scale)[0]
        end.record()
        end.synchronize()
        latencies_ms.append(start.elapsed_time(end) / sample_iters)
    return latencies_ms, bool(
        torch.isfinite(output).all().item()
        and torch.isfinite(
            state.index_select(0, payload.state_indices.long())
        )
        .all()
        .item()
    )


def _cuda_graph_metadata_evidence(
    payload: ProfilePayload,
    *,
    mode: str,
    scale: float,
) -> dict[str, object]:
    state = payload.initial_state.clone()
    cu_seqlens_data_ptr = payload.query_start_loc.data_ptr()
    state_indices_data_ptr = payload.state_indices.data_ptr()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            _launch(payload, state, mode=mode, scale=scale)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = _launch(payload, state, mode=mode, scale=scale)
    graph.replay()
    torch.cuda.synchronize()
    return {
        "cuda_graph_capture_succeeded": True,
        "captured_output_finite": bool(torch.isfinite(captured_output).all().item()),
        "cu_seqlens_data_ptr": cu_seqlens_data_ptr,
        "state_indices_data_ptr": state_indices_data_ptr,
        "cu_seqlens_identity_preserved": (
            payload.query_start_loc.data_ptr() == cu_seqlens_data_ptr
        ),
        "state_indices_identity_preserved": (
            payload.state_indices.data_ptr() == state_indices_data_ptr
        ),
        "interpretation": (
            "successful CUDA graph capture rejects host synchronization in the "
            "public packed launch boundary"
        ),
    }


def _measurement(
    payload: ProfilePayload,
    config: BenchmarkConfig,
    *,
    mode: str,
    scale: float,
) -> dict[str, object]:
    functional_samples, functional_finite = _measure_functional(
        payload,
        mode=mode,
        scale=scale,
        warmup_iters=config.warmup_iters,
        samples=config.samples,
    )
    stateful_samples, stateful_finite = _measure_stateful(
        payload,
        mode=mode,
        scale=scale,
        warmup_iters=config.warmup_iters,
        samples=config.samples,
        sample_iters=config.stateful_sample_iters,
    )
    return {
        "functional_identical_input": {
            "boundary": (
                "state reset precedes the start event; one operator launch "
                "is timed per raw sample"
            ),
            "sample_iters": 1,
            "state_finite_after_measurement": functional_finite,
            "latency_ms": _latency_summary(
                functional_samples,
                total_tokens=payload.total_tokens,
            ),
        },
        "stateful_steady_state": {
            "boundary": (
                "the state pool evolves in place; each raw sample is the "
                "per-launch average of consecutive operator executions"
            ),
            "sample_iters": config.stateful_sample_iters,
            "state_finite_after_measurement": stateful_finite,
            "latency_ms": _latency_summary(
                stateful_samples,
                total_tokens=payload.total_tokens,
            ),
        },
    }


def _run_case(
    profile: str,
    mode: str,
    config: BenchmarkConfig,
    tolerances: dict[str, Any],
    *,
    seed: int,
) -> dict[str, object]:
    seq_lens = PROFILE_LENGTHS[profile]
    payload = _make_payload(
        seq_lens,
        hidden_size=config.hidden_size,
        mode=mode,
        seed=seed,
    )
    limits = _mode_tolerances(tolerances, mode)
    correctness = _correctness(
        payload,
        mode=mode,
        scale=1.0,
        limits=limits,
    )
    metadata_evidence = (
        _cuda_graph_metadata_evidence(payload, mode=mode, scale=1.0)
        if profile == config.profiles[0] and correctness["passed"]
        else None
    )
    measurement: dict[str, object] | None = None
    invalid_reason: str | None = None
    if correctness["passed"] and config.measure:
        measurement = _measurement(
            payload,
            config,
            mode=mode,
            scale=1.0,
        )
    elif correctness["passed"]:
        invalid_reason = "measurement disabled by --correctness-only"
    elif correctness["status"] == "report_only":
        invalid_reason = (
            "versioned tolerance fixture has no numeric gate for this mode"
        )
    else:
        invalid_reason = "correctness gate failed for this exact case"

    result = {
        "profile": profile,
        "mode": mode,
        "provider": (
            "flash_rwkv.rwkv7_recurrent_stateful"
        ),
        "state_dtype": (
            "float32" if mode == "fp32io16" else "float16"
        ),
        "token_dtype": "float16",
        "accumulation_policy": (
            "fp32 state and recurrence"
            if mode == "fp32io16"
            else "fp16 half2 state and recurrence"
        ),
        "hidden_size": config.hidden_size,
        "head_size": HEAD_SIZE,
        "head_count": config.hidden_size // HEAD_SIZE,
        "sequence_count": len(seq_lens),
        "total_tokens": sum(seq_lens),
        "seq_lens": list(seq_lens),
        "min_length": min(seq_lens),
        "max_length": max(seq_lens),
        "padding_ratio": 1.0
        - sum(seq_lens) / (len(seq_lens) * max(seq_lens)),
        "state_slot_mapping": "reverse sequence order with seven untouched rows",
        "state_pool_size": int(payload.initial_state.shape[0]),
        "metadata_validation_complexity": "O(sequence_count + state_pool_size)",
        "metadata_validation_strategy": "device_slot_claim_bitmap",
        "metadata_validation_workspace_bytes": int(
            (payload.initial_state.shape[0] + 1) * torch.int32.itemsize
        ),
        "metadata_host_round_trip": False,
        "kernel_launches_per_operator": 2,
        "seed": seed,
        "correctness": correctness,
        "device_metadata": {
            "cu_seqlens_dtype": str(payload.query_start_loc.dtype),
            "state_indices_dtype": str(payload.state_indices.dtype),
            "device": str(payload.query_start_loc.device),
            "cuda_graph_evidence": metadata_evidence,
        },
        "measurement_valid": measurement is not None,
        "invalid_reason": invalid_reason,
        "measurements": measurement,
    }
    del payload
    torch.cuda.empty_cache()
    return result


def run(config: BenchmarkConfig) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if config.hidden_size <= 0 or config.hidden_size % HEAD_SIZE:
        raise ValueError("hidden_size must be a positive multiple of 64")
    if config.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if config.samples <= 0:
        raise ValueError("samples must be positive")
    if config.stateful_sample_iters <= 0:
        raise ValueError("stateful_sample_iters must be positive")

    tolerances = _load_tolerances()
    results: list[dict[str, object]] = []
    for profile_index, profile in enumerate(config.profiles):
        seed = _profile_seed(config.seed, profile_index)
        for mode in config.modes:
            results.append(
                _run_case(
                    profile,
                    mode,
                    config,
                    tolerances,
                    seed=seed,
                )
            )

    correct_cases = sum(
        bool(result["correctness"]["passed"]) for result in results
    )
    valid_measurements = sum(
        bool(result["measurement_valid"]) for result in results
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "flash_rwkv_recurrent",
        "baseline_profile_source": {
            "repository": "https://github.com/BlinkDL/Albatross",
            "implementation_repository": (
                "https://github.com/rwkv-rs/vllm-rwkv"
            ),
            "implementation_revision": (
                "6d683f9e49a2997e405c47edc147872c8609513b"
            ),
            "profile_file": "benchmarks/kernels/benchmark_rwkv7_wkv.py",
        },
        "operator_boundary": (
            "public rwkv7_recurrent_stateful packed serving API; includes "
            "structural validation, device-side value and uniqueness validation, "
            "output allocation, Python dispatch, native launch, and device "
            "metadata pass-through; excludes input "
            "generation, strict debug metadata validation, compilation, "
            "autotune, state reset, logging, and serialization"
        ),
        "synchronization_policy": (
            "warmup followed by explicit synchronization; CUDA start/end "
            "events and end-event synchronization for every raw sample"
        ),
        "warmup_iters": config.warmup_iters,
        "samples": config.samples,
        "stateful_sample_iters": config.stateful_sample_iters,
        "measurement_enabled": config.measure,
        "seed": config.seed,
        "hardware": _hardware_metadata(),
        "source": _source_metadata(),
        "result_count": len(results),
        "correct_case_count": correct_cases,
        "valid_measurement_count": valid_measurements,
        "all_cases_correct": correct_cases == len(results),
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
    if not run_log_dir:
        return None
    return Path(run_log_dir) / "flash-rwkv-recurrent.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILE_LENGTHS),
        default=list(PROFILE_LENGTHS),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
    )
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--stateful-sample-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, default=_default_output())
    parser.add_argument("--correctness-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(
        BenchmarkConfig(
            hidden_size=args.hidden_size,
            profiles=tuple(args.profiles),
            modes=tuple(args.modes),
            warmup_iters=args.warmup_iters,
            samples=args.samples,
            stateful_sample_iters=args.stateful_sample_iters,
            seed=args.seed,
            output=args.output,
            measure=not args.correctness_only,
        )
    )
    for result in payload["results"]:
        measurements = result["measurements"]
        if measurements is None:
            continue
        latency = measurements["stateful_steady_state"]["latency_ms"]
        print(
            " ".join(
                (
                    f"RESULT B={result['sequence_count']}",
                    f"T={result['total_tokens']}",
                    f"iters={args.samples * args.stateful_sample_iters}",
                    f"p10_ms={latency['p10']:.6f}",
                    f"p50_ms={latency['p50']:.6f}",
                    f"p90_ms={latency['p90']:.6f}",
                    f"tok_s_p50={latency['useful_tokens_per_s_p50']:.6f}",
                    f"label=packed-{result['profile']}-{result['mode']}",
                    f"provider={result['provider']}",
                    f"mode={result['mode']}",
                )
            )
        )
    print(
        json.dumps(
            {
                "benchmark": payload["benchmark"],
                "result_count": payload["result_count"],
                "correct_case_count": payload["correct_case_count"],
                "valid_measurement_count": payload[
                    "valid_measurement_count"
                ],
                "all_cases_correct": payload["all_cases_correct"],
                "output": str(args.output) if args.output else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
