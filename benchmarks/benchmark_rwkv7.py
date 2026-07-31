# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-gated unified RWKV-7 forward benchmark.

The serving profiles are adapted from the canonical vllm-rwkv benchmark at
commit 6d683f9e49a2997e405c47edc147872c8609513b. Functional measurements call
the public provider boundary with identical inputs. The separate stateful
measurement follows the Albatross/vLLM in-place state-pool contract and is
never compared as identical-input latency.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from flash_rwkv import _C, rwkv7, rwkv7_recurrent_stateful
from flash_rwkv.config import (
    chunk_tuning_key,
    select_chunk_config,
)


HEAD_SIZE = 64
SOURCE_ROOT = Path(__file__).resolve().parents[1]
TOLERANCE_PATH = SOURCE_ROOT / "tests/fixtures/tolerances-v1.json"
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class Profile:
    layout: str
    seq_lens: tuple[int, ...]


PROFILES: dict[str, Profile] = {
    "decode_b1": Profile("fixed", (1,)),
    "decode_b16": Profile("fixed", (1,) * 16),
    "decode_b32": Profile("fixed", (1,) * 32),
    "decode_b64": Profile("fixed", (1,) * 64),
    "decode_b128": Profile("fixed", (1,) * 128),
    "decode_b320": Profile("fixed", (1,) * 320),
    "equal_chunk16_b320": Profile("fixed", (16,) * 320),
    "ragged_1_16_b320": Profile("packed", tuple(range(1, 17)) * 20),
    "ragged_long_b32": Profile(
        "packed",
        (1, 4, 8, 16, 32, 64, 96, 128) * 4,
    ),
    "ragged_skewed_b32": Profile("packed", (128,) + (1,) * 31),
}
PROVIDER_NAMES = (
    "flash_recurrent",
    "flash_chunk",
    "fla_chunk",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    hidden_size: int
    profiles: tuple[str, ...]
    dtypes: tuple[str, ...]
    providers: tuple[str, ...]
    warmup_iters: int
    samples: int
    stateful_sample_iters: int
    seed: int
    output: Path | None
    measure: bool
    measure_stateful: bool
    profile_provider: str | None
    profile_case: str | None
    profile_dtype: str
    profile_iterations: int


@dataclass(frozen=True)
class Inputs:
    profile: Profile
    initial_state: torch.Tensor
    tensors: tuple[torch.Tensor, ...]
    flat_tensors: tuple[torch.Tensor, ...]
    cu_seqlens_cpu: torch.Tensor | None
    cu_seqlens_cuda: torch.Tensor | None

    @property
    def num_sequences(self) -> int:
        return len(self.profile.seq_lens)

    @property
    def total_tokens(self) -> int:
        return sum(self.profile.seq_lens)

    @property
    def num_heads(self) -> int:
        return self.flat_tensors[0].shape[1]


@dataclass(frozen=True)
class Provider:
    name: str
    configuration: dict[str, object]
    call: Callable[[], tuple[torch.Tensor, torch.Tensor | None]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _source_paths() -> tuple[Path, ...]:
    native = (
        path.relative_to(SOURCE_ROOT)
        for path in (SOURCE_ROOT / "csrc").rglob("*")
        if path.is_file() and path.suffix in {".cpp", ".cu", ".cuh", ".h"}
    )
    python = (
        path.relative_to(SOURCE_ROOT)
        for path in (SOURCE_ROOT / "flash_rwkv").glob("*.py")
    )
    fixed = (
        Path("setup.py"),
        Path("pyproject.toml"),
        Path("flash_rwkv/chunk-tuning-v1.json"),
    )
    return tuple(sorted((*native, *python, *fixed), key=str))


def _fla_metadata() -> dict[str, object]:
    import fla

    package_root = Path(fla.__file__).resolve().parent
    repository = package_root.parent
    try:
        version = importlib.metadata.version("flash-linear-attention")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        **_repository_metadata(repository),
        "package_version": version,
        "module_path": str(package_root),
    }


def _source_metadata() -> dict[str, object]:
    paths = _source_paths()
    hashes = {
        str(relative): _sha256(SOURCE_ROOT / relative)
        for relative in paths
    }
    extension_path = Path(_C.__file__).resolve()
    return {
        "leaf": _repository_metadata(SOURCE_ROOT),
        "parent": _repository_metadata(SOURCE_ROOT.parents[2]),
        "fla": _fla_metadata(),
        "benchmark_script_sha256": _sha256(Path(__file__).resolve()),
        "source_sha256": hashes,
        "source_set_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
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
        "compute_capability": (
            f"{properties.major}.{properties.minor}"
        ),
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _make_inputs(
    profile: Profile,
    *,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
) -> Inputs:
    num_heads = hidden_size // HEAD_SIZE
    total_tokens = sum(profile.seq_lens)
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
        normal(0.05).to(dtype),
        (
            -0.05
            - 0.15
            * torch.rand(
                flat_shape,
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
        ).to(dtype),
        normal(0.05).to(dtype),
        normal(0.05).to(dtype),
        (-direction).to(dtype),
        (direction * strength).to(dtype),
    )
    if profile.layout == "fixed":
        sequence_length = profile.seq_lens[0]
        if any(length != sequence_length for length in profile.seq_lens):
            raise ValueError("fixed profiles must have equal sequence lengths")
        surface_shape = (
            len(profile.seq_lens),
            sequence_length,
            num_heads,
            HEAD_SIZE,
        )
        tensors = tuple(
            tensor.reshape(surface_shape) for tensor in flat_tensors
        )
        cu_seqlens_cpu = None
        cu_seqlens_cuda = None
    elif profile.layout == "packed":
        surface_shape = (1, total_tokens, num_heads, HEAD_SIZE)
        tensors = tuple(
            tensor.reshape(surface_shape) for tensor in flat_tensors
        )
        offsets = [0]
        for length in profile.seq_lens:
            offsets.append(offsets[-1] + length)
        cu_seqlens_cpu = torch.tensor(offsets, dtype=torch.int64)
        cu_seqlens_cuda = cu_seqlens_cpu.to(device="cuda")
    else:
        raise ValueError(f"unknown profile layout: {profile.layout}")

    initial_state = 0.02 * torch.randn(
        (
            len(profile.seq_lens),
            num_heads,
            HEAD_SIZE,
            HEAD_SIZE,
        ),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    return Inputs(
        profile=profile,
        initial_state=initial_state,
        tensors=tensors,
        flat_tensors=flat_tensors,
        cu_seqlens_cpu=cu_seqlens_cpu,
        cu_seqlens_cuda=cu_seqlens_cuda,
    )


def _reference(
    inputs: Inputs,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = inputs.initial_state.clone()
    output = torch.empty_like(inputs.flat_tensors[3], dtype=torch.float32)
    offsets = [0]
    for length in inputs.profile.seq_lens:
        offsets.append(offsets[-1] + length)
    flat = tuple(tensor.float() for tensor in inputs.flat_tensors)

    for token_offset in range(max(inputs.profile.seq_lens)):
        active_sequences = [
            index
            for index, length in enumerate(inputs.profile.seq_lens)
            if token_offset < length
        ]
        sequence_indices = torch.tensor(
            active_sequences,
            dtype=torch.long,
            device="cuda",
        )
        token_indices = torch.tensor(
            [
                offsets[index] + token_offset
                for index in active_sequences
            ],
            dtype=torch.long,
            device="cuda",
        )
        previous_state = state.index_select(0, sequence_indices)
        r, log_decay, k, v, a, b = (
            tensor.index_select(0, token_indices) for tensor in flat
        )
        state_dot_a = torch.einsum("nhk,nhkv->nhv", a, previous_state)
        next_state = (
            torch.exp(log_decay).unsqueeze(-1) * previous_state
            + b.unsqueeze(-1) * state_dot_a.unsqueeze(-2)
            + k.unsqueeze(-1) * v.unsqueeze(-2)
        )
        token_output = float(scale) * torch.einsum(
            "nhk,nhkv->nhv",
            r,
            next_state,
        )
        state.index_copy_(0, sequence_indices, next_state)
        output.index_copy_(0, token_indices, token_output)

    return output.reshape(inputs.tensors[3].shape), state


def _providers(
    inputs: Inputs,
    names: tuple[str, ...],
    *,
    scale: float,
) -> dict[str, Provider]:
    r, log_decay, k, v, a, b = inputs.tensors
    packed = inputs.profile.layout == "packed"
    key = chunk_tuning_key(
        r,
        mode="fp32io16",
        packed=packed,
        max_sequence_length=max(inputs.profile.seq_lens),
    )
    selected = select_chunk_config(key)

    def flash_recurrent() -> tuple[torch.Tensor, torch.Tensor | None]:
        return rwkv7(
            r,
            log_decay,
            k,
            v,
            a,
            b,
            scale=scale,
            initial_state=inputs.initial_state,
            output_final_state=True,
            cu_seqlens=inputs.cu_seqlens_cpu,
            mode="fp32io16",
            algorithm="recurrent",
        )

    def flash_chunk() -> tuple[torch.Tensor, torch.Tensor | None]:
        return rwkv7(
            r,
            log_decay,
            k,
            v,
            a,
            b,
            scale=scale,
            initial_state=inputs.initial_state,
            output_final_state=True,
            cu_seqlens=inputs.cu_seqlens_cpu,
            mode="fp32io16",
            algorithm="chunk",
            chunk_config=selected.config,
        )

    def fla_chunk() -> tuple[torch.Tensor, torch.Tensor | None]:
        from fla.ops.rwkv7 import chunk_rwkv7

        return chunk_rwkv7(
            r,
            log_decay,
            k,
            v,
            a,
            b,
            scale=scale,
            initial_state=inputs.initial_state,
            output_final_state=True,
            cu_seqlens=inputs.cu_seqlens_cuda,
            cu_seqlens_cpu=inputs.cu_seqlens_cpu,
            safe_gate=True,
            chunk_size=16,
        )

    available = {
        "flash_recurrent": Provider(
            name="flash_recurrent",
            configuration={
                "implementation": "flash_rwkv.rwkv7",
                "algorithm": "recurrent",
                "mode": "fp32io16",
                "state_policy": "functional FP32 copy",
            },
            call=flash_recurrent,
        ),
        "flash_chunk": Provider(
            name="flash_chunk",
            configuration={
                "implementation": "flash_rwkv.rwkv7",
                "algorithm": "chunk",
                "mode": "fp32io16",
                "state_policy": "functional FP32 copy",
                "tuning_key": key.identifier,
                "selection_source": selected.source,
                "chunk_config": asdict(selected.config),
            },
            call=flash_chunk,
        ),
        "fla_chunk": Provider(
            name="fla_chunk",
            configuration={
                "implementation": "fla.ops.rwkv7.chunk_rwkv7",
                "mode": "fp32 state",
                "state_policy": "functional",
                "safe_gate": True,
                "chunk_size": 16,
            },
            call=fla_chunk,
        ),
    }
    return {name: available[name] for name in names}


def _relative_rmse(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    difference = actual.float() - expected.float()
    rmse = difference.square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return float((rmse / baseline).item())


def _max_absolute_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _correctness(
    provider: Provider,
    inputs: Inputs,
    expected_output: torch.Tensor,
    expected_state: torch.Tensor,
    limits: dict[str, float],
) -> dict[str, object]:
    initial_snapshot = inputs.initial_state.clone()
    with torch.inference_mode():
        output, final_state = provider.call()
    torch.cuda.synchronize()
    if final_state is None:
        raise RuntimeError(f"{provider.name} did not return final state")
    output_error = _relative_rmse(output, expected_output)
    state_error = _relative_rmse(final_state, expected_state)
    finite = bool(
        torch.isfinite(output).all().item()
        and torch.isfinite(final_state).all().item()
    )
    initial_state_preserved = bool(
        torch.equal(inputs.initial_state, initial_snapshot)
    )
    passed = bool(
        finite
        and initial_state_preserved
        and output_error <= limits["output_relative_rmse"]
        and state_error <= limits["state_relative_rmse"]
    )
    return {
        "passed": passed,
        "output_relative_rmse": output_error,
        "output_max_absolute_error": _max_absolute_error(
            output,
            expected_output,
        ),
        "final_state_relative_rmse": state_error,
        "final_state_max_absolute_error": _max_absolute_error(
            final_state,
            expected_state,
        ),
        "output_relative_rmse_limit": limits[
            "output_relative_rmse"
        ],
        "final_state_relative_rmse_limit": limits[
            "state_relative_rmse"
        ],
        "finite": finite,
        "initial_state_preserved": initial_state_preserved,
    }


def _percentile(samples: Sequence[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def _latency_summary(
    samples: list[float],
    *,
    total_tokens: int,
) -> dict[str, object]:
    p50 = statistics.median(samples)
    return {
        "min_ms": min(samples),
        "mean_ms": statistics.fmean(samples),
        "p10_ms": _percentile(samples, 0.10),
        "p50_ms": p50,
        "p90_ms": _percentile(samples, 0.90),
        "useful_tokens_per_s_p50": total_tokens * 1000.0 / p50,
        "raw_samples_ms": samples,
    }


def _measure_functional(
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
    with torch.inference_mode():
        for provider in correct.values():
            for _ in range(warmup_iters):
                provider.call()
    torch.cuda.synchronize()

    raw_samples = {name: [] for name in correct}
    ordering = list(correct)
    generator = random.Random(seed)
    with torch.inference_mode():
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
            with torch.inference_mode():
                for _ in range(iterations):
                    provider.call()
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
    finally:
        cudart.cudaProfilerStop()
    return {"name": range_name, "iterations": iterations}


def _stateful_payload(
    inputs: Inputs,
    *,
    seed: int,
) -> tuple[
    tuple[torch.Tensor, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    packed_tensors = tuple(
        tensor.reshape(
            1,
            inputs.total_tokens,
            inputs.num_heads,
            HEAD_SIZE,
        )
        for tensor in inputs.flat_tensors
    )
    offsets = [0]
    for length in inputs.profile.seq_lens:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64)
    state_indices = torch.arange(
        inputs.num_sequences - 1,
        -1,
        -1,
        dtype=torch.int64,
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state_pool = 0.02 * torch.randn(
        (
            inputs.num_sequences + 7,
            inputs.num_heads,
            HEAD_SIZE,
            HEAD_SIZE,
        ),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    state_pool.index_copy_(
        0,
        state_indices.to(device="cuda"),
        inputs.initial_state,
    )
    return packed_tensors, cu_seqlens, state_indices, state_pool


def _stateful_measurement(
    inputs: Inputs,
    expected_output: torch.Tensor,
    expected_state: torch.Tensor,
    *,
    scale: float,
    limits: dict[str, float],
    warmup_iters: int,
    samples: int,
    sample_iters: int,
    seed: int,
    measure: bool,
) -> dict[str, object]:
    tensors, cu_seqlens, state_indices, base_pool = _stateful_payload(
        inputs,
        seed=seed,
    )
    active_indices = state_indices.to(device="cuda")
    untouched_mask = torch.ones(
        base_pool.shape[0],
        dtype=torch.bool,
        device="cuda",
    )
    untouched_mask[active_indices] = False
    pool = base_pool.clone()
    with torch.inference_mode():
        output = rwkv7_recurrent_stateful(
            *tensors,
            state_pool=pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            scale=scale,
            mode="fp32io16",
        )
    torch.cuda.synchronize()
    actual_state = pool.index_select(0, active_indices)
    output_error = _relative_rmse(
        output,
        expected_output.reshape(output.shape),
    )
    state_error = _relative_rmse(actual_state, expected_state)
    untouched_preserved = bool(
        torch.equal(pool[untouched_mask], base_pool[untouched_mask])
    )
    finite = bool(
        torch.isfinite(output).all().item()
        and torch.isfinite(actual_state).all().item()
    )
    passed = bool(
        finite
        and untouched_preserved
        and output_error <= limits["output_relative_rmse"]
        and state_error <= limits["state_relative_rmse"]
    )
    correctness = {
        "passed": passed,
        "output_relative_rmse": output_error,
        "final_state_relative_rmse": state_error,
        "output_relative_rmse_limit": limits[
            "output_relative_rmse"
        ],
        "final_state_relative_rmse_limit": limits[
            "state_relative_rmse"
        ],
        "finite": finite,
        "untouched_slots_preserved": untouched_preserved,
    }

    measurements = None
    measurement_finite = None
    if passed and measure:
        warmup_pool = base_pool.clone()
        with torch.inference_mode():
            for _ in range(warmup_iters):
                rwkv7_recurrent_stateful(
                    *tensors,
                    state_pool=warmup_pool,
                    cu_seqlens=cu_seqlens,
                    state_indices=state_indices,
                    scale=scale,
                    mode="fp32io16",
                )
        torch.cuda.synchronize()

        measured_pool = base_pool.clone()
        raw_samples: list[float] = []
        last_output: torch.Tensor | None = None
        with torch.inference_mode():
            for _ in range(samples):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(sample_iters):
                    last_output = rwkv7_recurrent_stateful(
                        *tensors,
                        state_pool=measured_pool,
                        cu_seqlens=cu_seqlens,
                        state_indices=state_indices,
                        scale=scale,
                        mode="fp32io16",
                    )
                end.record()
                end.synchronize()
                raw_samples.append(
                    start.elapsed_time(end) / sample_iters
                )
        if last_output is None:
            raise AssertionError("stateful measurement produced no output")
        measurement_finite = bool(
            torch.isfinite(last_output).all().item()
            and torch.isfinite(
                measured_pool.index_select(0, active_indices)
            )
            .all()
            .item()
        )
        measurements = _latency_summary(
            raw_samples,
            total_tokens=inputs.total_tokens,
        )

    return {
        "provider": "flash_rwkv.rwkv7_recurrent_stateful",
        "boundary": (
            "in-place state-pool evolution; no state reset between timed "
            "iterations; latency is not identical-input latency"
        ),
        "state_slot_mapping": (
            "reverse sequence order with seven untouched rows"
        ),
        "sample_iters": sample_iters,
        "correctness": correctness,
        "measurement_valid": bool(
            measurements is not None and measurement_finite
        ),
        "measurement_finite": measurement_finite,
        "measurements": measurements,
    }


def _run_case(
    name: str,
    dtype_name: str,
    config: BenchmarkConfig,
    tolerances: dict[str, Any],
    *,
    seed: int,
) -> dict[str, object]:
    profile = PROFILES[name]
    inputs = _make_inputs(
        profile,
        hidden_size=config.hidden_size,
        dtype=DTYPES[dtype_name],
        seed=seed,
    )
    expected_output, expected_state = _reference(inputs, scale=1.0)
    providers = _providers(inputs, config.providers, scale=1.0)
    correctness = {
        provider_name: _correctness(
            provider,
            inputs,
            expected_output,
            expected_state,
            (
                tolerances["fp32io16_recurrent"]
                if provider_name == "flash_recurrent"
                else tolerances["fp32io16_chunk"]
            ),
        )
        for provider_name, provider in providers.items()
    }
    functional_measurements = (
        _measure_functional(
            providers,
            correctness,
            total_tokens=inputs.total_tokens,
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
        and config.profile_case == name
        and config.profile_dtype == dtype_name
    ):
        provider = providers[config.profile_provider]
        if not correctness[config.profile_provider]["passed"]:
            raise RuntimeError(
                "profiler provider failed correctness for the exact case"
            )
        profile_range = _profile_provider(
            provider,
            range_name=(
                f"flash_rwkv.unified.{config.profile_provider}."
                f"{name}.{dtype_name}"
            ),
            iterations=config.profile_iterations,
        )

    stateful = (
        _stateful_measurement(
            inputs,
            expected_output,
            expected_state,
            scale=1.0,
            limits=tolerances["fp32io16_recurrent"],
            warmup_iters=config.warmup_iters,
            samples=config.samples,
            sample_iters=config.stateful_sample_iters,
            seed=seed + 1543,
            measure=config.measure,
        )
        if config.measure_stateful
        else None
    )
    result = {
        "profile": name,
        "layout": profile.layout,
        "dtype": dtype_name,
        "mode": "fp32io16",
        "shape": list(inputs.tensors[0].shape),
        "seq_lens": list(profile.seq_lens),
        "sequence_count": inputs.num_sequences,
        "total_tokens": inputs.total_tokens,
        "min_length": min(profile.seq_lens),
        "max_length": max(profile.seq_lens),
        "padding_ratio": (
            0.0
            if profile.layout == "packed"
            else 1.0
            - inputs.total_tokens
            / (
                inputs.num_sequences
                * max(profile.seq_lens)
            )
        ),
        "hidden_size": config.hidden_size,
        "head_size": HEAD_SIZE,
        "head_count": inputs.num_heads,
        "seed": seed,
        "functional": {
            "boundary": (
                "public provider call returning output and functional final "
                "state; identical input tensors and initial state; CUDA "
                "start/end events with end-event synchronization"
            ),
            "providers": {
                provider_name: {
                    "configuration": provider.configuration,
                    "correctness": correctness[provider_name],
                    "measurement_valid": (
                        functional_measurements[provider_name] is not None
                    ),
                    "measurements": functional_measurements[
                        provider_name
                    ],
                }
                for provider_name, provider in providers.items()
            },
        },
        "stateful": stateful,
        "profile_range": profile_range,
    }
    del inputs, expected_output, expected_state, providers
    torch.cuda.empty_cache()
    return result


def run(config: BenchmarkConfig) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if config.hidden_size <= 0 or config.hidden_size % HEAD_SIZE:
        raise ValueError("hidden_size must be a positive multiple of 64")
    if config.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if config.samples <= 0 or config.stateful_sample_iters <= 0:
        raise ValueError("sample counts must be positive")
    if config.profile_iterations <= 0:
        raise ValueError("profile_iterations must be positive")
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

    tolerances = json.loads(TOLERANCE_PATH.read_text(encoding="utf-8"))
    results: list[dict[str, object]] = []
    counter = 0
    for profile_name in config.profiles:
        for dtype_name in config.dtypes:
            results.append(
                _run_case(
                    profile_name,
                    dtype_name,
                    config,
                    tolerances,
                    seed=config.seed + 1009 * counter,
                )
            )
            counter += 1

    functional_cases = [
        provider
        for result in results
        for provider in result["functional"]["providers"].values()
    ]
    stateful_cases = [
        result["stateful"]
        for result in results
        if result["stateful"] is not None
    ]
    all_functional_correct = all(
        provider["correctness"]["passed"]
        for provider in functional_cases
    )
    all_stateful_correct = all(
        stateful["correctness"]["passed"]
        for stateful in stateful_cases
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "flash_rwkv_unified_forward",
        "profile_source": {
            "repository": "https://github.com/BlinkDL/Albatross",
            "implementation_repository": (
                "https://github.com/rwkv-rs/vllm-rwkv"
            ),
            "implementation_revision": (
                "6d683f9e49a2997e405c47edc147872c8609513b"
            ),
            "profile_file": "benchmarks/kernels/benchmark_rwkv7_wkv.py",
        },
        "functional_boundary": (
            "public provider call; input generation, extension/JIT build, "
            "correctness, autotune selection, and warmup precede timing"
        ),
        "stateful_boundary": (
            "FlashRWKV recurrent state pool evolves in place without reset; "
            "reported separately from identical-input functional latency"
        ),
        "synchronization_policy": (
            "CUDA start/end events around each provider call and explicit "
            "end-event synchronization for every retained raw sample"
        ),
        "warmup_iters": config.warmup_iters,
        "samples": config.samples,
        "stateful_sample_iters": config.stateful_sample_iters,
        "measurement_enabled": config.measure,
        "stateful_enabled": config.measure_stateful,
        "seed": config.seed,
        "hardware": _hardware_metadata(),
        "source": _source_metadata(),
        "result_count": len(results),
        "functional_case_count": len(functional_cases),
        "stateful_case_count": len(stateful_cases),
        "all_functional_cases_correct": all_functional_correct,
        "all_stateful_cases_correct": all_stateful_correct,
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
        Path(run_log_dir) / "flash-rwkv-unified-forward.json"
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
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--stateful-sample-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, default=_default_output())
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--no-stateful", action="store_true")
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
        stateful_sample_iters=arguments.stateful_sample_iters,
        seed=arguments.seed,
        output=arguments.output,
        measure=not arguments.correctness_only,
        measure_stateful=not arguments.no_stateful,
        profile_provider=arguments.profile_provider,
        profile_case=arguments.profile_case,
        profile_dtype=arguments.profile_dtype,
        profile_iterations=arguments.profile_iterations,
    )
    payload = run(config)
    summary = {
        "output": None if config.output is None else str(config.output),
        "result_count": payload["result_count"],
        "functional_case_count": payload["functional_case_count"],
        "stateful_case_count": payload["stateful_case_count"],
        "all_functional_cases_correct": payload[
            "all_functional_cases_correct"
        ],
        "all_stateful_cases_correct": payload[
            "all_stateful_cases_correct"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not (
        payload["all_functional_cases_correct"]
        and payload["all_stateful_cases_correct"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
