# SPDX-License-Identifier: MIT
"""Correctness-gated A/B benchmark for fused RWKV-7 decay inference.

The timed product comparison is deliberately narrow:

* A performs a separate bias add, raw-logit to canonical-log-decay transform,
  and then the canonical recurrent kernel.
* B passes the same raw logits and separate bias to the fused recurrent kernel.

Both paths reuse the same prepared metadata ticket, state buffer, CUDA stream,
warmup policy, CUDA events, and synchronization boundary.  A precomputed
canonical WKV-only path is recorded only as a diagnostic and is never reported
as the end-to-end baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from flash_rwkv import _extension

HEAD_SIZE = 64
DECAY_RATE = 0.6065306597126334
SOURCE_ROOT = Path(__file__).resolve().parents[3]
CASES: dict[str, tuple[int, ...]] = {
    "b1_t1": (1,),
    "b320_t1": (1,) * 320,
    "b1_t128": (128,),
    "packed_b320_t16": (16,) * 320,
    "ragged_b320_t1_to_t16": tuple(1 + index % 16 for index in range(320)),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    num_heads: int
    modes: tuple[str, ...]
    warmup_iters: int
    samples: int
    seed: int
    cases: tuple[str, ...]
    output: Path | None
    trace_dir: Path | None


@dataclass(frozen=True)
class Inputs:
    mode: str
    seq_lens: tuple[int, ...]
    cu_seqlens: torch.Tensor
    state_indices: torch.Tensor
    validated_metadata: object
    initial_state: torch.Tensor
    state: torch.Tensor
    r: torch.Tensor
    decay_logits: torch.Tensor
    decay_bias: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    output: torch.Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SOURCE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
    }


def _make_inputs(
    seq_lens: tuple[int, ...],
    *,
    num_heads: int,
    mode: str,
    seed: int,
) -> Inputs:
    total_tokens = sum(seq_lens)
    batch_size = len(seq_lens)
    shape = (total_tokens, num_heads, HEAD_SIZE)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def normal(scale: float) -> torch.Tensor:
        return (
            scale
            * torch.randn(
                shape,
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            )
        ).to(torch.float16)

    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    state_indices = torch.arange(
        batch_size - 1,
        -1,
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
    initial_state = 0.01 * torch.randn(
        batch_size,
        num_heads,
        HEAD_SIZE,
        HEAD_SIZE,
        generator=generator,
        device="cuda",
        dtype=state_dtype,
    )
    state = initial_state.clone()
    r = normal(0.05)
    decay_logits = normal(1.0)
    k = normal(0.05)
    v = normal(0.05)
    a = normal(0.05)
    b = normal(0.05)
    output = torch.empty_like(v)
    decay_bias = torch.linspace(
        -0.25,
        0.25,
        num_heads * HEAD_SIZE,
        device="cuda",
        dtype=torch.float16,
    ).reshape(num_heads, HEAD_SIZE)
    validated_metadata = _extension.prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=total_tokens,
        state_pool_size=batch_size,
    )
    return Inputs(
        mode=mode,
        seq_lens=seq_lens,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        validated_metadata=validated_metadata,
        initial_state=initial_state,
        state=state,
        r=r,
        decay_logits=decay_logits,
        decay_bias=decay_bias,
        k=k,
        v=v,
        a=a,
        b=b,
        output=output,
    )


def _unfused_correct_product_call(inputs: Inputs) -> None:
    log_decay = inputs.decay_logits + inputs.decay_bias
    log_decay.sigmoid_()
    log_decay.mul_(-DECAY_RATE)
    recurrent = (
        _extension.recurrent_fp32
        if inputs.mode == "fp32io16"
        else _extension.recurrent_fp16
    )
    recurrent(
        inputs.cu_seqlens,
        inputs.state_indices,
        inputs.state,
        inputs.r,
        log_decay,
        inputs.k,
        inputs.v,
        inputs.a,
        inputs.b,
        inputs.output,
        1.0,
        inputs.validated_metadata,
    )


def _fused_product_call(inputs: Inputs) -> None:
    recurrent = (
        _extension.recurrent_fp32_from_decay_logits
        if inputs.mode == "fp32io16"
        else _extension.recurrent_fp16_from_decay_logits
    )
    recurrent(
        inputs.cu_seqlens,
        inputs.state_indices,
        inputs.state,
        inputs.r,
        inputs.decay_logits,
        inputs.k,
        inputs.v,
        inputs.a,
        inputs.b,
        inputs.output,
        1.0,
        decay_bias=inputs.decay_bias,
        validated_metadata=inputs.validated_metadata,
    )


def _canonical_diagnostic_call(
    inputs: Inputs,
    precomputed_log_decay: torch.Tensor,
) -> None:
    recurrent = (
        _extension.recurrent_fp32
        if inputs.mode == "fp32io16"
        else _extension.recurrent_fp16
    )
    recurrent(
        inputs.cu_seqlens,
        inputs.state_indices,
        inputs.state,
        inputs.r,
        precomputed_log_decay,
        inputs.k,
        inputs.v,
        inputs.a,
        inputs.b,
        inputs.output,
        1.0,
        inputs.validated_metadata,
    )


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    rmse = difference.square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return float((rmse / baseline).item())


def _measure(
    call: Callable[[], None],
    inputs: Inputs,
    *,
    warmup_iters: int,
    samples: int,
) -> tuple[list[float], float]:
    for _ in range(warmup_iters):
        inputs.state.copy_(inputs.initial_state)
        call()
    torch.cuda.synchronize()

    measurements: list[float] = []
    for _ in range(samples):
        inputs.state.copy_(inputs.initial_state)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        measurements.append(float(start.elapsed_time(end)))
    return measurements, statistics.median(measurements)


def _launch_trace(
    name: str,
    call: Callable[[], None],
    inputs: Inputs,
    *,
    trace_path: Path | None,
) -> dict[str, object]:
    inputs.state.copy_(inputs.initial_state)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        call()
        torch.cuda.synchronize()
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace.export_chrome_trace(str(trace_path))
    cuda_kernel_names = [
        event.name
        for event in trace.events()
        if "cuda" in str(event.device_type).lower()
    ]
    recurrent_count = sum(
        any(
            marker in kernel_name
            for marker in (
                "recurrent_fp32_kernel",
                "recurrent_fp16_kernel",
                "recurrent_fp16_generic_kernel",
            )
        )
        for kernel_name in cuda_kernel_names
    )
    validator_count = sum(
        "validate_recurrent_metadata_kernel" in kernel_name
        for kernel_name in cuda_kernel_names
    )
    return {
        "range": name,
        "cuda_kernel_count": len(cuda_kernel_names),
        "cuda_kernel_names": cuda_kernel_names,
        "recurrent_kernel_count": recurrent_count,
        "metadata_validator_kernel_count": validator_count,
        "non_recurrent_kernel_count": (
            len(cuda_kernel_names) - recurrent_count - validator_count
        ),
        "chrome_trace": str(trace_path) if trace_path is not None else None,
    }


def _trace_path(
    trace_dir: Path | None,
    case_name: str,
    variant_name: str,
) -> Path | None:
    if trace_dir is None:
        return None
    return trace_dir / f"{case_name}-{variant_name}.json"


def _check_launch_contract(
    unfused: dict[str, object],
    fused: dict[str, object],
    diagnostic: dict[str, object],
) -> None:
    if unfused["recurrent_kernel_count"] != 1:
        raise RuntimeError(f"unfused trace must contain one WKV kernel: {unfused}")
    if int(unfused["non_recurrent_kernel_count"]) < 3:
        raise RuntimeError(
            "unfused trace must include bias-add, sigmoid, and multiply "
            f"launches: {unfused}"
        )
    for label, trace in (("fused", fused), ("diagnostic", diagnostic)):
        if (
            trace["cuda_kernel_count"] != 1
            or trace["recurrent_kernel_count"] != 1
            or trace["metadata_validator_kernel_count"] != 0
            or trace["non_recurrent_kernel_count"] != 0
        ):
            raise RuntimeError(
                f"{label} ticket path must contain exactly one WKV kernel: {trace}"
            )
    if unfused["metadata_validator_kernel_count"] != 0:
        raise RuntimeError(f"unfused ticket path relaunched validation: {unfused}")


def _run_case(
    case_name: str,
    seq_lens: tuple[int, ...],
    mode: str,
    config: BenchmarkConfig,
    seed: int,
) -> dict[str, object]:
    inputs = _make_inputs(
        seq_lens,
        num_heads=config.num_heads,
        mode=mode,
        seed=seed,
    )
    precomputed_log_decay = inputs.decay_logits + inputs.decay_bias
    precomputed_log_decay.sigmoid_()
    precomputed_log_decay.mul_(-DECAY_RATE)

    inputs.state.copy_(inputs.initial_state)
    _unfused_correct_product_call(inputs)
    torch.cuda.synchronize()
    expected_output = inputs.output.clone()
    expected_state = inputs.state.clone()

    inputs.state.copy_(inputs.initial_state)
    _fused_product_call(inputs)
    torch.cuda.synchronize()
    fused_output_error = _relative_rmse(inputs.output, expected_output)
    fused_state_error = _relative_rmse(inputs.state, expected_state)
    relative_rmse_limit = 0.002 if mode == "fp32io16" else 0.003
    if (
        fused_output_error > relative_rmse_limit
        or fused_state_error > relative_rmse_limit
    ):
        raise RuntimeError(
            f"fused correctness failed for {case_name}: "
            f"output RRMSE={fused_output_error}, state RRMSE={fused_state_error}"
        )

    inputs.state.copy_(inputs.initial_state)
    _canonical_diagnostic_call(inputs, precomputed_log_decay)
    torch.cuda.synchronize()
    diagnostic_output_error = _relative_rmse(inputs.output, expected_output)
    diagnostic_state_error = _relative_rmse(inputs.state, expected_state)

    variants = {
        "unfused_correct_product": (
            lambda: _unfused_correct_product_call(inputs),
            "bias add + raw-to-log-decay + canonical recurrent",
            inputs.decay_logits.numel() * inputs.decay_logits.element_size(),
        ),
        "fused_raw_product": (
            lambda: _fused_product_call(inputs),
            "raw decay logits + separate bias fused into recurrent",
            0,
        ),
        "precomputed_log_decay_wkv_only_diagnostic": (
            lambda: _canonical_diagnostic_call(inputs, precomputed_log_decay),
            "diagnostic only; producer work is outside the timed region",
            0,
        ),
    }
    results: dict[str, object] = {}
    traces: dict[str, dict[str, object]] = {}
    for variant_name, (call, boundary, materialized_bytes) in variants.items():
        raw_samples, p50_ms = _measure(
            call,
            inputs,
            warmup_iters=config.warmup_iters,
            samples=config.samples,
        )
        launch_trace = _launch_trace(
            variant_name,
            call,
            inputs,
            trace_path=_trace_path(
                config.trace_dir,
                f"{case_name}-{mode}",
                variant_name,
            ),
        )
        traces[variant_name] = launch_trace
        results[variant_name] = {
            "measurement_boundary": boundary,
            "raw_samples_ms": raw_samples,
            "p50_ms": p50_ms,
            "timed_transform_materialization_bytes": materialized_bytes,
            "launch_trace": launch_trace,
        }
    _check_launch_contract(
        traces["unfused_correct_product"],
        traces["fused_raw_product"],
        traces["precomputed_log_decay_wkv_only_diagnostic"],
    )
    unfused_p50 = float(results["unfused_correct_product"]["p50_ms"])
    fused_p50 = float(results["fused_raw_product"]["p50_ms"])
    return {
        "case": case_name,
        "seq_lens": list(seq_lens),
        "batch_size": len(seq_lens),
        "total_tokens": sum(seq_lens),
        "num_heads": config.num_heads,
        "head_size": HEAD_SIZE,
        "mode": mode,
        "token_dtype": "float16",
        "state_dtype": "float32" if mode == "fp32io16" else "float16",
        "correctness": {
            "baseline": "unfused_correct_product",
            "fused_output_relative_rmse": fused_output_error,
            "fused_state_relative_rmse": fused_state_error,
            "diagnostic_output_relative_rmse": diagnostic_output_error,
            "diagnostic_state_relative_rmse": diagnostic_state_error,
            "relative_rmse_limit": relative_rmse_limit,
        },
        "fused_speedup_over_unfused": unfused_p50 / fused_p50,
        "variants": results,
    }


def run(config: BenchmarkConfig) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if config.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if not config.modes or not set(config.modes) <= {"fp32io16", "fp16"}:
        raise ValueError("modes must contain fp32io16 and/or fp16")
    if config.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if config.samples <= 0:
        raise ValueError("samples must be positive")

    results = []
    case_index = 0
    for case_name in config.cases:
        for mode in config.modes:
            results.append(
                _run_case(
                    case_name,
                    CASES[case_name],
                    mode,
                    config,
                    config.seed + 1009 * case_index,
                )
            )
            case_index += 1
    extension_path = Path(_extension._load_extension().__file__).resolve()
    payload: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "flash_rwkv_fused_decay_recurrent",
        "git_revision": _git_revision(),
        "source_sha256": _sha256(Path(__file__).resolve()),
        "extension_path": str(extension_path),
        "extension_sha256": _sha256(extension_path),
        "hardware": _hardware_metadata(),
        "warmup_iters": config.warmup_iters,
        "samples": config.samples,
        "seed": config.seed,
        "main_comparison": {
            "A": "unfused_correct_product",
            "B": "fused_raw_product",
            "same_inputs_state_mode_ticket_stream_events_and_sync": True,
            "precomputed_log_decay_is_diagnostic_only": True,
        },
        "operator_telemetry": {
            "A_recurrent": ("flash_rwkv._extension.recurrent_fp32|recurrent_fp16"),
            "B_recurrent": (
                "flash_rwkv._extension.recurrent_fp32_from_decay_logits|"
                "recurrent_fp16_from_decay_logits"
            ),
            "metadata": "flash_rwkv.prepare_recurrent_metadata",
        },
        "fp16_elapsed_t": (
            "None for both A and B; actual-dither FP16 performance remains "
            "covered by packed-recurrent-benchmark.json"
        ),
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
    return Path(run_log_dir) / "flash-rwkv-fused-decay-recurrent.json"


def _default_trace_dir() -> Path | None:
    run_log_dir = os.environ.get("REMOTE_RUN_LOG_DIR")
    if not run_log_dir:
        return None
    return Path(run_log_dir) / "flash-rwkv-fused-decay-recurrent-traces"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("fp32io16", "fp16"),
        default=["fp32io16", "fp16"],
    )
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASES),
        default=list(CASES),
    )
    parser.add_argument("--output", type=Path, default=_default_output())
    parser.add_argument("--trace-dir", type=Path, default=_default_trace_dir())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(
        BenchmarkConfig(
            num_heads=args.num_heads,
            modes=tuple(args.modes),
            warmup_iters=args.warmup_iters,
            samples=args.samples,
            seed=args.seed,
            cases=tuple(args.cases),
            output=args.output,
            trace_dir=args.trace_dir,
        )
    )
    print(
        json.dumps(
            {
                "benchmark": payload["benchmark"],
                "git_revision": payload["git_revision"],
                "cases": len(payload["results"]),
                "output": str(args.output) if args.output is not None else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
