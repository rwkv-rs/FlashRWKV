# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import subprocess
import time

import torch

from flashrwkv2.cmix.sparse import infer_cmix_sparse_forward_varlen

SOURCE_REVISION = "ee3308f6922e59f2166c7fac3c5a192340a2b48e"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--features", type=int, default=16384)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=str, default="/tmp/flashrwkv2-cmix-sparse.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch <= 0 or any(tokens <= 0 or tokens % args.batch for tokens in args.tokens):
        raise ValueError("tokens must be positive and divisible by batch")
    device = torch.device("cuda")
    torch.manual_seed(97)
    workloads = []
    for tokens in args.tokens:
        length = tokens // args.batch
        x = torch.randn(tokens, args.channels, device=device, dtype=torch.float16) * 0.1
        x_k = torch.randn(args.channels, device=device, dtype=torch.float16) * 0.1
        key_fc = (
            torch.randn(args.features, args.channels, device=device, dtype=torch.float16) * 0.1
        )
        value_fc = (
            torch.randn(args.features, args.channels, device=device, dtype=torch.float16) * 0.1
        )
        initial_shift = torch.zeros(
            args.batch, args.channels, device=device, dtype=torch.float16
        )
        cu = torch.arange(0, tokens + 1, length, device=device, dtype=torch.int32)
        slots = torch.arange(args.batch, device=device, dtype=torch.int32)

        mixed = x.float().clone()
        for sequence in range(args.batch):
            previous = initial_shift[sequence].float()
            start_row = sequence * length
            end_row = start_row + length
            for row in range(start_row, end_row):
                current = x[row].float()
                mixed[row] = current + (previous - current) * x_k.float()
                previous = current
        expected = torch.relu(mixed @ key_fc.float().t()).square() @ value_fc.float()

        modes = {}
        for mode, deterministic in (("atomic", False), ("deterministic", True)):
            correctness_outputs = []
            correctness_states = []
            for _ in range(2):
                shift = initial_shift.clone()
                correctness_outputs.append(
                    infer_cmix_sparse_forward_varlen(
                        x,
                        x_k,
                        key_fc,
                        value_fc,
                        shift_state_pool=shift,
                        cu_seqlens=cu,
                        state_indices=slots,
                        max_seqlen=length,
                        deterministic=deterministic,
                    )
                )
                correctness_states.append(shift)
            observed = correctness_outputs[0]
            finite = bool(torch.isfinite(observed).all().item())
            repeatable = torch.equal(observed, correctness_outputs[1]) and torch.equal(
                correctness_states[0], correctness_states[1]
            )
            error = (observed.float() - expected).abs()
            max_abs_error = float(error.max().item())
            rmse = float(error.square().mean().sqrt().item())
            correctness = finite and torch.allclose(
                observed.float(), expected, atol=128.0, rtol=0.12
            )
            if not correctness:
                raise RuntimeError(
                    f"{mode} correctness failed for B={args.batch},T={length}: "
                    f"max_abs_error={max_abs_error}, rmse={rmse}"
                )
            if deterministic and not repeatable:
                raise RuntimeError(
                    f"deterministic mode was not bitwise repeatable for "
                    f"B={args.batch},T={length}"
                )

            for _ in range(args.warmup):
                shift = initial_shift.clone()
                infer_cmix_sparse_forward_varlen(
                    x,
                    x_k,
                    key_fc,
                    value_fc,
                    shift_state_pool=shift,
                    cu_seqlens=cu,
                    state_indices=slots,
                    max_seqlen=length,
                    deterministic=deterministic,
                )
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.samples):
                shift = initial_shift.clone()
                start = time.perf_counter()
                infer_cmix_sparse_forward_varlen(
                    x,
                    x_k,
                    key_fc,
                    value_fc,
                    shift_state_pool=shift,
                    cu_seqlens=cu,
                    state_indices=slots,
                    max_seqlen=length,
                    deterministic=deterministic,
                )
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1e6)
            modes[mode] = {
                "deterministic_requested": deterministic,
                "bitwise_repeatable": repeatable,
                "finite": finite,
                "correctness": correctness,
                "max_abs_error": max_abs_error,
                "rmse": rmse,
                "raw_latency_us": samples,
                "p50_us": sorted(samples)[len(samples) // 2],
            }

        modes["deterministic"]["latency_ratio_vs_atomic"] = (
            modes["deterministic"]["p50_us"] / modes["atomic"]["p50_us"]
        )
        workloads.append(
            {
                "batch": args.batch,
                "tokens": tokens,
                "max_seqlen": length,
                "modes": modes,
            }
        )

    result = {
        "operator": "infer_cmix_sparse_forward_varlen",
        "channels": args.channels,
        "features": args.features,
        "workloads": workloads,
        "kernel_families": {
            "atomic": "canonical_albatross_sparse_atomic_down",
            "deterministic": "flashrwkv2_fixed_order_sparse_down",
        },
        "source_revision": SOURCE_REVISION,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_status": subprocess.run(["git", "status", "--short"], capture_output=True, text=True, check=False).stdout.strip(),
        "correctness": "validated-before-timing-and-by-tests/cmix/sparse/test.py",
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
