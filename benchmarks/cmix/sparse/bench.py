# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import subprocess
import time

import torch

from flash_rwkv.cmix.sparse import infer_cmix_sparse_forward_varlen


SOURCE_REVISION = "ee3308f6922e59f2166c7fac3c5a192340a2b48e"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--features", type=int, default=16384)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=str, default="/tmp/flash-rwkv-cmix-sparse.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.tokens % args.batch:
        raise ValueError("tokens must be divisible by batch")
    device = torch.device("cuda")
    length = args.tokens // args.batch
    x = torch.randn(args.tokens, args.channels, device=device, dtype=torch.float16)
    x_k = torch.randn(args.channels, device=device, dtype=torch.float16)
    key_fc = torch.randn(args.features, args.channels, device=device, dtype=torch.float16)
    value_fc = torch.randn(args.features, args.channels, device=device, dtype=torch.float16)
    shift = torch.zeros(args.batch, args.channels, device=device, dtype=torch.float16)
    cu = torch.arange(0, args.tokens + 1, length, device=device, dtype=torch.int32)
    slots = torch.arange(args.batch, device=device, dtype=torch.int32)
    for _ in range(args.warmup):
        infer_cmix_sparse_forward_varlen(
            x, x_k, key_fc, value_fc, shift_state_pool=shift, cu_seqlens=cu, state_indices=slots
        )
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        infer_cmix_sparse_forward_varlen(
            x, x_k, key_fc, value_fc, shift_state_pool=shift, cu_seqlens=cu, state_indices=slots
        )
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    result = {
        "operator": "infer_cmix_sparse_forward_varlen",
        "tokens": args.tokens,
        "channels": args.channels,
        "features": args.features,
        "batch": args.batch,
        "raw_latency_us": samples,
        "p50_us": sorted(samples)[len(samples) // 2],
        "selected_kernel_family": "cmix_sparse_up_projection_plus_relu_square_down",
        "source_revision": SOURCE_REVISION,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_status": subprocess.run(["git", "status", "--short"], capture_output=True, text=True, check=False).stdout.strip(),
        "correctness": "validated-by-tests/cmix/sparse/test.py",
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
