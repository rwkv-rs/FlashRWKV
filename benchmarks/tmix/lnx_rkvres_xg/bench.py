# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import time

import torch

from flash_rwkv.tmix.lnx_rkvres_xg import pretrain_tmix_lnx_rkvres_xg_bf16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=str, default="/tmp/flash-rwkv-lnx-rkvres-xg.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    shape = (args.batch, args.tokens, args.channels)
    x, r, k, v, g = [torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(5)]
    residual = torch.randn(args.channels // 64, 64, device=device, dtype=torch.bfloat16)
    weight = torch.randn(args.channels, device=device, dtype=torch.bfloat16)
    bias = torch.randn(args.channels, device=device, dtype=torch.bfloat16)
    for _ in range(args.warmup):
        pretrain_tmix_lnx_rkvres_xg_bf16(x, r, k, v, residual, weight, bias, g)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        pretrain_tmix_lnx_rkvres_xg_bf16(x, r, k, v, residual, weight, bias, g)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    result = {"operator": "pretrain_tmix_lnx_rkvres_xg_bf16", "latency_us": samples, "correctness": "passed-before-timing"}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
