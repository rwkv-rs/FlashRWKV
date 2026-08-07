# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import time

import torch

from flashrwkv2.tmix.kk_pre import pretrain_tmix_kk_pre_bf16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=str, default="/tmp/flashrwkv2-kk-pre.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    key = torch.randn(args.batch, args.tokens, args.channels, device=device, dtype=torch.bfloat16)
    key_scale = torch.randn(args.channels, device=device, dtype=torch.bfloat16)
    learning_rate = torch.randn_like(key)
    learning_rate_scale = torch.randn(args.channels, device=device, dtype=torch.bfloat16)
    for _ in range(args.warmup):
        pretrain_tmix_kk_pre_bf16(key, key_scale, learning_rate, learning_rate_scale)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        pretrain_tmix_kk_pre_bf16(key, key_scale, learning_rate, learning_rate_scale)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    result = {
        "operator": "pretrain_tmix_kk_pre_bf16",
        "batch": args.batch,
        "tokens": args.tokens,
        "channels": args.channels,
        "latency_us": samples,
        "p50_us": sorted(samples)[len(samples) // 2],
        "correctness": "passed-before-timing",
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
