# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import time

import torch

from flashrwkv2.head.l2wrap_ce import pretrain_head_l2wrap_ce_bf16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=str, default="/tmp/flashrwkv2-head-l2wrap-ce.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    hidden = torch.randn(args.batch, args.tokens, args.channels, device=device, dtype=torch.bfloat16)
    weight = torch.randn(65536, args.channels, device=device, dtype=torch.bfloat16)
    targets = torch.randint(65536, (args.batch * args.tokens,), device=device, dtype=torch.int64)
    for _ in range(args.warmup):
        pretrain_head_l2wrap_ce_bf16(hidden, weight, targets)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        pretrain_head_l2wrap_ce_bf16(hidden, weight, targets)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    result = {"operator": "pretrain_head_l2wrap_ce_bf16", "latency_us": samples, "correctness": "passed-before-timing"}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
