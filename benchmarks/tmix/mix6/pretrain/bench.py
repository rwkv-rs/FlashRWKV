# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import time

import torch

from flashrwkv2.tmix.mix6 import pretrain_tmix_mix6_bf16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=32)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    device = torch.device("cuda")
    x = torch.zeros(args.batch, args.seqlen, args.channels, device=device, dtype=torch.bfloat16)
    params = [torch.zeros(args.channels, device=device, dtype=torch.bfloat16) for _ in range(6)]
    for _ in range(10):
        pretrain_tmix_mix6_bf16(x, *params)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.iters):
        start = time.perf_counter()
        pretrain_tmix_mix6_bf16(x, *params)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    print(json.dumps({"operator": "pretrain_tmix_mix6_bf16", "raw_latency_us": samples, "p50_us": samples[len(samples) // 2], "gpu": torch.cuda.get_device_name()}))


if __name__ == "__main__":
    main()
