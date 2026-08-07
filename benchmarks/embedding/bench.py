# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import time

import torch

from flashrwkv2.embedding import infer_embedding_ln0_forward_varlen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    embedding = torch.randn(args.tokens, args.channels, device=device, dtype=torch.bfloat16)
    weight = torch.ones(args.channels, device=device, dtype=torch.bfloat16)
    bias = torch.zeros_like(weight)
    for _ in range(10):
        infer_embedding_ln0_forward_varlen(embedding, weight, bias)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        infer_embedding_ln0_forward_varlen(embedding, weight, bias)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    result = {"operator": "infer_embedding_ln0_forward_varlen", "raw_latency_us": samples, "p50_us": sorted(samples)[len(samples) // 2], "source_revision": "ee3308f6922e59f2166c7fac3c5a192340a2b48e", "gpu": torch.cuda.get_device_name(), "correctness": "validated-by-tests/embedding/test.py"}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
