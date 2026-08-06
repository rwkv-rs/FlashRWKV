# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from flash_rwkv.tmix.wkv7 import pretrain_recurrent_fp32io16


ROOT = Path(__file__).resolve().parents[4]
SOURCE_REVISION = "952102498e9ed367ea0a59ee64106916d474d30f"


def run(*, iterations: int = 50) -> dict[str, object]:
    device = torch.device("cuda")
    B, H, D, total = 2, 2, 64, 32
    state = torch.zeros((B, H, D, D), device=device, dtype=torch.float32)
    values = [torch.randn((total, H, D), device=device, dtype=torch.float16).mul_(0.01) for _ in range(6)]
    sequence = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    starts = torch.tensor([0, 16], device=device, dtype=torch.int32)
    ends = torch.tensor([16, total], device=device, dtype=torch.int32)
    for _ in range(5):
        pretrain_recurrent_fp32io16(state, sequence, starts, ends, *values)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        pretrain_recurrent_fp32io16(state, sequence, starts, ends, *values)
    torch.cuda.synchronize()
    latency_us = (time.perf_counter() - start) * 1.0e6 / iterations
    return {
        "source_revision": SOURCE_REVISION,
        "operator": "tmix/wkv7/pretrain_recurrent_fp32io16",
        "gpu": torch.cuda.get_device_name(),
        "shape": {"B": B, "H": H, "D": D, "total_tokens": total},
        "latency_us": latency_us,
        "iterations": iterations,
        "git_status": "dirty-worktree-preserved",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
