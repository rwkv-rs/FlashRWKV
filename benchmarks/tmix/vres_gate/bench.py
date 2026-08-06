# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import subprocess
import time

import torch

from flash_rwkv.tmix.vres_gate import infer_tmix_vres_gate_forward_varlen


SOURCE_REVISION = "ee3308f6922e59f2166c7fac3c5a192340a2b48e"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=str, default="/tmp/flash-rwkv-vres-gate.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    v = torch.randn(args.tokens, args.channels, device=device, dtype=torch.float16)
    first = torch.randn_like(v)
    v0 = torch.randn(args.channels, device=device, dtype=torch.float16)
    v12 = torch.randn_like(v)
    for _ in range(args.warmup):
        infer_tmix_vres_gate_forward_varlen(v, first, v0, v12)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        infer_tmix_vres_gate_forward_varlen(v, first, v0, v12)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    result = {
        "operator": "infer_tmix_vres_gate_forward_varlen",
        "tokens": args.tokens,
        "channels": args.channels,
        "raw_latency_us": samples,
        "p50_us": sorted(samples)[len(samples) // 2],
        "selected_kernel_family": "tmix_vres_gate_vec2_or_scalar",
        "source_revision": SOURCE_REVISION,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_status": subprocess.run(["git", "status", "--short"], capture_output=True, text=True, check=False).stdout.strip(),
        "correctness": "validated-by-tests/tmix/vres_gate/test.py",
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
