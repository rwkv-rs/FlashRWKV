# SPDX-License-Identifier: MIT

"""Benchmark the train_temp L2Wrap CE forward/backward operator family."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch

from flashrwkv2.loss.l2wrap_ce import pretrain_l2wrap_ce_bf16
from flashrwkv2.tmix.wkv7 import _extension


SOURCE_REVISION = "952102498e9ed367ea0a59ee64106916d474d30f"
SOURCE_PATHS = (
    "csrc/sm90/loss/l2wrap_ce/pretrain_bf16_forward.cpp",
    "csrc/sm90/loss/l2wrap_ce/pretrain_bf16_forward.cu",
    "csrc/sm90/loss/l2wrap_ce/pretrain_bf16_backward.cpp",
    "csrc/sm90/loss/l2wrap_ce/pretrain_bf16_backward.cu",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_metadata(root: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            args, cwd=root, check=False, capture_output=True, text=True
        )
        return result.stdout.strip()

    return {
        "revision": run("git", "rev-parse", "HEAD"),
        "status": run("git", "status", "--short"),
    }


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def _metadata() -> dict[str, object]:
    capability = torch.cuda.get_device_capability()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_memory_bytes": properties.total_memory,
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "python": platform.python_version(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--output", type=str, default="/tmp/flashrwkv2-l2wrap-ce.json"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rows <= 0 or args.vocab <= 0:
        raise ValueError("rows and vocab must be positive")

    device = torch.device("cuda")
    torch.manual_seed(17)
    logits = torch.randn(
        args.rows, args.vocab, device=device, dtype=torch.bfloat16
    ).mul_(0.2)
    targets = torch.randint(
        args.vocab, (args.rows,), device=device, dtype=torch.int64
    )

    # Public correctness gate, including the autograd boundary.
    public_logits = logits.detach().clone().requires_grad_(True)
    public_loss = pretrain_l2wrap_ce_bf16(public_logits, targets)
    reference = torch.nn.functional.cross_entropy(
        public_logits.float(), targets
    )
    public_loss.backward()
    max_error = float((public_loss - reference).abs().item())
    correctness = "passed" if max_error <= 2.0e-4 else "failed"
    if correctness != "passed":
        raise RuntimeError(f"correctness gate failed: max_error={max_error}")

    extension = _extension()
    grad_loss = torch.ones((), device=device, dtype=torch.float32)

    def launch() -> None:
        loss, lse, max_vals, argmax = extension.pretrain_l2wrap_ce_forward(
            logits, targets
        )
        extension.pretrain_l2wrap_ce_backward(
            grad_loss, logits, targets, lse, max_vals, argmax
        )
        del loss

    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end) * 1000.0))

    root = _repo_root()
    result = {
        "operator": "pretrain_l2wrap_ce_bf16",
        "rows": args.rows,
        "vocab": args.vocab,
        "timing_boundary": "native_forward_backward",
        "latency_us": samples,
        "p10_us": _percentile(samples, 0.10),
        "p50_us": _percentile(samples, 0.50),
        "p90_us": _percentile(samples, 0.90),
        "throughput_rows_per_second": args.rows / (_percentile(samples, 0.50) * 1.0e-6),
        "correctness": correctness,
        "max_abs_error": max_error,
        "source_repository": "https://github.com/BlinkDL/RWKV-LM",
        "source_revision": SOURCE_REVISION,
        "source_paths": SOURCE_PATHS,
        "source_hash": _source_hash(root),
        "compiled_extension_hash": None,
        **_metadata(),
        "git": _git_metadata(root),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
