# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import statistics

import torch

from flashrwkv2 import (
    infer_sampling_six_parameter_forward_varlen,
    infer_sampling_temperature_topk_topp_forward_varlen,
    setup_sampling_states,
)


def _time_cuda(call, *, warmup: int, samples: int) -> list[float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0)
    return values


def _profile(batch_size: int, vocab_size: int, provider: str, *, warmup: int, samples: int) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(20260807)
    logits = torch.randn(
        batch_size,
        vocab_size,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    num_slots = batch_size * 2
    slot_indices = torch.arange(0, num_slots, 2, dtype=torch.int32, device="cuda")
    states = setup_sampling_states(20260807, num_slots)
    top_k = 50
    top_p = 0.9

    deterministic = infer_sampling_temperature_topk_topp_forward_varlen(
        logits, states.clone(), slot_indices, top_k=1
    )
    expected = logits.argmax(dim=-1).to(torch.int32)
    if not torch.equal(deterministic, expected):
        raise RuntimeError("sampling top-k=1 correctness check failed")

    if provider == "scalar":
        call = lambda: infer_sampling_temperature_topk_topp_forward_varlen(
            logits, states, slot_indices, temperature=1.0, top_k=top_k, top_p=top_p
        )
    elif provider == "per_request":
        temperatures = torch.linspace(0.8, 1.2, batch_size, device="cuda")
        top_ks = torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda")
        top_ps = torch.linspace(0.8, top_p, batch_size, device="cuda")
        call = lambda: infer_sampling_temperature_topk_topp_forward_varlen(
            logits,
            states,
            slot_indices,
            temperature=temperatures,
            top_k=top_ks,
            top_p=top_ps,
        )
    elif provider == "six_parameter":
        penalties = torch.zeros(num_slots, vocab_size, dtype=torch.float32, device="cuda")
        presence = torch.full((batch_size,), 0.1, device="cuda")
        frequency = torch.full((batch_size,), 0.1, device="cuda")
        decays = torch.full((batch_size,), 0.996, device="cuda")
        temperatures = torch.ones(batch_size, device="cuda")
        top_ks = torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda")
        top_ps = torch.full((batch_size,), top_p, device="cuda")
        call = lambda: infer_sampling_six_parameter_forward_varlen(
            logits,
            penalties,
            states,
            slot_indices,
            presence_penalty=presence,
            frequency_penalty=frequency,
            penalty_decay=decays,
            temperature=temperatures,
            top_k=top_ks,
            top_p=top_ps,
        )
    else:
        raise ValueError(f"unknown provider: {provider}")

    raw = _time_cuda(call, warmup=warmup, samples=samples)
    return {
        "profile": f"{provider}/b{batch_size}/v{vocab_size}",
        "provider": provider,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "raw_latency_us": raw,
        "p50_us": statistics.median(raw),
        "correctness": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[65536, 131072])
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=("scalar", "per_request", "six_parameter"),
        default=["scalar", "per_request", "six_parameter"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    results = [
        _profile(batch_size, vocab_size, provider, warmup=args.warmup, samples=args.samples)
        for vocab_size in args.vocab_sizes
        for batch_size in args.batch_sizes
        for provider in args.providers
    ]
    print(
        json.dumps(
            {
                "operator": "sampling",
                "source_revision": "e0297f7830c3fa581d49ddddddba32f35ea7f733",
                "adaptation_revision": "fd440426689f10e240b5761e1a7c82e4c37deb8d",
                "gpu": torch.cuda.get_device_name(),
                "correctness": "passed",
                "results": results,
            }
        )
    )


if __name__ == "__main__":
    main()
