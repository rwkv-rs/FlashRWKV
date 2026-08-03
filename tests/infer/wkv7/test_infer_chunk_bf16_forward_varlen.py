# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional

from flash_rwkv import (
    infer_chunk_bf16_forward,
    infer_chunk_bf16_forward_varlen,
)
from flash_rwkv.reference import rwkv7_decay_logits_reference

HEAD_SIZE = 64
TOLERANCE = json.loads(
    (Path(__file__).parents[2] / "fixtures/tolerances-v1.json").read_text(
        encoding="utf-8"
    )
)["bf16_kda_chunk"]


@pytest.fixture(scope="module", autouse=True)
def require_flash_rwkv_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    try:
        from flash_rwkv import _C  # noqa: F401
    except ImportError as error:
        pytest.fail(f"FlashRWKV CUDA extension is unavailable: {error!r}")


def _inputs(
    batch_size: int,
    sequence_length: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch_size, sequence_length, 2, HEAD_SIZE)

    def normal(scale: float) -> torch.Tensor:
        return scale * torch.randn(
            shape,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )

    direction = functional.normalize(normal(1.0), dim=-1)
    strength = 0.08 * torch.rand(
        shape,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    tensors = (
        normal(0.05),
        normal(1.0),
        normal(0.05),
        normal(0.05),
        -direction,
        direction * strength,
    )
    return tuple(tensor.to(torch.bfloat16).contiguous() for tensor in tensors)


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return float((error / baseline).item())


@pytest.mark.parametrize("sequence_length", [1, 15, 16, 17, 33])
def test_kda_k1_k2_fixed_matches_fp32_oracle(sequence_length: int) -> None:
    inputs = _inputs(2, sequence_length, seed=2100 + sequence_length)
    initial_state = (
        0.02
        * torch.randn(
            2,
            2,
            HEAD_SIZE,
            HEAD_SIZE,
            device="cuda",
            dtype=torch.float32,
        )
    ).to(torch.bfloat16)
    decay_bias = torch.linspace(
        -0.2,
        0.2,
        2 * HEAD_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    ).reshape(2, HEAD_SIZE)
    expected_output, expected_state = rwkv7_decay_logits_reference(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        scale=0.125,
        decay_bias=decay_bias,
    )
    output, final_state = infer_chunk_bf16_forward(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        scale=0.125,
        decay_bias=decay_bias,
    )
    torch.cuda.synchronize()

    assert final_state is not None
    assert _relative_rmse(output, expected_output) <= TOLERANCE[
        "output_relative_rmse"
    ]
    assert _relative_rmse(final_state, expected_state) <= TOLERANCE[
        "state_relative_rmse"
    ]


def test_kda_k1_k2_varlen_isolates_sequences() -> None:
    sequence_lengths = (1, 17, 33)
    inputs = _inputs(1, sum(sequence_lengths), seed=2201)
    cu_seqlens = torch.tensor(
        [0, 1, 18, 51],
        device="cuda",
        dtype=torch.int32,
    )
    initial_state = (
        0.02
        * torch.randn(
            3,
            2,
            HEAD_SIZE,
            HEAD_SIZE,
            device="cuda",
            dtype=torch.float32,
        )
    ).to(torch.bfloat16)
    decay_bias = torch.linspace(
        0.1,
        -0.1,
        2 * HEAD_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected_output, expected_state = rwkv7_decay_logits_reference(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        scale=0.125,
        decay_bias=decay_bias,
    )
    output, final_state = infer_chunk_bf16_forward_varlen(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        scale=0.125,
        decay_bias=decay_bias,
    )
    torch.cuda.synchronize()

    assert final_state is not None
    assert _relative_rmse(output, expected_output) <= TOLERANCE[
        "output_relative_rmse"
    ]
    assert _relative_rmse(final_state, expected_state) <= TOLERANCE[
        "state_relative_rmse"
    ]
