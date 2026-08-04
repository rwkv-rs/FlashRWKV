# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flash_rwkv import infer_cmix_mix_fp16_varlen, infer_tmix_mix6_fp16_varlen


def test_packed_inference_rejects_cpu_metadata_before_native_dispatch() -> None:
    x = torch.zeros(3, 64, dtype=torch.float16)
    state_pool = torch.zeros(2, 64, dtype=torch.float16)
    state_indices = torch.zeros(1, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(ValueError, match="CUDA"):
        infer_cmix_mix_fp16_varlen(
            x,
            state_pool,
            state_indices,
            cu_seqlens,
            torch.zeros(64, dtype=torch.float16),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("provide_token_batch_indices", [False, True])
def test_cmix_packed_matches_reference_and_updates_state(
    provide_token_batch_indices: bool,
) -> None:
    device = torch.device("cuda")
    channels = 128
    x = torch.randn(5, channels, device=device, dtype=torch.float16)
    state_pool = torch.randn(4, channels, device=device, dtype=torch.float16)
    initial_state = state_pool.clone()
    state_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 5], device=device, dtype=torch.int32)
    token_batch_indices = (
        torch.tensor([0, 0, 1, 1, 1], device=device, dtype=torch.int32)
        if provide_token_batch_indices
        else None
    )
    mix = torch.randn(channels, device=device, dtype=torch.float16).mul_(0.2)

    output = infer_cmix_mix_fp16_varlen(
        x,
        state_pool,
        state_indices,
        cu_seqlens,
        mix,
        token_batch_indices=token_batch_indices,
    )

    previous = torch.empty_like(x)
    previous[:2] = initial_state[1]
    previous[2:] = initial_state[3]
    previous[1] = x[0]
    previous[3:] = x[2:-1]
    reference = (x.float() + (previous.float() - x.float()) * mix.float()).half()
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    torch.testing.assert_close(state_pool[1], x[1], rtol=0, atol=0)
    torch.testing.assert_close(state_pool[3], x[4], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tmix_packed_matches_reference_without_token_map() -> None:
    device = torch.device("cuda")
    channels = 128
    x = torch.randn(5, channels, device=device, dtype=torch.float16)
    state_pool = torch.randn(4, channels, device=device, dtype=torch.float16)
    initial_state = state_pool.clone()
    state_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 5], device=device, dtype=torch.int32)
    mixes = tuple(
        torch.randn(channels, device=device, dtype=torch.float16).mul_(0.2)
        for _ in range(6)
    )

    outputs = infer_tmix_mix6_fp16_varlen(
        x, state_pool, state_indices, cu_seqlens, mixes
    )

    previous = torch.empty_like(x)
    previous[:2] = initial_state[1]
    previous[2:] = initial_state[3]
    previous[1] = x[0]
    previous[3:] = x[2:-1]
    delta = previous.float() - x.float()
    reference = tuple((x.float() + delta * mix.float()).half() for mix in mixes)
    for output, expected in zip(outputs, reference, strict=True):
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(state_pool[1], x[1], rtol=0, atol=0)
    torch.testing.assert_close(state_pool[3], x[4], rtol=0, atol=0)
