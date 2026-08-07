# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flashrwkv2.cmix.sparse import (
    infer_cmix_sparse_down_relu_forward_varlen,
    infer_cmix_sparse_forward_varlen,
    infer_cmix_sparse_up_forward_varlen,
)
from flashrwkv2.tmix.wkv7 import prepare_recurrent_metadata


def test_cmix_sparse_down_dispatch_uses_batch_max_metadata() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "csrc/sm120/cmix/sparse/infer_fp16_forward_varlen.cu"
    ).read_text()
    assert "const int64_t dispatch_rows" in source
    assert "batch_size * max_seqlen" in source
    assert "if (dispatch_rows >= 8" in source
    assert "} else if (dispatch_rows == 1)" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cmix_sparse_ragged_up_down_and_combined() -> None:
    torch.manual_seed(41)
    device = torch.device("cuda")
    lengths = [2, 3]
    rows, channels, features = sum(lengths), 256, 128
    x = torch.randn(rows, channels, device=device, dtype=torch.float16)
    x_k = torch.randn(channels, device=device, dtype=torch.float16)
    key_fc = torch.randn(features, channels, device=device, dtype=torch.float16)
    value_fc = torch.randn(features, channels, device=device, dtype=torch.float16)
    initial_shift = torch.randn(4, channels, device=device, dtype=torch.float16)
    cu = torch.tensor([0, 2, 5], device=device, dtype=torch.int32)
    slots = torch.tensor([3, 1], device=device, dtype=torch.int32)
    mixed = x.float().clone()
    expected_shift = initial_shift.clone()
    for seq, slot in enumerate(slots.tolist()):
        start, end = cu[seq].item(), cu[seq + 1].item()
        previous = initial_shift[slot].float()
        for row in range(start, end):
            current = x[row].float()
            mixed[row] = current + (previous - current) * x_k.float()
            previous = current
        expected_shift[slot] = previous.to(torch.float16)
    expected_preact = mixed @ key_fc.float().t()
    expected_act = torch.relu(expected_preact).square()
    expected_output = expected_act @ value_fc.float()
    down_preact = torch.randn(rows, features, device=device, dtype=torch.float16)
    expected_down = torch.relu(down_preact.float()).square() @ value_fc.float()
    shift_for_up = initial_shift.clone()
    up = infer_cmix_sparse_up_forward_varlen(
        x, x_k, key_fc, shift_state_pool=shift_for_up, cu_seqlens=cu, state_indices=slots
    )
    down = infer_cmix_sparse_down_relu_forward_varlen(down_preact, value_fc)
    shift_for_combined = initial_shift.clone()
    combined = infer_cmix_sparse_forward_varlen(
        x,
        x_k,
        key_fc,
        value_fc,
        shift_state_pool=shift_for_combined,
        cu_seqlens=cu,
        state_indices=slots,
    )
    assert torch.allclose(up.float(), expected_act, atol=0.08, rtol=0.08)
    assert torch.allclose(down.float(), expected_down, atol=0.1, rtol=0.1)
    # The canonical down path accumulates half2 atomics in the same FP16
    # association as Albatross; use a scale-aware tolerance for the packed
    # reference rather than requiring a different FP32 reduction order.
    assert torch.allclose(combined.float(), expected_output, atol=16.0, rtol=0.1)
    assert torch.equal(shift_for_up, expected_shift)
    assert torch.equal(shift_for_combined, expected_shift)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cmix_sparse_ticket_rejects_metadata_mutation_without_state_write() -> None:
    device = torch.device("cuda")
    rows, channels, features = 5, 16, 8
    x = torch.randn(rows, channels, device=device, dtype=torch.float16)
    x_k = torch.randn(channels, device=device, dtype=torch.float16)
    key_fc = torch.randn(features, channels, device=device, dtype=torch.float16)
    shift_state = torch.randn(4, channels, device=device, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 2, rows], device=device, dtype=torch.int32)
    state_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    ticket = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=rows,
        state_pool_size=shift_state.shape[0],
    )

    before = shift_state.clone()
    state_indices[1] = 2
    with pytest.raises(RuntimeError, match="version"):
        infer_cmix_sparse_up_forward_varlen(
            x,
            x_k,
            key_fc,
            shift_state_pool=shift_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket,
        )
    assert torch.equal(shift_state, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cmix_sparse_rejects_wrong_value_layout() -> None:
    device = torch.device("cuda")
    preact = torch.zeros(2, 4, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match=r"\[F,C\]"):
        infer_cmix_sparse_down_relu_forward_varlen(
            preact, torch.zeros(5, 4, device=device, dtype=torch.float16)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cmix_sparse_one_and_t512_dispatch() -> None:
    torch.manual_seed(43)
    device = torch.device("cuda")

    # This is the exact upstream B=1,T=1 one-row family.
    channels, features = 256, 128
    x = torch.randn(1, channels, device=device, dtype=torch.float16)
    x_k = torch.randn(channels, device=device, dtype=torch.float16)
    key_fc = torch.randn(features, channels, device=device, dtype=torch.float16)
    value_fc = torch.randn(features, channels, device=device, dtype=torch.float16)
    shift = torch.randn(3, channels, device=device, dtype=torch.float16)
    cu = torch.tensor([0, 1], device=device, dtype=torch.int32)
    slots = torch.tensor([2], device=device, dtype=torch.int32)
    previous = shift[2].float()
    mixed = x.float() + (previous - x.float()) * x_k.float()
    expected = torch.relu(mixed @ key_fc.float().t()).square() @ value_fc.float()
    actual = infer_cmix_sparse_forward_varlen(
        x,
        x_k,
        key_fc,
        value_fc,
        shift_state_pool=shift,
        cu_seqlens=cu,
        state_indices=slots,
        max_seqlen=1,
    )
    assert torch.allclose(actual.float(), expected, atol=8.0, rtol=0.1)
    assert torch.equal(shift[2], x[0])

    # The canonical (B=8,T=1,C=4096,F=16384) caller policy selects the
    # two-accumulator T512 reuse family without exposing a forced selector.
    rows = 8
    preact = torch.randn(rows, 16384, device=device, dtype=torch.float16)
    value_fc = torch.randn(16384, 4096, device=device, dtype=torch.float16)
    expected_t512 = torch.relu(preact.float()).square() @ value_fc.float()
    actual_t512 = infer_cmix_sparse_down_relu_forward_varlen(
        preact, value_fc, batch_size=8, max_seqlen=1
    )
    assert torch.allclose(actual_t512.float(), expected_t512, atol=128.0, rtol=0.12)

    # The canonical no-FC value-loop table selects the split2 body before the
    # T512 threshold for both the one-row and packed-row families.
    preact_one = torch.randn(1, 16384, device=device, dtype=torch.float16)
    expected_one = torch.relu(preact_one.float()).square() @ value_fc.float()
    actual_one = infer_cmix_sparse_down_relu_forward_varlen(
        preact_one, value_fc, batch_size=1, max_seqlen=1
    )
    assert torch.allclose(actual_one.float(), expected_one, atol=128.0, rtol=0.12)

    preact_rows = torch.randn(4, 16384, device=device, dtype=torch.float16)
    expected_rows = torch.relu(preact_rows.float()).square() @ value_fc.float()
    actual_rows = infer_cmix_sparse_down_relu_forward_varlen(
        preact_rows, value_fc, batch_size=2, max_seqlen=2
    )
    assert torch.allclose(actual_rows.float(), expected_rows, atol=128.0, rtol=0.12)
