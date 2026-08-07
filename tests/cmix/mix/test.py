# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.cmix.mix import (
    infer_cmix_add_layer_norm_mix_forward_varlen,
    infer_cmix_linear_ffn_down_forward_varlen,
    infer_cmix_mix_forward_varlen,
    infer_cmix_relu_square_forward_varlen,
    pretrain_cmix_bf16,
)
from flashrwkv2.tmix.wkv7 import prepare_recurrent_metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pretrain_cmix_forward_backward() -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    b, t, c = 1, 3, 4
    x = (torch.randn(b, t, c, device=device) * 0.03).to(torch.bfloat16).requires_grad_()
    x_k = (torch.randn(c, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    key = (torch.randn(4 * c, c, device=device) * 0.05).to(torch.bfloat16).requires_grad_()
    value = (torch.randn(c, 4 * c, device=device) * 0.05).to(torch.bfloat16).requires_grad_()
    output = pretrain_cmix_bf16(x, x_k, key, value)
    previous = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
    mixed = x + (previous - x) * x_k
    preact = mixed.float().reshape(-1, c) @ key.float().t()
    activation = torch.relu(preact) ** 2
    expected = (activation @ value.float().t()).reshape_as(x)
    assert torch.allclose(output.float(), expected.float(), atol=0.03, rtol=0.03)
    relu_square_input = torch.randn(3, 8, device=device, dtype=torch.float16)
    relu_square_output = infer_cmix_relu_square_forward_varlen(relu_square_input)
    assert torch.allclose(
        relu_square_output.float(),
        torch.relu(relu_square_input.float()).square(),
        atol=0.002,
        rtol=0.002,
    )
    output.float().sum().backward()
    assert torch.isfinite(x.grad.float()).all()
    assert torch.isfinite(x_k.grad.float()).all()
    assert torch.isfinite(key.grad.float()).all()
    assert torch.isfinite(value.grad.float()).all()

    # CMix's dense FFN-down caller owns the Albatross C=4096 tuned table.
    # The table is selected internally for the canonical 48-row shape; this
    # test only observes the exact GEMM result, not a forced algorithm API.
    rows, hidden, channels = 48, 16384, 4096
    down_x = torch.randn(rows, hidden, device=device, dtype=torch.float16) * 0.01
    down_weight = torch.randn(hidden, channels, device=device, dtype=torch.float16) * 0.01
    down_output = infer_cmix_linear_ffn_down_forward_varlen(down_x, down_weight)
    expected_down = down_x.float() @ down_weight.float()
    assert torch.allclose(down_output.float(), expected_down, atol=0.04, rtol=0.04)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_cmix_ragged_ticket_updates_only_selected_shift_slots() -> None:
    device = torch.device("cuda")
    c = 8
    lengths = (1, 3)
    total = sum(lengths)
    x = torch.arange(total * c, device=device, dtype=torch.float16).reshape(total, c)
    x_k = torch.full((c,), 0.25, device=device, dtype=torch.float16)
    initial = torch.randn(5, c, device=device, dtype=torch.float16)
    shift_state = initial.clone()
    cu_seqlens = torch.tensor([0, lengths[0], total], device=device, dtype=torch.int32)
    state_indices = torch.tensor([0, 4], device=device, dtype=torch.int32)
    ticket = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=total,
        state_pool_size=shift_state.shape[0],
    )

    output = infer_cmix_mix_forward_varlen(
        x,
        x_k,
        shift_state_pool=shift_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        validated_metadata=ticket,
    )

    expected = []
    expected_shift = initial.clone()
    start = 0
    for sequence, length in enumerate(lengths):
        slot = int(state_indices[sequence].item())
        previous = initial[slot]
        for token in range(start, start + length):
            current = x[token]
            expected.append(current + (previous - current) * x_k)
            previous = current
        expected_shift[slot] = previous
        start += length

    assert torch.allclose(output, torch.stack(expected), atol=0.01, rtol=0.01)
    assert torch.allclose(shift_state, expected_shift, atol=0.01, rtol=0.01)
    untouched = [slot for slot in range(initial.shape[0]) if slot not in (0, 4)]
    assert torch.equal(shift_state[untouched], initial[untouched])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_cmix_ticket_rejects_metadata_mutation_without_state_write() -> None:
    device = torch.device("cuda")
    c = 8
    total = 5
    x = torch.randn(total, c, device=device, dtype=torch.float16)
    x_k = torch.randn(c, device=device, dtype=torch.float16)
    shift_state = torch.randn(4, c, device=device, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 2, total], device=device, dtype=torch.int32)
    state_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    ticket = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=total,
        state_pool_size=shift_state.shape[0],
    )

    before = shift_state.clone()
    state_indices[1] = 2
    with pytest.raises(RuntimeError, match="version"):
        infer_cmix_mix_forward_varlen(
            x,
            x_k,
            shift_state_pool=shift_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket,
        )
    assert torch.equal(shift_state, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_cmix_fused_add_layer_norm_matches_albatross_t1_path() -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")
    b, c = 2, 4096
    eps = 1.0e-5
    x = (torch.randn(b, c, device=device) * 0.03).to(torch.float16)
    residual = (torch.randn(b, c, device=device) * 0.02).to(torch.float16)
    weight = (torch.randn(c, device=device) * 0.1 + 1.0).to(torch.float16)
    bias = (torch.randn(c, device=device) * 0.01).to(torch.float16)
    x_k = torch.full((c,), 0.25, device=device, dtype=torch.float16)
    initial = torch.randn(6, c, device=device, dtype=torch.float16)
    shift_state = initial.clone()
    cu_seqlens = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    state_indices = torch.tensor([4, 1], device=device, dtype=torch.int32)
    ticket = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=b,
        state_pool_size=shift_state.shape[0],
        max_seqlen=1,
    )

    x_out, mixed = infer_cmix_add_layer_norm_mix_forward_varlen(
        x,
        residual,
        weight,
        bias,
        x_k,
        shift_state_pool=shift_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        validated_metadata=ticket,
    )

    summed = x.float() + residual.float()
    mean = summed.mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt((summed - mean).square().mean(dim=-1, keepdim=True) + eps)
    normalized = ((summed - mean) * rstd * weight.float() + bias.float()).to(torch.float16)
    expected_mixed = (
        normalized.float()
        + (initial[state_indices.long()].float() - normalized.float()) * x_k.float()
    ).to(torch.float16)

    assert torch.allclose(x_out, summed.to(torch.float16), atol=0.01, rtol=0.01)
    assert torch.allclose(mixed, expected_mixed, atol=0.01, rtol=0.01)
    expected_state = initial.clone()
    expected_state[state_indices.long()] = normalized
    assert torch.allclose(shift_state, expected_state, atol=0.01, rtol=0.01)
    assert torch.equal(shift_state[[0, 2, 3, 5]], initial[[0, 2, 3, 5]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_cmix_fused_rejects_non_t1_dispatch() -> None:
    device = torch.device("cuda")
    x = torch.zeros(2, 4096, device=device, dtype=torch.float16)
    residual = torch.zeros_like(x)
    parameter = torch.ones(4096, device=device, dtype=torch.float16)
    shift_state = torch.zeros(4, 4096, device=device, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    state_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="max_seqlen=1"):
        infer_cmix_add_layer_norm_mix_forward_varlen(
            x,
            residual,
            parameter,
            torch.zeros_like(parameter),
            parameter,
            shift_state_pool=shift_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            max_seqlen=2,
        )
