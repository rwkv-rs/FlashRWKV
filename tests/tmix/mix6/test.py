# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flash_rwkv.tmix.mix6 import (
    infer_tmix_mix6_add_layer_norm_forward_varlen,
    infer_tmix_mix6_forward_varlen,
    pretrain_tmix_mix6_bf16,
)
from flash_rwkv.tmix.wkv7 import prepare_recurrent_metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pretrain_mix6_forward_backward() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    x = (torch.randn(2, 3, 8, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    params = [(torch.randn(8, device=device) * 0.1).to(torch.bfloat16).requires_grad_() for _ in range(6)]
    outputs = pretrain_tmix_mix6_bf16(x, *params)
    reference = []
    previous = torch.zeros_like(x[:, :1])
    for parameter in params:
        mixed = x + (torch.cat((previous, x[:, :-1]), dim=1) - x) * parameter
        reference.append(mixed)
    for output, expected in zip(outputs, reference, strict=True):
        assert torch.allclose(output.float(), expected.float(), atol=0.01, rtol=0.01)
    sum(output.float().sum() for output in outputs).backward()
    assert torch.isfinite(x.grad.float()).all()
    for parameter in params:
        assert torch.isfinite(parameter.grad.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_mix6_consumes_packed_metadata_ticket_and_updates_last_shift() -> None:
    device = torch.device("cuda")
    b, c = 2, 8
    lengths = (2, 3)
    total = sum(lengths)
    x = torch.arange(total * c, device=device, dtype=torch.float16).reshape(total, c)
    params = [
        torch.full((c,), value, device=device, dtype=torch.float16)
        for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    ]
    initial = torch.randn(4, c, device=device, dtype=torch.float16)
    shift_state = initial.clone()
    cu_seqlens = torch.tensor([0, lengths[0], total], device=device, dtype=torch.int32)
    state_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    ticket = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=total,
        state_pool_size=shift_state.shape[0],
    )

    outputs = infer_tmix_mix6_forward_varlen(
        x,
        *params,
        shift_state_pool=shift_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        validated_metadata=ticket,
    )

    expected = [[] for _ in range(6)]
    expected_shift = initial.clone()
    start = 0
    for sequence, length in enumerate(lengths):
        slot = int(state_indices[sequence].item())
        previous = initial[slot]
        for token in range(start, start + length):
            current = x[token]
            for index, parameter in enumerate(params):
                expected[index].append(current + (previous - current) * parameter)
            previous = current
        expected_shift[slot] = previous
        start += length

    for output, rows in zip(outputs, expected, strict=True):
        assert torch.allclose(output, torch.stack(rows), atol=0.01, rtol=0.01)
    assert torch.allclose(shift_state, expected_shift, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_mix6_ticket_rejects_metadata_mutation_without_state_write() -> None:
    device = torch.device("cuda")
    c = 8
    total = 5
    x = torch.randn(total, c, device=device, dtype=torch.float16)
    params = [torch.randn(c, device=device, dtype=torch.float16) for _ in range(6)]
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
    cu_seqlens[1] += 1
    with pytest.raises(RuntimeError, match="version"):
        infer_tmix_mix6_forward_varlen(
            x,
            *params,
            shift_state_pool=shift_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket,
        )
    assert torch.equal(shift_state, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_mix6_fused_add_layer_norm_matches_albatross_t1_path() -> None:
    torch.manual_seed(19)
    device = torch.device("cuda")
    c = 4096
    eps = 1.0e-5
    x = (torch.randn(1, c, device=device) * 0.03).to(torch.float16)
    residual = (torch.randn(1, c, device=device) * 0.02).to(torch.float16)
    weight = (torch.randn(c, device=device) * 0.1 + 1.0).to(torch.float16)
    bias = (torch.randn(c, device=device) * 0.01).to(torch.float16)
    params = [
        (torch.randn(c, device=device) * 0.1).to(torch.float16)
        for _ in range(6)
    ]
    initial = torch.randn(5, c, device=device, dtype=torch.float16)
    shift_state = initial.clone()
    cu_seqlens = torch.tensor([0, 1], device=device, dtype=torch.int32)
    state_indices = torch.tensor([3], device=device, dtype=torch.int32)
    ticket = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=1,
        state_pool_size=shift_state.shape[0],
        max_seqlen=1,
    )

    outputs = infer_tmix_mix6_add_layer_norm_forward_varlen(
        x,
        residual,
        weight,
        bias,
        *params,
        shift_state_pool=shift_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        validated_metadata=ticket,
    )

    summed = x.float() + residual.float()
    mean = summed.mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt((summed - mean).square().mean(dim=-1, keepdim=True) + eps)
    normalized = ((summed - mean) * rstd * weight.float() + bias.float()).to(torch.float16)
    previous = initial[3]
    expected = [
        (
            normalized.float()
            + (previous.float() - normalized.float()) * parameter.float()
        ).to(torch.float16)
        for parameter in params
    ]

    assert torch.allclose(outputs[0], summed.to(torch.float16), atol=0.01, rtol=0.01)
    for output, expected_output in zip(outputs[1:], expected, strict=True):
        assert torch.allclose(output, expected_output, atol=0.01, rtol=0.01)
    expected_state = initial.clone()
    expected_state[3] = normalized[0]
    assert torch.allclose(shift_state, expected_state, atol=0.01, rtol=0.01)
    assert torch.equal(shift_state[[0, 1, 2, 4]], initial[[0, 1, 2, 4]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_mix6_fused_rejects_non_b1_dispatch() -> None:
    device = torch.device("cuda")
    x = torch.zeros(1, 4096, device=device, dtype=torch.float16)
    residual = torch.zeros_like(x)
    parameters = [torch.ones(4096, device=device, dtype=torch.float16) for _ in range(8)]
    cu_seqlens = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    state_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="B=1"):
        infer_tmix_mix6_add_layer_norm_forward_varlen(
            x,
            residual,
            *parameters[:2],
            *parameters[2:],
            shift_state_pool=torch.zeros(4, 4096, device=device, dtype=torch.float16),
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
        )
