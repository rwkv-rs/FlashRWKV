# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.tmix.vres_gate import (
    infer_tmix_vres_gate_forward_varlen,
    pretrain_tmix_vres_gate_bf16,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vres_gate_inference_packed() -> None:
    torch.manual_seed(31)
    device = torch.device("cuda")
    rows, channels = 5, 128
    v = torch.randn(rows, channels, device=device, dtype=torch.float16)
    first = torch.randn_like(v)
    v0 = torch.randn(channels, device=device, dtype=torch.float16)
    v12 = torch.randn_like(v)
    output = infer_tmix_vres_gate_forward_varlen(v, first, v0, v12)
    expected = v.float() + (first.float() - v.float()) * torch.sigmoid(v0.float() + v12.float())
    assert torch.allclose(output.float(), expected, atol=0.04, rtol=0.04)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vres_gate_training_forward_backward() -> None:
    torch.manual_seed(32)
    device = torch.device("cuda")
    value = (torch.randn(1, 2, 8, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    first = value.detach().clone().requires_grad_()
    v0 = torch.randn(8, device=device, dtype=torch.bfloat16, requires_grad=True)
    v12 = torch.randn_like(value, requires_grad=True)
    output = pretrain_tmix_vres_gate_bf16(value, first, v0, v12)
    output.float().square().mean().backward()
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in (value, first, v0, v12)
    )
