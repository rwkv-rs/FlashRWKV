# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from flashrwkv2.tmix.lnx_rkvres_xg import (
    infer_tmix_lnx_rkvres_xg_forward_varlen,
    pretrain_tmix_lnx_rkvres_xg_bf16,
)


def test_lnx_rkvres_xg_head_size_api() -> None:
    signature = inspect.signature(infer_tmix_lnx_rkvres_xg_forward_varlen)
    assert signature.parameters["head_size"].default == 64
    assert inspect.signature(pretrain_tmix_lnx_rkvres_xg_bf16).parameters["head_size"].default == 64


def test_lnx_rkvres_xg_dispatch_uses_scheduler_shape_metadata() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "csrc/sm120/tmix/lnx_rkvres_xg/infer_fp16_forward_varlen.cu"
    ).read_text()
    assert "const int64_t head_tasks" in source
    assert "batch_size) * max_seqlen * heads" in source
    assert "head_tasks >= 4096" in source
    assert "rows >= 4096" not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("head_size", (64, 128, 256))
def test_lnx_rkvres_xg_inference_packed(head_size: int) -> None:
    torch.manual_seed(21)
    device = torch.device("cuda")
    rows, heads = 3, 2
    channels = heads * head_size
    x, r, k, v, gate = [
        torch.randn(rows, channels, device=device, dtype=torch.float16)
        for _ in range(5)
    ]
    r_k = torch.randn(channels, device=device, dtype=torch.float16)
    weight = torch.randn(channels, device=device, dtype=torch.float16)
    bias = torch.randn(channels, device=device, dtype=torch.float16)
    output = infer_tmix_lnx_rkvres_xg_forward_varlen(
        x, r, k, v, r_k, weight, bias, gate, head_size=head_size
    )
    x_f = x.float().reshape(rows, heads, head_size)
    mean = x_f.mean(-1, keepdim=True)
    rstd = (x_f.var(-1, unbiased=False, keepdim=True) + 64.0e-5).rsqrt()
    residual = (
        r.float().reshape(rows, heads, head_size)
        * k.float().reshape(rows, heads, head_size)
        * r_k.float().reshape(1, heads, head_size)
    ).sum(-1, keepdim=True)
    expected = (
        (x_f - mean) * rstd * weight.float().reshape(1, heads, head_size)
        + bias.float().reshape(1, heads, head_size)
        + residual * v.float().reshape(rows, heads, head_size)
    ) * gate.float().reshape(rows, heads, head_size)
    assert torch.allclose(
        output.float().reshape_as(expected), expected, atol=0.04, rtol=0.04
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("head_size", (64, 128, 256))
def test_lnx_rkvres_xg_forward_backward(head_size: int) -> None:
    torch.manual_seed(23)
    device = torch.device("cuda")
    b, t, c = 2, 3, head_size * 2
    tokens = [
        (torch.randn(b, t, c, device=device) * 0.03).to(torch.bfloat16).requires_grad_()
        for _ in range(4)
    ]
    residual = (torch.randn(2, head_size, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    weight = torch.randn(c, device=device, dtype=torch.bfloat16).requires_grad_()
    bias = torch.randn(c, device=device, dtype=torch.bfloat16).requires_grad_()
    gate = (torch.randn(b, t, c, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    output = pretrain_tmix_lnx_rkvres_xg_bf16(
        tokens[0], tokens[1], tokens[2], tokens[3], residual, weight, bias, gate,
        head_size=head_size,
    )
    output.float().square().mean().backward()
    assert output.shape == tokens[0].shape
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in (*tokens, residual, weight, bias, gate))
