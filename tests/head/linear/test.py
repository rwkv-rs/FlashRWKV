# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashrwkv2.head.linear import (
    infer_head_last_norm_forward_varlen,
    infer_head_linear_all_forward_varlen,
    infer_head_linear_forward_varlen,
    infer_head_linear_last_forward_varlen,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_head_linear_and_indexed_last_norm() -> None:
    torch.manual_seed(59)
    device = torch.device("cuda")
    rows, channels, vocab = 4, 8, 11
    x = torch.randn(rows, channels, device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(vocab, channels, device=device, dtype=torch.float16)
    logits = infer_head_linear_forward_varlen(x, weight)
    assert torch.allclose(logits.float(), x.float() @ weight.float().t(), atol=0.04, rtol=0.04)
    all_logits = infer_head_linear_all_forward_varlen(x, weight)
    assert torch.allclose(all_logits.float(), x.float() @ weight.float().t(), atol=0.04, rtol=0.04)
    last_logits = infer_head_linear_last_forward_varlen(x, weight, tokens_count=2)
    assert torch.allclose(last_logits.float(), x.float() @ weight.float().t(), atol=0.04, rtol=0.04)
    last_indices = torch.tensor([3, 1], device=device, dtype=torch.int64)
    affine_weight = torch.randn(channels, device=device, dtype=torch.float16)
    bias = torch.randn(channels, device=device, dtype=torch.float16)
    output = infer_head_last_norm_forward_varlen(
        x, residual, last_indices, affine_weight, bias
    )
    expected = F.layer_norm((x + residual).float(), (channels,), affine_weight.float(), bias.float(), 1.0e-5)
    assert torch.allclose(
        output.float(), expected.index_select(0, last_indices), atol=0.04, rtol=0.04
    )
