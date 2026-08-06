# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flash_rwkv.tmix.kk_pre import pretrain_tmix_kk_pre_bf16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_kk_pre_forward_backward() -> None:
    torch.manual_seed(19)
    device = torch.device("cuda")
    b, t, c = 2, 3, 128
    tensors = [
        (torch.randn(b, t, c, device=device) * 0.1).to(torch.bfloat16).requires_grad_(),
        (torch.randn(c, device=device) * 0.1).to(torch.bfloat16).requires_grad_(),
        (torch.randn(b, t, c, device=device) * 0.1).to(torch.bfloat16).requires_grad_(),
        (torch.randn(c, device=device) * 0.1).to(torch.bfloat16).requires_grad_(),
    ]
    outputs = pretrain_tmix_kk_pre_bf16(*tensors)
    assert len(outputs) == 3
    sum(output.float().square().mean() for output in outputs).backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_kk_pre_rejects_non_head_aligned_channels() -> None:
    device = torch.device("cuda")
    key = torch.zeros(1, 1, 65, device=device, dtype=torch.bfloat16)
    vector = torch.zeros(65, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by 64"):
        pretrain_tmix_kk_pre_bf16(key, vector, key, vector)
