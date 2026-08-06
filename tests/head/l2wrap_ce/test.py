# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flash_rwkv.head.l2wrap_ce import pretrain_head_l2wrap_ce_bf16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_head_l2wrap_forward_backward() -> None:
    torch.manual_seed(29)
    device = torch.device("cuda")
    hidden = torch.randn(1, 2, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(65536, 8, device=device, dtype=torch.bfloat16) * 0.01).requires_grad_()
    targets = torch.tensor([3, 4096], device=device, dtype=torch.int64)
    loss = pretrain_head_l2wrap_ce_bf16(hidden, weight, targets, chunk_rows=1)
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_head_l2wrap_rejects_invalid_target() -> None:
    device = torch.device("cuda")
    hidden = torch.zeros(1, 1, 8, device=device, dtype=torch.bfloat16)
    weight = torch.zeros(65536, 8, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="targets"):
        pretrain_head_l2wrap_ce_bf16(hidden, weight, torch.tensor([65536], device=device))
