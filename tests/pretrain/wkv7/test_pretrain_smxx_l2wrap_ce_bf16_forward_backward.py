# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from flash_rwkv import pretrain_l2wrap_ce_bf16


def _inputs(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(731)
    logits = torch.randn(2, 3, 16, device=device, dtype=dtype).mul_(0.25)
    targets = torch.tensor(
        [[0, 7, 15], [3, 9, 1]],
        device=device,
        dtype=torch.int64,
    )
    return logits, targets


def _expected_gradient(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    rows = targets.numel()
    flattened = logits.float().reshape(rows, -1)
    gradient = torch.softmax(flattened, dim=-1)
    gradient[torch.arange(rows, device=logits.device), targets.reshape(-1)] -= 1
    gradient /= rows
    max_values, argmax = flattened.max(dim=-1)
    gradient[torch.arange(rows, device=logits.device), argmax] += (
        max_values * 1.0e-4 / rows
    )
    return gradient.reshape_as(logits).to(torch.bfloat16)


def test_l2wrap_rejects_non_cuda_inputs_before_loading_extension() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        pretrain_l2wrap_ce_bf16(*_inputs(device="cpu"))


def test_l2wrap_rejects_non_bf16_logits() -> None:
    with pytest.raises(TypeError, match="torch.bfloat16"):
        pretrain_l2wrap_ce_bf16(*_inputs(device="cpu", dtype=torch.float32))


def test_l2wrap_rejects_target_count_mismatch() -> None:
    logits, targets = _inputs(device="cpu")
    with pytest.raises(ValueError, match="exactly 6"):
        pretrain_l2wrap_ce_bf16(logits, targets.reshape(-1)[:-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_l2wrap_forward_and_surrogate_gradient_match_contract() -> None:
    logits, targets = _inputs(device="cuda")
    logits.requires_grad_(True)

    loss = pretrain_l2wrap_ce_bf16(logits, targets)
    reference_loss = functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
    loss.backward()

    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, reference_loss, atol=2.0e-4, rtol=2.0e-4)
    torch.testing.assert_close(
        logits.grad,
        _expected_gradient(logits.detach(), targets),
        atol=0.002,
        rtol=0.01,
    )
