# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from flash_rwkv import pretrain_head_l2wrap_ce_bf16

_VOCAB_SIZE = 65_536


def _inputs(
    *,
    device: torch.device | str,
    channels: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1701)
    hidden = torch.randn(1, 3, channels, device=device, dtype=torch.bfloat16).mul_(0.2)
    weight = torch.randn(
        _VOCAB_SIZE,
        channels,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.02)
    targets = torch.tensor([[0, 1234, _VOCAB_SIZE - 1]], device=device)
    return hidden, weight, targets


def test_head_l2wrap_rejects_non_cuda_inputs_before_extension() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        pretrain_head_l2wrap_ce_bf16(*_inputs(device="cpu"))


def test_head_l2wrap_rejects_noncanonical_vocabulary() -> None:
    hidden, weight, targets = _inputs(device="cpu")
    with pytest.raises(ValueError, match="weight must have shape"):
        pretrain_head_l2wrap_ce_bf16(hidden, weight[:-1], targets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_head_l2wrap_loss_and_gradients_match_torch_contract() -> None:
    hidden, weight, targets = _inputs(device="cuda")
    hidden.requires_grad_(True)
    weight.requires_grad_(True)
    reference_hidden = hidden.detach().float().requires_grad_(True)
    reference_weight = weight.detach().float().requires_grad_(True)

    loss = pretrain_head_l2wrap_ce_bf16(
        hidden,
        weight,
        targets,
        chunk_rows=2,
    )
    logits = reference_hidden.reshape(-1, hidden.shape[-1]) @ reference_weight.T
    plain_loss = functional.cross_entropy(logits, targets.reshape(-1))
    max_logits = logits.max(dim=-1).values
    surrogate = 0.5e-4 * max_logits.square().sum() / targets.numel()
    (plain_loss + surrogate).backward()
    loss.backward()

    torch.testing.assert_close(loss, plain_loss, atol=0.002, rtol=0.002)
    torch.testing.assert_close(
        hidden.grad,
        reference_hidden.grad.to(torch.bfloat16),
        atol=0.003,
        rtol=0.03,
    )
    torch.testing.assert_close(
        weight.grad,
        reference_weight.grad.to(torch.bfloat16),
        atol=0.003,
        rtol=0.03,
    )
