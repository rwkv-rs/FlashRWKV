# SPDX-License-Identifier: MIT

"""Fused RWKV cross-entropy with the canonical L2Wrap surrogate gradient."""

from __future__ import annotations

import torch

from . import _extension


def pretrain_l2wrap_ce_bf16(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return mean cross-entropy with RWKV L2Wrap applied in backward.

    The forward scalar is ordinary mean cross-entropy. Backward adds
    ``1e-4 * max_logit / rows`` to the first maximum logit of each row, which
    is the established RWKV L2Wrap training contract.
    """

    _validate_l2wrap_inputs(logits, targets)
    return _PretrainL2WrapCeBf16Function.apply(logits, targets)


def _validate_l2wrap_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a torch.Tensor")
    if logits.dtype != torch.bfloat16:
        raise TypeError("logits must have dtype torch.bfloat16")
    if targets.dtype != torch.int64:
        raise TypeError("targets must have dtype torch.int64")
    if not logits.is_contiguous() or not targets.is_contiguous():
        raise ValueError("logits and targets must be contiguous")
    if logits.ndim < 2 or logits.shape[-1] <= 0:
        raise ValueError("logits must have non-empty shape [..., vocab]")
    rows = logits.numel() // logits.shape[-1]
    if rows <= 0 or targets.numel() != rows:
        raise ValueError(f"targets must contain exactly {rows} entries")
    if targets.device != logits.device:
        raise ValueError("targets must be on the same device as logits")
    if not logits.is_cuda:
        raise ValueError("pretrain_l2wrap_ce_bf16 requires CUDA tensors")


class _PretrainL2WrapCeBf16Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        loss, logsumexp, max_values, argmax = (
            _extension.pretrain_l2wrap_ce_bf16_forward(logits, targets)
        )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(logits, targets, logsumexp, max_values, argmax)
        return loss

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_loss: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, None]:
        if grad_loss is None or not ctx.needs_input_grad[0]:
            return None, None
        logits, targets, logsumexp, max_values, argmax = ctx.saved_tensors
        grad_logits = _extension.pretrain_l2wrap_ce_bf16_backward(
            grad_loss.contiguous(),
            logits,
            targets,
            logsumexp,
            max_values,
            argmax,
        )
        return grad_logits, None
