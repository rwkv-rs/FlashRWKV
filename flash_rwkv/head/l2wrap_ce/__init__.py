# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ...tmix.wkv7 import _extension


def _check(hidden, weight, targets, chunk_rows: int) -> None:
    for name, tensor in (("hidden", hidden), ("weight", weight)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA bfloat16")
    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a torch.Tensor")
    if targets.dtype != torch.int64 or not targets.is_cuda or not targets.is_contiguous():
        raise ValueError("targets must be contiguous CUDA int64")
    if hidden.ndim != 3 or hidden.numel() == 0:
        raise ValueError("hidden must have shape [B,T,C]")
    if weight.shape != (65536, hidden.shape[-1]):
        raise ValueError("weight must have shape [65536,C]")
    if targets.numel() != hidden.shape[0] * hidden.shape[1]:
        raise ValueError("targets must contain one token per hidden row")
    if hidden.device != weight.device or hidden.device != targets.device:
        raise ValueError("head loss tensors must share a device")
    if torch.any((targets < 0) | (targets >= 65536)).item():
        raise ValueError("targets must be in [0,65536)")
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")


class _HeadL2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk_rows):
        loss, grad_hidden, grad_weight = _extension().pretrain_head_l2wrap_ce_forward(
            hidden, weight, targets, int(chunk_rows)
        )
        ctx.save_for_backward(grad_hidden, grad_weight)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        if grad_loss is None:
            return None, None, None, None
        grad_hidden, grad_weight = ctx.saved_tensors
        scale = grad_loss.to(grad_hidden.dtype)
        return (
            grad_hidden * scale if ctx.needs_input_grad[0] else None,
            grad_weight * grad_loss.to(grad_weight.dtype) if ctx.needs_input_grad[1] else None,
            None,
            None,
        )


def pretrain_head_l2wrap_ce_bf16(hidden, weight, targets, *, chunk_rows: int = 4096):
    """Memory-bounded train_temp output-head CE with L2Wrap backward."""

    _check(hidden, weight, targets, chunk_rows)
    return _HeadL2Wrap.apply(hidden, weight, targets, chunk_rows)


__all__ = ["pretrain_head_l2wrap_ce_bf16"]
