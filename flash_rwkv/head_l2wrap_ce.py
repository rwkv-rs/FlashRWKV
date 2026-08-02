# SPDX-License-Identifier: MIT

"""Memory-bounded RWKV output head, cross-entropy, and L2Wrap operator."""

from __future__ import annotations

import torch

from . import _extension

_VOCAB_SIZE = 65_536


def pretrain_head_l2wrap_ce_bf16(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_rows: int = 4096,
) -> torch.Tensor:
    """Compute the fixed-vocabulary RWKV head loss without full logits storage.

    The forward value is mean cross-entropy. Its saved, chunk-produced
    gradients include the canonical RWKV L2Wrap surrogate.
    """

    _validate_inputs(hidden, weight, targets, chunk_rows)
    return _PretrainHeadL2WrapCeBf16Function.apply(
        hidden,
        weight,
        targets,
        chunk_rows,
    )


def _validate_inputs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_rows: int,
) -> None:
    for name, tensor in {"hidden": hidden, "weight": weight}.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a torch.Tensor")
    if targets.dtype != torch.int64:
        raise TypeError("targets must have dtype torch.int64")
    if not targets.is_contiguous():
        raise ValueError("targets must be contiguous")
    if hidden.ndim != 3 or any(dimension <= 0 for dimension in hidden.shape):
        raise ValueError("hidden must have non-empty shape [B, T, C]")
    if weight.shape != (_VOCAB_SIZE, hidden.shape[2]):
        raise ValueError(f"weight must have shape [{_VOCAB_SIZE}, {hidden.shape[2]}]")
    if targets.numel() != hidden.shape[0] * hidden.shape[1]:
        raise ValueError("targets must contain one token ID per hidden row")
    if weight.device != hidden.device or targets.device != hidden.device:
        raise ValueError("hidden, weight, and targets must be on the same device")
    if not hidden.is_cuda:
        raise ValueError("pretrain_head_l2wrap_ce_bf16 requires CUDA tensors")
    if torch.any((targets < 0) | (targets >= _VOCAB_SIZE)).item():
        raise ValueError(f"targets must be in [0, {_VOCAB_SIZE})")
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool):
        raise TypeError("chunk_rows must be an integer")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")


class _PretrainHeadL2WrapCeBf16Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        chunk_rows: int,
    ) -> torch.Tensor:
        loss, grad_hidden, grad_weight = (
            _extension.pretrain_head_l2wrap_ce_bf16_forward(
                hidden,
                weight,
                targets,
                chunk_rows,
            )
        )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(grad_hidden, grad_weight)
        return loss

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_loss: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        if grad_loss is None:
            return None, None, None, None
        grad_hidden, grad_weight = ctx.saved_tensors
        hidden_result = (
            grad_hidden * grad_loss.to(grad_hidden.dtype)
            if ctx.needs_input_grad[0]
            else None
        )
        weight_result = (
            grad_weight * grad_loss.to(grad_weight.dtype)
            if ctx.needs_input_grad[1]
            else None
        )
        return hidden_result, weight_result, None, None
