# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from . import _extension


def _check(tensors: dict[str, torch.Tensor], x: torch.Tensor, heads: int) -> None:
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA bfloat16")
    for name in ("r", "k", "v", "g"):
        if tensors[name].shape != x.shape:
            raise ValueError(f"{name} must match x")
    if x.ndim != 3 or x.numel() == 0 or x.shape[-1] % 64:
        raise ValueError("x must have non-empty shape [B,T,C], C divisible by 64")
    if tensors["residual_scale"].shape != (heads, 64):
        raise ValueError("residual_scale must have shape [C/64,64]")
    if tensors["weight"].shape != (x.shape[-1],) or tensors["bias"].shape != tensors["weight"].shape:
        raise ValueError("weight and bias must have shape [C]")
    if any(tensor.device != x.device for tensor in tensors.values()):
        raise ValueError("lnx tensors must share a device")


class _Lnx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, r, k, v, residual_scale, weight, bias, g):
        output, mean, rstd = _extension().pretrain_tmix_lnx_rkvres_xg_forward(
            x, r, k, v, residual_scale, weight, bias, g
        )
        ctx.save_for_backward(x, r, k, v, residual_scale, weight, bias, g, mean, rstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return (None,) * 8
        gradients = _extension().pretrain_tmix_lnx_rkvres_xg_backward(
            grad_output.contiguous(), *ctx.saved_tensors
        )
        return tuple(
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad, strict=True)
        )


def pretrain_tmix_lnx_rkvres_xg_bf16(
    x, r, k, v, residual_scale, weight, bias, g
):
    """Train-temp head-wise LN, recurrent residual and output gate."""

    tensors = {
        "x": x,
        "r": r,
        "k": k,
        "v": v,
        "residual_scale": residual_scale,
        "weight": weight,
        "bias": bias,
        "g": g,
    }
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    _check(tensors, x, x.shape[-1] // 64 if x.ndim >= 3 else 0)
    return _Lnx.apply(x, r, k, v, residual_scale, weight, bias, g)


__all__ = ["pretrain_tmix_lnx_rkvres_xg_bf16"]
