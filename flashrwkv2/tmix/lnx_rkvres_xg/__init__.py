# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ..wkv7 import _extension


def infer_tmix_lnx_rkvres_xg_forward_varlen(
    x: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r_k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    g: torch.Tensor,
    *,
    head_size: int = 64,
    batch_size: int = 1,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    """Run Albatross TMix LN/rkv-residual/gate on packed rows."""

    if head_size not in {64, 128, 256}:
        raise ValueError("head_size must be one of 64, 128, or 256")
    tensors = (x, r, k, v, g)
    names = ("x", "r", "k", "v", "g")
    for name, tensor in zip(names, tensors, strict=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float16:
            raise TypeError(f"{name} must have dtype torch.float16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be CUDA and contiguous")
        if tensor.device != x.device:
            raise ValueError(f"{name} must share x's device")
    if (
        x.ndim != 2
        or x.shape[0] <= 0
        or x.shape[1] <= 0
        or x.shape[1] % head_size
    ):
        raise ValueError("x must have packed shape [total_tokens,H*head_size]")
    if any(tensor.shape != x.shape for tensor in (r, k, v, g)):
        raise ValueError("r, k, v and g must match x's packed shape")
    channels = x.shape[1]
    for name, tensor in (("r_k", r_k), ("weight", weight), ("bias", bias)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA float16")
        if tensor.device != x.device or tensor.shape != (channels,):
            raise ValueError(f"{name} must have shape [C] and share x's device")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if max_seqlen is None:
        max_seqlen = x.shape[0]
    if (
        not isinstance(max_seqlen, int)
        or isinstance(max_seqlen, bool)
        or max_seqlen <= 0
    ):
        raise ValueError("max_seqlen must be a positive integer")
    return _extension().tmix_lnx_rkvres_xg_forward_varlen(
        x,
        r,
        k,
        v,
        r_k,
        weight,
        bias,
        g,
        int(head_size),
        int(batch_size),
        int(max_seqlen),
    )


__all__ = ["infer_tmix_lnx_rkvres_xg_forward_varlen"]

from .pretrain import pretrain_tmix_lnx_rkvres_xg_bf16

__all__.append("pretrain_tmix_lnx_rkvres_xg_bf16")
