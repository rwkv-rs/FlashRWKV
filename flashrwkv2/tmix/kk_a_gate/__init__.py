# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ..wkv7 import _extension


def infer_tmix_kk_a_gate_forward_varlen(
    k: torch.Tensor,
    k_k: torch.Tensor,
    a0: torch.Tensor,
    a12: torch.Tensor,
    k_a: torch.Tensor,
    *,
    head_size: int = 64,
    batch_size: int = 1,
    max_seqlen: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the Albatross key/key-a gate on packed token rows."""

    if head_size not in {64, 128, 256}:
        raise ValueError("head_size must be one of 64, 128, or 256")
    tensors = (k, k_k, a0, a12, k_a)
    names = ("k", "k_k", "a0", "a12", "k_a")
    for name, tensor in zip(names, tensors, strict=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float16:
            raise TypeError(f"{name} must have dtype torch.float16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be CUDA and contiguous")
        if tensor.device != k.device:
            raise ValueError(f"{name} must share k's device")
    if (
        k.ndim != 2
        or k.shape[0] <= 0
        or k.shape[1] <= 0
        or k.shape[1] % head_size
    ):
        raise ValueError("k must have packed shape [total_tokens,H*head_size]")
    if any(tensor.shape != (k.shape[1],) for tensor in (k_k, a0, k_a)):
        raise ValueError("k_k, a0 and k_a must have shape [C]")
    if a12.shape != k.shape:
        raise ValueError("a12 must match k's packed shape")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if max_seqlen is None:
        max_seqlen = k.shape[0]
    if (
        not isinstance(max_seqlen, int)
        or isinstance(max_seqlen, bool)
        or max_seqlen <= 0
    ):
        raise ValueError("max_seqlen must be a positive integer")
    return tuple(
        _extension().tmix_kk_a_gate_forward_varlen(
            k,
            k_k,
            a0,
            a12,
            k_a,
            int(head_size),
            int(batch_size),
            int(max_seqlen),
        )
    )  # type: ignore[return-value]


__all__ = ["infer_tmix_kk_a_gate_forward_varlen"]
