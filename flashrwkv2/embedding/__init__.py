# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ..tmix.wkv7 import _extension


def infer_embedding_ln0_forward_varlen(
    embedding: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    for name, tensor in (("embedding", embedding), ("weight", weight), ("bias", bias)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be CUDA and contiguous")
        if tensor.device != embedding.device:
            raise ValueError(f"{name} must share embedding's device")
    if embedding.ndim != 2 or embedding.shape[0] <= 0:
        raise ValueError("embedding must have packed shape [rows,C]")
    if weight.shape != (embedding.shape[1],) or bias.shape != weight.shape:
        raise ValueError("weight and bias must have shape [C]")
    return _extension().embedding_ln0_forward_varlen(embedding, weight, bias, float(eps))


__all__ = ["infer_embedding_ln0_forward_varlen"]
