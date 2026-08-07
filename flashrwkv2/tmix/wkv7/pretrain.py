# SPDX-License-Identifier: MIT

"""Canonical RWKV-LM ``train_temp`` clampw v3 H100 training operator."""

from __future__ import annotations

import torch


HEAD_SIZE = 64
CHUNK_LEN = 16


def _validate_inputs(tensors: tuple[torch.Tensor, ...]) -> tuple[int, int, int]:
    if len(tensors) != 6:
        raise ValueError("clampw requires r, w, k, v, a, and b")
    r = tensors[0]
    if not isinstance(r, torch.Tensor):
        raise TypeError("r must be a torch.Tensor")
    if r.ndim != 3 or min(r.shape) <= 0:
        raise ValueError("r must have shape [B,T,C] with nonzero dimensions")
    if r.dtype != torch.bfloat16:
        raise TypeError("r must be bfloat16")
    if not r.is_cuda or not r.is_contiguous():
        raise ValueError("r must be contiguous CUDA")

    for name, tensor in zip(("w", "k", "v", "a", "b"), tensors[1:]):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA")
        if tensor.shape != r.shape:
            raise ValueError(f"{name} must match r shape")
        if tensor.device != r.device:
            raise ValueError(f"{name} must share the r device")

    batch, tokens, channels = r.shape
    if channels % HEAD_SIZE != 0:
        raise ValueError("C must be divisible by the canonical head size 64")
    if tokens % CHUNK_LEN != 0:
        raise ValueError("T must be divisible by the canonical chunk length 16")
    return batch, tokens, channels


class _PretrainRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, channels = _validate_inputs((r, w, k, v, a, b))
        heads = channels // HEAD_SIZE
        values = tuple(
            tensor.view(batch, tokens, heads, HEAD_SIZE)
            for tensor in (r, w, k, v, a, b)
        )
        output = torch.empty_like(values[3])
        boundary = torch.empty(
            (batch, heads, tokens // CHUNK_LEN, HEAD_SIZE, HEAD_SIZE),
            device=r.device,
            dtype=torch.float32,
        )
        state_dot_a = torch.empty(
            (batch, tokens, heads, HEAD_SIZE),
            device=r.device,
            dtype=torch.float32,
        )
        torch.ops.rwkv7_clampw_v3.forward(
            *values, output, boundary, state_dot_a
        )
        ctx.save_for_backward(*values, boundary, state_dot_a)
        ctx.input_shape = (batch, tokens, channels)
        return output.view(batch, tokens, channels)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        r, w, k, v, a, b, boundary, state_dot_a = ctx.saved_tensors
        batch, tokens, channels = ctx.input_shape
        heads = channels // HEAD_SIZE
        grad_output_4d = grad_output.contiguous().view(
            batch, tokens, heads, HEAD_SIZE
        )
        gradients = tuple(torch.empty_like(tensor) for tensor in (r, w, k, v, a, b))
        torch.ops.rwkv7_clampw_v3.backward(
            r,
            w,
            k,
            v,
            a,
            b,
            grad_output_4d,
            boundary,
            state_dot_a,
            *gradients,
        )
        return tuple(gradient.view(batch, tokens, channels) for gradient in gradients)


def pretrain_recurrent_bf16(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Run the canonical train_temp clampw v3 H100 forward/backward family."""

    _validate_inputs((r, w, k, v, a, b))
    return _PretrainRecurrent.apply(r, w, k, v, a, b)


__all__ = ["pretrain_recurrent_bf16"]
