# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import torch

from . import _extension
from .pretrain import _check_metadata, _check_token_inputs


class _StateTune(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        initial_state,
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        scale,
    ):
        state = initial_state.detach().contiguous().clone()
        output = torch.empty_like(v)
        boundary = torch.empty(
            (chunk_token_starts.numel(), state.shape[1], state.shape[2], state.shape[3]),
            device=state.device,
            dtype=torch.float32,
        )
        state_dot_a = torch.empty_like(r, dtype=torch.float32)
        _extension().statetune_recurrent_fp32io16_forward(
            sequence_chunk_offsets,
            chunk_token_starts,
            chunk_token_ends,
            state,
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            output,
            boundary,
            state_dot_a,
            float(scale),
        )
        ctx.save_for_backward(
            sequence_chunk_offsets,
            chunk_token_starts,
            chunk_token_ends,
            state,
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            state_dot_a,
            boundary,
        )
        ctx.scale = float(scale)
        ctx.mark_non_differentiable(boundary, state_dot_a)
        return output, state, boundary, state_dot_a

    @staticmethod
    def backward(ctx, grad_output, grad_final_state, _grad_boundary, _grad_state_dot_a):
        saved = ctx.saved_tensors
        (
            sequence_chunk_offsets,
            chunk_token_starts,
            chunk_token_ends,
            final_state,
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            state_dot_a,
            boundary,
        ) = saved
        gradients = [
            torch.empty_like(r),
            torch.empty_like(decay_logits),
            torch.empty_like(k),
            torch.empty_like(v),
            torch.empty_like(a),
            torch.empty_like(b),
            torch.empty_like(final_state),
        ]
        _extension().statetune_recurrent_fp32io16_backward(
            sequence_chunk_offsets,
            chunk_token_starts,
            chunk_token_ends,
            final_state,
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            state_dot_a,
            grad_output,
            grad_final_state,
            boundary,
            gradients[0],
            gradients[1],
            gradients[2],
            gradients[3],
            gradients[4],
            gradients[5],
            gradients[6],
            ctx.scale,
        )
        return (
            gradients[6],
            None,
            None,
            None,
            gradients[0],
            gradients[1],
            gradients[2],
            gradients[3],
            gradients[4],
            gradients[5],
            None,
        )


def statetune_recurrent_fp32io16(
    initial_state,
    sequence_chunk_offsets,
    chunk_token_starts,
    chunk_token_ends,
    r,
    decay_logits,
    k,
    v,
    a,
    b,
    *,
    scale: float = 1.0,
):
    """Run StateTune with a nonzero state and direct initial-state gradient."""

    _check_token_inputs(initial_state, r, decay_logits, k, v, a, b)
    _check_metadata(sequence_chunk_offsets, chunk_token_starts, chunk_token_ends, initial_state)
    if sequence_chunk_offsets.numel() != initial_state.shape[0] + 1:
        raise ValueError("sequence_chunk_offsets must have shape [B+1]")
    if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)):
        raise ValueError("scale must be finite")
    return _StateTune.apply(
        initial_state,
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        float(scale),
    )


__all__ = ["statetune_recurrent_fp32io16"]
