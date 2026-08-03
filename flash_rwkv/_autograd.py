# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Literal

import torch

from . import _extension


def pretrain_recurrent_fp32io16_autograd(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the RWKV-LM-derived fixed-length recurrent autograd boundary."""

    return _recurrent_fp32io16_autograd(
        "pretrain",
        "log_decay",
        r,
        log_decay,
        k,
        v,
        a,
        b,
        initial_state=initial_state,
        sequence_chunk_offsets=sequence_chunk_offsets,
        chunk_token_starts=chunk_token_starts,
        chunk_token_ends=chunk_token_ends,
        scale=scale,
    )


def pretrain_recurrent_fp32io16_from_decay_logits_autograd(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run recurrent autograd with the raw decay transform fused natively."""

    return _recurrent_fp32io16_autograd(
        "pretrain",
        "decay_logits",
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        initial_state=initial_state,
        sequence_chunk_offsets=sequence_chunk_offsets,
        chunk_token_starts=chunk_token_starts,
        chunk_token_ends=chunk_token_ends,
        scale=scale,
    )


def statetune_recurrent_fp32io16_autograd(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the StateTune nonzero-state recurrent autograd boundary."""

    return _recurrent_fp32io16_autograd(
        "statetune",
        "log_decay",
        r,
        log_decay,
        k,
        v,
        a,
        b,
        initial_state=initial_state,
        sequence_chunk_offsets=sequence_chunk_offsets,
        chunk_token_starts=chunk_token_starts,
        chunk_token_ends=chunk_token_ends,
        scale=scale,
    )


def statetune_recurrent_fp32io16_from_decay_logits_autograd(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run StateTune autograd with the raw decay transform fused natively."""

    return _recurrent_fp32io16_autograd(
        "statetune",
        "decay_logits",
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        initial_state=initial_state,
        sequence_chunk_offsets=sequence_chunk_offsets,
        chunk_token_starts=chunk_token_starts,
        chunk_token_ends=chunk_token_ends,
        scale=scale,
    )


def _recurrent_fp32io16_autograd(
    workload: Literal["pretrain", "statetune"],
    decay_input: Literal["log_decay", "decay_logits"],
    r: torch.Tensor,
    decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _RecurrentFp32io16Function.apply(
        r,
        decay,
        k,
        v,
        a,
        b,
        initial_state,
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        scale,
        workload,
        decay_input,
    )


class _RecurrentFp32io16Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        r: torch.Tensor,
        decay: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        initial_state: torch.Tensor | None,
        sequence_chunk_offsets: torch.Tensor,
        chunk_token_starts: torch.Tensor,
        chunk_token_ends: torch.Tensor,
        scale: float,
        workload: Literal["pretrain", "statetune"],
        decay_input: Literal["log_decay", "decay_logits"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, num_heads, head_size = r.shape
        flattened = tuple(
            tensor.reshape(-1, num_heads, head_size)
            for tensor in (r, decay, k, v, a, b)
        )
        working_state = (
            torch.zeros(
                batch_size,
                num_heads,
                head_size,
                head_size,
                dtype=torch.float32,
                device=r.device,
            )
            if initial_state is None
            else initial_state.clone()
        )
        workspace_shape = (
            chunk_token_starts.numel(),
            num_heads,
            head_size,
            head_size,
        )
        boundary = torch.empty(
            workspace_shape,
            dtype=torch.float32,
            device=r.device,
        )
        state_dot_a = torch.empty(
            flattened[0].shape,
            dtype=torch.float32,
            device=r.device,
        )
        output = torch.empty_like(flattened[3])

        if decay_input == "decay_logits":
            forward = (
                _extension.statetune_recurrent_fp32io16_from_decay_logits_forward
                if workload == "statetune"
                else _extension.pretrain_recurrent_fp32io16_from_decay_logits_forward
            )
        else:
            forward = (
                _extension.statetune_recurrent_fp32io16_forward
                if workload == "statetune"
                else _extension.pretrain_recurrent_fp32io16_forward
            )
        forward(
            sequence_chunk_offsets,
            chunk_token_starts,
            chunk_token_ends,
            working_state,
            *flattened,
            output,
            boundary,
            state_dot_a,
            float(scale),
        )

        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            sequence_chunk_offsets,
            chunk_token_starts,
            chunk_token_ends,
            *flattened,
            boundary,
            state_dot_a,
            working_state,
        )
        ctx.input_shape = tuple(r.shape)
        ctx.scale = float(scale)
        ctx.workload = workload
        ctx.decay_input = decay_input
        return output.reshape(v.shape), working_state

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor | None,
        grad_final_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        (
            sequence_chunk_offsets,
            chunk_token_starts,
            chunk_token_ends,
            r,
            decay,
            k,
            v,
            a,
            b,
            boundary,
            state_dot_a,
            final_state,
        ) = ctx.saved_tensors
        needs = ctx.needs_input_grad
        inputs = (r, decay, k, v, a, b)
        gradients = tuple(
            torch.empty_like(tensor) if needs[index] else None
            for index, tensor in enumerate(inputs)
        )
        grad_initial_state = (
            torch.empty_like(final_state) if needs[6] else None
        )
        flattened_grad_output = (
            None
            if grad_output is None
            else grad_output.contiguous().reshape_as(v)
        )
        contiguous_grad_final_state = (
            None
            if grad_final_state is None
            else grad_final_state.contiguous()
        )

        if ctx.decay_input == "decay_logits":
            backward = (
                _extension.statetune_recurrent_fp32io16_from_decay_logits_backward
                if ctx.workload == "statetune"
                else _extension.pretrain_recurrent_fp32io16_from_decay_logits_backward
            )
        else:
            backward = (
                _extension.statetune_recurrent_fp32io16_backward
                if ctx.workload == "statetune"
                else _extension.pretrain_recurrent_fp32io16_backward
            )
        backward(
            sequence_chunk_offsets,
            chunk_token_starts,
            chunk_token_ends,
            final_state,
            r,
            decay,
            k,
            v,
            a,
            b,
            state_dot_a,
            flattened_grad_output,
            contiguous_grad_final_state,
            boundary,
            *gradients,
            grad_initial_state,
            ctx.scale,
        )

        input_shape = ctx.input_shape
        shaped_gradients = tuple(
            None if gradient is None else gradient.reshape(input_shape)
            for gradient in gradients
        )
        return (
            *shaped_gradients,
            grad_initial_state,
            None,
            None,
            None,
            None,
            None,
            None,
        )
