# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import torch

from . import _extension
from .reference import rwkv7_reference
from .validation import ValidatedLayout, validate_rwkv7_inputs


_HEAD_SIZE = 64


def decay_logits_to_log_decay(decay_logits: torch.Tensor) -> torch.Tensor:
    """Convert RWKV-LM training logits to the core log-decay representation."""

    if not isinstance(decay_logits, torch.Tensor):
        raise TypeError("decay_logits must be a torch.Tensor")
    if not decay_logits.is_floating_point():
        raise TypeError("decay_logits must have a floating-point dtype")
    return -math.exp(-0.5) * torch.sigmoid(decay_logits)


def rwkv7(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    mode: str = "fp32io16",
    algorithm: str = "reference",
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run RWKV-7 with an explicit numerical mode and algorithm family.

    ``reference`` is the independent FP32 oracle. ``recurrent`` and ``chunk``
    select explicit forward-only CUDA families and fail if their extension is
    missing; no requested accelerated path silently falls back to the oracle.
    """

    if mode not in {"fp32io16", "fp16"}:
        raise ValueError(
            f"unsupported mode {mode!r}; supported modes: 'fp32io16', 'fp16'"
        )
    if algorithm == "reference":
        if mode != "fp32io16":
            raise ValueError(
                "algorithm='reference' is the FP32 oracle; "
                "use mode='fp32io16'"
            )
        output, final_state = rwkv7_reference(
            r,
            log_decay,
            k,
            v,
            a,
            b,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
        )
        return output.to(dtype=v.dtype), final_state
    if algorithm == "recurrent":
        return _rwkv7_recurrent_cuda(
            r,
            log_decay,
            k,
            v,
            a,
            b,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode=mode,
            mutate_state=False,
        )
    if algorithm == "chunk":
        return _rwkv7_chunk_cuda(
            r,
            log_decay,
            k,
            v,
            a,
            b,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode=mode,
            chunk_size=chunk_size,
        )
    raise ValueError(
        f"unsupported algorithm {algorithm!r}; "
        "supported algorithms: 'reference', 'recurrent', 'chunk'"
    )


def _cuda_metadata(
    layout: ValidatedLayout,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [layout.sequence_ranges[0][0]]
    offsets.extend(end for _, end in layout.sequence_ranges)
    if offsets[-1] > torch.iinfo(torch.int32).max:
        raise ValueError("packed token count must fit in int32")
    query_start_loc = torch.tensor(offsets, device=device, dtype=torch.int32)
    indices = (
        tuple(range(layout.num_sequences))
        if layout.state_indices is None
        else layout.state_indices
    )
    state_indices = torch.tensor(indices, device=device, dtype=torch.int32)
    return query_start_loc, state_indices


def _check_cuda_forward_only(tensors: tuple[torch.Tensor | None, ...]) -> None:
    if any(tensor is not None and tensor.requires_grad for tensor in tensors):
        raise RuntimeError(
            "the accelerated CUDA family is forward-only; "
            "use algorithm='reference' when gradients are required"
        )


def _cuda_chunk_metadata(
    layout: ValidatedLayout,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(chunk_size, bool) or chunk_size not in {16, 32, 64}:
        raise ValueError("chunk_size must be one of 16, 32, or 64")

    sequence_chunk_offsets = [0]
    chunk_token_starts: list[int] = []
    chunk_token_ends: list[int] = []
    for sequence_start, sequence_end in layout.sequence_ranges:
        if sequence_end > torch.iinfo(torch.int32).max:
            raise ValueError("chunk token offsets must fit in int32")
        for chunk_start in range(sequence_start, sequence_end, chunk_size):
            chunk_token_starts.append(chunk_start)
            chunk_token_ends.append(min(chunk_start + chunk_size, sequence_end))
        sequence_chunk_offsets.append(len(chunk_token_starts))

    indices = (
        tuple(range(layout.num_sequences))
        if layout.state_indices is None
        else layout.state_indices
    )
    return (
        torch.tensor(
            sequence_chunk_offsets,
            device=device,
            dtype=torch.int32,
        ),
        torch.tensor(
            chunk_token_starts,
            device=device,
            dtype=torch.int32,
        ),
        torch.tensor(
            chunk_token_ends,
            device=device,
            dtype=torch.int32,
        ),
        torch.tensor(indices, device=device, dtype=torch.int32),
    )


def _rwkv7_chunk_cuda(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    state_indices: torch.Tensor | None,
    mode: str,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    layout = validate_rwkv7_inputs(
        r,
        log_decay,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        required_head_size=_HEAD_SIZE,
    )
    if not r.is_cuda:
        raise ValueError("algorithm='chunk' requires CUDA inputs")
    if mode != "fp32io16":
        raise ValueError(
            "algorithm='chunk' currently supports only mode='fp32io16'"
        )
    _check_cuda_forward_only((r, log_decay, k, v, a, b, initial_state))

    if initial_state is None:
        working_state = torch.zeros(
            (
                layout.num_sequences,
                layout.num_heads,
                _HEAD_SIZE,
                _HEAD_SIZE,
            ),
            dtype=torch.float32,
            device=r.device,
        )
    else:
        working_state = initial_state.to(dtype=torch.float32).clone()

    (
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        cuda_state_indices,
    ) = _cuda_chunk_metadata(layout, chunk_size, r.device)
    num_chunks = chunk_token_starts.numel()
    workspace_shape = (
        num_chunks,
        layout.num_heads,
        _HEAD_SIZE,
        _HEAD_SIZE,
    )
    transform = torch.empty(
        workspace_shape,
        dtype=torch.float32,
        device=r.device,
    )
    bias = torch.empty_like(transform)
    boundary = torch.empty_like(transform)
    flattened_inputs = tuple(
        tensor.reshape(-1, layout.num_heads, _HEAD_SIZE)
        for tensor in (r, log_decay, k, v, a, b)
    )
    output = torch.empty_like(flattened_inputs[3])
    _extension.materialized_chunk_fp32(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        cuda_state_indices,
        working_state,
        *flattened_inputs,
        output,
        transform,
        bias,
        boundary,
        float(scale),
    )
    output = output.reshape(v.shape)
    return output, working_state if output_final_state else None


def _rwkv7_recurrent_cuda(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    state_indices: torch.Tensor | None,
    mode: str,
    mutate_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    layout = validate_rwkv7_inputs(
        r,
        log_decay,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        required_head_size=_HEAD_SIZE,
    )
    if not r.is_cuda:
        raise ValueError("algorithm='recurrent' requires CUDA inputs")
    if mode == "fp16" and r.dtype != torch.float16:
        raise TypeError("mode='fp16' requires fp16 token tensors")
    _check_cuda_forward_only((r, log_decay, k, v, a, b, initial_state))

    state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
    if initial_state is None:
        working_state = torch.zeros(
            (
                layout.num_sequences,
                layout.num_heads,
                _HEAD_SIZE,
                _HEAD_SIZE,
            ),
            dtype=state_dtype,
            device=r.device,
        )
    elif mutate_state:
        if initial_state.dtype != state_dtype:
            raise TypeError(
                f"stateful {mode} state_pool must have dtype {state_dtype}"
            )
        working_state = initial_state
    else:
        working_state = initial_state.to(dtype=state_dtype).clone()

    query_start_loc, cuda_state_indices = _cuda_metadata(layout, r.device)
    flattened_inputs = tuple(
        tensor.reshape(-1, layout.num_heads, _HEAD_SIZE)
        for tensor in (r, log_decay, k, v, a, b)
    )
    output = torch.empty_like(flattened_inputs[3])
    extension_op = (
        _extension.recurrent_fp32
        if mode == "fp32io16"
        else _extension.recurrent_fp16
    )
    extension_op(
        query_start_loc,
        cuda_state_indices,
        working_state,
        *flattened_inputs,
        output,
        float(scale),
    )
    output = output.reshape(v.shape)
    return output, working_state if output_final_state else None


def rwkv7_recurrent_stateful(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float = 1.0,
    mode: str = "fp32io16",
) -> torch.Tensor:
    """Run packed recurrent inference and update selected state rows in place."""

    output, _ = _rwkv7_recurrent_cuda(
        r,
        log_decay,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=state_pool,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        mode=mode,
        mutate_state=True,
    )
    return output


def rwkv7_from_decay_logits(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    mode: str = "fp32io16",
    algorithm: str = "reference",
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run RWKV-7 after the differentiable RWKV-LM decay-logit transform."""

    return rwkv7(
        r,
        decay_logits_to_log_decay(decay_logits),
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        mode=mode,
        algorithm=algorithm,
        chunk_size=chunk_size,
    )
