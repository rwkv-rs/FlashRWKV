# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from . import _extension
from ._autograd import (
    pretrain_recurrent_fp32io16_autograd,
    pretrain_recurrent_fp32io16_from_decay_logits_autograd,
    statetune_recurrent_fp32io16_autograd,
    statetune_recurrent_fp32io16_from_decay_logits_autograd,
)
from .config import (
    ChunkConfig,
    chunk_tuning_key,
    select_algorithm,
    select_chunk_config,
)
from .reference import rwkv7_reference
from .validation import (
    ValidatedLayout,
    validate_rwkv7_inputs,
)

_HEAD_SIZE = 64
RWKV7_RECURRENT_HEAD_SIZES = (64, 128, 256)


def prepare_recurrent_metadata(
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    total_tokens: int,
    state_pool_size: int,
) -> object:
    """Validate packed metadata once for same-stream multi-layer reuse.

    The returned object is an opaque native ticket bound to the exact metadata
    tensor identities, versions, layout, device, token/slot bounds, and current
    CUDA stream. Passing it to recurrent calls removes their per-layer metadata
    validation launch without weakening fail-closed validation.
    """

    return _extension.prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=total_tokens,
        state_pool_size=state_pool_size,
    )


def _pretrain_recurrent_fp32io16_from_log_decay(
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the RWKV-LM-derived fixed-length recurrent training operator."""

    layout = validate_rwkv7_inputs(
        r,
        log_decay,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=None,
        state_indices=None,
        required_head_size=RWKV7_RECURRENT_HEAD_SIZES,
    )
    if not r.is_cuda:
        raise ValueError(
            "pretrain_recurrent_fp32io16_from_log_decay requires CUDA inputs"
        )
    if r.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError(
            "pretrain_recurrent_fp32io16_from_log_decay requires fp16 or bf16 "
            "token tensors"
        )
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError(
            "pretrain_recurrent_fp32io16_from_log_decay requires an FP32 "
            "initial_state"
        )
    (
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        _,
    ) = _cuda_chunk_metadata(layout, 16, r.device)
    output, final_state = pretrain_recurrent_fp32io16_autograd(
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
        scale=float(scale),
    )
    return output, final_state if output_final_state else None


def _canonical_pretrain_recurrent_fp32io16(
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Explicit canonical-log-decay compatibility training boundary."""

    return _pretrain_recurrent_fp32io16_from_log_decay(
        r,
        log_decay,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )


def pretrain_recurrent_fp32io16_from_decay_logits(
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run fixed-length recurrent training with the raw transform fused."""

    layout = validate_rwkv7_inputs(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=None,
        state_indices=None,
        required_head_size=RWKV7_RECURRENT_HEAD_SIZES,
    )
    if not r.is_cuda:
        raise ValueError(
            "pretrain_recurrent_fp32io16_from_decay_logits requires CUDA inputs"
        )
    if r.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError(
            "pretrain_recurrent_fp32io16_from_decay_logits requires fp16 or "
            "bf16 token tensors"
        )
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError(
            "pretrain_recurrent_fp32io16_from_decay_logits requires an FP32 "
            "initial_state"
        )
    (
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        _,
    ) = _cuda_chunk_metadata(layout, 16, r.device)
    output, final_state = pretrain_recurrent_fp32io16_from_decay_logits_autograd(
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
        scale=float(scale),
    )
    return output, final_state if output_final_state else None


def _canonical_statetune_recurrent_fp32io16(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run StateTune recurrence with a nonzero FP32 initial state."""

    layout = validate_rwkv7_inputs(
        r,
        log_decay,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=None,
        state_indices=None,
        required_head_size=RWKV7_RECURRENT_HEAD_SIZES,
    )
    if not r.is_cuda:
        raise ValueError("statetune_recurrent_fp32io16_forward requires CUDA inputs")
    if r.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError(
            "statetune_recurrent_fp32io16_forward requires fp16 or bf16 "
            "token tensors"
        )
    if initial_state.dtype != torch.float32:
        raise TypeError(
            "statetune_recurrent_fp32io16_forward requires an FP32 initial_state"
        )
    (
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        _,
    ) = _cuda_chunk_metadata(layout, 16, r.device)
    output, final_state = statetune_recurrent_fp32io16_autograd(
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
        scale=float(scale),
    )
    return output, final_state if output_final_state else None


def pretrain_recurrent_fp32io16_forward(
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Standard fixed-length training API with the raw transform fused."""

    return pretrain_recurrent_fp32io16_from_decay_logits(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )


def pretrain_recurrent_fp32io16(
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Standard fixed-length training API with the raw transform fused."""

    return pretrain_recurrent_fp32io16_from_decay_logits(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )


def statetune_recurrent_fp32io16_forward(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Standard StateTune API with the raw decay transform fused natively."""

    layout = validate_rwkv7_inputs(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=None,
        state_indices=None,
        required_head_size=RWKV7_RECURRENT_HEAD_SIZES,
    )
    if not r.is_cuda:
        raise ValueError("statetune_recurrent_fp32io16_forward requires CUDA inputs")
    if r.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError(
            "statetune_recurrent_fp32io16_forward requires fp16 or bf16 "
            "token tensors"
        )
    if initial_state.dtype != torch.float32:
        raise TypeError(
            "statetune_recurrent_fp32io16_forward requires an FP32 initial_state"
        )
    (
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        _,
    ) = _cuda_chunk_metadata(layout, 16, r.device)
    output, final_state = statetune_recurrent_fp32io16_from_decay_logits_autograd(
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
        scale=float(scale),
    )
    return output, final_state if output_final_state else None


def _canonical_rwkv7(
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
    chunk_size: int | None = None,
    chunk_config: ChunkConfig | None = None,
    validated_metadata: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Explicit canonical-log-decay compatibility/oracle API.

    ``reference`` is the independent FP32 oracle. ``recurrent`` is an explicit
    forward-only CUDA family. ``chunk`` supports fixed-length autograd and
    forward-only packed execution. No requested accelerated path silently
    falls back to the oracle.
    """

    if mode not in {"fp32io16", "fp16"}:
        raise ValueError(
            f"unsupported mode {mode!r}; supported modes: 'fp32io16', 'fp16'"
        )
    if algorithm == "auto":
        algorithm = "reference"
    if algorithm == "reference":
        if validated_metadata is not None:
            raise ValueError(
                "validated_metadata is valid only for recurrent execution"
            )
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
            from_decay_logits=False,
            decay_bias=None,
            elapsed_t=None,
            validated_metadata=validated_metadata,
        )
    if algorithm == "chunk":
        if validated_metadata is not None:
            raise ValueError(
                "validated_metadata is valid only for recurrent execution"
            )
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
            chunk_config=chunk_config,
            from_decay_logits=False,
            decay_bias=None,
        )
    raise ValueError(
        f"unsupported algorithm {algorithm!r}; "
        "supported algorithms: 'reference', 'recurrent', 'chunk', 'auto'"
    )


def rwkv7(
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
    algorithm: str = "auto",
    chunk_size: int | None = None,
    chunk_config: ChunkConfig | None = None,
    decay_bias: torch.Tensor | None = None,
    elapsed_t: torch.Tensor | None = None,
    validated_metadata: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run RWKV-7 from raw decay logits.

    Accelerated recurrent and chunk paths fuse the raw decay transform in
    their existing native kernels. ``decay_bias`` keeps a producer delta and
    its channel bias separate so inference does not need an add kernel.
    ``elapsed_t`` enables the vLLM/Albatross FP16 state dither and is rejected
    by FP32-state and chunk modes.
    """

    if mode not in {"fp32io16", "fp16"}:
        raise ValueError(
            f"unsupported mode {mode!r}; supported modes: 'fp32io16', 'fp16'"
        )
    if algorithm == "auto":
        algorithm = select_algorithm(
            algorithm,
            mode=mode,
            max_sequence_length=0,
        )
    if algorithm == "reference":
        raise ValueError(
            "algorithm='reference' is not a product decay_logits path; "
            "use flash_rwkv.reference.rwkv7_reference with an independent "
            "canonical-log-decay oracle"
        )
    if algorithm == "recurrent":
        return _rwkv7_recurrent_cuda(
            r,
            decay_logits,
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
            from_decay_logits=True,
            decay_bias=decay_bias,
            elapsed_t=elapsed_t,
            validated_metadata=validated_metadata,
        )
    if algorithm == "chunk":
        if validated_metadata is not None:
            raise ValueError(
                "validated_metadata is valid only for recurrent execution"
            )
        if elapsed_t is not None:
            raise ValueError(
                "elapsed_t dithering is supported only by mode='fp16' "
                "recurrent execution"
            )
        return _rwkv7_chunk_cuda(
            r,
            decay_logits,
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
            chunk_config=chunk_config,
            from_decay_logits=True,
            decay_bias=decay_bias,
        )
    raise ValueError(
        f"unsupported algorithm {algorithm!r}; "
        "supported algorithms: 'recurrent', 'chunk', 'auto'"
    )


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
    algorithm: str = "auto",
    chunk_size: int | None = None,
    chunk_config: ChunkConfig | None = None,
    decay_bias: torch.Tensor | None = None,
    elapsed_t: torch.Tensor | None = None,
    validated_metadata: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compatibility alias for the standard raw-decay ``rwkv7`` API."""

    return rwkv7(
        r,
        decay_logits,
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
        chunk_config=chunk_config,
        decay_bias=decay_bias,
        elapsed_t=elapsed_t,
        validated_metadata=validated_metadata,
    )


def _cuda_metadata(
    layout: ValidatedLayout,
    device: torch.device,
    cu_seqlens: torch.Tensor | None,
    state_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is None:
        if layout.sequence_length > torch.iinfo(torch.int32).max:
            raise ValueError("sequence length must fit in int32")
        query_start_loc = torch.arange(
            0,
            (layout.batch_size + 1) * layout.sequence_length,
            layout.sequence_length,
            device=device,
            dtype=torch.int32,
        )
    else:
        query_start_loc = cu_seqlens
    cuda_state_indices = (
        torch.arange(layout.num_sequences, device=device, dtype=torch.int32)
        if state_indices is None
        else state_indices
    )
    return query_start_loc, cuda_state_indices


def _check_cuda_forward_only(tensors: tuple[torch.Tensor | None, ...]) -> None:
    if any(tensor is not None and tensor.requires_grad for tensor in tensors):
        raise RuntimeError(
            "the accelerated CUDA family is forward-only; "
            "use the fixed-length training API when gradients are required"
        )


def _cuda_chunk_metadata(
    layout: ValidatedLayout,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(chunk_size, bool) or chunk_size not in {16, 32, 64}:
        raise ValueError("chunk_size must be one of 16, 32, or 64")

    if layout.sequence_ranges is None:
        raise ValueError("chunk metadata requires strict packed validation")
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
    decay: torch.Tensor,
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
    chunk_size: int | None,
    chunk_config: ChunkConfig | None,
    from_decay_logits: bool,
    decay_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    layout = validate_rwkv7_inputs(
        r,
        decay,
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
    requires_grad = any(
        tensor is not None and tensor.requires_grad
        for tensor in (r, decay, k, v, a, b, initial_state, decay_bias)
    )
    if requires_grad:
        raise RuntimeError(
            "algorithm='chunk' autograd is unsupported; use "
            "algorithm='recurrent' for training"
        )

    tuning_key = chunk_tuning_key(
        r,
        mode=mode,
        packed=layout.packed,
        max_sequence_length=max(
            end - start for start, end in layout.sequence_ranges
        ),
    )
    selected_config = select_chunk_config(
        tuning_key,
        chunk_size=chunk_size,
        config=chunk_config,
    )
    config = selected_config.config
    (
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        cuda_state_indices,
    ) = _cuda_chunk_metadata(layout, config.chunk_size, r.device)
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
        for tensor in (r, decay, k, v, a, b)
    )
    output = torch.empty_like(flattened_inputs[3])
    extension_args = (
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
        config.build_warps,
        config.stages,
        config.state_tile,
        float(scale),
    )
    if from_decay_logits:
        _extension.materialized_chunk_fp32_from_decay_logits(
            *extension_args,
            decay_bias=decay_bias,
        )
    else:
        _extension.materialized_chunk_fp32(*extension_args)
    output = output.reshape(v.shape)
    return output, working_state if output_final_state else None


def _canonical_rl_infctx_chunk_fp32io16_factor_recompute(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the low-workspace RL-infctx factor-recompute chunk kernel.

    This forward-only FP32-state operator exposes the canonical registry
    strategy without leaking its private workspace or native ``_C`` ABI.
    Fixed and packed inputs share the ordinary RWKV7 state contract.
    """

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
        raise ValueError(
            "rl_infctx_chunk_fp32io16_factor_recompute requires CUDA inputs"
        )
    if r.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError(
            "rl_infctx_chunk_fp32io16_factor_recompute requires fp16 or bf16 "
            "token tensors"
        )
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError(
            "rl_infctx_chunk_fp32io16_factor_recompute requires an FP32 "
            "initial_state"
        )
    _check_cuda_forward_only((r, log_decay, k, v, a, b, initial_state))

    (
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        cuda_state_indices,
    ) = _cuda_chunk_metadata(layout, chunk_size, r.device)
    flattened_inputs = tuple(
        tensor.reshape(-1, layout.num_heads, _HEAD_SIZE)
        for tensor in (r, log_decay, k, v, a, b)
    )
    working_state = (
        torch.zeros(
            (
                layout.num_sequences,
                layout.num_heads,
                _HEAD_SIZE,
                _HEAD_SIZE,
            ),
            dtype=torch.float32,
            device=r.device,
        )
        if initial_state is None
        else initial_state.clone()
    )
    output = torch.empty_like(flattened_inputs[3])
    boundary = torch.empty(
        (
            chunk_token_starts.numel(),
            layout.num_heads,
            _HEAD_SIZE,
            _HEAD_SIZE,
        ),
        dtype=torch.float32,
        device=r.device,
    )
    _extension.recompute_chunk_fp32(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        cuda_state_indices,
        working_state,
        *flattened_inputs,
        output,
        boundary,
        float(scale),
    )
    return (
        output.reshape(v.shape),
        working_state if output_final_state else None,
    )


def rl_infctx_chunk_fp32io16_factor_recompute(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    chunk_size: int = 16,
    decay_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the raw-decay low-workspace factor-recompute chunk kernel."""

    layout = validate_rwkv7_inputs(
        r,
        decay_logits,
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
        raise ValueError(
            "rl_infctx_chunk_fp32io16_factor_recompute requires CUDA inputs"
        )
    if r.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError(
            "rl_infctx_chunk_fp32io16_factor_recompute requires fp16 or bf16 "
            "token tensors"
        )
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError(
            "rl_infctx_chunk_fp32io16_factor_recompute requires an FP32 "
            "initial_state"
        )
    _check_cuda_forward_only(
        (r, decay_logits, k, v, a, b, initial_state, decay_bias)
    )

    (
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        cuda_state_indices,
    ) = _cuda_chunk_metadata(layout, chunk_size, r.device)
    flattened_inputs = tuple(
        tensor.reshape(-1, layout.num_heads, _HEAD_SIZE)
        for tensor in (r, decay_logits, k, v, a, b)
    )
    working_state = (
        torch.zeros(
            (
                layout.num_sequences,
                layout.num_heads,
                _HEAD_SIZE,
                _HEAD_SIZE,
            ),
            dtype=torch.float32,
            device=r.device,
        )
        if initial_state is None
        else initial_state.clone()
    )
    output = torch.empty_like(flattened_inputs[3])
    boundary = torch.empty(
        (
            chunk_token_starts.numel(),
            layout.num_heads,
            _HEAD_SIZE,
            _HEAD_SIZE,
        ),
        dtype=torch.float32,
        device=r.device,
    )
    _extension.recompute_chunk_fp32_from_decay_logits(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        cuda_state_indices,
        working_state,
        *flattened_inputs,
        output,
        boundary,
        float(scale),
        decay_bias=decay_bias,
    )
    return (
        output.reshape(v.shape),
        working_state if output_final_state else None,
    )


def _rwkv7_recurrent_cuda(
    r: torch.Tensor,
    decay: torch.Tensor,
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
    from_decay_logits: bool,
    decay_bias: torch.Tensor | None,
    elapsed_t: torch.Tensor | None,
    validated_metadata: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if mode not in {"fp32io16", "fp16"}:
        raise ValueError(
            f"unsupported mode {mode!r}; supported modes: 'fp32io16', 'fp16'"
        )
    layout = validate_rwkv7_inputs(
        r,
        decay,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        required_head_size=RWKV7_RECURRENT_HEAD_SIZES,
        strict_packed_metadata=False,
    )
    if not r.is_cuda:
        raise ValueError("algorithm='recurrent' requires CUDA inputs")
    if mode == "fp16" and r.dtype != torch.float16:
        raise TypeError("mode='fp16' requires fp16 token tensors")
    if not from_decay_logits and (
        decay_bias is not None or elapsed_t is not None
    ):
        raise ValueError(
            "decay_bias and elapsed_t are valid only for raw decay_logits"
        )
    if mode != "fp16" and elapsed_t is not None:
        raise ValueError(
            "elapsed_t dithering is valid only for mode='fp16'"
        )
    _check_cuda_forward_only(
        (r, decay, k, v, a, b, initial_state, decay_bias)
    )
    if validated_metadata is not None and (
        cu_seqlens is None or state_indices is None
    ):
        raise ValueError(
            "validated_metadata requires explicit cu_seqlens and state_indices"
        )

    state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
    head_size = layout.key_size
    if initial_state is None:
        working_state = torch.zeros(
            (
                layout.num_sequences,
                layout.num_heads,
                head_size,
                head_size,
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

    query_start_loc, cuda_state_indices = _cuda_metadata(
        layout,
        r.device,
        cu_seqlens,
        state_indices,
    )
    flattened_inputs = tuple(
        tensor.reshape(-1, layout.num_heads, head_size)
        for tensor in (r, decay, k, v, a, b)
    )
    output = torch.empty_like(flattened_inputs[3])
    if from_decay_logits:
        extension_op = (
            _extension.recurrent_fp32_from_decay_logits
            if mode == "fp32io16"
            else _extension.recurrent_fp16_from_decay_logits
        )
        extension_op(
            query_start_loc,
            cuda_state_indices,
            working_state,
            *flattened_inputs,
            output,
            float(scale),
            decay_bias=decay_bias,
            elapsed_t=elapsed_t,
            validated_metadata=validated_metadata,
        )
    else:
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
            validated_metadata=validated_metadata,
        )
    output = output.reshape(v.shape)
    return output, working_state if output_final_state else None


def rwkv7_recurrent(
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
    decay_bias: torch.Tensor | None = None,
    elapsed_t: torch.Tensor | None = None,
    validated_metadata: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Standard functional recurrent API from raw decay logits."""

    return _rwkv7_recurrent_cuda(
        r,
        decay_logits,
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
        from_decay_logits=True,
        decay_bias=decay_bias,
        elapsed_t=elapsed_t,
        validated_metadata=validated_metadata,
    )


def rwkv7_chunk(
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
    chunk_size: int | None = None,
    chunk_config: ChunkConfig | None = None,
    decay_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Standard FP32-state chunk API with the raw transform fused natively."""

    return rwkv7(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        mode="fp32io16",
        algorithm="chunk",
        chunk_size=chunk_size,
        chunk_config=chunk_config,
        decay_bias=decay_bias,
    )


def _canonical_rwkv7_recurrent(
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
    validated_metadata: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Explicit canonical-log-decay functional compatibility boundary."""

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
        from_decay_logits=False,
        decay_bias=None,
        elapsed_t=None,
        validated_metadata=validated_metadata,
    )


def _canonical_rwkv7_recurrent_stateful(
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
    validated_metadata: object | None = None,
) -> torch.Tensor:
    """Run packed recurrent inference and update selected state rows in place.

    ``cu_seqlens`` and ``state_indices`` must be contiguous CUDA int32 tensors.
    The serving path validates their Python-visible structure and passes the same
    tensor objects to the native operator without synchronizing values to the
    host. The native boundary checks endpoints, monotonic ranges, slot bounds,
    and slot uniqueness on the current CUDA stream before consuming metadata.
    Use ``validate_packed_metadata_strict`` only for synchronous debug errors
    outside the hot path.
    """

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
        from_decay_logits=False,
        decay_bias=None,
        elapsed_t=None,
        validated_metadata=validated_metadata,
    )
    return output


def rwkv7_recurrent_stateful(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
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
    decay_bias: torch.Tensor | None = None,
    elapsed_t: torch.Tensor | None = None,
    validated_metadata: object | None = None,
) -> torch.Tensor:
    """Standard packed stateful API from raw decay logits."""

    output, _ = _rwkv7_recurrent_cuda(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        initial_state=state_pool,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        scale=scale,
        mode=mode,
        mutate_state=True,
        from_decay_logits=True,
        decay_bias=decay_bias,
        elapsed_t=elapsed_t,
        validated_metadata=validated_metadata,
    )
    return output


def _canonical_infer_recurrent_fp32io16_forward_varlen(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor | None = None,
    scale: float = 1.0,
    output_final_state: bool = True,
    validated_metadata: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run vllm-rwkv-derived packed inference with FP32 canonical state."""

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
        mode="fp32io16",
        mutate_state=False,
        from_decay_logits=False,
        decay_bias=None,
        elapsed_t=None,
        validated_metadata=validated_metadata,
    )


def _canonical_infer_recurrent_fp16_forward_varlen(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor | None = None,
    scale: float = 1.0,
    output_final_state: bool = True,
    validated_metadata: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run vllm-rwkv-derived packed inference with FP16 canonical state."""

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
        mode="fp16",
        mutate_state=False,
        from_decay_logits=False,
        decay_bias=None,
        elapsed_t=None,
        validated_metadata=validated_metadata,
    )


def infer_recurrent_fp32io16_forward_varlen(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor | None = None,
    scale: float = 1.0,
    output_final_state: bool = True,
    decay_bias: torch.Tensor | None = None,
    validated_metadata: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Standard packed FP32-state inference API from raw decay logits."""

    return _rwkv7_recurrent_cuda(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        scale=scale,
        output_final_state=output_final_state,
        mode="fp32io16",
        mutate_state=False,
        from_decay_logits=True,
        decay_bias=decay_bias,
        elapsed_t=None,
        validated_metadata=validated_metadata,
    )


def infer_recurrent_fp16_forward_varlen(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor | None = None,
    scale: float = 1.0,
    output_final_state: bool = True,
    decay_bias: torch.Tensor | None = None,
    elapsed_t: torch.Tensor | None = None,
    validated_metadata: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Standard packed FP16-state inference API from raw decay logits."""

    return _rwkv7_recurrent_cuda(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        scale=scale,
        output_final_state=output_final_state,
        mode="fp16",
        mutate_state=False,
        from_decay_logits=True,
        decay_bias=decay_bias,
        elapsed_t=elapsed_t,
        validated_metadata=validated_metadata,
    )


def _infer_chunk_bf16_forward_impl(
    r: torch.Tensor,
    decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    scale: float,
    output_final_state: bool,
    from_decay_logits: bool,
    decay_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    layout = validate_rwkv7_inputs(
        r,
        decay,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        state_indices=None,
        required_head_size=_HEAD_SIZE,
    )
    if not r.is_cuda:
        raise ValueError("infer_chunk_bf16_forward requires CUDA inputs")
    if r.dtype != torch.bfloat16:
        raise TypeError("infer_chunk_bf16_forward token tensors must be bf16")
    if initial_state is not None and initial_state.dtype != torch.bfloat16:
        raise TypeError("infer_chunk_bf16_forward initial_state must be bf16")
    _check_cuda_forward_only(
        (r, decay, k, v, a, b, initial_state, decay_bias)
    )

    (
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        _,
    ) = _cuda_chunk_metadata(layout, 16, r.device)
    flattened = tuple(
        tensor.reshape(-1, layout.num_heads, _HEAD_SIZE)
        for tensor in (r, decay, k, v, a, b)
    )
    state = (
        torch.zeros(
            layout.num_sequences,
            layout.num_heads,
            _HEAD_SIZE,
            _HEAD_SIZE,
            dtype=torch.bfloat16,
            device=r.device,
        )
        if initial_state is None
        else initial_state.clone()
    )
    chunk_workspace_shape = (
        chunk_token_starts.numel(),
        layout.num_heads,
        _HEAD_SIZE,
        _HEAD_SIZE,
    )
    chunk_transform = torch.empty(
        chunk_workspace_shape,
        dtype=torch.float32,
        device=r.device,
    )
    chunk_bias = torch.empty_like(chunk_transform)
    token_transform = torch.empty(
        flattened[0].shape,
        dtype=torch.float32,
        device=r.device,
    )
    token_bias = torch.empty_like(token_transform)
    output = torch.empty_like(flattened[3])

    if from_decay_logits:
        _extension.infer_chunk_bf16_forward_k1_prepare_from_decay_logits(
            chunk_token_starts,
            chunk_token_ends,
            *flattened,
            chunk_transform,
            chunk_bias,
            token_transform,
            token_bias,
            float(scale),
            decay_bias=decay_bias,
        )
    else:
        _extension.infer_chunk_bf16_forward_k1_prepare(
            chunk_token_starts,
            chunk_token_ends,
            *flattened,
            chunk_transform,
            chunk_bias,
            token_transform,
            token_bias,
            float(scale),
        )
    _extension.infer_chunk_bf16_forward_k2_recurrence(
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        state,
        output,
        chunk_transform,
        chunk_bias,
        token_transform,
        token_bias,
    )
    return (
        output.reshape(v.shape),
        state if output_final_state else None,
    )


def _canonical_infer_chunk_bf16_forward(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float = 1.0,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the KDA-derived K1/K2 fixed-length BF16 chunk operator."""

    return _infer_chunk_bf16_forward_impl(
        r,
        log_decay,
        k,
        v,
        a,
        b,
        initial_state=initial_state,
        cu_seqlens=None,
        scale=scale,
        output_final_state=output_final_state,
        from_decay_logits=False,
        decay_bias=None,
    )


def _canonical_infer_chunk_bf16_forward_varlen(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    scale: float = 1.0,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the KDA-derived K1/K2 packed-varlen BF16 chunk operator."""

    return _infer_chunk_bf16_forward_impl(
        r,
        log_decay,
        k,
        v,
        a,
        b,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        output_final_state=output_final_state,
        from_decay_logits=False,
        decay_bias=None,
    )


def infer_chunk_bf16_forward(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float = 1.0,
    output_final_state: bool = True,
    decay_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run fixed-length BF16 chunk inference from raw decay logits."""

    return _infer_chunk_bf16_forward_impl(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        initial_state=initial_state,
        cu_seqlens=None,
        scale=scale,
        output_final_state=output_final_state,
        from_decay_logits=True,
        decay_bias=decay_bias,
    )


def infer_chunk_bf16_forward_varlen(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    scale: float = 1.0,
    output_final_state: bool = True,
    decay_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run packed-varlen BF16 chunk inference from raw decay logits."""

    return _infer_chunk_bf16_forward_impl(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        output_final_state=output_final_state,
        from_decay_logits=True,
        decay_bias=decay_bias,
    )
