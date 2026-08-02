# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import torch

_CORE_NAMES = ("r", "log_decay", "k", "v", "a", "b")
_INTEGER_DTYPES = (torch.int32, torch.int64)


@dataclass(frozen=True)
class ValidatedLayout:
    batch_size: int
    sequence_length: int
    num_heads: int
    key_size: int
    value_size: int
    sequence_ranges: tuple[tuple[int, int], ...] | None
    state_indices: tuple[int, ...] | None
    cu_seqlens_tensor: torch.Tensor | None
    state_indices_tensor: torch.Tensor | None
    packed: bool

    @property
    def num_sequences(self) -> int:
        if self.cu_seqlens_tensor is None:
            return self.batch_size
        return self.cu_seqlens_tensor.numel() - 1

    @property
    def is_packed(self) -> bool:
        return self.packed


def _check_metadata_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if tensor.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must have dtype int32 or int64")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _metadata_values(name: str, tensor: torch.Tensor) -> tuple[int, ...]:
    _check_metadata_tensor(name, tensor)
    return tuple(int(value) for value in tensor.detach().cpu().tolist())


def validate_packed_metadata_strict(
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor | None = None,
    *,
    total_tokens: int,
    state_pool_size: int | None = None,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...] | None]:
    """Synchronously validate device metadata values for debug and tests.

    Production packed inference intentionally uses structural validation only
    so scheduler-owned CUDA metadata can flow directly into the native launch.
    Call this function outside the serving hot path when value-level checks are
    required.
    """

    offsets = _metadata_values("cu_seqlens", cu_seqlens)
    if len(offsets) < 2:
        raise ValueError("cu_seqlens must contain at least two offsets")
    if offsets[0] != 0:
        raise ValueError("cu_seqlens must start at 0")
    if offsets[-1] != total_tokens:
        raise ValueError(
            "the final cu_seqlens offset must equal the packed token count"
        )
    if any(
        end <= start
        for start, end in pairwise(offsets)
    ):
        raise ValueError(
            "cu_seqlens must be strictly increasing; empty sequences are unsupported"
        )
    sequence_ranges = tuple(pairwise(offsets))

    if state_indices is None:
        return sequence_ranges, None
    indices = _metadata_values("state_indices", state_indices)
    if len(indices) != len(sequence_ranges):
        raise ValueError(
            "state_indices length must equal the number of packed sequences"
        )
    if len(set(indices)) != len(indices):
        raise ValueError("state_indices must be unique within one call")
    if state_pool_size is None:
        raise ValueError("state_pool_size is required with state_indices")
    if any(index < 0 or index >= state_pool_size for index in indices):
        raise ValueError("state_indices entries must be within the state pool")
    return sequence_ranges, indices


def validate_rwkv7_inputs(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    state_indices: torch.Tensor | None,
    required_head_size: int | tuple[int, ...] | None = None,
    strict_packed_metadata: bool = True,
) -> ValidatedLayout:
    tensors = dict(zip(_CORE_NAMES, (r, log_decay, k, v, a, b), strict=True))
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [B, T, H, D]")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if not math.isfinite(float(scale)):
        raise ValueError("scale must be finite")

    core_shape = r.shape
    for name in ("log_decay", "k", "a", "b"):
        if tensors[name].shape != core_shape:
            raise ValueError(f"{name} must match r shape {tuple(core_shape)}")

    batch_size, sequence_length, num_heads, key_size = core_shape
    if batch_size <= 0 or sequence_length <= 0 or num_heads <= 0 or key_size <= 0:
        raise ValueError("B, T, H, and K must all be positive")
    if v.shape[:3] != core_shape[:3] or v.shape[3] <= 0:
        raise ValueError("v must have shape [B, T, H, V] matching r")
    value_size = v.shape[3]

    if required_head_size is not None:
        required_head_sizes = (
            (required_head_size,)
            if isinstance(required_head_size, int)
            else required_head_size
        )
        if key_size != value_size or key_size not in required_head_sizes:
            if len(required_head_sizes) == 1:
                requirement = f"K = V = {required_head_sizes[0]}"
            else:
                sizes = ", ".join(str(size) for size in required_head_sizes)
                requirement = f"equal K and V in {{{sizes}}}"
            raise ValueError(
                f"accelerated RWKV-7 requires {requirement}, "
                f"got K = {key_size}, V = {value_size}"
            )

    device = r.device
    dtype = r.dtype
    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")

    if cu_seqlens is None:
        if state_indices is not None:
            raise ValueError("state_indices requires cu_seqlens")
        sequence_ranges = tuple(
            (batch_index * sequence_length, (batch_index + 1) * sequence_length)
            for batch_index in range(batch_size)
        )
        expected_state_rows = batch_size
        validated_indices = None
    else:
        if batch_size != 1:
            raise ValueError("B must be 1 when cu_seqlens is provided")
        _check_metadata_tensor("cu_seqlens", cu_seqlens)
        expected_metadata_device = (
            None if strict_packed_metadata else device
        )
        if expected_metadata_device is None:
            if cu_seqlens.device.type != "cpu" and cu_seqlens.device != device:
                raise ValueError("cu_seqlens must be on CPU or the input device")
        elif cu_seqlens.device != expected_metadata_device:
            raise ValueError("hot-path cu_seqlens must be on the input device")
        if not strict_packed_metadata and cu_seqlens.dtype != torch.int32:
            raise TypeError("hot-path cu_seqlens must have dtype int32")
        expected_state_rows = cu_seqlens.numel() - 1
        if expected_state_rows <= 0:
            raise ValueError("cu_seqlens must contain at least two offsets")
        sequence_ranges = None

        if state_indices is None:
            validated_indices = None
        else:
            if initial_state is None:
                raise ValueError("state_indices requires an initial state pool")
            _check_metadata_tensor("state_indices", state_indices)
            if expected_metadata_device is None:
                if (
                    state_indices.device.type != "cpu"
                    and state_indices.device != device
                ):
                    raise ValueError(
                        "state_indices must be on CPU or the input device"
                    )
            elif state_indices.device != expected_metadata_device:
                raise ValueError("hot-path state_indices must be on the input device")
            if not strict_packed_metadata and state_indices.dtype != torch.int32:
                raise TypeError("hot-path state_indices must have dtype int32")
            if state_indices.numel() != expected_state_rows:
                raise ValueError(
                    "state_indices length must equal the number of packed sequences"
                )
            validated_indices = None

    if initial_state is not None:
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("initial_state must be a torch.Tensor")
        if initial_state.ndim != 4:
            raise ValueError("initial_state must have shape [N, H, K, V]")
        if not initial_state.is_floating_point():
            raise TypeError("initial_state must have a floating-point dtype")
        if not initial_state.is_contiguous():
            raise ValueError("initial_state must be contiguous")
        if initial_state.device != device:
            raise ValueError("initial_state must be on the input device")
        if initial_state.shape[1:] != (num_heads, key_size, value_size):
            raise ValueError(
                "initial_state must have trailing shape [H, K, V] matching the inputs"
            )

        if state_indices is None:
            if initial_state.shape[0] != expected_state_rows:
                raise ValueError(
                    f"initial_state must have {expected_state_rows} state rows"
                )
        else:
            state_pool_size = initial_state.shape[0]
            if strict_packed_metadata:
                sequence_ranges, validated_indices = validate_packed_metadata_strict(
                    cu_seqlens,
                    state_indices,
                    total_tokens=sequence_length,
                    state_pool_size=state_pool_size,
                )
    elif state_indices is not None:
        raise ValueError("state_indices requires an initial state pool")

    if cu_seqlens is not None and strict_packed_metadata and state_indices is None:
        sequence_ranges, _ = validate_packed_metadata_strict(
            cu_seqlens,
            total_tokens=sequence_length,
        )

    return ValidatedLayout(
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_heads=num_heads,
        key_size=key_size,
        value_size=value_size,
        sequence_ranges=sequence_ranges,
        state_indices=validated_indices,
        cu_seqlens_tensor=cu_seqlens,
        state_indices_tensor=state_indices,
        packed=cu_seqlens is not None,
    )
