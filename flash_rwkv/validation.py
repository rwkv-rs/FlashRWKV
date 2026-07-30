# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
import math

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
    sequence_ranges: tuple[tuple[int, int], ...]
    state_indices: tuple[int, ...] | None
    packed: bool

    @property
    def num_sequences(self) -> int:
        return len(self.sequence_ranges)

    @property
    def is_packed(self) -> bool:
        return self.packed


def _metadata_values(name: str, tensor: torch.Tensor) -> tuple[int, ...]:
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if tensor.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must have dtype int32 or int64")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tuple(int(value) for value in tensor.detach().cpu().tolist())


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
    required_head_size: int | None = None,
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

    if required_head_size is not None and (
        key_size != required_head_size or value_size != required_head_size
    ):
        raise ValueError(
            f"accelerated RWKV-7 requires K = V = {required_head_size}, "
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
        if cu_seqlens.device.type != "cpu" and cu_seqlens.device != device:
            raise ValueError("cu_seqlens must be on CPU or the input device")
        offsets = _metadata_values("cu_seqlens", cu_seqlens)
        if len(offsets) < 2:
            raise ValueError("cu_seqlens must contain at least two offsets")
        if offsets[0] != 0:
            raise ValueError("cu_seqlens must start at 0")
        if offsets[-1] != sequence_length:
            raise ValueError(
                "the final cu_seqlens offset must equal the packed token count"
            )
        if any(
            end <= start
            for start, end in zip(offsets[:-1], offsets[1:], strict=True)
        ):
            raise ValueError(
                "cu_seqlens must be strictly increasing; empty sequences are unsupported"
            )
        sequence_ranges = tuple(zip(offsets[:-1], offsets[1:], strict=True))
        expected_state_rows = len(sequence_ranges)

        if state_indices is None:
            validated_indices = None
        else:
            if initial_state is None:
                raise ValueError("state_indices requires an initial state pool")
            if state_indices.device.type != "cpu" and state_indices.device != device:
                raise ValueError("state_indices must be on CPU or the input device")
            validated_indices = _metadata_values("state_indices", state_indices)
            if len(validated_indices) != expected_state_rows:
                raise ValueError(
                    "state_indices length must equal the number of packed sequences"
                )
            if len(set(validated_indices)) != len(validated_indices):
                raise ValueError("state_indices must be unique within one call")

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

        if validated_indices is None:
            if initial_state.shape[0] != expected_state_rows:
                raise ValueError(
                    f"initial_state must have {expected_state_rows} state rows"
                )
        else:
            state_pool_size = initial_state.shape[0]
            if any(index < 0 or index >= state_pool_size for index in validated_indices):
                raise ValueError("state_indices entries must be within the state pool")
    elif state_indices is not None:
        raise ValueError("state_indices requires an initial state pool")

    return ValidatedLayout(
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_heads=num_heads,
        key_size=key_size,
        value_size=value_size,
        sequence_ranges=sequence_ranges,
        state_indices=validated_indices,
        packed=cu_seqlens is not None,
    )
