# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from .validation import validate_rwkv7_inputs


def decay_logits_to_log_decay_reference(
    decay_logits: torch.Tensor,
    decay_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Independent producer oracle for the canonical recurrence input.

    This reference-only helper intentionally spells out the RWKV-7 producer
    formula instead of calling any product-path conversion helper.
    """

    combined = decay_logits
    if decay_bias is not None:
        if decay_logits.ndim < 2:
            raise ValueError("decay_logits must include [H,D] dimensions")
        heads, head_size = decay_logits.shape[-2:]
        if decay_bias.shape == (heads, head_size):
            shaped_bias = decay_bias
        elif decay_bias.ndim == 1 and decay_bias.numel() == heads * head_size:
            shaped_bias = decay_bias.reshape(heads, head_size)
        else:
            raise ValueError("decay_bias must have shape [H,D] or [H*D]")
        combined = decay_logits + shaped_bias
    return -0.6065306597126334 * torch.sigmoid(combined)


def _run_sequence(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state.to(dtype=torch.float32)
    outputs: list[torch.Tensor] = []

    r_f32 = r.to(dtype=torch.float32)
    log_decay_f32 = log_decay.to(dtype=torch.float32)
    k_f32 = k.to(dtype=torch.float32)
    v_f32 = v.to(dtype=torch.float32)
    a_f32 = a.to(dtype=torch.float32)
    b_f32 = b.to(dtype=torch.float32)

    for token_index in range(r.shape[0]):
        previous_state = state
        a_state = torch.einsum(
            "hk,hkv->hv", a_f32[token_index], previous_state
        )
        state = (
            torch.exp(log_decay_f32[token_index]).unsqueeze(-1) * previous_state
            + b_f32[token_index].unsqueeze(-1) * a_state.unsqueeze(-2)
            + k_f32[token_index].unsqueeze(-1) * v_f32[token_index].unsqueeze(-2)
        )
        outputs.append(
            float(scale)
            * torch.einsum("hk,hkv->hv", r_f32[token_index], state)
        )

    return torch.stack(outputs, dim=0), state


def rwkv7_reference(
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the canonical RWKV-7 recurrence in FP32 PyTorch.

    The canonical state layout is ``[K, V]``. Fixed inputs use
    ``[B, T, H, D]``. Packed inputs use ``B = 1`` and strictly increasing
    ``cu_seqlens``. When ``state_indices`` is supplied, ``initial_state`` is a
    state pool and the returned final state is an updated functional copy of
    that pool.

    This function is the numerical oracle. It deliberately does not import or
    dispatch to the CUDA extension.
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
    )

    zero_state = torch.zeros(
        layout.num_heads,
        layout.key_size,
        layout.value_size,
        device=r.device,
        dtype=torch.float32,
    )

    if cu_seqlens is None:
        outputs: list[torch.Tensor] = []
        final_states: list[torch.Tensor] = []
        for batch_index in range(layout.batch_size):
            state = (
                zero_state
                if initial_state is None
                else initial_state[batch_index]
            )
            output, final_state = _run_sequence(
                r[batch_index],
                log_decay[batch_index],
                k[batch_index],
                v[batch_index],
                a[batch_index],
                b[batch_index],
                state,
                scale,
            )
            outputs.append(output)
            final_states.append(final_state)

        output = torch.stack(outputs, dim=0)
        final_state_result = torch.stack(final_states, dim=0)
    else:
        if initial_state is None:
            sequence_states = zero_state.unsqueeze(0).expand(
                layout.num_sequences, -1, -1, -1
            )
        elif layout.state_indices is None:
            sequence_states = initial_state
        else:
            indices = torch.tensor(
                layout.state_indices, device=initial_state.device, dtype=torch.long
            )
            sequence_states = initial_state.index_select(0, indices)

        packed_outputs: list[torch.Tensor] = []
        final_states = []
        for sequence_index, (start, end) in enumerate(layout.sequence_ranges):
            output, final_state = _run_sequence(
                r[0, start:end],
                log_decay[0, start:end],
                k[0, start:end],
                v[0, start:end],
                a[0, start:end],
                b[0, start:end],
                sequence_states[sequence_index],
                scale,
            )
            packed_outputs.append(output)
            final_states.append(final_state)

        output = torch.cat(packed_outputs, dim=0).unsqueeze(0)
        sequence_final_states = torch.stack(final_states, dim=0)
        if layout.state_indices is None:
            final_state_result = sequence_final_states
        else:
            indices = torch.tensor(
                layout.state_indices, device=initial_state.device, dtype=torch.long
            )
            final_state_result = initial_state.to(dtype=torch.float32).index_copy(
                0, indices, sequence_final_states
            )

    if not output_final_state:
        return output, None
    return output, final_state_result


def rwkv7_decay_logits_reference(
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
    decay_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate raw RWKV-7 decay logits through the independent FP32 oracle."""

    return rwkv7_reference(
        r,
        decay_logits_to_log_decay_reference(decay_logits, decay_bias),
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
