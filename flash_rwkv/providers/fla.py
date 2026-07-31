# SPDX-License-Identifier: MIT

"""FLA adapters named by the canonical FlashRWKV execution contract."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def pretrain_chunk_fp32io16_forward(
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
    safe_gate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run FLA's fixed-length chunk forward/autograd implementation."""

    from fla.ops.rwkv7 import chunk_rwkv7

    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError(
            "FLA pretrain_chunk_fp32io16_forward requires FP32 initial_state"
        )
    return chunk_rwkv7(
        r,
        log_decay,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        safe_gate=safe_gate,
        chunk_size=16,
    )


def pretrain_chunk_fp32io16_backward(
    output: torch.Tensor,
    final_state: torch.Tensor,
    inputs: Sequence[torch.Tensor],
    *,
    grad_output: torch.Tensor,
    grad_final_state: torch.Tensor,
    retain_graph: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    """Run the FLA autograd backward for a prepared forward graph."""

    if len(inputs) != 7:
        raise ValueError("FLA backward requires six token inputs and initial_state")
    return torch.autograd.grad(
        outputs=(output, final_state),
        inputs=tuple(inputs),
        grad_outputs=(grad_output, grad_final_state),
        retain_graph=retain_graph,
        allow_unused=False,
    )


def infer_recurrent_fp32io16_forward_varlen(
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
    """Run FLA's packed fused-recurrent forward with FP32 state."""

    from fla.ops.rwkv7 import fused_recurrent_rwkv7

    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError(
            "FLA infer_recurrent_fp32io16_forward_varlen requires "
            "FP32 initial_state"
        )
    return fused_recurrent_rwkv7(
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
    )
