#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""Small real-kernel workload intended for Compute Sanitizer racecheck."""

from __future__ import annotations

import json

import torch

from flash_rwkv import (
    _C,
    infer_cmix_mix_fp16,
    infer_tmix_kk_a_gate_fp16,
    infer_tmix_lnx_rkvres_xg_fp16,
    infer_tmix_mix6_fp16,
    infer_tmix_vres_gate_fp16,
    pretrain_cmix_bf16,
    pretrain_head_l2wrap_ce_bf16,
    pretrain_l2wrap_ce_bf16,
    pretrain_tmix_a_gate_bf16,
    pretrain_tmix_kk_pre_bf16,
    pretrain_tmix_lnx_rkvres_xg_bf16,
    pretrain_tmix_mix6_bf16,
    pretrain_tmix_vres_gate_bf16,
    rwkv7_recurrent_stateful,
)

HOSTILE_METADATA_CASES = (
    ("malformed_start", (1, 2, 3), (0, 1)),
    ("malformed_end", (0, 1, 2), (0, 1)),
    ("nonmonotonic_overlap", (0, 2, 1, 3), (0, 1, 2)),
    ("negative_slot", (0, 1, 3), (-1, 1)),
    ("out_of_range_slot", (0, 1, 3), (0, 5)),
    ("duplicate_slot", (0, 1, 3), (2, 2)),
)


def _backward(outputs: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
    tensors = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
    sum(tensor.float().square().mean() for tensor in tensors).backward()
    torch.cuda.synchronize()


def _bf16(shape: tuple[int, ...], *, scale: float = 0.2) -> torch.Tensor:
    return (
        torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        .mul_(scale)
        .requires_grad_(True)
    )


def _fp16(shape: tuple[int, ...], *, scale: float = 0.2) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=torch.float16).mul_(scale)


def run_training() -> list[str]:
    names: list[str] = []
    shape = (2, 3, 128)

    _backward(pretrain_tmix_a_gate_bf16(_bf16((128,)), _bf16(shape)))
    names.append("pretrain_tmix_a_gate_bf16")

    _backward(pretrain_tmix_mix6_bf16(_bf16(shape), *(_bf16((128,)) for _ in range(6))))
    names.append("pretrain_tmix_mix6_bf16")

    _backward(
        pretrain_tmix_kk_pre_bf16(
            _bf16(shape),
            _bf16((128,)),
            _bf16(shape),
            _bf16((128,)),
        )
    )
    names.append("pretrain_tmix_kk_pre_bf16")

    _backward(
        pretrain_tmix_vres_gate_bf16(
            _bf16(shape),
            _bf16(shape),
            _bf16((128,)),
            _bf16(shape),
        )
    )
    names.append("pretrain_tmix_vres_gate_bf16")

    _backward(
        pretrain_tmix_lnx_rkvres_xg_bf16(
            _bf16(shape),
            _bf16(shape),
            _bf16(shape),
            _bf16(shape),
            _bf16((2, 64)),
            _bf16((128,)),
            _bf16((128,)),
            _bf16(shape),
        )
    )
    names.append("pretrain_tmix_lnx_rkvres_xg_bf16")

    _backward(
        pretrain_cmix_bf16(
            _bf16(shape),
            _bf16((128,)),
            _bf16((512, 128)),
            _bf16((128, 512)),
        )
    )
    names.append("pretrain_cmix_bf16")

    logits = _bf16((2, 3, 256))
    targets = torch.arange(6, device="cuda", dtype=torch.int64).remainder(256)
    pretrain_l2wrap_ce_bf16(logits, targets).backward()
    torch.cuda.synchronize()
    names.append("pretrain_l2wrap_ce_bf16")

    hidden = _bf16((1, 2, 64))
    weight = _bf16((65_536, 64))
    head_targets = torch.tensor([[17, 23]], device="cuda", dtype=torch.int64)
    pretrain_head_l2wrap_ce_bf16(hidden, weight, head_targets, chunk_rows=1).backward()
    torch.cuda.synchronize()
    names.append("pretrain_head_l2wrap_ce_bf16")
    return names


def run_inference() -> list[str]:
    names: list[str] = []
    shape = (2, 3, 128)
    x = _fp16(shape)
    mixes = tuple(_fp16((128,)) for _ in range(6))
    infer_tmix_mix6_fp16(x, _fp16((2, 128)), mixes)
    torch.cuda.synchronize()
    names.append("infer_tmix_mix6_fp16")

    infer_tmix_kk_a_gate_fp16(
        _fp16(shape),
        _fp16((128,)),
        _fp16((128,)),
        _fp16(shape),
        _fp16((128,)),
    )
    torch.cuda.synchronize()
    names.append("infer_tmix_kk_a_gate_fp16")

    infer_tmix_lnx_rkvres_xg_fp16(
        _fp16(shape),
        _fp16(shape),
        _fp16(shape),
        _fp16(shape),
        _fp16((128,)),
        _fp16((128,)),
        _fp16((128,)),
        _fp16(shape),
    )
    torch.cuda.synchronize()
    names.append("infer_tmix_lnx_rkvres_xg_fp16")

    infer_tmix_vres_gate_fp16(
        _fp16(shape),
        _fp16(shape),
        _fp16((128,)),
        _fp16(shape),
    )
    torch.cuda.synchronize()
    names.append("infer_tmix_vres_gate_fp16")

    infer_cmix_mix_fp16(x, _fp16((2, 128)), _fp16((128,)))
    torch.cuda.synchronize()
    names.append("infer_cmix_mix_fp16")

    packed_shape = (1, 3, 1, 64)
    packed_inputs = tuple(_fp16(packed_shape, scale=0.02) for _ in range(6))
    cu_seqlens = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([4, 1], device="cuda", dtype=torch.int32)
    for mode, state_dtype in (
        ("fp32io16", torch.float32),
        ("fp16", torch.float16),
    ):
        state_pool = torch.zeros(
            5,
            1,
            64,
            64,
            device="cuda",
            dtype=state_dtype,
        )
        rwkv7_recurrent_stateful(
            *packed_inputs,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode=mode,
        )
        torch.cuda.synchronize()
        names.append(f"rwkv7_recurrent_stateful_{mode}")
    return names


def run_hostile_packed_metadata() -> list[str]:
    names: list[str] = []
    packed_shape = (1, 3, 1, 64)
    flattened_inputs = tuple(
        tensor.reshape(3, 1, 64)
        for tensor in (_fp16(packed_shape, scale=0.02) for _ in range(6))
    )
    for mode, state_dtype, operator in (
        ("fp32io16", torch.float32, _C.recurrent_fp32),
        ("fp16", torch.float16, _C.recurrent_fp16),
    ):
        for case, offsets, slots in HOSTILE_METADATA_CASES:
            state = torch.randn(
                5,
                1,
                64,
                64,
                device="cuda",
                dtype=state_dtype,
            )
            state_before = state.clone()
            output = torch.ones_like(flattened_inputs[3])
            query_start_loc = torch.tensor(
                offsets,
                device="cuda",
                dtype=torch.int32,
            )
            state_indices = torch.tensor(
                slots,
                device="cuda",
                dtype=torch.int32,
            )
            operator(
                query_start_loc,
                state_indices,
                state,
                *flattened_inputs,
                output,
                1.0,
            )
            torch.cuda.synchronize()
            if not torch.equal(state, state_before):
                raise AssertionError(f"hostile {case}/{mode} mutated state")
            if not torch.isnan(output).all().item():
                raise AssertionError(f"hostile {case}/{mode} did not fail closed")
            names.append(f"rwkv7_recurrent_raw_hostile_{case}_{mode}")
    return names


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(20260801)
    operators = [
        *run_training(),
        *run_inference(),
        *run_hostile_packed_metadata(),
    ]
    print(
        json.dumps(
            {
                "operators": operators,
                "operator_count": len(operators),
                "device": torch.cuda.get_device_name(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
