# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from flash_rwkv.registry import (
    KERNEL_SPECS,
    REFERENCE_ORACLES,
    WRAPPER_KERNELS,
    KernelSpec,
    get_kernel_spec,
)


EXPECTED_IDENTITIES = {
    ("rwkv-lm", "pretrain_recurrent_fp32io16_forward"),
    ("rwkv-lm", "pretrain_recurrent_fp32io16_backward"),
    ("vllm-rwkv", "infer_recurrent_fp32io16_forward_varlen"),
    ("vllm-rwkv", "infer_recurrent_fp16_forward_varlen"),
    ("flashkda-derived", "infer_chunk_bf16_forward"),
    ("flashkda-derived", "infer_chunk_bf16_forward_varlen"),
    ("fla", "pretrain_chunk_fp32io16_forward"),
    ("fla", "pretrain_chunk_fp32io16_backward"),
    ("fla", "infer_recurrent_fp32io16_forward_varlen"),
}


def test_registry_contains_exact_provider_specific_identities() -> None:
    assert {spec.identity for spec in KERNEL_SPECS} == EXPECTED_IDENTITIES


def test_duplicate_canonical_name_requires_provider() -> None:
    name = "infer_recurrent_fp32io16_forward_varlen"
    with pytest.raises(ValueError, match="ambiguous"):
        get_kernel_spec(name)
    assert get_kernel_spec(name, provider="vllm-rwkv").maturity == "stable"
    assert get_kernel_spec(name, provider="fla").maturity == "external"


def test_reference_and_stateful_helpers_are_not_kernel_identities() -> None:
    names = {spec.name for spec in KERNEL_SPECS}
    assert set(REFERENCE_ORACLES).isdisjoint(names)
    assert set(WRAPPER_KERNELS).isdisjoint(names)


@pytest.mark.parametrize(
    ("name", "layouts", "message"),
    [
        ("reference", ("fixed",), "invalid canonical kernel name"),
        (
            "infer_chunk_bf16_forward_varlen",
            ("fixed",),
            "_varlen identity requires exactly packed layout",
        ),
    ],
)
def test_registry_rejects_name_capability_mismatches(
    name: str,
    layouts: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        KernelSpec(
            provider="flashkda-derived",
            name=name,
            maturity="experimental",
            layouts=layouts,  # type: ignore[arg-type]
            autograd=False,
            state_behavior="functional",
            stages=("K1 prepare", "K2 recurrence"),
        )
