# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from flash_rwkv.registry import (
    KERNEL_SPECS,
    REFERENCE_ORACLES,
    TRAINING_OPERATOR_SPECS,
    WRAPPER_KERNELS,
    KernelSpec,
    get_kernel_spec,
    training_operator_specs,
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


def test_cmix_is_a_distinct_source_attributed_training_operator() -> None:
    assert training_operator_specs() == TRAINING_OPERATOR_SPECS
    assert len(TRAINING_OPERATOR_SPECS) == 1
    spec = TRAINING_OPERATOR_SPECS[0]
    assert spec.identity == ("rwkv-lm", "pretrain_cmix_bf16")
    assert spec.family == "cmix"
    assert spec.autograd is True
    assert spec.source_revision == "952102498e9ed367ea0a59ee64106916d474d30f"
    assert spec.native_ops == (
        "rwkv7_cmix_bf16_v5::forward",
        "rwkv7_cmix_bf16_v5::backward",
    )
    assert spec.identity not in {kernel.identity for kernel in KERNEL_SPECS}


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
