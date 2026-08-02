# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

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
    ("flash_rwkv", "statetune_recurrent_fp32io16_forward_backward"),
    ("flash_rwkv", "rl_infctx_chunk_fp32io16_materialized"),
    ("flash_rwkv", "rl_infctx_chunk_fp32io16_factor_recompute"),
    ("flash_rwkv", "rl_infctx_chunk_fp32io16_recurrent"),
}
ROOT = Path(__file__).parents[1]


def test_registry_contains_exact_provider_specific_identities() -> None:
    assert {spec.identity for spec in KERNEL_SPECS} == EXPECTED_IDENTITIES


def test_cmix_is_a_distinct_source_attributed_training_operator() -> None:
    assert training_operator_specs() == TRAINING_OPERATOR_SPECS
    spec = next(
        operator for operator in TRAINING_OPERATOR_SPECS if operator.family == "cmix"
    )
    assert spec.identity == ("rwkv-lm", "pretrain_cmix_bf16")
    assert spec.family == "cmix"
    assert spec.autograd is True
    assert spec.source_revision == "952102498e9ed367ea0a59ee64106916d474d30f"
    assert spec.native_ops == (
        "rwkv7_cmix_bf16_v5::forward",
        "rwkv7_cmix_bf16_v5::backward",
    )
    assert spec.identity not in {kernel.identity for kernel in KERNEL_SPECS}


def test_l2wrap_ce_preserves_its_distinct_loss_identity() -> None:
    spec = next(
        operator
        for operator in TRAINING_OPERATOR_SPECS
        if operator.family == "l2wrap_ce"
    )
    assert spec.identity == ("rwkv-lm", "pretrain_l2wrap_ce_bf16")
    assert spec.native_ops == (
        "rwkv7_l2wrap_ce_bf16_v2::forward",
        "rwkv7_l2wrap_ce_bf16_v2::backward",
    )
    assert spec.output_contract == (
        "mean_cross_entropy[]",
        "L2Wrap surrogate gradient",
    )


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


def test_workload_benchmark_identities_resolve_through_canonical_registry() -> None:
    statetune = get_kernel_spec(
        "statetune_recurrent_fp32io16_forward_backward",
        provider="flash_rwkv",
    )
    assert statetune.autograd is True
    assert statetune.layouts == ("fixed",)

    for strategy in ("materialized", "factor_recompute", "recurrent"):
        rl_infctx = get_kernel_spec(
            f"rl_infctx_chunk_fp32io16_{strategy}",
            provider="flash_rwkv",
        )
        assert rl_infctx.layouts == ("fixed", "packed")
        assert rl_infctx.maturity == "experimental"

    statetune_source = (
        ROOT
        / "benchmarks/statetune/wkv7/benchmark_statetune_recurrent_fp32io16_backward.py"
    ).read_text()
    rl_infctx_source = (
        ROOT / "benchmarks/rl_infctx/wkv7/benchmark_rl_infctx_chunk_fp32io16_forward.py"
    ).read_text()
    assert "OPERATOR_SPEC = get_kernel_spec(" in statetune_source
    assert "OPERATOR_SPECS = {" in rl_infctx_source
    assert '"provider": "flash_rwkv"' not in rl_infctx_source


def test_racecheck_covers_registered_statetune_result_identity() -> None:
    statetune = get_kernel_spec(
        "statetune_recurrent_fp32io16_forward_backward",
        provider="flash_rwkv",
    )
    source = (ROOT / "tests/racecheck/fused_operators.py").read_text()

    assert "STATETUNE_OPERATOR_SPEC = get_kernel_spec(" in source
    assert f'    "{statetune.name}",' in source
    assert f'    provider="{statetune.provider}",' in source
    assert "for input_dtype in (torch.float16, torch.bfloat16):" in source
    assert "output, final_state = statetune_recurrent_fp32io16_forward(" in source
    assert (
        "loss = output.float().square().mean() + final_state.square().mean()" in source
    )
    assert "initial_state_gradient = initial_state.grad" in source
    assert '"provider": STATETUNE_OPERATOR_SPEC.provider' in source
    assert '"name": STATETUNE_OPERATOR_SPEC.name' in source
    assert '"mode": STATETUNE_MODE' in source
    assert '"input_dtype": str(input_dtype).removeprefix("torch.")' in source
    assert '"statetune_results": statetune_results' in source


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
