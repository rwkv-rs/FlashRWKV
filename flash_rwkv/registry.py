# SPDX-License-Identifier: MIT

"""Canonical identities and capabilities for measurable RWKV-7 kernels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Provider = Literal["rwkv-lm", "vllm-rwkv", "flashkda-derived", "fla"]
Maturity = Literal["stable", "experimental", "external"]
Layout = Literal["fixed", "packed"]
StateBehavior = Literal["functional"]
OperatorFamily = Literal["tmix", "cmix", "l2wrap_ce", "head_l2wrap_ce"]

_NAME_PATTERN = re.compile(
    r"^(pretrain|infer)_(recurrent|chunk)_"
    r"(fp32io16|fp16|bf16)_(forward|backward)(_varlen)?$"
)


@dataclass(frozen=True, slots=True)
class KernelSpec:
    """One provider-specific implementation of a canonical kernel contract."""

    provider: Provider
    name: str
    maturity: Maturity
    layouts: tuple[Layout, ...]
    autograd: bool
    state_behavior: StateBehavior
    stages: tuple[str, ...]

    def __post_init__(self) -> None:
        match = _NAME_PATTERN.fullmatch(self.name)
        if match is None:
            raise ValueError(f"invalid canonical kernel name: {self.name!r}")
        workload, _, _, direction, varlen = match.groups()
        if not self.layouts or len(set(self.layouts)) != len(self.layouts):
            raise ValueError(f"{self.name}: layouts must be non-empty and unique")
        if bool(varlen) != (self.layouts == ("packed",)):
            raise ValueError(
                f"{self.name}: _varlen identity requires exactly packed layout"
            )
        if workload == "pretrain" and not self.autograd:
            raise ValueError(f"{self.name}: pretrain kernels must support autograd")
        if direction == "backward" and not self.autograd:
            raise ValueError(f"{self.name}: backward kernels must support autograd")
        if not self.stages or any(not stage for stage in self.stages):
            raise ValueError(f"{self.name}: stages must be non-empty")

    @property
    def identity(self) -> tuple[Provider, str]:
        return self.provider, self.name


@dataclass(frozen=True, slots=True)
class TrainingOperatorSpec:
    """A source-attributed non-recurrence training operator family."""

    provider: Provider
    name: str
    family: OperatorFamily
    dtype: str
    autograd: bool
    native_ops: tuple[str, ...]
    source_revision: str
    input_contract: tuple[str, ...]
    output_contract: tuple[str, ...]

    def __post_init__(self) -> None:
        expected_prefix = f"pretrain_{self.family}_"
        if not self.name.startswith(expected_prefix):
            raise ValueError(f"{self.name!r} must start with {expected_prefix!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_revision):
            raise ValueError("source_revision must be a canonical full Git OID")
        if not self.native_ops or len(set(self.native_ops)) != len(self.native_ops):
            raise ValueError("native_ops must be non-empty and unique")
        if not self.input_contract or not self.output_contract:
            raise ValueError("input and output contracts must be non-empty")

    @property
    def identity(self) -> tuple[Provider, str]:
        return self.provider, self.name


KERNEL_SPECS: tuple[KernelSpec, ...] = (
    KernelSpec(
        provider="rwkv-lm",
        name="pretrain_recurrent_fp32io16_forward",
        maturity="stable",
        layouts=("fixed",),
        autograd=True,
        state_behavior="functional",
        stages=("forward recurrence",),
    ),
    KernelSpec(
        provider="rwkv-lm",
        name="pretrain_recurrent_fp32io16_backward",
        maturity="stable",
        layouts=("fixed",),
        autograd=True,
        state_behavior="functional",
        stages=("backward recurrence",),
    ),
    KernelSpec(
        provider="vllm-rwkv",
        name="infer_recurrent_fp32io16_forward_varlen",
        maturity="stable",
        layouts=("packed",),
        autograd=False,
        state_behavior="functional",
        stages=("forward recurrence",),
    ),
    KernelSpec(
        provider="vllm-rwkv",
        name="infer_recurrent_fp16_forward_varlen",
        maturity="stable",
        layouts=("packed",),
        autograd=False,
        state_behavior="functional",
        stages=("forward recurrence",),
    ),
    KernelSpec(
        provider="flashkda-derived",
        name="infer_chunk_bf16_forward",
        maturity="experimental",
        layouts=("fixed",),
        autograd=False,
        state_behavior="functional",
        stages=("K1 prepare", "K2 recurrence"),
    ),
    KernelSpec(
        provider="flashkda-derived",
        name="infer_chunk_bf16_forward_varlen",
        maturity="experimental",
        layouts=("packed",),
        autograd=False,
        state_behavior="functional",
        stages=("K1 prepare", "K2 recurrence"),
    ),
    KernelSpec(
        provider="fla",
        name="pretrain_chunk_fp32io16_forward",
        maturity="external",
        layouts=("fixed",),
        autograd=True,
        state_behavior="functional",
        stages=("FLA chunk forward",),
    ),
    KernelSpec(
        provider="fla",
        name="pretrain_chunk_fp32io16_backward",
        maturity="external",
        layouts=("fixed",),
        autograd=True,
        state_behavior="functional",
        stages=("FLA chunk backward",),
    ),
    KernelSpec(
        provider="fla",
        name="infer_recurrent_fp32io16_forward_varlen",
        maturity="external",
        layouts=("packed",),
        autograd=False,
        state_behavior="functional",
        stages=("FLA fused recurrent forward",),
    ),
)

TRAINING_OPERATOR_SPECS: tuple[TrainingOperatorSpec, ...] = (
    TrainingOperatorSpec(
        provider="rwkv-lm",
        name="pretrain_head_l2wrap_ce_bf16",
        family="head_l2wrap_ce",
        dtype="bfloat16",
        autograd=True,
        native_ops=("rwkv7_head_l2wrap_ce_bf16_v4::forward",),
        source_revision="952102498e9ed367ea0a59ee64106916d474d30f",
        input_contract=("hidden[B,T,C]", "weight[65536,C]", "targets[B,T]"),
        output_contract=("mean_cross_entropy[]", "L2Wrap surrogate gradients"),
    ),
    TrainingOperatorSpec(
        provider="rwkv-lm",
        name="pretrain_tmix_a_gate_bf16",
        family="tmix",
        dtype="bfloat16",
        autograd=True,
        native_ops=(
            "rwkv7_tmix_a_gate_bf16::forward",
            "rwkv7_tmix_a_gate_bf16::backward",
        ),
        source_revision="952102498e9ed367ea0a59ee64106916d474d30f",
        input_contract=("a0[C]", "a12[B,T,C]"),
        output_contract=("sigmoid_gate[B,T,C]",),
    ),
    TrainingOperatorSpec(
        provider="rwkv-lm",
        name="pretrain_tmix_mix6_bf16",
        family="tmix",
        dtype="bfloat16",
        autograd=True,
        native_ops=(
            "rwkv7_tmix_mix6_bf16_v5::forward",
            "rwkv7_tmix_mix6_bf16_v5::backward",
        ),
        source_revision="952102498e9ed367ea0a59ee64106916d474d30f",
        input_contract=("x[B,T,C]", "x_{r,w,k,v,a,g}[C]"),
        output_contract=("mixed_{r,w,k,v,a,g}[B,T,C]",),
    ),
    TrainingOperatorSpec(
        provider="rwkv-lm",
        name="pretrain_tmix_vres_gate_bf16",
        family="tmix",
        dtype="bfloat16",
        autograd=True,
        native_ops=(
            "rwkv7_tmix_vres_gate_bf16_v3::forward",
            "rwkv7_tmix_vres_gate_bf16_v3::backward",
        ),
        source_revision="952102498e9ed367ea0a59ee64106916d474d30f",
        input_contract=("value[B,T,C]", "first_value[B,T,C]", "v0[C]", "v12[B,T,C]"),
        output_contract=("blended_value[B,T,C]",),
    ),
    TrainingOperatorSpec(
        provider="rwkv-lm",
        name="pretrain_cmix_bf16",
        family="cmix",
        dtype="bfloat16",
        autograd=True,
        native_ops=(
            "rwkv7_cmix_bf16_v5::forward",
            "rwkv7_cmix_bf16_v5::backward",
        ),
        source_revision="952102498e9ed367ea0a59ee64106916d474d30f",
        input_contract=("x[B,T,C]", "x_k[C]", "key[4C,C]", "value[C,4C]"),
        output_contract=("output[B,T,C]",),
    ),
    TrainingOperatorSpec(
        provider="rwkv-lm",
        name="pretrain_l2wrap_ce_bf16",
        family="l2wrap_ce",
        dtype="bfloat16",
        autograd=True,
        native_ops=(
            "rwkv7_l2wrap_ce_bf16_v2::forward",
            "rwkv7_l2wrap_ce_bf16_v2::backward",
        ),
        source_revision="952102498e9ed367ea0a59ee64106916d474d30f",
        input_contract=("logits[...,V]", "targets[...]"),
        output_contract=("mean_cross_entropy[]", "L2Wrap surrogate gradient"),
    ),
)

_BY_IDENTITY = {spec.identity: spec for spec in KERNEL_SPECS}
if len(_BY_IDENTITY) != len(KERNEL_SPECS):
    raise RuntimeError("kernel registry contains a duplicate provider/name identity")


def kernel_specs() -> tuple[KernelSpec, ...]:
    """Return the immutable provider-specific kernel registry."""

    return KERNEL_SPECS


def training_operator_specs() -> tuple[TrainingOperatorSpec, ...]:
    """Return imported training operators that are not WKV recurrences."""

    return TRAINING_OPERATOR_SPECS


def get_kernel_spec(name: str, *, provider: Provider | None = None) -> KernelSpec:
    """Resolve one kernel, failing when a canonical name has multiple providers."""

    matches = tuple(
        spec
        for spec in KERNEL_SPECS
        if spec.name == name and (provider is None or spec.provider == provider)
    )
    if not matches:
        qualifier = "" if provider is None else f" for provider {provider!r}"
        raise KeyError(f"unknown kernel {name!r}{qualifier}")
    if len(matches) > 1:
        providers = ", ".join(spec.provider for spec in matches)
        raise ValueError(
            f"kernel {name!r} is ambiguous; specify provider from: {providers}"
        )
    return matches[0]


# These public helpers reuse registry kernels; they are not additional kernels.
WRAPPER_KERNELS = {
    "rwkv7_recurrent_stateful": (
        ("vllm-rwkv", "infer_recurrent_fp32io16_forward_varlen"),
        ("vllm-rwkv", "infer_recurrent_fp16_forward_varlen"),
    ),
}
REFERENCE_ORACLES = ("rwkv7_reference",)
