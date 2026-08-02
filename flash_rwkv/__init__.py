# SPDX-License-Identifier: MIT

from .channel_mix import pretrain_cmix_bf16
from .config import ChunkConfig, enumerate_chunk_configs
from .head_l2wrap_ce import pretrain_head_l2wrap_ce_bf16
from .inference_blocks import (
    infer_cmix_mix_fp16,
    infer_tmix_kk_a_gate_fp16,
    infer_tmix_lnx_rkvres_xg_fp16,
    infer_tmix_mix6_fp16,
    infer_tmix_vres_gate_fp16,
)
from .l2wrap_ce import pretrain_l2wrap_ce_bf16
from .ops import (
    decay_logits_to_log_decay,
    infer_chunk_bf16_forward,
    infer_chunk_bf16_forward_varlen,
    infer_recurrent_fp16_forward_varlen,
    infer_recurrent_fp32io16_forward_varlen,
    pretrain_recurrent_fp32io16,
    pretrain_recurrent_fp32io16_forward,
    rwkv7,
    rwkv7_from_decay_logits,
    rwkv7_recurrent_stateful,
)
from .reference import rwkv7_reference
from .registry import (
    KernelSpec,
    TrainingOperatorSpec,
    get_kernel_spec,
    kernel_specs,
    training_operator_specs,
)
from .time_mix import (
    pretrain_tmix_a_gate_bf16,
    pretrain_tmix_kk_pre_bf16,
    pretrain_tmix_lnx_rkvres_xg_bf16,
    pretrain_tmix_mix6_bf16,
    pretrain_tmix_vres_gate_bf16,
)
from .validation import validate_packed_metadata_strict

__all__ = [
    "ChunkConfig",
    "KernelSpec",
    "TrainingOperatorSpec",
    "decay_logits_to_log_decay",
    "enumerate_chunk_configs",
    "get_kernel_spec",
    "infer_chunk_bf16_forward",
    "infer_chunk_bf16_forward_varlen",
    "infer_cmix_mix_fp16",
    "infer_recurrent_fp16_forward_varlen",
    "infer_recurrent_fp32io16_forward_varlen",
    "infer_tmix_kk_a_gate_fp16",
    "infer_tmix_lnx_rkvres_xg_fp16",
    "infer_tmix_mix6_fp16",
    "infer_tmix_vres_gate_fp16",
    "kernel_specs",
    "pretrain_cmix_bf16",
    "pretrain_head_l2wrap_ce_bf16",
    "pretrain_l2wrap_ce_bf16",
    "pretrain_recurrent_fp32io16",
    "pretrain_recurrent_fp32io16_forward",
    "pretrain_tmix_a_gate_bf16",
    "pretrain_tmix_kk_pre_bf16",
    "pretrain_tmix_lnx_rkvres_xg_bf16",
    "pretrain_tmix_mix6_bf16",
    "pretrain_tmix_vres_gate_bf16",
    "rwkv7",
    "rwkv7_from_decay_logits",
    "rwkv7_recurrent_stateful",
    "rwkv7_reference",
    "training_operator_specs",
    "validate_packed_metadata_strict",
]

__version__ = "0.1.0"
