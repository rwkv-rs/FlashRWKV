# SPDX-License-Identifier: MIT

from .config import ChunkConfig, enumerate_chunk_configs
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
from .registry import KernelSpec, get_kernel_spec, kernel_specs

__all__ = [
    "ChunkConfig",
    "KernelSpec",
    "decay_logits_to_log_decay",
    "enumerate_chunk_configs",
    "get_kernel_spec",
    "infer_chunk_bf16_forward",
    "infer_chunk_bf16_forward_varlen",
    "infer_recurrent_fp16_forward_varlen",
    "infer_recurrent_fp32io16_forward_varlen",
    "kernel_specs",
    "pretrain_recurrent_fp32io16",
    "pretrain_recurrent_fp32io16_forward",
    "rwkv7",
    "rwkv7_from_decay_logits",
    "rwkv7_reference",
    "rwkv7_recurrent_stateful",
]

__version__ = "0.1.0"
