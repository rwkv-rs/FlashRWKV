# SPDX-License-Identifier: MIT

# Load PyTorch before the native module.  The extension links against
# ``libc10`` and the other libraries loaded by PyTorch; importing ``_C``
# first makes the optional import fail even when the editable build exists.
import torch as _torch  # noqa: F401

try:
    from . import _C
except ImportError:
    _C = None

from .tmix.wkv7 import (
    infer_chunk_bf16_forward_varlen,
    infer_recurrent_fp16_forward_varlen,
    infer_recurrent_fp32io16_forward_varlen,
    pretrain_recurrent_fp32io16,
    prepare_recurrent_metadata,
)
from .tmix.wkv7.statetune import statetune_recurrent_fp32io16
from .tmix.a_gate import pretrain_tmix_a_gate_bf16
from .tmix.vres_gate import pretrain_tmix_vres_gate_bf16
from .tmix.mix6 import pretrain_tmix_mix6_bf16
from .tmix.kk_pre import pretrain_tmix_kk_pre_bf16
from .tmix.lnx_rkvres_xg import pretrain_tmix_lnx_rkvres_xg_bf16
from .cmix.mix import pretrain_cmix_bf16
from .head.l2wrap_ce import pretrain_head_l2wrap_ce_bf16
from .loss.l2wrap_ce import pretrain_l2wrap_ce_bf16
from .rl_infctx.wkv7 import (
    rl_infctx_chunk_fp32io16,
    rl_infctx_chunk_fp32io16_factor_recompute,
)

__all__ = [
    "infer_chunk_bf16_forward_varlen",
    "infer_recurrent_fp16_forward_varlen",
    "infer_recurrent_fp32io16_forward_varlen",
    "pretrain_recurrent_fp32io16",
    "statetune_recurrent_fp32io16",
    "prepare_recurrent_metadata",
    "pretrain_tmix_a_gate_bf16",
    "pretrain_tmix_vres_gate_bf16",
    "pretrain_tmix_mix6_bf16",
    "pretrain_tmix_kk_pre_bf16",
    "pretrain_tmix_lnx_rkvres_xg_bf16",
    "pretrain_cmix_bf16",
    "pretrain_head_l2wrap_ce_bf16",
    "pretrain_l2wrap_ce_bf16",
    "rl_infctx_chunk_fp32io16",
    "rl_infctx_chunk_fp32io16_factor_recompute",
]

__version__ = "0.1.0"
