// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV project

#include "bindings.h"

void register_flash_rwkv_bindings(py::module_& module) {
  register_infer_recurrent_bindings(module);
  register_infer_recurrent_fp16_bindings(module);
  register_tmix_mix6_bindings(module);
  register_tmix_kk_a_gate_bindings(module);
  register_tmix_lnx_rkvres_xg_bindings(module);
  register_tmix_vres_gate_bindings(module);
  register_cmix_mix_bindings(module);
  register_tmix_linear_bindings(module);
  register_tmix_normalization_bindings(module);
  register_embedding_bindings(module);
  register_head_linear_bindings(module);
  register_cmix_sparse_bindings(module);
  register_pretrain_l2wrap_ce_forward_bindings(module);
  register_pretrain_l2wrap_ce_backward_bindings(module);
  register_pretrain_recurrent_bindings(module);
  register_infer_chunk_bindings(module);
  register_pretrain_tmix_a_gate_forward_bindings(module);
  register_pretrain_tmix_a_gate_backward_bindings(module);
  register_pretrain_tmix_vres_gate_forward_bindings(module);
  register_pretrain_tmix_vres_gate_backward_bindings(module);
  register_pretrain_tmix_mix6_forward_bindings(module);
  register_pretrain_tmix_mix6_backward_bindings(module);
  register_pretrain_cmix_forward_bindings(module);
  register_pretrain_cmix_backward_bindings(module);
  register_pretrain_tmix_kk_pre_bindings(module);
  register_pretrain_tmix_kk_pre_backward_bindings(module);
  register_pretrain_tmix_lnx_rkvres_xg_forward_bindings(module);
  register_pretrain_tmix_lnx_rkvres_xg_backward_bindings(module);
  register_pretrain_head_l2wrap_ce_bindings(module);
  register_statetune_recurrent_forward_bindings(module);
  register_statetune_recurrent_backward_bindings(module);
  register_rl_infctx_forward_bindings(module);
  register_rl_infctx_backward_bindings(module);
}
