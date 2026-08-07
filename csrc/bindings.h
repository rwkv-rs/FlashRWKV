// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project

#pragma once

#include <torch/extension.h>

void register_flashrwkv2_bindings(py::module_&);
void register_infer_recurrent_bindings(py::module_&);
void register_infer_recurrent_fp16_bindings(py::module_&);
void register_tmix_mix6_bindings(py::module_&);
void register_tmix_kk_a_gate_bindings(py::module_&);
void register_tmix_lnx_rkvres_xg_bindings(py::module_&);
void register_tmix_vres_gate_bindings(py::module_&);
void register_cmix_mix_bindings(py::module_&);
void register_tmix_linear_bindings(py::module_&);
void register_tmix_normalization_bindings(py::module_&);
void register_embedding_bindings(py::module_&);
void register_head_linear_bindings(py::module_&);
void register_cmix_sparse_bindings(py::module_&);
void register_pretrain_l2wrap_ce_forward_bindings(py::module_&);
void register_pretrain_l2wrap_ce_backward_bindings(py::module_&);
void register_infer_chunk_bindings(py::module_&);
void register_pretrain_tmix_a_gate_forward_bindings(py::module_&);
void register_pretrain_tmix_a_gate_backward_bindings(py::module_&);
void register_pretrain_tmix_vres_gate_forward_bindings(py::module_&);
void register_pretrain_tmix_vres_gate_backward_bindings(py::module_&);
void register_pretrain_tmix_mix6_forward_bindings(py::module_&);
void register_pretrain_tmix_mix6_backward_bindings(py::module_&);
void register_pretrain_cmix_forward_bindings(py::module_&);
void register_pretrain_cmix_backward_bindings(py::module_&);
void register_pretrain_tmix_kk_pre_bindings(py::module_&);
void register_pretrain_tmix_kk_pre_backward_bindings(py::module_&);
void register_pretrain_tmix_lnx_rkvres_xg_forward_bindings(py::module_&);
void register_pretrain_tmix_lnx_rkvres_xg_backward_bindings(py::module_&);
void register_pretrain_head_l2wrap_ce_bindings(py::module_&);
void register_statetune_recurrent_forward_bindings(py::module_&);
void register_statetune_recurrent_backward_bindings(py::module_&);
void register_rl_infctx_forward_bindings(py::module_&);
void register_rl_infctx_backward_bindings(py::module_&);
