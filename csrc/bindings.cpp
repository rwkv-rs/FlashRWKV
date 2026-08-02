// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV project

#include "bindings.h"

void register_flash_rwkv_bindings(py::module_& module) {
  register_statetune_recurrent_bindings(module);
  register_infer_recurrent_bindings(module);
  register_pretrain_recurrent_bindings(module);
  register_rl_infctx_experimental_bindings(module);
  register_infer_experimental_bindings(module);
}
