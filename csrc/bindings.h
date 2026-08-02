// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV project

#pragma once

#include <torch/extension.h>

void register_flash_rwkv_bindings(py::module_&);
void register_statetune_recurrent_manifest(py::module_&);
void register_infer_recurrent_bindings(py::module_&);
void register_pretrain_recurrent_bindings(py::module_&);
void register_rl_infctx_experimental_bindings(py::module_&);
void register_infer_experimental_bindings(py::module_&);
