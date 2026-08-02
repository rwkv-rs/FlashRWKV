// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV project

#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  register_flash_rwkv_bindings(module);
}
