// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project

#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  register_flashrwkv2_bindings(module);
}
