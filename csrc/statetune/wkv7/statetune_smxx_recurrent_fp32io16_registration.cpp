// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV project

#include <torch/extension.h>

void register_statetune_recurrent_manifest(py::module_& module) {
  py::dict manifest;
  manifest["workload"] = "statetune";
  manifest["model_family"] = "wkv7";
  manifest["numerical_mode"] = "fp32io16";
  manifest["forward_source"] =
      "csrc/pretrain/wkv7/pretrain_smxx_recurrent_fp32io16_forward.cu";
  manifest["backward_source"] =
      "csrc/pretrain/wkv7/pretrain_smxx_recurrent_fp32io16_backward.cu";
  manifest["state_contract"] =
      "nonzero initial state, final state, and initial-state gradient";
  module.attr("_statetune_recurrent_source_manifest") = manifest;
}
