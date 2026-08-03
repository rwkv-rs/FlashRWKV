// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the RWKV-LM project
// Adapted from RWKV-LM at commit
// 952102498e9ed367ea0a59ee64106916d474d30f.

#include "../../common/wkv7/recurrent_common_fp32io16.h"
#include "../../bindings.h"

void register_pretrain_recurrent_bindings(py::module_& module) {
  module.def(
      "pretrain_recurrent_fp32io16_forward",
      &recurrent_common_fp32io16_forward,
      "RWKV-LM-derived recurrent training forward with FP32 state",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("state"), py::arg("r"),
      py::arg("log_decay"), py::arg("k"), py::arg("v"), py::arg("a"),
      py::arg("b"), py::arg("output"), py::arg("boundary"),
      py::arg("state_dot_a"), py::arg("scale"));
  module.def(
      "pretrain_recurrent_fp32io16_backward",
      &recurrent_common_fp32io16_backward,
      "RWKV-LM-derived recurrent training backward with FP32 state",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("final_state"), py::arg("r"),
      py::arg("log_decay"), py::arg("k"), py::arg("v"), py::arg("a"),
      py::arg("b"), py::arg("state_dot_a"), py::arg("grad_output"),
      py::arg("grad_final_state"), py::arg("boundary"), py::arg("grad_r"),
      py::arg("grad_log_decay"), py::arg("grad_k"), py::arg("grad_v"),
      py::arg("grad_a"), py::arg("grad_b"),
      py::arg("grad_initial_state"), py::arg("scale"));
  module.def(
      "pretrain_recurrent_fp32io16_from_decay_logits_forward",
      &recurrent_common_fp32io16_from_decay_logits_forward,
      "RWKV recurrent training forward with fused raw decay logits",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("state"), py::arg("r"),
      py::arg("decay_logits"), py::arg("k"), py::arg("v"), py::arg("a"),
      py::arg("b"), py::arg("output"), py::arg("boundary"),
      py::arg("state_dot_a"), py::arg("scale"));
  module.def(
      "pretrain_recurrent_fp32io16_from_decay_logits_backward",
      &recurrent_common_fp32io16_from_decay_logits_backward,
      "RWKV recurrent training backward returning raw decay-logit gradients",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("final_state"), py::arg("r"),
      py::arg("decay_logits"), py::arg("k"), py::arg("v"), py::arg("a"),
      py::arg("b"), py::arg("state_dot_a"), py::arg("grad_output"),
      py::arg("grad_final_state"), py::arg("boundary"), py::arg("grad_r"),
      py::arg("grad_decay_logits"), py::arg("grad_k"), py::arg("grad_v"),
      py::arg("grad_a"), py::arg("grad_b"),
      py::arg("grad_initial_state"), py::arg("scale"));
}
