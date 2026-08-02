// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV project

#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  register_statetune_recurrent_manifest(module);
  module.def("recurrent_fp32", &recurrent_fp32,
             "FlashRWKV recurrent forward with FP32 canonical state",
             py::arg("query_start_loc"), py::arg("state_indices"),
             py::arg("state"), py::arg("r"), py::arg("log_decay"),
             py::arg("k"), py::arg("v"), py::arg("a"), py::arg("b"),
             py::arg("output"), py::arg("scale"));
  module.def("recurrent_fp16", &recurrent_fp16,
             "FlashRWKV recurrent forward with FP16 canonical state",
             py::arg("query_start_loc"), py::arg("state_indices"),
             py::arg("state"), py::arg("r"), py::arg("log_decay"),
             py::arg("k"), py::arg("v"), py::arg("a"), py::arg("b"),
             py::arg("output"), py::arg("scale"));
  module.def(
      "pretrain_recurrent_fp32io16_forward",
      &pretrain_recurrent_fp32io16_forward,
      "RWKV-LM-derived recurrent training forward with FP32 state",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("state"), py::arg("r"),
      py::arg("log_decay"), py::arg("k"), py::arg("v"), py::arg("a"),
      py::arg("b"), py::arg("output"), py::arg("boundary"),
      py::arg("state_dot_a"), py::arg("scale"));
  module.def(
      "materialized_chunk_fp32", &materialized_chunk_fp32,
      "FlashRWKV materialized chunk forward with FP32 canonical state",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("state_indices"), py::arg("state"),
      py::arg("r"), py::arg("log_decay"), py::arg("k"), py::arg("v"),
      py::arg("a"), py::arg("b"), py::arg("output"), py::arg("transform"),
      py::arg("bias"), py::arg("boundary"), py::arg("build_warps"),
      py::arg("stages"), py::arg("state_tile"), py::arg("scale"),
      py::arg("state_dot_a") = py::none());
  module.def(
      "pretrain_recurrent_fp32io16_backward",
      &pretrain_recurrent_fp32io16_backward,
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
      "infer_chunk_bf16_forward_k1_prepare",
      &infer_chunk_bf16_forward_k1_prepare,
      "KDA-derived K1 chunk preparation for BF16 inference",
      py::arg("chunk_token_starts"), py::arg("chunk_token_ends"), py::arg("r"),
      py::arg("log_decay"), py::arg("k"), py::arg("v"), py::arg("a"),
      py::arg("b"), py::arg("chunk_transform"), py::arg("chunk_bias"),
      py::arg("token_transform"), py::arg("token_bias"), py::arg("scale"));
  module.def(
      "infer_chunk_bf16_forward_k2_recurrence",
      &infer_chunk_bf16_forward_k2_recurrence,
      "KDA-derived K2 boundary recurrence for BF16 inference",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("state"), py::arg("output"),
      py::arg("chunk_transform"), py::arg("chunk_bias"),
      py::arg("token_transform"), py::arg("token_bias"));
  module.def(
      "recompute_chunk_fp32", &recompute_chunk_fp32,
      "FlashRWKV DPLR-factor recompute chunk forward with FP32 state",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("state_indices"), py::arg("state"),
      py::arg("r"), py::arg("log_decay"), py::arg("k"), py::arg("v"),
      py::arg("a"), py::arg("b"), py::arg("output"), py::arg("boundary"),
      py::arg("scale"));
}
