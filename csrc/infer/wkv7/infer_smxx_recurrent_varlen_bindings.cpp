// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vllm-rwkv rwkv7_wkv_fp32_v2 at commit
// 6d683f9e49a2997e405c47edc147872c8609513b.

#include "../../bindings.h"
#include "../../validation.h"

void recurrent_fp32_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor metadata_status,
    double scale);
void recurrent_fp16_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor metadata_status,
    double scale);

using flash_rwkv::validation::check_recurrent_layout;
using flash_rwkv::validation::validate_recurrent_metadata_cuda;

void recurrent_fp32(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    double scale) {
  check_recurrent_layout(
      query_start_loc,
      state_indices,
      state,
      r,
      log_decay,
      k,
      v,
      a,
      b,
      output,
      scale);
  TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be fp32");
  TORCH_CHECK(
      r.scalar_type() == torch::kFloat16 ||
          r.scalar_type() == torch::kBFloat16 ||
          r.scalar_type() == torch::kFloat32,
      "FP32-state token tensors must be fp16, bf16, or fp32");

  auto metadata_status = validate_recurrent_metadata_cuda(
      query_start_loc, state_indices, r.size(0), state.size(0));
  recurrent_fp32_cuda(
      query_start_loc,
      state_indices,
      state,
      r,
      log_decay,
      k,
      v,
      a,
      b,
      output,
      metadata_status,
      scale);
}

void recurrent_fp16(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    double scale) {
  check_recurrent_layout(
      query_start_loc,
      state_indices,
      state,
      r,
      log_decay,
      k,
      v,
      a,
      b,
      output,
      scale);
  TORCH_CHECK(state.scalar_type() == torch::kFloat16, "state must be fp16");
  TORCH_CHECK(
      r.scalar_type() == torch::kFloat16,
      "FP16-state token tensors must be fp16");

  auto metadata_status = validate_recurrent_metadata_cuda(
      query_start_loc, state_indices, r.size(0), state.size(0));
  recurrent_fp16_cuda(
      query_start_loc,
      state_indices,
      state,
      r,
      log_decay,
      k,
      v,
      a,
      b,
      output,
      metadata_status,
      scale);
}

void register_infer_recurrent_bindings(py::module_& module) {
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
}
