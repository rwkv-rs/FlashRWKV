// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vllm-rwkv rwkv7_wkv_fp32_v2 at commit
// 6d683f9e49a2997e405c47edc147872c8609513b for the FlashRWKV core contract.

#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <utility>

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
    double scale);

namespace {

constexpr int64_t kHeadSize = 64;

struct RecurrentDimensions {
  int64_t num_sequences;
  int64_t num_heads;
};

void check_cuda_contiguous(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(
      tensor.device() == reference.device(),
      name,
      " must be on the same device as state");
}

RecurrentDimensions check_recurrent_layout(
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
  check_cuda_contiguous(query_start_loc, "query_start_loc");
  check_cuda_contiguous(state_indices, "state_indices");
  check_cuda_contiguous(state, "state");
  check_cuda_contiguous(r, "r");
  check_cuda_contiguous(log_decay, "log_decay");
  check_cuda_contiguous(k, "k");
  check_cuda_contiguous(v, "v");
  check_cuda_contiguous(a, "a");
  check_cuda_contiguous(b, "b");
  check_cuda_contiguous(output, "output");

  TORCH_CHECK(
      query_start_loc.scalar_type() == torch::kInt32,
      "query_start_loc must be int32");
  TORCH_CHECK(
      state_indices.scalar_type() == torch::kInt32,
      "state_indices must be int32");
  TORCH_CHECK(
      std::isfinite(scale),
      "scale must be finite");

  const int64_t num_sequences = state_indices.numel();
  TORCH_CHECK(
      num_sequences > 0 && num_sequences <= 65535,
      "state_indices must contain 1..65535 sequences");
  TORCH_CHECK(
      state_indices.dim() == 1,
      "state_indices must have shape [N]");
  TORCH_CHECK(
      query_start_loc.dim() == 1 &&
          query_start_loc.size(0) == num_sequences + 1,
      "query_start_loc must have shape [N+1]");
  TORCH_CHECK(
      state.dim() == 4 && state.size(0) > 0 &&
          state.size(1) > 0 &&
          state.size(2) == kHeadSize &&
          state.size(3) == kHeadSize,
      "state must have shape [slots,H,64,64]");

  const int64_t num_heads = state.size(1);
  TORCH_CHECK(
      num_heads <= std::numeric_limits<int>::max(),
      "head count must fit in int32");
  TORCH_CHECK(
      r.dim() == 3 && r.size(0) > 0 &&
          r.size(1) == num_heads &&
          r.size(2) == kHeadSize,
      "r must have shape [total_tokens,H,64]");
  TORCH_CHECK(
      r.sizes() == log_decay.sizes() &&
          r.sizes() == k.sizes() &&
          r.sizes() == v.sizes() &&
          r.sizes() == a.sizes() &&
          r.sizes() == b.sizes() &&
          r.sizes() == output.sizes(),
      "r,log_decay,k,v,a,b,output shape mismatch");
  TORCH_CHECK(
      r.scalar_type() == log_decay.scalar_type() &&
          r.scalar_type() == k.scalar_type() &&
          r.scalar_type() == v.scalar_type() &&
          r.scalar_type() == a.scalar_type() &&
          r.scalar_type() == b.scalar_type() &&
          r.scalar_type() == output.scalar_type(),
      "r,log_decay,k,v,a,b,output dtype mismatch");

  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &query_start_loc, "query_start_loc"},
           {&state_indices, "state_indices"},
           {&r, "r"},
           {&log_decay, "log_decay"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&output, "output"},
       }) {
    check_same_device(state, *item.first, item.second);
  }

  return RecurrentDimensions{num_sequences, num_heads};
}

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
      scale);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "recurrent_fp32",
      &recurrent_fp32,
      "FlashRWKV recurrent forward with FP32 canonical state",
      py::arg("query_start_loc"),
      py::arg("state_indices"),
      py::arg("state"),
      py::arg("r"),
      py::arg("log_decay"),
      py::arg("k"),
      py::arg("v"),
      py::arg("a"),
      py::arg("b"),
      py::arg("output"),
      py::arg("scale"));
  module.def(
      "recurrent_fp16",
      &recurrent_fp16,
      "FlashRWKV recurrent forward with FP16 canonical state",
      py::arg("query_start_loc"),
      py::arg("state_indices"),
      py::arg("state"),
      py::arg("r"),
      py::arg("log_decay"),
      py::arg("k"),
      py::arg("v"),
      py::arg("a"),
      py::arg("b"),
      py::arg("output"),
      py::arg("scale"));
}
