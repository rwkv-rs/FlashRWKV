// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
//
// The FP16 kernel body is adapted from Albatross faster3a_2607 at revision
// ee3308f6922e59f2166c7fac3c5a192340a2b48e.  vllm-rwkv is used only for the
// packed metadata/state-slot contract reference.

#include "../../../validation.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

void recurrent_fp16_from_decay_logits_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor elapsed_state,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor decay_bias,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor metadata_status,
    double scale,
    int64_t max_seqlen);
void recurrent_fp16_advance_i32_cuda(torch::Tensor x, int64_t amount);
void recurrent_fp16_advance_i32_varlen_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor elapsed_state,
    torch::Tensor metadata_status);
torch::Tensor recurrent_add_vec_forward_varlen_cuda(
    torch::Tensor x, torch::Tensor vec);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;
using flashrwkv2::validation::prepare_recurrent_metadata_cuda;

namespace {

void check_fp16_recurrent_layout(
    const torch::Tensor& query_start_loc,
    const torch::Tensor& state_indices,
    const torch::Tensor& elapsed_state,
    const torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& decay_logits,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& output,
    double scale) {
  check_cuda_contiguous(query_start_loc, "query_start_loc");
  check_cuda_contiguous(state_indices, "state_indices");
  check_cuda_contiguous(elapsed_state, "elapsed_state_pool");
  check_cuda_contiguous(state, "state");
  check_cuda_contiguous(r, "r");
  check_cuda_contiguous(decay_logits, "decay_logits");
  check_cuda_contiguous(k, "k");
  check_cuda_contiguous(v, "v");
  check_cuda_contiguous(a, "a");
  check_cuda_contiguous(b, "b");
  check_cuda_contiguous(output, "output");
  check_same_device(state, query_start_loc, "query_start_loc");
  check_same_device(state, state_indices, "state_indices");
  check_same_device(state, elapsed_state, "elapsed_state_pool");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&r, "r"},
           {&decay_logits, "decay_logits"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&output, "output"},
       }) {
    check_same_device(state, *item.first, item.second);
  }
  TORCH_CHECK(
      query_start_loc.scalar_type() == torch::kInt32 &&
          state_indices.scalar_type() == torch::kInt32 &&
          elapsed_state.scalar_type() == torch::kInt32,
      "recurrent metadata must be int32");
  TORCH_CHECK(
      state_indices.dim() == 1 && state_indices.numel() > 0 &&
          query_start_loc.dim() == 1 &&
          query_start_loc.numel() == state_indices.numel() + 1,
      "query_start_loc must have shape [B+1] and state_indices shape [B]");
  TORCH_CHECK(
      elapsed_state.dim() == 1 && elapsed_state.size(0) == state.size(0),
      "elapsed_state_pool must have shape [state_pool_slots]");
  TORCH_CHECK(
      state.dim() == 4 && state.scalar_type() == torch::kFloat16 &&
          state.size(0) > 0 && state.size(1) > 0 &&
          state.size(2) == state.size(3),
      "state must be contiguous float16 [slots,H,D,D]");
  TORCH_CHECK(
      r.dim() == 3 && r.size(0) > 0 && r.size(1) == state.size(1) &&
          r.size(2) == state.size(2),
      "r must have packed shape [total_tokens,H,D] matching state");
  TORCH_CHECK(
      r.sizes() == decay_logits.sizes() && r.sizes() == k.sizes() &&
          r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
          r.sizes() == b.sizes() && r.sizes() == output.sizes(),
      "r,decay_logits,k,v,a,b,output shape mismatch");
  TORCH_CHECK(
      r.scalar_type() == torch::kFloat16 &&
          decay_logits.scalar_type() == torch::kFloat16 &&
          k.scalar_type() == torch::kFloat16 &&
          v.scalar_type() == torch::kFloat16 &&
          a.scalar_type() == torch::kFloat16 &&
          b.scalar_type() == torch::kFloat16 &&
          output.scalar_type() == torch::kFloat16,
      "FP16-state recurrent token tensors must be float16");
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");
}

}  // namespace

void recurrent_fp16_advance_i32(torch::Tensor x, int64_t amount) {
  check_cuda_contiguous(x, "elapsed state");
  TORCH_CHECK(x.scalar_type() == torch::kInt32,
              "elapsed state must be contiguous CUDA int32");
  TORCH_CHECK(x.numel() > 0, "elapsed state must be non-empty");
  recurrent_fp16_advance_i32_cuda(x, amount);
}

void recurrent_fp16_advance_i32_varlen(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor elapsed_state,
    int64_t total_tokens,
    py::object validated_metadata) {
  check_cuda_contiguous(query_start_loc, "query_start_loc");
  check_cuda_contiguous(state_indices, "state_indices");
  check_cuda_contiguous(elapsed_state, "elapsed_state_pool");
  check_same_device(elapsed_state, query_start_loc, "query_start_loc");
  check_same_device(elapsed_state, state_indices, "state_indices");
  TORCH_CHECK(
      query_start_loc.scalar_type() == torch::kInt32 &&
          state_indices.scalar_type() == torch::kInt32 &&
          elapsed_state.scalar_type() == torch::kInt32,
      "varlen elapsed metadata must be contiguous CUDA int32");
  TORCH_CHECK(
      query_start_loc.dim() == 1 && state_indices.dim() == 1 &&
          state_indices.numel() > 0 &&
          query_start_loc.numel() == state_indices.numel() + 1,
      "query_start_loc must have shape [B+1] and state_indices shape [B]");
  TORCH_CHECK(
      state_indices.numel() <= 65535,
      "state_indices must contain at most 65535 sequences");
  TORCH_CHECK(
      elapsed_state.dim() == 1 && elapsed_state.size(0) > 0 &&
          total_tokens > 0 &&
          total_tokens <= std::numeric_limits<int32_t>::max(),
      "invalid varlen elapsed state or total_tokens");

  torch::Tensor launch_query_start_loc = query_start_loc;
  torch::Tensor launch_state_indices = state_indices;
  torch::Tensor metadata_status;
  if (!validated_metadata.is_none()) {
    validated_metadata.attr("_check_compatible")(
        query_start_loc, state_indices, total_tokens, elapsed_state.size(0),
        -1);
    launch_query_start_loc = validated_metadata
        .attr("_query_start_loc_snapshot")()
        .cast<torch::Tensor>();
    launch_state_indices = validated_metadata
        .attr("_state_indices_snapshot")()
        .cast<torch::Tensor>();
    metadata_status = validated_metadata.attr("_status")().cast<torch::Tensor>();
  } else {
    auto prepared = prepare_recurrent_metadata_cuda(
        query_start_loc, state_indices, total_tokens, elapsed_state.size(0));
    launch_query_start_loc = std::move(prepared.query_start_loc);
    launch_state_indices = std::move(prepared.state_indices);
    metadata_status = std::move(prepared.status);
  }
  recurrent_fp16_advance_i32_varlen_cuda(
      launch_query_start_loc, launch_state_indices, elapsed_state,
      metadata_status);
}

torch::Tensor recurrent_add_vec_forward_varlen(
    torch::Tensor x, torch::Tensor vec) {
  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(vec, "vec");
  check_same_device(x, vec, "vec");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat16 && vec.scalar_type() == torch::kFloat16,
      "x and vec must be float16");
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0 && (x.size(1) % 2) == 0,
      "x must have packed shape [total_tokens,C] with even C");
  TORCH_CHECK(
      vec.dim() == 1 && vec.size(0) == x.size(1),
      "vec must have shape [C]");
  return recurrent_add_vec_forward_varlen_cuda(x, vec);
}

void recurrent_fp16_from_decay_logits(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor elapsed_state,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    double scale,
    std::optional<torch::Tensor> decay_bias,
    py::object validated_metadata,
    int64_t max_seqlen) {
  check_fp16_recurrent_layout(
      query_start_loc, state_indices, elapsed_state, state, r, decay_logits,
      k, v, a, b, output, scale);
  if (decay_bias.has_value()) {
    check_cuda_contiguous(*decay_bias, "decay_bias");
    check_same_device(state, *decay_bias, "decay_bias");
    TORCH_CHECK(
        decay_bias->scalar_type() == torch::kFloat16 &&
            ((decay_bias->dim() == 1 &&
              decay_bias->numel() == state.size(1) * state.size(2)) ||
             (decay_bias->dim() == 2 && decay_bias->size(0) == state.size(1) &&
              decay_bias->size(1) == state.size(2))),
        "decay_bias must be float16 with shape [H*D] or [H,D]");
  }
  torch::Tensor launch_query_start_loc = query_start_loc;
  torch::Tensor launch_state_indices = state_indices;
  torch::Tensor metadata_status;
  if (!validated_metadata.is_none()) {
    // The ticket class is registered by the FP32 recurrent binding and is
    // intentionally consumed through its private pybind surface here.  This
    // keeps one metadata implementation while making FP16 obey the same
    // identity/version/stream/snapshot contract.
    validated_metadata.attr("_check_compatible")(
        query_start_loc, state_indices, r.size(0), state.size(0),
        max_seqlen);
    launch_query_start_loc = validated_metadata
        .attr("_query_start_loc_snapshot")()
        .cast<torch::Tensor>();
    launch_state_indices = validated_metadata
        .attr("_state_indices_snapshot")()
        .cast<torch::Tensor>();
    metadata_status = validated_metadata.attr("_status")().cast<torch::Tensor>();
    max_seqlen = validated_metadata.attr("_max_seqlen")().cast<int64_t>();
  } else {
    auto prepared = prepare_recurrent_metadata_cuda(
        query_start_loc, state_indices, r.size(0), state.size(0));
    launch_query_start_loc = std::move(prepared.query_start_loc);
    launch_state_indices = std::move(prepared.state_indices);
    metadata_status = std::move(prepared.status);
  }
  recurrent_fp16_from_decay_logits_cuda(
      launch_query_start_loc, launch_state_indices, elapsed_state, state, r,
      decay_logits, decay_bias.value_or(torch::Tensor()), k, v, a, b, output,
      metadata_status, scale, max_seqlen);
}

void register_infer_recurrent_fp16_bindings(py::module_& module) {
  module.def(
      "recurrent_fp16_advance_i32",
      &recurrent_fp16_advance_i32,
      "Advance the canonical FP16 recurrent elapsed/dither state",
      py::arg("elapsed_state"), py::arg("amount"));
  module.def(
      "recurrent_fp16_advance_i32_varlen",
      &recurrent_fp16_advance_i32_varlen,
      "Advance selected FP16 recurrent elapsed/dither slots by packed lengths",
      py::arg("query_start_loc"), py::arg("state_indices"),
      py::arg("elapsed_state_pool"), py::arg("total_tokens"),
      py::arg("validated_metadata") = py::none());
  module.def(
      "recurrent_add_vec_forward_varlen",
      &recurrent_add_vec_forward_varlen,
      "Packed Albatross WKV pre-add-vector helper",
      py::arg("x"), py::arg("vec"));
  module.def(
      "recurrent_fp16_from_decay_logits",
      &recurrent_fp16_from_decay_logits,
      "Albatross-first packed recurrent forward with FP16 state",
      py::arg("query_start_loc"), py::arg("state_indices"),
      py::arg("elapsed_state_pool"), py::arg("state"), py::arg("r"),
      py::arg("decay_logits"),
      py::arg("k"), py::arg("v"), py::arg("a"), py::arg("b"),
      py::arg("output"), py::arg("scale"),
      py::arg("decay_bias") = py::none(),
      py::arg("validated_metadata") = py::none(),
      py::arg("max_seqlen") = -1);
}
