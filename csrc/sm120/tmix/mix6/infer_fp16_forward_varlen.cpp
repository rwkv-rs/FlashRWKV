// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "../../../validation.h"

#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

void tmix_mix6_forward_varlen_cuda(
    int batch_size,
    int total_tokens,
    int channels,
    int max_seqlen,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor metadata_status,
    std::vector<torch::Tensor>& outputs);
std::vector<torch::Tensor> tmix_mix6_add_layer_norm_forward_varlen_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor shift_state,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor state_indices,
    torch::Tensor metadata_status,
    double eps);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;
using flashrwkv2::validation::prepare_recurrent_metadata_cuda;

namespace {

void check_tensor(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
  check_cuda_contiguous(tensor, name);
  check_same_device(reference, tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
}

}  // namespace

std::vector<torch::Tensor> tmix_mix6_forward_varlen(
    torch::Tensor x,
    torch::Tensor shift_state_pool,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor cu_seqlens,
    torch::Tensor state_indices,
    int64_t max_seqlen,
    py::object validated_metadata) {
  check_tensor(x, x, "x");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,C]");
  const int64_t total_tokens = x.size(0);
  const int64_t channels = x.size(1);
  check_tensor(shift_state_pool, x, "shift_state_pool");
  TORCH_CHECK(
      shift_state_pool.dim() == 2 && shift_state_pool.size(0) > 0 &&
          shift_state_pool.size(1) == channels,
      "shift_state_pool must have shape [slots,C]");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&x_r, "x_r"},
           {&x_w, "x_w"}, {&x_k, "x_k"}, {&x_v, "x_v"},
           {&x_a, "x_a"}, {&x_g, "x_g"},
       }) {
    check_tensor(*item.first, x, item.second);
    TORCH_CHECK(item.first->dim() == 1 && item.first->size(0) == channels,
                item.second, " must have shape [C]");
  }
  check_cuda_contiguous(cu_seqlens, "cu_seqlens");
  check_cuda_contiguous(state_indices, "state_indices");
  check_same_device(x, cu_seqlens, "cu_seqlens");
  check_same_device(x, state_indices, "state_indices");
  TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt32 &&
                  state_indices.scalar_type() == torch::kInt32,
              "packed metadata must be int32");
  TORCH_CHECK(cu_seqlens.dim() == 1 && state_indices.dim() == 1 &&
                  state_indices.numel() > 0 &&
                  cu_seqlens.numel() == state_indices.numel() + 1,
              "cu_seqlens must have shape [B+1] and state_indices [B]");
  TORCH_CHECK(channels % 2 == 0, "TMix mix6 requires an even channel count");
  const int batch_size = static_cast<int>(state_indices.numel());
  const int tokens = static_cast<int>(total_tokens);
  torch::Tensor launch_query_start_loc = cu_seqlens;
  torch::Tensor launch_state_indices = state_indices;
  torch::Tensor metadata_status;
  if (!validated_metadata.is_none()) {
    // Reuse the same identity/version/stream-bound ticket as WKV.  This keeps
    // shift-state operators on the scheduler's prepared metadata path rather
    // than rebuilding snapshots on every launch.
    validated_metadata.attr("_check_compatible")(
        cu_seqlens, state_indices, total_tokens, shift_state_pool.size(0),
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
    if (max_seqlen <= 0) {
      max_seqlen = 1;
    }
    auto prepared = prepare_recurrent_metadata_cuda(
        cu_seqlens, state_indices, total_tokens, shift_state_pool.size(0));
    launch_query_start_loc = std::move(prepared.query_start_loc);
    launch_state_indices = std::move(prepared.state_indices);
    metadata_status = std::move(prepared.status);
  }
  std::vector<torch::Tensor> outputs;
  outputs.reserve(6);
  for (int index = 0; index < 6; ++index) {
    outputs.push_back(torch::empty_like(x));
  }
  tmix_mix6_forward_varlen_cuda(
      batch_size, tokens, static_cast<int>(channels), static_cast<int>(max_seqlen),
      x, shift_state_pool, x_r, x_w, x_k, x_v, x_a, x_g,
      launch_query_start_loc, launch_state_indices, metadata_status, outputs);
  return outputs;
}

std::vector<torch::Tensor> tmix_mix6_add_layer_norm_forward_varlen(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor shift_state_pool,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor cu_seqlens,
    torch::Tensor state_indices,
    int64_t max_seqlen,
    double eps,
    py::object validated_metadata) {
  check_tensor(x, x, "x");
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) > 0 && x.size(1) == 4096,
      "canonical Albatross fused TMix requires packed shape [B,4096]");
  const int64_t total_tokens = x.size(0);
  const int64_t channels = x.size(1);
  check_tensor(residual, x, "residual");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual shape mismatch");
  check_tensor(shift_state_pool, x, "shift_state_pool");
  TORCH_CHECK(
      shift_state_pool.dim() == 2 && shift_state_pool.size(0) > 0 &&
          shift_state_pool.size(1) == channels,
      "shift_state_pool must have shape [slots,4096]");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&weight, "weight"},
           {&bias, "bias"}, {&x_r, "x_r"}, {&x_w, "x_w"},
           {&x_k, "x_k"}, {&x_v, "x_v"}, {&x_a, "x_a"}, {&x_g, "x_g"},
       }) {
    check_tensor(*item.first, x, item.second);
    TORCH_CHECK(
        item.first->dim() == 1 && item.first->size(0) == channels,
        item.second, " must have shape [4096]");
  }
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
  check_cuda_contiguous(cu_seqlens, "cu_seqlens");
  check_cuda_contiguous(state_indices, "state_indices");
  check_same_device(x, cu_seqlens, "cu_seqlens");
  check_same_device(x, state_indices, "state_indices");
  TORCH_CHECK(
      cu_seqlens.scalar_type() == torch::kInt32 &&
          state_indices.scalar_type() == torch::kInt32 &&
          cu_seqlens.dim() == 1 && state_indices.dim() == 1 &&
          state_indices.numel() > 0 &&
          cu_seqlens.numel() == state_indices.numel() + 1,
      "invalid packed metadata");
  const int64_t batch_size = state_indices.numel();
  TORCH_CHECK(
      total_tokens == batch_size,
      "canonical Albatross fused TMix requires one packed token per sequence");
  TORCH_CHECK(
      batch_size == 1,
      "canonical Albatross fused TMix is selected only for B=1,T=1");

  torch::Tensor launch_query_start_loc = cu_seqlens;
  torch::Tensor launch_state_indices = state_indices;
  torch::Tensor metadata_status;
  if (!validated_metadata.is_none()) {
    validated_metadata.attr("_check_compatible")(
        cu_seqlens, state_indices, total_tokens, shift_state_pool.size(0),
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
    if (max_seqlen <= 0) {
      max_seqlen = 1;
    }
    TORCH_CHECK(max_seqlen == 1, "canonical Albatross fused TMix requires max_seqlen=1");
    auto prepared = prepare_recurrent_metadata_cuda(
        cu_seqlens, state_indices, total_tokens, shift_state_pool.size(0));
    launch_query_start_loc = std::move(prepared.query_start_loc);
    launch_state_indices = std::move(prepared.state_indices);
    metadata_status = std::move(prepared.status);
  }
  TORCH_CHECK(max_seqlen == 1, "canonical Albatross fused TMix requires max_seqlen=1");
  (void)launch_query_start_loc;
  return tmix_mix6_add_layer_norm_forward_varlen_cuda(
      x, residual, shift_state_pool, weight, bias, x_r, x_w, x_k, x_v, x_a,
      x_g, launch_state_indices, metadata_status, eps);
}

void register_tmix_mix6_bindings(py::module_& module) {
  module.def(
      "tmix_mix6_forward_varlen", &tmix_mix6_forward_varlen,
      "Packed Albatross TMix mix6 with shift-state pool",
      py::arg("x"), py::arg("shift_state_pool"), py::arg("x_r"),
      py::arg("x_w"), py::arg("x_k"), py::arg("x_v"), py::arg("x_a"),
      py::arg("x_g"), py::arg("cu_seqlens"), py::arg("state_indices"),
      py::arg("max_seqlen") = -1,
      py::arg("validated_metadata") = py::none());
  module.def(
      "tmix_mix6_add_layer_norm_forward_varlen",
      &tmix_mix6_add_layer_norm_forward_varlen,
      "Packed Albatross fused TMix add-layer-norm mix6 with shift-state pool",
      py::arg("x"), py::arg("residual"), py::arg("shift_state_pool"),
      py::arg("weight"), py::arg("bias"), py::arg("x_r"), py::arg("x_w"),
      py::arg("x_k"), py::arg("x_v"), py::arg("x_a"), py::arg("x_g"),
      py::arg("cu_seqlens"), py::arg("state_indices"),
      py::arg("max_seqlen") = -1, py::arg("eps") = 1.0e-5,
      py::arg("validated_metadata") = py::none());
}
