// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV project
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "../../../validation.h"

#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

void cmix_mix_forward_varlen_cuda(
    int batch_size,
    int total_tokens,
    int channels,
    int max_seqlen,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor output,
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor metadata_status);
std::vector<torch::Tensor> cmix_add_layer_norm_mix_forward_varlen_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor shift_state,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor x_k,
    torch::Tensor state_indices,
    torch::Tensor metadata_status,
    double eps);
torch::Tensor cmix_relu_square_forward_varlen_cuda(torch::Tensor x);
torch::Tensor cmix_linear_ffn_down_forward_varlen_cuda(
    torch::Tensor x, torch::Tensor weight);

using flash_rwkv::validation::check_cuda_contiguous;
using flash_rwkv::validation::check_same_device;
using flash_rwkv::validation::prepare_recurrent_metadata_cuda;

namespace {

void check_half(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
  check_cuda_contiguous(tensor, name);
  check_same_device(reference, tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
}

}  // namespace

torch::Tensor cmix_mix_forward_varlen(
    torch::Tensor x,
    torch::Tensor shift_state_pool,
    torch::Tensor x_k,
    torch::Tensor cu_seqlens,
    torch::Tensor state_indices,
    int64_t max_seqlen,
    py::object validated_metadata) {
  check_half(x, x, "x");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,C]");
  const int64_t total_tokens = x.size(0);
  const int64_t channels = x.size(1);
  check_half(shift_state_pool, x, "shift_state_pool");
  TORCH_CHECK(shift_state_pool.dim() == 2 && shift_state_pool.size(0) > 0 &&
                  shift_state_pool.size(1) == channels,
              "shift_state_pool must have shape [slots,C]");
  check_half(x_k, x, "x_k");
  TORCH_CHECK(x_k.dim() == 1 && x_k.size(0) == channels,
              "x_k must have shape [C]");
  check_cuda_contiguous(cu_seqlens, "cu_seqlens");
  check_cuda_contiguous(state_indices, "state_indices");
  check_same_device(x, cu_seqlens, "cu_seqlens");
  check_same_device(x, state_indices, "state_indices");
  TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt32 &&
                  state_indices.scalar_type() == torch::kInt32 &&
                  cu_seqlens.dim() == 1 && state_indices.dim() == 1 &&
                  state_indices.numel() > 0 &&
                  cu_seqlens.numel() == state_indices.numel() + 1,
              "invalid packed metadata");
  TORCH_CHECK(channels % 2 == 0, "CMix mix requires an even channel count");
  const int batch_size = static_cast<int>(state_indices.numel());
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
    auto prepared = prepare_recurrent_metadata_cuda(
        cu_seqlens, state_indices, total_tokens, shift_state_pool.size(0));
    launch_query_start_loc = std::move(prepared.query_start_loc);
    launch_state_indices = std::move(prepared.state_indices);
    metadata_status = std::move(prepared.status);
  }
  auto output = torch::empty_like(x);
  cmix_mix_forward_varlen_cuda(
      batch_size, static_cast<int>(total_tokens), static_cast<int>(channels),
      static_cast<int>(max_seqlen), x, shift_state_pool, x_k, output,
      launch_query_start_loc, launch_state_indices, metadata_status);
  return output;
}

std::vector<torch::Tensor> cmix_add_layer_norm_mix_forward_varlen(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor shift_state_pool,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor x_k,
    torch::Tensor cu_seqlens,
    torch::Tensor state_indices,
    int64_t max_seqlen,
    double eps,
    py::object validated_metadata) {
  check_half(x, x, "x");
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) > 0 && x.size(1) == 4096,
      "canonical Albatross fused CMix requires packed shape [B,4096]");
  const int64_t total_tokens = x.size(0);
  const int64_t channels = x.size(1);
  check_half(residual, x, "residual");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual shape mismatch");
  check_half(shift_state_pool, x, "shift_state_pool");
  TORCH_CHECK(
      shift_state_pool.dim() == 2 && shift_state_pool.size(0) > 0 &&
          shift_state_pool.size(1) == channels,
      "shift_state_pool must have shape [slots,4096]");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&weight, "weight"},
           {&bias, "bias"},
           {&x_k, "x_k"},
       }) {
    check_half(*item.first, x, item.second);
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
      "canonical Albatross fused CMix requires one packed token per sequence");

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
    TORCH_CHECK(max_seqlen == 1, "canonical Albatross fused CMix requires max_seqlen=1");
    auto prepared = prepare_recurrent_metadata_cuda(
        cu_seqlens, state_indices, total_tokens, shift_state_pool.size(0));
    launch_query_start_loc = std::move(prepared.query_start_loc);
    launch_state_indices = std::move(prepared.state_indices);
    metadata_status = std::move(prepared.status);
  }
  TORCH_CHECK(max_seqlen == 1, "canonical Albatross fused CMix requires max_seqlen=1");
  (void)launch_query_start_loc;
  return cmix_add_layer_norm_mix_forward_varlen_cuda(
      x, residual, shift_state_pool, weight, bias, x_k, launch_state_indices,
      metadata_status, eps);
}

torch::Tensor cmix_relu_square_forward_varlen(torch::Tensor x) {
  check_half(x, x, "x");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,features]");
  TORCH_CHECK((x.numel() % 2) == 0,
              "cmix relu-square requires an even number of elements");
  return cmix_relu_square_forward_varlen_cuda(x);
}

torch::Tensor cmix_linear_ffn_down_forward_varlen(
    torch::Tensor x, torch::Tensor weight) {
  check_half(x, x, "x");
  check_half(weight, x, "weight");
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0 &&
          weight.dim() == 2 && weight.size(0) == x.size(1) &&
          weight.size(1) > 0,
      "CMix FFN down linear expects x [rows,K] and runtime weight [K,C]");
  return cmix_linear_ffn_down_forward_varlen_cuda(x, weight);
}

void register_cmix_mix_bindings(py::module_& module) {
  module.def(
      "cmix_mix_forward_varlen", &cmix_mix_forward_varlen,
      "Packed Albatross CMix mix with shift-state pool",
      py::arg("x"), py::arg("shift_state_pool"), py::arg("x_k"),
      py::arg("cu_seqlens"), py::arg("state_indices"),
      py::arg("max_seqlen") = -1,
      py::arg("validated_metadata") = py::none());
  module.def(
      "cmix_add_layer_norm_mix_forward_varlen",
      &cmix_add_layer_norm_mix_forward_varlen,
      "Packed Albatross fused CMix add-layer-norm mix with shift-state pool",
      py::arg("x"), py::arg("residual"), py::arg("shift_state_pool"),
      py::arg("weight"), py::arg("bias"), py::arg("x_k"),
      py::arg("cu_seqlens"), py::arg("state_indices"),
      py::arg("max_seqlen") = -1, py::arg("eps") = 1.0e-5,
      py::arg("validated_metadata") = py::none());
  module.def(
      "cmix_relu_square_forward_varlen",
      &cmix_relu_square_forward_varlen,
      "Packed Albatross CMix ReLU-square activation",
      py::arg("x"));
  module.def(
      "cmix_linear_ffn_down_forward_varlen",
      &cmix_linear_ffn_down_forward_varlen,
      "Packed Albatross CMix FFN down linear caller path",
      py::arg("x"), py::arg("weight"));
}
