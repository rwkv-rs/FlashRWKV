// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
//
// Source: BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_fast_ops_fp16.cu,
// revision ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "../../../validation.h"

#include <torch/extension.h>

#include <cstdint>
#include <utility>

torch::Tensor cmix_sparse_up_forward_varlen_cuda(
    int batch_size,
    int total_tokens,
    int channels,
    int features,
    int max_seqlen,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor metadata_status);
torch::Tensor cmix_sparse_down_relu_forward_varlen_cuda(
    torch::Tensor preact,
    torch::Tensor value_fc,
    int64_t batch_size,
    int64_t max_seqlen,
    bool deterministic);
torch::Tensor cmix_sparse_forward_varlen_cuda(
    int batch_size,
    int total_tokens,
    int channels,
    int features,
    int max_seqlen,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor value_fc,
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor metadata_status,
    bool deterministic);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;
using flashrwkv2::validation::prepare_recurrent_metadata_cuda;

namespace {

constexpr int64_t kMaxGridDimYZ = 65535;

void check_sparse_grid_rows(
    int64_t rows,
    const char* operator_name,
    const char* grid_dimension) {
  TORCH_CHECK(
      rows <= kMaxGridDimYZ,
      operator_name,
      " supports at most ",
      kMaxGridDimYZ,
      " packed rows because rows map to CUDA ",
      grid_dimension,
      "; got ",
      rows);
}

void check_half(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
  check_cuda_contiguous(tensor, name);
  check_same_device(reference, tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
}

void check_packed_metadata(
    const torch::Tensor& x,
    const torch::Tensor& shift_state,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& state_indices) {
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
      "cu_seqlens must have shape [B+1] and state_indices [B]");
  TORCH_CHECK(shift_state.size(0) > 0, "shift_state_pool must not be empty");
}

struct LaunchMetadata {
  torch::Tensor query_start_loc;
  torch::Tensor state_indices;
  torch::Tensor status;
  int64_t max_seqlen;
};

LaunchMetadata prepare_launch_metadata(
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& state_indices,
    int64_t total_tokens,
    int64_t state_pool_size,
    int64_t max_seqlen,
    py::object validated_metadata) {
  if (!validated_metadata.is_none()) {
    validated_metadata.attr("_check_compatible")(
        cu_seqlens,
        state_indices,
        total_tokens,
        state_pool_size,
        max_seqlen);
    return LaunchMetadata{
        validated_metadata.attr("_query_start_loc_snapshot")()
            .cast<torch::Tensor>(),
        validated_metadata.attr("_state_indices_snapshot")()
            .cast<torch::Tensor>(),
        validated_metadata.attr("_status")().cast<torch::Tensor>(),
        validated_metadata.attr("_max_seqlen")().cast<int64_t>()};
  }

  // Python callers prepare the reusable ticket before entering this binding.
  // A direct low-level caller without a ticket uses the conservative packed
  // upper bound, which never selects the B=1,T=1 fast path incorrectly.
  if (max_seqlen <= 0) {
    max_seqlen = total_tokens;
  }
  auto prepared = prepare_recurrent_metadata_cuda(
      cu_seqlens, state_indices, total_tokens, state_pool_size);
  return LaunchMetadata{
      std::move(prepared.query_start_loc),
      std::move(prepared.state_indices),
      std::move(prepared.status),
      max_seqlen};
}

}  // namespace

torch::Tensor cmix_sparse_up_forward_varlen(
    torch::Tensor x,
    torch::Tensor shift_state_pool,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor cu_seqlens,
    torch::Tensor state_indices,
    int64_t max_seqlen,
    py::object validated_metadata) {
  check_half(x, x, "x");
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
      "x must have packed shape [total_tokens,C]");
  check_sparse_grid_rows(x.size(0), "cmix sparse up", "grid.y");
  check_half(shift_state_pool, x, "shift_state_pool");
  TORCH_CHECK(
      shift_state_pool.dim() == 2 &&
          shift_state_pool.size(1) == x.size(1),
      "shift_state_pool must have shape [slots,C]");
  check_half(x_k, x, "x_k");
  TORCH_CHECK(x_k.dim() == 1 && x_k.size(0) == x.size(1),
              "x_k must have shape [C]");
  check_half(key_fc, x, "key_fc");
  TORCH_CHECK(
      key_fc.dim() == 2 && key_fc.size(1) == x.size(1),
      "key_fc must have shape [F,C]");
  TORCH_CHECK(x.size(1) % 8 == 0,
              "CMix sparse shift-state channels must be divisible by 8");
  check_packed_metadata(x, shift_state_pool, cu_seqlens, state_indices);
  const auto metadata = prepare_launch_metadata(
      cu_seqlens,
      state_indices,
      x.size(0),
      shift_state_pool.size(0),
      max_seqlen,
      validated_metadata);
  return cmix_sparse_up_forward_varlen_cuda(
      static_cast<int>(state_indices.numel()),
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)),
      static_cast<int>(key_fc.size(0)),
      static_cast<int>(metadata.max_seqlen),
      x,
      shift_state_pool,
      x_k,
      key_fc,
      metadata.query_start_loc,
      metadata.state_indices,
      metadata.status);
}

torch::Tensor cmix_sparse_down_relu_forward_varlen(
    torch::Tensor preact,
    torch::Tensor value_fc,
    int64_t batch_size,
    int64_t max_seqlen,
    bool deterministic) {
  check_half(preact, preact, "preact");
  check_half(value_fc, preact, "value_fc");
  TORCH_CHECK(
      preact.dim() == 2 && preact.size(0) > 0 && preact.size(1) > 0,
      "preact must have packed shape [total_tokens,F]");
  check_sparse_grid_rows(
      preact.size(0), "cmix sparse down", "grid.y/grid.z");
  TORCH_CHECK(
      value_fc.dim() == 2 && value_fc.size(0) == preact.size(1) &&
          value_fc.size(1) > 0 && value_fc.size(1) % 8 == 0,
      "value_fc must have shape [F,C] with C divisible by 8");
  TORCH_CHECK(
      (batch_size == -1 && max_seqlen == -1) ||
          (batch_size > 0 && max_seqlen > 0),
      "batch_size and max_seqlen must both be omitted or positive");
  if (batch_size > 0) {
    TORCH_CHECK(
        batch_size <= INT32_MAX / max_seqlen &&
            batch_size * max_seqlen >= preact.size(0),
        "batch_size * max_seqlen must cover packed rows");
  }
  return cmix_sparse_down_relu_forward_varlen_cuda(
      preact, value_fc, batch_size, max_seqlen, deterministic);
}

torch::Tensor cmix_sparse_forward_varlen(
    torch::Tensor x,
    torch::Tensor shift_state_pool,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor value_fc,
    torch::Tensor cu_seqlens,
    torch::Tensor state_indices,
    int64_t max_seqlen,
    py::object validated_metadata,
    bool deterministic) {
  check_half(x, x, "x");
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
      "x must have packed shape [total_tokens,C]");
  check_sparse_grid_rows(
      x.size(0), "cmix sparse combined", "grid.y/grid.z");
  check_half(shift_state_pool, x, "shift_state_pool");
  TORCH_CHECK(
      shift_state_pool.dim() == 2 &&
          shift_state_pool.size(1) == x.size(1),
      "shift_state_pool must have shape [slots,C]");
  check_half(x_k, x, "x_k");
  TORCH_CHECK(x_k.dim() == 1 && x_k.size(0) == x.size(1),
              "x_k must have shape [C]");
  check_half(key_fc, x, "key_fc");
  check_half(value_fc, x, "value_fc");
  TORCH_CHECK(
      key_fc.dim() == 2 && value_fc.dim() == 2 &&
          key_fc.size(1) == x.size(1) &&
          value_fc.size(0) == key_fc.size(0) &&
          value_fc.size(1) == x.size(1),
      "key_fc and value_fc must have shape [F,C]");
  TORCH_CHECK(
      x.size(1) % 256 == 0 && key_fc.size(0) % 128 == 0,
      "canonical sparse rows path requires C divisible by 256 and F by 128");
  check_packed_metadata(x, shift_state_pool, cu_seqlens, state_indices);
  const auto metadata = prepare_launch_metadata(
      cu_seqlens,
      state_indices,
      x.size(0),
      shift_state_pool.size(0),
      max_seqlen,
      validated_metadata);
  return cmix_sparse_forward_varlen_cuda(
      static_cast<int>(state_indices.numel()),
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1)),
      static_cast<int>(key_fc.size(0)),
      static_cast<int>(metadata.max_seqlen),
      x,
      shift_state_pool,
      x_k,
      key_fc,
      value_fc,
      metadata.query_start_loc,
      metadata.state_indices,
      metadata.status,
      deterministic);
}

void register_cmix_sparse_bindings(py::module_& module) {
  module.def(
      "cmix_sparse_up_forward_varlen",
      &cmix_sparse_up_forward_varlen,
      "Packed Albatross CMix sparse up projection",
      py::arg("x"),
      py::arg("shift_state_pool"),
      py::arg("x_k"),
      py::arg("key_fc"),
      py::arg("cu_seqlens"),
      py::arg("state_indices"),
      py::arg("max_seqlen") = -1,
      py::arg("validated_metadata") = py::none());
  module.def(
      "cmix_sparse_down_relu_forward_varlen",
      &cmix_sparse_down_relu_forward_varlen,
      "Packed Albatross CMix sparse ReLU-square/down projection",
      py::arg("preact"),
      py::arg("value_fc"),
      py::arg("batch_size") = -1,
      py::arg("max_seqlen") = -1,
      py::arg("deterministic") = false);
  module.def(
      "cmix_sparse_forward_varlen",
      &cmix_sparse_forward_varlen,
      "Packed Albatross CMix sparse up/down path",
      py::arg("x"),
      py::arg("shift_state_pool"),
      py::arg("x_k"),
      py::arg("key_fc"),
      py::arg("value_fc"),
      py::arg("cu_seqlens"),
      py::arg("state_indices"),
      py::arg("max_seqlen") = -1,
      py::arg("validated_metadata") = py::none(),
      py::arg("deterministic") = false);
}
