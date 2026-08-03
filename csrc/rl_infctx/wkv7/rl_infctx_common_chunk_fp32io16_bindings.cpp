// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV project

#include "../../bindings.h"
#include "../../validation.h"

#include <optional>
#include <utility>
#include <vector>

void materialized_chunk_fp32_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor transform,
    torch::Tensor bias,
    torch::Tensor boundary,
    torch::Tensor state_dot_a,
    int64_t build_warps,
    int64_t stages,
    int64_t state_tile,
    double scale);
void materialized_chunk_fp32_from_decay_logits_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor decay_bias,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor transform,
    torch::Tensor bias,
    torch::Tensor boundary,
    torch::Tensor state_dot_a,
    int64_t build_warps,
    int64_t stages,
    int64_t state_tile,
    double scale);
void recompute_chunk_fp32_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor boundary,
    double scale);
void recompute_chunk_fp32_from_decay_logits_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor decay_bias,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor boundary,
    double scale);

using flash_rwkv::validation::check_chunk_metadata;
using flash_rwkv::validation::check_cuda_contiguous;
using flash_rwkv::validation::check_recurrent_layout;
using flash_rwkv::validation::check_same_device;
using flash_rwkv::validation::kHeadSize;

void materialized_chunk_fp32_impl(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor transform,
    torch::Tensor bias,
    torch::Tensor boundary,
    int64_t build_warps,
    int64_t stages,
    int64_t state_tile,
    double scale,
    std::optional<torch::Tensor> state_dot_a,
    bool from_decay_logits,
    std::optional<torch::Tensor> decay_bias) {
  const auto dimensions = check_recurrent_layout(
      sequence_chunk_offsets,
      state_indices,
      state,
      r,
      decay,
      k,
      v,
      a,
      b,
      output,
      scale,
      kHeadSize);
  const int64_t num_chunks = check_chunk_metadata(
      chunk_token_starts,
      chunk_token_ends,
      state,
      dimensions);
  check_cuda_contiguous(transform, "transform");
  check_cuda_contiguous(bias, "bias");
  check_cuda_contiguous(boundary, "boundary");

  const std::vector<int64_t> workspace_shape{
      num_chunks,
      dimensions.num_heads,
      kHeadSize,
      kHeadSize,
  };
  TORCH_CHECK(
      transform.sizes().vec() == workspace_shape &&
          bias.sizes().vec() == workspace_shape &&
          boundary.sizes().vec() == workspace_shape,
      "transform, bias, and boundary must have shape [C,H,64,64]");
  TORCH_CHECK(
      transform.scalar_type() == torch::kFloat32 &&
          bias.scalar_type() == torch::kFloat32 &&
          boundary.scalar_type() == torch::kFloat32,
      "chunk workspaces must be fp32");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be fp32");
  TORCH_CHECK(
      r.scalar_type() == torch::kFloat16 ||
          r.scalar_type() == torch::kBFloat16 ||
          r.scalar_type() == torch::kFloat32,
      "FP32-state token tensors must be fp16, bf16, or fp32");
  TORCH_CHECK(
      (build_warps == 2 && stages == 1) ||
          (build_warps == 4 && (stages == 1 || stages == 2)),
      "chunk build config must be (warps,stages) in "
      "{(2,1),(4,1),(4,2)}");
  TORCH_CHECK(
      state_tile == 16 || state_tile == 32 || state_tile == 64,
      "chunk state_tile must be 16, 32, or 64");

  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &transform, "transform"},
           {&bias, "bias"},
           {&boundary, "boundary"},
       }) {
    check_same_device(state, *item.first, item.second);
  }
  if (state_dot_a.has_value()) {
    check_cuda_contiguous(*state_dot_a, "state_dot_a");
    check_same_device(state, *state_dot_a, "state_dot_a");
    TORCH_CHECK(
        state_dot_a->sizes() == r.sizes(),
        "state_dot_a must match the token tensor shape");
    TORCH_CHECK(
        state_dot_a->scalar_type() == torch::kFloat32,
        "state_dot_a must be fp32");
  }
  if (decay_bias.has_value()) {
    check_cuda_contiguous(*decay_bias, "decay_bias");
    check_same_device(state, *decay_bias, "decay_bias");
    TORCH_CHECK(
        from_decay_logits,
        "decay_bias is valid only for the raw decay_logits path");
    TORCH_CHECK(
        decay_bias->scalar_type() == r.scalar_type(),
        "decay_bias must match the token tensor dtype");
    TORCH_CHECK(
        (decay_bias->dim() == 1 &&
         decay_bias->numel() == dimensions.num_heads * kHeadSize) ||
            (decay_bias->dim() == 2 &&
             decay_bias->size(0) == dimensions.num_heads &&
             decay_bias->size(1) == kHeadSize),
        "decay_bias must have shape [H*64] or [H,64]");
  }

  if (from_decay_logits) {
    materialized_chunk_fp32_from_decay_logits_cuda(
        sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
        state_indices, state, r, decay, decay_bias.value_or(torch::Tensor()),
        k, v, a, b, output, transform, bias, boundary,
        state_dot_a.value_or(torch::Tensor()), build_warps, stages, state_tile,
        scale);
  } else {
    materialized_chunk_fp32_cuda(
        sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
        state_indices, state, r, decay, k, v, a, b, output, transform, bias,
        boundary, state_dot_a.value_or(torch::Tensor()), build_warps, stages,
        state_tile, scale);
  }
}

void materialized_chunk_fp32(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor transform,
    torch::Tensor bias,
    torch::Tensor boundary,
    int64_t build_warps,
    int64_t stages,
    int64_t state_tile,
    double scale,
    std::optional<torch::Tensor> state_dot_a) {
  materialized_chunk_fp32_impl(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
      state_indices, state, r, log_decay, k, v, a, b, output, transform,
      bias, boundary, build_warps, stages, state_tile, scale, state_dot_a,
      false, std::nullopt);
}

void materialized_chunk_fp32_from_decay_logits(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor transform,
    torch::Tensor bias,
    torch::Tensor boundary,
    int64_t build_warps,
    int64_t stages,
    int64_t state_tile,
    double scale,
    std::optional<torch::Tensor> state_dot_a,
    std::optional<torch::Tensor> decay_bias) {
  materialized_chunk_fp32_impl(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
      state_indices, state, r, decay_logits, k, v, a, b, output, transform,
      bias, boundary, build_warps, stages, state_tile, scale, state_dot_a,
      true, decay_bias);
}

void recompute_chunk_fp32_impl(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor boundary,
    double scale,
    bool from_decay_logits,
    std::optional<torch::Tensor> decay_bias) {
  const auto dimensions = check_recurrent_layout(
      sequence_chunk_offsets,
      state_indices,
      state,
      r,
      decay,
      k,
      v,
      a,
      b,
      output,
      scale,
      kHeadSize);
  const int64_t num_chunks = check_chunk_metadata(
      chunk_token_starts,
      chunk_token_ends,
      state,
      dimensions);
  check_cuda_contiguous(boundary, "boundary");

  const std::vector<int64_t> workspace_shape{
      num_chunks,
      dimensions.num_heads,
      kHeadSize,
      kHeadSize,
  };
  TORCH_CHECK(
      boundary.sizes().vec() == workspace_shape,
      "boundary must have shape [C,H,64,64]");
  TORCH_CHECK(
      boundary.scalar_type() == torch::kFloat32,
      "boundary must be fp32");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be fp32");
  TORCH_CHECK(
      r.scalar_type() == torch::kFloat16 ||
          r.scalar_type() == torch::kBFloat16 ||
          r.scalar_type() == torch::kFloat32,
      "FP32-state token tensors must be fp16, bf16, or fp32");
  check_same_device(state, boundary, "boundary");

  if (decay_bias.has_value()) {
    check_cuda_contiguous(*decay_bias, "decay_bias");
    check_same_device(state, *decay_bias, "decay_bias");
    TORCH_CHECK(
        from_decay_logits,
        "decay_bias is valid only for the raw decay_logits path");
    TORCH_CHECK(
        decay_bias->scalar_type() == r.scalar_type(),
        "decay_bias must match the token tensor dtype");
    TORCH_CHECK(
        (decay_bias->dim() == 1 &&
         decay_bias->numel() == dimensions.num_heads * kHeadSize) ||
            (decay_bias->dim() == 2 &&
             decay_bias->size(0) == dimensions.num_heads &&
             decay_bias->size(1) == kHeadSize),
        "decay_bias must have shape [H*64] or [H,64]");
  }

  if (from_decay_logits) {
    recompute_chunk_fp32_from_decay_logits_cuda(
        sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
        state_indices, state, r, decay,
        decay_bias.value_or(torch::Tensor()), k, v, a, b, output, boundary,
        scale);
  } else {
    recompute_chunk_fp32_cuda(
        sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
        state_indices, state, r, decay, k, v, a, b, output, boundary, scale);
  }
}

void recompute_chunk_fp32(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor boundary,
    double scale) {
  recompute_chunk_fp32_impl(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
      state_indices, state, r, log_decay, k, v, a, b, output, boundary, scale,
      false, std::nullopt);
}

void recompute_chunk_fp32_from_decay_logits(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor boundary,
    double scale,
    std::optional<torch::Tensor> decay_bias) {
  recompute_chunk_fp32_impl(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
      state_indices, state, r, decay_logits, k, v, a, b, output, boundary,
      scale, true, decay_bias);
}

void register_rl_infctx_experimental_bindings(py::module_& module) {
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
      "materialized_chunk_fp32_from_decay_logits",
      &materialized_chunk_fp32_from_decay_logits,
      "FlashRWKV materialized chunk forward with fused raw decay logits",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("state_indices"), py::arg("state"),
      py::arg("r"), py::arg("decay_logits"), py::arg("k"), py::arg("v"),
      py::arg("a"), py::arg("b"), py::arg("output"), py::arg("transform"),
      py::arg("bias"), py::arg("boundary"), py::arg("build_warps"),
      py::arg("stages"), py::arg("state_tile"), py::arg("scale"),
      py::arg("state_dot_a") = py::none(),
      py::arg("decay_bias") = py::none());
  module.def(
      "recompute_chunk_fp32", &recompute_chunk_fp32,
      "FlashRWKV DPLR-factor recompute chunk forward with FP32 state",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("state_indices"), py::arg("state"),
      py::arg("r"), py::arg("log_decay"), py::arg("k"), py::arg("v"),
      py::arg("a"), py::arg("b"), py::arg("output"), py::arg("boundary"),
      py::arg("scale"));
  module.def(
      "recompute_chunk_fp32_from_decay_logits",
      &recompute_chunk_fp32_from_decay_logits,
      "FlashRWKV factor-recompute chunk with fused raw decay logits",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("state_indices"), py::arg("state"),
      py::arg("r"), py::arg("decay_logits"), py::arg("k"), py::arg("v"),
      py::arg("a"), py::arg("b"), py::arg("output"), py::arg("boundary"),
      py::arg("scale"), py::arg("decay_bias") = py::none());
}
