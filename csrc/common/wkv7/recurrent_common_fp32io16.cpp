// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the RWKV-LM project
// Adapted from RWKV-LM at commit
// 952102498e9ed367ea0a59ee64106916d474d30f.

#include "recurrent_common_fp32io16.h"
#include "../../validation.h"

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

void recurrent_common_fp32io16_forward_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor boundary,
    torch::Tensor state_dot_a,
    double scale);
void recurrent_common_fp32io16_backward_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor final_state,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor state_dot_a,
    torch::Tensor grad_output,
    torch::Tensor grad_final_state,
    torch::Tensor boundary,
    torch::Tensor grad_r,
    torch::Tensor grad_log_decay,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_b,
    torch::Tensor grad_initial_state,
    double scale);

using flash_rwkv::validation::check_chunk_metadata;
using flash_rwkv::validation::check_cuda_contiguous;
using flash_rwkv::validation::check_optional_like;
using flash_rwkv::validation::check_same_device;
using flash_rwkv::validation::kHeadSize;
using flash_rwkv::validation::RecurrentDimensions;

void recurrent_common_fp32io16_forward(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor boundary,
    torch::Tensor state_dot_a,
    double scale) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &sequence_chunk_offsets, "sequence_chunk_offsets"},
           {&chunk_token_starts, "chunk_token_starts"},
           {&chunk_token_ends, "chunk_token_ends"},
           {&state, "state"},
           {&r, "r"},
           {&log_decay, "log_decay"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&output, "output"},
           {&boundary, "boundary"},
           {&state_dot_a, "state_dot_a"},
       }) {
    check_cuda_contiguous(*item.first, item.second);
  }
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");
  TORCH_CHECK(
      sequence_chunk_offsets.scalar_type() == torch::kInt32 &&
          chunk_token_starts.scalar_type() == torch::kInt32 &&
          chunk_token_ends.scalar_type() == torch::kInt32,
      "pretrain chunk metadata must be int32");
  TORCH_CHECK(
      state.dim() == 4 && state.size(0) > 0 &&
          state.size(1) > 0 &&
          state.size(2) == kHeadSize &&
          state.size(3) == kHeadSize,
      "state must have shape [B,H,64,64]");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be fp32");

  const int64_t num_sequences = state.size(0);
  const int64_t num_heads = state.size(1);
  TORCH_CHECK(
      sequence_chunk_offsets.dim() == 1 &&
          sequence_chunk_offsets.numel() == num_sequences + 1,
      "sequence_chunk_offsets must have shape [B+1]");
  const RecurrentDimensions dimensions{num_sequences, num_heads};
  const int64_t num_chunks = check_chunk_metadata(
      chunk_token_starts,
      chunk_token_ends,
      state,
      dimensions);
  TORCH_CHECK(
      r.dim() == 3 && r.size(0) > 0 &&
          r.size(1) == num_heads &&
          r.size(2) == kHeadSize,
      "r must have shape [B*T,H,64]");
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
  TORCH_CHECK(
      r.scalar_type() == torch::kFloat16 ||
          r.scalar_type() == torch::kBFloat16,
      "fp32io16 token tensors must be fp16 or bf16");

  const std::vector<int64_t> boundary_shape{
      num_chunks,
      num_heads,
      kHeadSize,
      kHeadSize,
  };
  TORCH_CHECK(
      boundary.sizes().vec() == boundary_shape &&
          boundary.scalar_type() == torch::kFloat32,
      "boundary must be fp32 with shape [C,H,64,64]");
  TORCH_CHECK(
      state_dot_a.sizes() == r.sizes() &&
          state_dot_a.scalar_type() == torch::kFloat32,
      "state_dot_a must be fp32 and match the token tensor shape");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &sequence_chunk_offsets, "sequence_chunk_offsets"},
           {&chunk_token_starts, "chunk_token_starts"},
           {&chunk_token_ends, "chunk_token_ends"},
           {&r, "r"},
           {&log_decay, "log_decay"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&output, "output"},
           {&boundary, "boundary"},
           {&state_dot_a, "state_dot_a"},
       }) {
    check_same_device(state, *item.first, item.second);
  }

  recurrent_common_fp32io16_forward_cuda(
      sequence_chunk_offsets,
      chunk_token_starts,
      chunk_token_ends,
      state,
      r,
      log_decay,
      k,
      v,
      a,
      b,
      output,
      boundary,
      state_dot_a,
      scale);
}

void recurrent_common_fp32io16_backward(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor final_state,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor state_dot_a,
    std::optional<torch::Tensor> grad_output,
    std::optional<torch::Tensor> grad_final_state,
    torch::Tensor boundary,
    std::optional<torch::Tensor> grad_r,
    std::optional<torch::Tensor> grad_log_decay,
    std::optional<torch::Tensor> grad_k,
    std::optional<torch::Tensor> grad_v,
    std::optional<torch::Tensor> grad_a,
    std::optional<torch::Tensor> grad_b,
    std::optional<torch::Tensor> grad_initial_state,
    double scale) {
  check_cuda_contiguous(sequence_chunk_offsets, "sequence_chunk_offsets");
  check_cuda_contiguous(final_state, "final_state");
  check_cuda_contiguous(r, "r");
  check_cuda_contiguous(log_decay, "log_decay");
  check_cuda_contiguous(k, "k");
  check_cuda_contiguous(v, "v");
  check_cuda_contiguous(a, "a");
  check_cuda_contiguous(b, "b");
  check_cuda_contiguous(state_dot_a, "state_dot_a");
  check_cuda_contiguous(boundary, "boundary");

  TORCH_CHECK(
      std::isfinite(scale),
      "scale must be finite");
  TORCH_CHECK(
      sequence_chunk_offsets.scalar_type() == torch::kInt32,
      "sequence_chunk_offsets must be int32");
  TORCH_CHECK(
      final_state.dim() == 4 && final_state.size(0) > 0 &&
          final_state.size(1) > 0 &&
          final_state.size(2) == kHeadSize &&
          final_state.size(3) == kHeadSize,
      "final_state must have shape [B,H,64,64]");
  TORCH_CHECK(
      final_state.scalar_type() == torch::kFloat32,
      "final_state must be fp32");

  const int64_t num_sequences = final_state.size(0);
  const int64_t num_heads = final_state.size(1);
  TORCH_CHECK(
      sequence_chunk_offsets.dim() == 1 &&
          sequence_chunk_offsets.numel() == num_sequences + 1,
      "sequence_chunk_offsets must have shape [B+1]");
  const RecurrentDimensions dimensions{num_sequences, num_heads};
  const int64_t num_chunks = check_chunk_metadata(
      chunk_token_starts,
      chunk_token_ends,
      final_state,
      dimensions);

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
          r.sizes() == b.sizes(),
      "r,log_decay,k,v,a,b shape mismatch");
  TORCH_CHECK(
      r.scalar_type() == log_decay.scalar_type() &&
          r.scalar_type() == k.scalar_type() &&
          r.scalar_type() == v.scalar_type() &&
          r.scalar_type() == a.scalar_type() &&
          r.scalar_type() == b.scalar_type(),
      "r,log_decay,k,v,a,b dtype mismatch");
  TORCH_CHECK(
      r.scalar_type() == torch::kFloat16 ||
          r.scalar_type() == torch::kBFloat16,
      "fp32io16 token tensors must be fp16 or bf16");
  TORCH_CHECK(
      state_dot_a.sizes() == r.sizes() &&
          state_dot_a.scalar_type() == torch::kFloat32,
      "state_dot_a must be fp32 and match the token tensor shape");
  const std::vector<int64_t> boundary_shape{
      num_chunks,
      num_heads,
      kHeadSize,
      kHeadSize,
  };
  TORCH_CHECK(
      boundary.sizes().vec() == boundary_shape &&
          boundary.scalar_type() == torch::kFloat32,
      "boundary must be fp32 with shape [C,H,64,64]");
  TORCH_CHECK(
      grad_output.has_value() || grad_final_state.has_value(),
      "at least one upstream gradient must be provided");

  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &sequence_chunk_offsets, "sequence_chunk_offsets"},
           {&r, "r"},
           {&log_decay, "log_decay"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&state_dot_a, "state_dot_a"},
           {&boundary, "boundary"},
       }) {
    check_same_device(final_state, *item.first, item.second);
  }
  check_optional_like(grad_output, r, "grad_output");
  check_optional_like(
      grad_final_state,
      final_state,
      "grad_final_state");
  check_optional_like(grad_r, r, "grad_r");
  check_optional_like(
      grad_log_decay,
      r,
      "grad_log_decay");
  check_optional_like(grad_k, r, "grad_k");
  check_optional_like(grad_v, r, "grad_v");
  check_optional_like(grad_a, r, "grad_a");
  check_optional_like(grad_b, r, "grad_b");
  check_optional_like(
      grad_initial_state,
      final_state,
      "grad_initial_state");

  recurrent_common_fp32io16_backward_cuda(
      sequence_chunk_offsets,
      chunk_token_starts,
      chunk_token_ends,
      final_state,
      r,
      log_decay,
      k,
      v,
      a,
      b,
      state_dot_a,
      grad_output.value_or(torch::Tensor()),
      grad_final_state.value_or(torch::Tensor()),
      boundary,
      grad_r.value_or(torch::Tensor()),
      grad_log_decay.value_or(torch::Tensor()),
      grad_k.value_or(torch::Tensor()),
      grad_v.value_or(torch::Tensor()),
      grad_a.value_or(torch::Tensor()),
      grad_b.value_or(torch::Tensor()),
      grad_initial_state.value_or(torch::Tensor()),
      scale);
}
