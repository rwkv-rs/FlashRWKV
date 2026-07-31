// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vllm-rwkv rwkv7_wkv_fp32_v2 at commit
// 6d683f9e49a2997e405c47edc147872c8609513b for the FlashRWKV core contract.

#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

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

void pretrain_recurrent_fp32io16_forward_cuda(
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

void pretrain_recurrent_fp32io16_backward_cuda(
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

void infer_chunk_bf16_forward_k1_prepare_cuda(
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor chunk_transform,
    torch::Tensor chunk_bias,
    torch::Tensor token_transform,
    torch::Tensor token_bias,
    double scale);

void infer_chunk_bf16_forward_k2_recurrence_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state,
    torch::Tensor output,
    torch::Tensor chunk_transform,
    torch::Tensor chunk_bias,
    torch::Tensor token_transform,
    torch::Tensor token_bias);

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

void check_optional_like(
    const std::optional<torch::Tensor>& tensor,
    const torch::Tensor& reference,
    const char* name) {
  if (!tensor.has_value()) {
    return;
  }
  check_cuda_contiguous(*tensor, name);
  check_same_device(reference, *tensor, name);
  TORCH_CHECK(
      tensor->sizes() == reference.sizes(),
      name,
      " must match the reference shape");
  TORCH_CHECK(
      tensor->scalar_type() == reference.scalar_type(),
      name,
      " must match the reference dtype");
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

int64_t check_chunk_metadata(
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    const torch::Tensor& state,
    const RecurrentDimensions& dimensions) {
  check_cuda_contiguous(chunk_token_starts, "chunk_token_starts");
  check_cuda_contiguous(chunk_token_ends, "chunk_token_ends");
  TORCH_CHECK(
      chunk_token_starts.scalar_type() == torch::kInt32 &&
          chunk_token_ends.scalar_type() == torch::kInt32,
      "chunk token metadata must be int32");
  TORCH_CHECK(
      chunk_token_starts.dim() == 1 &&
          chunk_token_starts.numel() > 0 &&
          chunk_token_starts.sizes() == chunk_token_ends.sizes(),
      "chunk_token_starts and chunk_token_ends must have shape [C]");
  check_same_device(state, chunk_token_starts, "chunk_token_starts");
  check_same_device(state, chunk_token_ends, "chunk_token_ends");

  const int64_t num_chunks = chunk_token_starts.numel();
  TORCH_CHECK(
      num_chunks * dimensions.num_heads <=
          std::numeric_limits<int>::max(),
      "chunk/head grid must fit in int32");
  TORCH_CHECK(
      dimensions.num_sequences * dimensions.num_heads <=
          std::numeric_limits<int>::max(),
      "sequence/head grid must fit in int32");
  return num_chunks;
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

void pretrain_recurrent_fp32io16_forward(
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

  pretrain_recurrent_fp32io16_forward_cuda(
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
  const auto dimensions = check_recurrent_layout(
      sequence_chunk_offsets,
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

  materialized_chunk_fp32_cuda(
      sequence_chunk_offsets,
      chunk_token_starts,
      chunk_token_ends,
      state_indices,
      state,
      r,
      log_decay,
      k,
      v,
      a,
      b,
      output,
      transform,
      bias,
      boundary,
      state_dot_a.value_or(torch::Tensor()),
      build_warps,
      stages,
      state_tile,
      scale);
}

void pretrain_recurrent_fp32io16_backward(
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

  pretrain_recurrent_fp32io16_backward_cuda(
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

void infer_chunk_bf16_forward_k1_prepare(
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor r,
    torch::Tensor log_decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor chunk_transform,
    torch::Tensor chunk_bias,
    torch::Tensor token_transform,
    torch::Tensor token_bias,
    double scale) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &chunk_token_starts, "chunk_token_starts"},
           {&chunk_token_ends, "chunk_token_ends"},
           {&r, "r"},
           {&log_decay, "log_decay"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&chunk_transform, "chunk_transform"},
           {&chunk_bias, "chunk_bias"},
           {&token_transform, "token_transform"},
           {&token_bias, "token_bias"},
       }) {
    check_cuda_contiguous(*item.first, item.second);
  }
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");
  TORCH_CHECK(
      chunk_token_starts.scalar_type() == torch::kInt32 &&
          chunk_token_ends.scalar_type() == torch::kInt32,
      "K1 chunk metadata must be int32");
  TORCH_CHECK(
      chunk_token_starts.dim() == 1 &&
          chunk_token_starts.numel() > 0 &&
          chunk_token_starts.sizes() == chunk_token_ends.sizes(),
      "K1 chunk metadata must have shape [C]");
  TORCH_CHECK(
      r.dim() == 3 && r.size(0) > 0 && r.size(1) > 0 &&
          r.size(2) == kHeadSize,
      "K1 token tensors must have shape [total_tokens,H,64]");
  TORCH_CHECK(
      r.sizes() == log_decay.sizes() &&
          r.sizes() == k.sizes() &&
          r.sizes() == v.sizes() &&
          r.sizes() == a.sizes() &&
          r.sizes() == b.sizes(),
      "K1 token tensor shape mismatch");
  TORCH_CHECK(
      r.scalar_type() == torch::kBFloat16 &&
          log_decay.scalar_type() == torch::kBFloat16 &&
          k.scalar_type() == torch::kBFloat16 &&
          v.scalar_type() == torch::kBFloat16 &&
          a.scalar_type() == torch::kBFloat16 &&
          b.scalar_type() == torch::kBFloat16,
      "K1 token tensors must be bf16");

  const int64_t num_chunks = chunk_token_starts.numel();
  const int64_t num_heads = r.size(1);
  const std::vector<int64_t> chunk_shape{
      num_chunks,
      num_heads,
      kHeadSize,
      kHeadSize,
  };
  TORCH_CHECK(
      chunk_transform.sizes().vec() == chunk_shape &&
          chunk_bias.sizes().vec() == chunk_shape &&
          chunk_transform.scalar_type() == torch::kFloat32 &&
          chunk_bias.scalar_type() == torch::kFloat32,
      "K1 chunk workspaces must be fp32 [C,H,64,64]");
  TORCH_CHECK(
      token_transform.sizes() == r.sizes() &&
          token_bias.sizes() == r.sizes() &&
          token_transform.scalar_type() == torch::kFloat32 &&
          token_bias.scalar_type() == torch::kFloat32,
      "K1 token workspaces must be fp32 [total_tokens,H,64]");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &chunk_token_starts, "chunk_token_starts"},
           {&chunk_token_ends, "chunk_token_ends"},
           {&log_decay, "log_decay"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&chunk_transform, "chunk_transform"},
           {&chunk_bias, "chunk_bias"},
           {&token_transform, "token_transform"},
           {&token_bias, "token_bias"},
       }) {
    check_same_device(r, *item.first, item.second);
  }

  infer_chunk_bf16_forward_k1_prepare_cuda(
      chunk_token_starts,
      chunk_token_ends,
      r,
      log_decay,
      k,
      v,
      a,
      b,
      chunk_transform,
      chunk_bias,
      token_transform,
      token_bias,
      scale);
}

void infer_chunk_bf16_forward_k2_recurrence(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state,
    torch::Tensor output,
    torch::Tensor chunk_transform,
    torch::Tensor chunk_bias,
    torch::Tensor token_transform,
    torch::Tensor token_bias) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &sequence_chunk_offsets, "sequence_chunk_offsets"},
           {&chunk_token_starts, "chunk_token_starts"},
           {&chunk_token_ends, "chunk_token_ends"},
           {&state, "state"},
           {&output, "output"},
           {&chunk_transform, "chunk_transform"},
           {&chunk_bias, "chunk_bias"},
           {&token_transform, "token_transform"},
           {&token_bias, "token_bias"},
       }) {
    check_cuda_contiguous(*item.first, item.second);
  }
  TORCH_CHECK(
      sequence_chunk_offsets.scalar_type() == torch::kInt32 &&
          chunk_token_starts.scalar_type() == torch::kInt32 &&
          chunk_token_ends.scalar_type() == torch::kInt32,
      "K2 chunk metadata must be int32");
  TORCH_CHECK(
      state.dim() == 4 && state.size(0) > 0 &&
          state.size(1) > 0 &&
          state.size(2) == kHeadSize &&
          state.size(3) == kHeadSize &&
          state.scalar_type() == torch::kBFloat16,
      "K2 state must be bf16 [N,H,64,64]");
  const int64_t num_sequences = state.size(0);
  const int64_t num_heads = state.size(1);
  TORCH_CHECK(
      sequence_chunk_offsets.dim() == 1 &&
          sequence_chunk_offsets.numel() == num_sequences + 1,
      "K2 sequence_chunk_offsets must have shape [N+1]");
  TORCH_CHECK(
      chunk_token_starts.dim() == 1 &&
          chunk_token_starts.numel() > 0 &&
          chunk_token_starts.sizes() == chunk_token_ends.sizes(),
      "K2 chunk metadata must have shape [C]");
  TORCH_CHECK(
      output.dim() == 3 && output.size(0) > 0 &&
          output.size(1) == num_heads &&
          output.size(2) == kHeadSize &&
          output.scalar_type() == torch::kBFloat16,
      "K2 output must be bf16 [total_tokens,H,64]");

  const int64_t num_chunks = chunk_token_starts.numel();
  const std::vector<int64_t> chunk_shape{
      num_chunks,
      num_heads,
      kHeadSize,
      kHeadSize,
  };
  TORCH_CHECK(
      chunk_transform.sizes().vec() == chunk_shape &&
          chunk_bias.sizes().vec() == chunk_shape &&
          chunk_transform.scalar_type() == torch::kFloat32 &&
          chunk_bias.scalar_type() == torch::kFloat32,
      "K2 chunk workspaces must be fp32 [C,H,64,64]");
  TORCH_CHECK(
      token_transform.sizes() == output.sizes() &&
          token_bias.sizes() == output.sizes() &&
          token_transform.scalar_type() == torch::kFloat32 &&
          token_bias.scalar_type() == torch::kFloat32,
      "K2 token workspaces must be fp32 [total_tokens,H,64]");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &sequence_chunk_offsets, "sequence_chunk_offsets"},
           {&chunk_token_starts, "chunk_token_starts"},
           {&chunk_token_ends, "chunk_token_ends"},
           {&output, "output"},
           {&chunk_transform, "chunk_transform"},
           {&chunk_bias, "chunk_bias"},
           {&token_transform, "token_transform"},
           {&token_bias, "token_bias"},
       }) {
    check_same_device(state, *item.first, item.second);
  }

  infer_chunk_bf16_forward_k2_recurrence_cuda(
      sequence_chunk_offsets,
      chunk_token_starts,
      chunk_token_ends,
      state,
      output,
      chunk_transform,
      chunk_bias,
      token_transform,
      token_bias);
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
  const auto dimensions = check_recurrent_layout(
      sequence_chunk_offsets,
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

  recompute_chunk_fp32_cuda(
      sequence_chunk_offsets,
      chunk_token_starts,
      chunk_token_ends,
      state_indices,
      state,
      r,
      log_decay,
      k,
      v,
      a,
      b,
      output,
      boundary,
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
  module.def(
      "pretrain_recurrent_fp32io16_forward",
      &pretrain_recurrent_fp32io16_forward,
      "RWKV-LM-derived recurrent training forward with FP32 state",
      py::arg("sequence_chunk_offsets"),
      py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"),
      py::arg("state"),
      py::arg("r"),
      py::arg("log_decay"),
      py::arg("k"),
      py::arg("v"),
      py::arg("a"),
      py::arg("b"),
      py::arg("output"),
      py::arg("boundary"),
      py::arg("state_dot_a"),
      py::arg("scale"));
  module.def(
      "materialized_chunk_fp32",
      &materialized_chunk_fp32,
      "FlashRWKV materialized chunk forward with FP32 canonical state",
      py::arg("sequence_chunk_offsets"),
      py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"),
      py::arg("state_indices"),
      py::arg("state"),
      py::arg("r"),
      py::arg("log_decay"),
      py::arg("k"),
      py::arg("v"),
      py::arg("a"),
      py::arg("b"),
      py::arg("output"),
      py::arg("transform"),
      py::arg("bias"),
      py::arg("boundary"),
      py::arg("build_warps"),
      py::arg("stages"),
      py::arg("state_tile"),
      py::arg("scale"),
      py::arg("state_dot_a") = py::none());
  module.def(
      "pretrain_recurrent_fp32io16_backward",
      &pretrain_recurrent_fp32io16_backward,
      "RWKV-LM-derived recurrent training backward with FP32 state",
      py::arg("sequence_chunk_offsets"),
      py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"),
      py::arg("final_state"),
      py::arg("r"),
      py::arg("log_decay"),
      py::arg("k"),
      py::arg("v"),
      py::arg("a"),
      py::arg("b"),
      py::arg("state_dot_a"),
      py::arg("grad_output"),
      py::arg("grad_final_state"),
      py::arg("boundary"),
      py::arg("grad_r"),
      py::arg("grad_log_decay"),
      py::arg("grad_k"),
      py::arg("grad_v"),
      py::arg("grad_a"),
      py::arg("grad_b"),
      py::arg("grad_initial_state"),
      py::arg("scale"));
  module.def(
      "infer_chunk_bf16_forward_k1_prepare",
      &infer_chunk_bf16_forward_k1_prepare,
      "KDA-derived K1 chunk preparation for BF16 inference",
      py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"),
      py::arg("r"),
      py::arg("log_decay"),
      py::arg("k"),
      py::arg("v"),
      py::arg("a"),
      py::arg("b"),
      py::arg("chunk_transform"),
      py::arg("chunk_bias"),
      py::arg("token_transform"),
      py::arg("token_bias"),
      py::arg("scale"));
  module.def(
      "infer_chunk_bf16_forward_k2_recurrence",
      &infer_chunk_bf16_forward_k2_recurrence,
      "KDA-derived K2 boundary recurrence for BF16 inference",
      py::arg("sequence_chunk_offsets"),
      py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"),
      py::arg("state"),
      py::arg("output"),
      py::arg("chunk_transform"),
      py::arg("chunk_bias"),
      py::arg("token_transform"),
      py::arg("token_bias"));
  module.def(
      "recompute_chunk_fp32",
      &recompute_chunk_fp32,
      "FlashRWKV DPLR-factor recompute chunk forward with FP32 state",
      py::arg("sequence_chunk_offsets"),
      py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"),
      py::arg("state_indices"),
      py::arg("state"),
      py::arg("r"),
      py::arg("log_decay"),
      py::arg("k"),
      py::arg("v"),
      py::arg("a"),
      py::arg("b"),
      py::arg("output"),
      py::arg("boundary"),
      py::arg("scale"));
}
