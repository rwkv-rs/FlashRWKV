// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vllm-rwkv rwkv7_wkv_fp32_v2 at commit
// 6d683f9e49a2997e405c47edc147872c8609513b for the FlashRWKV core contract.

#include <torch/extension.h>

#include <cmath>
#include <limits>
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
      build_warps,
      stages,
      state_tile,
      scale);
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
      py::arg("scale"));
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
