// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vllm-rwkv rwkv7_wkv_fp32_v2 at commit
// 6d683f9e49a2997e405c47edc147872c8609513b.

#include "../../bindings.h"
#include "../../validation.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>
#include <memory>
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
    torch::Tensor metadata_status,
    double scale);
void recurrent_fp32_from_decay_logits_cuda(
    torch::Tensor query_start_loc,
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
void recurrent_fp16_from_decay_logits_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor decay_bias,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor elapsed_t,
    torch::Tensor output,
    torch::Tensor metadata_status,
    double scale);

using flash_rwkv::validation::check_recurrent_layout;
using flash_rwkv::validation::check_cuda_contiguous;
using flash_rwkv::validation::check_same_device;
using flash_rwkv::validation::prepare_recurrent_metadata_cuda;

namespace {

std::optional<uint32_t> tensor_version(const torch::Tensor& tensor) {
  if (tensor.unsafeGetTensorImpl()->is_inference()) {
    return std::nullopt;
  }
  return tensor.unsafeGetTensorImpl()
      ->version_counter()
      .current_version();
}

class RecurrentMetadataTicket final {
 public:
  RecurrentMetadataTicket(
      torch::Tensor query_start_loc,
      torch::Tensor state_indices,
      torch::Tensor query_start_loc_snapshot,
      torch::Tensor state_indices_snapshot,
      torch::Tensor status,
      int64_t total_tokens,
      int64_t state_pool_size,
      cudaStream_t stream)
      : query_start_loc_(std::move(query_start_loc)),
        state_indices_(std::move(state_indices)),
        query_start_loc_snapshot_(std::move(query_start_loc_snapshot)),
        state_indices_snapshot_(std::move(state_indices_snapshot)),
        status_(std::move(status)),
        query_start_loc_version_(
            tensor_version(query_start_loc_)),
        state_indices_version_(tensor_version(state_indices_)),
        query_start_loc_data_(query_start_loc_.data_ptr()),
        state_indices_data_(state_indices_.data_ptr()),
        query_start_loc_sizes_(query_start_loc_.sizes().vec()),
        state_indices_sizes_(state_indices_.sizes().vec()),
        query_start_loc_strides_(query_start_loc_.strides().vec()),
        state_indices_strides_(state_indices_.strides().vec()),
        total_tokens_(total_tokens),
        state_pool_size_(state_pool_size),
        device_(query_start_loc_.device()),
        stream_(stream) {}

  void check_compatible(
      const torch::Tensor& query_start_loc,
      const torch::Tensor& state_indices,
      int64_t total_tokens,
      int64_t state_pool_size) const {
    TORCH_CHECK(
        query_start_loc.is_same(query_start_loc_),
        "validated_metadata query_start_loc identity mismatch");
    TORCH_CHECK(
        state_indices.is_same(state_indices_),
        "validated_metadata state_indices identity mismatch");
    TORCH_CHECK(
        query_start_loc.data_ptr() == query_start_loc_data_ &&
            state_indices.data_ptr() == state_indices_data_,
        "validated_metadata metadata data_ptr mismatch");
    TORCH_CHECK(
        query_start_loc.sizes().vec() == query_start_loc_sizes_ &&
            state_indices.sizes().vec() == state_indices_sizes_ &&
            query_start_loc.strides().vec() == query_start_loc_strides_ &&
            state_indices.strides().vec() == state_indices_strides_,
        "validated_metadata metadata shape or stride mismatch");
    if (query_start_loc_version_.has_value()) {
      TORCH_CHECK(
          tensor_version(query_start_loc) == query_start_loc_version_,
          "validated_metadata query_start_loc version mismatch");
    }
    if (state_indices_version_.has_value()) {
      TORCH_CHECK(
          tensor_version(state_indices) == state_indices_version_,
          "validated_metadata state_indices version mismatch");
    }
    TORCH_CHECK(
        query_start_loc.device() == device_ && state_indices.device() == device_,
        "validated_metadata device mismatch");
    TORCH_CHECK(
        total_tokens == total_tokens_,
        "validated_metadata total_tokens mismatch");
    TORCH_CHECK(
        state_pool_size == state_pool_size_,
        "validated_metadata state_pool_size mismatch");
    TORCH_CHECK(
        at::cuda::getCurrentCUDAStream(device_.index()).stream() == stream_,
        "validated_metadata stream mismatch; prepare and consume the ticket "
        "on the same CUDA stream");
  }

  const torch::Tensor& query_start_loc_snapshot() const {
    return query_start_loc_snapshot_;
  }

  const torch::Tensor& state_indices_snapshot() const {
    return state_indices_snapshot_;
  }

  const torch::Tensor& status() const { return status_; }

 private:
  torch::Tensor query_start_loc_;
  torch::Tensor state_indices_;
  torch::Tensor query_start_loc_snapshot_;
  torch::Tensor state_indices_snapshot_;
  torch::Tensor status_;
  std::optional<uint32_t> query_start_loc_version_;
  std::optional<uint32_t> state_indices_version_;
  void* query_start_loc_data_;
  void* state_indices_data_;
  std::vector<int64_t> query_start_loc_sizes_;
  std::vector<int64_t> state_indices_sizes_;
  std::vector<int64_t> query_start_loc_strides_;
  std::vector<int64_t> state_indices_strides_;
  int64_t total_tokens_;
  int64_t state_pool_size_;
  c10::Device device_;
  cudaStream_t stream_;
};

std::shared_ptr<RecurrentMetadataTicket> prepare_recurrent_metadata(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    int64_t total_tokens,
    int64_t state_pool_size) {
  check_cuda_contiguous(query_start_loc, "query_start_loc");
  check_cuda_contiguous(state_indices, "state_indices");
  check_same_device(query_start_loc, state_indices, "state_indices");
  TORCH_CHECK(
      query_start_loc.scalar_type() == torch::kInt32 &&
          state_indices.scalar_type() == torch::kInt32,
      "recurrent metadata must be int32");
  TORCH_CHECK(
      query_start_loc.dim() == 1 && state_indices.dim() == 1 &&
          state_indices.numel() > 0 &&
          query_start_loc.numel() == state_indices.numel() + 1,
      "query_start_loc must have shape [B+1] and state_indices shape [B]");
  TORCH_CHECK(
      total_tokens > 0 && total_tokens <= std::numeric_limits<int>::max(),
      "total_tokens must be positive and fit in int32");
  TORCH_CHECK(
      state_pool_size > 0 &&
          state_pool_size <= std::numeric_limits<int>::max(),
      "state_pool_size must be positive and fit in int32");
  const c10::cuda::CUDAGuard device_guard(query_start_loc.device());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(query_start_loc.device().index()).stream();
  auto prepared = prepare_recurrent_metadata_cuda(
      query_start_loc, state_indices, total_tokens, state_pool_size);
  return std::make_shared<RecurrentMetadataTicket>(
      std::move(query_start_loc), std::move(state_indices),
      std::move(prepared.query_start_loc), std::move(prepared.state_indices),
      std::move(prepared.status),
      total_tokens, state_pool_size, stream);
}

}  // namespace

void recurrent(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    double scale,
    bool fp16_state,
    bool from_decay_logits,
    std::optional<torch::Tensor> decay_bias,
    std::optional<torch::Tensor> elapsed_t,
    std::shared_ptr<RecurrentMetadataTicket> validated_metadata) {
  check_recurrent_layout(
      query_start_loc,
      state_indices,
      state,
      r,
      decay,
      k,
      v,
      a,
      b,
      output,
      scale);
  if (fp16_state) {
    TORCH_CHECK(state.scalar_type() == torch::kFloat16, "state must be fp16");
    TORCH_CHECK(
        r.scalar_type() == torch::kFloat16,
        "FP16-state token tensors must be fp16");
  } else {
    TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be fp32");
    TORCH_CHECK(
        r.scalar_type() == torch::kFloat16 ||
            r.scalar_type() == torch::kBFloat16 ||
            r.scalar_type() == torch::kFloat32,
        "FP32-state token tensors must be fp16, bf16, or fp32");
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
    const int64_t num_heads = state.size(1);
    const int64_t head_size = state.size(2);
    TORCH_CHECK(
        (decay_bias->dim() == 1 &&
         decay_bias->numel() == num_heads * head_size) ||
            (decay_bias->dim() == 2 &&
             decay_bias->size(0) == num_heads &&
             decay_bias->size(1) == head_size),
        "decay_bias must have shape [H*D] or [H,D]");
  }
  if (elapsed_t.has_value()) {
    check_cuda_contiguous(*elapsed_t, "elapsed_t");
    check_same_device(state, *elapsed_t, "elapsed_t");
    TORCH_CHECK(
        from_decay_logits && fp16_state,
        "elapsed_t dithering is valid only for raw mode='fp16'");
    TORCH_CHECK(
        elapsed_t->scalar_type() == torch::kInt32,
        "elapsed_t must be int32");
    TORCH_CHECK(
        elapsed_t->dim() == 1 && elapsed_t->numel() == state.size(0),
        "elapsed_t must have shape [state_pool_slots]");
  }

  torch::Tensor launch_query_start_loc = query_start_loc;
  torch::Tensor launch_state_indices = state_indices;
  torch::Tensor metadata_status;
  if (validated_metadata) {
    validated_metadata->check_compatible(
        query_start_loc, state_indices, r.size(0), state.size(0));
    launch_query_start_loc = validated_metadata->query_start_loc_snapshot();
    launch_state_indices = validated_metadata->state_indices_snapshot();
    metadata_status = validated_metadata->status();
  } else {
    auto prepared = prepare_recurrent_metadata_cuda(
        query_start_loc, state_indices, r.size(0), state.size(0));
    launch_query_start_loc = std::move(prepared.query_start_loc);
    launch_state_indices = std::move(prepared.state_indices);
    metadata_status = std::move(prepared.status);
  }
  if (from_decay_logits && fp16_state) {
    recurrent_fp16_from_decay_logits_cuda(
        launch_query_start_loc, launch_state_indices, state, r, decay,
        decay_bias.value_or(torch::Tensor()), k, v, a, b,
        elapsed_t.value_or(torch::Tensor()), output, metadata_status, scale);
  } else if (from_decay_logits) {
    recurrent_fp32_from_decay_logits_cuda(
        launch_query_start_loc, launch_state_indices, state, r, decay,
        decay_bias.value_or(torch::Tensor()), k, v, a, b, output,
        metadata_status, scale);
  } else if (fp16_state) {
    recurrent_fp16_cuda(
        launch_query_start_loc, launch_state_indices, state, r, decay,
        k, v, a, b, output,
        metadata_status, scale);
  } else {
    recurrent_fp32_cuda(
        launch_query_start_loc, launch_state_indices, state, r, decay,
        k, v, a, b, output,
        metadata_status, scale);
  }
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
    double scale,
    std::shared_ptr<RecurrentMetadataTicket> validated_metadata) {
  recurrent(
      query_start_loc, state_indices, state, r, log_decay, k, v, a, b,
      output, scale, false, false, std::nullopt, std::nullopt,
      std::move(validated_metadata));
}

void recurrent_fp32_from_decay_logits(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
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
    std::optional<torch::Tensor> elapsed_t,
    std::shared_ptr<RecurrentMetadataTicket> validated_metadata) {
  recurrent(
      query_start_loc, state_indices, state, r, decay_logits, k, v, a, b,
      output, scale, false, true, decay_bias, elapsed_t,
      std::move(validated_metadata));
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
    double scale,
    std::shared_ptr<RecurrentMetadataTicket> validated_metadata) {
  recurrent(
      query_start_loc, state_indices, state, r, log_decay, k, v, a, b,
      output, scale, true, false, std::nullopt, std::nullopt,
      std::move(validated_metadata));
}

void recurrent_fp16_from_decay_logits(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
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
    std::optional<torch::Tensor> elapsed_t,
    std::shared_ptr<RecurrentMetadataTicket> validated_metadata) {
  recurrent(
      query_start_loc, state_indices, state, r, decay_logits, k, v, a, b,
      output, scale, true, true, decay_bias, elapsed_t,
      std::move(validated_metadata));
}

void register_infer_recurrent_bindings(py::module_& module) {
  py::class_<
      RecurrentMetadataTicket,
      std::shared_ptr<RecurrentMetadataTicket>>(
      module, "_RecurrentMetadataTicket");
  module.def(
      "prepare_recurrent_metadata", &prepare_recurrent_metadata,
      "Validate packed recurrent metadata once for same-stream layer reuse",
      py::arg("query_start_loc"), py::arg("state_indices"),
      py::arg("total_tokens"), py::arg("state_pool_size"));
  module.def("recurrent_fp32", &recurrent_fp32,
             "FlashRWKV recurrent forward with FP32 canonical state",
             py::arg("query_start_loc"), py::arg("state_indices"),
             py::arg("state"), py::arg("r"), py::arg("log_decay"),
             py::arg("k"), py::arg("v"), py::arg("a"), py::arg("b"),
             py::arg("output"), py::arg("scale"),
             py::arg("validated_metadata") = py::none());
  module.def("recurrent_fp16", &recurrent_fp16,
             "FlashRWKV recurrent forward with FP16 canonical state",
             py::arg("query_start_loc"), py::arg("state_indices"),
             py::arg("state"), py::arg("r"), py::arg("log_decay"),
             py::arg("k"), py::arg("v"), py::arg("a"), py::arg("b"),
             py::arg("output"), py::arg("scale"),
             py::arg("validated_metadata") = py::none());
  module.def(
      "recurrent_fp32_from_decay_logits", &recurrent_fp32_from_decay_logits,
      "FlashRWKV recurrent forward with fused raw decay logits and FP32 state",
      py::arg("query_start_loc"), py::arg("state_indices"),
      py::arg("state"), py::arg("r"), py::arg("decay_logits"),
      py::arg("k"), py::arg("v"), py::arg("a"), py::arg("b"),
      py::arg("output"), py::arg("scale"),
      py::arg("decay_bias") = py::none(),
      py::arg("elapsed_t") = py::none(),
      py::arg("validated_metadata") = py::none());
  module.def(
      "recurrent_fp16_from_decay_logits", &recurrent_fp16_from_decay_logits,
      "FlashRWKV recurrent forward with fused raw decay logits and FP16 state",
      py::arg("query_start_loc"), py::arg("state_indices"),
      py::arg("state"), py::arg("r"), py::arg("decay_logits"),
      py::arg("k"), py::arg("v"), py::arg("a"), py::arg("b"),
      py::arg("output"), py::arg("scale"),
      py::arg("decay_bias") = py::none(),
      py::arg("elapsed_t") = py::none(),
      py::arg("validated_metadata") = py::none());
}
