// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.
// Modified by contributors to the FlashRWKV project.
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

torch::Tensor cmix_sparse_one_cuda(
    int C,
    int F,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_rows_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_one_cuda(
    int C,
    int F,
    torch::Tensor act,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_rows_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor act,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_relu_one_cuda(
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_relu_one_split2_cuda(
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_relu_rows_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_relu_rows_split2_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_relu_rows_t512_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc);
torch::Tensor cmix_sparse_down_relu_rows_t512_cfg_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    int accumulators);
void cmix_sparse_down_relu_rows_t512_cfg_out_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    torch::Tensor out,
    int accumulators);
torch::Tensor cmix_sparse_down_relu_rows_t512_reuse_cfg_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    int accumulators);
void cmix_sparse_down_relu_rows_t512_reuse_cfg_out_cuda(
    int B,
    int T,
    int C,
    int F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    torch::Tensor out,
    int accumulators);

torch::Tensor cmix_mix_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k);
torch::Tensor cmix_mix_cfg_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    int threads);
void cmix_mix_cfg_out_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor out,
    int threads);
torch::Tensor cmix_mix_3d_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k);
void cmix_mix_3d_out_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor out);

namespace {

void check_half_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, name, " must be fp16");
}

void check_3d(const torch::Tensor& x, int64_t B, int64_t T, int64_t C, const char* name) {
  TORCH_CHECK(B > 0 && T > 0 && C > 0, "B, T, and C must be positive");
  check_half_cuda_contig(x, name);
  TORCH_CHECK(x.dim() == 3, name, " must have shape [B,T,C]");
  TORCH_CHECK(x.size(0) == B && x.size(1) == T && x.size(2) == C, name, " shape mismatch");
}

void check_vec(const torch::Tensor& x, int64_t C, const char* name) {
  TORCH_CHECK(C > 0, "C must be positive");
  check_half_cuda_contig(x, name);
  TORCH_CHECK(x.dim() == 1 && x.size(0) == C, name, " must have shape [C]");
}

void check_same_device(
    const std::vector<std::pair<const torch::Tensor*, const char*>>& tensors) {
  TORCH_CHECK(!tensors.empty(), "internal error: empty tensor list");
  const int device = tensors.front().first->get_device();
  for (const auto& [tensor, name] : tensors) {
    TORCH_CHECK(tensor->get_device() == device, name, " must be on the same CUDA device as the primary input");
  }
}

void check_half2_aligned(const torch::Tensor& x, const char* name) {
  TORCH_CHECK((reinterpret_cast<std::uintptr_t>(x.data_ptr()) & 0x3u) == 0,
              name, " must be 4-byte aligned for half2 access");
}

void check_no_storage_overlap(
    const torch::Tensor& output,
    const char* output_name,
    const torch::Tensor& other,
    const char* other_name) {
  const auto output_begin = reinterpret_cast<std::uintptr_t>(output.data_ptr());
  const auto other_begin = reinterpret_cast<std::uintptr_t>(other.data_ptr());
  const auto output_end = output_begin + output.nbytes();
  const auto other_end = other_begin + other.nbytes();
  TORCH_CHECK(output_end <= other_begin || other_end <= output_begin,
              output_name, " must not overlap ", other_name);
}

int64_t check_head_grid_dims(int64_t B, int64_t T, int64_t C, int64_t H, bool grouped_heads) {
  TORCH_CHECK(B > 0 && B <= std::numeric_limits<int>::max(), "B must be positive int32");
  TORCH_CHECK(T > 0 && T <= std::numeric_limits<int>::max(), "T must be positive int32");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max() && (C % 64) == 0,
              "C must be positive int32 divisible by 64");
  TORCH_CHECK(H > 0 && H <= std::numeric_limits<int>::max() && H == C / 64,
              "only head size 64 is supported");
  if (grouped_heads) {
    TORCH_CHECK((H % 4) == 0, "2D KK head grid requires H divisible by 4");
  }
  TORCH_CHECK(B <= 65535 / T, "2D head grid requires B*T <= 65535");
  return B * T;
}

void check_half2_same_device(
    const std::vector<std::pair<const torch::Tensor*, const char*>>& tensors) {
  TORCH_CHECK(!tensors.empty(), "internal error: empty tensor list");
  const int device = tensors.front().first->get_device();
  for (const auto& [tensor, name] : tensors) {
    check_half2_aligned(*tensor, name);
    TORCH_CHECK(tensor->get_device() == device, "all tensors must be on the same CUDA device");
  }
}

}  // namespace

torch::Tensor cmix_sparse_one(
    int64_t C,
    int64_t F,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor value_fc) {
  check_3d(x, 1, 1, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == 1 && shift_state.size(1) == C,
              "shift_state must have shape [1,C]");
  check_vec(x_k, C, "x_k");
  check_half_cuda_contig(key_fc, "key_fc");
  TORCH_CHECK(key_fc.dim() == 2 && key_fc.size(0) == F && key_fc.size(1) == C,
              "key_fc must have shape [F,C]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_one_cuda(
      static_cast<int>(C), static_cast<int>(F), x, shift_state, x_k, key_fc, value_fc);
}

torch::Tensor cmix_sparse_rows(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor value_fc) {
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_k, C, "x_k");
  check_half_cuda_contig(key_fc, "key_fc");
  TORCH_CHECK(key_fc.dim() == 2 && key_fc.size(0) == F && key_fc.size(1) == C,
              "key_fc must have shape [F,C]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_rows_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      x, shift_state, x_k, key_fc, value_fc);
}

torch::Tensor cmix_sparse_down_one(
    int64_t C,
    int64_t F,
    torch::Tensor act,
    torch::Tensor value_fc) {
  check_half_cuda_contig(act, "act");
  TORCH_CHECK(act.dim() == 1 && act.size(0) == F, "act must have shape [F]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_one_cuda(static_cast<int>(C), static_cast<int>(F), act, value_fc);
}

torch::Tensor cmix_sparse_down_rows(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor act,
    torch::Tensor value_fc) {
  check_3d(act, B, T, F, "act");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_rows_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      act, value_fc);
}

torch::Tensor cmix_sparse_down_relu_one(
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc) {
  check_half_cuda_contig(preact, "preact");
  TORCH_CHECK(preact.dim() == 1 && preact.size(0) == F, "preact must have shape [F]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_relu_one_cuda(static_cast<int>(C), static_cast<int>(F), preact, value_fc);
}

torch::Tensor cmix_sparse_down_relu_one_split2(
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc) {
  check_half_cuda_contig(preact, "preact");
  TORCH_CHECK(preact.dim() == 1 && preact.size(0) == F, "preact must have shape [F]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_relu_one_split2_cuda(
      static_cast<int>(C), static_cast<int>(F), preact, value_fc);
}

torch::Tensor cmix_sparse_down_relu_rows(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc) {
  check_3d(preact, B, T, F, "preact");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_relu_rows_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      preact, value_fc);
}

torch::Tensor cmix_sparse_down_relu_rows_split2(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc) {
  check_3d(preact, B, T, F, "preact");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 128) == 0, "C must be divisible by 128");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_relu_rows_split2_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      preact, value_fc);
}

torch::Tensor cmix_sparse_down_relu_rows_t512(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc) {
  TORCH_CHECK(B > 0 && T > 0 && B <= 65535 / T,
              "t512 sparse down requires 0 < B*T <= 65535");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max(), "C must be positive int32");
  TORCH_CHECK(F > 0 && F <= std::numeric_limits<int>::max(), "F must be positive int32");
  check_3d(preact, B, T, F, "preact");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 512) == 0, "C must be divisible by 512");
  TORCH_CHECK((F % 512) == 0, "F must be divisible by 512");
  check_half2_same_device({{&preact, "preact"}, {&value_fc, "value_fc"}});
  check_no_storage_overlap(preact, "preact", value_fc, "value_fc");
  return cmix_sparse_down_relu_rows_t512_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      preact, value_fc);
}

torch::Tensor cmix_sparse_down_relu_rows_t512_cfg(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    int64_t accumulators) {
  TORCH_CHECK(accumulators == 1 || accumulators == 2 || accumulators == 4,
              "accumulators must be 1, 2, or 4");
  TORCH_CHECK(B > 0 && T > 0 && B <= 65535 / T,
              "t512 sparse down requires 0 < B*T <= 65535");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max(), "C must be positive int32");
  TORCH_CHECK(F > 0 && F <= std::numeric_limits<int>::max(), "F must be positive int32");
  check_3d(preact, B, T, F, "preact");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 512) == 0, "C must be divisible by 512");
  TORCH_CHECK((F % 512) == 0, "F must be divisible by 512");
  check_half2_same_device({{&preact, "preact"}, {&value_fc, "value_fc"}});
  check_no_storage_overlap(preact, "preact", value_fc, "value_fc");
  return cmix_sparse_down_relu_rows_t512_cfg_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      preact, value_fc, static_cast<int>(accumulators));
}

void cmix_sparse_down_relu_rows_t512_cfg_out(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    torch::Tensor out,
    int64_t accumulators) {
  TORCH_CHECK(accumulators == 1 || accumulators == 2 || accumulators == 4,
              "accumulators must be 1, 2, or 4");
  TORCH_CHECK(B > 0 && T > 0 && B <= 65535 / T,
              "t512 sparse down requires 0 < B*T <= 65535");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max(), "C must be positive int32");
  TORCH_CHECK(F > 0 && F <= std::numeric_limits<int>::max(), "F must be positive int32");
  check_3d(preact, B, T, F, "preact");
  check_3d(out, B, T, C, "out");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 512) == 0, "C must be divisible by 512");
  TORCH_CHECK((F % 512) == 0, "F must be divisible by 512");
  check_half2_same_device(
      {{&preact, "preact"}, {&value_fc, "value_fc"}, {&out, "out"}});
  // All three device pointers are restrict-qualified in the kernel. Read/read
  // overlap is invalid too, not only output aliasing.
  check_no_storage_overlap(preact, "preact", value_fc, "value_fc");
  check_no_storage_overlap(out, "out", preact, "preact");
  check_no_storage_overlap(out, "out", value_fc, "value_fc");
  cmix_sparse_down_relu_rows_t512_cfg_out_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      preact, value_fc, out, static_cast<int>(accumulators));
}

torch::Tensor cmix_sparse_down_relu_rows_t512_reuse_cfg(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    int64_t accumulators) {
  TORCH_CHECK(accumulators == 1 || accumulators == 2,
              "reuse accumulators must be 1 or 2");
  TORCH_CHECK(B > 0 && T > 0 && B <= 65535 / T,
              "t512 reuse sparse down requires 0 < B*T <= 65535");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max(), "C must be positive int32");
  TORCH_CHECK(F > 0 && F <= std::numeric_limits<int>::max(), "F must be positive int32");
  check_3d(preact, B, T, F, "preact");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 512) == 0, "C must be divisible by 512");
  TORCH_CHECK((F % 512) == 0, "F must be divisible by 512");
  check_half2_same_device({{&preact, "preact"}, {&value_fc, "value_fc"}});
  check_no_storage_overlap(preact, "preact", value_fc, "value_fc");
  return cmix_sparse_down_relu_rows_t512_reuse_cfg_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      preact, value_fc, static_cast<int>(accumulators));
}

void cmix_sparse_down_relu_rows_t512_reuse_cfg_out(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t F,
    torch::Tensor preact,
    torch::Tensor value_fc,
    torch::Tensor out,
    int64_t accumulators) {
  TORCH_CHECK(accumulators == 1 || accumulators == 2,
              "reuse accumulators must be 1 or 2");
  TORCH_CHECK(B > 0 && T > 0 && B <= 65535 / T,
              "t512 reuse sparse down requires 0 < B*T <= 65535");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max(), "C must be positive int32");
  TORCH_CHECK(F > 0 && F <= std::numeric_limits<int>::max(), "F must be positive int32");
  check_3d(preact, B, T, F, "preact");
  check_3d(out, B, T, C, "out");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
              "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 512) == 0, "C must be divisible by 512");
  TORCH_CHECK((F % 512) == 0, "F must be divisible by 512");
  check_half2_same_device(
      {{&preact, "preact"}, {&value_fc, "value_fc"}, {&out, "out"}});
  check_no_storage_overlap(preact, "preact", value_fc, "value_fc");
  check_no_storage_overlap(out, "out", preact, "preact");
  check_no_storage_overlap(out, "out", value_fc, "value_fc");
  cmix_sparse_down_relu_rows_t512_reuse_cfg_out_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(F),
      preact, value_fc, out, static_cast<int>(accumulators));
}

torch::Tensor cmix_mix(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_k, C, "x_k");
  check_same_device({{&x, "x"}, {&shift_state, "shift_state"}, {&x_k, "x_k"}});
  return cmix_mix_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), x, shift_state, x_k);
}

torch::Tensor cmix_mix_cfg(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    int64_t threads) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 || threads == 1024, "unsupported threads");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_k, C, "x_k");
  return cmix_mix_cfg_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), x, shift_state, x_k, static_cast<int>(threads));
}

void cmix_mix_cfg_out(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor out,
    int64_t threads) {
  TORCH_CHECK(B > 0 && T > 0 && C > 0 && (C % 2) == 0,
              "cmix_mix_cfg_out requires positive B/T and positive even C");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 || threads == 1024,
              "unsupported threads");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_k, C, "x_k");
  check_3d(out, B, T, C, "out");
  TORCH_CHECK(x.get_device() == shift_state.get_device() && x.get_device() == x_k.get_device() &&
                  x.get_device() == out.get_device(),
              "all tensors must be on the same CUDA device");
  check_half2_aligned(x, "x");
  check_half2_aligned(shift_state, "shift_state");
  check_half2_aligned(x_k, "x_k");
  check_half2_aligned(out, "out");
  check_no_storage_overlap(shift_state, "shift_state", x, "x");
  check_no_storage_overlap(shift_state, "shift_state", x_k, "x_k");
  check_no_storage_overlap(out, "out", x, "x");
  check_no_storage_overlap(out, "out", shift_state, "shift_state");
  check_no_storage_overlap(out, "out", x_k, "x_k");
  cmix_mix_cfg_out_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      x, shift_state, x_k, out, static_cast<int>(threads));
}

void check_cmix_mix_3d_inputs(
    int64_t B,
    int64_t T,
    int64_t C,
    const torch::Tensor& x,
    const torch::Tensor& shift_state,
    const torch::Tensor& x_k) {
  TORCH_CHECK(B > 0 && B <= 65535, "cmix_mix_3d requires 1 <= B <= 65535");
  TORCH_CHECK(T > 0 && T <= 65535, "cmix_mix_3d requires 1 <= T <= 65535");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max() && (C % 2) == 0,
              "cmix_mix_3d requires positive even C within int32 range");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_k, C, "x_k");
  TORCH_CHECK(x.get_device() == shift_state.get_device() && x.get_device() == x_k.get_device(),
              "all tensors must be on the same CUDA device");
  check_half2_aligned(x, "x");
  check_half2_aligned(shift_state, "shift_state");
  check_half2_aligned(x_k, "x_k");
  // shift_state is restrict read/write. Aliasing a read-only input would make
  // the T>1 two-launch recurrent-state contract invalid.
  check_no_storage_overlap(shift_state, "shift_state", x, "x");
  check_no_storage_overlap(shift_state, "shift_state", x_k, "x_k");
}

torch::Tensor cmix_mix_3d(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k) {
  check_cmix_mix_3d_inputs(B, T, C, x, shift_state, x_k);
  return cmix_mix_3d_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), x, shift_state, x_k);
}

void cmix_mix_3d_out(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor out) {
  check_cmix_mix_3d_inputs(B, T, C, x, shift_state, x_k);
  check_3d(out, B, T, C, "out");
  TORCH_CHECK(out.get_device() == x.get_device(), "out must be on the same CUDA device");
  check_half2_aligned(out, "out");
  // Byte-range checks also catch shifted contiguous views that share storage.
  check_no_storage_overlap(out, "out", x, "x");
  check_no_storage_overlap(out, "out", shift_state, "shift_state");
  check_no_storage_overlap(out, "out", x_k, "x_k");
  cmix_mix_3d_out_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), x, shift_state, x_k, out);
}

TORCH_LIBRARY_FRAGMENT(rwkv7_fast_ops_fp16, m) {
  m.def(
      "cmix_sparse_one(int C, int F, Tensor x, Tensor(a!) shift_state, Tensor x_k, Tensor key_fc, Tensor value_fc) -> Tensor");
  m.def(
      "cmix_sparse_rows(int B, int T, int C, int F, Tensor x, Tensor(a!) shift_state, Tensor x_k, Tensor key_fc, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_one(int C, int F, Tensor act, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_rows(int B, int T, int C, int F, Tensor act, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_relu_one(int C, int F, Tensor preact, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_relu_one_split2(int C, int F, Tensor preact, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_relu_rows(int B, int T, int C, int F, Tensor preact, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_relu_rows_split2(int B, int T, int C, int F, Tensor preact, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_relu_rows_t512(int B, int T, int C, int F, Tensor preact, Tensor value_fc) -> Tensor");
  m.def("cmix_sparse_down_relu_rows_t512_cfg(int B, int T, int C, int F, Tensor preact, Tensor value_fc, int accumulators) -> Tensor");
  m.def("cmix_sparse_down_relu_rows_t512_cfg_out(int B, int T, int C, int F, Tensor preact, Tensor value_fc, Tensor(a!) out, int accumulators) -> ()");
  m.def("cmix_sparse_down_relu_rows_t512_reuse_cfg(int B, int T, int C, int F, Tensor preact, Tensor value_fc, int accumulators) -> Tensor");
  m.def("cmix_sparse_down_relu_rows_t512_reuse_cfg_out(int B, int T, int C, int F, Tensor preact, Tensor value_fc, Tensor(a!) out, int accumulators) -> ()");
  m.def("cmix_mix(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor x_k) -> Tensor");
  m.def("cmix_mix_cfg(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor x_k, int threads) -> Tensor");
  m.def("cmix_mix_cfg_out(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor x_k, Tensor(b!) out, int threads) -> ()");
  m.def("cmix_mix_3d(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor x_k) -> Tensor");
  m.def("cmix_mix_3d_out(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor x_k, Tensor(b!) out) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_fast_ops_fp16, CUDA, m) {
  m.impl("cmix_sparse_one", &cmix_sparse_one);
  m.impl("cmix_sparse_rows", &cmix_sparse_rows);
  m.impl("cmix_sparse_down_one", &cmix_sparse_down_one);
  m.impl("cmix_sparse_down_rows", &cmix_sparse_down_rows);
  m.impl("cmix_sparse_down_relu_one", &cmix_sparse_down_relu_one);
  m.impl("cmix_sparse_down_relu_one_split2", &cmix_sparse_down_relu_one_split2);
  m.impl("cmix_sparse_down_relu_rows", &cmix_sparse_down_relu_rows);
  m.impl("cmix_sparse_down_relu_rows_split2", &cmix_sparse_down_relu_rows_split2);
  m.impl("cmix_sparse_down_relu_rows_t512", &cmix_sparse_down_relu_rows_t512);
  m.impl("cmix_sparse_down_relu_rows_t512_cfg", &cmix_sparse_down_relu_rows_t512_cfg);
  m.impl("cmix_sparse_down_relu_rows_t512_cfg_out", &cmix_sparse_down_relu_rows_t512_cfg_out);
  m.impl("cmix_sparse_down_relu_rows_t512_reuse_cfg", &cmix_sparse_down_relu_rows_t512_reuse_cfg);
  m.impl("cmix_sparse_down_relu_rows_t512_reuse_cfg_out", &cmix_sparse_down_relu_rows_t512_reuse_cfg_out);
  m.impl("cmix_mix", &cmix_mix);
  m.impl("cmix_mix_cfg", &cmix_mix_cfg);
  m.impl("cmix_mix_cfg_out", &cmix_mix_cfg_out);
  m.impl("cmix_mix_3d", &cmix_mix_3d);
  m.impl("cmix_mix_3d_out", &cmix_mix_3d_out);
}
