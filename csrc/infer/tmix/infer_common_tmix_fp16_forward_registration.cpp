// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.
// Modified by contributors to the FlashRWKV project.
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

std::vector<torch::Tensor> tmix_mix6_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g);
std::vector<torch::Tensor> tmix_mix6_cfg_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    int threads);
std::vector<torch::Tensor> tmix_mix6_3d_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g);
void tmix_mix6_cfg_out_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor out_r,
    torch::Tensor out_w,
    torch::Tensor out_k,
    torch::Tensor out_v,
    torch::Tensor out_a,
    torch::Tensor out_g);
void tmix_mix6_3d_out_cuda(
    int B,
    int T,
    int C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor out_r,
    torch::Tensor out_w,
    torch::Tensor out_k,
    torch::Tensor out_v,
    torch::Tensor out_a,
    torch::Tensor out_g);
std::vector<torch::Tensor> tmix_mix6_t1_c4096_cuda(
    int B,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    int threads,
    int vec,
    bool half_math);

std::vector<torch::Tensor> tmix_kk_a_gate_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor x,
    torch::Tensor shift_state,
    bool update_shift);
std::vector<torch::Tensor> tmix_kk_a_gate_2d_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a);

torch::Tensor tmix_lnx_rkvres_xg_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g);
torch::Tensor tmix_lnx_rkvres_xg_warp_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g);
torch::Tensor tmix_lnx_rkvres_xg_warp_2d_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g);

torch::Tensor tmix_vres_gate_cuda(
    int B,
    int T,
    int C,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12);
torch::Tensor tmix_vres_gate_cfg_cuda(
    int B,
    int T,
    int C,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12,
    int threads,
    bool vectorized);
void tmix_vres_gate_cfg_out_cuda(
    int B,
    int T,
    int C,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12,
    torch::Tensor out,
    int threads,
    bool vectorized);

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

std::vector<torch::Tensor> tmix_mix6(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_r, C, "x_r");
  check_vec(x_w, C, "x_w");
  check_vec(x_k, C, "x_k");
  check_vec(x_v, C, "x_v");
  check_vec(x_a, C, "x_a");
  check_vec(x_g, C, "x_g");
  check_same_device({
      {&x, "x"}, {&shift_state, "shift_state"}, {&x_r, "x_r"}, {&x_w, "x_w"},
      {&x_k, "x_k"}, {&x_v, "x_v"}, {&x_a, "x_a"}, {&x_g, "x_g"}});
  return tmix_mix6_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g);
}

std::vector<torch::Tensor> tmix_mix6_cfg(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    int64_t threads) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 || threads == 1024, "unsupported threads");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_r, C, "x_r");
  check_vec(x_w, C, "x_w");
  check_vec(x_k, C, "x_k");
  check_vec(x_v, C, "x_v");
  check_vec(x_a, C, "x_a");
  check_vec(x_g, C, "x_g");
  return tmix_mix6_cfg_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, static_cast<int>(threads));
}

void check_tmix_mix6_strict_inputs(
    int64_t B,
    int64_t T,
    int64_t C,
    const torch::Tensor& x,
    const torch::Tensor& shift_state,
    const torch::Tensor& x_r,
    const torch::Tensor& x_w,
    const torch::Tensor& x_k,
    const torch::Tensor& x_v,
    const torch::Tensor& x_a,
    const torch::Tensor& x_g,
    bool grid3d) {
  TORCH_CHECK(B > 0 && B <= std::numeric_limits<int>::max(), "B must be positive int32");
  TORCH_CHECK(T > 0 && T <= std::numeric_limits<int>::max(), "T must be positive int32");
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max() && (C % 2) == 0,
              "C must be positive even int32");
  if (grid3d) {
    TORCH_CHECK(B <= 65535 && T <= 65535, "3D TMix requires B/T <= 65535");
  }
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  const std::vector<std::pair<const torch::Tensor*, const char*>> mixes = {
      {&x_r, "x_r"}, {&x_w, "x_w"}, {&x_k, "x_k"},
      {&x_v, "x_v"}, {&x_a, "x_a"}, {&x_g, "x_g"}};
  check_half2_aligned(x, "x");
  check_half2_aligned(shift_state, "shift_state");
  TORCH_CHECK(x.get_device() == shift_state.get_device(), "all tensors must be on the same CUDA device");
  check_no_storage_overlap(shift_state, "shift_state", x, "x");
  for (const auto& [mix, name] : mixes) {
    check_vec(*mix, C, name);
    check_half2_aligned(*mix, name);
    TORCH_CHECK(x.get_device() == mix->get_device(), "all tensors must be on the same CUDA device");
    check_no_storage_overlap(shift_state, "shift_state", *mix, name);
  }
}

void check_tmix_mix6_outputs(
    int64_t B,
    int64_t T,
    int64_t C,
    const torch::Tensor& x,
    const torch::Tensor& shift_state,
    const torch::Tensor& x_r,
    const torch::Tensor& x_w,
    const torch::Tensor& x_k,
    const torch::Tensor& x_v,
    const torch::Tensor& x_a,
    const torch::Tensor& x_g,
    const torch::Tensor& out_r,
    const torch::Tensor& out_w,
    const torch::Tensor& out_k,
    const torch::Tensor& out_v,
    const torch::Tensor& out_a,
    const torch::Tensor& out_g) {
  const std::vector<std::pair<const torch::Tensor*, const char*>> inputs = {
      {&x, "x"}, {&shift_state, "shift_state"}, {&x_r, "x_r"}, {&x_w, "x_w"},
      {&x_k, "x_k"}, {&x_v, "x_v"}, {&x_a, "x_a"}, {&x_g, "x_g"}};
  const std::vector<std::pair<const torch::Tensor*, const char*>> outputs = {
      {&out_r, "out_r"}, {&out_w, "out_w"}, {&out_k, "out_k"},
      {&out_v, "out_v"}, {&out_a, "out_a"}, {&out_g, "out_g"}};
  for (const auto& [output, output_name] : outputs) {
    check_3d(*output, B, T, C, output_name);
    check_half2_aligned(*output, output_name);
    TORCH_CHECK(output->get_device() == x.get_device(), "all outputs must be on the input CUDA device");
    for (const auto& [input, input_name] : inputs) {
      check_no_storage_overlap(*output, output_name, *input, input_name);
    }
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    for (size_t j = i + 1; j < outputs.size(); ++j) {
      check_no_storage_overlap(*outputs[i].first, outputs[i].second,
                               *outputs[j].first, outputs[j].second);
    }
  }
}

std::vector<torch::Tensor> tmix_mix6_3d(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
  check_tmix_mix6_strict_inputs(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, true);
  return tmix_mix6_3d_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g);
}

void tmix_mix6_cfg_out(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor out_r,
    torch::Tensor out_w,
    torch::Tensor out_k,
    torch::Tensor out_v,
    torch::Tensor out_a,
    torch::Tensor out_g) {
  check_tmix_mix6_strict_inputs(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, false);
  check_tmix_mix6_outputs(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
                          out_r, out_w, out_k, out_v, out_a, out_g);
  tmix_mix6_cfg_out_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
      out_r, out_w, out_k, out_v, out_a, out_g);
}

void tmix_mix6_3d_out(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor out_r,
    torch::Tensor out_w,
    torch::Tensor out_k,
    torch::Tensor out_v,
    torch::Tensor out_a,
    torch::Tensor out_g) {
  check_tmix_mix6_strict_inputs(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, true);
  check_tmix_mix6_outputs(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
                          out_r, out_w, out_k, out_v, out_a, out_g);
  tmix_mix6_3d_out_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
      out_r, out_w, out_k, out_v, out_a, out_g);
}

std::vector<torch::Tensor> tmix_mix6_t1_c4096(
    int64_t B,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    int64_t threads,
    int64_t vec,
    bool half_math) {
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 || threads == 1024, "unsupported threads");
  TORCH_CHECK(vec == 1 || vec == 2 || vec == 4 || vec == 8, "unsupported vec");
  check_3d(x, B, 1, 4096, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == 4096,
              "shift_state must have shape [B,4096]");
  check_vec(x_r, 4096, "x_r");
  check_vec(x_w, 4096, "x_w");
  check_vec(x_k, 4096, "x_k");
  check_vec(x_v, 4096, "x_v");
  check_vec(x_a, 4096, "x_a");
  check_vec(x_g, 4096, "x_g");
  return tmix_mix6_t1_c4096_cuda(
      static_cast<int>(B), x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
      static_cast<int>(threads), static_cast<int>(vec), half_math);
}

std::vector<torch::Tensor> tmix_kk_a_gate(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_3d(k, B, T, C, "k");
  check_vec(k_k, C, "k_k");
  check_vec(a0, C, "a0");
  check_3d(a12, B, T, C, "a12");
  check_vec(k_a, C, "k_a");
  check_same_device({
      {&k, "k"}, {&k_k, "k_k"}, {&a0, "a0"}, {&a12, "a12"}, {&k_a, "k_a"}});
  return tmix_kk_a_gate_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      k, k_k, a0, a12, k_a, k, k, false);
}

std::vector<torch::Tensor> tmix_kk_a_gate_2d(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a) {
  check_head_grid_dims(B, T, C, H, true);
  check_3d(k, B, T, C, "k");
  check_vec(k_k, C, "k_k");
  check_vec(a0, C, "a0");
  check_3d(a12, B, T, C, "a12");
  check_vec(k_a, C, "k_a");
  check_half2_same_device({
      {&k, "k"}, {&k_k, "k_k"}, {&a0, "a0"}, {&a12, "a12"}, {&k_a, "k_a"}});
  return tmix_kk_a_gate_2d_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      k, k_k, a0, a12, k_a);
}

std::vector<torch::Tensor> tmix_kk_a_gate_update_shift(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor x,
    torch::Tensor shift_state) {
  TORCH_CHECK(T == 1, "tmix_kk_a_gate_update_shift currently requires T=1");
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_3d(k, B, T, C, "k");
  check_vec(k_k, C, "k_k");
  check_vec(a0, C, "a0");
  check_3d(a12, B, T, C, "a12");
  check_vec(k_a, C, "k_a");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B && shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  return tmix_kk_a_gate_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      k, k_k, a0, a12, k_a, x, shift_state, true);
}

torch::Tensor tmix_lnx_rkvres_xg(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_3d(x, B, T, C, "x");
  check_3d(r, B, T, C, "r");
  check_3d(k, B, T, C, "k");
  check_3d(v, B, T, C, "v");
  check_3d(g, B, T, C, "g");
  check_vec(r_k, C, "r_k");
  check_vec(weight, C, "weight");
  check_vec(bias, C, "bias");
  check_same_device({
      {&x, "x"}, {&r, "r"}, {&k, "k"}, {&v, "v"}, {&r_k, "r_k"},
      {&weight, "weight"}, {&bias, "bias"}, {&g, "g"}});
  return tmix_lnx_rkvres_xg_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      x, r, k, v, r_k, weight, bias, g);
}

torch::Tensor tmix_lnx_rkvres_xg_warp(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_3d(x, B, T, C, "x");
  check_3d(r, B, T, C, "r");
  check_3d(k, B, T, C, "k");
  check_3d(v, B, T, C, "v");
  check_3d(g, B, T, C, "g");
  check_vec(r_k, C, "r_k");
  check_vec(weight, C, "weight");
  check_vec(bias, C, "bias");
  return tmix_lnx_rkvres_xg_warp_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      x, r, k, v, r_k, weight, bias, g);
}

torch::Tensor tmix_lnx_rkvres_xg_warp_2d(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g) {
  check_head_grid_dims(B, T, C, H, false);
  check_3d(x, B, T, C, "x");
  check_3d(r, B, T, C, "r");
  check_3d(k, B, T, C, "k");
  check_3d(v, B, T, C, "v");
  check_3d(g, B, T, C, "g");
  check_vec(r_k, C, "r_k");
  check_vec(weight, C, "weight");
  check_vec(bias, C, "bias");
  check_half2_same_device({
      {&x, "x"}, {&r, "r"}, {&k, "k"}, {&v, "v"}, {&r_k, "r_k"},
      {&weight, "weight"}, {&bias, "bias"}, {&g, "g"}});
  return tmix_lnx_rkvres_xg_warp_2d_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      x, r, k, v, r_k, weight, bias, g);
}

torch::Tensor tmix_vres_gate(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12) {
  check_3d(v, B, T, C, "v");
  check_3d(v_first, B, T, C, "v_first");
  check_vec(v0, C, "v0");
  check_3d(v12, B, T, C, "v12");
  check_same_device({
      {&v, "v"}, {&v_first, "v_first"}, {&v0, "v0"}, {&v12, "v12"}});
  return tmix_vres_gate_cuda(static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), v, v_first, v0, v12);
}

torch::Tensor tmix_vres_gate_cfg(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12,
    int64_t threads,
    bool vectorized) {
  check_3d(v, B, T, C, "v");
  check_3d(v_first, B, T, C, "v_first");
  check_vec(v0, C, "v0");
  check_3d(v12, B, T, C, "v12");
  TORCH_CHECK(threads == 64 || threads == 128 || threads == 256 || threads == 512,
              "threads must be 64, 128, 256, or 512");
  TORCH_CHECK(v.get_device() == v_first.get_device() && v.get_device() == v0.get_device() &&
                  v.get_device() == v12.get_device(),
              "all tensors must be on the same CUDA device");
  if (vectorized) {
    TORCH_CHECK((C % 2) == 0, "vectorized vres gate requires even C");
    TORCH_CHECK(B * T <= 65535, "vectorized vres gate requires B*T <= 65535");
    check_half2_aligned(v, "v");
    check_half2_aligned(v_first, "v_first");
    check_half2_aligned(v0, "v0");
    check_half2_aligned(v12, "v12");
  }
  return tmix_vres_gate_cfg_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      v, v_first, v0, v12, static_cast<int>(threads), vectorized);
}

void tmix_vres_gate_cfg_out(
    int64_t B,
    int64_t T,
    int64_t C,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12,
    torch::Tensor out,
    int64_t threads,
    bool vectorized) {
  check_3d(v, B, T, C, "v");
  check_3d(v_first, B, T, C, "v_first");
  check_vec(v0, C, "v0");
  check_3d(v12, B, T, C, "v12");
  check_3d(out, B, T, C, "out");
  TORCH_CHECK(threads == 64 || threads == 128 || threads == 256 || threads == 512,
              "threads must be 64, 128, 256, or 512");
  TORCH_CHECK(v.get_device() == v_first.get_device() && v.get_device() == v0.get_device() &&
                  v.get_device() == v12.get_device() && v.get_device() == out.get_device(),
              "all tensors must be on the same CUDA device");
  // All CUDA pointers are __restrict__; output/input aliasing would invalidate that contract.
  if (out.numel() != 0) {
    TORCH_CHECK(out.data_ptr() != v.data_ptr() && out.data_ptr() != v_first.data_ptr() &&
                    out.data_ptr() != v0.data_ptr() && out.data_ptr() != v12.data_ptr(),
                "out must not alias any input");
  }
  if (vectorized) {
    TORCH_CHECK((C % 2) == 0, "vectorized vres gate requires even C");
    TORCH_CHECK(B * T <= 65535, "vectorized vres gate requires B*T <= 65535");
    check_half2_aligned(v, "v");
    check_half2_aligned(v_first, "v_first");
    check_half2_aligned(v0, "v0");
    check_half2_aligned(v12, "v12");
    check_half2_aligned(out, "out");
  }
  tmix_vres_gate_cfg_out_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C),
      v, v_first, v0, v12, out, static_cast<int>(threads), vectorized);
}

TORCH_LIBRARY_FRAGMENT(rwkv7_fast_ops_fp16, m) {
  m.def(
      "tmix_mix6(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]");
  m.def(
      "tmix_mix6_cfg(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, int threads) -> Tensor[]");
  m.def(
      "tmix_mix6_3d(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]");
  m.def(
      "tmix_mix6_cfg_out(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, "
      "Tensor(b!) out_r, Tensor(c!) out_w, Tensor(d!) out_k, Tensor(e!) out_v, Tensor(f!) out_a, Tensor(g!) out_g) -> ()");
  m.def(
      "tmix_mix6_3d_out(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, "
      "Tensor(b!) out_r, Tensor(c!) out_w, Tensor(d!) out_k, Tensor(e!) out_v, Tensor(f!) out_a, Tensor(g!) out_g) -> ()");
  m.def(
      "tmix_mix6_t1_c4096(int B, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, int threads, int vec, bool half_math=False) -> Tensor[]");
  m.def(
      "tmix_kk_a_gate(int B, int T, int C, int H, Tensor k, Tensor k_k, Tensor a0, Tensor a12, Tensor k_a) -> Tensor[]");
  m.def(
      "tmix_kk_a_gate_2d(int B, int T, int C, int H, Tensor k, Tensor k_k, Tensor a0, Tensor a12, Tensor k_a) -> Tensor[]");
  m.def(
      "tmix_kk_a_gate_update_shift(int B, int T, int C, int H, Tensor k, Tensor k_k, Tensor a0, Tensor a12, Tensor k_a, Tensor x, Tensor(a!) shift_state) -> Tensor[]");
  m.def(
      "tmix_lnx_rkvres_xg(int B, int T, int C, int H, Tensor x, Tensor r, Tensor k, Tensor v, "
      "Tensor r_k, Tensor weight, Tensor bias, Tensor g) -> Tensor");
  m.def(
      "tmix_lnx_rkvres_xg_warp(int B, int T, int C, int H, Tensor x, Tensor r, Tensor k, Tensor v, "
      "Tensor r_k, Tensor weight, Tensor bias, Tensor g) -> Tensor");
  m.def(
      "tmix_lnx_rkvres_xg_warp_2d(int B, int T, int C, int H, Tensor x, Tensor r, Tensor k, Tensor v, "
      "Tensor r_k, Tensor weight, Tensor bias, Tensor g) -> Tensor");
  m.def(
      "tmix_vres_gate(int B, int T, int C, Tensor v, Tensor v_first, Tensor v0, Tensor v12) -> Tensor");
  m.def(
      "tmix_vres_gate_cfg(int B, int T, int C, Tensor v, Tensor v_first, Tensor v0, Tensor v12, int threads, bool vectorized) -> Tensor");
  m.def(
      "tmix_vres_gate_cfg_out(int B, int T, int C, Tensor v, Tensor v_first, Tensor v0, Tensor v12, Tensor(a!) out, int threads, bool vectorized) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_fast_ops_fp16, CUDA, m) {
  m.impl("tmix_mix6", &tmix_mix6);
  m.impl("tmix_mix6_cfg", &tmix_mix6_cfg);
  m.impl("tmix_mix6_3d", &tmix_mix6_3d);
  m.impl("tmix_mix6_cfg_out", &tmix_mix6_cfg_out);
  m.impl("tmix_mix6_3d_out", &tmix_mix6_3d_out);
  m.impl("tmix_mix6_t1_c4096", &tmix_mix6_t1_c4096);
  m.impl("tmix_kk_a_gate", &tmix_kk_a_gate);
  m.impl("tmix_kk_a_gate_2d", &tmix_kk_a_gate_2d);
  m.impl("tmix_kk_a_gate_update_shift", &tmix_kk_a_gate_update_shift);
  m.impl("tmix_lnx_rkvres_xg", &tmix_lnx_rkvres_xg);
  m.impl("tmix_lnx_rkvres_xg_warp", &tmix_lnx_rkvres_xg_warp);
  m.impl("tmix_lnx_rkvres_xg_warp_2d", &tmix_lnx_rkvres_xg_warp_2d);
  m.impl("tmix_vres_gate", &tmix_vres_gate);
  m.impl("tmix_vres_gate_cfg", &tmix_vres_gate_cfg);
  m.impl("tmix_vres_gate_cfg_out", &tmix_vres_gate_cfg_out);
}
