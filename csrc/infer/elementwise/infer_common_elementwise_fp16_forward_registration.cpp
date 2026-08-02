// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.
// Modified by contributors to the FlashRWKV project.
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

torch::Tensor relu_square_cuda(torch::Tensor x);

torch::Tensor act_tanh_cuda(torch::Tensor x);

torch::Tensor act_sigmoid_cuda(torch::Tensor x);

torch::Tensor add_vec_cuda(int C, torch::Tensor x, torch::Tensor vec);
torch::Tensor add_vec_2d_cuda(int C, torch::Tensor x, torch::Tensor vec);
void add_vec_cfg_out_cuda(
    int C,
    torch::Tensor x,
    torch::Tensor vec,
    torch::Tensor out,
    bool grid2d);

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

torch::Tensor relu_square(torch::Tensor x) {
  check_half_cuda_contig(x, "x");
  TORCH_CHECK((x.numel() % 2) == 0, "x.numel() must be even");
  return relu_square_cuda(x);
}

torch::Tensor act_tanh(torch::Tensor x) {
  check_half_cuda_contig(x, "x");
  TORCH_CHECK((x.numel() % 2) == 0, "x.numel() must be even");
  return act_tanh_cuda(x);
}

torch::Tensor act_sigmoid(torch::Tensor x) {
  check_half_cuda_contig(x, "x");
  TORCH_CHECK((x.numel() % 2) == 0, "x.numel() must be even");
  return act_sigmoid_cuda(x);
}

int64_t check_add_vec_inputs(
    int64_t C,
    const torch::Tensor& x,
    const torch::Tensor& vec,
    bool grid2d) {
  TORCH_CHECK(C > 0 && C <= std::numeric_limits<int>::max() && (C % 2) == 0,
              "C must be positive even int32");
  check_half_cuda_contig(x, "x");
  check_vec(vec, C, "vec");
  TORCH_CHECK(x.numel() > 0 && (x.numel() % C) == 0,
              "x must contain complete C-wide rows");
  TORCH_CHECK(x.size(-1) == C, "x last dim must equal C");
  check_half2_same_device({{&x, "x"}, {&vec, "vec"}});
  // Both CUDA pointers are restrict-qualified, so even read/read aliasing is
  // outside the kernel contract and must be rejected before launch.
  check_no_storage_overlap(x, "x", vec, "vec");
  const int64_t rows = x.numel() / C;
  if (grid2d) {
    TORCH_CHECK(rows <= 65535, "2D add_vec requires rows <= 65535");
  }
  return rows;
}

torch::Tensor add_vec(int64_t C, torch::Tensor x, torch::Tensor vec) {
  check_add_vec_inputs(C, x, vec, false);
  return add_vec_cuda(static_cast<int>(C), x, vec);
}

torch::Tensor add_vec_2d(int64_t C, torch::Tensor x, torch::Tensor vec) {
  check_add_vec_inputs(C, x, vec, true);
  return add_vec_2d_cuda(static_cast<int>(C), x, vec);
}

void add_vec_cfg_out(
    int64_t C,
    torch::Tensor x,
    torch::Tensor vec,
    torch::Tensor out,
    bool grid2d) {
  check_add_vec_inputs(C, x, vec, grid2d);
  check_half_cuda_contig(out, "out");
  TORCH_CHECK(out.sizes() == x.sizes(), "out shape must match x");
  check_half2_same_device({{&x, "x"}, {&vec, "vec"}, {&out, "out"}});
  // The tuning entrypoint writes a caller-owned output; reject even shifted
  // aliases because restrict would otherwise turn overlap into undefined behavior.
  check_no_storage_overlap(out, "out", x, "x");
  check_no_storage_overlap(out, "out", vec, "vec");
  add_vec_cfg_out_cuda(static_cast<int>(C), x, vec, out, grid2d);
}

TORCH_LIBRARY_FRAGMENT(rwkv7_fast_ops_fp16, m) {
  m.def("relu_square(Tensor x) -> Tensor");
  m.def("act_tanh(Tensor x) -> Tensor");
  m.def("act_sigmoid(Tensor x) -> Tensor");
  m.def("add_vec(int C, Tensor x, Tensor vec) -> Tensor");
  m.def("add_vec_2d(int C, Tensor x, Tensor vec) -> Tensor");
  m.def("add_vec_cfg_out(int C, Tensor x, Tensor vec, Tensor(a!) out, bool grid2d) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_fast_ops_fp16, CUDA, m) {
  m.impl("relu_square", &relu_square);
  m.impl("act_tanh", &act_tanh);
  m.impl("act_sigmoid", &act_sigmoid);
  m.impl("add_vec", &add_vec);
  m.impl("add_vec_2d", &add_vec_2d);
  m.impl("add_vec_cfg_out", &add_vec_cfg_out);
}
