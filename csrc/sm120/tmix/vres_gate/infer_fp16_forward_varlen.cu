// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
//
// Source: BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_fast_ops_fp16.cu,
// revision ee3308f6922e59f2166c7fac3c5a192340a2b48e.
//
// The scalar and vectorized value-residual gate bodies are copied from
// Albatross.  Packed rows are already the native row layout of this operator;
// only the launch metadata is changed from B*T to total_tokens.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

using dtype = at::Half;

namespace {

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

__device__ inline float load_h1(const dtype* ptr) {
  return __half2float(*reinterpret_cast<const __half*>(ptr));
}

__device__ inline void store_h1(dtype* ptr, float value) {
  *reinterpret_cast<__half*>(ptr) = __float2half_rn(value);
}

__device__ inline __half2 load_h2(const dtype* ptr) {
  return *reinterpret_cast<const __half2*>(ptr);
}

__device__ inline void store_h2(dtype* ptr, float x0, float x1) {
  *reinterpret_cast<__half2*>(ptr) = __floats2half2_rn(x0, x1);
}

__device__ inline float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// Exact Albatross tmix_vres_gate_kernel.
__global__ void tmix_vres_gate_kernel(
    int C,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    const dtype* __restrict__ v12,
    dtype* __restrict__ out,
    int64_t total) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int c = static_cast<int>(idx % static_cast<int64_t>(C));
  const float vv = load_h1(v + idx);
  const float gate = sigmoid_fast(load_h1(v0 + c) + load_h1(v12 + idx));
  store_h1(out + idx, fmaf(load_h1(v_first + idx) - vv, gate, vv));
}

// Exact Albatross tmix_vres_gate_vec2_kernel.
template <int Threads>
__global__ __launch_bounds__(Threads) void tmix_vres_gate_vec2_kernel(
    int C,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    const dtype* __restrict__ v12,
    dtype* __restrict__ out,
    int64_t rows) {
  const int pair = static_cast<int>(blockIdx.x) * Threads + threadIdx.x;
  const int pairs_per_row = C >> 1;
  const int64_t row = blockIdx.y;
  if (pair >= pairs_per_row || row >= rows) {
    return;
  }
  const int c = pair << 1;
  const int64_t idx = row * C + c;
  const float2 vv = __half22float2(load_h2(v + idx));
  const float2 vf = __half22float2(load_h2(v_first + idx));
  const float2 base = __half22float2(load_h2(v0 + c));
  const float2 delta = __half22float2(load_h2(v12 + idx));
  const float gate0 = sigmoid_fast(base.x + delta.x);
  const float gate1 = sigmoid_fast(base.y + delta.y);
  store_h2(
      out + idx,
      fmaf(vf.x - vv.x, gate0, vv.x),
      fmaf(vf.y - vv.y, gate1, vv.y));
}

template <int Threads>
void launch_vec2(
    int C,
    int64_t rows,
    const at::Tensor& v,
    const at::Tensor& v_first,
    const at::Tensor& v0,
    const at::Tensor& v12,
    at::Tensor& out,
    cudaStream_t stream) {
  const int pairs_per_row = C >> 1;
  tmix_vres_gate_vec2_kernel<Threads><<<
      dim3(static_cast<unsigned int>(ceil_div(pairs_per_row, Threads)),
           static_cast<unsigned int>(rows), 1),
      Threads, 0, stream>>>(
      C, v.data_ptr<dtype>(), v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(),
      v12.data_ptr<dtype>(), out.data_ptr<dtype>(), rows);
}

}  // namespace

void tmix_vres_gate_forward_varlen_cuda(
    int total_tokens,
    int channels,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor v12,
    torch::Tensor output) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t rows = total_tokens;

  // This is the canonical tuned policy in rwkv7_fast_v3a.py expressed in the
  // packed API: vector2 is admitted only for C==4096 and the measured row
  // range.  Other shapes use the exact scalar family.
  const bool use_vec2 = channels == 4096 && rows >= 64 && rows <= 65535;
  if (use_vec2) {
    if (rows < 256) {
      launch_vec2<128>(channels, rows, v, v_first, v0, v12, output, stream);
    } else {
      launch_vec2<256>(channels, rows, v, v_first, v0, v12, output, stream);
    }
  } else {
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(total_tokens) * channels;
    tmix_vres_gate_kernel<<<
        static_cast<unsigned int>(ceil_div(total, threads)), threads, 0, stream>>>(
        channels, v.data_ptr<dtype>(), v_first.data_ptr<dtype>(),
        v0.data_ptr<dtype>(), v12.data_ptr<dtype>(), output.data_ptr<dtype>(), total);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
