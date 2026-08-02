// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.
// Modified by contributors to the FlashRWKV project.
#include <assert.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#include <vector>

using dtype = at::Half;

namespace {

constexpr int HEAD_SIZE = 64;
constexpr int WARPS_PER_BLOCK = 4;
constexpr float KK_NORMALIZE_EPS = 1.0e-12f;
constexpr float TMIX_LN_X_EPS = 64.0e-5f;
constexpr int FFN_SPMV_THREADS = 128;
constexpr int FFN_TILE = 128;

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

__device__ inline __half2 load_h2(const dtype* ptr) {
  return *reinterpret_cast<const __half2*>(ptr);
}

__device__ inline float load_h1(const dtype* ptr) {
  return __half2float(*reinterpret_cast<const __half*>(ptr));
}

__device__ inline void store_h1(dtype* ptr, float value) {
  *reinterpret_cast<__half*>(ptr) = __float2half_rn(value);
}

__device__ inline void store_h2(dtype* ptr, float x0, float x1) {
  *reinterpret_cast<__half2*>(ptr) = __floats2half2_rn(x0, x1);
}

__device__ inline float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ inline float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__global__ void update_shift_state_last_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int b = static_cast<int>(pair_idx / c_pairs);
  const int c = static_cast<int>(pair_idx - static_cast<int64_t>(b) * c_pairs) << 1;
  const int64_t src_idx = (static_cast<int64_t>(b) * T + (T - 1)) * C + c;
  *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(b) * C + c) = load_h2(x + src_idx);
}

__global__ void update_shift_state_last_2d_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state) {
  const int c_pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int c_pairs = C >> 1;
  if (c_pair >= c_pairs) {
    return;
  }

  const int b = static_cast<int>(blockIdx.y);
  const int c = c_pair << 1;
  const int64_t src_idx = (static_cast<int64_t>(b) * T + (T - 1)) * C + c;
  *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(b) * C + c) = load_h2(x + src_idx);
}

template<int THREADS>
__global__ void cmix_sparse_up_one_kernel(
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ key_fc,
    dtype* __restrict__ act) {
  const int f = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float acc = 0.0f;

  const auto x2 = reinterpret_cast<const __half2*>(x);
  const auto p2 = reinterpret_cast<const __half2*>(shift_state);
  const auto k2 = reinterpret_cast<const __half2*>(x_k);
  const auto w2 = reinterpret_cast<const __half2*>(key_fc + static_cast<int64_t>(f) * C);
  const int n = C / 2;
  for (int j = tid; j < n; j += THREADS) {
    const float2 xv = __half22float2(x2[j]);
    const float2 pv = __half22float2(p2[j]);
    const float2 kv = __half22float2(k2[j]);
    const float2 wv = __half22float2(w2[j]);
    acc = fmaf(xv.x + (pv.x - xv.x) * kv.x, wv.x, acc);
    acc = fmaf(xv.y + (pv.y - xv.y) * kv.y, wv.y, acc);
  }

  acc = warp_sum(acc);
  __shared__ float warp_sums[THREADS / 32];
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    float total = lane < (THREADS / 32) ? warp_sums[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) {
      const float relu = fmaxf(total, 0.0f);
      store_h1(act + f, relu * relu);
    }
  }
}

template<int THREADS>
__global__ void cmix_sparse_up_rows_kernel(
    int T,
    int C,
    int F,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ key_fc,
    dtype* __restrict__ act) {
  const int f = blockIdx.x;
  const int row = blockIdx.y;
  const int b = row / T;
  const int t = row - b * T;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float acc = 0.0f;

  const auto x2 = reinterpret_cast<const __half2*>(x + static_cast<int64_t>(row) * C);
  const auto p2 = (t == 0)
      ? reinterpret_cast<const __half2*>(shift_state + static_cast<int64_t>(b) * C)
      : reinterpret_cast<const __half2*>(x + static_cast<int64_t>(row - 1) * C);
  const auto k2 = reinterpret_cast<const __half2*>(x_k);
  const auto w2 = reinterpret_cast<const __half2*>(key_fc + static_cast<int64_t>(f) * C);
  const int n = C / 2;
  for (int j = tid; j < n; j += THREADS) {
    const float2 xv = __half22float2(x2[j]);
    const float2 pv = __half22float2(p2[j]);
    const float2 kv = __half22float2(k2[j]);
    const float2 wv = __half22float2(w2[j]);
    acc = fmaf(xv.x + (pv.x - xv.x) * kv.x, wv.x, acc);
    acc = fmaf(xv.y + (pv.y - xv.y) * kv.y, wv.y, acc);
  }

  acc = warp_sum(acc);
  __shared__ float warp_sums[THREADS / 32];
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    float total = lane < (THREADS / 32) ? warp_sums[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) {
      const float relu = fmaxf(total, 0.0f);
      store_h1(act + static_cast<int64_t>(row) * F + f, relu * relu);
    }
  }
}

__global__ void cmix_sparse_copy_zero_one_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    dtype* __restrict__ out,
    int C) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n4 = C / 8;
  if (i < n4) {
    reinterpret_cast<int4*>(shift_state)[i] = reinterpret_cast<const int4*>(x)[i];
    reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
  }
}

__global__ void cmix_sparse_copy_zero_rows_kernel(
    int B,
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    dtype* __restrict__ out,
    int64_t out_vec4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < out_vec4) {
    reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
  }
  const int64_t state_vec4 = static_cast<int64_t>(B) * (C / 8);
  if (i < state_vec4) {
    const int b = static_cast<int>(i / (C / 8));
    const int c4 = static_cast<int>(i - static_cast<int64_t>(b) * (C / 8));
    reinterpret_cast<int4*>(shift_state)[i] =
        reinterpret_cast<const int4*>(x + (static_cast<int64_t>(b) * T + (T - 1)) * C)[c4];
  }
}

__global__ void zero_vec4_kernel(dtype* __restrict__ out, int64_t n_vec4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_vec4) {
    reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
  }
}

template <bool UpdateShift>
__global__ void cmix_mix_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int64_t bt = pair_idx / c_pairs;
  const int c = static_cast<int>(pair_idx - bt * c_pairs) << 1;
  const int b = static_cast<int>(bt / T);
  const int t = static_cast<int>(bt - static_cast<int64_t>(b) * T);
  const int64_t idx = bt * C + c;

  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 = (t == 0) ? load_h2(shift_state + static_cast<int64_t>(b) * C + c) : load_h2(x + idx - C);
  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float2 mix = __half22float2(load_h2(x_k + c));
  store_h2(out + idx, cur.x + (prev.x - cur.x) * mix.x, cur.y + (prev.y - cur.y) * mix.y);

  if constexpr (UpdateShift) {
    if (t == T - 1) {
      *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(b) * C + c) = cur2;
    }
  }
}

template <bool UpdateShift>
__global__ void cmix_mix_3d_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ out) {
  const int c_pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int c_pairs = C >> 1;
  if (c_pair >= c_pairs) {
    return;
  }

  const int t = static_cast<int>(blockIdx.y);
  const int b = static_cast<int>(blockIdx.z);
  const int c = c_pair << 1;
  const int64_t bt = static_cast<int64_t>(b) * T + t;
  const int64_t idx = bt * C + c;

  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 = (t == 0)
      ? load_h2(shift_state + static_cast<int64_t>(b) * C + c)
      : load_h2(x + idx - C);
  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float2 mix = __half22float2(load_h2(x_k + c));
  store_h2(out + idx, cur.x + (prev.x - cur.x) * mix.x, cur.y + (prev.y - cur.y) * mix.y);

  if constexpr (UpdateShift) {
    // Only T==1 launches this specialization. Each half2 owner first reads and
    // then replaces its own state, so no CTA can overwrite state another CTA reads.
    *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(b) * C + c) = cur2;
  }
}

__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_one_kernel(
    int C,
    const dtype* __restrict__ act,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half mat_row_smem[2][2 * FFN_SPMV_THREADS];
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;

  if (tid < FFN_TILE / 2) {
    *reinterpret_cast<__half2*>(vec_slice + tid * 2) =
        *reinterpret_cast<const __half2*>(act + start_f + tid * 2);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
  }
  atomicAdd(reinterpret_cast<__half2*>(out + c_block * (2 * FFN_SPMV_THREADS) + tid * 2), acc);
}

__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_rows_kernel(
    int C,
    int F,
    const dtype* __restrict__ act,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;
  const dtype* act_row = act + static_cast<int64_t>(row) * F;

  if (tid < FFN_TILE / 2) {
    *reinterpret_cast<__half2*>(vec_slice + tid * 2) =
        *reinterpret_cast<const __half2*>(act_row + start_f + tid * 2);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2),
      acc);
}

template <bool Split2>
__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_relu_one_kernel(
    int C,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;

  if (tid < FFN_TILE) {
    const float v = fmaxf(load_h1(preact + start_f + tid), 0.0f);
    vec_slice[tid] = __float2half_rn(v * v);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  if constexpr (Split2) {
    // Two independent chains shorten the loop-carried HFMA2 dependency. This
    // deliberately changes FP16 association order without dropping precision.
    // Keep the odd-nnz guard: nnz_ids[nnz_count] is not an owned element.
    __half2 acc1;
    *reinterpret_cast<int*>(&acc1) = 0;
    for (int i = 0; i < nnz_count; i += 2) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
    }
    acc = __hadd2(acc, acc1);
  } else {
    for (int i = 0; i < nnz_count; ++i) {
      const int actual_f = start_f + nnz_ids[i];
      const __half2 mat = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
    }
  }
  atomicAdd(reinterpret_cast<__half2*>(out + c_block * (2 * FFN_SPMV_THREADS) + tid * 2), acc);
}

template <bool Split2>
__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_relu_rows_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

  if (tid < FFN_TILE) {
    const float v = fmaxf(load_h1(pre_row + start_f + tid), 0.0f);
    vec_slice[tid] = __float2half_rn(v * v);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  if constexpr (Split2) {
    // The two accumulators own the same output half2 and are combined before
    // the original atomicAdd. No extra CTA writes or output races are added.
    __half2 acc1;
    *reinterpret_cast<int*>(&acc1) = 0;
    for (int i = 0; i < nnz_count; i += 2) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
    }
    acc = __hadd2(acc, acc1);
  } else {
    for (int i = 0; i < nnz_count; ++i) {
      const int actual_f = start_f + nnz_ids[i];
      const __half2 mat = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
    }
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2),
      acc);
}

template <int Accumulators>
__global__ __launch_bounds__(256, 2) void cmix_sparse_spmv_relu_rows_t512_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  constexpr int TILE = 512;
  constexpr int THREADS = 256;
  __shared__ __align__(256) __half vec_slice[TILE];
  __shared__ __align__(256) int nnz_ids[TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[TILE / 32];
  __shared__ int warp_prefix[TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const float v = fmaxf(load_h1(pre_row + start_f + local_f), 0.0f);
    vec_slice[local_f] = __float2half_rn(v * v);
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    if (lane == 0) {
      warp_counts[warp_id + u * (THREADS / 32)] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    const int local_pos = __popc(mask & ((1u << lane) - 1u));
    const int group = warp_id + u * (THREADS / 32);
    if (nonzero) {
      nnz_ids[warp_prefix[group] + local_pos] = local_f;
    }
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  if constexpr (Accumulators == 1) {
    for (int i = 0; i < nnz_count; ++i) {
      const int local_f = nnz_ids[i];
      const int actual_f = start_f + local_f;
      const __half2 mat = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f]), mat, acc);
    }
  } else if constexpr (Accumulators == 2) {
    __half2 acc1;
    *reinterpret_cast<int*>(&acc1) = 0;
    for (int i = 0; i < nnz_count; i += 2) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      // nnz_count is data-dependent and may be odd. Do not speculatively read
      // nnz_ids[i + 1]: it is not owned by this compacted page in that case.
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
    }
    acc = __hadd2(acc, acc1);
  } else {
    static_assert(Accumulators == 4, "unsupported t512 accumulator count");
    __half2 acc1;
    __half2 acc2;
    __half2 acc3;
    *reinterpret_cast<int*>(&acc1) = 0;
    *reinterpret_cast<int*>(&acc2) = 0;
    *reinterpret_cast<int*>(&acc3) = 0;
    for (int i = 0; i < nnz_count; i += 4) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
      if (i + 2 < nnz_count) {
        const int local_f2 = nnz_ids[i + 2];
        const __half2 mat2 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f2) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc2 = __hfma2(__half2half2(vec_slice[local_f2]), mat2, acc2);
      }
      if (i + 3 < nnz_count) {
        const int local_f3 = nnz_ids[i + 3];
        const __half2 mat3 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f3) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc3 = __hfma2(__half2half2(vec_slice[local_f3]), mat3, acc3);
      }
    }
    acc = __hadd2(__hadd2(acc, acc1), __hadd2(acc2, acc3));
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + c_block * (2 * THREADS) + tid * 2),
      acc);
}

template <int Accumulators>
__global__ __launch_bounds__(256, 2) void cmix_sparse_spmv_relu_rows_t512_reuse_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  static_assert(Accumulators == 1 || Accumulators == 2,
                "unsupported reuse accumulator count");
  constexpr int TILE = 512;
  constexpr int THREADS = 256;
  constexpr int WARPS = THREADS / 32;
  __shared__ __align__(256) __half vec_slice[TILE];
  __shared__ __align__(256) int nnz_ids[TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[TILE / 32];
  __shared__ int warp_prefix[TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const float v = fmaxf(load_h1(pre_row + start_f + local_f), 0.0f);
    vec_slice[local_f] = __float2half_rn(v * v);
  }
  __syncthreads();

  // These per-thread masks intentionally remain live across two block
  // barriers. They eliminate the second shared reload/test/ballot, but must
  // never be moved to shared storage where warps would race on ownership.
  unsigned masks[2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    masks[u] = mask;
    if (lane == 0) {
      warp_counts[warp_id + u * WARPS] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const unsigned mask = masks[u];
    const bool nonzero = (mask & (1u << lane)) != 0;
    const int local_pos = __popc(mask & ((1u << lane) - 1u));
    const int group = warp_id + u * WARPS;
    if (nonzero) {
      nnz_ids[warp_prefix[group] + local_pos] = local_f;
    }
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  if constexpr (Accumulators == 1) {
    for (int i = 0; i < nnz_count; ++i) {
      const int local_f = nnz_ids[i];
      const __half2 mat = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f) * C +
          c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f]), mat, acc);
    }
  } else {
    __half2 acc1;
    *reinterpret_cast<int*>(&acc1) = 0;
    for (int i = 0; i < nnz_count; i += 2) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
    }
    acc = __hadd2(acc, acc1);
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C +
                                 c_block * (2 * THREADS) + tid * 2),
      acc);
}

}  // namespace

at::Tensor cmix_sparse_one_cuda(
    int C,
    int F,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_k,
    at::Tensor key_fc,
    at::Tensor value_fc) {
  auto act = at::empty({F}, x.options());
  auto out = at::empty({1, 1, C}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  cmix_sparse_up_one_kernel<64><<<F, 64, 0, stream>>>(
      C,
      x.data_ptr<dtype>(),
      shift_state.data_ptr<dtype>(),
      x_k.data_ptr<dtype>(),
      key_fc.data_ptr<dtype>(),
      act.data_ptr<dtype>());
  cmix_sparse_copy_zero_one_kernel<<<(C / 8 + 127) / 128, 128, 0, stream>>>(
      x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), out.data_ptr<dtype>(), C);
  cmix_sparse_spmv_one_kernel<<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), 1), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      act.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_rows_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_k,
    at::Tensor key_fc,
    at::Tensor value_fc) {
  const int rows = B * T;
  auto act = at::empty({rows, F}, x.options());
  auto out = at::empty({B, T, C}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  cmix_sparse_up_rows_kernel<64><<<dim3(F, rows, 1), 64, 0, stream>>>(
      T,
      C,
      F,
      x.data_ptr<dtype>(),
      shift_state.data_ptr<dtype>(),
      x_k.data_ptr<dtype>(),
      key_fc.data_ptr<dtype>(),
      act.data_ptr<dtype>());
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  cmix_sparse_copy_zero_rows_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(
      B,
      T,
      C,
      x.data_ptr<dtype>(),
      shift_state.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      out_vec4);
  cmix_sparse_spmv_rows_kernel<<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), rows), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      F,
      act.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_one_cuda(
    int C,
    int F,
    at::Tensor act,
    at::Tensor value_fc) {
  auto out = at::empty({1, 1, C}, act.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  zero_vec4_kernel<<<(C / 8 + 127) / 128, 128, 0, stream>>>(out.data_ptr<dtype>(), C / 8);
  cmix_sparse_spmv_one_kernel<<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), 1), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      act.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_rows_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor act,
    at::Tensor value_fc) {
  const int rows = B * T;
  auto out = at::empty({B, T, C}, act.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(out.data_ptr<dtype>(), out_vec4);
  cmix_sparse_spmv_rows_kernel<<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), rows), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      F,
      act.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_one_cuda(
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc) {
  auto out = at::empty({1, 1, C}, preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  zero_vec4_kernel<<<(C / 8 + 127) / 128, 128, 0, stream>>>(out.data_ptr<dtype>(), C / 8);
  cmix_sparse_spmv_relu_one_kernel<false><<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), 1), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      preact.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_one_split2_cuda(
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc) {
  auto out = at::empty({1, 1, C}, preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  zero_vec4_kernel<<<(C / 8 + 127) / 128, 128, 0, stream>>>(out.data_ptr<dtype>(), C / 8);
  cmix_sparse_spmv_relu_one_kernel<true><<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), 1), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      preact.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_rows_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc) {
  const int rows = B * T;
  auto out = at::empty({B, T, C}, preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(out.data_ptr<dtype>(), out_vec4);
  cmix_sparse_spmv_relu_rows_kernel<false><<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), rows), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      F,
      preact.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_rows_split2_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc) {
  const int rows = B * T;
  auto out = at::empty({B, T, C}, preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(out.data_ptr<dtype>(), out_vec4);
  cmix_sparse_spmv_relu_rows_kernel<true><<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), rows), FFN_SPMV_THREADS, 0, stream>>>(
      C,
      F,
      preact.data_ptr<dtype>(),
      value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void launch_cmix_sparse_down_relu_rows_t512(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc,
    at::Tensor out,
    int accumulators) {
  const int rows = B * T;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(out.data_ptr<dtype>(), out_vec4);
  if (accumulators == 1) {
    cmix_sparse_spmv_relu_rows_t512_kernel<1><<<dim3(F / 512, C / 512, rows), 256, 0, stream>>>(
        C, F, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(), out.data_ptr<dtype>());
  } else if (accumulators == 2) {
    cmix_sparse_spmv_relu_rows_t512_kernel<2><<<dim3(F / 512, C / 512, rows), 256, 0, stream>>>(
        C, F, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(), out.data_ptr<dtype>());
  } else {
    cmix_sparse_spmv_relu_rows_t512_kernel<4><<<dim3(F / 512, C / 512, rows), 256, 0, stream>>>(
        C, F, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(), out.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor cmix_sparse_down_relu_rows_t512_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc) {
  auto out = at::empty({B, T, C}, preact.options());
  launch_cmix_sparse_down_relu_rows_t512(B, T, C, F, preact, value_fc, out, 1);
  return out;
}

at::Tensor cmix_sparse_down_relu_rows_t512_cfg_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc,
    int accumulators) {
  auto out = at::empty({B, T, C}, preact.options());
  launch_cmix_sparse_down_relu_rows_t512(
      B, T, C, F, preact, value_fc, out, accumulators);
  return out;
}

void cmix_sparse_down_relu_rows_t512_cfg_out_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc,
    at::Tensor out,
    int accumulators) {
  launch_cmix_sparse_down_relu_rows_t512(
      B, T, C, F, preact, value_fc, out, accumulators);
}

void launch_cmix_sparse_down_relu_rows_t512_reuse(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc,
    at::Tensor out,
    int accumulators) {
  const int rows = B * T;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(
      out.data_ptr<dtype>(), out_vec4);
  if (accumulators == 1) {
    cmix_sparse_spmv_relu_rows_t512_reuse_kernel<1>
        <<<dim3(F / 512, C / 512, rows), 256, 0, stream>>>(
            C, F, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(),
            out.data_ptr<dtype>());
  } else {
    cmix_sparse_spmv_relu_rows_t512_reuse_kernel<2>
        <<<dim3(F / 512, C / 512, rows), 256, 0, stream>>>(
            C, F, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(),
            out.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor cmix_sparse_down_relu_rows_t512_reuse_cfg_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc,
    int accumulators) {
  auto out = at::empty({B, T, C}, preact.options());
  launch_cmix_sparse_down_relu_rows_t512_reuse(
      B, T, C, F, preact, value_fc, out, accumulators);
  return out;
}

void cmix_sparse_down_relu_rows_t512_reuse_cfg_out_cuda(
    int B,
    int T,
    int C,
    int F,
    at::Tensor preact,
    at::Tensor value_fc,
    at::Tensor out,
    int accumulators) {
  launch_cmix_sparse_down_relu_rows_t512_reuse(
      B, T, C, F, preact, value_fc, out, accumulators);
}

at::Tensor cmix_mix_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_k) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 1) {
    cmix_mix_kernel<true><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T,
        C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        out.data_ptr<dtype>(),
        total_pairs);
  } else {
    cmix_mix_kernel<false><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T,
        C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        out.data_ptr<dtype>(),
        total_pairs);
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    update_shift_state_last_kernel<<<static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), state_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_mix_cfg_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_k,
    int threads) {
  auto out = at::empty_like(x);
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 1) {
    cmix_mix_kernel<true><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T,
        C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        out.data_ptr<dtype>(),
        total_pairs);
  } else {
    cmix_mix_kernel<false><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T,
        C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        out.data_ptr<dtype>(),
        total_pairs);
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    update_shift_state_last_kernel<<<static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), state_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void cmix_mix_cfg_out_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_k,
    at::Tensor out,
    int threads) {
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 1) {
    cmix_mix_kernel<true><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(), out.data_ptr<dtype>(), total_pairs);
  } else {
    cmix_mix_kernel<false><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(), out.data_ptr<dtype>(), total_pairs);
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    update_shift_state_last_kernel<<<static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), state_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_cmix_mix_3d(
    int B,
    int T,
    int C,
    const at::Tensor& x,
    at::Tensor& shift_state,
    const at::Tensor& x_k,
    at::Tensor& out) {
  constexpr int threads = 256;
  const int c_pairs = C / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 mix_grid(static_cast<unsigned int>(ceil_div(c_pairs, threads)),
                      static_cast<unsigned int>(T),
                      static_cast<unsigned int>(B));
  if (T == 1) {
    cmix_mix_3d_kernel<true><<<mix_grid, threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(), out.data_ptr<dtype>());
  } else {
    cmix_mix_3d_kernel<false><<<mix_grid, threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(), out.data_ptr<dtype>());
    // T>1 must remain a second launch: an in-grid last-token write can race
    // with a different CTA that is still reading the old recurrent state.
    const dim3 state_grid(static_cast<unsigned int>(ceil_div(c_pairs, threads)),
                          static_cast<unsigned int>(B));
    update_shift_state_last_2d_kernel<<<state_grid, threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor cmix_mix_3d_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_k) {
  auto out = at::empty_like(x);
  launch_cmix_mix_3d(B, T, C, x, shift_state, x_k, out);
  return out;
}

void cmix_mix_3d_out_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_k,
    at::Tensor out) {
  launch_cmix_mix_3d(B, T, C, x, shift_state, x_k, out);
}

