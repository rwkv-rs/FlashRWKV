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

template <bool UpdateShift>
__global__ void tmix_mix6_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
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
  __half2 prev2;
  if (t == 0) {
    prev2 = load_h2(shift_state + static_cast<int64_t>(b) * C + c);
  } else {
    prev2 = load_h2(x + idx - C);
  }

  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float dx0 = prev.x - cur.x;
  const float dx1 = prev.y - cur.y;

  const float2 xr = __half22float2(load_h2(x_r + c));
  const float2 xw = __half22float2(load_h2(x_w + c));
  const float2 xk = __half22float2(load_h2(x_k + c));
  const float2 xv = __half22float2(load_h2(x_v + c));
  const float2 xa = __half22float2(load_h2(x_a + c));
  const float2 xg = __half22float2(load_h2(x_g + c));

  store_h2(out_r + idx, cur.x + dx0 * xr.x, cur.y + dx1 * xr.y);
  store_h2(out_w + idx, cur.x + dx0 * xw.x, cur.y + dx1 * xw.y);
  store_h2(out_k + idx, cur.x + dx0 * xk.x, cur.y + dx1 * xk.y);
  store_h2(out_v + idx, cur.x + dx0 * xv.x, cur.y + dx1 * xv.y);
  store_h2(out_a + idx, cur.x + dx0 * xa.x, cur.y + dx1 * xa.y);
  store_h2(out_g + idx, cur.x + dx0 * xg.x, cur.y + dx1 * xg.y);

  if constexpr (UpdateShift) {
    if (t == T - 1) {
      *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(b) * C + c) = cur2;
    }
  }
}

template <bool UpdateShift>
__global__ void tmix_mix6_3d_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g) {
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
  const float dx0 = prev.x - cur.x;
  const float dx1 = prev.y - cur.y;

  const float2 xr = __half22float2(load_h2(x_r + c));
  const float2 xw = __half22float2(load_h2(x_w + c));
  const float2 xk = __half22float2(load_h2(x_k + c));
  const float2 xv = __half22float2(load_h2(x_v + c));
  const float2 xa = __half22float2(load_h2(x_a + c));
  const float2 xg = __half22float2(load_h2(x_g + c));

  store_h2(out_r + idx, cur.x + dx0 * xr.x, cur.y + dx1 * xr.y);
  store_h2(out_w + idx, cur.x + dx0 * xw.x, cur.y + dx1 * xw.y);
  store_h2(out_k + idx, cur.x + dx0 * xk.x, cur.y + dx1 * xk.y);
  store_h2(out_v + idx, cur.x + dx0 * xv.x, cur.y + dx1 * xv.y);
  store_h2(out_a + idx, cur.x + dx0 * xa.x, cur.y + dx1 * xa.y);
  store_h2(out_g + idx, cur.x + dx0 * xg.x, cur.y + dx1 * xg.y);

  if constexpr (UpdateShift) {
    // This specialization is only legal for T==1; each half2 has one owner.
    *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(b) * C + c) = cur2;
  }
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

__device__ inline int packed_sequence_for_token(
    const int* __restrict__ cu_seqlens,
    int num_sequences,
    int token) {
  int low = 0;
  int high = num_sequences;
  while (low < high) {
    const int middle = low + (high - low) / 2;
    if (cu_seqlens[middle + 1] <= token) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

template <bool HasTokenBatchIndices>
__global__ void tmix_mix6_varlen_kernel(
    int num_sequences,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ state_pool,
    const int* __restrict__ state_indices,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ token_batch_indices,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    int64_t total_pairs) {
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int token = static_cast<int>(pair_idx / c_pairs);
  const int c = static_cast<int>(pair_idx - static_cast<int64_t>(token) * c_pairs) << 1;
  const int sequence = HasTokenBatchIndices
      ? token_batch_indices[token]
      : packed_sequence_for_token(cu_seqlens, num_sequences, token);
  const int token_start = cu_seqlens[sequence];
  const int64_t idx = static_cast<int64_t>(token) * C + c;
  const int state_row = state_indices[sequence];

  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 = token == token_start
      ? load_h2(state_pool + static_cast<int64_t>(state_row) * C + c)
      : load_h2(x + idx - C);
  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float dx0 = prev.x - cur.x;
  const float dx1 = prev.y - cur.y;

  const float2 xr = __half22float2(load_h2(x_r + c));
  const float2 xw = __half22float2(load_h2(x_w + c));
  const float2 xk = __half22float2(load_h2(x_k + c));
  const float2 xv = __half22float2(load_h2(x_v + c));
  const float2 xa = __half22float2(load_h2(x_a + c));
  const float2 xg = __half22float2(load_h2(x_g + c));

  store_h2(out_r + idx, cur.x + dx0 * xr.x, cur.y + dx1 * xr.y);
  store_h2(out_w + idx, cur.x + dx0 * xw.x, cur.y + dx1 * xw.y);
  store_h2(out_k + idx, cur.x + dx0 * xk.x, cur.y + dx1 * xk.y);
  store_h2(out_v + idx, cur.x + dx0 * xv.x, cur.y + dx1 * xv.y);
  store_h2(out_a + idx, cur.x + dx0 * xa.x, cur.y + dx1 * xa.y);
  store_h2(out_g + idx, cur.x + dx0 * xg.x, cur.y + dx1 * xg.y);
}

__global__ void update_shift_state_varlen_kernel(
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ state_pool,
    const int* __restrict__ state_indices,
    const int* __restrict__ cu_seqlens,
    int num_sequences,
    int64_t total_pairs) {
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int c_pairs = C >> 1;
  const int sequence = static_cast<int>(pair_idx / c_pairs);
  const int c = static_cast<int>(pair_idx - static_cast<int64_t>(sequence) * c_pairs) << 1;
  if (sequence >= num_sequences) {
    return;
  }
  const int last_token = cu_seqlens[sequence + 1] - 1;
  const int state_row = state_indices[sequence];
  *reinterpret_cast<__half2*>(state_pool + static_cast<int64_t>(state_row) * C + c) =
      load_h2(x + static_cast<int64_t>(last_token) * C + c);
}

template <bool HalfMath, int Vec>
__global__ void tmix_mix6_t1_c4096_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    int64_t total_pairs) {
  const int64_t base_pair = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * Vec;
#pragma unroll
  for (int u = 0; u < Vec; ++u) {
  const int64_t pair_idx = base_pair + u;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int c = static_cast<int>(pair_idx & 2047) << 1;
  const int64_t idx = pair_idx << 1;
  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 = load_h2(shift_state + idx);
  if constexpr (HalfMath) {
    const __half2 dx = __hsub2(prev2, cur2);
    *reinterpret_cast<__half2*>(out_r + idx) = __hfma2(dx, load_h2(x_r + c), cur2);
    *reinterpret_cast<__half2*>(out_w + idx) = __hfma2(dx, load_h2(x_w + c), cur2);
    *reinterpret_cast<__half2*>(out_k + idx) = __hfma2(dx, load_h2(x_k + c), cur2);
    *reinterpret_cast<__half2*>(out_v + idx) = __hfma2(dx, load_h2(x_v + c), cur2);
    *reinterpret_cast<__half2*>(out_a + idx) = __hfma2(dx, load_h2(x_a + c), cur2);
    *reinterpret_cast<__half2*>(out_g + idx) = __hfma2(dx, load_h2(x_g + c), cur2);
  } else {
    const float2 cur = __half22float2(cur2);
    const float2 prev = __half22float2(prev2);
    const float dx0 = prev.x - cur.x;
    const float dx1 = prev.y - cur.y;
    const float2 xr = __half22float2(load_h2(x_r + c));
    const float2 xw = __half22float2(load_h2(x_w + c));
    const float2 xk = __half22float2(load_h2(x_k + c));
    const float2 xv = __half22float2(load_h2(x_v + c));
    const float2 xa = __half22float2(load_h2(x_a + c));
    const float2 xg = __half22float2(load_h2(x_g + c));
    store_h2(out_r + idx, cur.x + dx0 * xr.x, cur.y + dx1 * xr.y);
    store_h2(out_w + idx, cur.x + dx0 * xw.x, cur.y + dx1 * xw.y);
    store_h2(out_k + idx, cur.x + dx0 * xk.x, cur.y + dx1 * xk.y);
    store_h2(out_v + idx, cur.x + dx0 * xv.x, cur.y + dx1 * xv.y);
    store_h2(out_a + idx, cur.x + dx0 * xa.x, cur.y + dx1 * xa.y);
    store_h2(out_g + idx, cur.x + dx0 * xg.x, cur.y + dx1 * xg.y);
  }
  *reinterpret_cast<__half2*>(shift_state + idx) = cur2;
  }
}

template <bool UpdateShift>
__global__ void tmix_kk_a_gate_kernel(
    int H,
    const dtype* __restrict__ k,
    const dtype* __restrict__ k_k,
    const dtype* __restrict__ a0,
    const dtype* __restrict__ a12,
    const dtype* __restrict__ k_a,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    dtype* __restrict__ new_k,
    dtype* __restrict__ neg_kk,
    dtype* __restrict__ kka,
    int64_t bth_size) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t bth = static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (bth >= bth_size) {
    return;
  }

  const int64_t h = bth % H;
  const int64_t base = bth * HEAD_SIZE;
  const int64_t c = h * HEAD_SIZE + static_cast<int64_t>(lane) * 2;
  const int64_t idx = base + static_cast<int64_t>(lane) * 2;

  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 kk_scale = __half22float2(load_h2(k_k + c));
  const float u0 = kv.x * kk_scale.x;
  const float u1 = kv.y * kk_scale.y;

  float sum_sq = u0 * u0 + u1 * u1;
  sum_sq = warp_sum(sum_sq);
  const float total = __shfl_sync(0xffffffffu, sum_sq, 0);
  const float inv_d = 1.0f / fmaxf(sqrtf(total), KK_NORMALIZE_EPS);
  const float kk0 = u0 * inv_d;
  const float kk1 = u1 * inv_d;

  const float2 a0v = __half22float2(load_h2(a0 + c));
  const float2 a12v = __half22float2(load_h2(a12 + idx));
  const float av0 = sigmoid_fast(a0v.x + a12v.x);
  const float av1 = sigmoid_fast(a0v.y + a12v.y);
  const float2 ka = __half22float2(load_h2(k_a + c));
  store_h2(new_k + idx, kv.x * fmaf(av0, ka.x, 1.0f - ka.x), kv.y * fmaf(av1, ka.y, 1.0f - ka.y));
  store_h2(neg_kk + idx, -kk0, -kk1);
  store_h2(kka + idx, kk0 * av0, kk1 * av1);
  if constexpr (UpdateShift) {
    *reinterpret_cast<__half2*>(shift_state + idx) = load_h2(x + idx);
  }
}

template <bool UpdateShift>
__global__ void tmix_kk_a_gate_2d_kernel(
    int H,
    const dtype* __restrict__ k,
    const dtype* __restrict__ k_k,
    const dtype* __restrict__ a0,
    const dtype* __restrict__ a12,
    const dtype* __restrict__ k_a,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    dtype* __restrict__ new_k,
    dtype* __restrict__ neg_kk,
    dtype* __restrict__ kka) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int h = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (h >= H) {
    return;
  }

  const int64_t row = blockIdx.y;
  const int64_t bth = row * H + h;
  const int64_t base = bth * HEAD_SIZE;
  const int64_t c = static_cast<int64_t>(h) * HEAD_SIZE + static_cast<int64_t>(lane) * 2;
  const int64_t idx = base + static_cast<int64_t>(lane) * 2;

  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 kk_scale = __half22float2(load_h2(k_k + c));
  const float u0 = kv.x * kk_scale.x;
  const float u1 = kv.y * kk_scale.y;

  float sum_sq = u0 * u0 + u1 * u1;
  sum_sq = warp_sum(sum_sq);
  const float total = __shfl_sync(0xffffffffu, sum_sq, 0);
  const float inv_d = 1.0f / fmaxf(sqrtf(total), KK_NORMALIZE_EPS);
  const float kk0 = u0 * inv_d;
  const float kk1 = u1 * inv_d;

  const float2 a0v = __half22float2(load_h2(a0 + c));
  const float2 a12v = __half22float2(load_h2(a12 + idx));
  const float av0 = sigmoid_fast(a0v.x + a12v.x);
  const float av1 = sigmoid_fast(a0v.y + a12v.y);
  const float2 ka = __half22float2(load_h2(k_a + c));
  store_h2(new_k + idx, kv.x * fmaf(av0, ka.x, 1.0f - ka.x), kv.y * fmaf(av1, ka.y, 1.0f - ka.y));
  store_h2(neg_kk + idx, -kk0, -kk1);
  store_h2(kka + idx, kk0 * av0, kk1 * av1);
  if constexpr (UpdateShift) {
    *reinterpret_cast<__half2*>(shift_state + idx) = load_h2(x + idx);
  }
}

__global__ void tmix_lnx_rkvres_xg_kernel(
    int C,
    int H,
    const dtype* __restrict__ x,
    const dtype* __restrict__ r,
    const dtype* __restrict__ k,
    const dtype* __restrict__ v,
    const dtype* __restrict__ r_k,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ g,
    dtype* __restrict__ out,
    int64_t bth_size) {
  __shared__ float partial[2];
  const int bth = blockIdx.x;
  if (bth >= bth_size) {
    return;
  }
  const int lane = threadIdx.x;
  const int warp = lane >> 5;
  const int warp_lane = lane & 31;
  const int h = bth % H;
  const int64_t base = static_cast<int64_t>(bth) * HEAD_SIZE;
  const int64_t cbase = static_cast<int64_t>(h) * HEAD_SIZE;
  const int64_t idx = base + lane;
  const int64_t c = cbase + lane;

  const float xv = load_h1(x + idx);
  float sum = xv;
  sum = warp_sum(sum);
  if (warp_lane == 0) {
    partial[warp] = sum;
  }
  __syncthreads();
  const float mean = (partial[0] + partial[1]) * (1.0f / 64.0f);
  __syncthreads();

  const float d = xv - mean;
  float ss = d * d;
  ss = warp_sum(ss);
  if (warp_lane == 0) {
    partial[warp] = ss;
  }
  __syncthreads();
  const float var = (partial[0] + partial[1]) * (1.0f / 64.0f);
  const float rstd = rsqrtf(var + TMIX_LN_X_EPS);
  __syncthreads();

  const float rv = load_h1(r + idx);
  const float kv = load_h1(k + idx);
  const float vv = load_h1(v + idx);
  float dot = rv * kv * load_h1(r_k + c);
  dot = warp_sum(dot);
  if (warp_lane == 0) {
    partial[warp] = dot;
  }
  __syncthreads();
  const float rkv = partial[0] + partial[1];
  __syncthreads();

  const float y = (d * rstd * load_h1(weight + c) + load_h1(bias + c) + rkv * vv)
                  * load_h1(g + idx);
  store_h1(out + idx, y);
}

__global__ __launch_bounds__(32) void tmix_lnx_rkvres_xg_warp_kernel(
    int H,
    const dtype* __restrict__ x,
    const dtype* __restrict__ r,
    const dtype* __restrict__ k,
    const dtype* __restrict__ v,
    const dtype* __restrict__ r_k,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ g,
    dtype* __restrict__ out,
    int64_t bth_size) {
  const int64_t bth = blockIdx.x;
  if (bth >= bth_size) {
    return;
  }
  const int lane = threadIdx.x;
  const int h = static_cast<int>(bth % H);
  const int64_t base = bth * HEAD_SIZE;
  const int64_t cbase = static_cast<int64_t>(h) * HEAD_SIZE;
  const int64_t pair = static_cast<int64_t>(lane) * 2;
  const int64_t idx = base + pair;
  const int64_t c = cbase + pair;

  const float2 xv = __half22float2(load_h2(x + idx));
  float sum = warp_sum(xv.x + xv.y);
  const float mean = __shfl_sync(0xffffffffu, sum, 0) * (1.0f / 64.0f);
  const float d0 = xv.x - mean;
  const float d1 = xv.y - mean;
  float ss = warp_sum(d0 * d0 + d1 * d1);
  const float var = __shfl_sync(0xffffffffu, ss, 0) * (1.0f / 64.0f);
  const float rstd = rsqrtf(var + TMIX_LN_X_EPS);

  const float2 rv = __half22float2(load_h2(r + idx));
  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 vv = __half22float2(load_h2(v + idx));
  const float2 rkv_weight = __half22float2(load_h2(r_k + c));
  float dot = warp_sum(rv.x * kv.x * rkv_weight.x + rv.y * kv.y * rkv_weight.y);
  const float rkv = __shfl_sync(0xffffffffu, dot, 0);

  const float2 ln_weight = __half22float2(load_h2(weight + c));
  const float2 ln_bias = __half22float2(load_h2(bias + c));
  const float2 gate = __half22float2(load_h2(g + idx));
  // Pairing adjacent channels changes only the FP32 reduction tree. The model
  // contract permits this small arithmetic-order difference; no precision is removed.
  store_h2(
      out + idx,
      (d0 * rstd * ln_weight.x + ln_bias.x + rkv * vv.x) * gate.x,
      (d1 * rstd * ln_weight.y + ln_bias.y + rkv * vv.y) * gate.y);
}

__global__ __launch_bounds__(32) void tmix_lnx_rkvres_xg_warp_2d_kernel(
    int H,
    const dtype* __restrict__ x,
    const dtype* __restrict__ r,
    const dtype* __restrict__ k,
    const dtype* __restrict__ v,
    const dtype* __restrict__ r_k,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ g,
    dtype* __restrict__ out) {
  const int lane = threadIdx.x;
  const int h = static_cast<int>(blockIdx.x);
  const int64_t row = blockIdx.y;
  const int64_t bth = row * H + h;
  const int64_t base = bth * HEAD_SIZE;
  const int64_t cbase = static_cast<int64_t>(h) * HEAD_SIZE;
  const int64_t pair = static_cast<int64_t>(lane) * 2;
  const int64_t idx = base + pair;
  const int64_t c = cbase + pair;

  const float2 xv = __half22float2(load_h2(x + idx));
  float sum = warp_sum(xv.x + xv.y);
  const float mean = __shfl_sync(0xffffffffu, sum, 0) * (1.0f / 64.0f);
  const float d0 = xv.x - mean;
  const float d1 = xv.y - mean;
  float ss = warp_sum(d0 * d0 + d1 * d1);
  const float var = __shfl_sync(0xffffffffu, ss, 0) * (1.0f / 64.0f);
  const float rstd = rsqrtf(var + TMIX_LN_X_EPS);

  const float2 rv = __half22float2(load_h2(r + idx));
  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 vv = __half22float2(load_h2(v + idx));
  const float2 rkv_weight = __half22float2(load_h2(r_k + c));
  float dot = warp_sum(rv.x * kv.x * rkv_weight.x + rv.y * kv.y * rkv_weight.y);
  const float rkv = __shfl_sync(0xffffffffu, dot, 0);

  const float2 ln_weight = __half22float2(load_h2(weight + c));
  const float2 ln_bias = __half22float2(load_h2(bias + c));
  const float2 gate = __half22float2(load_h2(g + idx));
  store_h2(
      out + idx,
      (d0 * rstd * ln_weight.x + ln_bias.x + rkv * vv.x) * gate.x,
      (d1 * rstd * ln_weight.y + ln_bias.y + rkv * vv.y) * gate.y);
}

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


}  // namespace

std::vector<at::Tensor> tmix_mix6_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g) {
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 1) {
    tmix_mix6_kernel<true><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(),
        x_w.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(),
        x_a.data_ptr<dtype>(),
        x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(),
        out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(),
        out_a.data_ptr<dtype>(),
        out_g.data_ptr<dtype>(),
        total_pairs);
  } else {
    tmix_mix6_kernel<false><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(),
        x_w.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(),
        x_a.data_ptr<dtype>(),
        x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(),
        out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(),
        out_a.data_ptr<dtype>(),
        out_g.data_ptr<dtype>(),
        total_pairs);
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    update_shift_state_last_kernel<<<static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), state_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> tmix_mix6_varlen_cuda(
    int num_sequences,
    int total_tokens,
    int C,
    at::Tensor x,
    at::Tensor state_pool,
    at::Tensor state_indices,
    at::Tensor cu_seqlens,
    at::Tensor token_batch_indices,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g) {
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = static_cast<int64_t>(total_tokens) * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool has_token_batch_indices = token_batch_indices.numel() != 0;
  if (has_token_batch_indices) {
    tmix_mix6_varlen_kernel<true><<<
        static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        num_sequences, C, x.data_ptr<dtype>(), state_pool.data_ptr<dtype>(),
        state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(),
        token_batch_indices.data_ptr<int>(), x_r.data_ptr<dtype>(),
        x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(), x_v.data_ptr<dtype>(),
        x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(), out_r.data_ptr<dtype>(),
        out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(),
        out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(), total_pairs);
  } else {
    tmix_mix6_varlen_kernel<false><<<
        static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        num_sequences, C, x.data_ptr<dtype>(), state_pool.data_ptr<dtype>(),
        state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(), nullptr,
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(),
        total_pairs);
  }
  const int64_t state_pairs = static_cast<int64_t>(num_sequences) * (C / 2);
  update_shift_state_varlen_kernel<<<
      static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
      C, x.data_ptr<dtype>(), state_pool.data_ptr<dtype>(),
      state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(), num_sequences,
      state_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> tmix_mix6_cfg_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    int threads) {
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 1) {
    tmix_mix6_kernel<true><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(),
        x_w.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(),
        x_a.data_ptr<dtype>(),
        x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(),
        out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(),
        out_a.data_ptr<dtype>(),
        out_g.data_ptr<dtype>(),
        total_pairs);
  } else {
    tmix_mix6_kernel<false><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C,
        x.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(),
        x_w.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(),
        x_a.data_ptr<dtype>(),
        x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(),
        out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(),
        out_a.data_ptr<dtype>(),
        out_g.data_ptr<dtype>(),
        total_pairs);
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    update_shift_state_last_kernel<<<static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), state_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_r, out_w, out_k, out_v, out_a, out_g};
}

void launch_tmix_mix6_flat_out(
    int B,
    int T,
    int C,
    const at::Tensor& x,
    at::Tensor& shift_state,
    const at::Tensor& x_r,
    const at::Tensor& x_w,
    const at::Tensor& x_k,
    const at::Tensor& x_v,
    const at::Tensor& x_a,
    const at::Tensor& x_g,
    at::Tensor& out_r,
    at::Tensor& out_w,
    at::Tensor& out_k,
    at::Tensor& out_v,
    at::Tensor& out_a,
    at::Tensor& out_g) {
  constexpr int threads = 256;
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (T == 1) {
    tmix_mix6_kernel<true><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(), total_pairs);
  } else {
    tmix_mix6_kernel<false><<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(), total_pairs);
    const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
    update_shift_state_last_kernel<<<static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(), state_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_tmix_mix6_3d(
    int B,
    int T,
    int C,
    const at::Tensor& x,
    at::Tensor& shift_state,
    const at::Tensor& x_r,
    const at::Tensor& x_w,
    const at::Tensor& x_k,
    const at::Tensor& x_v,
    const at::Tensor& x_a,
    const at::Tensor& x_g,
    at::Tensor& out_r,
    at::Tensor& out_w,
    at::Tensor& out_k,
    at::Tensor& out_v,
    at::Tensor& out_a,
    at::Tensor& out_g) {
  constexpr int threads = 256;
  const int c_pairs = C / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 mix_grid(static_cast<unsigned int>(ceil_div(c_pairs, threads)),
                      static_cast<unsigned int>(T),
                      static_cast<unsigned int>(B));
  if (T == 1) {
    tmix_mix6_3d_kernel<true><<<mix_grid, threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>());
  } else {
    tmix_mix6_3d_kernel<false><<<mix_grid, threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>());
    // Keep the recurrent write in a second launch. An in-grid t==T-1 store
    // could race a different CTA that has not read old state for t==0 yet.
    const dim3 state_grid(static_cast<unsigned int>(ceil_div(c_pairs, threads)),
                          static_cast<unsigned int>(B));
    update_shift_state_last_2d_kernel<<<state_grid, threads, 0, stream>>>(
        T, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<at::Tensor> tmix_mix6_3d_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g) {
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  launch_tmix_mix6_3d(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
                      out_r, out_w, out_k, out_v, out_a, out_g);
  return {out_r, out_w, out_k, out_v, out_a, out_g};
}

void tmix_mix6_cfg_out_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    at::Tensor out_r,
    at::Tensor out_w,
    at::Tensor out_k,
    at::Tensor out_v,
    at::Tensor out_a,
    at::Tensor out_g) {
  launch_tmix_mix6_flat_out(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
                            out_r, out_w, out_k, out_v, out_a, out_g);
}

void tmix_mix6_3d_out_cuda(
    int B,
    int T,
    int C,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    at::Tensor out_r,
    at::Tensor out_w,
    at::Tensor out_k,
    at::Tensor out_v,
    at::Tensor out_a,
    at::Tensor out_g) {
  launch_tmix_mix6_3d(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
                      out_r, out_w, out_k, out_v, out_a, out_g);
}

template <int Vec>
std::vector<at::Tensor> tmix_mix6_t1_c4096_cuda_impl(
    int B,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    int threads,
    bool half_math) {
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  const int64_t total_pairs = static_cast<int64_t>(B) * (4096 / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(threads) * Vec));
  if (half_math) {
    tmix_mix6_t1_c4096_kernel<true, Vec><<<blocks, threads, 0, stream>>>(
        x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(), x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(),
        total_pairs);
  } else {
    tmix_mix6_t1_c4096_kernel<false, Vec><<<blocks, threads, 0, stream>>>(
        x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(), x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(),
        total_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> tmix_mix6_t1_c4096_cuda(
    int B,
    at::Tensor x,
    at::Tensor shift_state,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    int threads,
    int vec,
    bool half_math) {
  if (vec == 2) {
    return tmix_mix6_t1_c4096_cuda_impl<2>(B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, threads, half_math);
  }
  if (vec == 4) {
    return tmix_mix6_t1_c4096_cuda_impl<4>(B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, threads, half_math);
  }
  if (vec == 8) {
    return tmix_mix6_t1_c4096_cuda_impl<8>(B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, threads, half_math);
  }
  return tmix_mix6_t1_c4096_cuda_impl<1>(B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, threads, half_math);
}

std::vector<at::Tensor> tmix_kk_a_gate_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor k,
    at::Tensor k_k,
    at::Tensor a0,
    at::Tensor a12,
    at::Tensor k_a,
    at::Tensor x,
    at::Tensor shift_state,
    bool update_shift) {
  (void)C;
  assert(C == H * HEAD_SIZE);
  auto new_k = at::empty_like(k);
  auto neg_kk = at::empty_like(k);
  auto kka = at::empty_like(k);
  const int64_t bth_size = static_cast<int64_t>(B) * T * H;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>(ceil_div(bth_size, static_cast<int64_t>(WARPS_PER_BLOCK)));
  if (update_shift) {
    tmix_kk_a_gate_kernel<true><<<blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
        H, k.data_ptr<dtype>(), k_k.data_ptr<dtype>(), a0.data_ptr<dtype>(), a12.data_ptr<dtype>(),
        k_a.data_ptr<dtype>(), x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        new_k.data_ptr<dtype>(), neg_kk.data_ptr<dtype>(), kka.data_ptr<dtype>(), bth_size);
  } else {
    tmix_kk_a_gate_kernel<false><<<blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
        H, k.data_ptr<dtype>(), k_k.data_ptr<dtype>(), a0.data_ptr<dtype>(), a12.data_ptr<dtype>(),
        k_a.data_ptr<dtype>(), nullptr, nullptr,
        new_k.data_ptr<dtype>(), neg_kk.data_ptr<dtype>(), kka.data_ptr<dtype>(), bth_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {new_k, neg_kk, kka};
}

std::vector<at::Tensor> tmix_kk_a_gate_2d_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor k,
    at::Tensor k_k,
    at::Tensor a0,
    at::Tensor a12,
    at::Tensor k_a) {
  (void)C;
  assert(C == H * HEAD_SIZE);
  auto new_k = at::empty_like(k);
  auto neg_kk = at::empty_like(k);
  auto kka = at::empty_like(k);
  const int64_t rows = static_cast<int64_t>(B) * T;
  // The C++ boundary requires H%4==0 and rows<=65535. Each 4-warp CTA
  // therefore owns one complete head group with no padded or aliased owner.
  const dim3 grid(static_cast<unsigned int>(ceil_div(H, WARPS_PER_BLOCK)),
                  static_cast<unsigned int>(rows));
  auto stream = at::cuda::getCurrentCUDAStream();
  tmix_kk_a_gate_2d_kernel<false><<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(
      H, k.data_ptr<dtype>(), k_k.data_ptr<dtype>(), a0.data_ptr<dtype>(), a12.data_ptr<dtype>(),
      k_a.data_ptr<dtype>(), nullptr, nullptr,
      new_k.data_ptr<dtype>(), neg_kk.data_ptr<dtype>(), kka.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {new_k, neg_kk, kka};
}

at::Tensor tmix_lnx_rkvres_xg_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor x,
    at::Tensor r,
    at::Tensor k,
    at::Tensor v,
    at::Tensor r_k,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor g) {
  (void)C;
  assert(C == H * HEAD_SIZE);
  auto out = at::empty_like(x);
  const int64_t bth_size = static_cast<int64_t>(B) * T * H;
  auto stream = at::cuda::getCurrentCUDAStream();
  tmix_lnx_rkvres_xg_kernel<<<static_cast<int>(bth_size), HEAD_SIZE, 0, stream>>>(
      C, H,
      x.data_ptr<dtype>(),
      r.data_ptr<dtype>(),
      k.data_ptr<dtype>(),
      v.data_ptr<dtype>(),
      r_k.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      g.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      bth_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor tmix_lnx_rkvres_xg_warp_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor x,
    at::Tensor r,
    at::Tensor k,
    at::Tensor v,
    at::Tensor r_k,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor g) {
  (void)C;
  assert(C == H * HEAD_SIZE);
  auto out = at::empty_like(x);
  const int64_t bth_size = static_cast<int64_t>(B) * T * H;
  auto stream = at::cuda::getCurrentCUDAStream();
  tmix_lnx_rkvres_xg_warp_kernel<<<static_cast<int>(bth_size), 32, 0, stream>>>(
      H,
      x.data_ptr<dtype>(),
      r.data_ptr<dtype>(),
      k.data_ptr<dtype>(),
      v.data_ptr<dtype>(),
      r_k.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      g.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      bth_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor tmix_lnx_rkvres_xg_warp_2d_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor x,
    at::Tensor r,
    at::Tensor k,
    at::Tensor v,
    at::Tensor r_k,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor g) {
  (void)C;
  assert(C == H * HEAD_SIZE);
  auto out = at::empty_like(x);
  const int64_t rows = static_cast<int64_t>(B) * T;
  const dim3 grid(static_cast<unsigned int>(H), static_cast<unsigned int>(rows));
  auto stream = at::cuda::getCurrentCUDAStream();
  tmix_lnx_rkvres_xg_warp_2d_kernel<<<grid, 32, 0, stream>>>(
      H,
      x.data_ptr<dtype>(),
      r.data_ptr<dtype>(),
      k.data_ptr<dtype>(),
      v.data_ptr<dtype>(),
      r_k.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      g.data_ptr<dtype>(),
      out.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <int Threads>
void launch_tmix_vres_gate_vec2(
    int B,
    int T,
    int C,
    const at::Tensor& v,
    const at::Tensor& v_first,
    const at::Tensor& v0,
    const at::Tensor& v12,
    at::Tensor& out,
    cudaStream_t stream) {
  const int64_t rows = static_cast<int64_t>(B) * T;
  const int pairs_per_row = C >> 1;
  tmix_vres_gate_vec2_kernel<Threads><<<
      dim3(static_cast<unsigned int>(ceil_div(pairs_per_row, Threads)), static_cast<unsigned int>(rows), 1),
      Threads, 0, stream>>>(
      C,
      v.data_ptr<dtype>(),
      v_first.data_ptr<dtype>(),
      v0.data_ptr<dtype>(),
      v12.data_ptr<dtype>(),
      out.data_ptr<dtype>(),
      rows);
}

void launch_tmix_vres_gate_cfg(
    int B,
    int T,
    int C,
    const at::Tensor& v,
    const at::Tensor& v_first,
    const at::Tensor& v0,
    const at::Tensor& v12,
    at::Tensor& out,
    int threads,
    bool vectorized,
    cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(B) * T * C;
  if (!vectorized) {
    tmix_vres_gate_kernel<<<static_cast<int>(ceil_div(total, threads)), threads, 0, stream>>>(
        C,
        v.data_ptr<dtype>(),
        v_first.data_ptr<dtype>(),
        v0.data_ptr<dtype>(),
        v12.data_ptr<dtype>(),
        out.data_ptr<dtype>(),
        total);
    return;
  }
  switch (threads) {
    case 64:
      launch_tmix_vres_gate_vec2<64>(B, T, C, v, v_first, v0, v12, out, stream);
      break;
    case 128:
      launch_tmix_vres_gate_vec2<128>(B, T, C, v, v_first, v0, v12, out, stream);
      break;
    case 256:
      launch_tmix_vres_gate_vec2<256>(B, T, C, v, v_first, v0, v12, out, stream);
      break;
    default:
      launch_tmix_vres_gate_vec2<512>(B, T, C, v, v_first, v0, v12, out, stream);
      break;
  }
}

at::Tensor tmix_vres_gate_cuda(
    int B,
    int T,
    int C,
    at::Tensor v,
    at::Tensor v_first,
    at::Tensor v0,
    at::Tensor v12) {
  auto out = at::empty_like(v);
  if (out.numel() == 0) {
    return out;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_tmix_vres_gate_cfg(B, T, C, v, v_first, v0, v12, out, 256, false, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor tmix_vres_gate_cfg_cuda(
    int B,
    int T,
    int C,
    at::Tensor v,
    at::Tensor v_first,
    at::Tensor v0,
    at::Tensor v12,
    int threads,
    bool vectorized) {
  auto out = at::empty_like(v);
  if (out.numel() == 0) {
    return out;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_tmix_vres_gate_cfg(B, T, C, v, v_first, v0, v12, out, threads, vectorized, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void tmix_vres_gate_cfg_out_cuda(
    int B,
    int T,
    int C,
    at::Tensor v,
    at::Tensor v_first,
    at::Tensor v0,
    at::Tensor v12,
    at::Tensor out,
    int threads,
    bool vectorized) {
  if (out.numel() == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_tmix_vres_gate_cfg(B, T, C, v, v_first, v0, v12, out, threads, vectorized, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
