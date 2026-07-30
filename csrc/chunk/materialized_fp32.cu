// SPDX-License-Identifier: MIT
// FlashRWKV materialized affine chunk forward.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadSize = 64;

template <typename io_t>
__device__ __forceinline__ float to_float(io_t value) {
  return static_cast<float>(value);
}

template <typename io_t>
__device__ __forceinline__ io_t from_float(float value) {
  return static_cast<io_t>(value);
}

struct BuildShared {
  float transform[kHeadSize][kHeadSize];
  float bias[kHeadSize][kHeadSize];
  float decay[kHeadSize];
  float a[kHeadSize];
  float b[kHeadSize];
  float k[kHeadSize];
};

template <typename io_t>
__global__ __launch_bounds__(kHeadSize, 2) void build_transforms_kernel(
    int num_heads,
    const int* __restrict__ chunk_token_starts,
    const int* __restrict__ chunk_token_ends,
    const io_t* __restrict__ log_decay_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    float* __restrict__ transform_ptr,
    float* __restrict__ bias_ptr) {
  const int linear_block = static_cast<int>(blockIdx.x);
  const int chunk_index = linear_block / num_heads;
  const int head_index = linear_block % num_heads;
  const int column = static_cast<int>(threadIdx.x);
  __shared__ BuildShared shared;

#pragma unroll
  for (int row = 0; row < kHeadSize; ++row) {
    shared.transform[row][column] = row == column ? 1.0f : 0.0f;
    shared.bias[row][column] = 0.0f;
  }
  __syncthreads();

  const int token_start = chunk_token_starts[chunk_index];
  const int token_end = chunk_token_ends[chunk_index];
  for (int token_index = token_start;
       token_index < token_end;
       ++token_index) {
    const int64_t input_index =
        (static_cast<int64_t>(token_index) * num_heads + head_index) *
            kHeadSize +
        column;
    shared.decay[column] = expf(to_float(log_decay_ptr[input_index]));
    shared.a[column] = to_float(a_ptr[input_index]);
    shared.b[column] = to_float(b_ptr[input_index]);
    shared.k[column] = to_float(k_ptr[input_index]);
    const float value = to_float(v_ptr[input_index]);
    __syncthreads();

    float transform_dot_a = 0.0f;
    float bias_dot_a = 0.0f;
#pragma unroll
    for (int row = 0; row < kHeadSize; ++row) {
      transform_dot_a = fmaf(
          shared.a[row],
          shared.transform[row][column],
          transform_dot_a);
      bias_dot_a = fmaf(
          shared.a[row],
          shared.bias[row][column],
          bias_dot_a);
    }

#pragma unroll
    for (int row = 0; row < kHeadSize; ++row) {
      shared.transform[row][column] = fmaf(
          shared.b[row],
          transform_dot_a,
          shared.decay[row] * shared.transform[row][column]);
      shared.bias[row][column] = fmaf(
          shared.k[row],
          value,
          fmaf(
              shared.b[row],
              bias_dot_a,
              shared.decay[row] * shared.bias[row][column]));
    }
    __syncthreads();
  }

  const int64_t workspace_base =
      (static_cast<int64_t>(chunk_index) * num_heads + head_index) *
      kHeadSize * kHeadSize;
#pragma unroll
  for (int row = 0; row < kHeadSize; ++row) {
    const int64_t workspace_index =
        workspace_base + row * kHeadSize + column;
    transform_ptr[workspace_index] = shared.transform[row][column];
    bias_ptr[workspace_index] = shared.bias[row][column];
  }
}

struct ScanShared {
  float transform[kHeadSize][kHeadSize];
  float next_state[kHeadSize][kHeadSize];
};

__global__ __launch_bounds__(kHeadSize, 2) void scan_boundaries_kernel(
    int num_heads,
    const int* __restrict__ sequence_chunk_offsets,
    const int* __restrict__ state_indices,
    float* __restrict__ state_ptr,
    const float* __restrict__ transform_ptr,
    const float* __restrict__ bias_ptr,
    float* __restrict__ boundary_ptr) {
  const int linear_block = static_cast<int>(blockIdx.x);
  const int sequence_index = linear_block / num_heads;
  const int head_index = linear_block % num_heads;
  const int value_index = static_cast<int>(threadIdx.x);
  const int state_slot = state_indices[sequence_index];
  __shared__ ScanShared shared;

  const int64_t state_base =
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  float state[kHeadSize];
#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state[key_index] =
        state_ptr[state_base + key_index * kHeadSize + value_index];
  }

  const int chunk_start = sequence_chunk_offsets[sequence_index];
  const int chunk_end = sequence_chunk_offsets[sequence_index + 1];
  for (int chunk_index = chunk_start;
       chunk_index < chunk_end;
       ++chunk_index) {
    const int64_t workspace_base =
        (static_cast<int64_t>(chunk_index) * num_heads + head_index) *
        kHeadSize * kHeadSize;
#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      const int64_t workspace_index =
          workspace_base + key_index * kHeadSize + value_index;
      boundary_ptr[workspace_index] = state[key_index];
      shared.transform[key_index][value_index] =
          transform_ptr[workspace_index];
    }
    __syncthreads();

#pragma unroll
    for (int output_key = 0;
         output_key < kHeadSize;
         ++output_key) {
      float updated =
          bias_ptr[
              workspace_base + output_key * kHeadSize + value_index];
#pragma unroll
      for (int input_key = 0;
           input_key < kHeadSize;
           ++input_key) {
        updated = fmaf(
            shared.transform[output_key][input_key],
            state[input_key],
            updated);
      }
      shared.next_state[output_key][value_index] = updated;
    }
    __syncthreads();
#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      state[key_index] = shared.next_state[key_index][value_index];
    }
    __syncthreads();
  }

#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state_ptr[state_base + key_index * kHeadSize + value_index] =
        state[key_index];
  }
}

struct OutputShared {
  float r[kHeadSize];
  float decay[kHeadSize];
  float k[kHeadSize];
  float a[kHeadSize];
  float b[kHeadSize];
};

template <typename io_t>
__global__ __launch_bounds__(kHeadSize, 2) void emit_outputs_kernel(
    int num_heads,
    const int* __restrict__ chunk_token_starts,
    const int* __restrict__ chunk_token_ends,
    const float* __restrict__ boundary_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ log_decay_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ output_ptr,
    float scale) {
  const int linear_block = static_cast<int>(blockIdx.x);
  const int chunk_index = linear_block / num_heads;
  const int head_index = linear_block % num_heads;
  const int value_index = static_cast<int>(threadIdx.x);
  __shared__ OutputShared shared;

  const int64_t boundary_base =
      (static_cast<int64_t>(chunk_index) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  float state[kHeadSize];
#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state[key_index] =
        boundary_ptr[
            boundary_base + key_index * kHeadSize + value_index];
  }

  const int token_start = chunk_token_starts[chunk_index];
  const int token_end = chunk_token_ends[chunk_index];
  for (int token_index = token_start;
       token_index < token_end;
       ++token_index) {
    const int64_t input_index =
        (static_cast<int64_t>(token_index) * num_heads + head_index) *
            kHeadSize +
        value_index;
    shared.r[value_index] = to_float(r_ptr[input_index]);
    shared.decay[value_index] =
        expf(to_float(log_decay_ptr[input_index]));
    shared.k[value_index] = to_float(k_ptr[input_index]);
    shared.a[value_index] = to_float(a_ptr[input_index]);
    shared.b[value_index] = to_float(b_ptr[input_index]);
    const float value = to_float(v_ptr[input_index]);
    __syncthreads();

    float state_dot_a = 0.0f;
#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      state_dot_a =
          fmaf(shared.a[key_index], state[key_index], state_dot_a);
    }

    float output = 0.0f;
#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      const float updated = fmaf(
          shared.k[key_index],
          value,
          fmaf(
              shared.b[key_index],
              state_dot_a,
              shared.decay[key_index] * state[key_index]));
      state[key_index] = updated;
      output = fmaf(shared.r[key_index], updated, output);
    }
    output_ptr[input_index] = from_float<io_t>(scale * output);
    __syncthreads();
  }
}

template <typename io_t>
void launch_materialized_chunk(
    int num_sequences,
    int num_chunks,
    int num_heads,
    const torch::Tensor& sequence_chunk_offsets,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& state_indices,
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& log_decay,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    torch::Tensor& transform,
    torch::Tensor& bias,
    torch::Tensor& boundary,
    float scale,
    cudaStream_t stream) {
  const int chunk_blocks = num_chunks * num_heads;
  const int sequence_blocks = num_sequences * num_heads;
  build_transforms_kernel<io_t>
      <<<chunk_blocks, kHeadSize, 0, stream>>>(
          num_heads,
          chunk_token_starts.data_ptr<int>(),
          chunk_token_ends.data_ptr<int>(),
          log_decay.data_ptr<io_t>(),
          k.data_ptr<io_t>(),
          v.data_ptr<io_t>(),
          a.data_ptr<io_t>(),
          b.data_ptr<io_t>(),
          transform.data_ptr<float>(),
          bias.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  scan_boundaries_kernel<<<sequence_blocks, kHeadSize, 0, stream>>>(
      num_heads,
      sequence_chunk_offsets.data_ptr<int>(),
      state_indices.data_ptr<int>(),
      state.data_ptr<float>(),
      transform.data_ptr<float>(),
      bias.data_ptr<float>(),
      boundary.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  emit_outputs_kernel<io_t>
      <<<chunk_blocks, kHeadSize, 0, stream>>>(
          num_heads,
          chunk_token_starts.data_ptr<int>(),
          chunk_token_ends.data_ptr<int>(),
          boundary.data_ptr<float>(),
          r.data_ptr<io_t>(),
          log_decay.data_ptr<io_t>(),
          k.data_ptr<io_t>(),
          v.data_ptr<io_t>(),
          a.data_ptr<io_t>(),
          b.data_ptr<io_t>(),
          output.data_ptr<io_t>(),
          scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

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
    double scale) {
  const c10::cuda::CUDAGuard device_guard(state.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int num_sequences = static_cast<int>(state_indices.numel());
  const int num_chunks = static_cast<int>(chunk_token_starts.numel());
  const int num_heads = static_cast<int>(state.size(1));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      r.scalar_type(),
      "flash_rwkv_materialized_chunk_fp32",
      [&] {
        launch_materialized_chunk<scalar_t>(
            num_sequences,
            num_chunks,
            num_heads,
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
            static_cast<float>(scale),
            stream);
      });
}
