// SPDX-License-Identifier: MIT

#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

void launch_chunk_replay_fp32(
    int num_chunks,
    int num_heads,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& boundary,
    const torch::Tensor& r,
    const torch::Tensor& log_decay,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    torch::Tensor* state_dot_a,
    float scale,
    cudaStream_t stream);
