// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV project

#pragma once

#include <cuda_runtime.h>

namespace flash_rwkv::wkv7 {

enum class RecurrentDecayInput {
  kLogDecay,
  kDecayLogits,
};

constexpr float kNegativeExpHalfLog2E = -0.8750387749145276f;
constexpr float kNegativeLog2E = -1.4426950408889634f;
constexpr float kExpNegativeHalf = 0.6065306597126334f;

template <RecurrentDecayInput Input>
__device__ __forceinline__ float recurrent_retention(float value) {
  if constexpr (Input == RecurrentDecayInput::kDecayLogits) {
    return exp2f(
        kNegativeExpHalfLog2E /
        (1.0f + exp2f(kNegativeLog2E * value)));
  }
  return expf(value);
}

__device__ __forceinline__ float log_decay_derivative_from_logits(
    float decay_logits) {
  const float sigmoid =
      1.0f / (1.0f + exp2f(kNegativeLog2E * decay_logits));
  return -kExpNegativeHalf * sigmoid * (1.0f - sigmoid);
}

}  // namespace flash_rwkv::wkv7
