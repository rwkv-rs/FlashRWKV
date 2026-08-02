// SPDX-License-Identifier: MIT

#pragma once

#include <cuda_bf16.h>

// CUDA exposes native bfloat162 arithmetic only on SM80 and newer. Pascal
// still supports scalar BF16 conversion, so emulate each pair through FP32.
__device__ __forceinline__ float2 flash_bfloat1622float2(__nv_bfloat162 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __bfloat1622float2(value);
#else
  return make_float2(__bfloat162float(value.x), __bfloat162float(value.y));
#endif
}

__device__ __forceinline__ __nv_bfloat162 flash_floats2bfloat162_rn(
    float x,
    float y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __floats2bfloat162_rn(x, y);
#else
  __nv_bfloat162 result;
  result.x = __float2bfloat16_rn(x);
  result.y = __float2bfloat16_rn(y);
  return result;
#endif
}

__device__ __forceinline__ __nv_bfloat162 flash_bfloat162_add(
    __nv_bfloat162 lhs,
    __nv_bfloat162 rhs) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hadd2(lhs, rhs);
#else
  const float2 a = flash_bfloat1622float2(lhs);
  const float2 b = flash_bfloat1622float2(rhs);
  return flash_floats2bfloat162_rn(a.x + b.x, a.y + b.y);
#endif
}

__device__ __forceinline__ __nv_bfloat162 flash_bfloat162_sub(
    __nv_bfloat162 lhs,
    __nv_bfloat162 rhs) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hsub2(lhs, rhs);
#else
  const float2 a = flash_bfloat1622float2(lhs);
  const float2 b = flash_bfloat1622float2(rhs);
  return flash_floats2bfloat162_rn(a.x - b.x, a.y - b.y);
#endif
}

__device__ __forceinline__ __nv_bfloat162 flash_bfloat162_mul(
    __nv_bfloat162 lhs,
    __nv_bfloat162 rhs) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hmul2(lhs, rhs);
#else
  const float2 a = flash_bfloat1622float2(lhs);
  const float2 b = flash_bfloat1622float2(rhs);
  return flash_floats2bfloat162_rn(a.x * b.x, a.y * b.y);
#endif
}

__device__ __forceinline__ float flash_bfloat162_low2float(
    __nv_bfloat162 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __low2float(value);
#else
  return __bfloat162float(value.x);
#endif
}

__device__ __forceinline__ float flash_bfloat162_high2float(
    __nv_bfloat162 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __high2float(value);
#else
  return __bfloat162float(value.y);
#endif
}

#define __bfloat1622float2 flash_bfloat1622float2
#define __floats2bfloat162_rn flash_floats2bfloat162_rn
#define __hadd2 flash_bfloat162_add
#define __hsub2 flash_bfloat162_sub
#define __hmul2 flash_bfloat162_mul
#define __low2float flash_bfloat162_low2float
#define __high2float flash_bfloat162_high2float
