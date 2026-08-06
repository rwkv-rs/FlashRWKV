// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Upstream repository: https://github.com/BlinkDL/Albatross
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e
// Original path: faster3a_2607/cuda/rwkv7_v3a_ops.cu
// Mechanical migration boundary: the head caller reuses the exact
// Albatross original-layout linear, add, and layer-normalization bodies owned
// by tmix/linear and tmix/normalization. Only packed-row selection is local
// varlen adaptation; no generic GEMM or normalization body is implemented here.

#include <ATen/ATen.h>
at::Tensor tmix_linear_original_caller_dispatch_cuda(
    at::Tensor x, at::Tensor weight_orig, int64_t caller_group);
at::Tensor linear_f16_orig_lt_cfg_cuda(
    at::Tensor x, at::Tensor weight_orig,
    int64_t workspace_mb, int64_t algo_index);
at::Tensor add_last_layer_norm_indexed_packed_f16_cuda(
    at::Tensor x, at::Tensor residual, at::Tensor last_indices,
    at::Tensor weight, at::Tensor bias, double eps);

at::Tensor head_linear_forward_varlen_cuda(
    at::Tensor x, at::Tensor weight) {
  // Group 3 is the canonical Albatross head caller.  The exact
  // rows/WMMA/CuBLASLt policy is owned by tmix/linear and is shared here
  // without a second implementation or a compatibility provider.
  return tmix_linear_original_caller_dispatch_cuda(x, weight, 3);
}

at::Tensor head_linear_all_forward_varlen_cuda(
    at::Tensor x, at::Tensor weight) {
  // Mechanical copy of HEAD_ALL_LOGITS_GEMM_4096 from the canonical
  // Albatross caller.  The table selects the existing original-layout Lt
  // body; an unlisted row count follows the existing head caller dispatch.
  if (x.size(1) == 4096) {
    switch (x.size(0)) {
      case 24:
        return linear_f16_orig_lt_cfg_cuda(x, weight, 0, 0);
      case 32:
        return linear_f16_orig_lt_cfg_cuda(x, weight, 0, 0);
      case 160:
        return linear_f16_orig_lt_cfg_cuda(x, weight, 0, 7);
      case 192:
        return linear_f16_orig_lt_cfg_cuda(x, weight, 0, 5);
      default:
        break;
    }
  }
  return tmix_linear_original_caller_dispatch_cuda(x, weight, 3);
}

at::Tensor head_linear_last_forward_varlen_cuda(
    at::Tensor x, at::Tensor weight, int64_t tokens_count) {
  // Mechanical copy of HEAD_LAST_LOGITS_GEMM_4096.  This is deliberately
  // keyed by (rows, tokens_count): the last-logits GEMM has B rows even when
  // the preceding packed request contained B*T tokens.
  if (x.size(1) == 4096 && weight.size(0) == 65536 &&
      weight.size(1) == 4096) {
    const int64_t rows = x.size(0);
    if ((rows == 24 || rows == 32) &&
        (tokens_count == 1 || tokens_count == 2 || tokens_count == 4 ||
         tokens_count == 8 || (rows == 32 && tokens_count == 16))) {
      return linear_f16_orig_lt_cfg_cuda(x, weight, 0, 0);
    }
    if (rows == 160 &&
        (tokens_count == 1 || tokens_count == 2 || tokens_count == 4 ||
         tokens_count == 32)) {
      return linear_f16_orig_lt_cfg_cuda(x, weight, 0, 7);
    }
    if (rows == 192 &&
        (tokens_count == 1 || tokens_count == 2 || tokens_count == 4 ||
         tokens_count == 8 || tokens_count == 16 || tokens_count == 32)) {
      return linear_f16_orig_lt_cfg_cuda(x, weight, 0, 5);
    }
  }
  return tmix_linear_original_caller_dispatch_cuda(x, weight, 3);
}

at::Tensor head_last_norm_forward_varlen_cuda(
    at::Tensor x, at::Tensor residual, at::Tensor indices,
    at::Tensor weight, at::Tensor bias, double eps) {
  // Mechanical packed adaptation of Albatross
  // add_last_layer_norm_indexed_f16: its indexed rows are request-local
  // outputs, while the packed caller supplies absolute source-row indices.
  // The add/LN body and B-dependent dispatch remain in normalization.
  return add_last_layer_norm_indexed_packed_f16_cuda(
      x, residual, indices, weight, bias, eps);
}
