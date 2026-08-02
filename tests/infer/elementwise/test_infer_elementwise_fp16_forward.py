# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flash_rwkv._extension import _load_extension


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_elementwise_fp16_operators_match_torch_reference() -> None:
    _load_extension()
    torch.manual_seed(2611)
    x = torch.randn(2, 3, 64, device="cuda", dtype=torch.float16).mul_(0.2)
    vector = torch.randn(64, device="cuda", dtype=torch.float16).mul_(0.2)
    ops = torch.ops.rwkv7_fast_ops_fp16

    torch.testing.assert_close(
        ops.relu_square(x),
        torch.relu(x.float()).square().half(),
        atol=0.002,
        rtol=0.002,
    )
    torch.testing.assert_close(
        ops.act_tanh(x),
        torch.tanh(x.float()).half(),
        atol=0.002,
        rtol=0.002,
    )
    torch.testing.assert_close(
        ops.act_sigmoid(x),
        torch.sigmoid(x.float()).half(),
        atol=0.002,
        rtol=0.002,
    )
    expected = (x.float() + vector.float()).half()
    torch.testing.assert_close(ops.add_vec(64, x, vector), expected, atol=0.002, rtol=0.002)
    torch.testing.assert_close(
        ops.add_vec_2d(64, x, vector), expected, atol=0.002, rtol=0.002
    )
