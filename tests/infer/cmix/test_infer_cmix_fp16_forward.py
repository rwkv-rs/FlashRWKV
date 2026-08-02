# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flash_rwkv import infer_cmix_mix_fp16


def test_cmix_inference_rejects_cpu_before_native_dispatch() -> None:
    x = torch.zeros(1, 1, 64, dtype=torch.float16)
    shift = torch.zeros(1, 64, dtype=torch.float16)
    mix = torch.zeros(64, dtype=torch.float16)
    with pytest.raises(ValueError, match="CUDA"):
        infer_cmix_mix_fp16(x, shift, mix)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cmix_mix_matches_reference_and_advances_state() -> None:
    torch.manual_seed(2610)
    x = torch.randn(2, 3, 128, device="cuda", dtype=torch.float16).mul_(0.2)
    shift_state = torch.randn(2, 128, device="cuda", dtype=torch.float16).mul_(0.2)
    initial_state = shift_state.clone()
    mix = torch.randn(128, device="cuda", dtype=torch.float16).mul_(0.2)

    output = infer_cmix_mix_fp16(x, shift_state, mix)

    previous = torch.cat((initial_state[:, None], x[:, :-1]), dim=1)
    reference = (x.float() + (previous.float() - x.float()) * mix.float()).half()
    torch.testing.assert_close(output, reference, atol=0.002, rtol=0.002)
    torch.testing.assert_close(shift_state, x[:, -1], atol=0, rtol=0)
