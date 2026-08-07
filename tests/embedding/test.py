# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashrwkv2.embedding import infer_embedding_ln0_forward_varlen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_embedding_ln0_packed() -> None:
    torch.manual_seed(57)
    device = torch.device("cuda")
    embedding = torch.randn(5, 8, device=device, dtype=torch.bfloat16)
    weight = torch.randn(8, device=device, dtype=torch.bfloat16)
    bias = torch.randn(8, device=device, dtype=torch.bfloat16)
    output = infer_embedding_ln0_forward_varlen(embedding, weight, bias)
    expected = F.layer_norm(embedding.float(), (8,), weight.float(), bias.float(), 1.0e-5).half()
    assert output.dtype == torch.float16
    assert torch.allclose(output.float(), expected.float(), atol=0.05, rtol=0.05)
