# SPDX-License-Identifier: MIT

import pytest
import torch

from flash_rwkv import (
    RWKV7_RECURRENT_HEAD_SIZES,
    pretrain_recurrent_fp32io16_forward,
    rwkv7,
)


def _inputs(
    head_size: int, *, value_size: int | None = None
) -> tuple[torch.Tensor, ...]:
    shape = (1, 2, 1, head_size)
    value_shape = (*shape[:-1], value_size or head_size)
    inputs = [torch.zeros(shape, dtype=torch.float16) for _ in range(6)]
    inputs[3] = torch.zeros(value_shape, dtype=torch.float16)
    return tuple(inputs)


@pytest.mark.parametrize("head_size", RWKV7_RECURRENT_HEAD_SIZES)
def test_public_recurrent_contract_accepts_supported_head_sizes_before_cuda(
    head_size: int,
) -> None:
    inputs = _inputs(head_size)

    with pytest.raises(ValueError, match="algorithm='recurrent' requires CUDA"):
        rwkv7(*inputs, algorithm="recurrent")
    with pytest.raises(ValueError, match="requires CUDA inputs"):
        pretrain_recurrent_fp32io16_forward(*inputs)


@pytest.mark.parametrize(
    ("head_size", "value_size"),
    [(32, 32), (64, 128), (128, 256)],
)
def test_public_recurrent_contract_rejects_unsupported_or_asymmetric_heads(
    head_size: int,
    value_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"requires equal K and V in \{64, 128, 256\}",
    ):
        rwkv7(*_inputs(head_size, value_size=value_size), algorithm="recurrent")
