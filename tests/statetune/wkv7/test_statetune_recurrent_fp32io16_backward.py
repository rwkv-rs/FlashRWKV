# SPDX-License-Identifier: MIT
"""Fixed-length gradient contract against Torch FP32 and FLA RWKV-7."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

from flash_rwkv import (
    pretrain_recurrent_fp32io16_forward,
    rwkv7_reference,
)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        importlib.util.find_spec("fla") is None
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] < 8,
        reason="FLA chunk kernels require SM80 or newer",
    ),
]

HEAD_SIZE = 64
TOLERANCE = json.loads(
    (
        Path(__file__).parents[2] / "fixtures/tolerances-v1.json"
    ).read_text(encoding="utf-8")
)["fp32io16_pretrain_recurrent"]


def _inputs(
    *,
    batch_size: int,
    sequence_length: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    shape = (batch_size, sequence_length, 2, HEAD_SIZE)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def normal(scale: float) -> torch.Tensor:
        return scale * torch.randn(
            shape,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )

    direction = functional.normalize(normal(1.0), dim=-1)
    strength = 0.1 * torch.rand(
        shape,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    log_decay = -0.05 - 0.15 * torch.rand(
        shape,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    inputs = (
        normal(0.05).to(dtype),
        log_decay.to(dtype),
        normal(0.05).to(dtype),
        normal(0.05).to(dtype),
        (-direction).to(dtype),
        (direction * strength).to(dtype),
    )
    initial_state = 0.02 * torch.randn(
        batch_size,
        2,
        HEAD_SIZE,
        HEAD_SIZE,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    return inputs, initial_state


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    rmse = difference.square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return float((rmse / baseline).item())


def _run_backward(
    implementation: str,
    base_inputs: tuple[torch.Tensor, ...],
    base_initial_state: torch.Tensor,
    *,
    requires_grad: tuple[bool, ...],
    grad_output: torch.Tensor,
    grad_final_state: torch.Tensor,
    include_final_state_loss: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    inputs = tuple(
        tensor.detach().clone().requires_grad_(required)
        for tensor, required in zip(
            base_inputs,
            requires_grad[:6],
            strict=True,
        )
    )
    initial_state = (
        base_initial_state.detach()
        .clone()
        .requires_grad_(requires_grad[6])
    )
    if implementation == "torch":
        output, final_state = rwkv7_reference(
            *inputs,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
        )
    elif implementation == "flash":
        output, final_state = pretrain_recurrent_fp32io16_forward(
            *inputs,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
        )
    elif implementation == "fla":
        from fla.ops.rwkv7 import chunk_rwkv7

        output, final_state = chunk_rwkv7(
            *inputs,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=16,
        )
    else:
        raise AssertionError(f"unknown implementation: {implementation}")

    assert final_state is not None
    loss = (output.float() * grad_output).sum()
    if include_final_state_loss:
        loss = loss + (final_state * grad_final_state).sum()
    loss.backward()
    gradients = tuple(
        tensor.grad for tensor in (*inputs, initial_state)
    )
    return output, final_state, gradients


@pytest.mark.parametrize(
    (
        "dtype",
        "sequence_length",
        "include_final_state_loss",
        "seed",
    ),
    [
        (torch.bfloat16, 17, False, 1201),
        (torch.float16, 33, True, 1202),
        (torch.float16, 257, True, 1203),
    ],
)
def test_pretrain_recurrent_autograd_matches_torch_and_fla(
    dtype: torch.dtype,
    sequence_length: int,
    include_final_state_loss: bool,
    seed: int,
) -> None:
    inputs, initial_state = _inputs(
        batch_size=1,
        sequence_length=sequence_length,
        dtype=dtype,
        seed=seed,
    )
    generator = torch.Generator(device="cuda").manual_seed(seed + 17)
    grad_output = (
        0.05
        * torch.randn(
            inputs[3].shape,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
    ).to(dtype).float()
    grad_final_state = 0.02 * torch.randn(
        initial_state.shape,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    results = {
        implementation: _run_backward(
            implementation,
            inputs,
            initial_state,
            requires_grad=(True,) * 7,
            grad_output=grad_output,
            grad_final_state=grad_final_state,
            include_final_state_loss=include_final_state_loss,
            scale=0.125,
        )
        for implementation in ("torch", "flash", "fla")
    }
    torch.cuda.synchronize()

    expected_output, expected_state, expected_gradients = results["torch"]
    for implementation in ("flash", "fla"):
        output, final_state, gradients = results[implementation]
        assert (
            _relative_rmse(output, expected_output)
            <= TOLERANCE["output_relative_rmse"]
        )
        assert (
            _relative_rmse(final_state, expected_state)
            <= TOLERANCE["state_relative_rmse"]
        )
        for gradient, expected_gradient in zip(
            gradients,
            expected_gradients,
            strict=True,
        ):
            assert gradient is not None
            assert expected_gradient is not None
            assert (
                _relative_rmse(gradient, expected_gradient)
                <= TOLERANCE["gradient_relative_rmse"]
            )


def test_pretrain_recurrent_only_materializes_requested_gradients() -> None:
    inputs, initial_state = _inputs(
        batch_size=2,
        sequence_length=17,
        dtype=torch.bfloat16,
        seed=1301,
    )
    generator = torch.Generator(device="cuda").manual_seed(1302)
    grad_output = (
        0.05
        * torch.randn(
            inputs[3].shape,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
    ).to(torch.bfloat16).float()
    grad_final_state = torch.zeros_like(initial_state)
    requires_grad = (True, False, False, True, False, True, False)
    results = {
        implementation: _run_backward(
            implementation,
            inputs,
            initial_state,
            requires_grad=requires_grad,
            grad_output=grad_output,
            grad_final_state=grad_final_state,
            include_final_state_loss=False,
            scale=1.0,
        )
        for implementation in ("torch", "flash", "fla")
    }
    torch.cuda.synchronize()

    expected_gradients = results["torch"][2]
    for implementation in ("flash", "fla"):
        gradients = results[implementation][2]
        for required, gradient, expected_gradient in zip(
            requires_grad,
            gradients,
            expected_gradients,
            strict=True,
        ):
            if not required:
                assert gradient is None
                assert expected_gradient is None
                continue
            assert gradient is not None
            assert expected_gradient is not None
            assert (
                _relative_rmse(gradient, expected_gradient)
                <= TOLERANCE["gradient_relative_rmse"]
            )
