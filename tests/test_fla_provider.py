# SPDX-License-Identifier: MIT

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flash_rwkv.providers import fla


@pytest.fixture(autouse=True)
def clear_fla_operator_cache() -> None:
    fla._raw_decay_operator.cache_clear()
    yield
    fla._raw_decay_operator.cache_clear()


def _tokens() -> tuple[torch.Tensor, ...]:
    return tuple(torch.randn(1, 2, 1, 4) for _ in range(6))


def test_chunk_adapter_forwards_raw_decay_and_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def chunk_rwkv7(
        r: torch.Tensor,
        decay_logits: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        safe_gate: bool = False,
        chunk_size: int | None = None,
        disable_recompute: bool = False,
        cp_context: object | None = None,
        decay_bias: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        calls.append(
            (
                (r, decay_logits, k, v, a, b),
                {
                    "scale": scale,
                    "initial_state": initial_state,
                    "output_final_state": output_final_state,
                    "cu_seqlens": cu_seqlens,
                    "cu_seqlens_cpu": cu_seqlens_cpu,
                    "safe_gate": safe_gate,
                    "chunk_size": chunk_size,
                    "disable_recompute": disable_recompute,
                    "cp_context": cp_context,
                    "decay_bias": decay_bias,
                    **kwargs,
                },
            )
        )
        return v, initial_state

    monkeypatch.setattr(
        fla.importlib,
        "import_module",
        lambda _name: SimpleNamespace(chunk_rwkv7=chunk_rwkv7),
    )
    inputs = _tokens()
    initial_state = torch.randn(1, 1, 4, 4, dtype=torch.float32)
    decay_bias = torch.randn(1, 4)

    output, final_state = fla.pretrain_chunk_fp32io16_forward(
        *inputs,
        scale=0.125,
        initial_state=initial_state,
        output_final_state=True,
        safe_gate=True,
        decay_bias=decay_bias,
    )

    assert output is inputs[3]
    assert final_state is initial_state
    assert calls[0][0][1] is inputs[1]
    keywords = calls[0][1]
    assert keywords["scale"] == 0.125
    assert keywords["initial_state"] is initial_state
    assert keywords["output_final_state"] is True
    assert keywords["cu_seqlens"] is None
    assert keywords["cu_seqlens_cpu"] is None
    assert keywords["safe_gate"] is True
    assert keywords["chunk_size"] == 16
    assert keywords["disable_recompute"] is False
    assert keywords["cp_context"] is None
    assert keywords["decay_bias"] is decay_bias


def test_recurrent_adapter_uses_standard_raw_fla_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call: dict[str, object] = {}

    def recurrent_rwkv7(
        r: torch.Tensor,
        decay_logits: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
        mode: str = "fp32io16",
        safe_gate: bool = False,
        chunk_size: int | None = None,
        disable_recompute: bool = False,
        cp_context: object | None = None,
        decay_bias: torch.Tensor | None = None,
        elapsed_t: torch.Tensor | None = None,
        validated_metadata: object | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        call.update(locals())
        return v, initial_state

    monkeypatch.setattr(
        fla.importlib,
        "import_module",
        lambda _name: SimpleNamespace(recurrent_rwkv7=recurrent_rwkv7),
    )
    inputs = _tokens()
    initial_state = torch.randn(3, 1, 4, 4, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 1, 2, 2], dtype=torch.int32)
    state_indices = torch.tensor([2, 0, 1], dtype=torch.int32)
    decay_bias = torch.randn(4)
    metadata = object()

    output, final_state = fla.infer_recurrent_fp32io16_forward_varlen(
        *inputs,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
        validated_metadata=metadata,
    )

    assert output is inputs[3]
    assert final_state is initial_state
    assert call["decay_logits"] is inputs[1]
    assert call["mode"] == "fp32io16"
    assert call["cu_seqlens"] is cu_seqlens
    assert call["state_indices"] is state_indices
    assert call["decay_bias"] is decay_bias
    assert call["elapsed_t"] is None
    assert call["validated_metadata"] is metadata


def test_incompatible_legacy_fla_log_decay_abi_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def chunk_rwkv7(
        r: torch.Tensor,
        log_decay: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        decay_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        return v, None

    monkeypatch.setattr(
        fla.importlib,
        "import_module",
        lambda _name: SimpleNamespace(chunk_rwkv7=chunk_rwkv7),
    )

    with pytest.raises(RuntimeError, match="second parameter is 'log_decay'"):
        fla.pretrain_chunk_fp32io16_forward(*_tokens())
