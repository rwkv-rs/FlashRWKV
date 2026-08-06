# FlashRWKV

FlashRWKV 是 RWKV7 的 module-local CUDA kernel 后端。当前迁移树采用
Albatross-first、train_temp-second、vllm-varlen-reference 的来源边界：

- inference 数学、shape-specific family 和主 dispatch 来自 Albatross
  `faster3a_2607`，revision
  `ee3308f6922e59f2166c7fac3c5a192340a2b48e`；
- train/pretrain forward/backward 来自 RWKV-LM `train_temp`，revision
  `952102498e9ed367ea0a59ee64106916d474d30f`；
- vllm-rwkv revision `6d683f9e49a2997e405c47edc147872c8609513b` 只参考
  packed metadata、state slot 和 scheduler boundary，不是 kernel body 或 fallback；
- BF16 chunk 保留 HANDOFF 指定的 FlashKDA revision
  `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` 的 RWKV7 chunk algebra。

实际可达 family、caller ownership、源码 revision 和未完成边界见
[`docs/kernels/albatross-train-temp-manifest.md`](/home/caizus/Projects/MachineLearning/rwkv/flash-rwkv/docs/kernels/albatross-train-temp-manifest.md)。

## Inference packed contract

sequence-dependent operator 统一使用 packed token rows：

```text
tokens:        [total_tokens, ...]
cu_seqlens:    [batch_size + 1], CUDA contiguous int32
state_indices: [batch_size], CUDA contiguous int32
```

`cu_seqlens[0]` 必须为 `0`，最后一个 offset 必须等于
`total_tokens`，每个 sequence 非空；同一个 launch 中 `state_indices` 不得重复，
并且必须落在对应 state pool 范围内。所有输入必须 CUDA、contiguous、同 device；
Python wrapper 不做 padding、CPU copy、dtype conversion 或隐式 contiguous copy。

WKV recurrent 的 state pool 是 `[slots,H,D,D]`，canonical 内存解释为 `[K,V]`，
kernel 原地更新选中的 slots。FP32-state path 支持 `D=64/128/256`，token I/O
支持 FP16/BF16；FP16-state path 使用 Albatross family 的 FP16 state contract。
raw `decay_logits` 是唯一 public decay boundary，kernel 内完成 retention transform；
不提供 `log_decay`、`from_log_decay` 或独立 `elapsed_t` compatibility operator。

metadata preparation 可以生成 reusable native ticket。ticket 绑定 metadata tensor
identity、data pointer、shape/stride、version、device、stream、token 数量、state
pool size 和 `max_seqlen` snapshot；ticket 与 launch 不匹配时 fail closed。

典型 WKV 调用：

```python
from flash_rwkv import (
    infer_recurrent_fp32io16_forward_varlen,
    prepare_recurrent_metadata,
)

ticket = prepare_recurrent_metadata(
    cu_seqlens,
    state_indices,
    total_tokens=r.shape[0],
    state_pool_size=state_pool.shape[0],
)
output = infer_recurrent_fp32io16_forward_varlen(
    r,
    decay_logits,
    k,
    v,
    a,
    b,
    state_pool=state_pool,
    cu_seqlens=cu_seqlens,
    state_indices=state_indices,
    validated_metadata=ticket,
)
```

WKV FP32 inference 已收录 large、small-warp、short-block 三路 family；FP16
inference 已收录 clone、exact、seq-v2、one-cp、one-direct family，并保留
Albatross 的 `Tis1`、`AddW0`、`Grid2D` template dispatch。Albatross
81-case operator correctness matrix 已在 SM120 上通过。canonical Albatross FP16
source 只提供 `D=64`；`D=128/256` 不选择 generic 或临时替代，当前 binding 对其
fail closed。FP16 `elapsed_state_pool` 已作为 request-state bundle 的内部组成部分
接入；`infer_recurrent_fp16_advance_i32_varlen` 按 packed sequence length 机械迁移
上游 `advance_i32` 的 slot advancement，并复用 recurrent metadata ticket；该
operator 的 focused slot/state correctness 已通过。外部 caller 负责按自己的
sequence lifecycle 调用该 module-local helper；FlashRWKV 不定义 model 或 layer loop。

## Module ownership

当前 active module path 包括：

- `tmix/wkv7`：recurrent、BF16 chunk、pretrain、StateTune；
- `tmix/mix6`、`tmix/kk_a_gate`、`tmix/lnx_rkvres_xg`、`tmix/vres_gate`；
- `cmix/mix`、`cmix/sparse`；
- `tmix/linear`、`tmix/normalization`、`embedding`、`head/linear`；
- `tmix/a_gate`、`tmix/kk_pre`；
- `loss/l2wrap_ce`、`head/l2wrap_ce`；
- `rl_infctx/wkv7`。

每个 family 有自己的 Python、test、benchmark 和同 stem native binding/implementation
pair。`csrc/sm90` 保存 train_temp/训练相关实现，`csrc/sm120` 保存当前目标 GPU
上的 inference/packed 实现。当前没有新的 `elementwise` module、`csrc/common`、
global registry/provider 或 vllm fallback。

Albatross lightweight caller identity 已拆出 mix6、KK-A gate、LN/RKV/residual/XG、
v-res gate、CMix mix/sparse 等 module-local path。`update_shift_state_last_kernel`
是相应 stateful family 的内部 closure，不注册成独立 public operator。

`rwkv7_v3a_ops` 的 `.cu` body 已按真实 caller 机械迁移到 `linear`、normalization、
embedding、head 等 module；`tmix/linear` 已接入 `linear_t_f16`、tanh/sigmoid
`linear_t_act_f16` 和 `linear_t_vres_f16` caller binding。FlashRWKV 是 operator
library，不定义完整 model graph；外部 model caller 负责组合这些 module-local
operator。当前仍需继续补齐各 operator 的 caller-specific split-K、row-tile、
WMMA/CuBLASLt dispatch 证据，以及 fused TMix/CMix LN 的 packed request-boundary
operator acceptance。

## Training and auxiliary

`train_temp` canonical body 已接入：

- recurrent forward/backward：final state、final-state gradient、initial-state
  gradient、state-dot-a、checkpoint/chunk metadata 和 tail chunk；
- TMix a-gate、v-res gate、mix6、KK-pre、LN/RKV/residual/XG；
- CMix forward/backward；
- L2Wrap CE loss 和 head L2Wrap CE。

训练 operator 保留 training 自己的 tensor layout、workspace、autograd、recompute、
loss scaling 和 gradient 语义，不套用 inference state pool API。RL/Infctx 和
StateTune 保留独立 public entry；RL 的 materialized/recompute/replay 与 StateTune
的 recurrent forward/backward body 已独立机械迁移，完整 strategy/workspace
acceptance 仍需继续验证。

## Build

使用仓库自己的 `./.venv`。当前迁移 slice 的 native build 要求 SM120 或更高：

```bash
uv sync
TORCH_CUDA_ARCH_LIST=12.0 \
  ./.venv/bin/python -m pip install -v --no-build-isolation -e .
```

不要把 `build/`、`artifacts/`、`.egg-info` 或缓存生成物加入 worktree。构建源列表
和 architecture gate 位于 [`setup.py`](/home/caizus/Projects/MachineLearning/rwkv/flash-rwkv/setup.py)。

## Verification

当前 focused native regression：

```bash
CUDA_LAUNCH_BLOCKING=1 ./.venv/bin/python -m pytest -q \
  tests/tmix/wkv7/test.py \
  tests/tmix/wkv7/chunk/test.py \
  tests/tmix/wkv7/pretrain/test.py \
  tests/tmix/wkv7/statetune/test.py \
  tests/tmix/a_gate/test.py tests/tmix/vres_gate/test.py \
  tests/tmix/mix6/test.py tests/tmix/kk_pre/test.py \
  tests/tmix/lnx_rkvres_xg/test.py tests/tmix/kk_a_gate/test.py \
  tests/tmix/linear/test.py tests/tmix/normalization/test.py \
  tests/cmix/mix/test.py tests/cmix/sparse/test.py \
  tests/embedding/test.py tests/head/linear/test.py \
  tests/head/l2wrap_ce/test.py tests/loss/l2wrap_ce/test.py \
  tests/rl_infctx/wkv7/test.py
```

WKV operator correctness matrix：

```bash
./.venv/bin/python -m benchmarks.tmix.wkv7.bench \
  --shapes h32d64 h40d64 h64d64 --dtype bfloat16 --correctness-only \
  --stress --decay-bias \
  --output /tmp/flash-rwkv-wkv7-operator-shapes-correctness.json
```

benchmark JSON 应记录 source revision、compiled extension hash、GPU/SM、Torch/CUDA/
Python、selected kernel family、raw latency samples、p10/p50/p90、throughput、
correctness tolerance 和 failure reason。benchmark 只测单个 operator，不定义或组合 model，
也不把任何 model-level layer count 乘进 latency。

完整 Albatross fixed-vs-uniform-varlen-vs-ragged-vs-vllm operator diagnostic、
各 stateful operator 的 ragged correctness、目标 GPU 全量 operator benchmark、
compute-sanitizer/racecheck 和完整 `rwkv7_v3a_ops` tuned dispatch 证据不是当前
smoke 结果可以替代的验收项；FlashRWKV 不负责完整 model 定义或 model-level
benchmark。

## Worktree boundary

当前树是有意保留的 dirty migration tree。`_old` 目录、已有删除、用户修改和本地
验证生成的 native `.so` 不因文档或测试更新被自动清理。本项目当前不会自动
commit、push、reset 或 checkout 无关路径。
