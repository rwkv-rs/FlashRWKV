# FlashRWKV Kernel 迁移交接

更新时间：2026-08-05

本文替代本文件之前的“只完成 FP32 recurrent slice”状态记录。当前工作树仍然是用户有意保留的 dirty migration tree；本文记录的是已经落地并验证过的实现范围，以及不能被当前证据覆盖的剩余工作。

## 1. 已锁定的迁移边界

本轮采用 Albatross-first、train_temp-second、vllm-varlen-reference 的边界：

- RWKV7 inference 的数学、shape-specific family 和主 dispatch 来源是 Albatross `faster3a_2607`，revision `ee3308f6922e59f2166c7fac3c5a192340a2b48e`。
- train/pretrain 的 forward/backward 语义来源是 RWKV-LM `train_temp`，revision `952102498e9ed367ea0a59ee64106916d474d30f`。
- vllm-rwkv revision `6d683f9e49a2997e405c47edc147872c8609513b` 只用于 packed metadata、state slot、query boundary 和 scheduler contract 参考；没有作为 FlashRWKV 的 kernel body 或 fallback provider。
- BF16 chunk 继续使用 HANDOFF 指定的 FlashKDA revision `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` 的 RWKV7 chunk algebra；K1/K2 是内部阶段，不改名为 KDA attention。
- 不创建 `elementwise` 模块。逐元素 helper 只归属实际调用它的 `cmix`、`tmix`、`head` 或 normalization caller；不创建 `csrc/common/`、`flash_rwkv/providers/`、`flash_rwkv/registry.py` 或旧 global compatibility namespace。
- 当前源码和 `_old` 目录中的已有删除、备份和用户修改必须保留；本轮没有 commit、push、reset、checkout 或无关清理。

每个新增 `.cu`/`.cpp` 的头部应继续记录 upstream、exact revision、原始路径、SPDX/license 和本地适配边界。当前迁移文件已经按上述四个来源写入 provenance；若后续替换 compact body，必须更新对应头部而不是只改文件名。

## 2. 当前已经落地的 source family

### 2.1 Inference WKV7

`flash_rwkv/tmix/wkv7/` 与 `csrc/sm120/tmix/wkv7/` 已包含：

- FP32-state recurrent packed-varlen：
  - `infer_recurrent_fp32io16_forward_varlen.cpp/.cu`
  - Albatross 的 `wkv_fp32_v2_kernel`、`wkv_fp32_v2_small_warp_kernel`、`wkv_fp32_v2_short_block_kernel` 三路 family；small/short/large 由 `batch_size`、`max_seqlen` 和 dtype dispatch。
  - raw `decay_logits` retention transform、可选 decay bias、`state_pool [slots,H,D,D]` 原地更新。
- FP16-state recurrent packed-varlen：
  - `infer_recurrent_fp16_forward_varlen.cpp/.cu`
  - `wkv_fp16_v1_clone_kernel`、`wkv_fp16_v1_exact_kernel`、`wkv_fp16_seq_v2_kernel`、`wkv_fp16_one_cp_kernel`、`wkv_fp16_one_direct_kernel` family symbol 和 all-`T=1`/sequence dispatch。
  - D=64 使用 half2、cp.async 和 Albatross XOR-swizzled state staging；`Tis1`、`AddW0` 和 `Grid2D` 作为 upstream 同构 template specialization，packed varlen 只改 sequence/state 地址。canonical Albatross FP16 source 只实现 D=64；D=128/256 不选择任何 generic 或临时替代，而是在 binding 中 fail closed。
- 所有 recurrent public API 的 token layout 是 packed `[total_tokens,H,D]`，metadata 是 `cu_seqlens [B+1]`、`state_indices [B]`，sequence 非空且 slot 不重复。
- `prepare_recurrent_metadata` 生成带 identity、data pointer、shape/stride、tensor version、device、stream、`total_tokens`、`state_pool_size`、`max_seqlen` snapshot 的 native ticket。FP32/FP16 都能消费同一个 ticket；ticket 失配 fail closed。
- 不暴露 `log_decay`、`from_log_decay`、`elapsed_t` 或旧 recurrent alias。FP16
  recurrent 已接收按 state slot 管理的 `elapsed_state_pool`，并在 kernel 内按
  Albatross phase/dither 规则使用；`infer_recurrent_fp16_advance_i32_varlen`
  现在直接按 packed sequence length 迁移上游 `advance_i32` 的 slot advancement
  语义，复用同一 metadata ticket 并保持未选 slot 不变。外部 model caller 在完成
  一段上下文后调用该 module-local helper；FlashRWKV 本身不定义或串联完整 model，
  也不能把旧的独立 `elapsed_t` operator 当作 public compatibility API。

### 2.2 Lightweight inference

已经有 module-local Python/test/benchmark/native pair：

- `tmix/mix6`
- `tmix/kk_a_gate`
- `tmix/lnx_rkvres_xg`
- `tmix/vres_gate`
- `cmix/mix`
- `cmix/sparse`
- `tmix/linear`
- `tmix/normalization`
- `embedding`
- `head/linear`

`mix6`、`kk_a_gate`、`lnx_rkvres_xg`、`vres_gate`、`cmix/mix`、`cmix/sparse` 已有 module-local CUDA family 和 packed row/sequence contract。`update_shift_state_last_kernel` 的语义在 stateful mix family 内处理，不注册为独立 `elementwise` public API。

`tmix/linear`、`tmix/normalization`、`embedding` 和 `head/linear` 的 `.cu` body 已按 revision `ee3308f...` 机械迁移对应的 Albatross `rwkv7_v3a_ops.cu` caller body；`tmix/linear` 已接入 `linear_t_f16`、tanh/sigmoid `linear_t_act_f16`、`linear_t_vres_f16`、low-rank 和 caller-owned original-layout dispatch，`cmix/mix` 也已接入 dense FFN-down 的 C=4096 tuned table。fused TMix/CMix LN、indexed last-LN、head tuned caller 和 token helper 已有 module-local packed binding。FlashRWKV 只提供这些 standalone operator，不定义或串联完整 model；剩余差距是各 operator 的全量 shape/config、ragged boundary 和 racecheck acceptance。

### 2.3 train_temp recurrent and auxiliary

已加入 `sm90` source pair、Python wrapper、独立 test/benchmark 的 family：

- `tmix/wkv7` pretrain recurrent forward/backward：final state、grad final state、grad initial state、state-dot-a、checkpoint/chunk metadata、tail chunk 和 raw decay logits boundary。
- `tmix/a_gate`
- `tmix/vres_gate`
- `tmix/mix6`
- `tmix/kk_pre`
- `tmix/lnx_rkvres_xg`
- `cmix/mix`
- `head/l2wrap_ce`
- `loss/l2wrap_ce`

其中 recurrent body 保留 train_temp 的 training state/gradient 语义。最近一轮已经把
以下 auxiliary/loss/head `.cu` body 切换为 revision `9521024...` 对应的 canonical
train_temp body，并按 FlashRWKV 的 forward/backward module pair 拆开：a-gate、v-res
gate、mix6、KK-pre、LN/RKV/residual/XG、CMix、L2Wrap CE loss 和 head L2Wrap CE。
现有 C++ binding 只做函数名、固定 `head_size=64`、`vocab` 和 module-local output
适配；这些 family 的 focused correctness 已在新 native build 上通过。训练
recurrent 本身仍是保持 FlashRWKV initial-state/packed checkpoint contract 的
adapted body，因此不能把它描述成未经适配的 train_temp 原文件复制。

`loss/l2wrap_ce` 现在拥有：

- `flash_rwkv/loss/l2wrap_ce/__init__.py`
- `tests/loss/l2wrap_ce/test.py`
- `benchmarks/loss/l2wrap_ce/bench.py`
- `csrc/sm90/loss/l2wrap_ce/pretrain_bf16_{forward,backward}.cpp/.cu`

该入口验证 raw logits/targets、CE forward、L2Wrap argmax surrogate gradient、target shape/range rejection 和 native forward/backward timing boundary。

### 2.4 Chunk、RL/Infctx、StateTune

- `tmix/wkv7/chunk`：BF16 packed chunk public API，K1 prepare/K2 recurrence workspace boundary 保留在同一 public operator 内；raw decay logits 和 state pool contract 已验证。
- `rl_infctx/wkv7`：保留旧 source 中独立的 materialized affine、factor-recompute 和 output-replay body，已按 raw decay logits 和 module-local binding 机械迁移；没有通过 pretrain recurrent family call-through。focused contract 已覆盖 FP16/BF16、chunk size 16/32/64、tail、decay bias、materialized/recompute 一致性、caller-owned state pool 不变和 packed boundary rejection；native workspace/racecheck 及 scheduler-facing acceptance 仍未完成。
- `tmix/wkv7/statetune`：独立 binding/Python/test/benchmark；train_temp recurrent forward/backward body 已机械复制并改名为独立 StateTune symbols，保留 StateTune 的 initial-state gradient 和 nonzero-state 语义。现有独立测试已覆盖 train_temp recurrence、chunk boundary、boundary/state-dot-a、final state、initial-state gradient、输入梯度和 caller-owned initial-state 不被 forward 改写；workspace/racecheck acceptance 仍待完成。

## 3. 统一 contract 当前状态

sequence-dependent inference 使用：

```text
tokens:        [total_tokens,...]
cu_seqlens:    [B+1], int32
state_indices: [B], int32
```

当前 native ticket/validation 已覆盖：

- 起点为 0、终点等于 `total_tokens`、严格递增、sequence 非空；
- state slot 范围、同一 launch 内重复 slot；
- CUDA、contiguous、same device、dtype、shape、max-seqlen；
- ticket identity/data pointer/version/stream 失配；
- invalid metadata 时 output 填 invalid sentinel 且 state 不写入。

Python wrapper 不做 padding，也不暴露固定 `[1,T,...]` 为 canonical varlen layout。FP16 在未显式传入 `max_seqlen` 时由 metadata preparation ticket 计算并缓存，不再在每次 FP16 launch 中读取 CUDA offsets；BF16 chunk 由于 native workspace 必须预分配 max chunks，仍在 wrapper 的 metadata preparation 阶段解析 max seqlen。

需要继续修正或明确记录的边界：

- FP32 ticket preparation 的旧校验路径仍用 C++ `to(torch::kCPU)` 读取 metadata 以推导 max seqlen；这是 preparation 阶段的显式同步，不应扩展到每次 kernel launch。若要完全满足无 host metadata copy，需要改成 GPU-side reduction/status ticket。
- RL/Infctx Python metadata helper 仍有 `.cpu().tolist()`，目前只作为 chunk metadata preparation 使用；它不应被带入最终 vLLM scheduler hot path。
- TMix/CMix shift-state closure 需要在 operator 层再做一次 `T=1`、`T>1`、ragged boundary 和 racecheck 验证；完整 model graph 不属于 FlashRWKV 的实现范围。

## 4. Albatross canonical call-graph audit 结果

已对 canonical `rwkv7_fast_v3a.py` 的实际 caller 和以下 source family 做过审计：

```text
rwkv7_fast_v3a.py
rwkv7_fast_ops_fp16.{cpp,cu}
rwkv7_v3a_ops.{cpp,cu}
rwkv7_wkv_fp16_v2.{cpp,cu}
rwkv7_wkv_fp32_v2.{cpp,cu}
```

确认的实际 inference operator family 包括：

- WKV FP32/FP16 的 shape dispatch；
- TMix mix6、3D mix6、KK-A gate/2D、LN/RKV/residual/XG generic/warp/2D、v-res gate/vector2；
- CMix mix/3D、sparse up/down、T512 accumulator/reuse 变体；
- `emb_ln0_bf16_to_f16`、tokenwise/original-layout/transpose/activated/low-rank linear；
- add/LN、fused TMix/CMix LN、last-LN/indexed last-LN、`add_f16`、`advance_i32`。

当前已实现的 module path 覆盖了 canonical upstream caller graph 实际使用的主要 operator identity。FlashRWKV 不负责完整 Albatross model graph；operator library 仍需完成以下内容：

1. 对已接入的 `rwkv7_fast_v3a.py` shape/config predicate 做全量 operator matrix 验证，包括 C=4096/H=64 白名单、3D/2D/warp/vectorized threshold、CMix T512、sparse split2 和 last-layer/head policy；主要 predicate 已恢复，仍缺统一 operator trace/benchmark 证据；
2. 在每个 packed stateful operator 上验证 request boundary、shift state、FP16 elapsed slot 和 state immutability；不新增 model-level wrapper；
3. 为这些 v3a operator family 生成和运行 Albatross operator shape/config plus ragged benchmark matrix。

因此，当前 HANDOFF 的状态是“canonical body 已按 family 机械迁移，部分 operator-level binding/packed dispatch/acceptance 尚未闭环”；完整 model 定义、layer loop 和 model wrapper 不在目标范围内。

## 5. Source/build contract

当前 `setup.py` 构建 72 个 native source，其中新增 module source 使用相同 stem 的 `.cpp/.cu` pair。`tests/test_native_source_layout.py` 已把这组约束自动化。已检查的约束：

- 没有新的 `csrc/common/`；
- 没有新的 `elementwise` module；
- 没有新的 `infer_common_*`、`pretrain_common_*`、`_registration.cpp`；
- WKV 文件使用 `recurrent`/`chunk` marker，非 WKV operator 使用 `infer_fp16_forward_varlen` 或 `pretrain_bf16_{forward,backward}`；
- global bindings/registration 只负责 module registration glue；
- `_old` source、legacy compatibility Python、旧 provider/registry 没有被重新加入当前 setup source list；
- `build/`、`artifacts/`、`.egg-info`、cache 不属于迁移 source set。editable native `.so` 仅作为本地验证产物，不应提交。

当前仍应在后续 source-contract test 中自动化检查“每个 module 的 Python/tests/benchmarks/CUDA 路径一致”，特别是 `cmix/mix`、`tmix/mix6` 下兼容旧 workload 子目录的 benchmark entry。

## 6. 已执行验证证据

环境：

```text
GPU: NVIDIA GB10, compute capability 12.1
Torch: 2.13.0+cu130
CUDA runtime: 13.0
Python: 3.12.13
build target: TORCH_CUDA_ARCH_LIST=12.0
```

已成功执行：

```text
TORCH_CUDA_ARCH_LIST=12.0 \
  ./.venv/bin/python -m pip install -v --no-build-isolation -e .

./.venv/bin/python -m compileall -q flash_rwkv tests benchmarks
git diff --check
```

最近一次 full native build 已包含独立 RL/Infctx materialized/recompute/replay body、独立 StateTune body、FP16 elapsed varlen slot helper、Albatross FP16 五路独立 body（含 `Tis1`/`AddW0`/`Grid2D` specialization），以及 CMix sparse 的 ticket-based dispatch，并成功完成 `72/72` source pair 的编译、链接和 editable install。当前验证为：定向 WKV/sparse/lnx/source-contract 为 `74 passed, 1 warning`，全量 pytest 为 `116 passed, 1 warning`，compileall 和 `git diff --check` 通过。source-contract 还检查 `flash_rwkv` 没有 model/transformer class 或模型 forward 入口。已知 warning 是环境缺少 NumPy 导致 Torch functional tensor 的 warning，不是 kernel correctness failure。

重新构建命令为：

```text
TORCH_CUDA_ARCH_LIST=12.0 ./.venv/bin/python -m pip install -v --no-build-isolation -e .
```

构建完成后，Albatross WKV operator correctness matrix 在当前重链接 native extension 上以
`h32d64`、`h40d64`、`h64d64` 三个 operator shape、BF16 token I/O、decay bias 和
`stress` 打开重新验证，81 个 case 全部通过：每个 shape 的 21-case matrix、B=320/2048
decode、B=320 T=16，以及三组真实 ragged lengths。结果写入
`/tmp/flash-rwkv-wkv7-operator-shapes-correctness.json`，没有写入 worktree；JSON 中
`correctness.passed` 失败数为 0，compiled extension hash、GPU/SM、source revision 和
git status 均有记录。

`loss/l2wrap_ce` benchmark smoke 已在同一 GPU 上通过 correctness gate，并记录 native forward/backward latency、p10/p50/p90、throughput、source revision/hash、GPU/Torch/CUDA/Python 和 git status。FP16 五路/grid boundary 以及 TMix/CMix/KK-A/LN-XG/v-res packed stateful operator 的 `compute-sanitizer --tool racecheck` 共运行 13 个 cases，结果均为 `0 hazards (0 errors, 0 warnings)`；Albatross fixed-vs-uniform-varlen-vs-ragged-vs-vllm operator diagnostic、全量 standalone operator benchmark 和其余 stateful family 的 racecheck 尚未完成，不能以当前 correctness 结果代替这些验收。

## 7. 下一步执行顺序

1. 继续按 manifest 对 `rwkv7_v3a_ops` 的实际可达 split-K、row-tile、WMMA/CuBLASLt、transposed/original-layout、low-rank 和 fused norm operator 做全量 shape/config 对照；不得把整个 upstream monolith 复制成 global module，也不新增完整 model 定义。
2. 把 `rwkv7_fast_v3a.py` 的 C=4096/H=64、3D/2D/warp/vectorized、CMix T512、last-layer/head tuned predicate 收敛到 module-local automatic dispatch，并在 packed ragged 输入上验证边界。
3. 完成每个 stateful operator 的 packed token/state/shift-state API correctness 和 shift-state racecheck；不在 FlashRWKV 内新增完整 model caller path。
4. 对 WKV FP16 已接入的 elapsed/dither state-slot contract、`advance_i32` 的 packed
   slot advancement 和所有 Albatross tuned override 做独立 operator-level 验证；canonical
   FP16 body 只支持 D=64，D=128/256 必须保持 fail-closed，除非上游提供新的
   canonical family。
5. 将 reusable metadata ticket 继续扩展到其它 stateful fused family；`tmix/mix6`、`cmix/mix`、`cmix/sparse` 已复用 ticket snapshot/status/max-seqlen，RL/Infctx 的 `.cpu().tolist()` 只允许留在 preparation 边界，不得进入 scheduler hot path。
6. train_temp auxiliary/loss/head canonical body 已接入；StateTune 已完成独立 recurrence/gradient 对照，后续补 workspace alias/recompute/tail/chunk 的更完整证据，以及 RL 独立 strategy body 的 acceptance。
7. 最后运行 Albatross fixed-vs-uniform-varlen-vs-ragged-vs-vllm diagnostic、operator shape/config plus ragged benchmark、目标 GPU racecheck 和 clean rebuild；更新本文件为真实 acceptance status。

## 8. Worktree 操作边界

- 不执行 `git reset --hard`、`git checkout --`、删除 `_old`、清理用户已有删除或覆盖 `README.md`、`pyproject.toml`、`setup.py` 等用户修改。
- 不自动 commit、push、开 PR。
- 如果后续需要提交，先按路径区分本轮新增/修改和既有 dirty 内容，并保留 provenance、测试命令、GPU/SM、source revision 和 benchmark JSON。
