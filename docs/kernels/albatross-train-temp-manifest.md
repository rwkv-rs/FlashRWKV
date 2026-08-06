# RWKV7 Kernel Migration Manifest

更新时间：2026-08-05

这个 manifest 是当前迁移树的 source-of-truth。它只记录 canonical upstream
snapshot 中被 Python caller 实际选中的 family；历史目录、forced wrapper、死
symbol 和只在其它模型路径出现的实现不因为存在于 upstream tree 就进入
FlashRWKV public API。

## 来源与优先级

| 角色 | repository / revision | 允许的用途 |
| --- | --- | --- |
| inference body 与 shape policy | `BlinkDL/Albatross`, `ee3308f6922e59f2166c7fac3c5a192340a2b48e`, `faster3a_2607` | RWKV7 inference 数学、kernel family、默认 dispatch |
| train/pretrain body | `BlinkDL/RWKV-LM`, `952102498e9ed367ea0a59ee64106916d474d30f`, `RWKV-v7/train_temp` | training forward/backward、workspace、gradient 语义 |
| packed scheduler contract | `rwkv-rs/vllm-rwkv`, `6d683f9e49a2997e405c47edc147872c8609513b` | `cu_seqlens`、slot mapping、packed launch 参考；不提供 kernel body 或 fallback |
| BF16 chunk algebra | HANDOFF 指定的 FlashKDA revision `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` | K1 prepare/K2 recurrence 的 RWKV7 chunk 语义 |

所有 CUDA body 的本地头部还必须写明原始文件、SPDX/license、改动边界和
state/varlen 适配。`vllm-rwkv` 与 Albatross 有分歧时，Albatross inference
body 和 state layout 优先。

## Albatross `rwkv7_fast_ops_fp16` 实际 caller family

下表中的“当前状态”描述 FlashRWKV 当前树，而不是 upstream 是否存在该 symbol。

| caller identity | canonical upstream symbol/family | module ownership | 当前状态 |
| --- | --- | --- | --- |
| TMix shifted preparation | `tmix_mix6`, `tmix_mix6_3d`, `update_shift_state_last_kernel` | `tmix/mix6` | packed stateful family、metadata ticket 和 upstream C=4096/B/T whitelist 已迁移；外部 caller 负责组合该 module-local operator，FlashRWKV 不定义 model |
| TMix key/key-a gate | `tmix_kk_a_gate`, `tmix_kk_a_gate_2d` | `tmix/kk_a_gate` | generic/2D body 和 packed binding 已迁移；自动使用 C=4096/H=64/B*T<=65535 的 upstream head-grid predicate |
| TMix LN/RKV/residual/XG | `tmix_lnx_rkvres_xg`, `_warp`, `_warp_2d` | `tmix/lnx_rkvres_xg` | generic/warp/2D body 和 packed binding 已迁移；warp 使用 upstream head-task threshold 与 B=1 whitelist，tuned lnx 2D 保持关闭 |
| TMix value residual gate | `tmix_vres_gate`, vectorized/config path | `tmix/vres_gate` | scalar/vector2 body 和 packed binding 已迁移；C=4096、rows 64..65535 自动选择 upstream 128/256-thread policy |
| CMix shifted preparation | `cmix_mix`, `cmix_mix_3d`, `update_shift_state_last_kernel` | `cmix/mix` | packed stateful family、metadata ticket 和 C=4096/B/T whitelist 已迁移；外部 caller 负责组合该 module-local operator，FlashRWKV 不定义 model |
| CMix sparse up/down | `cmix_sparse_*`, `relu_square`、T512 accumulator/reuse | `cmix/sparse` | exact up/rows、SPMV、split2、T512 accumulator/reuse body、packed ticket 和 upstream split2/T512 dispatch 已迁移 |
| tokenwise helpers | `add_vec`, `add_vec_2d`, `add_f16`, `advance_i32` | 实际 caller 的 `tmix`/`head`/normalization | exact helper body 已按 caller 文件保留；add_vec、add_f16、advance_i32 均已有 caller-owned binding，不创建 `elementwise` |

`T=1` 和 `T>1` 的 shift-state 行为属于对应 `tmix`/`cmix` family；
`update_shift_state_last_kernel` 不注册为独立用户 API。

## Albatross `rwkv7_v3a_ops` 实际 caller family

canonical `rwkv7_fast_v3a.py` 的 operator-caller 审计确认以下 family 会被上游完整
RWKV7 caller graph 实际调用：

| caller identity | upstream family | FlashRWKV owner | 当前状态 |
| --- | --- | --- | --- |
| embedding + initial LN | `emb_ln0_bf16_to_f16` | `embedding` | exact upstream body/launch 已迁移；packed row caller 已接入 |
| ordinary linear | `linear_f16`, split-K/rows variants | `tmix/linear` 或实际 caller | exact ordinary/split-K/row body和 packed caller dispatch 已迁移；CMix dense FFN-down 的 C=4096 rows 48/256 table 在 `cmix/mix` owner 中接入 |
| original-layout linear | `linear_f16_orig`, row/exact/WMMA variants | `tmix/linear` 或实际 caller | exact original/row/exact/WMMA body 和 generic/attention/ffn/head caller-specific policy 已迁移 |
| transposed/activated linear | `linear_t_f16`, `linear_t_act_f16`, `linear_t_vres_f16` | `tmix/linear` | exact vectorized body 已迁移，并已接入 packed caller binding |
| low-rank TMix | `linear_wag_rank_in/out`, `linear_wagv_rank_in/out` | `tmix/linear` | exact low-rank body 已迁移并接入 caller-owned module dispatch；model-level composition 留给外部 caller |
| ordinary/add/LN | `layer_norm_f16`, `add_f16`, `add_layer_norm_f16` | `tmix/normalization` | exact body、small/stats config family 和 packed functional binding 已迁移 |
| fused TMix/CMix LN | `add_layer_norm_tmix_mix6_f16`, `add_layer_norm_cmix_mix_f16` | 对应 `tmix`/`cmix` caller | T=1 packed state-slot binding、Welford tuned stats policy 和 shift-state update 已迁移；非 T=1 保持 canonical caller fail-closed |
| last-layer LN | `add_last_layer_norm_f16`, indexed variant | `head`/normalization caller | exact rectangular/indexed body和绝对 packed-row index binding 已迁移 |
| head logits | ordinary/transposed/CuBLASLt/WMMA linear | `head/linear` | exact original-layout body、all-logits/last-logits tuned table 和 packed caller binding 已迁移 |

这里的“待迁移”不是允许复制整个 `rwkv7_v3a_ops.cu`。实现时必须按上述
caller/module 拆分，public API 只保留实际 call-reachable identity。

## WKV family

| dtype/state | canonical family | FlashRWKV path | 当前状态 |
| --- | --- | --- | --- |
| FP32 state | `wkv_fp32_v2_kernel`, `small_warp`, `short_block` | `tmix/wkv7/infer_recurrent_fp32io16_forward_varlen` | 三路 packed dispatch 已有；canonical state 为 `[K,V]`，21-case correctness 已通过 |
| FP16 state, `T=1` | `clone`, `one_cp`, `one_direct` | `tmix/wkv7/infer_recurrent_fp16_forward_varlen` | family symbol、`Tis1`/`AddW0`/`Grid2D` specialization、one-path dispatch、state-slot elapsed/dither phase 和按 packed sequence length 迁移的 `advance_i32` helper 已有；仍需补齐 operator-level boundary/racecheck acceptance |
| FP16 state, sequence | `exact`, `seq_v2` | 同上 | canonical D=64 tuned body、`AddW0`/`Grid2D` dispatch 已验证；上游没有 D=128/256 FP16 family，非 D=64 fail closed |

统一 public sequence contract 是 packed token rows、`cu_seqlens [B+1]`、
`state_indices [B]` 和 reusable metadata ticket。raw `decay_logits` 是唯一
decay boundary；不恢复 `log_decay`、`from_log_decay`、独立 `elapsed_t` alias。

## train_temp family

canonical source root：`RWKV-v7/train_temp/cuda`，revision
`952102498e9ed367ea0a59ee64106916d474d30f`，license Apache-2.0。

| source | FlashRWKV module | 当前 body |
| --- | --- | --- |
| `rwkv7_clampw_v3.cpp`, `rwkv7_clampw_v3_for_h100.cu` | `tmix/wkv7` | canonical BF16、N=64、chunk-16 forward/backward body 原样迁移；soft-clamp、零初始 state、内部 `s`/`sa` workspace 和 H100 launch 保持上游 contract |
| `rwkv7_tmix_a_gate_bf16.cu` | `tmix/a_gate` | canonical body 已切分为 forward/backward pair |
| `rwkv7_tmix_vres_gate_bf16_v3.cu` | `tmix/vres_gate` | canonical body 已切分为 forward/backward pair |
| `rwkv7_tmix_mix6_bf16_v5.cu` | `tmix/mix6` | canonical body 已切分为 forward/backward pair |
| `rwkv7_tmix_kk_pre_bf16_v5.cu` | `tmix/kk_pre` | canonical body 已切分；固定 `head_size=64` 是 module-local 适配 |
| `rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cu` | `tmix/lnx_rkvres_xg` | canonical body 已切分为 forward/backward pair |
| `rwkv7_cmix_bf16_v5.cu` | `cmix/mix` | canonical body 已切分；CMix helper 仍归属 cmix |
| `rwkv7_l2wrap_ce_bf16_v2.cu` | `loss/l2wrap_ce` | canonical body 已切分；vocab 为 binding-local shape parameter |
| `rwkv7_head_l2wrap_ce_bf16_v4.cu` | `head/l2wrap_ce` | canonical row-chunk/reduction body 已接入 module-local wrapper |

pretrain recurrent API 不复用 inference state pool contract，也不暴露 initial/final
state、packed metadata 或兼容旧 `wkv7_cuda` 的入口；公开 wrapper 直接保持 clampw v3
的 BF16 `[B,T,C]`、`T % 16 == 0` 和内部 workspace contract。

## RL/Infctx 与 StateTune body

| source | FlashRWKV module | 当前 body |
| --- | --- | --- |
| retained `rl_infctx_common_chunk_fp32io16_forward_materialized.cu` | `rl_infctx/wkv7` | materialized affine transform/build/scan/replay body 已机械迁移为独立 raw-decay source；当前固定保留的 `(2 warps, 1 stage, 64-row tile)` 是 canonical source configuration，不是 generic fallback；FP16/BF16、chunk 32/64、tail 和 state immutability 已通过 focused contract |
| retained `rl_infctx_common_chunk_fp32io16_forward_recompute.cu` | `rl_infctx/wkv7` | factor-recompute boundary scan/replay body 已机械迁移为独立 raw-decay source；与 materialized 在 ragged/tail 输入上通过 focused oracle |
| retained `rl_infctx_common_chunk_fp32io16_backward_replay.cu` | `rl_infctx/wkv7` | output replay body 已独立 binding；它不是 train_temp backward alias；packed boundary rejection 已在 Python preparation 层 fail closed |
| `wkv7_cuda.cu`, `wkv7_cuda_fp32.cu` from `train_temp` | `tmix/wkv7/statetune` | forward/backward body 已机械复制并独立改名；StateTune 保留 initial-state gradient 和 nonzero-state contract，独立 oracle 已覆盖 recurrence/checkpoint/gradient |

## 不进入 manifest 的内容

- `faster2`、`faster3b`、`mega` 和其它历史目录；
- `*_forced` 作为 FlashRWKV public API；forced 选择只能是内部 dispatch policy；
- 只声明没有 `rwkv7_fast_v3a.py` 实际 caller 的 wrapper/dead symbol；
- `elementwise`、`csrc/common`、global `registry/provider`；
- vllm-rwkv kernel body、fallback provider 或默认 benchmark provider；
- 旧 `log_decay`、`from_log_decay`、独立 `elapsed_t` compatibility operator。

## 验收状态

已验证：native build、focused CUDA pytest、全量 pytest、WKV Albatross 81-case operator
correctness matrix、FP16 五路/grid boundary 及 TMix/CMix/KK-A/LN-XG/v-res packed
stateful operator 的 compute-sanitizer racecheck（13 cases、0 hazards）、Python compile
和 source provenance 的基础检查。

未完成：完整 standalone operator-level shape/config matrix、Albatross
fixed-vs-uniform-varlen-vs-ragged-vs-vllm operator diagnostic、RL/Infctx strategy/workspace
acceptance、racecheck 和目标 GPU 全量 operator benchmark。FP16 D=64 五路 family、
elapsed/dither slot advancement 和已接入 stateful operator 的 focused correctness 已通过；
其余 acceptance 不能用当前 smoke/correctness 结果替代。FlashRWKV 不定义完整 RWKV7
model；model-level composition 由外部 caller 负责。
FP16 D=128/256 不在当前
canonical Albatross source 覆盖范围内，不能用临时 generic body 填补。
