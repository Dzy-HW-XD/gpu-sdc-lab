# Phase 0：长上下文 agent prefill 的 NVBitFI 注错验证

日期：2026-10-21  
状态：**PASS（NVBitFI kernel 映射与注入链路通过）**  
节点：node20，共享 A100  
模型：`Qwen/Qwen2.5-1.5B-Instruct`（BF16）  
Case：`policy_001`，2,264 prompt tokens

## 1. Phase 0 目标

验证在 HuggingFace Qwen 的第一次长 prefill 中，NVBitFI 是否能：

1. 识别每层、每类 GEMM 的 kernel；
2. 用 `(kernel_name, kernel_count)` 精确命中目标 GEMM；
3. 在 HMMA FP32 accumulator 上翻转 bit 28/29/30；
4. 注入后仍能完成 decode；
5. 至少有一个注入能改变最终 agent 答案。

## 2. 长上下文 Case

| 项目 | 值 |
|---|---|
| case_id | policy_001 |
| category | policy_qa |
| system context | 内部支持手册，含退款政策、干扰段落 |
| prompt tokens | 2,264 |
| user query | Tier 2 损坏商品最高退款多少？ |
| expected answer | `120` |
| clean output | `120` |

clean 运行：

- 输出：`120`
- 耗时：0.815 s（无 NVBitFI）
- 显存：3,714 MiB

## 3. Kernel Profile

使用 NVBitFI profiler 对 **同一条 generate() 路径**做 profile：

- profile 耗时：160.9 s
- profile 中 `mma > 0` 的 kernel：**253 个**
- 结构：Qwen1.5B 有 28 层；每层 9 个 GEMM；最后 1 个 LM head kernel

**28 × 9 + 1 = 253**

每层 9 个 GEMM 的顺序：

| 顺序 | stage | kernel 形状 |
|---:|---|---|
| 1 | qkv_q | `128x256 ... tn` |
| 2 | qkv_k | `128x64 ... tn` |
| 3 | qkv_v | `128x64 ... tn` |
| 4 | qk | `256x128 ... tn` |
| 5 | pv | `128x128 ... nn` |
| 6 | wo | `128x256 ... tn` |
| 7 | mlp_gate | `128x256 ... tn`，ctas=630 |
| 8 | mlp_up | `128x256 ... tn`，ctas=630 |
| 9 | mlp_down | `128x256 ... tn` |

Layer 14（M）的实际 global kernel index：

| stage | global index | kernel |
|---|---:|---|
| qkv_q | 1219 | `ampere_bf16_s16816gemm_bf16_128x256_ldg8_relu_f2f_stages_64x3_tn` |
| qkv_k | 1220 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x4_tn` |
| qkv_v | 1221 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x4_tn` |
| qk | 1238 | `ampere_bf16_s1688gemm_bf16_256x128_ldg8_f2f_stages_32x1_tn` |
| pv | 1246 | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nn` |
| wo | 1249 | `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` |
| mlp_gate | 1261 | `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` |
| mlp_up | 1263 | `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` |
| mlp_down | 1265 | `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` |

完整 253-kernel 映射：`results/agent_phase0/mapping_253.csv`。

## 4. 注入结果

注入方式：NVBitFI `mma` 组，bitflip single-bit，M 层 layer 14，middle instruction。

| stage | bit | kernel index | mask | before | after | output | 结果 |
|---|---:|---:|---|---|---|---|---|
| qkv_q | 30 | 1219 | `0x40000000` | `0xbdb2b9c2` | `0xfdb2b9c2` | `120` | masked |
| qk | 30 | 1238 | `0x40000000` | `0xc313cb30` | `0x8313cb30` | `120` | masked |
| qk | 28 | 1238 | `0x10000000` | `0xbea641a0` | `0xaea641a0` | `120` | masked |
| **qkv_v** | **30** | **1221** | `0x40000000` | `0x3f5d10dc` | `0x7f5d10dc` | **`!!!!!!!!!!!!!!!!`** | **task failure / literal garble** |
| pv | 30 | 1246 | `0x40000000` | `0xb97818ac` | `0xf97818ac` | `120` | masked |
| wo | 30 | 1249 | `0x40000000` | `0x3e8d1a50` | `0x7e8d1a50` | `120` | masked |
| mlp_down | 30 | 1265 | `0x40000000` | `0xbe321c52` | `0xfe321c52` | `120` | masked |

所有 7 个 run 的日志都出现：

- `Injection data`
- `opcode: HMMA`
- 正确 kernel index 和 kernel name
- 没有 `Error not injected`

### 关键结果

**qkv_v, layer 14, bit 30：**

- clean：`120`
- fault：`!!!!!!!!!!!!!!!!`
- 16 个重复感叹号，目标答案完全丢失
- 判为 literal garble / task failure

这证明：

- Qwen 的 qkv_v GEMM kernel 可以被 NVBitFI 精确命中；
- 长上下文 prefill 中注入的 bit 可以传播到最终回答；
- 输出可以变成完全不可用的乱码，而不是 NaN/Inf。

## 5. 运行时间

| 项目 | 时间 |
|---|---:|
| clean prefill + decode | 0.815 s |
| NVBitFI profile | 160.9 s |
| 单次注入 + decode | 424–769 s（约 7–13 分钟） |

**重要结论：** 原计划的 12-case Phase 1 矩阵（1,944 个 fault）如果全部用 NVBitFI，单卡约需：

**1,944 × 7 分钟 ≈ 226 小时（约 9.4 天）**

即使用 4 张 GPU 并行，也约需 **2.4 天**。原矩阵在 NVBitFI 下不可行，需要缩小或分层。

## 6. Phase 0 结论

**PASS：**

- kernel 映射成功：253 个 MMA kernel，严格符合 28×9+1；
- 7 个注入都成功写入 HMMA，mask/opcode/kernel 正确；
- qkv_v bit 30 产生了明确的任务失败/乱码；
- clean 输出和注入输出都能正常 decode；
- 没有 Python/CUDA crash、没有 NaN/Inf。

**注意事项：**

- 单次 single-bit 注入大多数被 masked（7 个中 6 个答案不变）；
- 要估计 stage/bit 的失败率，每个 cell 需要多次注入，不能只做 1 次；
- NVBitFI 长 prefill 成本高，原 1,944-run 矩阵不现实。

## 7. 建议的 Phase 1 重设计

### 方案 A（推荐）：先做 NVBitFI stage screen

- 3 个代表 case；
- Qwen1.5B-Instruct；
- E/M/L 三层；
- 9 个 stage；
- 3 个 bit；
- 每 cell 1 次；
- 3 × 3 × 9 × 3 = **243 runs**。

单卡约 28 小时，4 GPU 并行约 **7 小时**。  
先回答“哪几个 stage/bit 最敏感”，再决定是否扩到 12 case。

### 方案 B：NVBitFI 只做验证，软件注入做完整矩阵

- NVBitFI 保留 Phase 0 和少量代表 cell；
- 完整 12-case ×9-stage 矩阵用软件 GEMM 输出注入；
- 优点：可跑完；缺点：不是全程 NVBitFI。

### 方案 C：缩短上下文

- 将 system context 从 2.2k 缩短到 ~800 token；
- 单次注入可能从 7 分钟降到 2–3 分钟；
- 但仍不是“很长专有内容”，agent 场景真实性下降。

## 8. 产物

- 报告：`docs/PHASE0_AGENT_NVBITFI.md`
- 计划：`docs/EXPERIMENT_PLAN_AGENT_GEMM.md`
- case：`data/agent_phase0_case.json`
- clean：`results/agent_phase0/clean.json`
- kernel 映射：`results/agent_phase0/mapping_253.csv`
- 注入汇总：`results/agent_phase0/phase0_injections.csv`
- profile：`results/agent_phase0/profile_gen/`
- 各 stage 注入目录：`results/agent_phase0/inject_gen_*`
