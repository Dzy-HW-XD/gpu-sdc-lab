# Long-context agent GEMM fault-injection plan

Status: **design proposal, not executed**.  
Scope: long proprietary prefill → agent answer → GEMM-output bit injection.  
Reference: `047_attn_5gemm_full` (five attention GEMM stages, bits 24–30, early/middle/late positions, random head).

## 1. What changes from the previous experiment

| 项目 | 上一版 | 新实验 |
|---|---|---|
| 输入 | 24 条短 prompt | 12–24 条长上下文 agent 场景 |
| 注入位置 | 最后一个 prompt token 的 residual activation | GEMM 输出元素 |
| 注入 bit | 28–31 | **28–30** |
| 注入阶段 | 任意 block 输入 | **QKV_Q / QKV_K / QKV_V / QK / PV** |
| 注入位置 | 最后一个 prompt token | 长 prefill 内的 early / middle / late token |
| 目标 | 输出是否乱码/错误 | agent 任务答案是否被破坏 + 输出形态 |

新实验不再只打最后一个 token，而是模拟：**很长的 system/proprietary context + user query 一次 prefill，随后回答**。

## 2. Agent 场景与数据集

### 2.1 对话形式

每个 case 固定为一次 prefill：

```
system: 很长的专有内容（内部政策、工具目录、API 文档、产品/价格表、干扰信息）
user:   一个需要依据上述内容回答的问题
assistant: 短答案（金额、工具名、endpoint、参数或 JSON）
```

- 专有内容是**合成的内部资料**，不使用真实公司数据。
- 目标答案在 clean 输出开头，便于严格判定。
- 长上下文目标长度：**1,500–3,000 tokens**；Qwen2.5 支持 32k 上下文。
- 采用 Qwen chat template；注入发生在第一次 prefill，decode 不再注错。

### 2.2 数据集规模

建议 **24 个 case**，每个 case 有唯一可验证答案：

| 类别 | 数量 | 例子 | 目标答案 |
|---|---:|---|---|
| policy QA | 8 | 内部退款政策中 Tier 2 损坏商品最高退多少？ | `120` |
| tool routing | 8 | 用户要求重置 MFA，应调用哪个工具和参数？ | `reset_mfa(user_id="U-1042")` |
| API lookup | 8 | 修改订单状态的 endpoint 是什么？ | `/v2/orders/{id}/status` |

- 每个 case 的上下文包含唯一 fact 和多段干扰内容。
- 8 个 policy + 8 个 tool + 8 个 API = 24。
- Pilot 先用 12 个 case（每类 4 个），通过 clean 100% 后再扩到 24。

### 2.3 Clean 门槛

候选模型：

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`

正式统计前必须满足：

- 两个模型都在最终 case 集上 **clean 100%**（目标答案严格匹配）；
- 如果 0.5B 无法达到 100%，只保留 1.5B，或简化 case；
- 不把 clean 错题带入注错统计。

## 3. GEMM 注入模型

### 3.1 五个 GEMM stage

参考 047：

| stage | LLM 中的位置 | 注入元素 |
|---|---|---|
| `qkv_q` | 某层 Q projection GEMM 输出 | 选定 token 的一个 hidden element |
| `qkv_k` | 某层 K projection GEMM 输出 | 选定 token 的一个 hidden element |
| `qkv_v` | 某层 V projection GEMM 输出 | 选定 token 的一个 hidden element |
| `qk` | 某层某 head 的 QK^T score GEMM | 一个 (query token, key token) score |
| `pv` | 某层某 head 的 P·V GEMM | 一个 (query token, value channel) 元素 |

- **QKV** 是 projection-level fault，影响该 token 的整个 Q/K/V 向量；每次仍只翻转一个元素。
- **QK/PV** 使用随机 head，和 047 的 random-head 设定对齐。
- 第一轮只做这 5 个 GEMM；`Wo`、MLP up/down、LM head 作为后续扩展。

### 3.2 Fault 参数

| 参数 | 取值 |
|---|---|
| 精度 | FP32 activation/GEMM 输出 |
| bit | **28, 29, 30** |
| 注入元素 | 每个 run 随机选择一个元素（固定 seed） |
| 注入次数 | 每个 run 恰好 1 bit |
| 注入时机 | 第一次 prefill |
| 注入位置 | 长上下文内的 early / middle / late token |
| 层 | E / M / L（每模型 3 层） |
| 校验 | 注入前后 int32 XOR，恰好 1 bit |

### 3.3 与 047 的对应关系

047 的故障模型是 **NVBitFI 在 HMMA destination register（FP32 accumulator）翻转一个 bit**，阶段为 `qkv_q/qkv_k/qkv_v/qk/pv`，位置为 early/middle/late。

本实验先采用**软件 GEMM 输出注入**：

- 在 PyTorch 的 Q/K/V projection、QK^T、PV GEMM 输出上注入；
- 注入一个 FP32 元素的 1 个 bit；
- 对每个 run 验证 bit 数；
- 保证长上下文和大量组合可复现。

后续用 NVBitFI 对少量 stage/bit/层组合做**硬件验证**：

- 验证软件 GEMM 注入与真实 HMMA destination-bit fault 的失败形态一致；
- 不把全矩阵都放到 NVBitFI 上，避免长上下文 × 多阶段的运行成本。

## 4. 注入位置与矩阵

### 4.1 Token 位置

每个 case 的 prefill 包含：长 system/context + user query。  
在 **context 内容**内选三个 token 位置：

- early：上下文约 10% 位置
- middle：约 50%
- late：约 90%

不使用“最后一个 prompt token”作为唯一位置。QK/PV 的 query token 使用该位置；QKV 的注入 token 也使用该位置。

### 4.2 Pilot 矩阵

12 case pilot：

| 因子 | 取值 |
|---|---|
| 模型 | Qwen2.5-0.5B-Instruct / 1.5B-Instruct |
| case | 12 |
| 层 | E / M / L（3） |
| stage | qkv_q / qkv_k / qkv_v / qk / pv（5） |
| position | early / middle / late（3） |
| bit | 28 / 29 / 30（3） |

计算：

- fault：2 × 12 × 3 × 5 × 3 × 3 = **3,240**
- sham：2 × 12 × 3 × 5 = **360**
- clean：2 × 12 = **24**
- 合计约 **3,624 个生成**

正式版扩到 24 case 后：

- fault：2 × 24 × 3 × 5 × 3 × 3 = **6,480**
- 其他控制同比增加。

### 4.3 控制

- clean：每个 case 在每个模型上生成一次；
- sham：安装 GEMM hook 但不翻 bit；每个模型每个 case × 3 层 × 5 stage；
- 重复 clean：至少对 pilot 模型重复一次，确认 greedy 输出稳定；
- 所有 fault run 记录 bit 改变数、注入值是否 finite、runtime error。

## 5. 判定指标

### 5.1 任务答案

- **task correct**：目标答案/JSON 在答案开头严格匹配；
- **answer wrong**：目标答案缺失或与 clean 不一致；
- **JSON invalid**（tool/API 类）：不能解析为预期结构。

### 5.2 输出形态

沿用上一版：

- **literal garble**：替换字符、控制字符、空输出、退化重复、异常书写系统；
- **semantic wrong**：答案缺失、char similarity < 0.4、clean PPL ratio > 5；
- **catastrophic**：literal garble 或 semantic wrong；
- **partial / preserved**：答案仍出现但被改写，或完全保持。

### 5.3 GEMM 局部效应

对每个 run 额外记录：

- 注入 stage/层/head/位置的 local delta；
- 下游 GEMM 输出是否超过 047 的 L4 阈值（`|Δ| > 1e-4 + 1e-3|ref|`）；
- 受影响的最大层和 token 后缀；
- 用于比较“局部数值异常”与“最终答案错误”。

## 6. 输出与图

每个 run 的 CSV 至少包含：

- case_id, category, context_tokens
- model, layer, stage, bit, position, token_index, head, element_index
- old_int, new_int, bit_diff, fault_value, fault_finite
- clean_output, fault_output
- task_correct, literal_garble, semantic_wrong, catastrophic, partial, preserved
- local_l4_affected, downstream_l4_layers, runtime_error

一张总结图：

- **(a)** stage × bit 的 catastrophic rate；
- **(b)** stage × layer（bit 28–30 合并）；
- **(c)** early/middle/late position 的 catastrophic rate，或 outcome composition。

## 7. 主要风险

1. **长上下文 clean 100% 难**：0.5B 可能无法稳定检索；先用 Instruct、唯一 fact、短答案，并做 clean gate。
2. **GEMM hook 实现复杂**：Qwen 默认 SDPA，需要改成 explicit eager attention 才能注入 QK/PV；必须用 clean 输出验证 hook 无副作用。
3. **bit 28–30 可能被 masked**：同一层/stage 的某些元素可能完全不影响答案；需要按 stage/位置分层报告，而不是只看平均。
4. **NVBitFI 与软件注入不完全等价**：先软件全矩阵，后用 NVBitFI 做小规模硬件验证。
5. **长上下文运行成本**：12-case pilot 先行；clean 100% 和 hook 验证通过后再扩 24。
6. **Agent 答案格式**：tool/API 类要明确 JSON/字段；否则 0.5B 容易格式错误，clean gate 过不了。

## 8. 待确认

1. 模型：`Qwen2.5-0.5B-Instruct` + `1.5B-Instruct`（推荐），还是继续 base？
2. Pilot 规模：12 case（推荐）还是直接 24 case？
3. GEMM stage：仅 047 的 5 个 stage（推荐），还是加 `Wo` / MLP？
4. 注入实现：软件 GEMM 全矩阵 + NVBitFI 小规模验证（推荐），还是全部 NVBitFI？

确认后先做 **12-case clean 校准和 hook 自测**，通过后再跑 fault 矩阵。
