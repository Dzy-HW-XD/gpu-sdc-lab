# Long-context agent GEMM fault-injection plan (NVBitFI)

Status: **design proposal, Phase 0 executed and passed; Phase 1 pending**.  
Confirmed choices: Qwen Instruct ×2; 12-case pilot; add Wo/MLP; **NVBitFI injection during prefill**.

> **Phase 0 result (2026-10-21):** Qwen1.5B-Instruct, 2,264-token case, clean answer `120`.
> Profile found exactly **253 MMA kernels = 28 layers × 9 GEMMs + LM head**. NVBitFI successfully
> injected bit 30 into layer-14 `qkv_v` (kernel index 1221, mask `0x40000000`); output became
> `!!!!!!!!!!!!!!!!`. Other tested stage/bit points were masked. Single injection costs
> **~7–13 min** with NVBitFI; the original 1,944-run matrix would take ~9 days on one GPU.
> See `docs/PHASE0_AGENT_NVBITFI.md` for full evidence and the Phase-1 redesign options.

## 1. 实验目标

模拟 agent 对话的第一次长 prefill：

1. system 中放入很长的专有内容（内部政策、工具目录、API 文档、干扰段落）；
2. user 提问，要求依据这些内容给出短答案；
3. 在 **prefill 的 GEMM 内部**用 NVBitFI 翻转一个 bit（28/29/30）；
4. decode 回答，判断答案是否仍正确，以及是否出现乱码或完全错误。

与上一版的区别：

| 项目 | 上一版 | 本实验 |
|---|---|---|
| 输入 | 短 prompt | 1,500–3,000 token 的长专有上下文 |
| 注入位置 | 最后一个 prompt token 的 residual activation | **GEMM 内部 HMMA 目的寄存器** |
| 注入 bit | 28–31 | **28, 29, 30** |
| 注入阶段 | block 输入 | QKV_Q / QKV_K / QKV_V / QK / PV / Wo / MLP |
| 注入时机 | 最后一个 token | 第一次 prefill，长上下文内的 GEMM |
| 注入工具 | 软件 hook | **NVBitFI** |

## 2. Agent 数据集

### 2.1 对话格式

```
system:
  你是一个内部支持 agent。只能依据下面的专有资料回答。
  [很长的内部政策 / 工具目录 / API 文档 / 价格表 / 干扰段落]

user:
  [一个需要依据专有内容回答的问题]

assistant:
  [短答案，例如金额、工具名、endpoint、参数或 JSON]
```

- 合成“专有内容”，不使用真实公司数据。
- 每个 case 有一个唯一可验证答案，答案放在输出开头。
- 上下文长度目标：**1,500–3,000 tokens**。
- 使用 Qwen chat template；注错只发生在第一次 prefill，decode 不再注错。

### 2.2 Pilot 数据量

12 个 case，4 个/类：

| 类别 | 数量 | 查询 | 目标答案 |
|---|---:|---|---|
| policy QA | 4 | 某等级客户的最高退款/折扣/时限 | 金额或天数 |
| tool routing | 4 | 某请求应调用哪个内部工具和参数 | 工具名 + 参数 |
| API lookup | 4 | 修改/查询对象应使用哪个 endpoint | endpoint 字符串 |

后续扩到 24 个 case（每类 8 个）。

### 2.3 Clean 100% 门槛

模型：

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`

正式 fault 矩阵前必须满足：

- 两个模型在 12 个 case 上均 **clean 100%**；
- 目标答案严格匹配（金额/endpoint/工具名 + 参数）；
- 任何 clean 错题直接替换或删除，不进入统计；
- 如果 0.5B 不能稳定达到 100%，只保留 1.5B。

### 2.4 数据格式

每个 case 一条 JSON：

| 字段 | 含义 |
|---|---|
| `case_id` | 唯一 ID |
| `category` | `policy_qa` / `tool_routing` / `api_lookup` |
| `system_context` | 1.5–3k token 的专有资料 |
| `user_query` | 用户问题 |
| `expected_answer` | 严格答案 |
| `answer_regex` | 可选，用于工具名/endpoint/JSON |
| `context_tokens` | clean tokenizer 下的上下文长度 |
| `difficulty` | 用于分层统计 |

构造规则：

- 每个 case 的答案在专有内容中只出现一次；
- 干扰段落包含相似术语，避免模型仅靠关键词命中；
- 答案短、可精确判定；
- 系统内容不含真实公司数据或密钥；
- 12 个 pilot case 通过 clean gate 后再扩到 24。

## 3. NVBitFI 注入设计

### 3.1 故障模型

对齐 047：

- NVBitFI `mma` 组；
- 在 HMMA 的 **FP32 destination register / accumulator** 上翻转 1 个 bit；
- bit：**28, 29, 30**；
- 每个 run 只注入 1 次；
- decode 不再注入。

### 3.2 GEMM 阶段

第一轮包含 047 的五个阶段并加入 Wo/MLP：

| stage | LLM 中的 GEMM | 说明 |
|---|---|---|
| `qkv_q` | Q projection | 每层一个 GEMM |
| `qkv_k` | K projection | 每层一个 GEMM |
| `qkv_v` | V projection | 每层一个 GEMM |
| `qk` | QK^T score | eager attention，按层 |
| `pv` | P·V | eager attention，按层 |
| `wo` | attention output projection | 每层一个 GEMM |
| `mlp_gate` | SwiGLU gate projection | 每层一个 GEMM |
| `mlp_up` | SwiGLU up projection | 每层一个 GEMM |
| `mlp_down` | MLP down projection | 每层一个 GEMM |

**层：** E / M / L，每模型 3 层。

### 3.3 长上下文中的位置

NVBitFI 通过动态指令序号选择 fault；在 047 中使用 early/middle/late instruction fraction。本实验同样记录：

- injection instruction fraction；
- 对应 kernel 的 `kernel_count`；
- 若可映射，记录大致 token 范围；
- 不把“最后一个 prompt token”作为唯一位置。

正式矩阵的三种位置：

- early：GEMM instruction 范围前 1/3；
- middle：中 1/3；
- late：后 1/3。

### 3.4 NVBitFI 与 HuggingFace 的接入

需要一个专门的长上下文 agent workload，而不是 047 的 `attention_forward` 小程序：

1. 读取 12 个 case；
2. 用 Qwen chat template 拼 system + user；
3. 做一次 prefill；
4. 在 prefill 期间被 NVBitFI 注入；
5. decode 短答案并写 JSON；
6. 保存 clean/fault 输出和注入日志。

关键工程点：

- 必须使用 **eager attention** 或显式 Q/K/V matmul，避免 SDPA 把 QK/PV 融合成不可映射的 kernel。
- 需要用 **NVTX range** 标出每层、每 stage；再用 PyTorch profiler / CUPTI 把 kernel launch 与 stage 对齐。
- NVBitFI profile log 给出 kernel name、global index、mma 计数；将 stage/layer 映射到 `(kernel_name, kernel_count)`。
- 每个 stage/layer 至少做一次“已知 fault 注入 → 输出变化”验证，确认映射没有错位。
- 如果某个 stage 的 GEMM 被融合/拆成多个 kernel，先记录，不强行猜测；无法稳定映射的 stage 暂不进入正式矩阵。

### 3.5 为什么先做 Phase 0

NVBitFI 在 047 的自定义 CUDA workload 上已验证；但 HuggingFace Qwen 的 GEMM launch 顺序、cuBLAS kernel 名称和融合行为与 047 不同。正式跑矩阵前必须先验证：

- 一次长 prefill 的 profile 中，能否识别出每层 9 个 GEMM；
- 能否用 `(kernel_name, kernel_count)` 精确命中目标 layer/stage；
- 一次 bit 28/29/30 注入是否真的 `Injection data`，且 clean 输出在无 fault 时不变；
- 注入后是否仍能完成 decode，而不是进程崩溃。

## 4. 运行矩阵

### 4.1 Phase 0：NVBitFI 可行性

| 项目 | 设置 |
|---|---|
| 模型 | Qwen2.5-1.5B-Instruct |
| case | 1 个长上下文 case |
| layer | M |
| stage | qkv_q, qk, pv, wo, mlp_down |
| bit | 30 |
| 注入 | 1 次/组合 |
| 目的 | 验证 kernel 映射、注入日志、decode 不崩 |

通过标准：

- profile 能定位目标 kernel；
- 注入日志出现 `Injection data`；
- fault 输出与 clean 不同或明确记录 masked；
- 无 Python/CUDA 崩溃。

### 4.2 Phase 1：12-case pilot

| 因子 | 取值 |
|---|---|
| 模型 | Qwen0.5B-Instruct / 1.5B-Instruct |
| case | 12 |
| layer | E / M / L（3） |
| stage | 9（上表） |
| bit | 28 / 29 / 30（3） |
| position | 每 cell 1 个，early/middle/late 均匀分配并记录 |

fault 数：

**2 × 12 × 3 × 9 × 3 = 1,944**

控制：

- clean：2 × 12 = 24；
- sham：2 × 12 × 3 × 9 = 648（不翻 bit，验证 hook/流程）；
- 总计约 **2,616 个 prefill + decode**。

### 4.3 Phase 2：位置与规模扩展

Phase 1 通过后：

- 每个 cell 补 early/middle/late 三个 position；
- fault 数变为 **2 × 12 × 3 × 9 × 3 × 3 = 5,832**；
- 如需统计功效，再扩到 24 case。

## 5. 输出与判定

### 5.1 每个 run 的输出

- case_id, category, context_tokens
- model, layer, stage, bit, position
- kernel_name, kernel_count, inst_id, inst_fraction
- mask, before_val, after_val, opcode, register/pc
- injected / error_not_injected / kernel_error
- clean_output, fault_output
- task_correct, literal_garble, semantic_wrong, catastrophic, partial, preserved
- runtime_error / finite checks

### 5.2 任务答案

- **task correct**：金额/工具名/endpoint/JSON 严格匹配；
- **wrong**：答案缺失、错误或 JSON 无效；
- 只对 clean 100% 的 case 统计。

### 5.3 输出形态

沿用上一版：

- literal garble：替换字符、控制字符、空输出、退化重复、异常书写系统；
- semantic wrong：答案缺失、char similarity < 0.4、clean PPL ratio > 5；
- catastrophic：literal garble 或 semantic wrong；
- partial / preserved。

### 5.4 GEMM 局部效应

可选记录：

- 注入 kernel 的 HMMA bit 数据；
- 后续层输出是否超过 047 的 L4 阈值；
- 受影响的最大层/token；
- 用于解释“局部数值错误 → 最终答案错误”的传播。

## 6. 图与报告

一张总结图：

- **(a)** stage × bit 的 catastrophic rate（两个模型）；
- **(b)** stage × layer（bit 28–30 合并）；
- **(c)** early/middle/late position 的 catastrophic rate，或 outcome composition。

报告至少包含：

- 长上下文 dataset 统计；
- clean 100% 证明；
- NVBitFI kernel 映射表；
- 注入日志核验；
- stage/bit/layer/position 结果；
- 典型乱码/错误/保持样本；
- 与 047 的对应关系和差异。

## 7. 风险

1. **Kernel 映射是最大风险**：Qwen 的 GEMM 可能被 cuBLAS/inductor 融合或拆成多个 kernel。Phase 0 必须先证明能按 layer/stage 命中。
2. **NVBitFI 长上下文开销**：每次 run 都要重新 prefill 1.5–3k token；5,832 次会非常慢。先跑 1,944 次 pilot。
3. **0.5B clean 100% 难**：长上下文检索和工具格式可能失败；clean gate 不过就只保留 1.5B。
4. **QK/PV 可能不走独立 GEMM**：必须显式 eager attention；如果仍然融合，暂不纳入 QK/PV，或改用可注入的自定义 attention。
5. **bit 28–30 可能被 masked**：同一 stage 的某些 HMMA 元素不影响 answer；需要按 stage/position 分层统计。
6. **NVBitFI 注入结果不稳定**：需要重复 clean/sham 和已知 fault 验证；不能把 crash 当作普通 SDC。

## 8. 已确认 / 待确认

已确认：

- 模型：Qwen2.5-0.5B-Instruct + 1.5B-Instruct；
- 规模：12-case pilot；
- stage：047 五个 + Wo + MLP（9 个）；
- 注入：NVBitFI，prefill，bits 28–30，decode 看回答质量。

待确认：

- QK/PV 若无法用 NVBitFI 稳定命中，是否允许暂时只做 QKV/Wo/MLP；
- 12-case pilot 的每个 cell 先做 1 个 position（1,944 runs），还是直接 3 个 position（5,832 runs）；
- clean 100% 失败时，是否只保留 1.5B。
