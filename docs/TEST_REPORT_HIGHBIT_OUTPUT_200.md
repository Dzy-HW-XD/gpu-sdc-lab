# 高位激活注错实验测试报告

测试编号：`SDC-HB-200-2026-10-21`  
测试日期：2026-10-21  
测试对象：Qwen2.5-0.5B / Qwen2.5-1.5B 高位 activation 注错  
测试状态：**PASS（目标答案 clean 100% 口径）**  
执行节点：`node20`，共享 A100

## 1. 测试目标

验证以下事项：

1. 200 条 prompt 在两个 Qwen 模型上是否达到 **clean 200/200**；
2. 高位 activation bit flip 是否被执行且每次只改变 1 个 bit；
3. sham 控制是否与 clean 完全一致；
4. 注错后输出是否出现 literal garble 或答案完全错误；
5. 是否出现 NaN/Inf、runtime error 或 OOM；
6. 是否能生成完整的逐 run 数据、汇总表和总结图。

## 2. 测试环境

| 项目 | 配置 |
|---|---|
| 节点 | node20 |
| GPU | 共享 NVIDIA A100，80 GB |
| 精度 | FP32 |
| Python 环境 | `/home/dzy/llm_sdc_env` |
| PyTorch | 2.14.1+cu126 |
| CUDA | 12.6 |
| Transformers | 5.20.2 |
| decoding | greedy，`max_new_tokens=48`，batch size 1，seed 0 |
| 显存保护 | 单进程 15% GPU memory fraction |
| 结果 | 无 OOM、无 runtime error |

## 3. 测试对象与输入

### 3.1 模型

| 模型 | Hugging Face ID | 层数 | hidden | E/M/L | attention | residual |
|---|---|---:|---:|---|---|---|
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | 24 | 896 | 0 / 12 / 23 | GQA 14Q/2KV | sequential |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | 28 | 1536 | 0 / 14 / 27 | GQA 12Q/2KV | sequential |

### 3.2 Prompt

最终 200 条 prompt，全部使用 few-shot / cloze 形式，目标答案为短答案。

| 类别 | 数量 |
|---|---:|
| factual | 53 |
| arithmetic | 40 |
| list / sequence | 28 |
| code | 28 |
| continuation | 26 |
| definition / opposite | 25 |
| **合计** | **200** |

clean 校准：

- 第一版 200 条：Qwen2.5-0.5B 191/200，Qwen2.5-1.5B 196/200；
- 替换两个模型不合格的 13 条 prompt；
- 重新 clean-only 校准后：两个模型均为 **200/200**。

### 3.3 注错输入

| 参数 | 取值 |
|---|---|
| 注入对象 | decoder block 输入的 residual-stream activation |
| 注入层 | E/M/L 三层 |
| 注入 token | 最后一个 prompt token |
| 注入元素 | 该位置绝对值最大的单个元素 |
| bit | 28, 29, 30, 31 |
| 注错方式 | int32 视图 XOR 一个 bit |
| 校验 | 每次恰好改变 1 bit |

### 3.4 测试矩阵

| 条件 | 数量 |
|---|---:|
| fault | 2 × 200 × 3 × 4 = 4,800 |
| sham | 2 × 200 × 3 = 1,200 |
| clean | 2 × 200 = 400 |
| **总计** | **6,400 runs** |

## 4. 测试项与验收结果

| ID | 测试项 | 验收标准 | 实际结果 | 结论 |
|---|---|---|---|---|
| T00 | clean 校准 | 两个模型均为 200/200 | Qwen0.5B 200/200；Qwen1.5B 200/200 | PASS |
| T01 | clean 可重复性 | clean-only 与 full-run 的 clean 输出一致 | 400/400 一致 | PASS |
| T02 | sham 控制 | sham 与 clean 完全一致 | 1,200/1,200 char similarity = 1.000000 | PASS |
| T03 | bit 完整性 | 每个 fault run 恰好 1 bit 改变 | 4,800/4,800 | PASS |
| T04 | 数值有限性 | 无 NaN/Inf | 4,800/4,800 fault value finite；0 nonfinite logits | PASS |
| T05 | 运行稳定性 | 无 runtime error / OOM | 0 error；6,400 runs 全部完成 | PASS |
| T06 | 注错执行 | 4,800 个 fault 全部生成 | 4,800/4,800 | PASS |
| T07 | 输出判定 | 生成 literal garble / wrong / partial / preserved / catastrophic 标签 | 全部生成 | PASS |
| T08 | 汇总与图 | 生成 CSV、报告和图 | 已生成 | PASS |

## 5. 功能测试结果

### 5.1 总体结果

统计分母为两个模型各自 200 条 prompt、3 层、4 bit，共 2,400 个有效 fault runs。

| 模型 | clean | 有效 fault runs | catastrophic (95% CI) | literal garble | wrong | partial | preserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 200/200 | 2,400 | 38.0% (36.1–40.0) | 4.5% | 37.9% | 10.0% | 52.0% |
| Qwen2.5-1.5B | 200/200 | 2,400 | 15.8% (14.4–17.3) | 1.3% | 15.6% | 14.9% | 69.3% |

合计：

- catastrophic：1,292/4,800 = **26.9%**
- literal garble：141/4,800 = **2.9%**

### 5.2 按 bit

| 模型 | bit 28 | bit 29 | bit 30 | bit 31 |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 63.8% | 43.8% | 34.2% | 10.3% |
| Qwen2.5-1.5B | 21.2% | 23.7% | 16.2% | 2.2% |

### 5.3 按层

| 模型 | E | M | L |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 15.9% | 42.0% | 56.3% |
| Qwen2.5-1.5B | 11.4% | 18.4% | 17.6% |

### 5.4 关键组合

| 组合 | catastrophic |
|---|---:|
| Qwen2.5-0.5B, L, bit 28 | 97.0% |
| Qwen2.5-0.5B, M, bit 28 | 94.0% |
| Qwen2.5-0.5B, M, bit 29 | 68.0% |
| Qwen2.5-0.5B, L, bit 29 | 63.0% |
| Qwen2.5-0.5B, E, bit 30 | 61.0% |
| Qwen2.5-1.5B, E, bit 30 | 42.0% |
| Qwen2.5-1.5B, M, bit 29 | 39.0% |
| Qwen2.5-1.5B, L, bit 28 | 32.5% |

### 5.5 结局构成（bit 28–29）

| 模型 | garbled | wrong | partial | preserved | n |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 7.8% | 46.0% | 10.0% | 36.2% | 1,200 |
| Qwen2.5-1.5B | 2.0% | 20.4% | 21.0% | 56.6% | 1,200 |

## 6. 缺陷与观察记录

| ID | 描述 | 影响 | 处理 | 状态 |
|---|---|---|---|---|
| D01 | 第一版 200 条 prompt 的 clean 不是 100%：Qwen0.5B 191/200，Qwen1.5B 196/200 | 不能直接进入最终统计 | 替换 13 条不合格 prompt，重新 clean-only 校准到 200/200 | CLOSED |
| D02 | Qwen base 在答对后继续生成无关内容 | 不影响目标答案判定，但整段文本不一定是“整段完全正确” | 统计口径限定为目标答案严格匹配；整段 judge 留作后续 | OPEN（口径限制） |
| D03 | 个别高位注错产生语言漂移/重复退化 | 输出 literal garble | 已由 literal garble 标签覆盖 | OBSERVED |
| D04 | 没有出现 NaN/Inf | 语义错误不会触发 NaN/Inf 监控 | 需要输出级检测 | OBSERVED |

## 7. 测试结论

**PASS（目标答案 clean 100% 口径）：**

- 两个 Qwen 模型均达到 clean **200/200**；
- 4,800 个 fault run 全部完成；
- 1,200 个 sham 与 clean 完全一致；
- 4,800 个 fault 均恰好改变 1 bit；
- 没有 NaN/Inf、runtime error 或 OOM；
- 注错导致 Qwen0.5B 38.0%、Qwen1.5B 15.8% catastrophic；
- 主要失败是答案完全错误，literal garble 仅占 2.9%。

**未覆盖：**

- 整段自由续写的 100% 正确性；
- 多元素、多 token、连续 bit；
- 权重、KV-cache、attention head 注入；
- instruction-tuned 模型；
- 人工/LLM judge 的整段语义评分。

## 8. 产物

| 产物 | 路径 |
|---|---|
| 测试报告 | `docs/TEST_REPORT_HIGHBIT_OUTPUT_200.md` |
| 实验报告 | `docs/REPORT_HIGHBIT_OUTPUT_200.md` |
| 最终 prompt | `data/highbit_prompts_200_calibrated.json` |
| 逐 run 数据 | `results/llm_highbit_200_calibrated/runs_all.csv` |
| 汇总表 | `results/llm_highbit_200_calibrated/summary_by_*.csv` |
| 结局构成 | `results/llm_highbit_200_calibrated/outcome_by_model_2829.csv` |
| 示例 | `results/llm_highbit_200_calibrated/examples.md` |
| 图 | `docs/figures/highbit_output_summary_200.pdf` / `.png` |
| 图数据副本 | `results/llm_highbit_200_calibrated/figures/` |

## 9. 签署

测试执行：自动实验脚本  
测试日期：2026-10-21  
结论：**PASS（clean 200/200，注错执行与数据完整性通过）**
