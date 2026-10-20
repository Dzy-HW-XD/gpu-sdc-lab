# 高位激活注错与 LLM 输出乱码/错误

日期：2026-10-20  
范围：prefill 后的单元素 FP32 激活高位翻转，观察 decode 输出  
模型：Pythia-160M / 410M / 1B，Qwen2.5-0.5B / 1.5B

## 结论摘要

1. **高位 bit 不是简单的“越高越坏”。** 在 5 个模型上，bit 28–29 在 middle/late 层最容易造成答案错误；bit 30 在 early 层很强；bit 30/31 在 late 层通常几乎没有影响。唯一明显例外是 Pythia-160M 的 early bit 31。
2. **失败以“答案完全错误”为主，不是 NaN/Inf。** 1,440 个 fault run 中，1,140 个进入 clean-correct 统计；其中 358 个为 catastrophic（31.4%）。literal garble 只有 30/1,440（2.1%），但确实存在，包括越南语重复、中文片段、替换字符和标点/重复退化。
3. **模型间没有单调的参数规模关系。** 总体 catastrophic rate：Pythia-160M 41.7%、Qwen2.5-0.5B 36.6%、Pythia-1B 30.8%、Pythia-410M 27.9%、Qwen2.5-1.5B 24.7%。
4. **单元素高位翻转不会触发 NaN/Inf 检测。** 所有 1,440 个 fault run 的 logits 都是 finite，注入值也都是 finite，没有 runtime error。NaN/Inf 监控会完全漏掉这些语义错误。
5. **Sham 与重复性控制干净。** 360 个 sham run 与 clean 输出逐字符一致；Pythia-160M 的 24 个 clean prompt 重复生成无差异；所有 fault run 都只改变了 1 个 bit。

![高位激活注错总结图](figures/highbit_output_summary.png)

矢量版：[`figures/highbit_output_summary.pdf`](figures/highbit_output_summary.pdf)

图注：**(a)** 各模型在 bit 28–31 上的 catastrophic rate，按 E/M/L 三层聚合；**(b)** 各模型在 E/M/L 注入深度上的 catastrophic rate，按 bit 28–31 聚合；**(c)** bit 28–29 合并后的输出结局构成（garbled / wrong / partial / preserved）。行标签括号内为 clean-correct prompt 数；**(c)** 右侧 `n` 为对应有效 run 数。所有比率都基于 clean-correct prompt；garbled 在 **(c)** 中优先归类，因此与 wrong 不重复计数。

## 1. 实验参数

| 项目 | 设置 |
|---|---|
| 模型 | `EleutherAI/pythia-160m`, `pythia-410m`, `pythia-1b`, `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B` |
| 精度 | FP32 |
| 注入对象 | decoder block 输入的 residual-stream activation |
| 注入位置 | 最后一个 prompt token |
| 注入元素 | 该位置绝对值最大的单个元素 |
| 注入层 | 每个模型 3 层：E = 0，M = N/2，L = N−1 |
| 注入 bit | 28, 29, 30, 31 |
| Prompt | 24 条固定 prompt，6 类 × 4 条 |
| 解码 | greedy，`max_new_tokens = 48`，batch size 1 |
| 控制 | 120 个 clean 生成；360 个 sham（hook 安装但不翻转）；Pythia-160M 额外 24 个 clean 重复 |
| 总 run | 1,944 行：1,440 fault + 360 sham + 120 clean + 24 clean-repeat |
| GPU | 共享 A100；单进程显存上限 15%；最大模型占用约 5.9 GiB，无 OOM |

### Prompt 集

| 类别 | 条数 | 形式 | 例子 |
|---|---:|---|---|
| factual | 4 | few-shot / cloze | “The capital of France is Paris. The capital of Japan is” |
| arithmetic | 4 | few-shot / cloze | “1 + 1 = 2. 2 + 2 = 4. 2 + 3 =” |
| list | 4 | 续写 / 序列 | “The first three letters are A, B, and” |
| code | 4 | 代码续写 / 关键字 | “def add(a, b): ... def mul(a, b): return” |
| definition | 4 | 反义 / 常识 | “Hot is the opposite of cold. Up is the opposite of” |
| continuation | 4 | 固定文本续写 | “The quick brown fox jumps over the lazy” |

Prompt 均为短答案设计，答案位置明确。Pythia 是 base model，不是 instruction-tuned，因此 clean 正确率不高：

| 模型 | clean-correct prompts |
|---|---:|
| Pythia-160M | 11 / 24 |
| Pythia-410M | 17 / 24 |
| Pythia-1B | 20 / 24 |
| Qwen2.5-0.5B | 23 / 24 |
| Qwen2.5-1.5B | 24 / 24 |

所有语义错误率都以 clean-correct prompt 为分母；clean 已经答错的 prompt 不参与归因。literal garble 仍对全部 fault run 统计。

## 2. 判定指标

- **literal garble**：出现 U+FFFD、控制字符、空输出、异常书写系统，或相对 clean 输出出现明显退化重复。
- **wrong / semantic wrong**：答案缺失；或字符相似度 `< 0.4`；或 clean continuation 的 PPL ratio `> 5`。
  - 短答案类别（arithmetic / list / code）要求答案在前 6 个 token 内；
  - 其余类别要求答案出现在第一行。
- **catastrophic**：literal garble 或 wrong。
- **partial**：答案仍出现，但位置/格式/上下文发生明显改变，或字符相似度不足。
- **preserved**：答案出现在期望位置，且字符相似度 `≥ 0.8`。
- **clean-correct**：clean 输出中答案位于前 4 个 token 内。

这是**规则化、可复现**的判定，不是 LLM judge，也不是人工语义评分。它刻意把“答案完全消失”与“答案还在但格式变了”分开。

## 3. 总体结果

| 模型 | clean-correct | 有效 fault runs | catastrophic (95% CI) | literal garble（全部 runs） | wrong | partial | preserved | mean char sim | median PPL ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pythia-160M | 11/24 | 132 | 41.7% (33.6–50.2) | 1.7% | 41.7% | 9.8% | 48.5% | 0.692 | 1.017 |
| Pythia-410M | 17/24 | 204 | 27.9% (22.2–34.5) | 1.7% | 27.9% | 12.3% | 59.8% | 0.751 | 1.011 |
| Pythia-1B | 20/24 | 240 | 30.8% (25.3–36.9) | 1.0% | 30.8% | 15.0% | 54.2% | 0.767 | 1.010 |
| Qwen2.5-0.5B | 23/24 | 276 | 36.6% (31.1–42.4) | 4.2% | 36.2% | 10.5% | 52.9% | 0.729 | 1.062 |
| Qwen2.5-1.5B | 24/24 | 288 | 24.7% (20.0–29.9) | 1.7% | 24.7% | 8.7% | 66.7% | 0.796 | 1.006 |

- 合计：358/1,140 = **31.4% catastrophic**（95% CI 28.8–34.2%）；30/1,440 = **2.1% literal garble**。
- `wrong` 与 `catastrophic` 几乎重合，说明主要失败是答案丢失，而不是编码层乱码。
- `median PPL ratio` 接近 1，说明“模型仍认为 clean continuation 合理”并不保证 greedy 输出仍正确；仅靠 PPL 监控会漏报。
- Qwen2.5-0.5B 的 literal garble 最高（12/288 = 4.2%），表现为语言漂移和退化重复。

## 4. 按 bit 聚合

单元格为 catastrophic rate；每个模型每个 bit 聚合 E/M/L 三层。

| 模型 | bit 28 | bit 29 | bit 30 | bit 31 |
|---|---:|---:|---:|---:|
| Pythia-160M | 63.6% | 48.5% | 30.3% | 24.2% |
| Pythia-410M | 49.0% | 37.3% | 15.7% | 9.8% |
| Pythia-1B | 46.7% | 43.3% | 25.0% | 8.3% |
| Qwen2.5-0.5B | 56.5% | 42.0% | 39.1% | 8.7% |
| Qwen2.5-1.5B | 37.5% | 40.3% | 19.4% | 1.4% |

聚合后 bit 28/29 通常比 bit 30/31 更危险，但这不是逐层规律：bit 30 在 early 层很强，bit 30/31 在 late 层常常完全无效。原因来自 FP32 exponent/sign 位的非线性影响：翻转同一个 bit 在不同数值上可能放大、缩小、变号，甚至几乎不改变下游输出。

## 5. 按深度聚合

单元格为 catastrophic rate；每个模型每个深度聚合 bit 28–31。

| 模型 | E | M | L |
|---|---:|---:|---:|
| Pythia-160M | 45.5% | 43.2% | 36.4% |
| Pythia-410M | 13.2% | 41.2% | 29.4% |
| Pythia-1B | 18.8% | 43.8% | 30.0% |
| Qwen2.5-0.5B | 15.2% | 41.3% | 53.3% |
| Qwen2.5-1.5B | 17.7% | 29.2% | 27.1% |

- 对 Pythia-410M、Pythia-1B，**middle 层最敏感**；Qwen2.5-0.5B 的 **late 层最敏感**（53.3%），middle 层次之（41.3%）。
- Pythia-160M 的 early 层被 bit 30/31 拉高，总体与 middle 接近。
- Qwen2.5-1.5B 的 early 层最稳，middle/late 也较低。

## 6. 关键 layer × bit 组合

| 组合 | catastrophic | 说明 |
|---|---:|---|
| Pythia-160M, M, bit 28 | 10/11 = 90.9% | 首个答案被替换为无关词（如 “bulletproof”） |
| Pythia-410M, M, bit 28 | 15/17 = 88.2% | 输出出现 “Eastern Europe” 等无关前缀 |
| Pythia-1B, M, bit 28 | 17/20 = 85.0% | 首个 token 被 “un...” 类词替换 |
| Qwen2.5-0.5B, L, bit 28 | 21/23 = 91.3% | 最高单格；常出现语言漂移/重复 |
| Qwen2.5-1.5B, E, bit 30 | 14/24 = 58.3% | early exponent 翻转，输出变成 `!`/代码片段/多项选择 |
| Pythia-160M, E, bit 30 | 9/11 = 81.8% | early 层最强的单格之一 |
| Pythia-160M, E, bit 31 | 7/11 = 63.6% | sign bit 翻转在 early 层也可致命 |
| Qwen2.5-0.5B, L, bit 30 | 12/23 = 52.2% | late 层 bit 30 的明显例外 |
| Pythia-160M, L, bit 30/31 | 0/11 | late 层翻转几乎完全恢复 |
| Qwen2.5-1.5B, M, bit 31 | 0/24 | 完全无影响 |

## 7. 典型输出

### 7.1 乱码/退化

**Qwen2.5-1.5B，definition 类，M 层，bit 28，char_sim 0.090**

- clean：`down. Left is the opposite of right. ...`
- fault：`． A. down B. up ... 答案: A ... 下列关于细胞结构和功能的叙述，正确的是____ ...`

输出从英文常识题变成中文生物题，并含全角标点和中文片段；判为 literal garble + wrong。

**Qwen2.5-0.5B，arithmetic 类，L 层，bit 28，char_sim 0.160**

- clean：`11. 6 + 7 = 13. 7 + 8 = 15. ...`
- fault：`thói quen, thói quen, thói quen, ...`

越南语词重复，属于退化重复和语言漂移；判为 literal garble。

**Pythia-1B，continuation 类，L 层，bit 28，char_sim 0.086**

- clean：`blue, and the sun is yellow. ...`
- fault：`- - - - - - - - ...`

重复标点退化；判为 literal garble。

### 7.2 完全错误但文本仍流畅

**Pythia-160M，factual 类，M 层，bit 28，char_sim 0.275**

- clean：`Tokyo. The city of Tokyo is ...`
- fault：`bulletproof. The city of Tokyo is ...`

文本连贯，但首答案从 Tokyo 变成 bulletproof；判为 wrong。

**Pythia-1B，arithmetic 类，E 层，bit 28，char_sim 0.968**

- clean：`5. 2 + 4 = 7. ...`
- fault：`6. 2 + 4 = 8. ...`

只有首数字变化，字符相似度很高，但答案错误；说明只看文本相似度会漏报。

**Qwen2.5-1.5B，continuation 类，L 层，bit 28**

- clean：`dog. The quick brown fox jumps over the lazy dog.`
- fault：`120000000000000000000000000...`

语义完全丢失，但字符层面不是“乱码”；判为 wrong。

### 7.3 保持

- Pythia-1B，L 层，bit 30，arithmetic：与 clean 完全一致。
- Qwen2.5-1.5B，E 层，bit 28，arithmetic：与 clean 完全一致。
- Pythia-410M，E 层，bit 28，code：与 clean 完全一致。

## 8. 控制与有效性

| 检查 | 结果 |
|---|---|
| sham hook vs clean | 360/360 字符相似度 1.000000 |
| Pythia-160M clean 重复 | 24/24 完全一致 |
| fault run bit 改变数 | 1,440/1,440 恰好 1 bit |
| fault value finite | 1,440/1,440 finite |
| 生成 logits nonfinite | 0 |
| runtime error | 0 |
| clean 输出本身退化 | Pythia-160M 3/24；410M 1/24；1B 1/24；Qwen0.5B 0/24；Qwen1.5B 1/24 |

Sham 和 clean 重复说明 hook 本身没有副作用；fault 值 finite 说明这些不是 NaN/Inf 传播，而是数值方向/幅度改变导致的语义错误。

## 9. 解释

1. **bit 28/29 更危险**：它们改变 FP32 exponent 的高位，容易把单个激活放大到 `1e9–1e10` 量级，足以主导后续 residual stream 和 attention/MLP 计算。
2. **bit 30 在 early 层危险、在 late 层弱**：early 层的扰动会经过全部后续层放大；late 层的同类扰动只影响最后几层，常被 residual 吸收。
3. **bit 31 是符号位**：在 early 层可以翻转大激活的符号，造成强扰动；在 late 层通常不足以改变最终 argmax。
4. **Qwen2.5-0.5B 对 late 层 bit 28 特别敏感**：L 层 91.3% catastrophic，且出现语言漂移；可能与 GQA/sequential residual 和较小 hidden size 有关，但本实验不能分离这些因素。
5. **参数更多不一定更稳**：Pythia-1B/410M 低于 Pythia-160M，Qwen2.5-1.5B 低于 Qwen2.5-0.5B；模型结构、训练数据和激活分布共同决定敏感性。
6. **PPL 不是好检测器**：median PPL ratio 约 1.0，说明 clean continuation 在 fault model 下仍可能高概率；但 greedy 输出已经错误。需要输出级 answer/consistency 检查。

## 10. 局限

- 每个条件只翻转 **一个元素、一个 bit、最后一个 prompt token**；没有多元素、多 token、连续 bit 翻转。
- 只做 activation 注入，没有权重、KV-cache 或 attention head 注入。
- 每个模型只有 24 条 prompt；Pythia-160M 只有 11 条 clean-correct，模型间比较的置信区间较宽。
- 规则化答案匹配不能完全替代人工/LLM judge；格式改变会被归为 partial，但边界仍有主观性。
- 仅 greedy decoding；没有采样、beam search 或多轮对话。
- 共享 A100 上运行，未做严格的吞吐/延迟测量。

## 11. 产物

- 原始与后处理数据：`results/llm_highbit_output/runs_all.csv`
- 汇总表：`summary_by_model.csv`、`summary_by_model_bit.csv`、`summary_by_model_layer.csv`、`summary_by_model_layer_bit.csv`、`summary_by_category.csv`
- 结局构成：`outcome_by_model_2829.csv`
- 示例：`examples.md`
- 图：`results/llm_highbit_output/figures/highbit_output_summary.pdf` / `.png`
- 文档图副本：`docs/figures/highbit_output_summary.pdf` / `.png`
- 实验计划：`docs/EXPERIMENT_PLAN_HIGHBIT_OUTPUT.md`
- 控制元数据：`analysis_meta.json`

## 12. 下一步建议

1. 对 20–30 个 catastrophic/partial 样本做人工复核，并补一个 LLM judge，校准“完全错误”的边界。
2. 扩展到 top-k 元素、多个 token、连续 bit 翻转，观察是否存在阈值/饱和。
3. 对权重矩阵重复同一协议，比较“参数错误”和“状态错误”的失败形态。
4. 把 answer consistency、字符相似度、PPL ratio、语言/重复检测组合成运行时检测器，并与 NaN/Inf 监控对比漏报率。
5. 在 prompt 集上增加 instruction-tuned 模型，验证 base model 的结论是否迁移。
