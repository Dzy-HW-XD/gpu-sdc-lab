# Attention SDC 注入热图重绘与传播范围分析

- **日期**：2026-10-20
- **实验**：`047_attn_5gemm_full`（复用已有 3150 次注入，本次**没有新增注入**）
- **计算环境**：node20（4×A100-80GB，sm_80）负责从原始 Y 重算数据；本地 Windows 负责绘图
- **模型规模**：S=128，D=2048，H=32，dh=64，BF16
- **判据**：L4，`|ΔY| > 1e-4 + 1e-3·|Y|`
- **目的**：重绘 attention 注入热图，并用一张图总结“注入位置 → 输出影响范围（单 token vs 多 token）”

---

## 1. 实验设置回顾

对 attention forward 的五个 GEMM 阶段做单故障注入：

| 阶段 | 含义 |
|---|---|
| `qkv_q` | Q 投影，`Q = XWq` |
| `qkv_k` | K 投影，`K = XWk` |
| `qkv_v` | V 投影，`V = XWv` |
| `qk` | score GEMM，`QK^T` |
| `pv` | `P·V` |

- bits：24–30
- 注入位置：early / middle / late（kernel 动态指令流中的早/中/晚，不是 token 位置）
- 每个 `(stage, bit, position)` 30 次注入，每个 stage 共 7×3×30 = 630 次
- 总计 **3150 次注入**，每次一个故障
- `qk` / `pv` 每次从 0–31 中随机选一个 head；`qkv_*` 是投影级故障，影响所有 head

结果：**1507 / 3150 SDC（47.8%）**，其余为 MASKED，无 CRASH。

---

## 2. 旧版热图的问题

旧版是 17×9.5 in、140 dpi 的 dashboard，包含误差图、support 图、累计覆盖率曲线和统计文本面板。主要问题：

### 2.1 版式与排版

- 画布不是论文尺寸；缩到 7 in 栏宽时字号只剩约 4–5 pt。
- 右侧统计面板占用整列但只用上半部分，右下大片留白。
- 图内标题、轴标题、统计面板重复同一组 stage/bit/SDC 信息。
- 没有 `(a)/(b)` 面板标号；默认无衬线字体与论文正文不匹配；只输出 PNG，不是矢量图。

### 2.2 颜色与编码

- support 是 1/30 的离散值，线性 0–1 色标把 1/30 画得几乎和 0 一样，局部足迹几乎不可见。
- “从未命中”和 `NaN` 共用同一种灰色。
- 固定 `[-3, 6]` 的误差色标把 75% 的数值（1.5–2.1）压在很窄的色带里。
- 底部 “cumulative impact ratio” 在 feature coverage 100% 时就是一条对角线，实际只是累计覆盖率，叫 “impact” 属于过度解读。
- 误差面板对所有非零变化取平均，包含低于 L4 的浮点噪声，导致 `qkv_k` 出现整片均匀红色。

### 2.3 标注与数据一致性

- `qk/pv` 的 30 次注入使用**随机 head**，旧图标题却写成 `h0/h13`，看起来像 30 次都用同一个 head。
- bit 30 的 NaN/Inf 被分类器计为 SDC，但旧热图脚本跳过了非有限值。例如 `qkv_q early bit30` 的 footer 写着 **SDC 13/30**，图却是空的——13 次运行各有 2048 个 NaN。这是必须修的数据显示问题。

---

## 3. 学术制图规范调研

参考了两套公开的学术绘图 skill：

| skill | 来源 | 采用的做法 |
|---|---|---|
| `paper-figure` | wanshuiyin/Auto-claude-code-research-in-sleep | 图内不放 title；Times 衬线字体；300 dpi + PDF 矢量；色盲友好；单栏/双栏宽度；render-then-verify |
| `scientific-visualization` | K-Dense-AI/scientific-agent-skills（MIT，v1.4） | 真实编码；区分缺失/零/非有限值；可比面板固定色标；导出与 provenance 检查 |

没有把第三方 skill 装进仓库，绘图代码保持自包含。

---

## 4. 数据重算（v2）

在 node20 上用原始 `Y_faulty` / `Y_golden` 重算，纯 stdlib，未占用 GPU：

- `support` / `count`：L4 命中次数（0–30）
- `nfcount`：NaN/Inf 运行次数
- `meanlog`：**只在 L4 命中的运行上**取平均，不再混入亚阈值噪声
- `maxlog`：L4 命中运行中的最大 `log10|ΔY|`
- NaN/Inf 计入 affected，误差哨兵值设为 40

全 campaign 统计：

- **55 / 3150 次运行**在 **11 个 cell** 中出现非有限值
- 全部集中在 bit 30

| stage | early | middle | late | 合计 |
|---|---:|---:|---:|---:|
| QKV-Q | 13 | 4 | 5 | 22 |
| QKV-K | 9 | 5 | 7 | 21 |
| QKV-V | 6 | 2 | 2 | 10 |
| QK | 0 | 0 | 0 | 0 |
| PV | 0 | 1 | 1 | 2 |
| **合计** | **28** | **12** | **15** | **55** |

> 这解释了旧图“footer 有 SDC、图却是空”的矛盾。v2 中这些位置会显示为黑色，并标注 `NF k/30`。

---

## 5. 新图设计

### 5.1 单 cell 图（105 张）

7.2 in 双栏宽、两面板：

- **`(a)` support**：0–30 次命中，`Blues` + 平方根拉伸，固定 0–30；1/30 也可见。
- **`(b)` 误差**：`log10|ΔY|`，`YlOrRd`，固定 `[0, 3]`；`>10^3` 截断。
  - 灰色 = 从未 L4 命中
  - 黑色 = 至少一次 NaN/Inf
- Times New Roman + STIX，8 pt 正文 / 7 pt 刻度，`(a)/(b)` 面板标号，无图内标题。
- 审阅版带 footer（stage/position/bit/n/SDC/NF/图例）；论文版无 footer。
- PDF 矢量文字 + 600 dpi PNG。

### 5.2 空间总览（5×3）

`overview_support_stage_position`：

- 行：5 个 stage；列：early / middle / late
- 每个 panel 是 128×2048 的 support 图
- 颜色：`max over bits 24–30` 的命中次数（0–30）
- 用途：看“影响落在哪些 token 行”，是空间 companion

### 5.3 单图总结（推荐主图）

`token_count_distribution`：

- 5×7 heatmap
- 行：5 个 stage
- 列：一次 SDC 运行影响的 **token 行数**分箱：`1`、`2–3`、`4–7`、`8–15`、`16–31`、`32–63`、`64–128`
- 颜色/数字：该 stage 的 SDC 运行中落入该区间的比例，每行归一化到 100%
- 左侧：`SDC 次数 / 630` 和中位数
- 这张图直接回答“单 token 还是多 token”

---

## 6. 关键结果

### 6.1 SDC 率

| stage | SDC / 630 | SDC 率 |
|---|---:|---:|
| QKV-Q | 343 | 54.4% |
| QKV-K | 399 | 63.3% |
| QKV-V | 346 | 54.9% |
| QK | 92 | 14.6% |
| PV | 327 | 51.9% |
| **总计** | **1507 / 3150** | **47.8%** |

位置：early 43.2%，middle 48.3%，late 52.0%。  
Bit：24 28.0%，25 39.3%，26 51.8%，27 58.2%，28 54.4%，29 57.8%，30 45.3%。

### 6.2 传播宽度（SDC 运行内，影响的 token 行数）

| stage | SDC/630 | median | 1 | 2–3 | 4–7 | 8–15 | 16–31 | 32–63 | 64–128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QKV-Q | 343 | 1 | **100%** | 0 | 0 | 0 | 0 | 0 | 0 |
| QKV-K | 399 | 20 | 17% | 8.8% | 7.3% | 12% | 20% | **31%** | 4.0% |
| QKV-V | 346 | 2 | **45%** | **34%** | 18% | 0.6% | 0 | 0 | 2.9% |
| QK | 92 | 1 | **100%** | 0 | 0 | 0 | 0 | 0 | 0 |
| PV | 327 | 1 | **100%** | 0 | 0 | 0 | 0 | 0 | 0 |

> 因四舍五入，行和可能略偏离 100%。

### 6.3 解读

- **QKV-Q（`Q=XWq`）**：故障只污染被注入 query 的 Q 行，最终只影响该 query 的输出行 → **严格单 token**。
- **QK（`QK^T`）**：故障只影响该 query 的 score 行，softmax 后只改变该 query 的输出 → **严格单 token**。
- **PV**：故障只影响该 query 的 `P·V` 输出行 → **严格单 token**。
- **QKV-K**：K 行被污染后，所有 `i >= j` 的 query 都通过 causal attention 受影响 → **宽范围多 token**（median 20，31% 在 32–63）。
- **QKV-V**：V 行被污染后，所有关注该 key 的 query 都可能受影响，但只有少数超过 L4 → **中间态**（median 2，97% 在 1–7）。
- V 的 `64–128` 那一档（2.9%）**正好是 bit 30 的 10 次 NaN/Inf 传播**，不是有限误差的宽尾。

### 6.4 一个容易误读的点

现在这张 5×3 空间总览按**绝对 token 索引**聚合。Q/QK/PV 每次只影响 1 个 token，但 30 次注入落在不同 token 上，叠加后会显示成很多条横线，看起来像“多 token”。

因此：

- 要表达“单 token vs 多 token”，用 **token_count_distribution**；
- 要表达“影响落在哪些 token 行”，用 **overview**；
- 两者不要互相替代。

---

## 7. 产出文件

| 路径 | 内容 |
|---|---|
| `results/047_attn_5gemm_full/final_heatmaps_v2/summary/token_count_distribution.pdf` | **推荐主图**：单 token vs 多 token |
| `results/047_attn_5gemm_full/final_heatmaps_v2/summary/token_count_distribution.png` | 主图 PNG 预览 |
| `results/047_attn_5gemm_full/final_heatmaps_v2/overview/overview_support_stage_position.pdf` | 5×3 空间总览 |
| `results/047_attn_5gemm_full/final_heatmaps_v2/` | 105 个 cell 的审阅版 PDF + 600 dpi PNG |
| `results/047_attn_5gemm_full/final_heatmaps_v2_paper/` | 105 个 cell 的论文版 PDF（无 footer） |
| `results/047_attn_5gemm_full/final_heatmaps_data_v2_nf/` | v2 数据（含 `nfcount`），约 368 MB |
| `docs/HEATMAP_REDESIGN.md` | 完整分析、skill 对比、caption 模板、复现命令 |

> 语雀中插入图片时，直接上传对应的 `.png` 即可；本地路径不会自动变成语雀图片。

---

## 8. 复现命令

```bash
# node20：从原始 Y 重算 v2 数据（stdlib，无需 numpy）
python3 scripts/attention_final_heatmaps_data_v2.py \
  --exp results/047_attn_5gemm_full \
  --out results/047_attn_5gemm_full/final_heatmaps_data_v2_nf

# 本地：105 个单 cell 图（审阅版）
python scripts/attention_final_heatmaps_plot_v2.py \
  --data results/047_attn_5gemm_full/final_heatmaps_data_v2_nf \
  --outdir results/047_attn_5gemm_full/final_heatmaps_v2 \
  --formats pdf,png --dpi 600

# 本地：105 个论文版 PDF（无 footer）
python scripts/attention_final_heatmaps_plot_v2.py \
  --data results/047_attn_5gemm_full/final_heatmaps_data_v2_nf \
  --outdir results/047_attn_5gemm_full/final_heatmaps_v2_paper \
  --formats pdf --no-annotate

# 本地：5×3 空间总览
python scripts/attention_overview_heatmaps_v2.py \
  --data results/047_attn_5gemm_full/final_heatmaps_data_v2_nf \
  --outdir results/047_attn_5gemm_full/final_heatmaps_v2/overview \
  --formats pdf,png

# 本地：单图总结（单 token vs 多 token）
python scripts/attention_token_count_heatmap.py \
  --footprint results/047_attn_5gemm_full/final_heatmaps_data_v2_nf/footprint.csv \
  --outdir results/047_attn_5gemm_full/final_heatmaps_v2/summary \
  --formats pdf,png
```

---

## 9. 结论

1. **Q/QK/PV 是严格单 token**：100% 的 SDC 运行只影响 1 个输出 token 行。
2. **K 是宽范围多 token**：median 20，31% 的 SDC 运行影响 32–63 个 token 行。
3. **V 是中间态**：median 2，97% 的 SDC 运行影响 1–7 个 token 行；`64–128` 的尾部来自 bit 30 的 NaN/Inf，不是有限误差。
4. 表达“单 vs 多”用 `token_count_distribution.pdf`；表达“落在哪些 token”用 `overview_support_stage_position.pdf`。
5. 按绝对 token 索引聚合的空间图，会让单 token 效应看起来像多 token；这是本次重绘最重要的方法学修正。

## 10. 后续可做

- 增加 **relative-lag heatmap**（以每次运行最早受影响的 token 为原点，看影响随 `i-j0` 的衰减），作为空间 companion。
- 按目标期刊核对单栏/双栏尺寸与字号（当前是 7.2 in 双栏）。
- 补 L0–L5 tolerance sweep，回答“传播宽度随容差如何变化”。
- 将本次改动 commit（目前仅在工作区，未提交）。
