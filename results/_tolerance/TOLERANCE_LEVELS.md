# 容忍度档位（L1–L5）与敏感性热力图

## 统一说明

- **测试目的**：用**同一批故障样本**（配对设计），只改变“判定容忍度”，测量
  有多少“静默错误”其实落在可接受范围内（会被判为 MASKED），从而量化
  **SDC 严重度的敏感区**与**良性误差比例**。
- **判定规则**：某元素 `|fault − golden| > abs_tol + rel_tol·|golden|` 才算损坏；
  一档越松 → 越多小误差被“容忍”→ SDC 率下降。
- **主要变量（自变量）**：容忍度档位 `(abs_tol, rel_tol)`（**同步放大**：每步 abs、rel 同时 ×10）。
- **固定变量**：输入分布、bit 位、指令、规模、注入时机、寄存器（同一批注入，不重跑 GPU）。
- **因变量**：总体 SDC 率、被判 MASKED 的数量、以及逐格 `input × bit` 的 SDC 率。
- **数据来源**：从保留的 raw 输出离线重算（`scripts/tolerance_sweep.py`），
  与运行期严格比较 `(0,0)` 完全解耦。

## 档位表（起点 L1=(1e-7,1e-6)，每步同时 ×10）

| 档 | abs_tol | rel_tol | 相对 L1 | 测试目的 | 主要变量 | 总体 SDC 率 | 判为 MASKED |
|---|---|---|---|---|---|---|---|
| **L1** | 1e-7 | 1e-6 | ×1 | **起点/最严实验口径**：仅过滤极小舍入噪声 | abs=1e-7, rel=1e-6 | 0.727 | 2465 |
| **L2** | 1e-6 | 1e-5 | ×10 | 放大 10×，看有多少“低幅”SDC 被容忍 | abs=1e-6, rel=1e-5 | 0.665 | 3110 |
| **L3** | 1e-5 | 1e-4 | ×100 | 放大 100×，进入敏感区，观察下降斜率 | abs=1e-5, rel=1e-4 | 0.612 | 3655 |
| **L4** | 1e-4 | 1e-3 | ×1000 | 放大 1000×，接近“相对误差千分之一” | abs=1e-4, rel=1e-3 | 0.552 | 4272 |
| **L5** | 1e-3 | 1e-2 | ×10000 | **最松**：近似“相对误差 1% 可接受” | abs=1e-3, rel=1e-2 | 0.499 | 4822 |

（总计 10331 有效样本；CRASH 355 在所有档位保持不变。）

## 图像

每个档位一张 `input × bit` 的 SDC 率热力图（在同一批样本上重算）：

```
results/_tolerance/figs/
  heatmap_015_input_bit_gemm_L1_1e-7_sdc_rate.svg
  heatmap_015_input_bit_gemm_L2_1e-6_sdc_rate.svg
  heatmap_015_input_bit_gemm_L3_1e-5_sdc_rate.svg
  heatmap_015_input_bit_gemm_L4_1e-4_sdc_rate.svg
  heatmap_015_input_bit_gemm_L5_1e-3_sdc_rate.svg
```

## 怎么读

- 从 L1→L5，颜色整体变浅（更多格被判 MASKED）。
- **下降最快**的格 = “误差幅度小、最容易被容忍”的地方，通常是**尾数低 bit**；
  **下降最慢**的格 = 即便容忍度很松仍判 SDC，通常是**指数位**（误差被放大到很大）。
- 同一 bit 列内不同输入行的差异 = 输入分布对“多大误差会被暴露”的影响。
- L5 的 SDC 率（0.499）可视为**保守下界**：“即使允许 1% 相对误差，仍有约一半
  的位翻转会造成不可接受的输出偏差”。

## 复现

```bash
# 离线重算（无需 GPU）
python3 scripts/tolerance_sweep.py 015_input_bit_gemm 016_input_instruction_gemm \
    017_input_size_gemm 018_input_position_gemm 019_register_class_gemm \
    --out results/_tolerance
# 分档热力图
python3 scripts/tolerance_heatmaps.py \
    --cells results/_tolerance/tolerance_by_cell.csv \
    --experiment 015_input_bit_gemm --outdir results/_tolerance/figs
```
