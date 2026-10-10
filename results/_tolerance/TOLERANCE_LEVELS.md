# 容忍度档位（L1–L5）与敏感性分析

## 统一说明

- **测试目的**：用**同一批故障样本**（配对设计），只改变“判定容忍度”，测量
  有多少“静默错误”其实落在可接受范围内（应判为 MASKED），从而量化
  **SDC 严重度的敏感区**与**良性误差比例**。
- **判定规则**：某元素 `|fault − golden| > abs_tol + rel_tol·|golden|` 才算损坏；
  一档越松 → 越多小误差被“容忍”→ SDC 率下降。
- **主要变量（自变量）**：容忍度档位 `(abs_tol, rel_tol)`，**同步放大**：每步 abs、rel 同时 ×10。
- **固定变量**：输入分布、bit、指令、规模、时机、寄存器（同一批注入，不重跑 GPU）。
- **因变量**：SDC 率、被判 MASKED 数、逐格 `input × 第二变量` 的 SDC 率。
- **数据来源**：从保留的 raw 输出**离线重算**（`scripts/tolerance_sweep.py`），
  与运行期严格比较 `(0,0)` 完全解耦；改档位只需重跑该脚本。

## 档位表（起点 L1=(1e-7,1e-6)，每步同时 ×10）

| 档 | abs_tol | rel_tol | 相对 L1 | 测试目的（主要变量：容忍度） | 总体 SDC 率 | MASKED |
|---|---|---|---|---|---|---|
| **L1** | 1e-7 | 1e-6 | ×1 | 起点/最严实验口径，仅过滤极小舍入噪声 | 0.727 | 2465 |
| **L2** | 1e-6 | 1e-5 | ×10 | 放大 10×，看有多少低幅 SDC 被容忍 | 0.665 | 3110 |
| **L3** | 1e-5 | 1e-4 | ×100 | 放大 100×，进入敏感区，观察下降斜率 | 0.612 | 3655 |
| **L4** | 1e-4 | 1e-3 | ×1000 | 放大 1000×，接近“相对误差千分之一” | 0.552 | 4272 |
| **L5** | 1e-3 | 1e-2 | ×10000 | 最松，近似“相对误差 1% 可接受” | 0.499 | 4822 |

（10331 个有效样本；CRASH 355 在所有档位不变。）

## 各实验 × 各档位 SDC 率

各实验的**第二变量**不同，但“主要变量”始终是容忍度档位：

| 实验 | 第二变量（逐格维度） | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| 015_input_bit_gemm | bit 0–31 | 0.727 | 0.666 | 0.612 | 0.547 | 0.492 |
| 016_input_instruction_gemm | FFMA / FADD / FMUL | 0.710 | 0.647 | 0.599 | 0.559 | 0.521 |
| 017_input_size_gemm | 128 / 512 / 1024 | 0.806 | 0.739 | 0.685 | 0.607 | 0.533 |
| 018_input_position_gemm | early / middle / late | 0.761 | 0.709 | 0.674 | 0.635 | 0.591 |
| 019_register_class_gemm | accumulator / temporary | 0.626 | 0.540 | 0.462 | 0.445 | 0.402 |

## 015：按 FP32 位区的 SDC 率（关键结论）

| 档 | 尾数位(0–22) | 指数位(23–30) | 符号位(31) |
|---|---|---|---|
| L1 | 0.698 | 0.802 | 0.792 |
| L2 | 0.613 | 0.802 | 0.792 |
| L3 | 0.539 | 0.802 | 0.792 |
| L4 | 0.448 | 0.802 | 0.792 |
| L5 | 0.373 | 0.798 | 0.787 |

- **尾数位**随容忍度大幅下降（0.70→0.37）——误差小，容易被容忍；
- **指数位**几乎不变（~0.80）——误差被指数级放大，任何容忍度都盖不住；
- **符号位**也几乎不变（~0.79）——误差≈2×|值|，必超阈值。

## 图像（5 实验 × 5 档 = 25 张，均在 `figs/`）

每张为 `input × 第二变量` 的 SDC 率热力图（同一批样本，格内标数值）：

```
figs/heatmap_015_input_bit_gemm_bit_L{1..5}_*_sdc_rate.svg
figs/heatmap_016_input_instruction_gemm_instruction_L{1..5}_*_sdc_rate.svg
figs/heatmap_017_input_size_gemm_size_L{1..5}_*_sdc_rate.svg
figs/heatmap_018_input_position_gemm_position_L{1..5}_*_sdc_rate.svg
figs/heatmap_019_register_class_gemm_register_L{1..5}_*_sdc_rate.svg
```

读图：同一实验内 L1→L5 颜色变浅（更多判 MASKED）；**变浅最快**的格是最容易被
容忍的（通常尾数低 bit / 小误差组合），**几乎不变**的格是指数位/大误差组合。

## 复现（无需 GPU）

```bash
python3 scripts/tolerance_sweep.py 015_input_bit_gemm 016_input_instruction_gemm \
    017_input_size_gemm 018_input_position_gemm 019_register_class_gemm \
    --out results/_tolerance
python3 scripts/tolerance_heatmaps.py \
    --cells results/_tolerance/tolerance_by_cell.csv \
    --outdir results/_tolerance/figs
```
