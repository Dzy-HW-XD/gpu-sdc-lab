# 容忍度档位（L1–L5）与敏感性分析

> 注意：`near_overflow` 与 `extreme` 两个输入族会**让 GEMM 自身溢出**（golden ~92%/100% 为 Inf），
> 其 golden 退化、幅度指标无定义、容忍度也无法掩盖 Inf，故已从本研究中**完全移除**
> （含定义、记录与 raw）。因此本文与图表仅涵盖 6 个有限输入族。

## 统一说明

- **测试目的**：用**同一批故障样本**（配对设计），只改变“判定容忍度”，测量
  有多少“静默错误”落在可接受范围内（应判为 MASKED），量化 **SDC 严重度敏感区**与**良性误差比例**。
- **判定规则**：某元素 `|fault − golden| > abs_tol + rel_tol·|golden|` 才算损坏；与 golden **逐位相同**的运行在**所有档位**都判 MASKED（含输出本就是 Inf/NaN 的情形）。非有限元素只有**与 golden 不同**时才计损坏。
- **主要变量（自变量）**：容忍度档位 `(abs_tol, rel_tol)`，**同步放大**：每步 abs、rel 同时 ×10。
- **固定变量**：输入分布、bit、指令、规模、时机、寄存器（同一批注入，不重跑 GPU）。
- **数据来源**：从保留的 raw 输出**离线重算**（`scripts/tolerance_sweep.py`），与运行期严格比较 `(0,0)` 解耦。

## 档位表（起点 L1=(1e-7,1e-6)，每步同时 ×10）

| 档 | abs_tol | rel_tol | 相对 L1 | 测试目的（主要变量：容忍度） | 总体 SDC 率 | MASKED |
|---|---|---|---|---|---|---|
| **L1** | 1e-7 | 1e-6 | ×1 | 起点/最严实验口径，仅滤极小舍入噪声 | 0.639 | 2465 |
| **L2** | 1e-6 | 1e-5 | ×10 | 放大 10×，看多少低幅 SDC 被容忍 | 0.553 | 3110 |
| **L3** | 1e-5 | 1e-4 | ×100 | 进入敏感区，看下降斜率 | 0.481 | 3655 |
| **L4** | 1e-4 | 1e-3 | ×1000 | 接近“相对误差千分之一” | 0.399 | 4272 |
| **L5** | 1e-3 | 1e-2 | ×10000 | 最松，近似“相对误差 1% 可接受” | 0.326 | 4822 |

（7540 个有效样本；CRASH 260 在所有档位不变。）

## 各实验 × 各档位 SDC 率

| 实验（第二变量） | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| 015_input_bit_gemm（bit 0–31） | 0.649 | 0.566 | 0.495 | 0.409 | 0.335 |
| 016_input_instruction_gemm（FFMA/FADD/FMUL） | 0.561 | 0.466 | 0.393 | 0.332 | 0.275 |
| 017_input_size_gemm（128/512/1024） | 0.731 | 0.631 | 0.550 | 0.433 | 0.322 |
| 018_input_position_gemm（early/middle/late） | 0.681 | 0.603 | 0.550 | 0.492 | 0.425 |
| 019_register_class_gemm（累加器/临时） | 0.483 | 0.363 | 0.253 | 0.230 | 0.170 |

## 015：按 FP32 位区的 SDC 率（关键结论）

| 档 | 尾数位(0–22) | 指数位(23–30) | 符号位(31) |
|---|---|---|---|
| L1 | 0.608 | 0.751 | 0.756 |
| L2 | 0.494 | 0.751 | 0.756 |
| L3 | 0.395 | 0.751 | 0.756 |
| L4 | 0.275 | 0.751 | 0.756 |
| L5 | 0.174 | 0.745 | 0.750 |

- **尾数位**随容忍度大幅下降（0.61→0.17）——误差小，容易被容忍；
- **指数位**几乎不变（~0.75）——误差被指数级放大，任何容忍度都盖不住；
- **符号位**也几乎不变（~0.75）——误差≈2×|值|，必超阈值。

## 图像（5 实验 × 5 档 = 25 张，均在 `figs/`）

```
figs/heatmap_015_input_bit_gemm_bit_L{1..5}_*_sdc_rate.svg
figs/heatmap_016_input_instruction_gemm_instruction_L{1..5}_*_sdc_rate.svg
figs/heatmap_017_input_size_gemm_size_L{1..5}_*_sdc_rate.svg
figs/heatmap_018_input_position_gemm_position_L{1..5}_*_sdc_rate.svg
figs/heatmap_019_register_class_gemm_register_L{1..5}_*_sdc_rate.svg
```

读图：同一实验内 L1→L5 颜色变浅（更多判 MASKED）；**变浅最快**的格 = 最容易被容忍
（通常尾数低 bit 的小误差），**几乎不变**的格 = 指数位/大误差组合。

## 复现（无需 GPU）

```bash
python3 scripts/tolerance_sweep.py 015_input_bit_gemm 016_input_instruction_gemm \
    017_input_size_gemm 018_input_position_gemm 019_register_class_gemm \
    --out results/_tolerance
python3 scripts/tolerance_heatmaps.py \
    --cells results/_tolerance/tolerance_by_cell.csv \
    --outdir results/_tolerance/figs
```
