# gpu-sdc-lab

[English](README.md) | [简体中文](README.zh-CN.md)

一个可复现的 **GPU 静默数据错误（SDC）实验台**。仓库分为两层：

- **平台**：workload / injector / observer / oracle / runner 组成的一条可复用流水线，与具体 kernel、注错器和观测方式解耦。
- **实验**：建立在平台之上的 YAML 配置。主实验集是 **001–019**（FP32 标量 GEMM + 输入敏感性）和 **047**（attention Tensor-Core 实验）。

实验不会 fork 平台代码：每个实验都定义在
`experiments/<id>/experiment.yaml`，通过同一套 `sdc-lab` / `campaign.py`
运行。

---

## 平台

### 流水线

```
Workload -> Golden 基准 -> NVBitFI 注错 -> 故障运行
      -> 观测器（输出 / 运行时 / GPU） -> SDC 分类
      -> results/（JSONL + CSV + 图）
```

设计原则：

- golden 运行逐位可复现，`MASKED` 严格等于“输出逐位一致”；
- 每次只注入一个故障，作用于一条动态指令的目标寄存器；
- fault 元数据与 observation 元数据分开保存；
- 图和统计可从保留的原始输出复现；
- GPU 端只需要 Python 标准库。

### 目录结构

```
core/
  workload/       Workload 接口 + 注册表
  injection/      FaultInjector 接口 + NVBitFI 后端
  oracle/         输出比对器 + MASKED/SDC/CRASH 分类器
  observer/       输出 / 运行时 / GPU 观测器
  runner/         编排单个实验
  experiments.py  各实验种类的用例生成与聚合
  taxonomy.py     指令分类树（故障目标选择）
  inputgen.py     合成输入族 + 真实张量加载
  regclass.py     静态 SASS 寄存器分类
workloads/        gemm, gemm_bf16, reduce, conv2d, softmax, attention,
                  attention_forward, mlp
fault_models/     NVBitFI 说明 + 兼容性补丁
experiments/      实验定义（001–047）
scripts/          构建、批量运行、分析、绘图、容忍度、统计
docs/             研究方案、workload 与实验文档
results/          生成的产物
sdc-lab           CLI 入口
```

### 工作负载

| workload | 计算 | 可注入路径 |
|---|---|---|
| `gemm` | FP32 `C=A×B`，确定性 CUDA-Core kernel | FFMA / FADD / FMUL |
| `gemm_bf16` | BF16×BF16 → FP32，cuBLAS Tensor Core | HMMA 累加器 |
| `reduce` | block reduction + `atomicAdd` | atomics / FP32 / load |
| `conv2d` | 直接 2-D 卷积 | FP32 / load |
| `softmax` | 行级 softmax | FP32 / MUFU / load |
| `attention` | 显式 FP32 单头 attention | FP32 / load / MUFU |
| `attention_forward` | BF16 causal 多头 attention，五个 GEMM | HMMA 累加器 |
| `mlp` | 2 层前向 + 一步反向 | FP32 / load / MUFU |

所有 workload 都是确定性的。`mlp` 输出 6 个具名张量，用于逐层传播研究。

### 故障模型与目标

一次故障 = 某个 kernel 调用中，某条动态线程指令的目标寄存器。故障类型：
`bitflip`、`twobit`、`random`、`zero`。

目标从指令分类树中选择：`arithmetic`（fp32、ffma、fadd、fmul、fp16、int、mma）、`memory`（load、atomic）、`control`（predicate）、`meta`（gpr、gp）。NVBitFI 的 `mma` 组已修补，以正确处理 HMMA.16816 的 4 个 FP32 目标寄存器。详见
[docs/TC_BF16_HMMA.md](docs/TC_BF16_HMMA.md)。

### 输入

`core/inputgen.py` 提供确定性输入族：`uniform`、`normal`、`lognormal`、
`sparse`、`cancellation`、`near_overflow`、`correlated`、`adversarial`、
`ones`、`near_zero`、`near_one`、`extreme`、`small`、`large`，以及面向
LLM 激活的 `heavy_tail_act`、`residual_stream`、`swiglu_gated` 和 `real`
（从磁盘加载张量）。输入可按 FP32 或 BF16 物化。详见
[docs/INPUT_FAMILIES.md](docs/INPUT_FAMILIES.md)。

### 分析工具

- `scripts/campaign.py`：批量运行/汇总，把每个实验的 `observations.jsonl` 合并为 master/cells CSV，并计算 Wilson 区间。
- `scripts/heatmap.py`：纯标准库 SVG 热力图 + 透视表 CSV。
- `scripts/tolerance_sweep.py`：对保留的原始输出按 L1–L5 重新分类。
- `scripts/stats.py`：Wilson CI、McNemar、双比例 z、Holm。
- `scripts/features.py`：可解释的输入特征。
- `scripts/predictor.py`、`scripts/sampling.py`：输入感知风险模型与采样。
- `scripts/analyze.py`：单实验汇总。

### 环境（已实测）

| 项 | 值 |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (sm_80) |
| Driver | 580.178.04 (CUDA 13.0) |
| Toolkit | `/usr/local/cuda-13.0`（nvcc 13.0.88） |
| NVBit | v1.8.1 |
| NVBitFI | NVlabs/nvbitfi（已打补丁） |
| Python | 3.12；GPU 节点上仅用标准库 |

### 构建与运行

```bash
./scripts/setup_nvbitfi.sh     # 拉取 + 打补丁 + 编译 NVBit/NVBitFI
./scripts/build.sh             # 编译 workloads
./sdc-lab doctor               # 环境 / GPU / 工具自检

./sdc-lab baseline --experiment 001_gemm_baseline
./sdc-lab run      --experiment 002_fault_position
./sdc-lab run      --experiment 047_attn_5gemm_full
```

`run` 会输出 total / masked / SDC / crash / not-injected 计数和 SDC rate。
常用覆盖参数：`--groups`、`--positions`、`--bits`、`--M/--N/--K`、`--gpu`、
`--seed`、`--num-injections`。

### 结果目录

```
results/<experiment>/
  config.yaml                 # 实验定义
  experiment.json             # 计数 + 聚合
  observations.jsonl          # 每次注错一行 JSON
  summary.csv                 # 总体计数
  <axis>_summary.csv          # 按维度的拆分
  report.md                   # 人类可读报告（部分实验有）
  raw/                        # 每次注错的原始输出（较大，归档）
```

`observations.jsonl` 把 **fault 元数据** 与 **observation 元数据** 分开保存，
因此可以直接做“故障点 → SDC 概率 → 可观测特征”的研究。大文件（`*.bin`）
和 raw archive 默认不进 git。

### SDC 分类

| 分类 | 条件 |
|---|---|
| `CRASH` | 超时、非零退出码、CUDA 错误 / 非法访存 |
| `MASKED` | 正常结束，且输出与 golden 逐位一致 |
| `SDC` | 正常结束，但输出与 golden 不同（NaN/Inf 额外标记） |
| `NOT_INJECTED` | 故障点从未执行到（仅记账） |

阈值来自 `oracle.abs_tol/rel_tol`；由于 golden 逐位可复现，默认精确比较。

### 扩展

- **Workload**：实现 `core/workload/base.py` 并注册。
- **Injector**：实现 `core/injection/base.py` 并注册。
- **Observer**：在 `core/observer/` 增加观测器。
- **Oracle / 实验种类**：在 `core/experiments.py` 增加构建器。

### 兼容性（A100 + CUDA 13.0）

NVBitFI（2020 年的工具）需要两处源码补丁，由
`scripts/setup_nvbitfi.sh` 自动应用：`ballot()` →
`__activemask()`/`__ballot_sync`；移除 device 端
`printf()/assert()`（仅 CUDA ≥ 13.1 支持）。不需要改动系统驱动。

---

## 实验

下面所有实验都是平台上的配置。**主实验集**是 001–019 和 047；
020–023、030–040 是开发/辅助运行，放在最后。

### 001–007：平台打通与 FP32 GEMM 基础

| ID | 问题 | 成果 |
|---|---|---|
| [001_gemm_baseline](experiments/001_gemm_baseline/experiment.yaml) | golden GEMM 是否逐位可复现？ | [报告](results/001_gemm_baseline/report.md) |
| [002_fault_position](experiments/002_fault_position/experiment.yaml) | 指令位置 early/middle/late 如何影响 SDC？ | [报告](results/002_fault_position/report.md) |
| [003_fault_type](experiments/003_fault_type/experiment.yaml) | bitflip / twobit / random / zero 的差异？ | [报告](results/003_fault_type/report.md) |
| [004_fault_bit](experiments/004_fault_bit/experiment.yaml) | 符号位 / 指数位 / 尾数位哪个最危险？ | [报告](results/004_fault_bit/report.md) |
| [005_gemm_size](experiments/005_gemm_size/experiment.yaml) | GEMM 规模是否改变 SDC 概率？ | [报告](results/005_gemm_size/report.md) |
| [006_data_pattern](experiments/006_data_pattern/experiment.yaml) | 输入数据模式是否影响 SDC？ | [报告](results/006_data_pattern/report.md) |
| [007_reduce_atomic](experiments/007_reduce_atomic/experiment.yaml) | atomics 与 FP32/load/global 注错有何不同？ | [报告](results/007_reduce_atomic/report.md) |

组级汇总：[results/SDCLAB_SUMMARY.md](results/SDCLAB_SUMMARY.md)。

### 008–014：跨 workload 的输入敏感性

这组实验采用配对设计：固定故障点，只改变输入分布，用来建立“输入依赖
SDC”的方向。论文中的输入敏感性分析来自这组实验；当前 check-out 没有为
每个实验单独保留结果目录。

| ID | 问题 | 链接 |
|---|---|---|
| [008_input_sensitivity_gemm](experiments/008_input_sensitivity_gemm/experiment.yaml) | GEMM 的输入族敏感性 | [方案](docs/RESEARCH_PLAN.md)、[摘要](PAPER_ABSTRACT.md) |
| [009_input_sensitivity_reduce](experiments/009_input_sensitivity_reduce/experiment.yaml) | reduce / atomics 的输入敏感性 | [方案](docs/RESEARCH_PLAN.md) |
| [010_input_sensitivity_softmax](experiments/010_input_sensitivity_softmax/experiment.yaml) | softmax 的输入敏感性 | [方案](docs/RESEARCH_PLAN.md) |
| [011_input_sensitivity_conv2d](experiments/011_input_sensitivity_conv2d/experiment.yaml) | 2-D 卷积的输入敏感性 | [方案](docs/RESEARCH_PLAN.md) |
| [012_input_sensitivity_attention](experiments/012_input_sensitivity_attention/experiment.yaml) | 显式 attention 的输入敏感性 | [方案](docs/RESEARCH_PLAN.md) |
| [013_input_sensitivity_mlp](experiments/013_input_sensitivity_mlp/experiment.yaml) | MLP 输入敏感性与逐层传播 | [方案](docs/RESEARCH_PLAN.md) |
| [014_cross_workload](experiments/014_cross_workload/experiment.yaml) | 跨 workload 对比 | [方案](docs/RESEARCH_PLAN.md) |

### 015–019：输入 × 变量批量实验（FP32 标量 GEMM）

`kind: input_grid` 在固定其它变量的前提下，遍历“输入分布 × 单个第二变量”。
这是 FP32 标量 CUDA-Core GEMM 阶段，已归档；见
[docs/ARCHIVE_scalar_gemm.md](docs/ARCHIVE_scalar_gemm.md)。

| ID | 第二变量 | 成果 |
|---|---|---|
| [015_input_bit_gemm](experiments/015_input_bit_gemm/experiment.yaml) | FP32 bit 0–31 | [结果](results/015_input_bit_gemm/) |
| [016_input_instruction_gemm](experiments/016_input_instruction_gemm/experiment.yaml) | FFMA / FADD / FMUL | [结果](results/016_input_instruction_gemm/) |
| [017_input_size_gemm](experiments/017_input_size_gemm/experiment.yaml) | 128 / 512 / 1024 | [结果](results/017_input_size_gemm/) |
| [018_input_position_gemm](experiments/018_input_position_gemm/experiment.yaml) | early / middle / late | [结果](results/018_input_position_gemm/) |
| [019_register_class_gemm](experiments/019_register_class_gemm/experiment.yaml) | accumulator / temporary | [结果](results/019_register_class_gemm/) |

组级链接：
[per-cell 汇总](results/_campaign/CAMPAIGN_SUMMARY.md)、
[L1–L5 容忍度](results/_tolerance/TOLERANCE_LEVELS.md)。

### 047：attention 五 GEMM 实验

`047_attn_5gemm_full` 在 BF16 causal attention forward（`S=128, D=2048,
H=32`）的 HMMA 累加器上注入单个 bit flip。覆盖五个 GEMM 阶段
（`qkv_q`、`qkv_k`、`qkv_v`、`qk`、`pv`）、bit 24–30、early/middle/late
位置，每个 cell 30 次注入，共 3150 次。判据为 L4
（`|ΔY| > 1e-4 + 1e-3|Y|`）。

主要结果：**1507/3150 SDC（47.8%）**。QKV-Q、QK、PV 严格单 token；
QKV-K 影响范围宽（中位 20 个 token 行）；QKV-V 介于两者之间。bit 30 有
55 次运行产生 NaN/Inf，已单独处理。

链接：
[实验产物](results/047_attn_5gemm_full/)、
[图说明](results/047_attn_5gemm_full/final_heatmaps_v2/README.md)、
[单/多 token 总结图](results/047_attn_5gemm_full/final_heatmaps_v2/summary/token_count_distribution.png)、
[golden 数值差距图](results/047_attn_5gemm_full/final_heatmaps_v2/summary/golden_vs_delta_hexbin.png)、
[attention 设计](docs/ATTENTION.md)、
[热图重绘说明](docs/HEATMAP_REDESIGN.md)。

### 辅助 / 开发实验

| ID | 用途 | 链接 |
|---|---|---|
| 020–023 | BF16 Tensor-Core 输入 × bit/size（early/middle/late） | [TC 汇总](results/_tc_campaign/TC_SUMMARY.md)、[容忍度](results/_tc_tolerance/TOLERANCE_LEVELS.md)、[文档](docs/TC_BF16_HMMA.md) |
| 030–033 | attention 开发/目标规模基线 | [attention 文档](docs/ATTENTION.md)、[配置](experiments/030_attn_single_dev_baseline/experiment.yaml) |
| 040 | attention stage smoke test | [attention 文档](docs/ATTENTION.md)、[配置](experiments/040_attn_stage_smoke/experiment.yaml) |

---

## 文档索引

- [PAPER_ABSTRACT.md](PAPER_ABSTRACT.md) — 论文摘要（中英文）
- [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) — 实验方案
- [docs/INPUT_FAMILIES.md](docs/INPUT_FAMILIES.md) — 输入族说明
- [docs/TC_BF16_HMMA.md](docs/TC_BF16_HMMA.md) — Tensor-Core/HMMA 实验
- [docs/ATTENTION.md](docs/ATTENTION.md) — attention workload 与 047
- [docs/HEATMAP_REDESIGN.md](docs/HEATMAP_REDESIGN.md) — 热图重绘与 caption
- [docs/ARCHIVE_scalar_gemm.md](docs/ARCHIVE_scalar_gemm.md) — FP32 标量 GEMM 归档
