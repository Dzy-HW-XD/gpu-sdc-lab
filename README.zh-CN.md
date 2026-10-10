# gpu-sdc-lab

一个长期维护、可渐进扩展的 **GPU 静默数据错误（SDC）实验台**。

Phase 1 在单张 A100 上打通了一条完整、可复用的链路：

```
Workload(GEMM) -> Golden 黄金基准 -> 故障注入(NVBitFI) -> 故障运行
      -> 观测器(输出/运行时/GPU) -> SDC 分类(MASKED/SDC/CRASH)
      -> results/ (JSONL + CSV)
```

整个框架刻意与 workload、injector、observer 解耦：以后加入新的
workload、故障模型、观测方法，只需插入同一个 `Runner`，不需要推倒重来。

## 架构

```
                    SDC LAB
                       |
        +--------------+--------------+
        |              |              |
     Workload       Injector       Observer
     (gemm)        (nvbitfi)     output/runtime/gpu
        |              |              |
        +--------------+--------------+
                       |
                    Oracle (比对 + 分类)
                       |
                    results/
```

```
core/
  workload/    Workload 接口 + GEMM + 注册表
  injection/   FaultInjector 接口 + NVBitFI + 注册表
  oracle/      输出比对器 + MASKED/SDC/CRASH 分类器
  observer/    OutputObserver / RuntimeObserver / GPUObserver
  runner/      编排单个实验
  experiments.py  各实验种类的用例生成与聚合（含 input_grid）
  config.py, util.py, yamlmini.py
workloads/          gemm, reduce, conv2d, softmax, attention, mlp（各含 README）
fault_models/nvbitfi/  后端说明 + 兼容性补丁
experiments/     001..019 实验定义
core/taxonomy.py  指令分类树（故障目标选择）
core/inputgen.py  输入分布生成器（14 个合成族 + 真实张量）
core/regclass.py  静态 SASS 寄存器分类（累加器 / 临时）
scripts/          build.sh, setup_nvbitfi.sh, smoke_workloads.sh,
                  analyze.py, stats.py, features.py, predictor.py, sampling.py,
                  campaign.py（批量遍历+汇总）, heatmap.py（纯标准库 SVG 热力图）
docs/RESEARCH_PLAN.md  论文实验方案
results/         生成的产物
scripts/         setup_nvbitfi.sh, build.sh, env.sh
sdc-lab          CLI 入口
```

## 环境（已实测）

| 项          | 值                                           |
|-------------|----------------------------------------------|
| GPU         | NVIDIA A100-SXM4-80GB (sm_80)，GPU index 1   |
| Driver      | 580.178.04 (CUDA 13.0)                       |
| Toolkit     | /usr/local/cuda-13.0 (nvcc 13.0.88)          |
| NVBit       | v1.8.1                                       |
| NVBitFI     | NVlabs/nvbitfi（已打补丁，见下）             |
| Python      | 3.12（仅标准库；不需要 numpy/pandas）        |

## 安装 / 构建

```bash
# 1. 拉取 + 打补丁 + 编译 NVBit 与 NVBitFI（不改动系统环境）
./scripts/setup_nvbitfi.sh

# 2. 编译 GEMM workload
./scripts/build.sh

# 3. 自检
./sdc-lab doctor
```

**可任意路径执行（可移植）。** 工程内没有任何硬编码路径：所有路径都在运行时
从工程根目录推导（`realpath`），因此可以放在任意目录、从任意当前目录调用，
甚至通过软链接调用。CUDA 工具链自动探测（`$CUDA_HOME` -> `PATH` 上的 `nvcc`
-> `/usr/local/cuda*`）；也可用 `CUDA_HOME=...` 或 `config.yaml` 里的
`cuda_home:` 覆盖。`results/` 与 `third_party/` 都相对于工程根目录。

## 使用

```bash
# 环境 / GPU / 工具自检
./sdc-lab doctor
./sdc-lab profile --M 1024 --N 1024 --K 1024

# 基线确定性（连续 10 次，不注错）
./sdc-lab baseline --experiment 001_gemm_baseline

# 故障注入实验
./sdc-lab run --experiment 002_fault_position
./sdc-lab run --experiment 003_fault_type
./sdc-lab run --experiment 004_fault_bit
./sdc-lab run --experiment 005_gemm_size
./sdc-lab run --experiment 006_data_pattern
./sdc-lab run --experiment 007_reduce_atomic

# 输入相关性实验（input × bit / 指令 / 规模 / 时机）
./sdc-lab run --experiment 008_input_sensitivity_gemm
./sdc-lab run --experiment 015_input_bit_gemm
./sdc-lab run --experiment 016_input_instruction_gemm

# 批量活动：运行 + 汇总 + 热力图（见下节）
python3 scripts/campaign.py run 015_input_bit_gemm
python3 scripts/campaign.py collect 015_input_bit_gemm
python3 scripts/heatmap.py results/_campaign/cells.csv --axis bit

# 临时实验：100 次 bitflip，混合目标
./sdc-lab run --workload gemm --injector nvbitfi \
    --fault-type bitflip --num-injections 100

# 查看指令分类树
./sdc-lab taxonomy
```

常用覆盖参数：`--groups memory/atomic,arithmetic/fp32,memory/load,meta/gp`、
`--group fp32`、`--num-injections N`、`--positions`、`--bits`、
`--M/--N/--K`、`--gpu`、`--seed`、`--id`。

`run` 输出示例：

```
Total   : 100
Masked  : 6
SDC     : 82
Crash   : 12
NotInj  : 0
SDC rate: 0.8200
```

## 结果目录结构

```
results/002_fault_position/
  config.yaml                 # 实验定义
  experiment.json             # 计数 + 按维度的聚合
  observations.jsonl          # 每次注错一行 JSON
  summary.csv                 # 总体计数
  fault_position_summary.csv  # position,total,masked,sdc,crash,sdc_rate
  report.md
  raw/0001_fp32/              # injection-info, injection-log, out.bin/json,
                              # stdout, stderr, result.json
```

`observations.jsonl` 把 **fault 元数据** 与 **observation 元数据** 分开保存，
因此可以直接用 `pandas.read_json(..., lines=True)` 做
"故障位置 -> SDC 概率 -> 可观测特征" 的研究。

## 故障模型（NVBitFI）

一次故障 = 某个 kernel 调用中，某条动态线程指令的目标寄存器。定位方式为
`(kernel, 调用序号, 指令组, 组内动态指令 id)` 加上 `(bit-flip 模型, bit id seed)`。
故障类型：`bitflip`、`twobit`、`random`、`zero`。
详见 `fault_models/nvbitfi/README.md`。

## 故障目标（分类树）

目标从一棵小型指令树（`core/taxonomy.py`）中选择；每个叶子映射到一个
NVBitFI 组，并可选带 opcode 白名单：

```
arithmetic: fp64, fp32, ffma*, fadd*, fmul*, fp16*, int*, mma*
            (* 按 opcode 过滤：ffma/fadd/fmul 过滤 fp32 桶，其余过滤 others 桶)
memory:     load, atomic, store(不可注)
control:    predicate, nodest(不可注)
meta:       gppr, gp, other
```

`./sdc-lab taxonomy` 可打印该树。选择支持叶子名（`--group atomic`）、路径
（`--group memory/atomic`）或整棵子树（`--group arithmetic`）。`memory/atomic`
使用由 `scripts/patch_nvbitfi.py` 新增的专用 NVBitFI 组 `atom`；标了
opcode-filtered 的叶子在共享的 `others` 桶里采样，只有命中的 opcode 属于白名单
才计入统计，否则记为 `NOT_TARGETED`。

## 工作负载与输入

六个确定性 workload：`gemm / reduce / conv2d / softmax / attention / mlp`
（mlp 为“前向+一步反向”的小型训练步，输出 6 个具名张量，用于研究误差**逐层传播**）。

输入生成器（`core/inputgen.py`）：`uniform / normal / lognormal / sparse /
cancellation / near_overflow / correlated / adversarial / ones / near_zero /
near_one / extreme / small / large` + `real`。带 `input_spec` 的实验会把输入
物化到 `results/_inputs/` 并传给 kernel。`./sdc-lab gen-input` 可直接生成。
`scripts/features.py` 提取可解释的输入特征（动态范围、抵消、指数统计等）。
每种输入分布的含义与示例见 [docs/INPUT_FAMILIES.md](docs/INPUT_FAMILIES.md)。

## 输入相关性批量实验（015–019）

`kind: input_grid` 在固定其它变量的前提下，遍历“输入分布 × 单个第二变量”
（配对设计），cell 命名为 `input|<值>`：

| 实验 | 第二变量 | 说明 |
|---|---|---|
| `015_input_bit_gemm` | FP32 bit 0–31 | 核心 input × bit 热力图 |
| `016_input_instruction_gemm` | FFMA / FADD / FMUL | 按 opcode 过滤；FADD/FMUL 用 `--op-mode muladd` |
| `017_input_size_gemm` | 128/512/1024 | |
| `018_input_position_gemm` | early / middle / late | |
| `019_register_class_gemm` | 累加器 / 临时寄存器 | 静态 SASS 分类，见 `core/regclass.py` |

批量运行 + 汇总 + 绘图：

```bash
python3 scripts/campaign.py run 015_input_bit_gemm 016_input_instruction_gemm
python3 scripts/campaign.py collect 015_input_bit_gemm 016_input_instruction_gemm
python3 scripts/heatmap.py results/_campaign/cells.csv --axis bit
# 大动态范围指标：对数色标 + 每格标注数值
python3 scripts/heatmap.py results/_campaign/cells.csv --axis bit \
    --metric mean_abs_err --log-color --annotate
python3 scripts/stats.py --cells results/_campaign/cells.csv uniform bit
```

`campaign.py` 把每个实验的 `observations.jsonl` 合并为
`results/_campaign/master.csv` 与 `results/_campaign/cells.csv`（按维度给出
Wilson 置信区间）；`heatmap.py` 用纯标准库渲染 SVG 热力图并导出透视表 CSV。

## 指令类型 / 寄存器类型

NVBitFI 的 `fp32` 是一个整组；`core/taxonomy.py` 为它增加了按 opcode 过滤的
视图（`arithmetic/ffma`、`arithmetic/fadd`、`arithmetic/fmul`），只有命中白名单
opcode 才计入。GEMM 内层循环几乎全是 FFMA，因此 `gemm` 支持 `--op-mode muladd`
强制使用 `__fmul_rn`/`__fadd_rn`（仍然确定性），在同一 kernel 内造出 FADD/FMUL；
默认 `--op-mode fma` 与原始 GEMM 逐位一致。

`core/regclass.py` 用 `cuobjdump -sass`（缓存于 `results/_sass/`）把每条 FP32
指令的目标寄存器标为 `accumulator`（目标同时是操作数，即原地累加）或
`temporary`；runner 会在每条记录写入 `fault.register_class` 并输出
`register_class_summary.csv`。这是静态、启发式的分类。

## SDC 分类

| 分类         | 条件 |
|--------------|------|
| CRASH        | 超时、非零退出码、CUDA 错误 / 非法访存 / 地址未对齐 |
| MASKED       | 正常运行结束，且输出与黄金基准逐位一致 |
| SDC          | 正常运行结束，但输出与黄金基准不同（NaN/Inf 会额外标记） |
| NOT_INJECTED | 故障点从未执行到（仅用于记账） |

阈值来自 `oracle.abs_tol/rel_tol`，默认精确比较——因为 GEMM 黄金基准
逐位可复现（已在实验 001 验证）。

## 扩展接口

- **Workload**：实现 `core/workload/base.py` 并注册。
- **Injector**：实现 `core/injection/base.py`（如软件注错器）并注册；
  runner 从不直接依赖 NVBitFI。
- **Observer**：在 `core/observer/` 增加 `Observer`（激活/梯度/损失/权重/DCGM），
  runner 会自动聚合。
- **Oracle / 实验种类**：在 `core/experiments.py` 增加构建器。

## 兼容性说明（A100 + CUDA 13.0）

NVBitFI（2020 年的工具）需要两处源码补丁，已由 `scripts/setup_nvbitfi.sh`
自动应用；**未改动系统驱动/CUDA**：

1. profiler：`ballot()` -> `__activemask()` / `__ballot_sync`（sm_70+ 已移除）。
2. injector：编译时剥离 device 端 `printf()/assert()`（它们要求 CUDA >= 13.1，
   否则会触发 `CUDA_ERROR_INVALID_SOURCE`）。
