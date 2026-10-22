# 同一逻辑 GEMM、不同 Kernel 实现的 HMMA 故障传播实验设计

日期：2026-10-22  
定位：**实验设计，尚未执行；当前阶段只做 HMMA 指令级注错，不做软件输出注错**  

> 当前主实验已经收窄到 `qkv_v` 单 stage，专用设计见 `docs/EXPERIMENT_PLAN_QKV_V_HMMA.md`。本文件保留为后续 kernel 变体/机制分析的参考框架。  
目标：先用 NVBit 在 HMMA 指令边界注错，描述故障影响分布；再研究 kernel 实现方式是否改变该分布。三个候选机制：

1. 指令调度与依赖链；
2. 累加器寄存器生命周期与复用；
3. 累加器转存/共享内存往返与流水线阶段。

> 重要边界：NVBit 注入 HMMA 目的寄存器测到的是**指令边界/累加器寄存器级**故障传播，不是 Tensor Core 内部 adder tree 的真实 AVF。本文所有 “AVF-like” 指标都是 instruction-level proxy，不能写成完整微架构 AVF。

> 当前优先级：**L1/HMMA 指令级注错为主**。L0 软件输出注错（SVF）暂时不做，只保留为后续可选的对照，不进入当前主矩阵。这里“软件层”只作为**观察下游影响**的 consumer，不作为注错位置。

---

## 1. 研究问题与假设

### 1.1 核心问题

先固定一个逻辑 GEMM：

- 相同 `M, N, K`；
- 相同输入矩阵、数据类型、累加精度、输出布局；
- 相同下游 consumer；

然后只通过 NVBit 在 **HMMA 指令执行后的 FP32 destination register** 翻转 1 bit，回答：

> **同一个逻辑 GEMM 中，不同 HMMA 指令位置、不同 bit、不同输出位置的故障，经过后续指令和 consumer 后，影响如何分布？**

在这个分布基线建立后，再改变 kernel 实现：

> **同一个逻辑输出元素、同一个 K 位置、同一个 bit 的故障，最终到达 consumer 的概率和误差分布，是否会随 kernel 实现改变？**

### 1.2 三个机制假设

| 机制 | 假设 | 主要操纵 |
|---|---|---|
| H1 指令调度 | 不同 pipeline/stage 配置改变 HMMA 依赖链和错误被后续指令读取的机会 | K2a |
| H2 寄存器生命周期 | 不同 CTA/warp/accumulator 布局改变寄存器复用、写回和读取距离 | K3 |
| H3 转存/流水线 | 累加器在 RF 与 SMEM 之间往返会改变错误存活和可见性 | K1/K2/K3 spill |

**零假设**：在控制数值舍入顺序和 clean 差异后，kernel 实现不改变故障传播。

> AVF 与 SVF 的总体差距本身不能证明上述任一机制是主因。当前实验不把总体差距直接归因于累加器转存或流水线，只检验指令级注错后的影响分布及其机制相关性。

---

## 2. 总体设计

### 2.1 三层故障抽象与当前范围

| 层级 | 注入位置 | 当前是否做 | 说明 |
|---|---|---|---|
| **L1 / HMMA** | HMMA 指令执行后的 FP32 destination register | **是，主实验** | NVBit 在指令边界翻转 1 bit |
| **L2 / Spill** | 累加器写入 SMEM 前/后的状态 | **是，Phase 3** | 人为增加 RF→SMEM→RF 往返 |
| **L0 / SVF** | GEMM 最终输出元素 | **暂不做** | 软件输出注错，留作后续对照 |

L2 在 L1 基础上增加显式转存；L0 暂不进入主矩阵。

### 2.2 为什么仍然需要 consumer

当前不做软件注错，但仍需要一个固定的下游 consumer 来观察影响是否到达软件可见层：

```text
C = A × B                    （FP32 accumulator / FP32 output）
Cq = quantize_bf16(C)        （模拟写回/精度截断）
Y = Cq × Wc                  （固定 consumer GEMM）
metric = top1(Y) 或 ||Y - Y_clean||
```

- consumer 是**观察器**，不是注错器；
- `Wc` 固定、随机、所有 kernel 变体共用；
- 同时记录 **local visibility**（C 是否变化）和 **consumer visibility**（Y/top1 是否变化）。

这样可以直接观察：

> 某个 HMMA 指令级 fault 在 C 上可见，但经过 BF16 量化和 consumer 后不可见。

---

## 3. Phase 0：可控性与映射验证

### 3.1 固定条件

| 项目 | 设置 |
|---|---|
| GPU | 单张 A100 80GB，固定 GPU UUID |
| 架构 | `sm_80`，BF16 Tensor Core |
| 输入 | BF16，固定随机种子，数值范围避免溢出 |
| 累加 | FP32 |
| GEMM 输出 | FP32，便于观察 bit 28–30 |
| 逻辑形状 | 第一阶段 `M=N=K=4096` |
| 输出布局 | Row-major，固定 epilogue |
| 编译 | 固定 CUTLASS commit、CUDA、编译选项；保留 `-lineinfo` |
| 精度 | 关闭 TF32；不允许 fast-math 改变累加 |
| 注入 | 每个 run 只翻转 1 个 bit |

选择 FP32 输出而不是 BF16 输出的原因：

- bit 28–30 在 FP32 accumulator 中有明确位置；
- 避免 BF16 epilogue 直接把这些 bit 舍入掉，先把传播机制看清楚；
- BF16 写回作为后续单独因素，而不是混在 kernel 比较里。

### 3.2 Kernel 校验

对每个 kernel 变体记录：

- SASS hash；
- HMMA 指令数；
- 实际使用的 MMA 形状，例如 `HMMA.16816.F32`；
- CTA tile、warp tile、K tile、stage 数；
- 寄存器数、shared memory、occupancy；
- 是否存在编译器生成的 spill；
- clean 输出与 CPU FP64 参考的误差；
- 重复 clean 是否 bitwise 一致。

### 3.3 逻辑故障坐标

每个 fault 用逻辑坐标定义，而不是只看动态指令编号：

```text
logical_site = (output_i, output_j, k_index, bit)
```

其中：

- `output_i, output_j`：输出矩阵元素；
- `k_index`：K 方向逻辑位置，early / middle / late；
- `bit`：28 / 29 / 30。

对每个 kernel 变体，映射到：

```text
physical_site = (
  cta_id, warp_id, lane_id,
  accumulator_register, hmma_dynamic_id, hmma_pc,
  output_tile, k_tile
)
```

映射必须通过 SASS 和注入定位实验验证，不能假设所有 kernel 的寄存器布局相同。

### 3.4 注入有效性检查

每个 run 必须记录并验证：

- opcode 是 HMMA；
- 目标 register 正确；
- `before` / `after` 恰好差 1 bit；
- 目标 register 在注入后是否被读取；
- 是否在读取前被覆盖；
- 最终是否到达 epilogue；
- sham 输出是否与 clean 完全一致。

### 3.5 Phase 0 通过标准

- 至少 1 个 CUTLASS kernel 可稳定编译和运行；
- clean 输出可重复；
- SASS 中能看到 HMMA；
- 能定位至少 9 个逻辑/物理注入点；
- 至少 1 个 fault 改变 C；
- 至少 1 个 fault 在 C 上变化但 consumer 不变；
- sham 100% 等于 clean；
- 注入日志没有 “not injected”。

---

## 4. Phase 1：HMMA 故障传播基线

### 4.1 Kernel K1

基础 CUTLASS Tensor Core GEMM：

| 参数 | 建议值 |
|---|---|
| CTA tile | 128×128 |
| Warp tile | 64×64 |
| K tile | 32 |
| Stages | 3 |
| Warp 数 | 4 |
| Instruction | `m16n8k16` BF16 → FP32 |

实际 tile 以编译后 SASS 为准。

> 术语说明：这里的 **`Stages` 是 CUTLASS 主循环的 software pipeline depth**，也就是共享内存多级缓冲/预取级数，不是 Qwen 里的 `qkv_q/qk/...` 逻辑 GEMM stage。下文用大写 `Stages` 表示 pipeline depth，用小写 `stage` 表示逻辑 GEMM 阶段。

### 4.2 故障采样

先做一个**位置分布预扫描**，再铺正式 135 点矩阵。

#### 4.2a HMMA 位置分布预扫描

固定 K1、中心输出元素、bit 30，沿 K 方向取 12 个位置：

```text
0%, 9%, 18%, ..., 100% 的 K 进度
= 12 次 HMMA fault run
```

目的不是估计总体失败率，而是先看：

- 故障影响是否随 HMMA 在 K-loop 中的位置单调变化；
- 是否存在“早期可见、晚期被吸收”或相反的区间；
- early/middle/late 三分层是否足够；
- 误差幅度是否在某个位置出现长尾。

如果 12 点分布明显不均匀，正式矩阵增加 K 分层；如果近似平坦，再使用 3 个 K 位置。

#### 4.2b 正式采样

每个 kernel 变体：

```text
3 bits × 3 K positions × 3 output regions × 5 output elements
= 135 个独立 HMMA fault run
```

这里的 135 是 **135 次独立实验**，不是“一次实验里翻 135 个 bit”。

- 每次 run 只注入 **1 个 bit**；
- 每次 run 都重新启动 kernel、重新 prefill/执行；
- 3 个 bit：28 / 29 / 30；
- 3 个 K 位置：early / middle / late；
- 3 个输出区域：中心、边缘、随机；
- 每个区域 5 个**不同的逻辑输出元素**，用来覆盖不同 lane/register/CTA；
- 不是同一个元素重复 5 次；5 个元素是 5 个不同的采样点；
- 每个物理点只注入一次；确定性重复只用于检查，不计入 135 个独立样本。

另外做：

- **10 次 clean**：每个 kernel 变体各做 10 次**不注错**的完整执行。用途是检查同一输入下输出是否 bitwise 稳定、checksum 是否一致、时钟/温度是否造成漂移。clean 不进入 fault 分母。
- **27 次 sham**：每次 run 仍然安装 NVBit 注入路径、选择故障点、执行完整 kernel，但**不翻 bit**。27 = `3 K positions × 3 output regions × 3 次重复`。用途是验证“注入流程本身不会改变输出”。sham 也不进入 fault 分母。
- **3 次 dead-register 对照**：故意选择一个在读取前就被覆盖、或者写入后不再被读取的寄存器，预期最终输出不变。用途是验证注入工具不会产生假阳性。

### 4.3 影响分布分析

当前不做 L0 软件输出注错。Phase 1 的主结果是 **135 个 HMMA 指令级 fault 的影响分布**：

- 按 bit 28/29/30 分组；
- 按 K 位置 early/middle/late 分组；
- 按输出区域（中心、边缘、随机）分组；
- 按输出元素/lane/CTA 分组；
- 按 HMMA 依赖类别分组：dead / accumulated / epilogue-only；
- 按影响类别分组：local-visible / consumer-visible / consumer-masked / NaN/Inf；
- 按误差幅度分组：`rel_l2`、`max_abs`、bit mismatch count。

输出至少包括：

```text
每个 cell 的 n、visible rate、masked rate、dead rate
每个 cell 的误差中位数/分位数/最大值
不同 K 位置和 bit 的分布直方图
```

Phase 1 的正式 fault 总数：

```text
L1 HMMA fault: 135
```

另有 12 次位置预扫描 run，单独分析，不混入 135 点矩阵；clean/sham/dead 控制 run 也不计入 135 个 fault 样本。对每个变体分别做 clean；Phase 1 只有 K1 时，就是 10 次 clean、27 次 sham、3 次 dead。

当前阶段**不做** L0 软件输出注错；软件层只作为 consumer 观察输出影响。L0 若以后需要，作为单独的可选对照实验。

### 4.4 需要的 SASS 依赖信息

对每个注入点记录：

- 该 HMMA 的 `D` 和 `C` 是否为同一寄存器；
- 注入后到下一条读取该寄存器的 HMMA 之间隔了多少动态 HMMA；
- 后续读取该寄存器的 HMMA 数；
- 寄存器是否在读取前被覆盖；
- 到 epilogue 的生存距离；
- 是否进入非 HMMA 的 epilogue 读写。

分类：

| 类别 | 含义 |
|---|---|
| dead | 注入后、读取前被覆盖 |
| accumulated | 被后续 HMMA 继续读取/累加 |
| epilogue-only | 只在最终写回时被读 |
| output-visible | 最终 C bitwise 改变 |
| consumer-visible | 经过 BF16 + consumer 后仍改变 |
| consumer-masked | C 改变但 consumer 不变 |

---

## 5. Phase 2：改变调度与流水线

### 5.1 四个版本

| 版本 | CTA tile | K tile | Stages | Warp tile | 目的 |
|---|---|---:|---:|---|---|
| K1 | 128×128 | 32 | 3 | 64×64 | 基线 |
| K2a | 128×128 | 32 | 5 | 64×64 | 只改 pipeline stage/双缓冲 |
| K2b | 128×128 | 64 | 3 | 64×64 | 改 K tile，改变累加分组/舍入 |
| K3 | 128×256 | 32 | 3 | 64×64 | 改 CTA tile、warp 数和 accumulator 布局 |

K2a 用于尽量隔离“调度/流水线”，K2b 用于暴露“调度 + 数值顺序”的混合效应，K3 用于研究寄存器生命周期。

### 5.2 保持不变的量

- 相同 A/B 输入；
- 相同数学 `C=A×B`；
- 相同输出布局；
- 相同 epilogue；
- 相同 consumer；
- 相同逻辑 fault 坐标；
- 相同 bit。

### 5.3 数值顺序混杂

不同 K tile/warp 布局会改变 FP32 累加顺序，因此 clean 输出可能不同。

必须区分：

1. **clean 差异**：K1/K2/K3 的无故障输出差异；
2. **fault 增量**：`F_variant - C_variant`；
3. **跨变体 fault 差异**：在各自 clean 基础上的归一化误差。

如果 fault 差异与 clean 数值差异同量级，不能直接归因于调度或寄存器生命周期。

### 5.4 机制指标

- 动态 HMMA 消费者数量；
- 目标寄存器 lifetime（以 HMMA 指令数计）；
- register reuse distance；
- dead 比例；
- consumer-visible 比例；
- local error / consumer error；
- SASS 中 HMMA 顺序和依赖链是否改变。

### 5.5 样本量

每个变体 135 个 fault：

```text
K1 + K2a + K2b + K3 = 4 × 135 = 540 个 HMMA fault
```

K1 可复用 Phase 1 数据，因此 Phase 2 新增约 405 个 run。

---

## 6. Phase 3：验证转存是否是关键因素

### 6.1 实现方式

Phase 3 使用一个可控的 Tensor Core microkernel，基于 CUTLASS warp-level MMA / `cutlass::arch::Mma`，以便显式控制 accumulator fragment：

> Phase 3 的 S0 不是 Phase 1 的 CUTLASS K1；它是同一个 spill microkernel 的无 spill 版本。S0/S1/S2 只在 Phase 3 内部比较，不能把 Phase 3 的绝对可见率直接和 Phase 1/2 的 CUTLASS kernel 混在一起解释。

| 版本 | 行为 |
|---|---|
| S0 | accumulator 始终留在 RF，直到最终 epilogue |
| S1 | 在 K/2 处将 accumulator 写入 SMEM，再读回 RF，继续累加 |
| S2 | 每 K/8 做一次 SMEM 往返 |

要求：

- 三个版本使用同一套 HMMA 顺序和同一逻辑输出；
- SMEM 只做精确 FP32 存取，不做舍入；
- 使用正确的 fragment 线程分布、shared memory 布局和同步；
- 用调试模式在 spill 前后 dump accumulator fragment。

### 6.2 关键控制

1. **clean 一致性**：S0/S1/S2 的 clean 输出必须 bitwise 相同；
2. **pre-spill 状态一致**：同一 fault 在 spill 前 dump 的 bit pattern 必须相同；
3. **spill 前后对比**：记录 spill 前、reload 后、最终输出的错误状态；
4. **同一注入点**：同一逻辑输出元素、同一 K 位置、同一 bit；
5. **sham**：有 spill 但无 bit flip，输出必须等于 clean。

如果 S1/S2 的 clean 输出或 pre-spill 状态不一致，不能比较传播差异，必须先修实现。

### 6.3 注入时机

对 S1/S2：

- **before-spill**：在写入 SMEM 前的 HMMA 注入；
- **after-spill**：在 reload 后的 HMMA 注入；
- **final**：在最后一次 HMMA 后注入。

分别回答：

- 转存是否保留错误；
- 转存是否让错误更早/更晚被读取；
- 转存是否改变最终可见性；
- 如果错误在 spill 前可见、spill 后不可见，是 masking 还是实现问题。

### 6.4 样本量

```text
S0 + S1 + S2 = 3 × 135 = 405 个 HMMA fault
```

加上 spill 前后状态检查、clean、sham，预计约 450–500 个 run。

### 6.5 预期与解释边界

可能结果：

- **S1/S2 与 S0 无差异**：转存不是主要因素，支持零假设；
- **S1/S2 可见性下降**：转存/生命周期使错误被覆盖或延迟；
- **S1/S2 可见性上升**：转存改变了寄存器分配或消费者路径；
- **仅 K 顺序变化导致差异**：不能归因于转存，必须用 pre-spill 状态和 clean 差异排除。

> 这里的 SMEM 往返是人为增加的存储路径，不是 A100 内部 RF 物理实现的复现。

---

## 7. Phase 4（可选）：Qwen 形状与真实 stage

微基准通过后，再做两类外部有效性检查：

### 7.1 Qwen-like 形状

选择：

- QKV-V 形状：`M=2264, N=256, K=1536`；
- MLP-up 形状：`M=2264, N=8960, K=1536`。

用同一套 K1/K2/K3 逻辑，检查 tile/CTA 数量变化后结论是否稳定。

### 7.2 Qwen 真实 stage

在 Qwen2.5 prefill 的 `qkv_v` 或 `mlp_up` 上做少量 NVBitFI 验证：

- 同一逻辑输出元素/bit；
- 对比 HMMA 级和软件输出级；
- 不追求完整 AVF，只验证微基准发现的方向是否在真实 LLM stage 出现。

---

## 8. 指标与统计

### 8.1 单 run 指标

| 指标 | 定义 |
|---|---|
| `local_visible` | `F_C != C_clean`（bitwise） |
| `consumer_visible` | `F_Y != Y_clean` 或 top1 改变 |
| `rel_l2` | `||F_C-C_clean||_2 / ||C_clean||_2` |
| `max_abs` | 最大绝对误差 |
| `bit_mismatch_count` | 输出中 bit 不同的元素数 |
| `nan_inf` | 是否出现 NaN/Inf |
| `consumer_top1_changed` | 下游 top1 是否变化 |
| `dead` | 注入后读取前被覆盖 |
| `consumer_count` | 后续读取目标 register 的 HMMA 数 |
| `lifetime_hmma` | 注入到最后一次读取之间的动态 HMMA 数 |
| `reuse_distance` | 目标 register 被重新分配前的距离 |

### 8.2 聚合指标

```text
AVF_like(variant, bit, kpos) =
    consumer_visible_count / valid_injection_count
```

这是 instruction-level proxy，不是完整 AVF。

当前阶段的主分析是**分布**，不是 L0/L1 对比：

- `consumer_visible`、`consumer_masked`、`dead` 的组成；
- `rel_l2` / `max_abs` 的中位数、四分位数、尾部；
- 按 bit、K 位置、输出区域、HMMA 依赖类别分组；
- 按 kernel 变体分组的分布差异。

L0 软件输出注错暂不做。以后若需要 SVF 对照，再单独计算：

```text
SVF_visible = consumer_visible_count / output_flip_count
```

但该指标不进入当前主矩阵。

### 8.3 统计方法

- 以逻辑 site 为配对单位；
- 用 paired bootstrap 估计可见率差异的 95% CI；
- 用 mixed-effects logistic regression：

```text
visible ~ variant + bit + k_position + output_region
          + (1 | logical_site)
```

- 机制分析加入 `consumer_count`、`lifetime_hmma`、`dead`；
- 报告效应量和 CI，不只报告 p 值；
- 预先指定 primary outcome：**consumer_visible**；
- local_visible、rel_l2、dead 作为 secondary。

---

## 9. 控制与混杂因素

### 9.1 必须控制

| 因素 | 控制方式 |
|---|---|
| GPU 差异 | 固定 GPU UUID；配对实验在同一卡；4 卡只做重复 |
| 时钟/温度 | 锁定 clocks（若权限允许）；warmup；记录时钟 |
| 输入 | 固定 seed、固定 A/B、所有变体共用 |
| 编译 | 固定 CUTLASS/CUDA/编译器/flag；记录 SASS hash |
| 数值 | 关闭 TF32；固定 FP32 accumulate；不用 fast-math |
| 注入 | 每 run 只 1 bit；记录 before/after；sham 对照 |
| clean 差异 | 每个变体单独 clean；用 delta 而非跨变体绝对输出 |
| consumer | 固定 Wc、固定量化、固定 top1 规则 |
| 注入位置 | 同一逻辑坐标；物理位置记录并验证 |
| 顺序效应 | 随机化 run 顺序；重复 clean 插入 |

### 9.2 必须区分的失败

- **dead**：寄存器在读取前被覆盖；
- **rounding-masked**：C 改变，但 BF16 量化后消失；
- **consumer-masked**：C 改变，但 consumer 输出/top1 不变；
- **implementation bug**：spill 前后状态不一致、sham 改变输出；
- **clean drift**：无故障输出变化。

这些不能混成“fault masked”。

---

## 10. 预期图与交付物

### 10.1 一张主图

建议四个 panel：

- **(a)** K1/K2a/K2b/K3 的 consumer-visible rate，按 bit 分组；
- **(b)** local-visible 与 consumer-visible 的差异，展示 masking；
- **(c)** consumer_count / lifetime 与可见性的关系；
- **(d)** S0/S1/S2 的 spill 结果，按 before/after spill 分组。

### 10.2 交付物

| 交付物 | 内容 |
|---|---|
| 实验设计 | 本文件 |
| Kernel 变体 | K1/K2a/K2b/K3 + S0/S1/S2 |
| SASS 映射 | HMMA PC、寄存器、依赖、logical→physical 映射 |
| 注入日志 | 每个 run 的 before/after、bit、register、CTA/warp/lane |
| 结果 CSV | local/consumer visibility、误差、lifetime、dead |
| 分析脚本 | 统计和绘图 |
| 报告 | 参数、结果、机制证据、限制 |
| 图 | 一张 publication-style 四面板图 |

---

## 11. 时间与风险

### 11.1 粗略时间

| 阶段 | 预计 |
|---|---|
| Phase 0：kernel/SASS/NVBit 映射 | 2–4 天 |
| Phase 1：基线矩阵 | 2–3 天 |
| Phase 2：三个变体 | 3–5 天 |
| Phase 3：spill microkernel | 3–5 天 |
| Phase 4：Qwen 形状/真实 stage | 3–5 天 |
| 分析、图、报告 | 2–3 天 |

实际 run 时间取决于 NVBit 单次注入开销，必须先做 Phase 0 pilot 估算；不能直接按 clean GEMM 时间推算。

### 11.2 主要风险

1. **SASS→输出元素映射错误**：这是最大风险；Phase 0 必须先验证注入确实改变预期输出位置。
2. **Kernel 变体数值顺序变化**：clean 差异可能大于 fault 效应；必须单独量化。
3. **Spill 实现改变编译器调度**：即使 clean 一致，SASS 也可能变化；记录并解释。
4. **NVBit 只能测寄存器边界**：不能声称内部 adder tree 或完整 AVF。
5. **样本量不足**：可见率差异可能很小；pilot 后做功效分析。
6. **单形状外部有效性**：至少补 Qwen-like 形状。
7. **GPU 非确定性**：固定时钟/顺序，重复 clean 检查。

---

## 12. 结论与不应当声称的内容

### 12.1 可以回答

- 同一逻辑 GEMM 中，不同 HMMA 指令位置、bit、输出位置的故障影响如何分布；
- 同一逻辑 GEMM 的不同 kernel 实现，是否改变 HMMA 级故障的 consumer 可见性；
- 这种变化是否与 HMMA 消费者数量、寄存器 lifetime、dead 比例相关；
- 显式累加器 SMEM 转存是否改变故障传播；
- 哪些差异只是数值舍入顺序变化，哪些在控制后仍然存在。

### 12.2 不应声称

- 不能声称测到了 Tensor Core 内部真实 adder tree/partial sum 的 AVF；
- 不能把 instruction-level proxy 直接等同于论文的 architecture-level AVF；
- 不能仅凭 AVF 与 SVF 的总体差距就把主因归为累加器转存；
- 不能把人为 SMEM spill 说成 A100 RF 物理实现；
- 不能从单一 GEMM 形状直接外推到所有 LLM GEMM 和端到端 agent。

### 12.3 推荐的最小可发表闭环

1. K1 基线 + HMMA 影响分布；
2. K2a（仅 pipeline stage）与 K3（寄存器布局）；
3. S0/S1/S2 spill；
4. Qwen-like 形状复现；
5. 一张四面板图 + 机制相关性分析。

这个闭环足以回答“kernel 实现是否改变故障传播”，但不会过度声称完整 AVF。L0 软件注错只在后续需要 SVF 对照时补充。
