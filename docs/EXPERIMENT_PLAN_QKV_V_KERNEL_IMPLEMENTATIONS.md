# QKV-V 单 GEMM 的 Kernel 实现变体实验设计

日期：2026-10-22  
定位：**实验设计，尚未执行**  
范围：**只研究 qkv_v 一个逻辑 GEMM，不扩展到 9 个 stage**  
目标：在相同输入、权重、数值精度和下游 block 下，研究 kernel 实现是否改变 HMMA 指令级故障的传播与 SDC/Mask 分布。

---

## 1. 研究问题

固定：

```text
qkv_v: V = X_norm × W_v
X_norm: [2264, 1536]
W_v:    [1536, 256]
V:      [2264, 256]
BF16 输入
FP32 accumulate
BF16 输出
```

只改变 qkv_v 的 kernel implementation：

> 同一个逻辑 GEMM 的不同 kernel，在同一个 HMMA 单 bit 故障下，到达 attention 和 block 输出的概率与误差分布是否不同？

### 1.1 三个机制假设

| 机制 | 假设 | 主要操纵 |
|---|---|---|
| H1 指令调度 | 只改变 pipeline/Stages，不改变累加顺序时，消费者数量和可见性是否改变 | K2a |
| H2 累加器布局/生命周期 | CTA/warp/accumulator 布局改变寄存器 lifetime 和读取距离 | K3 |
| H3 数值顺序 + 传播 | K tile 改变累加分组，既改变 clean 数值也改变故障传播 | K2b |

**H1 是最干净的因果检验**：如果 K2a 与 K1 的 clean 输出 bitwise 相同，但 fault 可见性不同，就可以把差异归因于调度/生命周期，而不是数值舍入顺序。

---

## 2. Kernel 版本

| 版本 | 实现 | 目的 |
|---|---|---|
| **K0** | Qwen 当前实际 cuBLAS qkv_v kernel | 真实基线 |
| **K1** | 等价 CUTLASS 基线 | 可控基线 |
| **K2a** | 同 CTA/warp tile，只改 Stages/双缓冲 | 只测流水线调度 |
| **K2b** | 同 CTA/warp，改 K tile | 改变累加分组和数值顺序 |
| **K3** | 改 CTA tile / warp 数 / accumulator 布局 | 改变寄存器生命周期 |

建议配置：

| 版本 | CTA tile | Warp tile | K tile | Stages |
|---|---|---|---|---|
| K0 | 128×64（实际 cuBLAS） | 由 SASS 记录 | 由 kernel 记录 | 由 kernel 记录 |
| K1 | 128×64 | 64×32 | 32 | 3 |
| K2a | 128×64 | 64×32 | 32 | 5 |
| K2b | 128×64 | 64×32 | 64 | 3 |
| K3 | 128×128 | 64×64 | 32 | 3 |

具体 tile 以编译和 SASS 检查为准；所有版本必须输出相同的 `[2264,256]` BF16 V。

### 2.1 必要的 kernel 元数据

每个 variant 记录：

- kernel name；
- SASS hash；
- CTA tile、warp tile、K tile、Stages；
- CTAs、HMMA 总数；
- 寄存器数、shared memory、occupancy；
- 是否有编译器 spill；
- clean 输出 checksum。

### 2.2 K0 与 CUTLASS 的边界

K0 是 cuBLAS 实现，K1–K3 是 CUTLASS 实现。  
因此 K0 vs K1 的差异包含：

- 库实现差异；
- 可能的 epilogue/转换差异；
- 不同的累加顺序。

K0 作为“真实系统基线”，K1–K3 作为“可控实现变体”。  
不能把 K0 vs K1 的差异全部归因于调度。

---

## 3. 固定条件

| 项目 | 设置 |
|---|---|
| GPU | 单张 A100 80GB，固定 GPU UUID |
| 模型 | Qwen2.5-1.5B-Instruct |
| block | 只跑一个 decoder block |
| layer | 第一轮 layer 14；复现用 0/27 |
| prompt | 2,264-token Phase0 case |
| block 输入 | 从 clean prefill 捕获，固定 |
| dtype | BF16 输入，FP32 accumulate，BF16 输出 |
| attention | eager，固定 clean Q/K 产生的 P |
| Wo / MLP | 固定 clean 权重 |
| 输出 | V、attention（Wo 后）、block（MLP+residual） |
| 注入 | NVBitFI `mma`，HMMA destination register，1 bit/run |
| bit | 28 / 29 / 30 |
| 采样 | 随机 HMMA 站点，不做 early/middle/late 分层 |
| 判定 | 与各自 variant clean 做 bitwise 比较 |

### 3.1 每 variant 必须有自己的 clean

不同 kernel 的加法顺序不同，clean V 可能不同。  
因此：

```text
K0 fault 只能和 K0 clean 比
K1 fault 只能和 K1 clean 比
...
```

绝不能拿 K0 clean 去逐 bit 比较 K1/K2a/K2b/K3。

---

## 4. 实验设计

### 4.1 Phase A：Clean 与一致性验证

每个 variant：

```text
30 次 clean run
```

检查：

- 内部 bitwise 稳定性；
- 与 K1 的 clean 差异；
- K2a 是否与 K1 bitwise 相同；
- K2b/K3 的 clean 差异量级；
- SASS 是否真的不同；
- 是否存在 NaN/Inf。

通过标准：

- 每个 variant 30/30 自身一致；
- SASS 和 kernel 元数据记录完整；
- K2a 与 K1 的 clean 差异被明确量化。

> 如果 K2a 与 K1 clean 不相同，H1 不能作为纯调度实验解释，必须退化为“调度+数值顺序”的混合实验。

### 4.2 Phase B：随机站点 screen

第一轮只做 layer 14：

```text
1 layer × 5 variants × 3 bits × 15 random HMMA sites
= 225 个 fault run
```

每个 run：

- 一个 HMMA destination register；
- 一个 bit；
- 一次注入；
- 记录 V/attention/block；
- 统计 SDC/Mask/DUE。

目的：

- 验证每个 variant 可注入；
- 粗看 SDC/Mask 是否随 variant 改变；
- 估计方差和所需样本量；
- 避免直接铺满全矩阵。

### 4.3 Phase C：配对逻辑站点实验

随机站点的分布可能不同，因此 screen 只能回答“分布是否不同”，不能严格回答“同一个逻辑故障是否传播不同”。

选择可映射到所有 CUTLASS variants 的逻辑站点：

```text
15 个输出元素 × 3 个 K 区间 × 3 bits
= 135 个逻辑站点
```

对 K1/K2a/K2b/K3 做配对注入：

```text
4 variants × 135 = 540 个 fault run
```

如果 K0 也能可靠映射，再加 K0：

```text
5 variants × 135 = 675 个 fault run
```

每个逻辑站点定义：

```text
site = (output_i, output_j, K_segment, bit)
```

对每个 variant 映射到：

```text
cta, warp, lane, accumulator_register, HMMA PC
```

要求：

- 同一逻辑输出元素；
- 同一 K segment；
- 同一 bit；
- 每个 variant 用自己的 clean。

> 配对实验是判断“kernel 实现改变传播”的核心，不是随机 screen 的替代品。

### 4.4 Phase D：扩展统计

如果 screen 或配对实验显示 variant 之间有稳定差异，再扩到：

```text
3 layers × 5 variants × 3 bits × 45 random sites
= 2,025 个 fault run
```

第一轮不建议直接跑 2,025 点；先完成 225 screen + 540/675 paired。

---

## 5. 判定与指标

### 5.1 主判定

每个 variant 使用自己的 clean：

```text
runtime error → DUE
checkpoint 与 clean bitwise 不同 → SDC
checkpoint 与 clean bitwise 相同 → Mask
```

检查点：

- V；
- attention（Wo 后）；
- block（MLP+residual 后）。

主结果：

```text
P(SDC-block), P(Mask-block), P(DUE)
P(SDC-attn), P(Mask-attn)
P(SDC-V), P(Mask-V)
```

### 5.2 故障增量

对每个 fault：

```text
D_v = F_v - C_v
```

其中：

- `F_v` 是 variant v 的 fault 输出；
- `C_v` 是 variant v 的 clean 输出。

不要用 `F_v - C_K1` 作为主误差。

### 5.3 Clean 差异

记录：

```text
C_v - C_K1
```

用于判断：

- K2a 是否真的一致；
- K2b/K3 的差异是否只是数值顺序；
- fault 差异是否大于 clean 差异。

### 5.4 Kernel 机制指标

- HMMA 后目标 register 的 lifetime；
- 后续读取该 register 的 HMMA 数；
- dead-register 比例；
- register reuse distance；
- SASS 中 HMMA 顺序；
- 每个 variant 的 CTA/warp 数。

这些指标用于解释“为什么 visible rate 变了”，而不是只报 SDC 率。

---

## 6. 统计方法

### 6.1 比例

```text
P(SDC) = N_SDC / N_valid
P(Mask) = N_Mask / N_valid
P(DUE) = N_DUE / N_valid
```

给出 Wilson 95% CI。

### 6.2 Screen

- 每个 variant 的 SDC/Mask 比例；
- 5 个 variant 之间的描述性比较；
- 不把 15 点 screen 当成最终显著性结论。

### 6.3 配对实验

- 以 logical site 为配对单位；
- McNemar test：K2a vs K1、K3 vs K1；
- paired bootstrap 估计 visible rate 差异的 95% CI；
- mixed-effects logistic regression：

```text
visible ~ variant + bit + K_segment + (1 | logical_site)
```

- 报告 effect size 和 CI，不只报 p 值；
- 多重比较做校正。

### 6.4 预注册主结果

- Primary: `visible_block`；
- Secondary: `visible_attn`、`visible_V`、`rel_l2`、`max_abs`；
- Mechanism: lifetime、consumer count、clean diff。

---

## 7. 控制组

每个 variant：

```text
10 次 clean（正式统计前）
9 次 sham
3 次 dead-register
```

- clean：确认 variant 自身确定性；
- sham：DUMMY injector，不翻 bit，输出必须等于 clean；
- dead-register：注入到读取前被覆盖的寄存器，预期 Mask；
- invalid injection：单独记录，不进入分母。

控制组不进入 SDC/DUE/Mask 分母。

---

## 8. 主要混杂因素

| 混杂 | 处理 |
|---|---|
| K2b/K3 改变数值顺序 | 每个 variant 自己的 clean；报告 clean diff |
| K0 是 cuBLAS，K1–K3 是 CUTLASS | K0 只作为真实基线，不混入“同实现”比较 |
| 不同 CTA/warp 导致注入站点分布不同 | 增加 paired logical-site 实验 |
| 不同 HMMA 数量 | 用随机站点比例和 logical-site 配对两种设计 |
| GPU/时钟/温度 | 固定 GPU UUID；warmup；随机化 run 顺序 |
| NVBit 插桩影响 | sham 27 次；SASS hash；固定环境 |
| golden 漂移 | 每 variant 30 次 clean；bitwise gate |
| 统计功效不足 | screen 后估计方差，再决定 2,025 点是否必要 |

---

## 9. 执行顺序

1. 编译 K1/K2a/K2b/K3；
2. 检查 SASS，确认 tile/stage/寄存器布局确实不同；
3. 每个 variant 生成 30 次 clean，检查 bitwise 稳定；
4. 计算 K2a/K2b/K3 与 K1 的 clean 差异；
5. 跑 layer 14 的 225 点随机 screen；
6. 跑 135 个逻辑站点的 paired 实验（K1–K3，K0 可选）；
7. 统计 variant 之间的 SDC/Mask 差异；
8. 只有 screen 或 paired 有稳定差异时，才扩到 3 layers × 2,025 点；
9. 再决定是否做 spill/register lifetime 专项实验。

---

## 10. 预期结果与解释

### 可能结果 A

```text
K1 与 K2a clean bitwise 相同
但 K2a 的 SDC-block 明显不同
```

支持：pipeline/调度/寄存器 lifetime 影响故障传播，而不是数值顺序。

### 可能结果 B

```text
K1 与 K2a clean 相同
SDC/Mask 也相同
```

支持零假设：仅改变 pipeline 深度不改变传播。

### 可能结果 C

```text
K2b/K3 clean 不同
fault 差异与 clean 差异同量级
```

不能把差异归因于 kernel 调度；可能只是数值顺序变化。

### 可能结果 D

```text
paired logical-site 实验显示 K3 与 K1 差异稳定
且 lifetime/consumer count 与 visible 相关
```

支持 accumulator 布局/生命周期机制。

### 不能声称

- 不能声称测到了 Tensor Core 内部 adder tree 的真实 AVF；
- 不能把 K0 与 CUTLASS 的差异全部归因于 pipeline；
- 不能把 qkv_v 的结论外推到其他 GEMM stage；
- 不能把单 case/单 layer 的结果外推到完整 Qwen；
- 不能把“错误输出”算成 DUE。

---

## 11. 预期产物

- 5 个 kernel variant 的 SASS/profile 元数据；
- 每 variant 30 次 clean 记录；
- 225 点随机 screen CSV；
- 540/675 点 paired CSV；
- 每个逻辑站点的 physical mapping；
- SDC/Mask/DUE 表；
- clean diff 与 fault delta 表；
- 一张 variant × bit 的总结图；
- 报告：kernel implementation 是否改变 qkv_v 故障传播。

---

## 12. 一句话总结

本实验不是“换 9 个 stage”，而是：

> **固定 qkv_v 的逻辑 GEMM，用 K0 + K1/K2a/K2b/K3 五种 kernel implementation，先做 225 点随机 screen，再做 135 个逻辑站点的配对注入，并用每个 variant 自己的 clean 做 bitwise SDC/Mask 判定。**

---

## 13. 后续扩展路径

扩展顺序固定为：

```text
paired logical-site
→ layer replication
→ case replication
→ bit expansion
→ full-Qwen validation
```

每一步都要先通过 golden/sham/injection 的控制门槛，不能直接跳到大矩阵。

### 13.1 E1：配对逻辑站点

第一优先级，不是增加随机点，而是把“同一条逻辑故障”在 K0–K3 上对齐。

```text
4 CUTLASS variants × 135 logical sites
= 540 runs
```

K0 若可稳定映射，则：

```text
5 variants × 135
= 675 runs
```

这是判断“kernel implementation 是否改变传播”的主实验。

### 13.2 E2：layer replication

在 layer 14 结论稳定后，复制到 layer 0 / 27。

若使用配对站点：

```text
5 variants × 135 sites × 3 layers
= 2,025 runs
```

如果 K0 无法做逻辑映射，则只对 4 个 CUTLASS variants：

```text
4 variants × 135 sites × 3 layers
= 1,620 runs
```

这一阶段回答：

> 现象是否只在 middle layer 出现，还是 early/late layer 也存在。

### 13.3 E3：case replication

选择 3 个长上下文 case：

- policy QA；
- tool routing；
- API lookup。

对每个 case 重新捕获 block 输入和 clean 输出，再选一个通过 screen 的 variant 子集，例如 3 个 variant：

```text
3 cases × 3 variants × 3 layers × 3 bits × 15 sites
= 1,215 runs
```

这一阶段是 screen 级复现，不是最终全矩阵。  
若方向稳定，再选 1–2 个最关键 variant 扩到 45 sites。

### 13.4 E4：低 bit 扩展

当前只测 28–30；后续增加 24–27：

```text
4 bits × 3 layers × 3 variants × 15 sites
= 540 runs
```

若低 bit 出现明显不同的 SDC/Mask 分布，再扩到：

```text
4 bits × 3 layers × 3 variants × 45 sites
= 1,620 runs
```

低位更可能被 BF16 写回/舍入 mask，这一步用于验证“是否只有高位才危险”。

### 13.5 E5：完整 Qwen 验证

只有 block 级结论稳定后，才做：

- 完整 Qwen prefill；
- qkv_v kernel 的小样本 NVBitFI 注入；
- 不追求全矩阵，只验证 block 级方向是否在完整模型中出现；
- 仍不与 agent answer 混在一起。

### 13.6 扩展决策规则

| 当前结果 | 下一步 |
|---|---|
| K2a clean 与 K1 相同，但 SDC 分布不同 | 优先扩大 paired/layer 复现 |
| K2a clean 不同 | 不把差异归因于 pipeline，先分析数值顺序 |
| K3 差异与 lifetime 相关 | 增加 register lifetime / spill 专项 |
| 所有 variant 无差异 | 停止扩展 kernel 变体，转向其他机制 |
| 低 bit 全部 Mask | 只保留高位结论，不再扩低 bit |
| 某 case 方向不稳定 | 增加 case 数，不增加无关 variant |

### 13.7 不要做的扩展

- 不扩到 9 个 stage；
- 不直接铺 3,150 或更大随机矩阵；
- 不在 K0/K1 clean 不相同的情况下做跨 kernel 逐 bit 比较；
- 不在完整 Qwen 上重复 block 级实验；
- 不用容差阈值替代 bitwise 主判据。


