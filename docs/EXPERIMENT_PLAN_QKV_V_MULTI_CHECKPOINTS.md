# QKV-V 单层注入 + 全链路多检查点实验设计

日期：2026-10-22  
定位：**实验设计，尚未执行**  
模型：Qwen2.5-1.5B-Instruct  
范围：只在 **layer 14** 的 `qkv_v` 注入；不扩展到 early/late layer；不扩展 bit 24–27。

---

## 1. 核心目标

固定 layer 14 的 qkv_v 逻辑 GEMM：

```text
V = X_norm × W_v
X_norm: [2264, 1536]
W_v:    [1536, 256]
V:      [2264, 256]
```

注入 qkv_v HMMA 的单个 bit 后，沿完整前向链路记录：

```text
C0  qkv_v GEMM 输出 V
C1  layer 14 attention 输出（Wo 后）
C2  layer 14 Transformer block 输出
C3  layer 27 最终 Transformer block 输出
```

然后比较：

```text
golden_clean
vs
fault_actual
```

在每个检查点分别记录数值差异和 SDC / DUE / Masked / crash。

---

## 2. 检查点定义

| 检查点 | 内容 | 形状 |
|---|---|---|
| **C0** | layer14 qkv_v 输出 V | `[1,2264,256]` |
| **C1** | layer14 attention 模块输出，Wo 后 | `[1,2264,1536]` |
| **C2** | layer14 block 输出，MLP + residual 后 | `[1,2264,1536]` |
| **C3** | layer27 最终 block 输出 | `[1,2264,1536]` |

C3 是 Qwen 最后一个 decoder layer 的输出；不接 final norm、LM head 或 decode。

### 2.1 全链路怎么得到 C3

采用两阶段 replay，避免对整个 28 层都做 NVBit 插桩：

```text
Stage A：
  layer14 qkv_v 注入
  → 捕获 C0、C1、C2
  → 保存 C2 fault hidden

Stage B：
  用 C2 fault hidden 作为输入
  → 依次运行 layer15 ... layer27
  → 捕获 C3
```

Downstream 的 layer15–27 使用 clean 权重，不再注入。

Golden 对应：

```text
clean C2
→ 同样运行 layer15 ... layer27
→ clean C3
```

这样：

- 注入成本仍然接近单 block；
- C3 仍反映完整下游传播；
- 不需要在 28 层上重复做 NVBit 注入。

---

## 3. 注入设置

| 项目 | 设置 |
|---|---|
| 注入层 | **只 layer 14** |
| 注入 stage | qkv_v |
| bit | 28 / 29 / 30 |
| 采样 | random HMMA destination-register site |
| 每个 run | 只翻 1 bit |
| 不注入 | layer 0 / 27 / 其它 stage |
| 不扩展 | bit 24–27 |

### 3.1 站点数量

当前 layer14 每个 bit 是 45 个站点：

```text
3 bits × 45 = 135
```

新计划适当增加：

**第一轮**

```text
3 bits × 60 random sites
= 180 个 fault run
```

**如果分布稳定**

```text
3 bits × 90 random sites
= 270 个 fault run
```

不增加 layer，只增加 layer14 内的随机站点。

### 3.2 随机规则

- 从 layer14 qkv_v kernel 的全部有效 HMMA destination-register 站点中均匀随机采样；
- 不放回；
- 固定 RNG seed；
- 先生成并保存 site manifest；
- 不做 early/middle/late 分层；
- 不预先指定输出 channel；
- 每个 run 只记录一次注入。

---

## 4. 多维度 golden vs actual 对比

每个 run 在 C0–C3 分别计算：

| 指标 | 含义 |
|---|---|
| `bitwise_equal` | 是否与 golden bitwise 完全相同 |
| `mismatch_count` | 有多少元素不同 |
| `max_abs` | 最大绝对误差 |
| `mean_abs` | 平均绝对误差 |
| `rel_l2` | 相对 L2 误差 |
| `max_rel` | 最大相对误差 |
| `nan_inf` | 是否出现 NaN/Inf |
| `finite_ratio` | 有限值比例 |

主表按检查点分开：

```text
C0 GEMM
C1 attention
C2 layer14 block
C3 layer27 block
```

不把四个检查点混成一个总误差。

### 4.1 比较对象

每个 variant/run 使用自己的 golden：

```text
golden_clean = 同一 layer14 输入、同一权重、同一 kernel、无故障运行
fault_actual = 同一配置下注入 1 bit 后的输出
```

不能拿 C0 的 clean 去比较 C3 的 fault。

### 4.2 传播矩阵

同时记录：

```text
C0 outcome
× C1 outcome
× C2 outcome
× C3 outcome
```

例如：

| C0 | C1 | C2 | C3 | 含义 |
|---|---|---|---|---|
| Masked | Masked | Masked | Masked | 全链路吸收 |
| SDC | SDC | Masked | Masked | attention 后恢复 |
| SDC | SDC | SDC | Masked | layer14 后恢复 |
| SDC | SDC | SDC | SDC | 传播到 layer27 |
| crash | — | — | — | 进程/GPU 失败 |

---

## 5. 分类：SDC / DUE / Masked / crash

每个检查点独立分类，同时给一个 run-level 总分类。

### 5.1 crash

```text
CUDA error
非法内存访问
kernel abort
进程崩溃
timeout
没有输出文件
```

→ **crash**

crash 与 DUE 分开记录。

### 5.2 DUE

DUE 是“检测到但无法纠正”的错误：

```text
显式 NaN/Inf 检测器触发
运行时检测到不可恢复错误
应用层 validator 触发并中止
```

→ **DUE-DET / DUE-HW**

没有检测器时，NaN/Inf 不自动算 DUE；需要单独报告 `SDC-NaN`。

### 5.3 SDC

```text
没有 crash
没有检测到错误
检查点输出与 golden bitwise 不同
```

→ **SDC**

包括：

- 数值错误但仍有限；
- `!!!!!!!!` 等乱码（如果到文本层，但本实验没有文本层）；
- NaN/Inf 但未启用检测器。

### 5.4 Masked

```text
检查点输出与 golden bitwise 完全相同
```

→ **Masked**

只要有一个 bit 不同，就不是 Masked。

### 5.5 检查点级和 run 级

每个检查点都给：

```text
C0: SDC / Masked / DUE / crash
C1: ...
C2: ...
C3: ...
```

Run-level 主分类以 **C3** 为准：

```text
C3 differs, no detection → SDC
C3 identical → Masked
detector/runtime → DUE
process fails → crash
```

C0–C2 用于解释传播路径。

---

## 6. 统计表

### 6.1 总表

| 检查点 | SDC | Masked | DUE | crash |
|---|---:|---:|---:|---:|
| C0 qkv_v |  |  |  |  |
| C1 attention |  |  |  |  |
| C2 layer14 block |  |  |  |  |
| C3 layer27 block |  |  |  |  |

### 6.2 按 bit

```text
bit 28 / 29 / 30
× C0/C1/C2/C3
× SDC/Masked/DUE/crash
```

### 6.3 数值分布

每个检查点：

- `rel_l2` 中位数/分位数；
- `max_abs` 中位数/分位数；
- `mismatch_count` 分布；
- NaN/Inf 数量。

---

## 7. 样本矩阵

### 第一轮

```text
layer14 × 3 bits × 60 random sites
= 180 个 fault run
```

### 扩展

```text
layer14 × 3 bits × 90 random sites
= 270 个 fault run
```

### 控制

```text
30 次 clean
27 次 sham
3 次 dead-register
```

控制组不进入 SDC/DUE/Masked/crash 分母。

### 成本

Stage A + Stage B 两阶段：

- 每 run 约一次单 block 注入 + 13 层 clean downstream；
- 180 点四卡预计约 20–40 分钟；
- 270 点四卡预计约 30–60 分钟；
- 实际以 pilot 为准。

---

## 8. 实现顺序

1. 固定 layer14 输入、W_v、position embeddings、mask；
2. 捕获 clean C0/C1/C2/C3；
3. 生成 layer14 qkv_v 随机 site manifest；
4. 跑 180 点 pilot；
5. 检查 C3 downstream replay 是否与完整 clean 路径 bitwise 一致；
6. 统计 C0–C3 的多维差异；
7. 若稳定，扩到 270 点；
8. 不扩展到 layer0/27 注入；
9. 不扩展 bit24–27。

---

## 9. 解释边界

### 可以回答

- 单个 qkv_v HMMA fault 是否改变 layer14 的 GEMM/attention/block；
- 它是否会传播到 layer27；
- 在哪一层被吸收；
- bit28/29/30 的差异；
- SDC/DUE/Masked/crash 的分布；
- C0–C3 的数值差大小。

### 不能声称

- 不能把 layer14 的结论外推到所有层；
- 不能外推到其它 GEMM stage；
- 不能把 NaN/Inf 自动算成 DUE；
- 不能把 crash 和 DUE 混为一类；
- 不能声称测到了 Tensor Core 内部 adder tree 的真实 AVF；
- 不能用 C3 代替完整 Qwen 的最终 answer。

---

## 10. 一句话总结

> **只在 layer14 qkv_v 注入，bit28–30，站点数从 45/bit 增到 60–90/bit；记录 C0 GEMM、C1 attention、C2 layer14 block、C3 layer27 block 四个检查点的 golden-vs-actual 数值差，并分别分类 SDC / DUE / Masked / crash。**
