# QKV-V HMMA 指令级注错：统计到单 Transformer Block 输出

日期：2026-10-22  
定位：**当前主实验设计，尚未执行**  
目标：在单个 Qwen2.5 decoder block 的 `qkv_v` 逻辑 GEMM 上做 HMMA 指令级单 bit 注错，统计：

1. **attention 输出**；
2. **Transformer block 输出**；

的 SDC / DUE / Mask 分布。

> 当前不跑完整 Qwen，不跑 28 层模型，不做 LM head，不做 decode，不做最终 answer 判定。  
> 只跑一个 Transformer block，统计到 block 输出为止。  
> 不做软件输出注错；软件层只作为观察器。

---

## 1. 实验边界

### 1.1 只跑一个 Transformer block

使用 Qwen1.5B-Instruct 的一个 decoder layer：

```text
hidden_in [1,S,1536]
  │
  ├─ input RMSNorm
  ├─ qkv_q / qkv_k / qkv_v
  ├─ RoPE
  ├─ QK^T
  ├─ softmax
  ├─ P·V
  ├─ Wo
  ├─ residual
  ├─ post-attention RMSNorm
  ├─ gate / up / down
  └─ residual
hidden_out [1,S,1536]
```

只选择 layer 0 / 14 / 27 中某一层做一次 block forward，不串起 28 层，不接 LM head。

### 1.2 固定输入

从 Qwen1.5B clean prefill 中捕获：

- layer 0 / 14 / 27 的 block 输入 hidden；
- 每层对应的 qkv_v 权重；
- 每层 clean attention 的 Q/K/V 和 attention 概率；
- 每层 clean block 输出。

所有 fault run 共用同一份输入；不重新采样 prompt。

### 1.3 只注 qkv_v

```text
V = X_norm × W_v
X_norm: [S,1536]
W_v:    [1536,256]
V:      [S,256]
```

注入位置：

```text
qkv_v 的 HMMA destination register
```

不做 qkv_q、qkv_k、QK、PV、Wo、MLP 的注错。

---

## 2. 三个统计检查点

每个 fault run 记录三个输出检查点。

### C0：V 输出

```text
V_fault [1,S,256]
```

这是 qkv_v 的直接输出，用于判断 fault 是否到达 stage 输出。

### C1：attention 内部输出

```text
attn_context = P · V
```

- 形状：`[1,12,S,128]`；
- 在 `Wo` 之前；
- 使用 clean Q/K 产生的固定 attention 概率 `P`；
- 用来观察 V 错误经过 attention 加权后的变化。

### C2：attention 模块输出

```text
attn_module_output = Wo(attn_context)
```

- 形状：`[1,S,1536]`；
- 这是 self-attention 子模块的返回输出；
- 还没有加 residual；
- 这是“attention 出来”的主统计点。

### C3：Transformer block 输出

```text
block_output =
    hidden_in
    + attn_module_output
    + MLP(
        RMSNorm(
            hidden_in + attn_module_output
        )
      )
```

- 形状：`[1,S,1536]`；
- 包含 attention residual、post-attention RMSNorm、MLP gate/up/down、MLP residual；
- 这是本实验的最终统计点，不再往 LM head 走。

> 如果后续需要 attention 子层输出（含 residual），可以额外记录：
> `attn_residual = hidden_in + attn_module_output`。
> 但主表以 C2（Wo 后）和 C3（block 输出）为准。

---

## 3. HMMA 随机采样

### 3.1 采样单位

每个 run 注入一个：

```text
一个动态 HMMA 指令
+ 一个 lane
+ 一个 destination register
+ 一个 bit
```

记为：

```text
site = (layer, bit, hmma_dynamic_id, cta, warp, lane, register)
```

输出 V 坐标 `(output_i, output_j)` 由站点映射得到。

### 3.2 随机规则

- 从该层 qkv_v kernel 的全部有效 HMMA destination-register 站点中均匀随机采样；
- 每层、每个 bit 独立采样；
- 不放回；
- 固定 RNG seed；
- 先生成并保存 site manifest，再执行注入；
- **不按 early/middle/late 分层**；
- 不预先指定输出 channel/head；
- 不因为某个站点看起来会 mask 就替换。

### 3.3 矩阵

```text
3 layers × 3 bits × 45 random HMMA sites
= 405 个 HMMA fault run
```

每个 run：

- 只翻 1 个 bit；
- 只跑一个 block；
- 不做 decode；
- 不做 LM head；
- 记录 C0/C1/C2/C3。

### 3.4 dead register

随机站点可能落在：

- 注入后读取前被覆盖的 register；
- 只被 epilogue 读取的 register；
- 被后续 HMMA 继续累加的 register；
- 不影响 V 输出的 register。

这些都属于指令级影响分布，不能预先过滤。

主表分母：

```text
全部有效 HMMA 站点
```

辅助表：

```text
注入后 destination register 仍然 live 的站点
```

### 3.5 无效注入

预生成 10% reserve sites。  
invalid injection 单独记录，不进入 SDC/DUE/Mask 分母，用 reserve 补足到 405 个有效 run。

---

## 4. SDC / DUE / Mask 定义

### 4.1 主配置 D0：不启用软件检测器

DUE 只表示运行时失败：

```text
CUDA error / kernel abort / 非法访问 / trap / timeout
→ DUE-HW
```

否则：

```text
某个检查点输出与 clean 不同
→ 该检查点 SDC

某个检查点输出与 clean bitwise 相同
→ 该检查点 Mask
```

### 4.2 每个检查点分别统计

对 C0/C1/C2/C3 分别定义：

| 检查点 | SDC | Mask |
|---|---|---|
| C0 V | `SDC-V` | `Mask-V` |
| C1 attn_context | `SDC-attn-context` | `Mask-attn-context` |
| C2 attn_module_output | `SDC-attn` | `Mask-attn` |
| C3 block_output | `SDC-block` | `Mask-block` |

主结果：

```text
SDC-attn     attention 模块输出错误
SDC-block    Transformer block 输出错误
```

### 4.3 不应把错误答案算成 DUE

本实验没有最终 answer，因此：

- 不定义 `SDC-task`；
- 不把 `!!!!!!!!` 或错误文本算成 DUE；
- DUE 只有 CUDA/runtime error；
- NaN/Inf 如果没有触发运行时错误，是 `SDC-NaN`，不是 DUE。

### 4.4 D1 敏感性配置

可选启用 NaN/Inf 检测器：

```text
D1:
  运行时错误 → DUE-HW
  NaN/Inf 检测器触发 → DUE-DET
  正常但输出错误 → SDC
  正常且输出相同 → Mask
```

D1 单独列表，不混入 D0 主表。

---

## 5. 统计关系

### 5.1 单检查点

对 C2：

```text
N_valid = N_SDC-attn + N_DUE + N_Mask-attn
```

对 C3：

```text
N_valid = N_SDC-block + N_DUE + N_Mask-block
```

DUE 是全局 run 状态；如果运行失败，则该 run 在所有检查点都记为 DUE，不能计入 Mask。

### 5.2 传播矩阵

同时报告：

```text
V outcome × attention outcome × block outcome
```

例如：

| V | attention | block | 含义 |
|---|---|---|---|
| Mask | Mask | Mask | 完全被吸收 |
| SDC | SDC | Mask | 到 attention 仍错误，但 block 输出恢复 |
| SDC | Mask | Mask | V 变化但 attention 加权后消失 |
| SDC | SDC | SDC | 错误传播到 Transformer 输出 |
| DUE | DUE | DUE | 运行时失败 |

这张表比单一 SDC 率更重要。

### 5.3 误差幅度

对每个检查点记录：

```text
rel_l2
max_abs
bit_mismatch_count
NaN/Inf
```

按 layer、bit、实际 K 分位数、V channel/head、HMMA 依赖类别分组。

---

## 6. 统计维度

不再按 early/middle/late 采样。以下都是**事后分析变量**：

- layer：0 / 14 / 27；
- bit：28 / 29 / 30；
- 实际 K 位置：按随机站点的 `hmma_dynamic_id` 分位数分箱；
- V channel/head；
- HMMA 依赖类别：dead / accumulated / epilogue-only；
- 输出区域；
- 误差幅度。

主表不应该因为随机样本的 K 分布偏斜而重新分层。

---

## 7. 控制组

| 控制 | 数量 | 用途 |
|---|---:|---|
| clean | 每层 10 次 | 检查 block 输出确定性、checksum、时钟漂移 |
| sham | 每层 9 次，合计 27 | 安装注入路径但不翻 bit；C2/C3 必须等于 clean |
| dead-register | 每层 3 次，合计 9 | 注入到读取前被覆盖的 register；预期全 Mask |
| invalid | 单独记录 | 不进入分母 |

sham 也从随机 HMMA 站点中选，只是不翻 bit。

---

## 8. 数据记录

每个 run 至少记录：

| 字段 | 内容 |
|---|---|
| model | Qwen1.5B-Instruct |
| layer | 0 / 14 / 27 |
| stage | qkv_v |
| kernel | kernel name、CTA 数、SASS hash |
| shape | M=S, N=256, K=1536 |
| bit | 28/29/30 |
| hmma_dynamic_id | 目标动态 HMMA |
| k_index | 实际 K 位置（事后变量） |
| output_i, output_j | V 输出坐标 |
| cta/warp/lane/register | 物理站点 |
| dependency | D 是否等于 C、后续读取数、lifetime、dead |
| before/after | 十六进制寄存器值 |
| injection_ok | 是否真正注入、bitdiff=1 |
| V diff | C0 checksum / diff |
| attn_context diff | C1 diff |
| attn_output diff | C2 diff |
| block_output diff | C3 diff |
| runtime | CUDA error / timeout |
| detector | D1 是否触发 |
| outcome-V | SDC / Mask / DUE |
| outcome-attn | SDC / Mask / DUE |
| outcome-block | SDC / Mask / DUE |
| error | rel_l2、max_abs、bit mismatch、NaN/Inf |

---

## 9. 主图

一张图四个 panel：

- **(a)** C2 attention 与 C3 block 的 SDC/DUE/Mask 堆叠比例，按 layer/bit；
- **(b)** V → attention → block 的传播漏斗（Mask/SDC 转移）；
- **(c)** C2/C3 的误差幅度分布（rel_l2、max_abs）；
- **(d)** 实际 K 分位数与 SDC-block 的关系。

不画 early/middle/late 预设分层。

---

## 10. 执行顺序

1. 捕获 layer 0/14/27 的 block 输入、qkv_v 权重、clean C0/C1/C2/C3；
2. 验证单 block replay 与 Qwen 原始 block 输出 bitwise 一致；
3. 生成 405 个随机 HMMA site manifest；
4. Phase 0：先跑 30 个随机站点，验证注入、C2/C3 记录、SDC/Mask 分类；
5. 跑完 405 个 HMMA fault；
6. 统计 C2/C3 的 SDC/DUE/Mask；
7. 生成传播矩阵和主图。

---

## 11. 解释边界

### 可以回答

- qkv_v 的 HMMA 故障有多少到达 attention 输出；
- 有多少继续到达 Transformer block 输出；
- attention 输出可见但 block 输出被 mask 的比例；
- block 输出 SDC 与 layer/bit/实际 K/寄存器生命周期的关系；
- D0/D1 下 DUE 的差异。

### 不能声称

- 不能把 block 输出 SDC 率外推到完整 Qwen 或 agent answer；
- 不能把 `!!!!!!!!` 或错误文本算成 DUE；
- 不能把没有检测器的 NaN/Inf 算成 DUE；
- 不能声称测到 Tensor Core 内部 adder tree 的真实 AVF；
- 不能把 qkv_v 的结论直接外推到 Q/K/QK/PV/Wo/MLP。
