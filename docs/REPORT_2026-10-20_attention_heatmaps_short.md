# Attention SDC 注入实验记录

## 一、实验参数

- 实验编号：047_attn_5gemm_full
- 实验对象：causal self-attention forward（标准 attention，非 Flash/SDPA，无 KV cache）
- 模型规模：S = 128，D = 2048，H = 32，dh = 64
- 数据类型：BF16 输入/权重，FP32 softmax 与累加
- 注入阶段：
  - QKV-Q：Q 投影，Q = XWq
  - QKV-K：K 投影，K = XWk
  - QKV-V：V 投影，V = XWv
  - QK：score GEMM，QK^T
  - PV：P·V
- 故障位置：early / middle / late，指 kernel 动态指令流中的早/中/晚位置，不是 token 位置
- 故障位：bit 24–30
- 注入次数：每个 stage × bit × position 30 次，共 5 × 7 × 3 × 30 = 3150 次
- 故障模型：每次注入一个 bit flip，作用于 HMMA accumulator 寄存器
- head 选择：QK/PV 每次从 0–31 中随机选一个 head；QKV 投影级故障影响所有 head
- 判据：L4，|ΔY| > 1e-4 + 1e-3·|Y|
- 平台：node20，4×A100-80GB，sm_80
- 说明：本次复用已有 3150 次注入，未新增注入；从原始 Y 重新计算统计量

## 二、主要结果

### 1. SDC 率

| 阶段 | SDC / 630 | SDC 率 |
|---|---:|---:|
| QKV-Q | 343 | 54.4% |
| QKV-K | 399 | 63.3% |
| QKV-V | 346 | 54.9% |
| QK | 92 | 14.6% |
| PV | 327 | 51.9% |
| 总计 | 1507 / 3150 | 47.8% |

其他趋势：

- 注入位置：early 43.2%，middle 48.3%，late 52.0%
- bit：24 28.0%，25 39.3%，26 51.8%，27 58.2%，28 54.4%，29 57.8%，30 45.3%
- SDC 率排序：QKV-K（63.3%）> QKV-V（54.9%）≈ QKV-Q（54.4%）> PV（51.9%）> QK（14.6%）
- QK 的 SDC 率最低，但一旦发生 SDC，100% 只影响单个 token

### 2. 传播范围（只在 SDC 运行内统计）

| 阶段 | SDC / 630 | 中位受影响 token 数 | 1 个 | 2–3 个 | 4–7 个 | 8–15 个 | 16–31 个 | 32–63 个 | 64–128 个 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QKV-Q | 343 | 1 | 100% | 0 | 0 | 0 | 0 | 0 | 0 |
| QKV-K | 399 | 20 | 17% | 8.8% | 7.3% | 12% | 20% | 31% | 4.0% |
| QKV-V | 346 | 2 | 45% | 34% | 18% | 0.6% | 0 | 0 | 2.9% |
| QK | 92 | 1 | 100% | 0 | 0 | 0 | 0 | 0 | 0 |
| PV | 327 | 1 | 100% | 0 | 0 | 0 | 0 | 0 | 0 |

因四舍五入，行和可能略偏离 100%。

### 3. 非有限值

- 55 / 3150 次运行出现 NaN/Inf，分布在 11 个 cell
- 全部集中在 bit 30
- 非有限值按“受影响”计入统计
- QKV-V 的 64–128 档（2.9%）正好对应 bit 30 的 10 次 NaN/Inf 传播，不是有限误差的宽尾

## 三、结论

1. QKV-Q（Q = XWq）是严格单 token 影响。故障只污染被注入 query 的 Q 行，最终只改变该 query 的输出行。

2. QK（QK^T）是严格单 token 影响。故障只影响该 query 的 score 行，softmax 后只改变该 query 的输出。

3. PV 也是严格单 token 影响。故障只影响该 query 的 P·V 输出行。

4. QKV-K 是宽范围多 token 影响。K 行被污染后，所有 i ≥ j 的 query 都通过 causal attention 受影响；中位 20 个 token，31% 的 SDC 运行影响 32–63 个 token。

5. QKV-V 是中间态。V 行被污染后，所有关注该 key 的 query 都可能受影响，但只有少数超过 L4；中位 2 个 token，97% 的 SDC 运行影响 1–7 个 token。

6. bit 30 的 NaN/Inf 必须显式处理。如果只统计有限值，会出现“SDC 13/30 但热图为空”的矛盾；本次已将非有限值计入 affected，并在图中用黑色标出。

7. 表达“单 token 还是多 token”时，不能用按绝对 token 索引聚合的空间热图。Q/QK/PV 每次只影响 1 个 token，但 30 次注入落在不同 token 上，叠加后会看起来像多 token。应使用“受影响 token 数分布”热图：行是注入阶段，列是受影响 token 数分箱，颜色是该阶段 SDC 运行的比例。

8. 建议的图组合：
   - 主图：受影响 token 数分布热图（直接体现单 vs 多）
   - 辅图：stage × position 的空间 support 总览（体现影响落在哪些 token 行）

9. 对故障传播的含义：
   - Q/QK/PV 的错误是 query-local，影响范围最容易界定
   - K 的错误会沿 causal attention 向后续 query 扩散，是最需要注意的宽范围故障
   - V 的错误介于两者之间，且极端情况下会通过 NaN/Inf 放大到大量 token
