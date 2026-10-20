# Attention SDC 总结图（保留两张）

按用户要求，只保留两张总结图；单 cell 图、论文版 105 张 PDF、5×3 overview 已删除。

## 文件

| 文件 | 内容 |
|---|---|
| `summary/token_count_distribution.pdf` / `.png` | 单 token vs 多 token：行是注入阶段，列是受影响 token 数分箱，颜色是该阶段 SDC 运行的比例 |
| `summary/golden_vs_delta_hexbin.pdf` / `.png` | 与 golden 的实际数值差距：横轴 `\|Y_golden\|`，纵轴 `\|ΔY\|`，虚线为 L4 边界；5 个 stage 分面板 |

## 读图

- 第一张回答“影响多少个 token”：QKV-Q、QK、PV 为 100% 单 token；QKV-K 中位 20；QKV-V 中位 2。
- 第二张回答“单个元素差多少”：Q/QK 主要在 1–10^4；K 量级相近但覆盖元素多约 40 倍；V/PV 可到 10^38，并呈离散指数带。
- 第二张使用每个 cell 相对误差最大的代表运行，不是全部 3150 次；NF 与 golden=0 已排除。

## 保留的数据与脚本

- 数据：`../final_heatmaps_data_v2_nf/`
- 原始数据：`../final_heatmaps_data/`
- 绘图脚本：`scripts/attention_token_count_heatmap.py`、`scripts/attention_golden_delta_hexbin.py`
- 说明文档：`docs/HEATMAP_REDESIGN.md`
