# Attention SDC 总结图

按用户要求，最终保留两张总结图。

## 文件

| 文件 | 内容 |
|---|---|
| `summary/token_count_distribution.pdf` / `.png` | 单 token vs 多 token：行是注入阶段，列是受影响 token 数分箱，颜色是该阶段 SDC 运行的比例 |
| `summary/golden_vs_delta_hexbin.pdf` / `.png` | 与 golden 的实际数值差距：横轴 `|Y_golden|`，纵轴 `|ΔY|`，按 5 个 stage 分面 |

## 当前 golden-vs-delta 图

- **已去掉 L4 阈值虚线**；
- 每个 stage 显示当前 representative run 中**所有有限、正的 `|ΔY|` 元素**；
- 不再只保留 L4 以上的点；
- 每个 cell 仍使用相对 L2 最大的代表 run；
- 零 `|ΔY|` 和非有限输出无法显示在 log 轴上，已在图脚注说明；
- 旧版带 L4 线的图保留在 `summary/archive/golden_vs_delta_hexbin_with_l4.*`。

## 保留的数据与脚本

- 数据：`../final_heatmaps_data_v2_nf/`
- 原始数据：`../final_heatmaps_data/`
- 当前绘图脚本：`scripts/attention_golden_delta_hexbin_all.py`
- 旧版绘图脚本：`scripts/attention_golden_delta_hexbin.py`
- 说明文档：`docs/HEATMAP_REDESIGN.md`
