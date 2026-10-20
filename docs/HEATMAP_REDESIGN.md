# Attention SDC heatmaps: critique and publication redesign

> **Cleanup note (2026-10-20):** only the two summary figures
> (`summary/token_count_distribution.*` and
> `summary/golden_vs_delta_hexbin.*`) are retained.  The 105 per-cell
> figures, the 5 x 3 overview, the paper set, and their index were removed
> on user request; the data and plotting scripts remain for regeneration.

Status: **v2 figures generated locally from the `047_attn_5gemm_full`
campaign**.  The old `final_heatmaps/` figures are left untouched; all new
outputs live under `results/047_attn_5gemm_full/final_heatmaps_v2*/`.

## 1. What the v1 figures contained

Each v1 figure was a 17 x 9.5 in, 140 dpi dashboard with four elements:

1. mean `log10|Y_faulty - Y_golden|` over 30 runs,
2. support frequency over 30 runs,
3. a cumulative Token/Feature "impact ratio" curve,
4. a monospaced text panel with stage/head/position/bit/seq and summary stats.

The underlying campaign is a single-fault, single-GPU L4 study of the
attention forward pass (`S=128, D=2048, H=32`), 3150 injections over five
stages (`qkv_q`, `qkv_k`, `qkv_v`, `qk`, `pv`), bits 24-30, and three
instruction positions.

## 2. Problems found

### 2.1 Figure size, typography, and layout

* **Physical size was not a paper size.** 17 x 9.5 in at 140 dpi becomes
  ~2380 x 1330 px; at a 7 in column width the 9-12 pt labels shrink to
  ~4-5 pt.  A 300/600 dpi raster cannot compensate for text that is too
  small at the final physical size.
* **Large dead areas.**  The stats panel occupied the whole right column but
  only used its top third; the bottom-right quadrant was empty.
* **Redundant titles.**  The suptitle, both axes titles, and the stats panel
  repeated the same stage/bit/SDC/relL2 information.
* **No panel labels.**  Journals require `(a)`, `(b)`, ... for multi-panel
  figures; v1 had none.
* **Default sans-serif font.**  v1 used DejaVu Sans; paper body text is
  usually Times/Computer Modern.  The two should match.
* **Ticks at the last index** (`127`, `2047`) instead of the array extent
  (`128`, `2048`), which is inconsistent with `extent=(0, D, 0, S)`.

### 2.2 Colour and encoding

* **The support scale hid low counts.**  Support is a multiple of 1/30, but
  a linear 0-1 `magma` scale maps 1/30 to almost the same colour as 0.  In a
  typical `qkv_q` cell most affected elements are hit in 1-2 runs, so the
  footprint was nearly invisible.
* **Zero and missing were the same grey.**  v1 masked `support <= 0` and
  `NaN` to `#eeeeee`; "never affected", "not evaluated", and "zero" were
  visually identical.
* **The global error range wasted most of the colormap.**  The fixed
  `[-3, 6]` range was chosen for the extremes, but 75% of affected elements
  have `log10|dY|` between 1.5 and 2.1.  The mid-range was compressed into a
  narrow band.
* **The cumulative "impact ratio" was uninformative.**  For almost every
  cell the feature coverage reaches 100%, so the red curve is a diagonal
  from (0,0) to (1,1); it is a cumulative-coverage curve, not an impact
  measure, and the name overclaims.
* **The v1 error panel averaged all non-zero changes.**  This includes
  sub-threshold FP noise.  For broad stages such as `qkv_k`, almost every
  element has some non-zero change, so the panel became a uniform red block
  even when L4 was never crossed.  v2 averages only over L4-affected runs.

### 2.3 Head label

For `qk` and `pv`, the 30 runs in a cell use a **random head per run**.  The
v1 title printed `h0`, `h13`, ... taken from the representative run, which
reads as if all 30 runs used that head.  v2 never prints a head in the title;
it states "random head per run" in the footer, and the CSV index keeps the
representative head only as a provenance field.

### 2.4 Non-finite outputs (important)

At bit 30, some injections produce `NaN`/`Inf` in `Y`.  The classifier counts
these as SDC, but the v1 data builder skipped non-finite elements.  This
produced contradictions such as:

```
final_qkv_q_hx_early_bit30:  SDC 13/30 (43.3%)  ... but the map is empty
```

The 13 SDC runs each contain 2048 `NaN` values (one token row).  v2 now
counts non-finite outputs as affected, assigns them a sentinel
`log10|dY| = 40`, and overlays them in **black** on the error panel.  The
footer reports the number of runs with non-finite output (`NF k/30`).
Across the campaign, 55 of 3150 runs in 11 cells contain non-finite output.
Cells in which more than half of the 262144 output elements are non-finite
carry an in-panel note (e.g., "all output elements non-finite in 6/30 runs").

### 2.5 Export

v1 wrote only 140 dpi PNG.  Text in a PNG is raster; the figure cannot be
edited or scaled cleanly.  v2 writes **vector PDF** with rasterised heatmaps
(300 dpi inside the PDF) plus a 600 dpi PNG fallback, and embeds TrueType
fonts (`pdf.fonttype = 42`).

## 3. Academic-plotting skills surveyed

The redesign was checked against two public agent skills:

| skill | source | what was adopted |
|---|---|---|
| `paper-figure` | [wanshuiyin/Auto-claude-code-research-in-sleep](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/main/skills/paper-figure/SKILL.md) | no titles inside figures; serif font; 300 dpi / vector PDF; colourblind-safe; single/double-column widths; render-then-verify checklist |
| `scientific-visualization` | [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills/tree/main/skills/scientific-visualization) (MIT, v1.4) | truthful encoding guardrails; explicit missing vs zero vs non-finite; fixed scales across comparable panels; export/provenance checks |

Other candidates that are reasonable for future figures:
[davila7/claude-code-templates scientific-visualization](https://github.com/davila7/claude-code-templates/blob/main/cli-tool/components/skills/scientific/scientific-visualization/SKILL.md),
[peizhou/scholarPlotSkill](https://github.com/peizhou/scholarPlotSkill), and
[O0000-code/SSCI-Plots](https://github.com/O0000-code/SSCI-Plots).

No third-party skill was installed into the repository; the checklists were
applied directly, and the figure code remains self-contained.

## 4. v2 design

### 4.1 Data (v2, non-finite aware)

`scripts/attention_final_heatmaps_data_v2.py` (stdlib only; runs on node20)
writes, for every `(stage, bit, position)` cell:

```
support/<tag>.bin    float32  affected runs / n_runs
count/<tag>.bin      uint8    affected runs (0..n)
nfcount/<tag>.bin    uint8    runs with NaN/Inf output
meanlog/<tag>.bin    float32  mean log10|dY| over L4-affected runs
maxlog/<tag>.bin     float32  max  log10|dY| over L4-affected runs
```

`count` includes non-finite runs; `meanlog` uses the sentinel 40 for them.
The manifest adds `nf_runs`; `footprint.csv` adds per-run `nonfinite`.

Validation against v1: for every cell, `count >= round(v1_support * n)` and
`nfcount <= count`; the 105 tags match exactly.  In cells without
non-finite runs the finite support is unchanged.

### 4.2 Individual-cell figure

`scripts/attention_final_heatmaps_plot_v2.py` produces a 7.2 x 2.85 in
(double-column) two-panel figure:

* **(a) support** — discrete count 0-30, square-root colour stretch
  (`PowerNorm(gamma=0.5)`), `Blues`, fixed 0-30 range across all figures;
  `1/30` is visible, `30/30` is dark.
* **(b) error magnitude** — mean `log10|dY|` over L4-affected runs,
  `YlOrRd`, fixed `[0, 3]` (values above `10^3` clip to the top colour);
  never-affected cells are grey `#e9e9e9`; non-finite cells are solid black.
* 8 pt Times New Roman / STIX math, 7 pt ticks, panel labels `(a)/(b)`,
  no in-figure title, no stats panel, no cumulative-coverage curve.
* Footer (review mode only): stage, position, bit, n, SDC rate, NF runs,
  grey/black/clipping legend, and "random head per run" where applicable.
* PDF (vector text, 300 dpi heatmap) + 600 dpi PNG.

The **paper mode** (`--no-annotate`) removes the footer and uses a 7.2 x
2.65 in canvas for a LaTeX caption.

### 4.3 Campaign overview

`scripts/attention_overview_heatmaps_v2.py` produces a 7.2 x 4.9 in,
5 x 3 grid (stages x positions) of the support map, reduced over bits
24-30 with `max` (default).  This is the **spatial companion** to the
token-count summary: K projection faults fan out across token rows, while
Q/V/PV faults are query-local.  The shared vertical colourbar is fixed 0-30.

### 4.4 Token-count distribution (one-figure summary)

`scripts/attention_token_count_heatmap.py` produces
`summary/token_count_distribution.pdf/png`: a 5 x 7 heatmap whose rows are
stages and whose columns bin the number of affected output token rows per
SDC run (`1`, `2–3`, `4–7`, `8–15`, `16–31`, `32–63`, `64–128`).  Cells are
row-normalised over SDC runs and annotated with percentages; the left
labels give `SDC runs / 630` and the median token count.

This is the most direct single figure for the "single-token vs multi-token"
claim:

* QKV-Q, QK, PV: **100%** of SDC runs affect exactly one token row.
* QKV-K: median **20** rows; 31% affect 32-63 rows, 4.0% affect 64-128.
* QKV-V: median **2** rows; 45% affect 1, 34% affect 2-3, 18% affect 4-7.
  The 64-128 bin (2.9%) is exactly the ten bit-30 runs with non-finite
  output, not a finite-error tail.

### 4.5 Actual numeric gap to golden

`scripts/attention_golden_delta_hexbin.py` produces
`summary/golden_vs_delta_hexbin.pdf/png`: five log-log hexbin panels, one per
stage, with x = `|Y_golden|`, y = `|delta Y|`, and the L4 boundary
`|delta Y| = 1e-4 + 1e-3|Y|` overlaid as a dashed line.  This directly shows
the **actual magnitude** of the deviation, not just whether it crosses L4.

The current version uses the representative run of each cell (the run with
the largest relative L2 error), so it is a worst-case illustration rather
than an all-run distribution; the footer states this.  Non-finite outputs
and exactly zero golden values are excluded from the hexbin.

## 5. Outputs

| path | contents |
|---|---|
| `results/047_attn_5gemm_full/final_heatmaps_data_v2_nf/` | 105 cells x 5 arrays + `final_heatmaps_manifest.csv` + `footprint.csv` (368 MB) |
| `results/047_attn_5gemm_full/final_heatmaps_v2/` | 105 PDF + 105 PNG with review footer + `index_v2.csv` (40 MB) |
| `results/047_attn_5gemm_full/final_heatmaps_v2_paper/` | 105 PDF without footer, for the manuscript (6.9 MB) |
| `results/047_attn_5gemm_full/final_heatmaps_v2/overview/` | 5 x 3 overview, PDF + PNG |
| `results/047_attn_5gemm_full/final_heatmaps_v2/summary/` | token-count distribution heatmap + golden-vs-delta hexbin, PDF + PNG |

The old `final_heatmaps/`, `stage_heatmaps/`, and their scripts are not
modified.

## 6. Caption templates

Individual cell (paper mode):

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/final_qkv_k_hx_early_bit29.pdf}
  \caption{Spatial footprint of an L4 SDC in the attention output for the
  QKV-K projection (early instruction, bit 29; 30 injections). (a) Number of
  runs (out of 30) in which an output element exceeds the L4 tolerance
  $|\Delta Y| > 10^{-4} + 10^{-3}|Y|$; the colour scale is square-root
  stretched so that low counts remain visible. (b) Mean
  $\log_{10}|\Delta Y|$ over the runs in which the element is affected.
  Grey cells are never L4-affected and black cells are non-finite
  (NaN/Inf) outputs; magnitudes above $10^3$ are clipped to the top colour.}
  \label{fig:attn-qkvk-early-b29}
\end{figure}
```

For `qk`/`pv` cells, append: *The head is drawn uniformly at random for
each injection.*

Overview:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/overview_support_stage_position.pdf}
  \caption{L4 spatial footprint by stage and fault position. Each panel
  shows the maximum number of affected runs over bits 24-30 for the 30
  injections at that position. Non-finite outputs are counted as affected.
  QKV-K faults spread over many query rows through the causal softmax,
  whereas QKV-Q, QK, and PV faults are query-local; QKV-V is intermediate
  and becomes broad when a non-finite value propagates through the PV GEMM.}
  \label{fig:attn-overview}
\end{figure}
```

Token-count distribution:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/token_count_distribution.pdf}
  \caption{Distribution of the number of output token rows affected by a
  single L4 injection, conditional on the run being classified SDC.  Rows
  are injection stages and columns bin the affected token count; cells are
  row-normalised over SDC runs and the left labels give the number of SDC
  runs (out of 630) and the median.  QKV-Q is the $Q=XW_q$ projection and
  QK is the $QK^\top$ score GEMM.  QKV-Q, QK, and PV affect exactly one
  token row in 100\% of SDC runs.  QKV-K is broad (median 20; 31\% affect
  32--63 rows).  QKV-V is intermediate (median 2; 45\% affect one row and
  34\% affect two to three).  The 64--128 tail for QKV-V is the ten bit-30
  runs with non-finite output.}
  \label{fig:attn-token-count}
\end{figure}
```

Actual numeric gap:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/golden_vs_delta_hexbin.pdf}
  \caption{Actual numerical gap to golden.  Each panel is a log-log hexbin
  of all L4-affected elements in the representative run of every
  (stage, bit, position) cell (largest relative L2 per cell); the
  $x$-axis is $|Y_{\mathrm{golden}}|$ and the $y$-axis is
  $|\Delta Y|$.  The dashed line is the L4 boundary
  $|\Delta Y| = 10^{-4} + 10^{-3}|Y|$.  QKV-Q/QK/PV remain in the
  1--$10^4$ range, whereas QKV-V/PV reach $10^{20}$--$10^{38}$ through
  discrete catastrophic-error bands.  Non-finite outputs and exactly zero
  golden values are excluded; this is a worst-case representative sample,
  not an all-run distribution.}
  \label{fig:attn-golden-delta}
\end{figure}
```

## 7. Remaining caveats

* **L4 only.**  There is no L0-L5 sweep in `047`; the maps answer "where does
  an L4 SDC appear", not "how does the footprint shrink with tolerance".
* **One fault per run, one GPU, synthetic inputs.**  No decode/KV-cache or
  LLM replay.
* **Error clipping at `10^3`.**  Panel (b) is deliberately clipped so that
  the 1.5-2.1 range remains visible; the full magnitudes reach `10^38` in
  `qkv_v`/`pv`.  Use `maxlog` or a separate distribution figure if the tail
  is the message.
* **Non-finite outputs are counted as affected.**  They are black in panel
  (b) and included in the support count.  This is a change from v1 and is
  the reason some bit-30 cells now have non-empty maps.
* **Random heads.**  `qk`/`pv` cells aggregate 30 random heads; the `h` in
  the filename is the representative run's head and is provenance only.
* **Feature order is arbitrary.**  The 2048 feature columns have no semantic
  grouping; the maps show token-local vs token-broad structure, not
  feature-level structure.
* **Double-column width.**  At 7.2 in, 2048 features are below one printed
  pixel per column; the figure is intended for row-level footprint, not
  per-feature reading.  For per-feature inspection use the PDF zoom or
  aggregate features.

## 8. Reproduction

```bash
# on node20 (stdlib only)
python3 scripts/attention_final_heatmaps_data_v2.py \
    --exp results/047_attn_5gemm_full \
    --out results/047_attn_5gemm_full/final_heatmaps_data_v2_nf

# locally, after downloading the bins
python scripts/attention_final_heatmaps_plot_v2.py \
    --data results/047_attn_5gemm_full/final_heatmaps_data_v2_nf \
    --outdir results/047_attn_5gemm_full/final_heatmaps_v2 \
    --formats pdf,png --dpi 600

python scripts/attention_final_heatmaps_plot_v2.py \
    --data results/047_attn_5gemm_full/final_heatmaps_data_v2_nf \
    --outdir results/047_attn_5gemm_full/final_heatmaps_v2_paper \
    --formats pdf --no-annotate

python scripts/attention_overview_heatmaps_v2.py \
    --data results/047_attn_5gemm_full/final_heatmaps_data_v2_nf \
    --outdir results/047_attn_5gemm_full/final_heatmaps_v2/overview \
    --formats pdf,png
```

`scripts/attention_final_heatmaps_plot_v2.py --tags <tag1>,<tag2>` redraws
only selected cells and writes `index_v2_selected.csv`.
