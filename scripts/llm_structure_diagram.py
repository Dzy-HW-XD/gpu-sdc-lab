#!/usr/bin/env python3
"""Schematic: LLM block structure and injection points (prefill)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

INK = "0.15"
W_COL = "#D55E00"   # weight injection
A_COL = "#0072B2"   # activation injection
N_COL = "#666666"   # norm injection
R_COL = "#009E73"   # residual
PY_COL = "#0072B2"
QW_COL = "#009E73"


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def box(ax, x, y, w, h, text, fc="#eaf2fb", ec="#4a7fb5", fs=6.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.07",
                       facecolor=fc, edgecolor=ec, linewidth=0.7, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.2)
    return (x + w / 2, y + h / 2)


def arrow(ax, x1, y1, x2, y2, color="0.3", lw=0.8, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=5.5, linewidth=lw, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle="arc3,rad=%s" % rad, zorder=1))


def mark(ax, x, y, letter, color, fs=5.2):
    ax.add_patch(Circle((x, y), 0.17, facecolor=color, edgecolor="white",
                        linewidth=0.5, zorder=10))
    ax.text(x, y, letter, color="white", ha="center", va="center",
            fontsize=fs, fontweight="bold", zorder=11)


def base(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")


def pythia(ax):
    base(ax)
    ax.set_title("Pythia / GPT-NeoX: parallel residual, MHA", fontsize=8, pad=3)
    # residual stream
    ax.plot([0.3, 9.7], [1.1, 1.1], color="0.3", lw=1.2, zorder=1)
    arrow(ax, 0.3, 0.45, 0.3, 1.1)
    arrow(ax, 9.7, 1.1, 9.9, 1.1)
    ax.text(0.3, 0.18, r"$x_{\mathrm{in}}$", ha="center", fontsize=6)
    ax.text(9.7, 0.72, r"$x_{\mathrm{out}}$", ha="center", fontsize=6)
    ax.text(5.0, 1.55, r"residual stream  (repeat $\times N$)", ha="center",
            fontsize=6, color="0.35")

    # shared LN + branches
    box(ax, 0.9, 5.2, 1.2, 0.7, "LayerNorm", fs=6)
    arrow(ax, 1.5, 1.1, 1.5, 5.2)
    box(ax, 2.9, 6.2, 1.6, 0.75, "W$_{QKV}$\n(Q,K,V)", fc="#fdece1", ec=W_COL)
    box(ax, 4.9, 6.2, 1.8, 0.75, "RoPE + MHA\n12/16/8 heads", fs=5.6)
    box(ax, 7.2, 6.2, 0.9, 0.75, "W$_O$", fc="#fdece1", ec=W_COL)
    arrow(ax, 2.1, 5.9, 2.9, 6.55)
    arrow(ax, 4.5, 6.55, 4.9, 6.55)
    arrow(ax, 6.7, 6.55, 7.2, 6.55)
    arrow(ax, 7.65, 6.2, 7.65, 1.1)

    box(ax, 2.9, 3.7, 1.5, 0.75, "W$_{up}$", fc="#fdece1", ec=W_COL)
    box(ax, 4.8, 3.7, 0.9, 0.75, "GELU", fs=6)
    box(ax, 6.1, 3.7, 1.3, 0.75, "W$_{down}$", fc="#fdece1", ec=W_COL)
    arrow(ax, 2.1, 5.2, 2.9, 4.1)
    arrow(ax, 4.4, 4.08, 4.8, 4.08)
    arrow(ax, 5.7, 4.08, 6.1, 4.08)
    arrow(ax, 6.75, 3.7, 6.75, 1.1)

    ax.text(8.0, 2.9, "+", fontsize=9, color="0.3")
    ax.text(7.1, 1.9, "+", fontsize=9, color="0.3")

    # injection markers
    mark(ax, 2.55, 7.15, "W", W_COL)
    mark(ax, 7.95, 7.15, "W", W_COL)
    mark(ax, 2.55, 3.15, "W", W_COL)
    mark(ax, 7.0, 3.15, "W", W_COL)
    mark(ax, 7.65, 4.1, "A", A_COL)
    mark(ax, 6.75, 2.25, "A", A_COL)
    mark(ax, 4.6, 1.1, "R", R_COL)
    mark(ax, 1.5, 4.75, "N", N_COL)

    ax.text(4.5, 0.35, r"inject at layer $\ell$, token $p$", ha="center",
            fontsize=6, color="0.35")


def qwen(ax):
    base(ax)
    ax.set_title("Qwen2.5: sequential pre-norm, GQA, SwiGLU", fontsize=8, pad=3)
    # top attention sublayer
    ax.plot([0.3, 9.7], [6.8, 6.8], color="0.3", lw=1.2, zorder=1)
    arrow(ax, 0.3, 6.2, 0.3, 6.8)
    arrow(ax, 9.7, 6.8, 9.9, 6.8)
    ax.text(0.3, 5.95, r"$x$", ha="center", fontsize=6)
    box(ax, 0.9, 7.9, 1.2, 0.65, "RMSNorm", fs=6)
    box(ax, 2.7, 7.9, 1.7, 0.65, "W$_Q$, W$_K$, W$_V$\nGQA 14Q / 2KV", fs=5.3,
        fc="#e8f6f1", ec=QW_COL)
    box(ax, 4.9, 7.9, 1.6, 0.65, "RoPE +\nattention", fs=5.4)
    box(ax, 7.0, 7.9, 0.8, 0.65, "W$_O$", fc="#e8f6f1", ec=QW_COL)
    arrow(ax, 1.5, 6.8, 1.5, 7.9)
    arrow(ax, 2.1, 8.22, 2.7, 8.22)
    arrow(ax, 4.4, 8.22, 4.9, 8.22)
    arrow(ax, 6.5, 8.22, 7.0, 8.22)
    arrow(ax, 7.4, 7.9, 7.4, 6.8)
    ax.text(7.75, 7.3, "+", fontsize=9, color="0.3")

    # sequential arrow
    arrow(ax, 5.0, 6.0, 5.0, 5.2, color="0.45", lw=1.0)
    ax.text(5.2, 5.6, "sequential", fontsize=5.8, color="0.4")

    # bottom MLP sublayer
    ax.plot([0.3, 9.7], [3.1, 3.1], color="0.3", lw=1.2, zorder=1)
    arrow(ax, 0.3, 2.5, 0.3, 3.1)
    arrow(ax, 9.7, 3.1, 9.9, 3.1)
    ax.text(0.3, 2.25, r"$x$", ha="center", fontsize=6)
    box(ax, 0.9, 4.2, 1.2, 0.65, "RMSNorm", fs=6)
    box(ax, 2.7, 4.2, 1.7, 0.65, "W$_{gate}$, W$_{up}$", fs=5.8,
        fc="#e8f6f1", ec=QW_COL)
    box(ax, 4.9, 4.2, 0.9, 0.65, "SiLU", fs=6)
    box(ax, 6.0, 4.2, 1.1, 0.65, "W$_{down}$", fc="#e8f6f1", ec=QW_COL)
    arrow(ax, 1.5, 3.1, 1.5, 4.2)
    arrow(ax, 2.1, 4.52, 2.7, 4.52)
    arrow(ax, 4.4, 4.52, 4.9, 4.52)
    arrow(ax, 5.8, 4.52, 6.0, 4.52)
    arrow(ax, 6.55, 4.2, 6.55, 3.1)
    ax.text(6.9, 3.65, "+", fontsize=9, color="0.3")

    # injection markers
    mark(ax, 3.0, 8.9, "W", W_COL)
    mark(ax, 7.25, 8.9, "W", W_COL)
    mark(ax, 7.4, 7.35, "A", A_COL)
    mark(ax, 1.5, 7.6, "N", N_COL)
    mark(ax, 3.0, 5.2, "W", W_COL)
    mark(ax, 6.35, 5.2, "W", W_COL)
    mark(ax, 6.55, 3.6, "A", A_COL)
    mark(ax, 1.5, 3.9, "N", N_COL)
    mark(ax, 5.0, 6.8, "R", R_COL)
    ax.text(4.5, 0.35, r"inject at layer $\ell$, token $p$", ha="center",
            fontsize=6, color="0.35")


def layer_stack(ax):
    models = [
        ("Pythia-160M", 12, "d=768, H=12", PY_COL, "MHA / parallel"),
        ("Pythia-410M", 24, "d=1024, H=16", PY_COL, "MHA / parallel"),
        ("Pythia-1B", 16, "d=2048, H=8, hd=256", PY_COL, "MHA / parallel"),
        ("Qwen2.5-0.5B", 24, "d=896, 14Q/2KV", QW_COL, "GQA / sequential"),
        ("Qwen2.5-1.5B", 28, "d=1536, 12Q/2KV", QW_COL, "GQA / sequential"),
    ]
    ax.set_xlim(0, 5.3)
    ax.set_ylim(0, 31)
    for i, (name, n, cfg, col, note) in enumerate(models):
        x = i + 0.5
        ax.add_patch(Rectangle((x - 0.15, 0), 0.30, n, facecolor=col,
                               alpha=0.78, edgecolor="0.25", linewidth=0.5))
        # layer ticks
        for y in range(n + 1):
            ax.plot([x - 0.15, x + 0.15], [y, y], color="white", lw=0.25,
                    alpha=0.55)
        # early/middle/late markers
        for label, y, c in (("E", 0.5, "#D55E00"), ("M", n / 2.0, "#E69F00"),
                            ("L", n - 0.5, "#CC79A7")):
            ax.plot([x + 0.24], [y], marker="o", markersize=3.0, color=c,
                    markeredgecolor="white", markeredgewidth=0.3,
                    linestyle="None", zorder=5)
            ax.text(x + 0.36, y, label, fontsize=5, va="center", color="0.3")
        ax.text(x, -1.5, name, ha="center", va="top", fontsize=6)
        ax.text(x, -2.5, cfg, ha="center", va="top", fontsize=5.2, color="0.35")
        ax.text(x, -3.5, note, ha="center", va="top", fontsize=5.2, color="0.35")
    ax.set_ylabel("layer index")
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_xticks([])
    ax.set_xlim(0, 5.3)
    ax.set_ylim(0, 30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.text(0.1, 28.8, "injection depth:", fontsize=6, color="0.35")
    for label, x, c in (("E", 1.1, "#D55E00"), ("M", 1.7, "#E69F00"),
                        ("L", 2.3, "#CC79A7")):
        ax.plot([x], [28.2], marker="o", markersize=3.0, color=c,
                markeredgecolor="white", markeredgewidth=0.3,
                linestyle="None")
        ax.text(x + 0.12, 28.2, label, fontsize=5, va="center", color="0.3")


def flow(ax):
    """Horizontal prefill flow for Pythia and Qwen2.5."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    lanes = [
        (1.55, "Pythia",
         r"160M 12L  ·  410M 24L  ·  1B 16L  ·  MHA  ·  parallel branches",
         PY_COL, "Block 0\n(E)", "Block N/2\n(M)", "Block N-1\n(L)"),
        (0.55, "Qwen2.5",
         r"0.5B 24L  ·  1.5B 28L  ·  GQA  ·  sequential sublayers",
         QW_COL, "Block 0\n(E)", "Block N/2\n(M)", "Block N-1\n(L)"),
    ]
    for y, label, note, col, b0, bm, bl in lanes:
        ax.text(0.04, y, label, rotation=90, ha="center", va="center",
                fontsize=7, color=col)
        items = [
            (0.35, 0.55, "tokens", None),
            (1.05, 0.65, "Embed", None),
            (1.95, 0.82, b0, "#D55E00"),
            (3.00, 0.72, "Block 1", None),
            (3.95, 0.32, r"$\cdots$", None),
            (4.42, 0.88, bm, "#E69F00"),
            (5.48, 0.32, r"$\cdots$", None),
            (6.02, 0.88, bl, "#CC79A7"),
            (7.12, 0.72, "Final\nNorm", None),
            (8.05, 0.52, "Logits", None),
        ]
        centers = []
        for x, w, text, ec in items:
            if ec is None:
                ax.text(x + w / 2, y, text, ha="center", va="center",
                        fontsize=5.8, color="0.2",
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                  edgecolor="0.55", linewidth=0.6))
            else:
                p = FancyBboxPatch((x, y - 0.19), w, 0.38,
                                   boxstyle="round,pad=0.02,rounding_size=0.06",
                                   facecolor="white", edgecolor=ec,
                                   linewidth=1.0, zorder=3)
                ax.add_patch(p)
                ax.text(x + w / 2, y, text, ha="center", va="center",
                        fontsize=5.3, color="0.2", zorder=4, linespacing=1.1)
            centers.append(x + w / 2)
        for i in range(len(items) - 1):
            x1, w1 = items[i][0], items[i][1]
            x2 = items[i + 1][0]
            arrow(ax, x1 + w1 + 0.03, y, x2 - 0.03, y, color="0.35", lw=0.8)
        # residual stream
        ax.annotate("", xy=(7.15, y - 0.34), xytext=(1.95, y - 0.34),
                    arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.9,
                                    shrinkA=0, shrinkB=0), zorder=1)
        ax.text(4.55, y - 0.46, "residual stream", ha="center", fontsize=5.2,
                color="0.35")
        ax.text(4.75, y + 0.35, note, ha="center", fontsize=5.2, color="0.4")
    ax.text(5.0, 0.02,
            r"inject at layer $\ell$, token $p$; causal suffix $p \ldots S-1$ only",
            ha="center", va="bottom", fontsize=5.8, color="0.4")


def main():
    setup()
    fig = plt.figure(figsize=(7.2, 8.6))
    ax0 = fig.add_axes([0.06, 0.57, 0.90, 0.34])
    ax1 = fig.add_axes([0.035, 0.05, 0.45, 0.48])
    ax2 = fig.add_axes([0.515, 0.05, 0.45, 0.48])
    flow(ax0)
    pythia(ax1)
    qwen(ax2)

    for ax, tag in ((ax0, "(a)"), (ax1, "(b)"), (ax2, "(c)")):
        ax.text(0.0, 1.02, tag, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8, fontweight="bold")

    fig.text(0.5, 0.975, "LLM block structure and injection points (prefill)",
             ha="center", va="top", fontsize=9)
    fig.text(0.5, 0.025,
             "W/A/N/R = weight/activation/norm/residual injection;  "
             "E/M/L = early/middle/late depth.  "
             "KV cache is decode-only and not part of this prefill figure.",
             ha="center", va="bottom", fontsize=6.0, color="0.35")

    out = "docs/figures"
    os.makedirs(out, exist_ok=True)
    for fmt in ("pdf", "png"):
        path = os.path.join(out, "llm_structure_flow." + fmt)
        fig.savefig(path, dpi=600 if fmt == "png" else 300)
        print("wrote", path)
    plt.close(fig)



if __name__ == "__main__":
    main()
