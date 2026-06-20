# -*- coding: utf-8 -*-
"""读取 4 个 report_multi_*.json，逐 model/data 组合产出（幻灯片级，Palatino 字体）：
- 含误差的相似度矩阵：渲染成图片 figures/*_table.png（并存 tables/*.csv）
- Fig1 横向 error-bar：以 Taffy_composite 为基准，灰带=判断阈值(Echo/Fries/Taffy vs composite, μ±σ)
- Fig2 纵向 error-bar：以 Echo/Fries/Taffy 各为基准，每列灰带=判断阈值(composite vs persona, ±1σ)
数值标注：横图统一在点上方，竖图统一在点右侧。speaker/baseline/Model·Data 加粗。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "Palatino", "serif"],
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
})

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
TBL = os.path.join(ROOT, "tables"); os.makedirs(TBL, exist_ok=True)

COMBOS = [("ecapa", "audio", "ECAPA", "raw"),
          ("ecapa", "audio_vocals", "ECAPA", "vocals"),
          ("campp", "audio", "CAM++", "raw"),
          ("campp", "audio_vocals", "CAM++", "vocals")]
COMP = "Taffy_composite"
REFS = ["Echo", "Fries", "Taffy"]
PROBES = ["Cui", "Monaka", "Miya"]
COL = {"Cui": "#1D9E75", "Monaka": "#D85A30", "Miya": "#6A3FA0"}   # Miya 改紫，避开灰


def load(model, tag):
    with open(os.path.join(ROOT, "results", f"report_multi_{model}_{tag}.json"), encoding="utf-8") as f:
        r = json.load(f)
    ents = r["entities"]
    mean = np.array(r["mean"]); std = np.array(r["std"])
    ix = {e: i for i, e in enumerate(ents)}
    return ents, mean, std, ix


def m_of(mean, std, ix, a, b):
    return mean[ix[a], ix[b]], std[ix[a], ix[b]]


def render_table(model, data, fname, ents, mean, std):
    n = len(ents)
    sh = {e: ("T_comp" if e == COMP else e) for e in ents}
    cmap = mpl.colormaps["Blues"]
    fig, ax = plt.subplots(figsize=(1.25 * (n + 1) + 0.5, 0.62 * (n + 1) + 0.9))
    ax.set_xlim(0, n + 1); ax.set_ylim(0, n + 1); ax.axis("off"); ax.invert_yaxis()
    for j, e in enumerate(ents):                       # 表头（加粗 speaker 名）
        ax.text(j + 1.5, 0.5, sh[e], ha="center", va="center", fontweight="bold", fontsize=12.5)
        ax.text(0.5, j + 1.5, sh[e], ha="center", va="center", fontweight="bold", fontsize=12.5)
    for i in range(n):
        for j in range(n):
            m, s = mean[i, j], std[i, j]
            if i == j:
                fc, txt = "#ECECEC", "—"
            else:
                fc = cmap(0.04 + (np.clip(m, 0.5, 1.0) - 0.5) / 0.5 * 0.42)   # 淡蓝，越像越深
                txt = f"{m:.2f}±{s:.2f}"
            ax.add_patch(Rectangle((j + 1, i + 1), 1, 1, facecolor=fc, edgecolor="white", lw=1.4))
            ax.text(j + 1.5, i + 1.5, txt, ha="center", va="center", fontsize=11)
    ax.set_title(f"{model} · {data}   —   similarity matrix (mean ± std)",
                 fontweight="bold", fontsize=14, pad=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), bbox_inches="tight"); plt.close(fig)
    # CSV
    rows = [[""] + ents] + [[ents[i]] + [f"{mean[i, j]:.3f}±{std[i, j]:.3f}" for j in range(n)] for i in range(n)]
    with open(os.path.join(TBL, f"{model.replace('+','p')}_{data}.csv"), "w", encoding="utf-8-sig") as f:
        f.write("\n".join(",".join(r) for r in rows))


def fig1(model, data, fname, mean, std, ix):
    band = [m_of(mean, std, ix, r, COMP)[0] for r in REFS]
    mu, sd = float(np.mean(band)), float(np.std(band))
    order = PROBES[::-1]                                # 自下而上 → Cui 在顶
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.axvspan(mu - sd, mu + sd, color="0.6", alpha=0.28, zorder=0)
    ax.axvline(mu, color="0.4", ls="--", lw=1.1, zorder=1)
    for y, p in enumerate(order):
        m, s = m_of(mean, std, ix, p, COMP)
        ax.errorbar(m, y, xerr=s, fmt="o", color=COL[p], ms=9, lw=2.4, capsize=5, zorder=3)
        ax.annotate(f"{m:.3f}±{s:.3f}", (m, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold", color=COL[p])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontweight="bold")
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Cosine similarity  to  Taffy_composite")
    ax.set_title(f"{model} · {data}      baseline = Taffy_composite", fontweight="bold")
    ax.grid(axis="x", color="0.9", lw=0.6)
    lo = min(mu - sd, min(m_of(mean, std, ix, p, COMP)[0] - m_of(mean, std, ix, p, COMP)[1] for p in PROBES))
    ax.set_xlim(lo - 0.05, max(mu + sd, 1.0) + 0.02)
    ax.legend(handles=[Patch(facecolor="0.6", alpha=0.28, label="judgment threshold")],
              loc="lower left", frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)


def fig2(model, data, fname, mean, std, ix):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    off = {"Cui": -0.24, "Monaka": 0.0, "Miya": 0.24}
    for j, r in enumerate(REFS):
        cm, cs = m_of(mean, std, ix, COMP, r)
        ax.add_patch(Rectangle((j - 0.42, cm - cs), 0.84, 2 * cs, color="0.6", alpha=0.28, zorder=0))
        ax.plot([j - 0.42, j + 0.42], [cm, cm], color="0.4", ls="--", lw=1.1, zorder=1)
        for p in PROBES:
            m, s = m_of(mean, std, ix, p, r)
            x = j + off[p]
            ax.errorbar(x, m, yerr=s, fmt="o", color=COL[p], ms=8, lw=2.4, capsize=5, zorder=3)
            ax.annotate(f"{m:.2f}", (x, m), textcoords="offset points",
                        xytext=(9, 0), va="center", ha="left", fontsize=9, fontweight="bold", color=COL[p])
    ax.set_xticks(range(len(REFS)))
    ax.set_xticklabels(REFS, fontweight="bold")
    ax.set_xlim(-0.6, len(REFS) - 0.4)
    ax.set_xlabel("baseline")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"{model} · {data}", fontweight="bold")
    ax.grid(axis="y", color="0.9", lw=0.6)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COL[p], label=p, ms=8) for p in PROBES]
    handles.append(Patch(facecolor="0.6", alpha=0.28, label="judgment threshold"))
    leg = ax.legend(handles=handles, loc="lower center", frameon=False, ncol=4,
                    bbox_to_anchor=(0.5, -0.30))
    for t in leg.get_texts():
        if t.get_text() in PROBES:
            t.set_fontweight("bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), bbox_inches="tight"); plt.close(fig)


for model, tag, M, D in COMBOS:
    ents, mean, std, ix = load(model, tag)
    base = f"{model}_{tag}"
    render_table(M, D, f"{base}_table.png", ents, mean, std)
    fig1(M, D, f"{base}_fig1_composite.png", mean, std, ix)
    fig2(M, D, f"{base}_fig2_personas.png", mean, std, ix)

print("完成。figures/:")
for p in sorted(os.listdir(FIG)):
    print("  ", p)
