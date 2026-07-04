# -*- coding: utf-8 -*-
"""Taffy 声音演化出图（读 analyze_evolution 的产物）。

Fig1  voice Hubble diagram：θ(现在, epoch) 随日期的演化曲线；
      叠 same-period floor 带(session 噪声)、different-person ceiling(Monaka/Miya)、
      previous-identity 参考线(Echo/Fries)。
Fig2  两张 2D 轨迹图：PCA on centroids  +  classical MDS on θ-matrix；
      epoch 按时间上色连成轨迹，landmark 用不同 marker 标注（看是平滑漂移还是断裂/聚成一团）。
标签全 ASCII，无需 CJK 字体。
用法：python make_figures_evolution.py
"""
import os
import json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "Palatino", "serif"],
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
RES = os.path.join(ROOT, "results")

COL_EPOCH = "#2A6FDB"
COL_FLOOR = "0.60"
COL_CEIL = "#D03B3B"       # different person
COL_PREV = "#E08A1E"       # previous identity (Echo/Fries)


def load():
    with open(os.path.join(RES, "report_evolution_campp.json"), encoding="utf-8") as f:
        r = json.load(f)
    z = np.load(os.path.join(RES, "evolution_centroids_campp.npz"), allow_pickle=True)
    return r, z


def pca2(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    return Xc @ Vt[:2].T, evr[:2]


def mds2(D):
    D2 = D ** 2
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    Bm = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(Bm)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    L = np.sqrt(np.clip(w[:2], 0, None))
    return V[:, :2] * L


def fig1_hubble(r):
    ents = r["entities"]; kinds = r["kinds"]; dates = r["dates"]
    tta = r["theta_to_anchor"]
    ep = [(datetime.strptime(dates[i], "%Y-%m-%d"), tta[i][0], tta[i][1], ents[i])
          for i in range(len(ents)) if kinds[i] == "epoch" and dates[i]]
    ep.sort()
    xs = [e[0] for e in ep]; ys = [e[1] for e in ep]; es = [e[2] for e in ep]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    # same-period floor
    fl = r["floor"]
    if fl.get("mean") is not None:
        ax.axhspan(fl["mean"] - fl["std"], fl["mean"] + fl["std"], color=COL_FLOOR, alpha=0.30, zorder=0)
        ax.axhline(fl["mean"], color="0.45", ls="--", lw=1.0, zorder=1)
    # different-person ceiling
    for n, (m, s) in r["ceiling"].items():
        ax.axhline(m, color=COL_CEIL, ls=":", lw=1.4, zorder=1)
        ax.text(xs[-1], m, f" {n}", color=COL_CEIL, va="center", ha="left", fontsize=9)
    # previous-identity references
    idx = {e: i for i, e in enumerate(ents)}
    for n in ("Echo", "Fries"):
        if n in idx:
            m = tta[idx[n]][0]
            ax.axhline(m, color=COL_PREV, ls="-.", lw=1.2, zorder=1)
            ax.text(xs[0], m, f"{n} ", color=COL_PREV, va="center", ha="right", fontsize=9)
    # evolution curve
    ax.plot(xs, ys, "-", color=COL_EPOCH, lw=1.6, zorder=2)
    ax.errorbar(xs, ys, yerr=es, fmt="o", color=COL_EPOCH, ms=6, lw=1.4, capsize=3, zorder=3)
    # "现在"锚点：独立参照星，不连趋势线（θ到自己=0 只是参照点，不是突变）
    if "anchor" in kinds:
        ainow = kinds.index("anchor")
        now_x = datetime.strptime(dates[ainow], "%Y-%m-%d")
        ax.scatter([now_x], [0], marker="*", s=280, color="#D03B3B", edgecolor="k", lw=0.5, zorder=5)
        ax.annotate("now", (now_x, 0), textcoords="offset points", xytext=(-6, 9),
                    ha="right", va="bottom", fontsize=10, color="#D03B3B", fontweight="bold")

    ax.set_ylabel(r"$\theta$ = arccos(cos)  to  now  (rad)")
    ax.set_xlabel("stream date")
    ax.set_title("Taffy voice evolution — angular distance to current voice (CAM++ / vocals)",
                 fontweight="bold")
    ax.set_ylim(0, max([m for m, _ in r["ceiling"].values()] + [max(ys) + max(es)]) * 1.12)
    ax.grid(axis="y", color="0.9", lw=0.6)
    # 明确标注起点 2021.7
    xt = [datetime(2021, 7, 1)] + [datetime(y, 1, 1) for y in range(2022, 2027)]
    ax.set_xticks(xt)
    ax.set_xticklabels(["2021.7", "2022", "2023", "2024", "2025", "2026"])
    ax.set_xlim(left=datetime(2021, 7, 1))
    # legend proxies
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=COL_EPOCH, marker="o", label="epoch θ to now"),
        Line2D([], [], marker="*", color="#D03B3B", ls="", label="now (reference, θ=0)"),
        Patch(facecolor=COL_FLOOR, alpha=0.30, label="same-period floor (session noise)"),
        Line2D([], [], color=COL_CEIL, ls=":", label="different person (Monaka soundalike / Miya)"),
        Line2D([], [], color=COL_PREV, ls="-.", label="previous identity (Echo/Fries)"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False)
    fig.tight_layout()
    p = os.path.join(FIG, "evolution_fig1_hubble.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("  saved", p)


def _scatter_map(ax, coords, kinds, dts, names, title):
    ep = [i for i, k in enumerate(kinds) if k in ("epoch", "anchor")]
    ep = sorted(ep, key=lambda i: (np.inf if np.isnan(dts[i]) else -dts[i]))  # old->new
    # epoch trajectory colored by time (older dark -> newer bright)
    xs = coords[ep, 0]; ys = coords[ep, 1]
    ax.plot(xs, ys, "-", color="0.8", lw=0.8, alpha=0.7, zorder=1)
    order = np.array([dts[i] for i in ep])
    sc = ax.scatter(xs, ys, c=-order, cmap="viridis", s=45, zorder=3, edgecolor="w", lw=0.5)
    # anchor star
    for i in ep:
        if kinds[i] == "anchor":
            ax.scatter(coords[i, 0], coords[i, 1], marker="*", s=240, color="#D03B3B",
                       edgecolor="k", lw=0.5, zorder=4, label="now (anchor)")
    # landmarks
    mk = {"prev": ("s", COL_PREV), "diff": ("X", COL_CEIL)}
    seen = set()
    for i, k in enumerate(kinds):
        if k in mk:
            m, c = mk[k]
            lbl = {"prev": "previous identity", "diff": "different person"}[k]
            ax.scatter(coords[i, 0], coords[i, 1], marker=m, s=80, color=c, edgecolor="k",
                       lw=0.5, zorder=4, label=(lbl if lbl not in seen else None))
            seen.add(lbl)
            ax.annotate(names[i], (coords[i, 0], coords[i, 1]), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=c)
    ax.set_title(title, fontweight="bold")
    ax.grid(color="0.92", lw=0.5)
    return sc


def fig2_maps(r, z):
    cents = z["centroids"]; names = list(z["names"]); kinds = list(z["kinds"])
    dts = z["dt_months"].astype(float)
    coords_p, evr = pca2(cents)
    coords_m = mds2(np.array(r["theta_mean"]))

    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.4))
    sc = _scatter_map(axs[0], coords_p, kinds, dts, names,
                      f"PCA on centroids  (PC1 {evr[0]*100:.0f}% · PC2 {evr[1]*100:.0f}%)")
    _scatter_map(axs[1], coords_m, kinds, dts, names, "classical MDS on θ-matrix")
    for ax in axs:
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    # colorbar for time
    cb = fig.colorbar(sc, ax=axs, fraction=0.035, pad=0.02)
    epdt = dts[[i for i, k in enumerate(kinds) if k in ("epoch", "anchor")]]
    cvals = -epdt
    cb.set_ticks([cvals.min(), cvals.max()])
    cb.set_ticklabels(["older", "newer"])
    cb.set_label("stream time")
    # de-dup legend
    h, l = axs[0].get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    axs[0].legend(seen.values(), seen.keys(), loc="best", frameon=False)
    fig.suptitle("Taffy voice — speaker-embedding trajectory over 5 years (landmarks for scale)",
                 fontweight="bold", y=1.02)
    p = os.path.join(FIG, "evolution_fig2_pca_mds.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("  saved", p)


if __name__ == "__main__":
    r, z = load()
    print("figures/:")
    fig1_hubble(r)
    fig2_maps(r, z)
