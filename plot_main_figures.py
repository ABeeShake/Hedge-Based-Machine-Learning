#!/usr/bin/env python3
"""plot_main_figures.py -- print-friendly main-text figures that replace dense results tables.

  fig_rmse_vs_hbml.pdf     replaces the cohort RMSE tables and the baselines table in the main text:
                           cohort-mean RMSE of every expert and the uniform ensemble relative to HBML
                           (full-history context), one panel per cohort x horizon.
  fig_paired_forest.pdf    replaces the paired-comparison table: mean paired difference
                           (best expert - HBML) with its 95% CI and 90% bootstrap TOST interval for
                           all 24 configurations, against the +/-1 mg/dL equivalence band.
  fig_population.pdf       replaces the population-model table: RMSE of population models and of
                           HBML + population relative to personal HBML.

The full tables move to the Supplementary Information unchanged. Inputs: overleaf/tables/
baselines_participant.csv, statistical_tests_best_expert.csv, population_comparison.csv.
Output: overleaf/images/main/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, "overleaf", "tables")
OUT = os.path.join(BASE, "overleaf", "images", "main")

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1-3 (validated all-pairs)
DS_LABEL = {"weinstock": "Weinstock 2016", "cgmacros": "CGMacros"}
H_LABEL = {0.5: "30 min", 2.0: "2 h", 5.0: "5 h"}
CTX_LABEL = {"6": "6 h", "12": "12 h", "24": "24 h", "full": "Full"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})


def _clean(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_rmse_vs_hbml():
    d = pd.read_csv(os.path.join(TABLES, "baselines_participant.csv"), dtype={"context": str, "id": str})
    d = d[d.context == "full"]
    models = ["ARIMA", "ETS", "NHITS", "NODE", "XGBoost", "Persistence", "RunningMean", "uniform"]
    names = {"uniform": "Uniform ensemble", "NHITS": "N-HITS"}
    lo, hi = -8.0, 25.0
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3), sharey=True)
    for i, ds in enumerate(("weinstock", "cgmacros")):
        for j, h in enumerate((0.5, 2.0, 5.0)):
            ax = axes[i, j]
            g = d[(d.dataset == ds) & (d.horizon == h)]
            hb = g["hbml"].mean()
            vals = [100.0 * (g[m].mean() - hb) / hb for m in models]
            y = np.arange(len(models))[::-1]
            ax.axvline(0, color=INK, linewidth=0.9)
            ax.axvspan(lo, 0, color="#f0efec", zorder=0)
            for yy, v, m in zip(y, vals, models):
                col = ORANGE if v < 0 else (BLUE if m != "uniform" else INK2)
                if v > hi:
                    ax.plot(hi, yy, marker=">", color=col, markersize=5, clip_on=False)
                    ax.text(hi - 0.8, yy, f"+{v:.0f}%", ha="right", va="center", fontsize=7, color=INK2)
                else:
                    ax.plot([0, v], [yy, yy], color=col, linewidth=1.2, alpha=0.5)
                    ax.plot(v, yy, "o", color=col, markersize=4.5, markeredgecolor="white", markeredgewidth=0.6)
                    if v < 0:
                        ax.text(v - 0.6, yy, f"{v:.1f}%", ha="right", va="center", fontsize=7, color=INK)
                    elif v == min(x for x in vals if x >= 0):   # closest competitor
                        ax.text(v + 0.8, yy, f"+{v:.1f}%", ha="left", va="center", fontsize=7, color=INK)
            ax.set_xlim(lo, hi)
            ax.set_yticks(y)
            ax.set_yticklabels([names.get(m, m) for m in models])
            ax.set_title(f"{DS_LABEL[ds]}, {H_LABEL[h]}\nHBML {hb:.1f} mg/dL, N = {len(g)}", loc="left",
                         color=INK, fontsize=7.8)
            _clean(ax)
    fig.supxlabel("Cohort-mean RMSE relative to HBML (%): right of 0 = worse than HBML, shaded = better",
                  fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_rmse_vs_hbml.pdf"))
    plt.close(fig)


def fig_paired_forest():
    d = pd.read_csv(os.path.join(TABLES, "statistical_tests_best_expert.csv"), dtype={"context": str})
    margin = float(d["tost_margin_mgdl"].iloc[0]) if "tost_margin_mgdl" in d else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.9), sharex=True)
    for ax, ds in zip(axes, ("weinstock", "cgmacros")):
        g = d[d.dataset == ds].copy()
        g["ctx_order"] = g.context.map({"6": 0, "12": 1, "24": 2, "full": 3})
        g = g.sort_values(["horizon", "ctx_order"])
        ypos, labels, y, gap = [], [], 0, 0
        prev_h = None
        for _, r in g.iterrows():
            if prev_h is not None and r.horizon != prev_h:
                y += 0.8
            ypos.append(y)
            labels.append(f"{H_LABEL[r.horizon]} · {CTX_LABEL[r.context]} · {r.expert.replace('NHITS', 'N-HITS')}")
            prev_h = r.horizon
            y += 1
        ypos = -np.array(ypos)
        ax.axvspan(-margin, margin, color="#e7f1fb", zorder=0)
        ax.axvline(0, color=INK, linewidth=0.8)
        for yy, (_, r) in zip(ypos, g.iterrows()):
            sig = r["direction_bh"] != "ns"
            ax.plot([r.ci95_mean_low, r.ci95_mean_high], [yy, yy], color=INK2, linewidth=0.8)
            ax.plot([r.tost_boot_ci90_low, r.tost_boot_ci90_high], [yy, yy], color=BLUE, linewidth=3.2,
                    solid_capstyle="round", alpha=0.85)
            ax.plot(r.paired_diff_mean, yy, "o", markersize=4.2, color=INK if sig else "white",
                    markeredgecolor=INK, markeredgewidth=0.8, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=7)
        nmin, nmax = g.n_participants.min(), g.n_participants.max()
        ax.set_title(f"{DS_LABEL[ds]} (N = {nmin}{'' if nmin == nmax else f'–{nmax}'})", loc="left",
                     color=INK, pad=14)
        _clean(ax)
    xl = max(abs(d.ci95_mean_low.min()), abs(d.ci95_mean_high.max()), margin) + 0.3
    for ax in axes:
        ax.set_xlim(-xl, xl)
        ax.text(xl * 0.97, ax.get_ylim()[1], "HBML better →", ha="right", va="bottom", fontsize=7, color=INK2)
        ax.text(-xl * 0.97, ax.get_ylim()[1], "← expert better", ha="left", va="bottom", fontsize=7, color=INK2)
    fig.supxlabel("Paired RMSE difference, best single expert − HBML (mg/dL)", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_paired_forest.pdf"))
    plt.close(fig)


def fig_population():
    d = pd.read_csv(os.path.join(TABLES, "population_comparison.csv"), dtype={"context": str})
    series = [("Pop-NHITS_rmse", "Pop-NHITS", ORANGE, "s"), ("Pop-XGBoost_rmse", "Pop-XGBoost", BLUE, "D"),
              ("HBML+Pop_rmse", "HBML + population", AQUA, "o")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharex=True, sharey=True)
    for ax, scheme, title in zip(axes, ("within", "cross"),
                                 ("Within-cohort (5-fold, participant-level)", "Cross-cohort transfer")):
        g = d[d.scheme == scheme]
        rows = [(ds, h) for ds in ("weinstock", "cgmacros") for h in (0.5, 2.0, 5.0)]
        ypos = -np.array([k + (0.6 if k >= 3 else 0) for k in range(len(rows))])
        ax.axvline(0, color=INK, linewidth=0.9)
        ax.axvspan(-30, 0, color="#f0efec", zorder=0)
        for yy, (ds, h) in zip(ypos, rows):
            r = g[(g.dataset == ds) & (g.horizon == h)]
            if r.empty:
                continue
            r = r.iloc[0]
            for k, (col, lab, c, mk) in enumerate(series):
                v = 100.0 * (r[col] - r["HBML_rmse"]) / r["HBML_rmse"]
                ax.plot(v, yy + (k - 1) * 0.22, mk, color=c, markersize=4.2, markeredgecolor="white",
                        markeredgewidth=0.5, label=lab if yy == ypos[0] else None)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{DS_LABEL[ds]}, {H_LABEL[h]}" for ds, h in rows])
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("RMSE relative to personal HBML (%)")
        _clean(ax)
    allv = []
    for col, *_ in series:
        allv += list(100.0 * (d[col] - d["HBML_rmse"]) / d["HBML_rmse"])
    axes[0].set_xlim(min(-6, np.nanmin(allv) - 1), np.nanmax(allv) + 2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(OUT, "fig_population.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_rmse_vs_hbml()
    fig_paired_forest()
    fig_population()
    print(f"Wrote figures to {OUT}")
