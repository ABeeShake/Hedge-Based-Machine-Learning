#!/usr/bin/env python3
"""calculate_statistics.py -- participant-level paired comparisons of HBML with each expert.

Input (in order of preference)
  1. overleaf/tables/rmse_participant.csv   raw per-participant RMSE written by generate_figures.py
  2. overleaf/tables/rmse_dev_*.csv + rmse_pivot_mean.csv   (fallback: RMSE reconstructed from rounded
     tables; accurate to about 0.01 mg/dL; reported as source = "reconstructed")

For every dataset x horizon x context and every expert:
  - paired difference  delta_i = RMSE_expert,i - RMSE_HBML,i   (positive = HBML better)
  - BCa bootstrap 95% CI for the mean and the median difference (10,000 resamples)
  - two-sided Wilcoxon signed-rank test (primary) and paired t-test (with degrees of freedom)
  - Shapiro-Wilk test of the paired differences
  - Benjamini-Hochberg adjusted Wilcoxon p-values (within the best-expert family, and within all experts)
  - win / tie / loss counts

Outputs (all in overleaf/tables/)
  statistical_tests_best_expert.csv, statistical_tests_all_experts.csv,
  statistical_summary.csv (counts quoted in the text), table_statistical_tests.tex

Run generate_figures.py --rmse_only first, and generate_results_macros.py afterwards.
"""
import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, norm, shapiro, ttest_1samp, ttest_rel, wilcoxon

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(BASE_DIR, "overleaf", "tables")

DATASETS = ["weinstock", "cgmacros"]
HORIZONS = [0.5, 2.0, 5.0]
CONTEXTS = ["6", "12", "24", "full"]
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ExpMethods.globals import GlobalValues  # noqa: E402
EXPERTS = list(GlobalValues.experts)
EQUIV_MARGIN_MGDL = float(getattr(GlobalValues, "equiv_margin_mgdl", 1.0))
HORIZON_MAP = {0.5: "0p5", 2.0: "2", 5.0: "5"}
HORIZON_LABEL = {0.5: "0.5 hr (30 min)", 2.0: "2 hr", 5.0: "5 hr"}
CONTEXT_LABEL = {"6": "6 hr", "12": "12 hr", "24": "24 hr", "full": "Full history"}
DATASET_LABEL = {"weinstock": "Weinstock", "cgmacros": "CGMacros"}
ALPHA = 0.05
TIE_TOL = 1e-3


def hodges_lehmann_ci(d, alpha=ALPHA):
    """Exact paired Hodges-Lehmann pseudomedian and distribution-free Walsh 95% CI."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        v = float(d[0])
        return v, v, v
    iu = np.triu_indices(n)
    walsh = np.sort(0.5 * (d[:, None] + d[None, :])[iu])
    est = float(np.median(walsh))
    m = len(walsh)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    mu = n * (n + 1) / 4.0
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    k = int(np.floor(mu - z * sd))
    lo_idx = max(0, min(m - 1, k - 1))
    hi_idx = max(0, min(m - 1, m - k))
    return est, float(walsh[lo_idx]), float(walsh[hi_idx])


def tost_equivalence(d, margin):
    """Two one-sided tests (TOST) on paired differences d for [-margin, +margin]."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 6 or margin <= 0:
        return np.nan, np.nan
    try:
        p1 = float(wilcoxon(d + margin, alternative="greater").pvalue)
        p2 = float(wilcoxon(d - margin, alternative="less").pvalue)
        w_tost = max(p1, p2)
    except Exception:
        w_tost = np.nan
    try:
        tp1 = float(ttest_1samp(d + margin, 0.0, alternative="greater").pvalue)
        tp2 = float(ttest_1samp(d - margin, 0.0, alternative="less").pvalue)
        t_tost = max(tp1, tp2)
    except Exception:
        t_tost = np.nan
    return w_tost, t_tost


def clean_num(val):
    s = re.sub(r"[^0-9.\-]", "", str(val))
    try:
        return float(s)
    except ValueError:
        return np.nan


def _to_h(v):
    s = str(v).strip()
    return 0.5 if s in ("half", "0.5") else float(s)


def load_participant_rmse():
    """Return ({(dataset, horizon, context): DataFrame[id x model]}, source)."""
    out = {}
    raw_path = os.path.join(TABLES_DIR, "rmse_participant.csv")
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, dtype={"id": str, "context": str})
        df["horizon"] = df["horizon"].map(_to_h)
        for (ds, h, ctx), g in df.groupby(["dataset", "horizon", "context"]):
            out[(ds, float(h), str(ctx))] = g.pivot_table(index="id", columns="model", values="rmse")
        return out, "raw"

    warnings.warn("rmse_participant.csv not found: reconstructing participant RMSE from rounded tables. "
                  "Run generate_figures.py --rmse_only to create it.")
    piv = pd.read_csv(os.path.join(TABLES_DIR, "rmse_pivot_mean.csv"))
    means = {}
    for _, row in piv.iterrows():
        for ctx in CONTEXTS:
            v = clean_num(row.get(f"mean.{ctx}", np.nan))
            if not np.isnan(v):
                means[(str(row["dataset"]).strip().lower(), float(row["horizon"]), ctx, str(row["model"]).strip())] = v
    for ds in DATASETS:
        for h in HORIZONS:
            for ctx in CONTEXTS:
                p = os.path.join(TABLES_DIR, f"rmse_dev_{ds}_h{HORIZON_MAP[h]}_c{ctx}.csv")
                if not os.path.exists(p):
                    continue
                dev = pd.read_csv(p, dtype={"id": str}).set_index("id")
                rec = pd.DataFrame(index=dev.index)
                for col in dev.columns:
                    if (ds, h, ctx, col) in means:
                        rec[col] = means[(ds, h, ctx, col)] * (1.0 + dev[col] / 100.0)
                out[(ds, h, ctx)] = rec
    return out, "reconstructed"


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (NaNs are left as NaN)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    if len(ok) == 0:
        return q
    order = ok[np.argsort(p[ok])]
    m = len(order)
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.minimum(adj, 1.0)
    return q


def boot_ci(d, stat, n_boot, seed=42):
    """BCa bootstrap CI; falls back to the percentile method if BCa is degenerate."""
    d = d[~np.isnan(d)]
    if len(d) < 5 or np.allclose(d, d[0]):
        return np.nan, np.nan, "none"
    for method in ("BCa", "percentile"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = bootstrap((d,), stat, confidence_level=0.95, n_resamples=n_boot, method=method,
                                random_state=seed, vectorized=False if stat is np.median else None)
            lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
            if np.isfinite(lo) and np.isfinite(hi):
                return lo, hi, method
        except Exception:
            continue
    return np.nan, np.nan, "none"


def fmt_p(p):
    if pd.isna(p):
        return "N/A"
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def tex_p(p):
    """p-value for LaTeX text and tables: 0.0123 or $1.2\\times10^{-5}$."""
    if pd.isna(p):
        return "N/A"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def run_paired_analysis(n_boot=10000):
    data, source = load_participant_rmse()
    records = []
    for ds in DATASETS:
        for h in HORIZONS:
            for ctx in CONTEXTS:
                raw = data.get((ds, h, ctx))
                if raw is None or "HBML" not in raw.columns:
                    print(f"Warning: no HBML results for {ds} h={h} ctx={ctx}")
                    continue
                experts = [m for m in EXPERTS if m in raw.columns]
                raw = raw[["HBML"] + experts].dropna()
                n = len(raw)
                hb = raw["HBML"].to_numpy()
                exp_means = {m: raw[m].mean() for m in experts}
                best = min(exp_means, key=exp_means.get)
                hbml_lowest = bool(hb.mean() < min(exp_means.values()))
                for m in experts:
                    ex = raw[m].to_numpy()
                    d = ex - hb
                    lo, hi, meth = boot_ci(d, np.mean, n_boot)
                    mlo, mhi, mmeth = boot_ci(d, np.median, n_boot)
                    hl_est, hl_lo, hl_hi = hodges_lehmann_ci(d)
                    tost_w_p, tost_t_p = tost_equivalence(d, EQUIV_MARGIN_MGDL)
                    tost_rel5_w_p, _ = tost_equivalence(d, 0.05 * ex.mean())
                    try:
                        w = wilcoxon(ex, hb, alternative="two-sided")
                        w_stat, w_p = float(w.statistic), float(w.pvalue)
                    except Exception:
                        w_stat, w_p = np.nan, np.nan
                    t = ttest_rel(ex, hb)
                    try:
                        sw_p = float(shapiro(d).pvalue)
                    except Exception:
                        sw_p = np.nan
                    n_imp = int((d > TIE_TOL).sum())
                    n_worse = int((d < -TIE_TOL).sum())
                    records.append(dict(
                        dataset=ds, horizon=h, context=ctx, n_participants=n, expert=m, is_best_expert=(m == best),
                        hbml_lowest_mean=hbml_lowest,
                        hbml_mean=hb.mean(), hbml_sd=hb.std(ddof=1), hbml_median=np.median(hb),
                        hbml_iqr=np.percentile(hb, 75) - np.percentile(hb, 25),
                        expert_mean=ex.mean(), expert_sd=ex.std(ddof=1), expert_median=np.median(ex),
                        expert_iqr=np.percentile(ex, 75) - np.percentile(ex, 25),
                        paired_diff_mean=d.mean(), paired_diff_se=d.std(ddof=1) / np.sqrt(n),
                        paired_diff_median=np.median(d),
                        hodges_lehmann=hl_est, hl_ci95_low=hl_lo, hl_ci95_high=hl_hi,
                        ci95_mean_low=lo, ci95_mean_high=hi, ci95_med_low=mlo, ci95_med_high=mhi,
                        ci_method_mean=meth, ci_method_median=mmeth,
                        wilcoxon_w=w_stat, wilcoxon_p=w_p,
                        tost_margin_mgdl=EQUIV_MARGIN_MGDL, tost_wilcoxon_p=tost_w_p, tost_ttest_p=tost_t_p,
                        tost_rel5_wilcoxon_p=tost_rel5_w_p,
                        equiv_wilcoxon=bool(pd.notna(tost_w_p) and tost_w_p < ALPHA),
                        ttest_t=float(t.statistic), ttest_df=n - 1, ttest_p=float(t.pvalue),
                        shapiro_p=sw_p,
                        n_improved=n_imp, n_tied=n - n_imp - n_worse, n_worse=n_worse,
                        pct_improved=100.0 * n_imp / n, pct_worse=100.0 * n_worse / n,
                        rel_reduction_pct=100.0 * (ex.mean() - hb.mean()) / ex.mean(),
                    ))
    df_all = pd.DataFrame(records)
    if df_all.empty:
        raise SystemExit("No statistics could be computed (no input tables).")
    df_all["wilcoxon_p_bh"] = bh_adjust(df_all["wilcoxon_p"].to_numpy())
    best_mask = df_all["is_best_expert"].to_numpy()
    df_all["wilcoxon_p_bh_best_family"] = np.nan
    df_all.loc[best_mask, "wilcoxon_p_bh_best_family"] = bh_adjust(df_all.loc[best_mask, "wilcoxon_p"].to_numpy())

    def direction(r, col):
        if pd.isna(r[col]) or r[col] >= ALPHA:
            return "ns"
        return "hbml_better" if r["paired_diff_mean"] > 0 else "hbml_worse"
    df_all["direction_bh"] = df_all.apply(lambda r: direction(r, "wilcoxon_p_bh"), axis=1)
    df_best = df_all[best_mask].copy()
    df_best["direction_raw"] = df_best.apply(lambda r: direction(r, "wilcoxon_p"), axis=1)
    df_best["direction_bh"] = df_best.apply(lambda r: direction(r, "wilcoxon_p_bh_best_family"), axis=1)

    df_best.to_csv(os.path.join(TABLES_DIR, "statistical_tests_best_expert.csv"), index=False)
    df_all.to_csv(os.path.join(TABLES_DIR, "statistical_tests_all_experts.csv"), index=False)

    # ---- counts quoted in the text ----
    rows = []
    def add(scope, key, value):
        rows.append(dict(scope=scope, key=key, value=value))
    for scope, g in [("all", df_best)] + [(ds, df_best[df_best.dataset == ds]) for ds in DATASETS]:
        if g.empty:
            continue
        add(scope, "n_configs", len(g))
        add(scope, "n_hbml_lowest_mean", int(g["hbml_lowest_mean"].sum()))
        add(scope, "n_mean_diff_positive", int((g["paired_diff_mean"] > 0).sum()))
        add(scope, "n_sig_raw_hbml_better", int((g["direction_raw"] == "hbml_better").sum()))
        add(scope, "n_sig_raw_hbml_worse", int((g["direction_raw"] == "hbml_worse").sum()))
        add(scope, "n_sig_bh_hbml_better", int((g["direction_bh"] == "hbml_better").sum()))
        add(scope, "n_sig_bh_hbml_worse", int((g["direction_bh"] == "hbml_worse").sum()))
        add(scope, "n_not_sig_bh", int((g["direction_bh"] == "ns").sum()))
        add(scope, "pct_improved_min", round(float(g["pct_improved"].min()), 1))
        add(scope, "pct_improved_max", round(float(g["pct_improved"].max()), 1))
        add(scope, "n_participants_min", int(g["n_participants"].min()))
        add(scope, "n_participants_max", int(g["n_participants"].max()))
        add(scope, "n_shapiro_reject", int((g["shapiro_p"] < ALPHA).sum()))
        add(scope, "n_equiv_wilcoxon", int(g["equiv_wilcoxon"].sum()))
        g30 = g[g["horizon"] == 0.5]
        add(scope, "n_configs_30min", len(g30))
        add(scope, "n_equiv_wilcoxon_30min", int(g30["equiv_wilcoxon"].sum()))
        add(scope, "n_equiv_ttest_30min", int((g30["tost_ttest_p"] < ALPHA).sum()))
    add("all", "equiv_margin_mgdl", EQUIV_MARGIN_MGDL)
    add("all", "source", source)
    add("all", "n_boot", n_boot)
    add("all", "fdr_q", ALPHA)
    pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, "statistical_summary.csv"), index=False)

    generate_latex_table(df_best, os.path.join(TABLES_DIR, "table_statistical_tests.tex"))
    print(f"Saved statistics for {len(df_best)} configurations (participant RMSE source: {source}).")
    return df_best


def generate_latex_table(df_best, output_tex):
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{lllclcccccc}", r"\toprule",
             r"Dataset & Horizon & Context & $N$ & Best expert & $\Delta$ mean [95\% CI] & HL est.\ [95\% CI] & HBML better & $p$ & $p_{\mathrm{BH}}$ & $p_{\mathrm{TOST}}$ \\",
             r"\midrule"]
    cur = None
    for _, r in df_best.iterrows():
        ds = DATASET_LABEL[r["dataset"]]
        if cur is not None and cur != ds:
            lines.append(r"\midrule")
        cur = ds
        n = int(r["n_participants"])
        q = r["wilcoxon_p_bh_best_family"]
        q_str = tex_p(q)
        if pd.notna(q) and q < ALPHA:
            q_str = f"\\textbf{{\\boldmath {q_str}}}"
        tost_p = r.get("tost_wilcoxon_p", np.nan)
        tost_str = tex_p(tost_p)
        if pd.notna(tost_p) and tost_p < ALPHA:
            tost_str = f"\\textbf{{\\boldmath {tost_str}}}"
        lines.append(
            f"{ds} & {HORIZON_LABEL[r['horizon']]} & {CONTEXT_LABEL[r['context']]} & {n} & {r['expert']} & "
            f"{r['paired_diff_mean']:+.2f} [{r['ci95_mean_low']:+.2f}, {r['ci95_mean_high']:+.2f}] & "
            f"{r['hodges_lehmann']:+.2f} [{r['hl_ci95_low']:+.2f}, {r['hl_ci95_high']:+.2f}] & "
            f"{int(r['n_improved'])}/{n} ({r['pct_improved']:.1f}\\%) & {tex_p(r['wilcoxon_p'])} & {q_str} & {tost_str} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}",
              r"\caption{\textbf{Participant-level paired comparisons and TOST equivalence between HBML and the best constituent expert.} "
              r"For each configuration, the best expert is the constituent model with the lowest cohort mean RMSE across the $N$ retained participants. "
              r"$\Delta = \text{RMSE}_{\text{best}} - \text{RMSE}_{\text{HBML}}$ (mg/dL) is computed per participant; "
              r"positive values favour HBML and negative values favour the expert. Mean 95\% confidence intervals use BCa bootstrap "
              r"(10,000 resamples); HL est.\ reports the exact Hodges--Lehmann pseudomedian of paired differences with its "
              r"95\% Walsh distribution-free interval. $p$: two-sided paired Wilcoxon signed-rank test; $p_{\mathrm{BH}}$: Benjamini--Hochberg "
              r"adjustment across the configurations shown; $p_{\mathrm{TOST}}$: two one-sided Wilcoxon tests for equivalence within the "
              f"pre-specified margin $\\pm {EQUIV_MARGIN_MGDL:.1f}$~mg/dL (bold: $p < 0.05$). ``HBML better'' counts participants with "
              r"strictly lower RMSE under HBML.}",
              r"\label{tab:statistical_tests}", r"\end{table*}"]
    with open(output_tex, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--tables_dir", default=None, help="Override overleaf/tables (used for testing).")
    a = ap.parse_args()
    if a.tables_dir:
        TABLES_DIR = a.tables_dir
    run_paired_analysis(a.n_boot)
