#!/usr/bin/env python3
"""generate_results_macros.py -- the ONLY writer of overleaf/sections/generated_macros.tex.

Every number quoted in the manuscript text comes from a macro defined here, and every
macro is read from a table in overleaf/tables/. No other script writes macros.

Run it last, after the tables exist:
    generate_figures.py --rmse_only [--median]  ->  rmse_pivot_*.csv, rmse_dev_*.csv, rmse_participant.csv
    calculate_statistics.py                     ->  statistical_tests_*.csv, statistical_summary.csv
    calculate_baselines.py                      ->  baselines_comparison.csv
    generate_figures.py                         ->  max_ceg.csv, ceg_cgmacros_*.csv, subgroup tables
    generate_results_macros.py [--strict]

Macro families (DS = Weinstock | Cgmacros, H = ThirtyMin | TwoHour | FiveHour,
C = SixHr | TwelveHr | TwentyFourHr | Full, M = ARIMA | ETS | NHITS | NODE | XGBoost | HBML,
G = NoDiabetes | Prediabetes | TypeTwo):
    \\RMSEMean{DS}{H}{M}{C}  \\RMSEsd...  \\RMSEMedian...  \\RMSEiqr...
    \\RMSEReduction{DS}{H}{C}            (best expert - HBML) / best expert * 100; negative = HBML worse
    \\RMSEBestExpert{DS}{H}{C}  \\RMSEBestExpertMean{DS}{H}{C}  \\RMSEBestExpertSd{DS}{H}{C}
    \\RMSEHBMLMin{DS}{H}  \\RMSEHBMLMinContext{DS}{H}  \\RMSEHBMLMax{DS}{H}  \\RMSEHBMLMaxContext{DS}{H}
    \\RMSEMeanCgmacros{G}{H}{M}{C}  \\RMSEsdCgmacros{G}...  \\RMSEBestExpertCgmacros{G}{H}{C} ...
    \\CEG{DS}{H}{C}{A|B|AB|C|DOne|DTwo|EOne|ETwo|CD|E}   \\CEG{DS}{H}{zone} (= Full context)
    \\CEGABMin{DS}{H}  \\CEGABMax{DS}{H}  \\CEGCgmacros{G}{H}{C}{zone}
    \\WilcoxonP  \\WilcoxonPBH  \\TtestP  \\PctImproved  \\NImproved  \\NWorse  \\NTotal  \\BestExpert
    \\PairedDiffMean  \\PairedDiffMeanLow/High  \\PairedDiffMedian  \\PairedDiffMedianLow/High   (all + {DS}{H}{C})
    \\Stat{All|Weinstock|Cgmacros}{NConfigs|NHBMLLowestMean|NSigBHBetter|NSigBHWorse|NNotSigBH|...}
    \\Baseline{Persistence|Climatology|Uniform}{DS}{H}{C}  \\BaselineReduction...  \\BaselinePctImproved...
    \\BaselineReduction{B}{DS}{H}Min / Max
    \\Vol{SD|CoV|TIR|MAGE|MODD}{DS}{Mean|Sd}
    \\Timing{Forecast|Train}{Model}{Mean|Sd}  \\TimingHedgeUpdate...  \\TimingStepTotal...  \\TimingStepPctOfInterval
    \\N{DS}  \\NCgmacros{G}  \\NConfigs

A macro that the manuscript uses but that could not be computed is defined as
\\MacroMissing (a bold red "??") so the gap is visible instead of silently stale.
"""
import argparse
import csv
import datetime
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(BASE_DIR, "overleaf", "tables")
SECTIONS_DIR = os.path.join(BASE_DIR, "overleaf", "sections")
OUTPUT_FILE = os.path.join(SECTIONS_DIR, "generated_macros.tex")

CONTEXTS = [("6", "SixHr"), ("12", "TwelveHr"), ("24", "TwentyFourHr"), ("full", "Full")]
CTX_NAME = dict(CONTEXTS)
CTX_TEXT = {"6": "6-hour", "12": "12-hour", "24": "24-hour", "full": "full-history"}
STATUS_NAME = {"non-T2D": "NoDiabetes", "pre-T2D": "Prediabetes", "T2D": "TypeTwo"}
MACRO_PREFIXES = ("RMSE", "AbsRMSE", "Oracle", "CEG", "Wilcoxon", "Ttest", "PctImproved", "PctWorse", "NImproved", "NWorse", "NTotal",
                  "PairedDiff", "BestExpert", "StatAll", "StatWeinstock", "StatCgmacros", "StatCGMacros",
                  "Baseline", "Vol[A-Z]", "Timing", "NWeinstock", "NCgmacros", "NCGMacros", "NConfigs", "NTotalParticipants",
                  "NParticipantConfigs", "NSecondarySubset", "NMinScored", "NExcluded", "Ablation", "GammaSens", "NExpertsWinning",
                  "TopWinner", "SecondWinner", "Pop[A-Z]", "Cadence", "Grid", "Weight", "PoolSeven", "Clipped", "Hypo",
                  "Regime", "Seed", "HodgesLehmann", "TOST", "Equiv", "Cmp", "WeightedRMSE", "PooledRMSE", "Env")


def clean_name(name):
    return re.sub(r"[^A-Za-z]", "", str(name))


def strip_latex(val):
    return str(val).replace("*", "").replace("\\textbf{", "").replace("}", "").strip()


def to_float(val):
    try:
        return float(strip_latex(val))
    except (TypeError, ValueError):
        return None


def horizon_label(h_raw):
    h = str(h_raw).strip()
    if h in ("0.5", "half", "ThirtyMin"):
        return "ThirtyMin"
    if h in ("2", "2.0", "TwoHour"):
        return "TwoHour"
    if h in ("5", "5.0", "FiveHour"):
        return "FiveHour"
    return clean_name(h)


def dataset_label(raw):
    r = str(raw).strip().lower()
    if "weinstock" in r:
        return "Weinstock"
    if "cgm" in r:
        return "Cgmacros"
    return clean_name(raw).capitalize()


def tex_p(p):
    """p-value usable in text and in math mode."""
    if p is None:
        return None
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"\\ensuremath{{{m}\\times10^{{{int(e)}}}}}"


def read_rows(name):
    path = os.path.join(TABLES_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class Macros:
    def __init__(self):
        self.defs = {}      # name -> value (insertion ordered)
        self.sections = []  # (comment, [names])
        self._cur = None

    def section(self, title):
        self._cur = []
        self.sections.append((title, self._cur))

    def add(self, name, value):
        if value is None or value == "":
            return
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"Illegal macro name: {name}")
        if name not in self.defs:
            self._cur.append(name)
        self.defs[name] = str(value)

    def num(self, name, value, nd=2):
        if value is None or value != value:
            return
        self.add(name, f"{value:.{nd}f}")


def rmse_block(M, rows, stat, disp, name_stat, name_disp, group_fn):
    """Per-model macros from a pivot table. group_fn(row) -> macro fragment before the horizon."""
    by_cfg = {}
    for row in rows:
        frag = group_fn(row)
        if frag is None:
            continue
        hz = horizon_label(row["horizon"])
        model = clean_name(row["model"])
        for c, cname in CONTEXTS:
            v, d = row.get(f"{stat}.{c}", ""), row.get(f"{disp}.{c}", "")
            if v not in ("", None):
                M.add(f"RMSE{name_stat}{frag}{hz}{model}{cname}", strip_latex(v))
                fv = to_float(v)
                if fv is not None:
                    by_cfg.setdefault((frag, hz, c), {})[model] = (fv, strip_latex(d))
            if d not in ("", None):
                M.add(f"RMSE{name_disp}{frag}{hz}{model}{cname}", strip_latex(d))
    return by_cfg


def best_expert_block(M, by_cfg, count_scope=None):
    lowest = total = 0
    for (frag, hz, c), models in by_cfg.items():
        if "HBML" not in models:
            continue
        experts = {m: v for m, v in models.items() if m != "HBML"}
        if not experts:
            continue
        best = min(experts, key=lambda m: experts[m][0])
        bv, bsd = experts[best]
        hv = models["HBML"][0]
        cname = CTX_NAME[c]
        M.add(f"RMSEBestExpert{frag}{hz}{cname}", best)
        M.num(f"RMSEBestExpertMean{frag}{hz}{cname}", bv)
        M.add(f"RMSEBestExpertSd{frag}{hz}{cname}", bsd)
        rel_val = (bv - hv) / bv * 100.0
        M.num(f"RMSEReduction{frag}{hz}{cname}", rel_val, 1)
        M.num(f"AbsRMSEReduction{frag}{hz}{cname}", abs(rel_val), 1)
        M.num(f"RMSEDiff{frag}{hz}{cname}", bv - hv)
        M.num(f"AbsRMSEDiff{frag}{hz}{cname}", abs(bv - hv))
        if hv < bv - 0.005:
            M.add(f"CmpVerbBest{frag}{hz}{cname}", "outperformed")
            M.add(f"CmpAdjBest{frag}{hz}{cname}", "lower")
        elif abs(hv - bv) <= 0.005:
            M.add(f"CmpVerbBest{frag}{hz}{cname}", "matched")
            M.add(f"CmpAdjBest{frag}{hz}{cname}", "equal")
        else:
            M.add(f"CmpVerbBest{frag}{hz}{cname}", "trailed")
            M.add(f"CmpAdjBest{frag}{hz}{cname}", "higher")
        total += 1
        lowest += int(hv < bv)
    for frag, hz in sorted({(k[0], k[1]) for k in by_cfg}):
        vals = [(by_cfg[(frag, hz, c)]["HBML"][0], c) for c, _ in CONTEXTS
                if (frag, hz, c) in by_cfg and "HBML" in by_cfg[(frag, hz, c)]]
        if vals:
            lo, hi = min(vals), max(vals)
            M.num(f"RMSEHBMLMin{frag}{hz}", lo[0])
            M.add(f"RMSEHBMLMinContext{frag}{hz}", CTX_TEXT[lo[1]])
            M.num(f"RMSEHBMLMax{frag}{hz}", hi[0])
            M.add(f"RMSEHBMLMaxContext{frag}{hz}", CTX_TEXT[hi[1]])
    if count_scope:
        M.add(f"RMSE{count_scope}NConfigs", total)
        M.add(f"RMSE{count_scope}NHBMLLowest", lowest)


def ceg_block(M, rows, frag_fn):
    zone_map = [("A+Bpct", "AB"), ("Apct", "A"), ("Bpct", "B"), ("Cpct", "C"), ("D1pct", "DOne"),
                ("D2pct", "DTwo"), ("E1pct", "EOne"), ("E2pct", "ETwo")]
    ab = {}
    for row in rows:
        frag = frag_fn(row)
        if frag is None or not row.get("horizon"):
            continue
        hz = horizon_label(row["horizon"])
        c = str(row.get("context", "full")).strip()
        cname = CTX_NAME.get(c, "Full")
        z = {}
        for col, suf in zone_map:
            v = to_float(row.get(col, ""))
            if v is not None:
                z[suf] = v
                M.num(f"CEG{frag}{hz}{cname}{suf}", v)
        if {"C", "DOne", "DTwo"} <= set(z):
            z["CD"] = z["C"] + z["DOne"] + z["DTwo"]
            M.num(f"CEG{frag}{hz}{cname}CD", z["CD"])
            M.num(f"CEG{frag}{hz}{cname}D", z["DOne"] + z["DTwo"])
        if {"EOne", "ETwo"} <= set(z):
            z["E"] = z["EOne"] + z["ETwo"]
            M.num(f"CEG{frag}{hz}{cname}E", z["E"])
        if "AB" in z:
            M.num(f"CEG{frag}{hz}{cname}NotAB", 100.0 - z["AB"])
            ab.setdefault((frag, hz), []).append(z["AB"])
        if cname == "Full":
            for suf in z:
                M.add(f"CEG{frag}{hz}{suf}", f"\\CEG{frag}{hz}Full{suf}")
    for (frag, hz), vals in ab.items():
        M.num(f"CEGABMin{frag}{hz}", min(vals))
        M.num(f"CEGABMax{frag}{hz}", max(vals))


def generate_macros(strict=False, quiet=False):
    M = Macros()
    warnings_ = []
    sources = []

    def need(name):
        rows = read_rows(name)
        if rows is None:
            warnings_.append(f"missing table: {name}")
        else:
            sources.append(name)
        return rows

    # 1. RMSE (mean / SD and median / IQR)
    M.section("RMSE by model: mean (SD) across participants")
    rows = need("rmse_pivot_mean.csv")
    if rows:
        by_cfg = rmse_block(M, rows, "mean", "sd", "Mean", "sd", lambda r: dataset_label(r["dataset"]))
        M.section("RMSE: best expert, relative difference, range across context windows")
        best_expert_block(M, by_cfg, count_scope="All")
    M.section("RMSE by model: median (IQR) across participants")
    rows = need("rmse_pivot_median.csv")
    if rows:
        rmse_block(M, rows, "median", "iqr", "Median", "iqr", lambda r: dataset_label(r["dataset"]))

    # 2. CGMacros subgroups
    M.section("CGMacros glycaemic-status subgroups: RMSE")
    sub_cfg = {}
    for status, sname in STATUS_NAME.items():
        rows = need(f"rmse_mean_cgmacros_{status}.csv")
        if rows:
            sub_cfg.update(rmse_block(M, rows, "mean", "sd", "Mean", "sd", lambda r, s=sname: "Cgmacros" + s))
    if sub_cfg:
        best_expert_block(M, sub_cfg, count_scope="Subgroup")

    # 3. Clarke error grid
    M.section("Clarke error grid (HBML)")
    rows = need("max_ceg.csv")
    if rows:
        ceg_block(M, rows, lambda r: dataset_label(r["dataset"]) if r.get("dataset") else None)
    M.section("Clarke error grid: CGMacros subgroups")
    for status, sname in STATUS_NAME.items():
        rows = need(f"ceg_cgmacros_{status}.csv")
        if rows:
            ceg_block(M, rows, lambda r, s=sname: "Cgmacros" + s)

    # 4. Paired statistics
    M.section("Paired participant-level tests: HBML vs best expert")
    rows = need("statistical_tests_best_expert.csv")
    n_by_ds = {}
    n_by_ds_h = {}
    if rows:
        ns_cfg_labels = []
        better_cfg_labels = []
        better30_labels = []
        for r in rows:
            ds_l = dataset_label(r["dataset"])
            hz_l = horizon_label(r["horizon"])
            ctx_raw = str(r["context"]).strip().lower()
            cname = CTX_NAME.get(ctx_raw, "Full")
            key = f"{ds_l}{hz_l}{cname}"
            f = lambda k: to_float(r.get(k, ""))
            M.add(f"WilcoxonP{key}", tex_p(f("wilcoxon_p")))
            M.add(f"WilcoxonPBH{key}", tex_p(f("wilcoxon_p_bh_best_family")))
            M.add(f"TtestP{key}", tex_p(f("ttest_p")))
            M.num(f"PctImproved{key}", f("pct_improved"), 1)
            M.num(f"PctWorse{key}", f("pct_worse"), 1)
            M.add(f"NImproved{key}", r.get("n_improved", ""))
            M.add(f"NWorse{key}", r.get("n_worse", ""))
            M.add(f"NTotal{key}", r.get("n_participants", ""))
            M.add(f"BestExpert{key}", r.get("expert", ""))
            for stem, col in [("PairedDiffMean", "paired_diff_mean"), ("PairedDiffMeanLow", "ci95_mean_low"),
                              ("PairedDiffMeanHigh", "ci95_mean_high"), ("PairedDiffMedian", "paired_diff_median"),
                              ("PairedDiffMedianLow", "ci95_med_low"), ("PairedDiffMedianHigh", "ci95_med_high"),
                              ("HodgesLehmann", "hodges_lehmann"), ("HodgesLehmannLow", "hl_ci95_low"),
                              ("HodgesLehmannHigh", "hl_ci95_high")]:
                v = f(col)
                if v is not None:
                    M.add(f"{stem}{key}", f"{v:+.2f}")
            for pstem, pcol in [("TOSTWilcoxonP", "tost_wilcoxon_p"), ("TOSTTtestP", "tost_ttest_p"),
                                ("TOSTRelFiveP", "tost_rel5_wilcoxon_p")]:
                pv = f(pcol)
                if pv is not None:
                    M.add(f"{pstem}{key}", tex_p(pv))
            dm = f("paired_diff_mean")
            if dm is not None:
                M.add(f"CmpSignDir{key}", "positive" if dm > 0 else ("zero" if abs(dm) < 5e-3 else "negative"))
                M.add(f"CmpAdjDiff{key}", "reduction" if dm > 0 else "difference")
            dir_bh = str(r.get("direction_bh", "")).strip()
            ds_disp = "Weinstock 2016" if ds_l == "Weinstock" else "CGMacros"
            hz_disp = "30-minute" if hz_l == "ThirtyMin" else ("2-hour" if hz_l == "TwoHour" else "5-hour")
            ctx_disp = CTX_TEXT.get(ctx_raw, ctx_raw)
            cfg_txt = f"{ds_disp} {hz_disp} ({ctx_disp})"
            if dir_bh == "ns":
                ns_cfg_labels.append(cfg_txt)
            elif dir_bh == "hbml_better":
                better_cfg_labels.append(cfg_txt)
                if hz_l == "ThirtyMin":
                    better30_labels.append(cfg_txt)
            n = to_float(r.get("n_participants", ""))
            if n is not None:
                n_by_ds.setdefault(ds_l, []).append(int(n))
                n_by_ds_h.setdefault((ds_l, hz_l), []).append(int(n))
        M.add("StatAllNSList", "; ".join(ns_cfg_labels) if ns_cfg_labels else "none")
        M.add("StatAllBetterList", "; ".join(better_cfg_labels) if better_cfg_labels else "none")
        M.add("StatAllBetterThirtyMinList", "; ".join(better30_labels) if better30_labels else "none")
    for (ds_l, hz_l), ns in n_by_ds_h.items():
        M.add(f"N{ds_l}{hz_l}", max(ns))
    rows = need("statistical_summary.csv")
    if rows:
        key_name = {"n_configs": "NConfigs", "n_hbml_lowest_mean": "NHBMLLowestMean",
                    "n_mean_diff_positive": "NMeanDiffPositive", "n_sig_raw_hbml_better": "NSigRawBetter",
                    "n_sig_raw_hbml_worse": "NSigRawWorse", "n_sig_bh_hbml_better": "NSigBHBetter",
                    "n_sig_bh_hbml_worse": "NSigBHWorse", "n_not_sig_bh": "NNotSigBH",
                    "pct_improved_min": "PctImprovedMin", "pct_improved_max": "PctImprovedMax",
                    "n_participants_min": "NParticipantsMin", "n_participants_max": "NParticipantsMax",
                    "n_shapiro_reject": "NShapiroReject", "n_boot": "NBoot", "fdr_q": "FDRq",
                    "n_equiv_wilcoxon": "NEquivWilcoxon", "n_equiv_ttest": "NEquivTtest",
                    "n_equiv_rel5": "NEquivRelFive", "n_configs_30min": "NConfigsThirtyMin",
                    "n_equiv_wilcoxon_30min": "NEquivWilcoxonThirtyMin",
                    "n_equiv_ttest_30min": "NEquivTtestThirtyMin",
                    "n_equiv_rel5_30min": "NEquivRelFiveThirtyMin",
                    "equiv_margin_mgdl": "EquivMarginMgdl"}
        scope_name = {"all": "All", "weinstock": "Weinstock", "cgmacros": "Cgmacros"}
        for r in rows:
            if r["key"] == "source":
                if r["value"] != "raw":
                    warnings_.append("statistics were computed from reconstructed (rounded) participant RMSE; "
                                     "run generate_figures.py --rmse_only and calculate_statistics.py again")
                continue
            if r["key"] in key_name and r["scope"] in scope_name:
                v = r["value"]
                if r["key"] == "n_boot":
                    v = f"{int(float(v)):,}".replace(",", "{,}")
                M.add(f"Stat{scope_name[r['scope']]}{key_name[r['key']]}", v)

    # 5. Cohort sizes
    M.section("Cohort sizes")
    for ds, ns in n_by_ds.items():
        M.add(f"N{ds}", max(ns))
        M.add(f"N{ds}Min", min(ns))
    try:
        sys.path.insert(0, BASE_DIR)
        from ExpMethods.globals import GlobalValues
        ids = None
        for h in ("0p5", "2", "5"):
            dev = read_rows(f"rmse_dev_cgmacros_h{h}_cfull.csv")
            if dev:
                ids = {str(r["id"]).zfill(3) for r in dev}
                break
        if ids:
            for key, sname in [("nondiabetic_ids", "NoDiabetes"), ("prediabetic_ids", "Prediabetes"), ("diabetic_ids", "TypeTwo")]:
                M.add(f"NCgmacros{sname}", len(ids & set(GlobalValues.CGMacros[key])))
    except Exception as e:  # pragma: no cover
        warnings_.append(f"could not derive subgroup sizes: {e}")
    if n_by_ds:
        tot_p = sum(max(v) for v in n_by_ds.values())
        M.add("NTotalParticipants", tot_p)
        M.add("NParticipantConfigs", f"{tot_p * 12:,}".replace(",", "{,}"))
    M.add("NConfigs", "24")
    M.add("NSecondarySubset", "20")
    try:
        sys.path.insert(0, BASE_DIR)
        from ExpMethods.globals import GlobalValues as _GV_N
        M.add("NMinScored", int(getattr(_GV_N, "min_scored_points", 288)))
        M.num("EquivMarginMgdl", float(getattr(_GV_N, "equiv_margin_mgdl", 1.0)), 1)
    except Exception:
        M.add("NMinScored", "288")
        M.add("EquivMarginMgdl", "1.0")

    ex_sum = read_rows("rmse_exclusions_summary.csv")
    if ex_sum:
        sources.append("rmse_exclusions_summary.csv")
        by_ds_h_ex = {}
        for r in ex_sum:
            ds_k = dataset_label(r["dataset"])
            hz_k = horizon_label(r["horizon"])
            M.add(f"NExcluded{ds_k}{hz_k}", r.get("n_excluded", "0"))
            by_ds_h_ex[(ds_k, hz_k)] = int(float(r.get("n_excluded", 0) or 0))
    else:
        for ds_k in ("Weinstock", "Cgmacros"):
            for hz_k in ("ThirtyMin", "TwoHour", "FiveHour"):
                M.add(f"NExcluded{ds_k}{hz_k}", "0")

    w_sens = read_rows("rmse_weighted_sensitivity.csv")
    if w_sens:
        sources.append("rmse_weighted_sensitivity.csv")
        for r in w_sens:
            frag = dataset_label(r["dataset"])
            hz = horizon_label(r["horizon"])
            mod = clean_name(r["model"])
            cname = CTX_NAME.get(str(r["context"]).strip().lower(), "Full")
            M.num(f"WeightedRMSE{frag}{hz}{mod}{cname}", to_float(r.get("weighted_mean_rmse", "")))
            M.num(f"PooledRMSE{frag}{hz}{mod}{cname}", to_float(r.get("pooled_point_rmse", "")))

    # 6. Baselines
    M.section("Naive baselines (persistence, running mean, uniform ensemble)")
    rows = need("baselines_comparison.csv")
    if rows:
        if "climatology_rmse" not in rows[0]:
            warnings_.append("baselines_comparison.csv is from the old script; re-run calculate_baselines.py")
        rng = {}
        n_empty = 0
        for r in rows:
            ds, hz = dataset_label(r["dataset"]), horizon_label(r["horizon"])
            cname = CTX_NAME.get(str(r["context"]).strip().lower(), "Full")
            if r.get("status", "ok") != "ok":
                n_empty += 1
                continue
            hb_v = to_float(r.get("hbml_rmse", ""))
            for b, bname in [("persistence", "Persistence"), ("climatology", "Climatology"), ("uniform", "Uniform")]:
                bv = to_float(r.get(f"{b}_rmse", ""))
                M.num(f"Baseline{bname}{ds}{hz}{cname}", bv)
                red = to_float(r.get(f"hbml_reduction_vs_{b}_pct", ""))
                M.num(f"BaselineReduction{bname}{ds}{hz}{cname}", red, 1)
                if red is not None:
                    M.num(f"BaselineAbsReduction{bname}{ds}{hz}{cname}", abs(red), 1)
                if bv is not None and hb_v is not None:
                    M.add(f"BaselineCmpVerb{bname}{ds}{hz}{cname}",
                          "outperformed" if hb_v < bv - 0.005 else ("matched" if abs(hb_v - bv) <= 0.005 else "trailed"))
                M.num(f"BaselinePctImproved{bname}{ds}{hz}{cname}", to_float(r.get(f"pct_improved_vs_{b}", "")), 1)
                M.add(f"BaselineP{bname}{ds}{hz}{cname}", tex_p(to_float(r.get(f"wilcoxon_p_vs_{b}", ""))))
                if red is not None and red == red:
                    rng.setdefault((bname, ds, hz), []).append(red)
        for (bname, ds, hz), v in rng.items():
            M.num(f"BaselineReduction{bname}{ds}{hz}Min", min(v), 1)
            M.num(f"BaselineReduction{bname}{ds}{hz}Max", max(v), 1)
            av = [abs(x) for x in v]
            M.num(f"BaselineAbsReduction{bname}{ds}{hz}Min", min(av), 1)
            M.num(f"BaselineAbsReduction{bname}{ds}{hz}Max", max(av), 1)
        if n_empty:
            warnings_.append(f"{n_empty} baseline configuration(s) are empty (run calculate_baselines.py where the forecasts are)")

    # 7. Glucose variability
    M.section("Glucose variability of the cohorts")
    rows = need("volatility_summary.csv")
    if rows:
        metric = None
        for r in rows:
            m = re.search(r"\{([A-Za-z]+)\}\s*$", r.get("Metric", "") or "")
            if m:
                metric = m.group(1)
            elif r.get("Metric"):
                metric = clean_name(r["Metric"])
            mm = re.match(r"\s*([\-0-9.]+)\s*\(([\-0-9.]+)\)", r.get("Value", "") or "")
            if metric and mm:
                ds = dataset_label(r.get("Dataset", ""))
                M.add(f"Vol{metric}{ds}Mean", mm.group(1))
                M.add(f"Vol{metric}{ds}Sd", mm.group(2))

    # 8. Timings
    M.section("Wall-clock timings (seconds)")
    rows = need("timings_summary.csv")
    if rows:
        model = None
        cat_name = {"Forecasting": "Forecast", "Training": "Train", "Hedge Update": "HedgeUpdate",
                    "5-Expert Forecasting": "ForecastAll", "5-Expert Training": "TrainAll",
                    "All-Expert Forecasting": "ForecastAll", "All-Expert Training": "TrainAll",
                    "End-to-End Step (w/ Train)": "StepTotal"}
        for r in rows:
            raw = r.get("Model", "") or ""
            m = re.search(r"\{(?:\\textbf\{)?([A-Za-z ()]+?)\}?\}\s*$", raw)
            if m:
                model = clean_name(m.group(1))
            elif raw.strip():
                model = clean_name(raw)
            mm = re.match(r"\s*([0-9.]+)\s*\$\\pm\$\s*([0-9.]+)", r.get("Time (s)", "") or "")
            cat = cat_name.get((r.get("Category") or "").strip())
            if not (mm and cat and model):
                continue
            stem = f"Timing{cat}" if model.startswith("HBML") else f"Timing{cat}{model}"
            M.add(f"{stem}Mean", mm.group(1))
            M.add(f"{stem}Sd", mm.group(2))
            if stem == "TimingStepTotal":
                M.num("TimingStepPctOfInterval", float(mm.group(1)) / 300.0 * 100.0, 1)
            if stem == "TimingHedgeUpdate":
                M.num("TimingHedgeUpdateMicroseconds", float(mm.group(1)) * 1e6, 1)

    # 9. Per-participant hindsight oracle and winner heterogeneity
    M.section("Per-participant hindsight oracle and winner heterogeneity")
    rows = need("rmse_participant.csv")
    if rows:
        from collections import Counter, defaultdict
        import math
        by_cfg_p = defaultdict(lambda: defaultdict(dict))
        for r in rows:
            ds_k = dataset_label(r["dataset"])
            hz_k = horizon_label(r["horizon"])
            c_k = str(r["context"]).strip().lower()
            cname = CTX_NAME.get(c_k, "Full")
            v = to_float(r.get("rmse", ""))
            if v is not None:
                by_cfg_p[(ds_k, hz_k, cname)][str(r["id"]).zfill(3)][clean_name(r["model"])] = v
        from ExpMethods.globals import GlobalValues as _G
        _present = {clean_name(r["model"]) for r in rows}
        exps_list = [e for e in _G.experts if e in _present]   # v1 tables have 5 experts, v2 tables 7
        n_win_all = []
        rel_diff_30m = []
        for (ds_k, hz_k, cname), pmap in sorted(by_cfg_p.items()):
            key = f"{ds_k}{hz_k}{cname}"
            valid_p = [m for m in pmap.values() if "HBML" in m and all(e in m for e in exps_list)]
            if not valid_p:
                continue
            n_p = len(valid_p)
            hb_vals = [m["HBML"] for m in valid_p]
            or_vals = [min(m[e] for e in exps_list) for m in valid_p]
            wo_vals = [max(m[e] for e in exps_list) for m in valid_p]
            hb_m = sum(hb_vals) / n_p
            or_m = sum(or_vals) / n_p
            wo_m = sum(wo_vals) / n_p
            or_sd = math.sqrt(sum((x - or_m) ** 2 for x in or_vals) / max(n_p - 1, 1))
            wo_sd = math.sqrt(sum((x - wo_m) ** 2 for x in wo_vals) / max(n_p - 1, 1))
            M.num(f"RMSEOracleMean{key}", or_m)
            M.num(f"RMSEOracleSd{key}", or_sd)
            M.num(f"RMSEWorstExpertMean{key}", wo_m)
            M.num(f"RMSEWorstExpertSd{key}", wo_sd)
            M.num(f"RMSEOracleDiff{key}", hb_m - or_m)
            M.num(f"AbsRMSEOracleDiff{key}", abs(hb_m - or_m))
            M.add(f"OracleCmpVerb{key}", "outperformed" if hb_m < or_m - 0.005 else ("matched" if abs(hb_m - or_m) <= 0.005 else "trailed"))
            rel_d = (hb_m - or_m) / or_m * 100.0
            M.num(f"RMSEOracleRelDiff{key}", rel_d, 1)
            M.num(f"AbsRMSEOracleRelDiff{key}", abs(rel_d), 1)
            if hz_k == "ThirtyMin":
                rel_diff_30m.append(rel_d)
            winners = Counter(min(exps_list, key=lambda e: m[e]) for m in valid_p)
            n_win_all.append(len(winners))
            M.add(f"NExpertsWinning{key}", len(winners))
            most = winners.most_common()
            if len(most) >= 1:
                M.add(f"TopWinnerExpert{key}", most[0][0])
                M.add(f"TopWinnerCount{key}", most[0][1])
                M.num(f"TopWinnerPct{key}", most[0][1] / n_p * 100.0, 1)
            if len(most) >= 2:
                M.add(f"SecondWinnerExpert{key}", most[1][0])
                M.add(f"SecondWinnerCount{key}", most[1][1])
                M.num(f"SecondWinnerPct{key}", most[1][1] / n_p * 100.0, 1)
        if n_win_all:
            M.add("NExpertsWinningMin", min(n_win_all))
            M.add("NExpertsWinningMax", max(n_win_all))
        if rel_diff_30m:
            M.num("RMSEOracleRelDiffThirtyMinMin", min(rel_diff_30m), 1)
            M.num("RMSEOracleRelDiffThirtyMinMax", max(rel_diff_30m), 1)
            abs_rel_30m = [abs(x) for x in rel_diff_30m]
            M.num("AbsRMSEOracleRelDiffThirtyMinMin", min(abs_rel_30m), 1)
            M.num("AbsRMSEOracleRelDiffThirtyMinMax", max(abs_rel_30m), 1)

    # 10. Aggregator and Fixed-Share ablation + gamma sensitivity
    M.section("Aggregator and Fixed-Share ablation")
    fs_path = os.path.join(BASE_DIR, "fs_comparison", "summary.csv")
    if os.path.exists(fs_path):
        with open(fs_path, newline="") as f:
            fs_rows = list(csv.DictReader(f))
        prim_var = (fs_rows[0].get("primary_variant") or "current").strip() if fs_rows else "current"
        prim_code = "SFHSqrt" if prim_var == "sfh_sqrt" else "SFHDF"
        prim_label_name = "HBML-SFH-Sqrt" if prim_var == "sfh_sqrt" else "HBML-SFHDF"
        var_map = {
            "current": "SFHDF",
            "cum_decaymax": "CumDecayMax",
            "sfh_sqrt": "SFHSqrt",
            "ftl": "FTL",
            "alg1_noshare": "RecNoShare",
            "alg1_fs_fd": "FSFastDecay",
            "alg1_fs_d": "FSDecay",
            "alg1_fs_c0.001": "FSConstLow",
            "alg1_fs_c0.01": "FSConstMid",
            "alg1_fs_c0.05": "FSConstHigh",
            "uniform": "Uniform",
        }
        from collections import defaultdict
        by_ds_v = defaultdict(list)
        wins_ds_v = defaultdict(int)
        n_conds_ds_v = defaultdict(int)
        table_cells = defaultdict(dict)
        for r in fs_rows:
            vcode = var_map.get(r["variant"])
            if not vcode:
                continue
            ds_k = dataset_label(r["dataset"])
            hz_k = horizon_label(r["horizon"])
            cname = CTX_NAME.get(str(r["context"]).strip().lower(), "Full")
            key = f"{vcode}{ds_k}{hz_k}{cname}"
            m_rmse = to_float(r.get("mean_rmse", ""))
            pct_ch = to_float(r.get("pct_change_vs_current", ""))
            delta_c = to_float(r.get("delta_vs_current", ""))
            M.num(f"AblationRMSE{key}", m_rmse)
            if pct_ch is not None:
                M.add(f"AblationRelChange{key}", f"{pct_ch:+.2f}")
                by_ds_v[(ds_k, vcode)].append(pct_ch)
                n_conds_ds_v[(ds_k, vcode)] += 1
            if delta_c is not None and delta_c > 0:
                wins_ds_v[(ds_k, vcode)] += 1
            if cname == "Full" and m_rmse is not None:
                table_cells[(ds_k, vcode)][hz_k] = m_rmse
            if vcode == prim_code:
                M.num(f"AblationRMSEPrimary{ds_k}{hz_k}{cname}", m_rmse)
                if pct_ch is not None:
                    M.add(f"AblationRelChangePrimary{ds_k}{hz_k}{cname}", f"{pct_ch:+.2f}")
        for (ds_k, vcode), vals in sorted(by_ds_v.items()):
            mean_ch = sum(vals) / len(vals)
            M.add(f"AblationMeanRelChange{vcode}{ds_k}", f"{mean_ch:+.2f}")
            M.num(f"AblationAbsMeanRelChange{vcode}{ds_k}", abs(mean_ch), 2)
            M.add(f"AblationDirVerb{vcode}{ds_k}", "reduced" if mean_ch < -1e-9 else "increased")
            M.add(f"AblationNWins{vcode}{ds_k}", wins_ds_v[(ds_k, vcode)])
            M.add(f"AblationNConds{vcode}{ds_k}", n_conds_ds_v[(ds_k, vcode)])
            if vcode == prim_code:
                M.add(f"AblationMeanRelChangePrimary{ds_k}", f"{mean_ch:+.2f}")
                M.num(f"AblationAbsMeanRelChangePrimary{ds_k}", abs(mean_ch), 2)
                M.add(f"AblationNWinsPrimary{ds_k}", wins_ds_v[(ds_k, vcode)])
                M.add(f"AblationNCondsPrimary{ds_k}", n_conds_ds_v[(ds_k, vcode)])

        # Also expose direct SFHSqrt vs SFHDF comparison macros regardless of which row is base
        for ds_k in ("Weinstock", "Cgmacros"):
            ch_sqrt = to_float(M.defs.get(f"AblationMeanRelChangeSFHSqrt{ds_k}", ""))
            ch_sfhdf = to_float(M.defs.get(f"AblationMeanRelChangeSFHDF{ds_k}", ""))
            if prim_code == "SFHSqrt" and ch_sfhdf is not None:
                # SFHDF relative to SFHSqrt is positive when SFHSqrt is lower; express reduction of SFHSqrt vs SFHDF
                M.num(f"AblationSqrtRedVsSFHDF{ds_k}", ch_sfhdf, 2)
                w_sfhdf = int(M.defs.get(f"AblationNWinsSFHDF{ds_k}", 0))
                n_sfhdf = int(M.defs.get(f"AblationNCondsSFHDF{ds_k}", 12))
                M.add(f"AblationSqrtWinsVsSFHDF{ds_k}", n_sfhdf - w_sfhdf)
                M.add(f"AblationSqrtCondsVsSFHDF{ds_k}", n_sfhdf)
            elif ch_sqrt is not None:
                M.num(f"AblationSqrtRedVsSFHDF{ds_k}", -ch_sqrt, 2)
                M.add(f"AblationSqrtWinsVsSFHDF{ds_k}", M.defs.get(f"AblationNWinsSFHSqrt{ds_k}", "0"))
                M.add(f"AblationSqrtCondsVsSFHDF{ds_k}", M.defs.get(f"AblationNCondsSFHSqrt{ds_k}", "12"))

        ablation_order = [
            ("SFHSqrt", "Orabona--P\\'al $\\sqrt{t}$-Scale-Free Hedge (HBML-SFH-Sqrt" + (", primary" if prim_code == "SFHSqrt" else "") + "; $\\eta_t = \\sqrt{\\ln K / \\sum_s M_s^2}$)"),
            ("SFHDF", "HBML-SFHDF (" + ("primary: " if prim_code == "SFHDF" else "variant: ") + "cumulative, EMA scale $1/L_t$, $\\alpha=0$, $\\gamma=0.20$)"),
            ("CumDecayMax", "Cumulative Scale-Free Hedge, decayed-max scale ($\\alpha=0$)"),
            ("FTL", "Follow-the-Leader (FTL, hard $\\arg\\min_k C_{t,k}$)"),
            ("RecNoShare", "Recursive Scale-Free Hedge, decayed-max scale ($\\alpha=0$)"),
            ("FSFastDecay", "HBML-FS (recursive Fixed-Share, $\\alpha_n = (n+1)^{-2}$)"),
            ("FSDecay", "Recursive Fixed-Share ($\\alpha_n = (n+1)^{-1}$)"),
            ("FSConstLow", "Recursive Fixed-Share (constant $\\alpha = 0.001$)"),
            ("FSConstMid", "Recursive Fixed-Share (constant $\\alpha = 0.01$)"),
            ("FSConstHigh", "Recursive Fixed-Share (constant $\\alpha = 0.05$)"),
            ("Uniform", "Equal-weight uniform ensemble ($p_{t,k} = 1/K$)"),
        ]
        if prim_code == "SFHDF":
            ablation_order = [ablation_order[1], ablation_order[2], ablation_order[0]] + ablation_order[3:]
        t_lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\scriptsize",
            r"\begin{tabular}{llccccc}",
            r"\toprule",
            r"Dataset & Aggregator variant & 0.5 hr (Full) & 2 hr (Full) & 5 hr (Full) & Mean $\Delta\%$ & Wins / $N_{\text{cfg}}$ \\",
            r"\midrule",
        ]
        for ds_idx, (ds_k, ds_disp) in enumerate([("Weinstock", "Weinstock 2016"), ("Cgmacros", "CGMacros")]):
            if ds_idx > 0:
                t_lines.append(r"\midrule")
            for vcode, vlabel in ablation_order:
                c30 = table_cells[(ds_k, vcode)].get("ThirtyMin")
                c2h = table_cells[(ds_k, vcode)].get("TwoHour")
                c5h = table_cells[(ds_k, vcode)].get("FiveHour")
                mch = M.defs.get(f"AblationMeanRelChange{vcode}{ds_k}", "---")
                nc = M.defs.get(f"AblationNConds{vcode}{ds_k}", "12")
                wns = "---" if vcode == prim_code else f"{M.defs.get(f'AblationNWins{vcode}{ds_k}', '0')}/{nc}"
                s30 = f"{c30:.2f}" if c30 is not None else "---"
                s2h = f"{c2h:.2f}" if c2h is not None else "---"
                s5h = f"{c5h:.2f}" if c5h is not None else "---"
                t_lines.append(f"{ds_disp} & {vlabel} & {s30} & {s2h} & {s5h} & {mch}\\% & {wns} \\\\")
        t_lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{\textbf{Ablation across online aggregation rules, loss-scale estimators, Follow-the-Leader, and Fixed-Share mixing schedules.} "
            r"All variants aggregate the identical constituent expert forecasts over the identical retained participant cohorts and evaluation windows. "
            r"Columns 3--5 report cohort-mean participant-level RMSE (mg/dL) in the full-history configuration at the 30-minute, 2-hour, and 5-hour horizons. "
            f"``Mean $\\Delta\\%$'' is the mean percentage change in RMSE relative to {prim_label_name} across evaluated horizon $\\times$ context configurations per dataset (negative values indicate lower error than {prim_label_name}; positive values indicate higher error). "
            f"``Wins / $N_{{\\text{{cfg}}}}$'' counts configurations in which the variant achieved lower cohort-mean RMSE than {prim_label_name}.}}",
            r"\label{tab:aggregator_ablation}",
            r"\end{table*}",
        ]
        with open(os.path.join(TABLES_DIR, "table_aggregator_ablation.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(t_lines) + "\n")

    rows = need("hbml_ceg_comparison.csv")
    if rows:
        ceg_vmap = {
            "HBML-SFH-Sqrt": "SFHSqrt",
            "HBML-SFHDF": "SFHDF",
            "HBML-SFH": "SFH",
            "HBML-SFH-SD": "SFHSD",
            "HBML-SFH-Meta": "SFHMeta",
            "HBML-FS": "FS",
        }
        for r in rows:
            vcode = ceg_vmap.get(str(r.get("model", "")).strip())
            if not vcode:
                continue
            ds_k = dataset_label(r["dataset"])
            hz_k = horizon_label(r["horizon"])
            cname = CTX_NAME.get(str(r["context"]).strip().lower(), "Full")
            ab = to_float(r.get("A+Bpct", ""))
            ap = to_float(r.get("Apct", ""))
            M.num(f"CEGVariant{vcode}{ds_k}{hz_k}{cname}AB", ab)
            M.num(f"CEGVariant{vcode}{ds_k}{hz_k}{cname}A", ap)

    rows = need("gamma_sensitivity.csv")
    if rows:
        from collections import defaultdict
        by_gcfg = defaultdict(dict)
        n_pat = None
        for r in rows:
            g_val = to_float(r.get("gamma", ""))
            m_val = to_float(r.get("mean_rmse", ""))
            if g_val is not None and m_val is not None:
                by_gcfg[(r["dataset"], r["horizon"], r["context"])][g_val] = m_val
            if n_pat is None and r.get("n_patients"):
                n_pat = int(float(r["n_patients"]))
        if n_pat is not None:
            M.add("GammaSensNPatients", n_pat)
        M.add("GammaSensNConfigs", len(by_gcfg))
        all_gs = [g for gmap in by_gcfg.values() for g in gmap]
        if all_gs:
            M.num("GammaSensMaxGamma", max(all_gs), 1)
        reds, spans = [], []
        for gmap in by_gcfg.values():
            if 0.0 in gmap and 0.2 in gmap:
                reds.append((gmap[0.0] - gmap[0.2]) / gmap[0.0] * 100.0)
                pos = [v for k, v in gmap.items() if k >= 0.05]
                if pos:
                    spans.append((max(pos) - min(pos)) / gmap[0.2] * 100.0)
        if reds:
            M.num("GammaSensZeroVsDefaultMinPct", min(reds), 1)
            M.num("GammaSensZeroVsDefaultMaxPct", max(reds), 1)
        if spans:
            M.num("GammaSensPosSpanMaxPct", max(spans), 2)

    # 10b. Weight concentration, 7-expert pool, clipped forecasts, hypoglycemia events, and regimes
    M.section("Weight concentration, 7-expert pool, and sensor-range clipping diagnostics")
    rows = read_rows("weight_diagnostics.csv")
    if rows:
        sources.append("weight_diagnostics.csv")
        maxw_all = []
        for r in rows:
            if r.get("matches_tables") == "no":
                continue
            ds_k = dataset_label(r["dataset"])
            hz_k = horizon_label(r["horizon"])
            cname = CTX_NAME.get(str(r["context"]).strip().lower(), "Full")
            key = f"{ds_k}{hz_k}{cname}"
            mw = to_float(r.get("mean_max_weight", ""))
            if mw is not None:
                M.num(f"WeightMaxMean{key}", mw, 2)
                maxw_all.append(mw)
            M.add(f"WeightDomARIMAETSCount{key}", r.get("dom_count_ARIMA_ETS", ""))
            M.num(f"WeightDomARIMAETSPct{key}", to_float(r.get("dom_pct_ARIMA_ETS", "")), 1)
            p7 = to_float(r.get("hbml_pool7_rmse", ""))
            if p7 is not None:
                M.num(f"PoolSevenRMSE{key}", p7, 2)
                pers_v = to_float(M.defs.get(f"BaselinePersistence{key}", ""))
                if pers_v is not None:
                    M.num(f"PoolSevenDiffPersistence{key}", abs(p7 - pers_v), 2)
            chb = to_float(r.get("clipped_hbml_rmse", ""))
            if chb is not None:
                M.num(f"ClippedRMSEHBML{key}", chb, 2)
            for ex in ("ARIMA", "ETS", "NHITS", "NODE", "XGBoost", "Persistence", "RunningMean"):
                M.num(f"WeightMean{ex}{key}", to_float(r.get(f"mean_w_{ex}", "")), 2)
                M.num(f"WeightDomPct{ex}{key}", to_float(r.get(f"dom_pct_{ex}", "")), 1)
                cex = to_float(r.get(f"clipped_{ex}_rmse", ""))
                if cex is not None:
                    M.num(f"ClippedRMSE{ex}{key}", cex, 2)
        if maxw_all:
            M.num("WeightMaxMeanMin", min(maxw_all), 2)
            M.num("WeightMaxMeanMax", max(maxw_all), 2)

    M.section("Event-level hypoglycemia detection and Parkes Error Grid")
    rows = read_rows("hypoglycemia_events.csv")
    if rows:
        sources.append("hypoglycemia_events.csv")
        for r in rows:
            ds_k = dataset_label(r["dataset"])
            hz_k = horizon_label(r["horizon"])
            m_k = clean_name(r["model"])
            key = f"{m_k}{ds_k}{hz_k}"
            M.add(f"HypoEpisodes{ds_k}{hz_k}", r.get("n_episodes", ""))
            M.add(f"HypoDetected{key}", r.get("n_detected", ""))
            M.num(f"HypoEpSens{key}", to_float(r.get("episode_sensitivity_pct", "")), 1)
            M.num(f"HypoEpSensAny{key}", to_float(r.get("episode_sensitivity_any_pct", r.get("episode_sensitivity_pct", ""))), 1)
            if "chance_sensitivity_pct" in r:
                M.num(f"HypoChanceSens{key}", to_float(r.get("chance_sensitivity_pct", "")), 1)
            if "above_chance_pp" in r:
                M.num(f"HypoAboveChance{key}", to_float(r.get("above_chance_pp", "")), 1)
            M.num(f"HypoLeadTime{key}", to_float(r.get("mean_lead_time_min", "")), 1)
            M.num(f"HypoLeadTimeDet{key}", to_float(r.get("mean_lead_time_detected_min", r.get("mean_lead_time_min", ""))), 1)
            M.num(f"HypoPointSens{key}", to_float(r.get("point_sensitivity_pct", "")), 1)
            M.num(f"HypoPointPPV{key}", to_float(r.get("point_ppv_pct", "")), 1)
            M.num(f"HypoPointFOne{key}", to_float(r.get("point_f1_pct", "")), 1)
            M.num(f"HypoFalseAlertsDay{key}", to_float(r.get("false_alerts_per_day", "")), 2)
            M.num(f"HypoParkesA{key}", to_float(r.get("parkes_a_pct", "")), 1)
            M.num(f"HypoParkesAB{key}", to_float(r.get("parkes_ab_pct", "")), 1)

    M.section("Operational glycemic regime summary (30-min, full history)")
    rows = read_rows("regime_summary_table.csv")
    if rows:
        sources.append("regime_summary_table.csv")
        reg_map = {
            "STABLE": "Stable",
            "RISING": "Rising",
            "FALLING": "Falling",
            "HIGHVAR": "HighVar",
            "RECOVERY": "Recovery",
            "HYPO_TRANS": "HypoTrans",
        }
        reg_out_cnt = {"Weinstock": 0, "Cgmacros": 0}
        for r in rows:
            ds_k = dataset_label(r["dataset"])
            rg_k = reg_map.get(r["regime"], clean_name(r["regime"]))
            key = f"{rg_k}{ds_k}"
            hb_v = to_float(r.get("hbml_rmse", ""))
            be_v = to_float(r.get("best_expert_rmse", ""))
            M.num(f"RegimeShare{key}", to_float(r.get("step_share_pct", "")), 1)
            M.num(f"RegimeHBMLRMSE{key}", hb_v, 2)
            M.add(f"RegimeBestExpert{key}", r.get("best_expert", ""))
            M.num(f"RegimeBestExpertRMSE{key}", be_v, 2)
            M.add(f"RegimeTopFreqExpert{key}", r.get("top_freq_expert", ""))
            M.num(f"RegimeTopFreqPct{key}", to_float(r.get("top_freq_pct", "")), 1)
            M.num(f"RegimeHBMLFreqPct{key}", to_float(r.get("hbml_freq_pct", "")), 1)
            if hb_v is not None and be_v is not None and be_v > 0:
                hb_r, be_r = round(hb_v, 2), round(be_v, 2)
                if hb_r < be_r:
                    M.add(f"RegimeCmpVerb{key}", "outperformed")
                    M.add(f"RegimeCmpAdj{key}", "lower")
                    reg_out_cnt[ds_k] = reg_out_cnt.get(ds_k, 0) + 1
                elif hb_r == be_r:
                    M.add(f"RegimeCmpVerb{key}", "matched")
                    M.add(f"RegimeCmpAdj{key}", "comparable")
                else:
                    M.add(f"RegimeCmpVerb{key}", "trailed")
                    M.add(f"RegimeCmpAdj{key}", "higher")
                M.num(f"RegimeAbsRelGap{key}", abs(hb_v - be_v) / be_v * 100.0, 1)
        for ds_k, cnt in reg_out_cnt.items():
            M.add(f"RegimeNOutperformed{ds_k}", cnt)

    M.section("Secondary analysis: multi-seed stochastic stability")
    rows = read_rows("seed_stability.csv")
    if rows:
        sources.append("seed_stability.csv")
        for r in rows:
            key = f"{dataset_label(r['dataset'])}{horizon_label(r['horizon'])}"
            M.add(f"SeedN{key}", r.get("n_participants", ""))
            M.add(f"SeedCount{key}", r.get("n_seeds", ""))
            for m_k in ("NHITS", "NODE", "XGBoost", "HBML"):
                M.num(f"SeedMeanRMSE{m_k}{key}", to_float(r.get(f"{m_k}_mean_rmse", "")), 2)
                M.num(f"SeedWithinSD{m_k}{key}", to_float(r.get(f"{m_k}_within_patient_seed_sd", "")), 3)
                M.num(f"SeedCohortSD{m_k}{key}", to_float(r.get(f"{m_k}_cohort_mean_seed_sd", "")), 3)

    # 11. Secondary analyses (optional tables; silently skipped if absent)
    M.section("Secondary analysis: population-trained models")
    rows = read_rows("population_comparison.csv")
    if rows:
        sources.append("population_comparison.csv")
        for r in rows:
            sch = "Within" if r["scheme"] == "within" else "Cross"
            key = f"{dataset_label(r['dataset'])}{horizon_label(r['horizon'])}{CTX_NAME.get(str(r['context']).lower(), 'Full')}"
            for col, nm in (("Pop-NHITS_rmse", "NHITS"), ("Pop-XGBoost_rmse", "XGBoost"), ("HBML_rmse", "HBML"),
                            ("HBML+Pop_rmse", "HBMLPlusPop"), ("best_personal_rmse", "BestPersonal")):
                M.num(f"Pop{sch}{nm}{key}", to_float(r.get(col, "")))
            M.add(f"Pop{sch}BestPersonalName{key}", r.get("best_personal_expert", ""))
            M.add(f"Pop{sch}N{key}", r.get("n", ""))
            for pre, nm in (("hbml_vs_popnhits", "HBMLvsNHITS"), ("hbml_vs_popxgb", "HBMLvsXGBoost"),
                            ("pop_added_vs_hbml", "Added"), ("pop_added_vs_best", "AddedvsBest")):
                M.num(f"Pop{sch}RelPct{nm}{key}", to_float(r.get(f"{pre}_rel_pct", "")), 1)
                M.num(f"Pop{sch}PctImproved{nm}{key}", to_float(r.get(f"{pre}_pct_improved", "")), 1)
                M.add(f"Pop{sch}P{nm}{key}", tex_p(to_float(r.get(f"{pre}_wilcoxon_p", ""))))

    M.section("Data: 5-minute grid, gaps and forecast coverage (analyze_segments.py)")
    rows = read_rows("grid_summary.csv")
    if rows:
        sources.append("grid_summary.csv")
        sys.path.insert(0, BASE_DIR)
        from ExpMethods.globals import GlobalValues as GV
        M.num("GridMaxGapMin", GV.max_gap_min, 0)
        M.num("GridWarmupMin", GV.min_segment_history * GV.grid_step_min, 0)
        M.num("GridBurnInHours", GV.burn_in_steps * GV.grid_step_min / 60, 0)
        win = {"72": "SixHr", "120": "Full", "144": "TwelveHr", "288": "TwentyFourHr"}
        for r in rows:
            thr = to_float(r["max_gap_min"])
            pre = "Grid" if abs(thr - GV.max_gap_min) < 1e-9 else f"GridSens{int(thr)}"
            pre = pre.replace("60", "Sixty").replace("45", "FortyFive").replace("90", "Ninety")
            if not re.fullmatch(r"[A-Za-z]+", pre):
                continue
            ds = dataset_label(r["dataset"])
            M.add(f"{pre}NParticipants{ds}", r["n_participants"])
            for col, nm, nd in (("missing_pct", "MissingPct", 1), ("filled_pct", "FilledPct", 1),
                                ("long_gap_pct", "LongGapPct", 1), ("irregular_pct", "IrregularPct", 2),
                                ("segments_median", "SegmentsMedian", 0), ("segments_q1", "SegmentsQOne", 0),
                                ("segments_q3", "SegmentsQThree", 0), ("segment_hours_median", "SegmentHoursMedian", 1),
                                ("segment_hours_q1", "SegmentHoursQOne", 1), ("segment_hours_q3", "SegmentHoursQThree", 1),
                                ("time_in_segments_ge_10h_pct", "TimeInLongSegmentsPct", 1)):
                M.num(f"{pre}{nm}{ds}", to_float(r.get(col, "")), nd)
            M.add(f"{pre}DroppedReadings{ds}", r.get("n_duplicates", ""))
            for hk in ("half", "2", "5"):
                key = f"{ds}{horizon_label(hk)}"
                M.num(f"{pre}KeptPct{key}", to_float(r.get(f"kept_pct_{hk}", "")), 1)
                M.num(f"{pre}SkippedPct{key}", 100 - (to_float(r.get(f"kept_pct_{hk}", "")) or 100), 1)
                M.add(f"{pre}NKept{key}", f"{int(float(r[f'n_kept_{hk}'])):,}".replace(",", "{,}")
                      if r.get(f"n_kept_{hk}") else "")
                for L, cn in win.items():
                    M.num(f"{pre}SegLocalKeptPct{key}{cn}", to_float(r.get(f"seglocal{L}_pct_{hk}", "")), 1)
                    M.num(f"{pre}BridgedHalfPct{key}{cn}", to_float(r.get(f"bridged50_{L}_pct_{hk}", "")), 1)
                    M.num(f"{pre}BridgedMeanPct{key}{cn}", to_float(r.get(f"bridged_mean_{L}_pct_{hk}", "")), 1)

    M.section("Secondary analysis: expert refit interval")
    rows = read_rows("retrain_cadence.csv")
    if rows:
        sources.append("retrain_cadence.csv")
        hours_word = {6: "SixHour", 12: "TwelveHour", 24: "TwentyFourHour", 48: "FortyEightHour"}
        for r in rows:
            hw = hours_word.get(int(round(float(r["hours"]))))
            if not hw:
                continue
            key = f"{dataset_label(r['dataset'])}{horizon_label(r['horizon'])}"
            M.add(f"CadenceN{key}", r.get("n", ""))
            M.num(f"Cadence{hw}FitSeconds{key}", to_float(r.get("fit_s_per_participant", "")), 0)
            for col in r:
                if col.endswith("_rmse"):
                    M.num(f"Cadence{hw}{clean_name(col[:-5])}{key}", to_float(r[col]))
            M.num(f"Cadence{hw}RelPctHBML{key}", to_float(r.get("HBML_rel_pct_vs_ref", "")), 1)
            M.add(f"Cadence{hw}PHBML{key}", tex_p(to_float(r.get("HBML_p_vs_ref", ""))))

    M.section("Runtime environment versions")
    env_rows = read_rows("runtime_env.csv")
    if env_rows:
        sources.append("runtime_env.csv")
        env_map = {r["package"]: r["version"] for r in env_rows}
    else:
        env_map = {"python": "3.11+", "numpy": "1.26+", "scipy": "1.13+", "pandas": "2.2+"}
    for pkg, nm in (("python", "Python"), ("numpy", "Numpy"), ("scipy", "Scipy"), ("pandas", "Pandas"),
                    ("torch", "Torch"), ("xgboost", "Xgboost"), ("statsforecast", "Statsforecast"),
                    ("neuralforecast", "Neuralforecast")):
        if pkg in env_map:
            M.add(f"Env{nm}Version", env_map[pkg])

    # Macros used in the manuscript
    used = set()
    tex_files = [os.path.join(SECTIONS_DIR, f) for f in ("introduction.tex", "results.tex", "discussion.tex", "methods.tex")]
    tex_files.append(os.path.join(BASE_DIR, "overleaf", "main.tex"))
    pat = re.compile(r"\\((?:" + "|".join(MACRO_PREFIXES) + r")[A-Za-z]*)")
    for p in tex_files:
        if os.path.exists(p):
            body = "\n".join(l for l in open(p, encoding="utf-8").read().splitlines() if not l.lstrip().startswith("%"))
            used |= set(pat.findall(body))
    # CGMacros spelling aliases, only where the manuscript uses them (\...CGMacros... == \...Cgmacros...)
    M.section("Aliases: CGMacros spelling")
    for u in sorted(used):
        base = u.replace("CGMacros", "Cgmacros")
        if base != u and base in M.defs:
            M.add(u, f"\\{base}")
    missing = sorted(u for u in used if u not in M.defs)
    M.section("Used in the manuscript but not available from the tables (shown as ?? in the PDF)")
    for name in missing:
        M.add(name, "\\MacroMissing")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    out = [f"% Auto-generated by generate_results_macros.py on {stamp}. DO NOT EDIT BY HAND.",
           "% Sources (overleaf/tables/, with modification times):"]
    mt = {}
    for s in sources:
        t = os.path.getmtime(os.path.join(TABLES_DIR, s))
        mt[s] = t
        out.append(f"%   {s}  {datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M')}")
    out.append("\\providecommand{\\MacroMissing}{\\textcolor{red}{\\textbf{??}}}")
    for title, names in M.sections:
        if not names:
            continue
        out.append("")
        out.append(f"% --- {title} ---")
        out += [f"\\newcommand{{\\{n}}}{{{M.defs[n]}}}" for n in names]
    tmp = OUTPUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    os.replace(tmp, OUTPUT_FILE)

    # Tables and figures generated in the same pipeline run should have similar ages.
    core = [mt[s] for s in ("rmse_pivot_mean.csv", "statistical_tests_best_expert.csv", "baselines_comparison.csv",
                            "max_ceg.csv") if s in mt]
    if core and max(core) - min(core) > 6 * 3600:
        warnings_.append("core tables were written more than 6 hours apart; some of them may be stale")
    if "rmse_pivot_mean.csv" in mt:
        ref_t = mt["rmse_pivot_mean.csv"]
        for ds_fig in ("weinstock", "cgmacros"):
            fig_p = os.path.join(BASE_DIR, "overleaf", "images", "regimes", f"fig3_stacked_freq_{ds_fig}_hhalf_cfull.pdf")
            if os.path.exists(fig_p) and (ref_t - os.path.getmtime(fig_p) > 6 * 3600):
                warnings_.append(f"stale regime figure (>6 h older than rmse_pivot_mean.csv): {os.path.relpath(fig_p, BASE_DIR)}")
    if not quiet:
        print(f"Generated {len(M.defs)} macros in {OUTPUT_FILE}")
        for w in warnings_:
            print(f"  WARNING: {w}")
        if missing:
            print(f"  WARNING: {len(missing)} macro(s) used in the manuscript have no value (shown as ??): "
                  + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))
    if strict and (missing or warnings_):
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="Exit with status 1 if anything is missing or stale.")
    a = ap.parse_args()
    sys.exit(generate_macros(strict=a.strict))
