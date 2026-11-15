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
MACRO_PREFIXES = ("RMSE", "CEG", "Wilcoxon", "Ttest", "PctImproved", "PctWorse", "NImproved", "NWorse", "NTotal",
                  "PairedDiff", "BestExpert", "StatAll", "StatWeinstock", "StatCgmacros", "StatCGMacros",
                  "Baseline", "Vol[A-Z]", "Timing", "NWeinstock", "NCgmacros", "NCGMacros", "NConfigs", "NTotalParticipants")


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
        M.num(f"RMSEReduction{frag}{hz}{cname}", (bv - hv) / bv * 100.0, 1)
        M.num(f"RMSEDiff{frag}{hz}{cname}", bv - hv)
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
    if rows:
        for r in rows:
            key = f"{dataset_label(r['dataset'])}{horizon_label(r['horizon'])}{CTX_NAME.get(str(r['context']).strip().lower(), 'Full')}"
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
                              ("PairedDiffMedianLow", "ci95_med_low"), ("PairedDiffMedianHigh", "ci95_med_high")]:
                v = f(col)
                if v is not None:
                    M.add(f"{stem}{key}", f"{v:+.2f}")
            n = to_float(r.get("n_participants", ""))
            if n is not None:
                n_by_ds.setdefault(dataset_label(r["dataset"]), []).append(int(n))
    rows = need("statistical_summary.csv")
    if rows:
        key_name = {"n_configs": "NConfigs", "n_hbml_lowest_mean": "NHBMLLowestMean",
                    "n_mean_diff_positive": "NMeanDiffPositive", "n_sig_raw_hbml_better": "NSigRawBetter",
                    "n_sig_raw_hbml_worse": "NSigRawWorse", "n_sig_bh_hbml_better": "NSigBHBetter",
                    "n_sig_bh_hbml_worse": "NSigBHWorse", "n_not_sig_bh": "NNotSigBH",
                    "pct_improved_min": "PctImprovedMin", "pct_improved_max": "PctImprovedMax",
                    "n_participants_min": "NParticipantsMin", "n_participants_max": "NParticipantsMax",
                    "n_shapiro_reject": "NShapiroReject", "n_boot": "NBoot", "fdr_q": "FDRq"}
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
        M.add("NTotalParticipants", sum(max(v) for v in n_by_ds.values()))
    M.add("NConfigs", "24")

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
            for b, bname in [("persistence", "Persistence"), ("climatology", "Climatology"), ("uniform", "Uniform")]:
                M.num(f"Baseline{bname}{ds}{hz}{cname}", to_float(r.get(f"{b}_rmse", "")))
                red = to_float(r.get(f"hbml_reduction_vs_{b}_pct", ""))
                M.num(f"BaselineReduction{bname}{ds}{hz}{cname}", red, 1)
                M.num(f"BaselinePctImproved{bname}{ds}{hz}{cname}", to_float(r.get(f"pct_improved_vs_{b}", "")), 1)
                M.add(f"BaselineP{bname}{ds}{hz}{cname}", tex_p(to_float(r.get(f"wilcoxon_p_vs_{b}", ""))))
                if red is not None and red == red:
                    rng.setdefault((bname, ds, hz), []).append(red)
        for (bname, ds, hz), v in rng.items():
            M.num(f"BaselineReduction{bname}{ds}{hz}Min", min(v), 1)
            M.num(f"BaselineReduction{bname}{ds}{hz}Max", max(v), 1)
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

    # Tables generated in the same pipeline run should have similar ages.
    core = [mt[s] for s in ("rmse_pivot_mean.csv", "statistical_tests_best_expert.csv", "baselines_comparison.csv",
                            "max_ceg.csv") if s in mt]
    if core and max(core) - min(core) > 6 * 3600:
        warnings_.append("core tables were written more than 6 hours apart; some of them may be stale")
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
