#!/usr/bin/env python3
"""calculate_statistics.py — Rigorous participant-level statistical evaluation.

Calculates paired statistical tests comparing HBML against constituent experts
across all 24 experimental configurations (dataset × horizon × context):
  - Two-sided Wilcoxon signed-rank test (W statistic and p-value)
  - Two-sided paired Student's t-test (t statistic, df, p-value)
  - 95% bootstrap confidence intervals for both mean and median paired differences
  - Individual participant win / tie / loss counts and proportion of participants improved
  - Relative percentage RMSE reduction

Outputs:
  - overleaf/tables/statistical_tests_best_expert.csv
  - overleaf/tables/statistical_tests_all_experts.csv
  - overleaf/tables/table_statistical_tests.tex
  - Appends statistical macros to overleaf/sections/generated_macros.tex
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel, bootstrap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(BASE_DIR, "overleaf", "tables")
SECTIONS_DIR = os.path.join(BASE_DIR, "overleaf", "sections")

DATASETS = ["weinstock", "cgmacros"]
HORIZONS = [0.5, 2.0, 5.0]
CONTEXTS = ["6", "12", "24", "full"]

HORIZON_MAP = {0.5: "0p5", 2.0: "2", 5.0: "5"}
HORIZON_LABEL = {0.5: "0.5 hr (30 min)", 2.0: "2 hr", 5.0: "5 hr"}
CONTEXT_LABEL = {"6": "6 hr", "12": "12 hr", "24": "24 hr", "full": "Full history"}
DATASET_LABEL = {"weinstock": "Weinstock", "cgmacros": "CGMacros"}

MACRO_HORIZON = {0.5: "ThirtyMin", 2.0: "TwoHour", 5.0: "FiveHour"}
MACRO_CONTEXT = {"6": "SixHr", "12": "TwelveHr", "24": "TwentyFourHr", "full": "Full"}


def clean_num(val):
    """Strip LaTeX markup and convert to float."""
    s = re.sub(r"[^0-9.]", "", str(val))
    return float(s) if s else np.nan


def load_group_means():
    """Load group means from rmse_pivot_mean.csv."""
    path = os.path.join(TABLES_DIR, "rmse_pivot_mean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    
    means = {}
    for _, row in df.iterrows():
        ds = str(row["dataset"]).strip().lower()
        h = float(row["horizon"])
        model = str(row["model"]).strip()
        for ctx in CONTEXTS:
            col = f"mean.{ctx}"
            if col in row and pd.notna(row[col]):
                means[(ds, h, ctx, model)] = clean_num(row[col])
    return means


def compute_bootstrap_ci(data, stat_func=np.mean, n_resamples=10000, ci=0.95):
    """Compute bootstrap CI for a 1D numpy array."""
    data = data[~np.isnan(data)]
    if len(data) < 5:
        return np.nan, np.nan
    try:
        res = bootstrap((data,), stat_func, confidence_level=ci, n_resamples=n_resamples, method="percentile", random_state=42)
        return res.confidence_interval.low, res.confidence_interval.high
    except Exception:
        # Fallback to numpy percentile bootstrap
        rng = np.random.default_rng(42)
        boot_dist = [stat_func(rng.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)]
        alpha = (1.0 - ci) / 2.0
        return np.percentile(boot_dist, alpha * 100), np.percentile(boot_dist, (1.0 - alpha) * 100)


def format_p_value(p):
    """Format p-value for scientific publication."""
    if pd.isna(p):
        return "N/A"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def run_paired_analysis():
    group_means = load_group_means()
    
    best_expert_records = []
    all_expert_records = []
    macros = []
    
    for ds in DATASETS:
        for h in HORIZONS:
            h_code = HORIZON_MAP[h]
            for ctx in CONTEXTS:
                dev_path = os.path.join(TABLES_DIR, f"rmse_dev_{ds}_h{h_code}_c{ctx}.csv")
                if not os.path.exists(dev_path):
                    print(f"Warning: file not found: {dev_path}")
                    continue
                
                dev_df = pd.read_csv(dev_path, index_col="id")
                n_participants = len(dev_df)
                
                # Reconstruct participant-level RMSE
                raw_rmse = pd.DataFrame(index=dev_df.index)
                for col in dev_df.columns:
                    key = (ds, h, ctx, col)
                    if key in group_means:
                        col_mean = group_means[key]
                        raw_rmse[col] = col_mean * (1.0 + dev_df[col] / 100.0)
                    else:
                        print(f"Missing group mean for {key}")
                
                if "HBML" not in raw_rmse.columns:
                    print(f"HBML not in columns for {ds} h={h} ctx={ctx}")
                    continue
                
                hbml_rmse = raw_rmse["HBML"].values
                hbml_mean = np.mean(hbml_rmse)
                hbml_sd = np.std(hbml_rmse, ddof=1)
                hbml_median = np.median(hbml_rmse)
                hbml_iqr = np.percentile(hbml_rmse, 75) - np.percentile(hbml_rmse, 25)
                
                experts = [m for m in raw_rmse.columns if m != "HBML"]
                
                # Identify Best Constituent Expert (lowest group mean RMSE)
                expert_means = {m: np.mean(raw_rmse[m].values) for m in experts}
                best_expert_name = min(expert_means, key=expert_means.get)
                
                for exp_name in experts:
                    is_best = (exp_name == best_expert_name)
                    exp_rmse = raw_rmse[exp_name].values
                    exp_mean = np.mean(exp_rmse)
                    exp_sd = np.std(exp_rmse, ddof=1)
                    exp_median = np.median(exp_rmse)
                    exp_iqr = np.percentile(exp_rmse, 75) - np.percentile(exp_rmse, 25)
                    
                    # Paired difference: positive means HBML has lower error (improvement)
                    diff = exp_rmse - hbml_rmse
                    mean_diff = np.mean(diff)
                    median_diff = np.median(diff)
                    se_diff = np.std(diff, ddof=1) / np.sqrt(n_participants)
                    
                    # 95% CIs
                    ci_mean_low, ci_mean_high = compute_bootstrap_ci(diff, stat_func=np.mean)
                    ci_med_low, ci_med_high = compute_bootstrap_ci(diff, stat_func=np.median)
                    
                    # Statistical hypothesis tests (two-sided)
                    # Wilcoxon test: test if median difference between HBML and Expert is zero
                    try:
                        w_res = wilcoxon(hbml_rmse, exp_rmse, alternative="two-sided")
                        w_stat = w_res.statistic
                        w_p = w_res.pvalue
                    except Exception:
                        w_stat, w_p = np.nan, np.nan
                    
                    # Paired t-test
                    try:
                        t_res = ttest_rel(exp_rmse, hbml_rmse)
                        t_stat = t_res.statistic
                        t_p = t_res.pvalue
                    except Exception:
                        t_stat, t_p = np.nan, np.nan
                    
                    # Participant win / tie / loss
                    n_improved = int(np.sum(diff > 0.001))
                    n_tied = int(np.sum(np.isclose(diff, 0.0, atol=0.001)))
                    n_worse = int(np.sum(diff < -0.001))
                    pct_improved = (n_improved / n_participants) * 100.0
                    
                    # Relative mean percentage reduction
                    rel_reduction = ((exp_mean - hbml_mean) / exp_mean) * 100.0
                    
                    record = {
                        "dataset": ds,
                        "horizon": h,
                        "context": ctx,
                        "n_participants": n_participants,
                        "expert": exp_name,
                        "is_best_expert": is_best,
                        "hbml_mean": hbml_mean,
                        "hbml_sd": hbml_sd,
                        "hbml_median": hbml_median,
                        "hbml_iqr": hbml_iqr,
                        "expert_mean": exp_mean,
                        "expert_sd": exp_sd,
                        "expert_median": exp_median,
                        "expert_iqr": exp_iqr,
                        "paired_diff_mean": mean_diff,
                        "paired_diff_se": se_diff,
                        "paired_diff_median": median_diff,
                        "ci95_mean_low": ci_mean_low,
                        "ci95_mean_high": ci_mean_high,
                        "ci95_med_low": ci_med_low,
                        "ci95_med_high": ci_med_high,
                        "wilcoxon_w": w_stat,
                        "wilcoxon_p": w_p,
                        "ttest_t": t_stat,
                        "ttest_p": t_p,
                        "n_improved": n_improved,
                        "n_tied": n_tied,
                        "n_worse": n_worse,
                        "pct_improved": pct_improved,
                        "rel_reduction_pct": rel_reduction,
                    }
                    
                    all_expert_records.append(record)
                    if is_best:
                        best_expert_records.append(record)
                        
                        # Generate LaTeX macros for key benchmarks
                        m_ds = DATASET_LABEL[ds]
                        m_h = MACRO_HORIZON[h]
                        m_ctx = MACRO_CONTEXT[ctx]
                        
                        macros.append(f"\\newcommand{{\\WilcoxonP{m_ds}{m_h}{m_ctx}}}{{{format_p_value(w_p)}}}")
                        macros.append(f"\\newcommand{{\\PctImproved{m_ds}{m_h}{m_ctx}}}{{{pct_improved:.1f}}}")
                        macros.append(f"\\newcommand{{\\NImproved{m_ds}{m_h}{m_ctx}}}{{{n_improved}}}")
                        macros.append(f"\\newcommand{{\\NTotal{m_ds}{m_h}{m_ctx}}}{{{n_participants}}}")
                        macros.append(f"\\newcommand{{\\PairedDiffMean{m_ds}{m_h}{m_ctx}}}{{{mean_diff:.2f}}}")
                        macros.append(f"\\newcommand{{\\PairedDiffMedian{m_ds}{m_h}{m_ctx}}}{{{median_diff:.2f}}}")
                        macros.append(f"\\newcommand{{\\BestExpert{m_ds}{m_h}{m_ctx}}}{{{best_expert_name}}}")

    df_best = pd.DataFrame(best_expert_records)
    df_all = pd.DataFrame(all_expert_records)
    
    # Save CSVs
    best_csv_path = os.path.join(TABLES_DIR, "statistical_tests_best_expert.csv")
    all_csv_path = os.path.join(TABLES_DIR, "statistical_tests_all_experts.csv")
    df_best.to_csv(best_csv_path, index=False)
    df_all.to_csv(all_csv_path, index=False)
    print(f"Saved best-expert tests to {best_csv_path}")
    print(f"Saved all-expert tests to {all_csv_path}")
    
    # Generate publication-grade LaTeX table
    tex_path = os.path.join(TABLES_DIR, "table_statistical_tests.tex")
    generate_latex_table(df_best, tex_path)
    print(f"Saved LaTeX statistical table to {tex_path}")
    
    # Append macros to generated_macros.tex
    macros_file = os.path.join(SECTIONS_DIR, "generated_macros.tex")
    if os.path.exists(macros_file):
        with open(macros_file, "r") as f:
            existing_content = f.read()
        
        # Filter out macros that might already be defined
        existing_lines = set(existing_content.splitlines())
        new_macros = [m for m in macros if m not in existing_lines]
        
        if new_macros:
            with open(macros_file, "a") as f:
                f.write("\n\n% --- Paired Participant-Level Statistical Tests ---\n")
                f.write("\n".join(new_macros) + "\n")
            print(f"Appended {len(new_macros)} statistical macros to {macros_file}")


def generate_latex_table(df_best, output_tex):
    """Generate a clean, professional LaTeX table of paired statistical tests."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Horizon & Context & $N$ & Best Expert & Paired $\Delta$ Mean [95\% CI] & Paired $\Delta$ Med [95\% CI] & Improved & Wilcoxon $p$ \\")
    lines.append(r"\midrule")
    
    current_ds = None
    for _, row in df_best.iterrows():
        ds = DATASET_LABEL[row["dataset"]]
        h = HORIZON_LABEL[row["horizon"]]
        ctx = CONTEXT_LABEL[row["context"]]
        n = int(row["n_participants"])
        best_exp = row["expert"]
        
        mean_diff = row["paired_diff_mean"]
        ci_m_low = row["ci95_mean_low"]
        ci_m_high = row["ci95_mean_high"]
        
        med_diff = row["paired_diff_median"]
        ci_med_low = row["ci95_med_low"]
        ci_med_high = row["ci95_med_high"]
        
        n_imp = int(row["n_improved"])
        pct_imp = row["pct_improved"]
        p_val = row["wilcoxon_p"]
        p_str = format_p_value(p_val)
        if p_val < 0.05:
            p_str = f"\\textbf{{{p_str}}}"
            
        if current_ds is not None and current_ds != ds:
            lines.append(r"\midrule")
        current_ds = ds
        
        line = (
            f"{ds} & {h} & {ctx} & {n} & {best_exp} & "
            f"{mean_diff:+.2f} [{ci_m_low:+.2f}, {ci_m_high:+.2f}] & "
            f"{med_diff:+.2f} [{ci_med_low:+.2f}, {ci_med_high:+.2f}] & "
            f"{n_imp}/{n} ({pct_imp:.1f}\\%) & {p_str} \\\\"
        )
        lines.append(line)
        
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\textbf{Participant-level paired statistical comparisons between HBML and the best constituent expert.} "
        r"For each configuration, the best constituent expert is the non-HBML model achieving the lowest cohort mean RMSE. "
        r"Paired differences ($\Delta = \text{RMSE}_{\text{best}} - \text{RMSE}_{\text{HBML}}$) are computed per participant, "
        r"where positive values denote HBML superiority. Confidence intervals (95\% CI) are computed via 10,000 bootstrap resamples. "
        r"Hypothesis testing is evaluated via two-sided paired Wilcoxon signed-rank tests ($p < 0.05$ bolded). "
        r"Participant improvement counts ($n_{\text{improved}}/N$) report individuals where HBML achieved strictly lower RMSE.}"
    )
    lines.append(r"\label{tab:statistical_tests}")
    lines.append(r"\end{table*}")
    
    with open(output_tex, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_paired_analysis()
