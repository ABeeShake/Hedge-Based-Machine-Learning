import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from linecache import getline
from itertools import product
from scipy.stats import wilcoxon, bootstrap

import sys
sys.path.append(os.path.abspath('.'))

import ExpMethods.utils as util
from ExpMethods.globals import GlobalValues
import ExpMethods.simulate as sim 
import ExpMethods.utils as utils
import ExpMethods.visualizations as viz
from ExpMethods.simulate import MixingMethods, AlphaMethods

# ---- Shared Plot Formatting Helpers ----
_DATASET_DISPLAY = {"weinstock": "Weinstock", "cgmacros": "CGMacros"}
_HORIZON_DISPLAY = {"half": "0.5", "0.5": "0.5", "2": "2", "5": "5"}

def fmt_dataset(d):
    """Return display name for a dataset key."""
    return _DATASET_DISPLAY.get(d.lower(), d.capitalize())

def fmt_horizon(h):
    """Return display string for a horizon value."""
    return _HORIZON_DISPLAY.get(str(h), str(h))

def make_plot_title(dataset, horizon):
    """Format the standard plot title: 'Dataset (h hr Horizon)'."""
    return f"{fmt_dataset(dataset)} ({fmt_horizon(horizon)} hr Horizon)"

# ---- Global rcParams ----
plt.rcParams.update(GlobalValues.plot_params)

if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Generate figures and tables.")
    _parser.add_argument(
        "--rmse_only", action="store_true",
        help="Skip all non-RMSE analyses (CEG, regrets, visualizations)."
    )
    _parser.add_argument(
        "--median", action="store_true",
        help="Use median and IQR instead of mean and SD for the RMSE table."
    )
    args = _parser.parse_args()

    # Post Processing
    ## find corrupted IDs for Weinstock
    print("Finding corrupted IDs for Weinstock...")
    all_files = glob("../weinstock/**/*_forecasts.csv", recursive = True)
    corrupted_paths = [file for file in all_files if re.search(r"^[0\.,]+$", getline(file, 23))]
    corrupted_files = [re.findall(r"../(.*)/h-(.{1,4})hr/context-(.{1,4})hr/.*/(\d{3})", file)[0] for file in corrupted_paths]
    comparison_cols = ["dataset","horizon","context","id"]
    corrupted_df = pd.DataFrame(corrupted_files, columns = comparison_cols)
    print(corrupted_df)

    # RMSE Analysis
    print("Running RMSE Analysis...")
    def is_outlier(x):
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5*iqr
        upper = q3 + 1.5*iqr
        return np.logical_or(x > upper, x < lower).values

    rmse_dfs = []
    for ds in ["weinstock", "cgmacros"]:
        path = f"../{ds}/rmse.csv"
        if os.path.exists(path):
            rmse_dfs.append(pd.read_csv(path))
    if not rmse_dfs:
        raise FileNotFoundError("No rmse.csv files found in ../weinstock or ../cgmacros")
    rmse = pd.concat(rmse_dfs, ignore_index=True)
    rmse = rmse.dropna(axis=0)
    rmse.id = rmse.id.astype(str).str.pad(width=3,side="left",fillchar="0")

    min_idx = rmse.groupby(["dataset","horizon","context","model","id"])["rmse"].idxmin()
    min_rmse = rmse.loc[min_idx].drop(["eta"],axis=1)
    na_idx = min_rmse.loc[min_rmse.rmse.isna(),"id"]
    min_rmse = min_rmse.loc[np.logical_not(min_rmse.id.isin(na_idx))]
    min_rmse = min_rmse.loc[np.logical_not(min_rmse.model.isin(["HBML (c)","HBML (d)"])),:]    
    # Remap both old and new HBML primary-model names to the canonical "HBML" label
    min_rmse.loc[min_rmse.model == "HBML-SFHDF", "model"] = "HBML"
    min_rmse.loc[min_rmse.horizon == "half","horizon"] = "0.5"

    cleaned_rmse = min_rmse

    # Exclude individuals with abnormally high HBML RMSE within each
    # (dataset, horizon, context) slab, using per-(horizon, context) thresholds.
    # All models are dropped for flagged IDs so group sizes stay balanced.
    # Keys are (horizon, context) string tuples; default fallback is 200.
    _rmse_thresholds = {
        ("0.5", "6"):    200, ("0.5", "12"): 200, ("0.5", "24"): 200, ("0.5", "full"): 200,
        ("2",   "6"):    200, ("2",   "12"): 200, ("2",   "24"): 200, ("2",   "full"): 200,
        ("5",   "6"):    200, ("5",   "12"): 200, ("5",   "24"): 200, ("5",   "full"): 200,
    }
    _hbml_rows = cleaned_rmse[cleaned_rmse["model"] == "HBML"].copy()
    _hbml_rows["_threshold"] = _hbml_rows.apply(
        lambda r: _rmse_thresholds.get((str(r["horizon"]), str(r["context"])), 200), axis=1
    )
    _flagged = _hbml_rows[_hbml_rows["rmse"] > _hbml_rows["_threshold"]]
    _high_rmse_ids = _flagged.groupby(["dataset", "horizon", "context"])["id"].apply(set)
    if not _high_rmse_ids.empty:
        _keep = pd.Series(True, index=cleaned_rmse.index)
        for (ds, h, ctx), bad_ids in _high_rmse_ids.items():
            threshold = _rmse_thresholds.get((str(h), str(ctx)), 200)
            _slab = (
                (cleaned_rmse["dataset"] == ds) &
                (cleaned_rmse["horizon"] == h) &
                (cleaned_rmse["context"] == ctx) &
                (cleaned_rmse["id"].isin(bad_ids))
            )
            _keep[_slab] = False
            print(f"  Excluding {len(bad_ids)} individual(s) from {ds} h={h} ctx={ctx} "
                  f"(HBML RMSE > {threshold}): {sorted(bad_ids)}")
        cleaned_rmse = cleaned_rmse[_keep].copy()
    print(f"  {cleaned_rmse['id'].nunique()} individuals retained after RMSE filter.")

    # Compute both mean/SD and median/IQR stats; select based on --median flag.
    _rmse_mean_stats = cleaned_rmse.groupby(["dataset","horizon","context","model"]).rmse.apply(
        lambda x: pd.Series(
            {"mean": np.round(x.mean(), 2),
             "sd":   np.round(x.std(ddof=0), 2)},
            index=["mean", "sd"]
        )
    ).reset_index().pivot(
        index=["dataset","horizon","context","model"],
        columns=["level_4"], values="rmse"
    ).reset_index()

    _rmse_median_stats = cleaned_rmse.groupby(["dataset","horizon","context","model"]).rmse.apply(
        lambda x: pd.Series(
            {"median": np.round(x.median(), 2),
             "iqr":    np.round(x.quantile(0.75) - x.quantile(0.25), 2)},
            index=["median", "iqr"]
        )
    ).reset_index().pivot(
        index=["dataset","horizon","context","model"],
        columns=["level_4"], values="rmse"
    ).reset_index()

    cleaned_rmse_stats = _rmse_median_stats if args.median else _rmse_mean_stats
    _stat_col  = "median" if args.median else "mean"
    _disp_col  = "iqr"    if args.median else "sd"
    _out_file  = "rmse_pivot_median.csv" if args.median else "rmse_pivot_mean.csv"

    def format_rmse(group):
        min_val = group[_stat_col].min()
        expert_group = group[group["model"] != "HBML"]
        min_expert = expert_group[_stat_col].min() if not expert_group.empty else np.nan

        def format_cell(row):
            val = f"{row[_stat_col]:.2f}"
            if row[_stat_col] == min_val:
                val = f"\\textbf{{{val}}}"
            return val

        # Name columns with the actual stat names so set_levels ordering is correct
        group[f"{_stat_col}_str"] = group.apply(format_cell, axis=1)
        group[f"{_disp_col}_str"] = group[_disp_col].apply(lambda x: f"{x:.2f}")
        return group

    cleaned_rmse_stats = cleaned_rmse_stats.groupby(["dataset", "horizon", "context"], group_keys=False).apply(format_rmse)

    model_order = dict(
        ARIMA=0, ETS=1, NHITS=2, NODE=3, XGBoost=4,
        HBML=5, **{"HBML-AE (clip-loss)": 6, "HBML-AE (sqrt)": 7, "HBML-AE (ema0.2)": 8,
                   "HBML-AE (ema0.5)": 9, "HBML-AE (ema0.8)": 10, "HBML-AE (ema1)": 11,
                   "HBML-VS": 12, "HBML-AH": 13, "HBML-SFH": 14, "HBML-SFHDF": 15}
    )
    horizon_order = {"0.5":0,"2":1,"5":2}
    dataset_order = {"weinstock": 0, "cgmacros": 1}
    context_order = {"6":0,"12":1,"24":2,"full":3}

    _s_col = f"{_stat_col}_str"  # e.g. "mean_str" or "median_str"
    _d_col = f"{_disp_col}_str"  # e.g. "sd_str"   or "iqr_str"
    cleaned_pivot = cleaned_rmse_stats.pivot(
        index=["horizon","model","dataset"],
        columns=["context"],
        values=[_s_col, _d_col]
    )
    # Explicit column flattening: match _s_col / _d_col by name, not by sort position.
    # This is immune to set_levels alphabetical-order swapping.
    def _flatten_col(x):
        if not isinstance(x, tuple) or x[1] == "":
            return x[0] if isinstance(x, tuple) else x
        if x[0] == _s_col:    # e.g. ("median_str", "6") → "median.6"
            return f"{_stat_col}.{x[1]}"
        if x[0] == _d_col:    # e.g. ("iqr_str",    "6") → "iqr.6"
            return f"{_disp_col}.{x[1]}"
        return ".".join(str(v) for v in x)

    cleaned_pivot = cleaned_pivot.sort_values(
        by=["dataset","horizon","model"],
        key=lambda x: x.map(dataset_order | horizon_order | model_order)
    ).reset_index()
    cleaned_pivot['h_freq'] = cleaned_pivot.groupby('horizon')['horizon'].transform('count')
    cleaned_pivot["htab"] = np.where(
        cleaned_pivot.horizon.shift() == cleaned_pivot.horizon, "",
        "\\multirow[t]{" + cleaned_pivot['h_freq'].astype(str) + "}{*}{" + cleaned_pivot['horizon'].astype(str) + "}"
    )
    cleaned_pivot.columns = cleaned_pivot.columns.map(_flatten_col)
    # Stat (median/mean) columns before disp (iqr/sd) columns.
    _other_cols = [c for c in cleaned_pivot.columns if not c.startswith(_stat_col + ".") and not c.startswith(_disp_col + ".")]
    _stat_cols  = [c for c in cleaned_pivot.columns if c.startswith(_stat_col + ".")]
    _disp_cols  = [c for c in cleaned_pivot.columns if c.startswith(_disp_col + ".")]
    cleaned_pivot = cleaned_pivot[_other_cols + _stat_cols + _disp_cols]

    cleaned_pivot["endings"] = np.where(
        np.logical_and(
            cleaned_pivot["htab"].shift(-1) != "",
            cleaned_pivot.dataset.shift(-1) == cleaned_pivot.dataset
        ), "\\\\\\cmidrule{1-6}", "\\\\\\")
    cleaned_pivot.loc[cleaned_pivot.shape[0]-1, "endings"] = "\\\\\\"
    cleaned_pivot.to_csv(f"../overleaf/tables/{_out_file}", index=False)
    print(f"Saved RMSE table → {_out_file}")

    # ---- Max-AE Analysis ----
    print("Running Max-AE Analysis...")
    try:
        maxae_dfs = []
        for ds in ["weinstock", "cgmacros"]:
            path = f"../{ds}/maxae.csv"
            if os.path.exists(path):
                maxae_dfs.append(pd.read_csv(path))
        if not maxae_dfs:
            raise FileNotFoundError("No maxae.csv files found in ../weinstock or ../cgmacros")
        maxae = pd.concat(maxae_dfs, ignore_index=True)
        maxae = maxae.dropna(axis=0)
        maxae.id = maxae.id.astype(str).str.pad(width=3, side="left", fillchar="0")
        
        # Keep only the optimal base models/HBML like in RMSE block
        min_maxae = maxae.loc[np.logical_not(maxae.model.isin(["HBML (c)","HBML (d)"])),:]
        # Remap both old and new HBML primary-model names to the canonical "HBML" label
        min_maxae.loc[min_maxae.model.isin(["HBML (fd)", "HBML-SFHDF"]), "model"] = "HBML"
        min_maxae.loc[min_maxae.horizon == "half", "horizon"] = "0.5"
        
        _maxae_stats = min_maxae.groupby(["dataset","horizon","context","model"]).maxae.apply(
            lambda x: pd.Series(
                {"median": np.round(x.median(), 2),
                 "iqr":    np.round(x.quantile(0.75) - x.quantile(0.25), 2)},
                index=["median", "iqr"]
            )
        ).reset_index().pivot(
            index=["dataset","horizon","context","model"],
            columns=["level_4"], values="maxae"
        ).reset_index()
        
        _maxae_stats = _maxae_stats.groupby(["dataset", "horizon", "context"], group_keys=False).apply(format_rmse)
        
        maxae_pivot = _maxae_stats.pivot(
            index=["horizon","model","dataset"],
            columns=["context"],
            values=[_s_col, _d_col]
        ).sort_values(
            by=["dataset","horizon","model"],
            key=lambda x: x.map(dataset_order | horizon_order | model_order)
        ).reset_index()
        
        maxae_pivot['h_freq'] = maxae_pivot.groupby('horizon')['horizon'].transform('count')
        maxae_pivot["htab"] = np.where(maxae_pivot.horizon.shift() == maxae_pivot.horizon, "", "\\multirow[t]{" + maxae_pivot['h_freq'].astype(str) + "}{*}{" + maxae_pivot['horizon'].astype(str) + "}")
        maxae_pivot.columns = maxae_pivot.columns.map(_flatten_col)
        
        _m_other = [c for c in maxae_pivot.columns if not c.startswith(_stat_col + ".") and not c.startswith(_disp_col + ".")]
        _m_stat  = [c for c in maxae_pivot.columns if c.startswith(_stat_col + ".")]
        _m_disp  = [c for c in maxae_pivot.columns if c.startswith(_disp_col + ".")]
        maxae_pivot = maxae_pivot[_m_other + _m_stat + _m_disp]
        
        maxae_pivot["endings"] = np.where(
            np.logical_and(maxae_pivot["htab"].shift(-1) != "", maxae_pivot.dataset.shift(-1) == maxae_pivot.dataset), 
            "\\\\\\cmidrule{1-6}", "\\\\\\"
        )
        maxae_pivot.loc[maxae_pivot.shape[0]-1, "endings"] = "\\\\\\"
        maxae_out = "maxae_pivot_median.csv" if args.median else "maxae_pivot_mean.csv"
        maxae_pivot.to_csv(f"../overleaf/tables/{maxae_out}", index=False)
        print(f"Saved Max-AE table → {maxae_out}")
    except Exception as e:
        print(f"Skipping Max-AE Analysis due to: {e}")

    # ---- Individual RMSE Contribution Analysis ----
    print("Running Individual RMSE Contribution Analysis...")
    os.makedirs("../overleaf/images/rmse_contrib", exist_ok=True)

    # Define a consistent colour for each model (re-use GlobalValues where possible,
    # fall back to a tab10 cycle for any model not in color_params).
    # All models whose name starts with "HBML" share the single "HBML" palette entry.
    _all_models = sorted(cleaned_rmse["model"].unique(), key=lambda m: model_order.get(m, 99))
    _cmap = plt.cm.get_cmap("tab10")
    _model_colors = {}
    _ci = 0
    for _m in _all_models:
        if _m.startswith("HBML") and "HBML" in GlobalValues.color_params:
            # All HBML variants share one colour so they read as a family.
            _model_colors[_m] = GlobalValues.color_params["HBML"]
        elif _m in GlobalValues.color_params:
            _model_colors[_m] = GlobalValues.color_params[_m]
        else:
            _model_colors[_m] = _cmap(_ci % 10)
            _ci += 1

    # Separate model lists used by the two strip plot views.
    _EXPERT_MODELS  = [m for m in _all_models if not m.startswith("HBML")]
    _HBML_MODELS    = [m for m in _all_models if m.startswith("HBML")]
    # Preferred HBML representative shown alongside experts
    _HBML_REPR      = "HBML-SFHDF" if "HBML-SFHDF" in _all_models else (
                       "HBML" if "HBML" in _all_models else (_HBML_MODELS[0] if _HBML_MODELS else None)
                      )
    _EXPERT_PLUS    = _EXPERT_MODELS + ([_HBML_REPR] if _HBML_REPR else [])

    # ---- Plot 1a: Strip plots — Expert models + HBML-SFHDF ----
    def _draw_strip_plot(ax, model_list, ctx_data, colors):
        """Draw a single strip-plot panel onto ax for the given model_list."""
        for m_idx, model in enumerate(model_list):
            m_data = ctx_data[ctx_data["model"] == model]["rmse"].dropna()
            if m_data.empty:
                continue
            col = colors[model]
            mean_val   = m_data.mean()
            sd_val     = m_data.std(ddof=0)
            median_val = m_data.median()
            q25_val    = m_data.quantile(0.25)
            q75_val    = m_data.quantile(0.75)

            rng = np.random.default_rng(seed=42)
            jitter = rng.uniform(-0.18, 0.18, size=len(m_data))
            ax.scatter(
                m_idx + jitter, m_data.values,
                color=col, alpha=0.65, s=28, zorder=3, linewidths=0,
            )
            ax.hlines(mean_val, m_idx - 0.35, m_idx + 0.35,
                      colors=col, linewidths=2.2, zorder=4)
            ax.fill_between(
                [m_idx - 0.35, m_idx + 0.35],
                mean_val - sd_val, mean_val + sd_val,
                color=col, alpha=0.12, zorder=2,
            )
            ax.hlines(median_val, m_idx - 0.35, m_idx + 0.35,
                      colors=col, linewidths=2.2, linestyles="dotted", zorder=5)
            ax.add_patch(plt.Rectangle(
                (m_idx - 0.35, q25_val),
                width=0.70, height=q75_val - q25_val,
                linewidth=1.4, linestyle="dotted",
                edgecolor=col, facecolor="none", zorder=5,
            ))
        ax.set_xticks(range(len(model_list)))
        ax.set_xticklabels(model_list, rotation=35, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_xlim(-0.6, len(model_list) - 0.4)

    for ds in cleaned_rmse["dataset"].unique():
        ds_data  = cleaned_rmse[cleaned_rmse["dataset"] == ds]
        horizons = sorted(ds_data["horizon"].unique(), key=lambda h: horizon_order.get(str(h), 99))
        contexts = sorted(ds_data["context"].unique(), key=lambda c: context_order.get(str(c), 99))

        for h in horizons:
            h_data  = ds_data[ds_data["horizon"] == h]
            n_ctx   = len(contexts)
            if n_ctx == 0:
                continue
            h_str = str(h).replace(".", "p")

            strip_dir = "../overleaf/images/rmse_contrib"
            os.makedirs(strip_dir, exist_ok=True)

            # --- Figure A: Expert models + HBML-SFHDF representative ---
            fig_a, axes_a = plt.subplots(
                1, n_ctx, figsize=(4.5 * n_ctx, 5), sharey=True, squeeze=False
            )
            fig_a.suptitle(
                f"{fmt_dataset(ds)} — {fmt_horizon(h)} hr Horizon: Expert Models + HBML-SFHDF",
                fontsize=13,
            )
            for col_idx, ctx in enumerate(contexts):
                ax = axes_a[0][col_idx]
                _draw_strip_plot(ax, _EXPERT_PLUS, h_data[h_data["context"] == ctx], _model_colors)
                ax.set_title(f"Context {ctx} hr", fontsize=10)
                ax.set_xlabel("")
                if col_idx == 0:
                    ax.set_ylabel("RMSE (mg/dL)", fontsize=10)

            fig_a.tight_layout()
            path_a = f"{strip_dir}/strip_{ds}_h{h_str}_experts.pdf"
            fig_a.savefig(path_a, dpi=300, bbox_inches="tight")
            plt.close(fig_a)
            print(f"  Saved expert strip plot → {path_a}")

            # --- Figure B: HBML variants only ---
            if not _HBML_MODELS:
                continue
            fig_b, axes_b = plt.subplots(
                1, n_ctx, figsize=(4.5 * n_ctx, 5), sharey=True, squeeze=False
            )
            fig_b.suptitle(
                f"{fmt_dataset(ds)} — {fmt_horizon(h)} hr Horizon: HBML Variants",
                fontsize=13,
            )
            for col_idx, ctx in enumerate(contexts):
                ax = axes_b[0][col_idx]
                _draw_strip_plot(ax, _HBML_MODELS, h_data[h_data["context"] == ctx], _model_colors)
                ax.set_title(f"Context {ctx} hr", fontsize=10)
                ax.set_xlabel("")
                if col_idx == 0:
                    ax.set_ylabel("RMSE (mg/dL)", fontsize=10)

            fig_b.tight_layout()
            path_b = f"{strip_dir}/strip_{ds}_h{h_str}_hbml.pdf"
            fig_b.savefig(path_b, dpi=300, bbox_inches="tight")
            plt.close(fig_b)
            print(f"  Saved HBML strip plot  → {path_b}")


    # ---- Plot 2: Per-individual deviation heatmap (one figure per dataset × horizon × context) ----
    for ds in cleaned_rmse["dataset"].unique():
        ds_data = cleaned_rmse[cleaned_rmse["dataset"] == ds]
        horizons = sorted(ds_data["horizon"].unique(), key=lambda h: horizon_order.get(str(h), 99))
        contexts = sorted(ds_data["context"].unique(), key=lambda c: context_order.get(str(c), 99))

        for h in horizons:
            for ctx in contexts:
                ctx_data = ds_data[(ds_data["horizon"] == h) & (ds_data["context"] == ctx)]
                if ctx_data.empty:
                    continue

                # Pivot to (id × model) RMSE matrix
                pivot = ctx_data.pivot_table(index="id", columns="model", values="rmse", aggfunc="mean")
                pivot = pivot[[m for m in _all_models if m in pivot.columns]]

                # Signed % deviation from each model's column mean
                col_means = pivot.mean(axis=0)
                dev_pct = (pivot - col_means) / col_means * 100

                # Sort individuals by their mean absolute deviation across all models
                dev_pct["_mad"] = dev_pct.abs().mean(axis=1)
                dev_pct = dev_pct.sort_values("_mad", ascending=False).drop(columns="_mad")

                # Also save as CSV for inspection
                dev_pct.to_csv(
                    f"../overleaf/tables/rmse_dev_{ds}_h{str(h).replace('.','p')}_c{ctx}.csv"
                )

                n_ids, n_mods = dev_pct.shape
                fig_h = max(4, 0.28 * n_ids)
                fig, ax = plt.subplots(figsize=(max(5, 1.1 * n_mods), fig_h))

                vmax = np.nanpercentile(dev_pct.abs().values, 95)
                im = ax.imshow(
                    dev_pct.values,
                    aspect="auto",
                    cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax,
                )
                cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
                cbar.set_label("% deviation from group mean", fontsize=9)

                ax.set_xticks(range(n_mods))
                ax.set_xticklabels(dev_pct.columns, rotation=35, ha="right", fontsize=9)
                ax.set_yticks(range(n_ids))
                ax.set_yticklabels(dev_pct.index, fontsize=7)
                ax.set_title(
                    f"{fmt_dataset(ds)} — {fmt_horizon(h)} hr, Context {ctx} hr\n"
                    f"Individual RMSE deviation from group mean (sorted by mean |dev|)",
                    fontsize=10,
                )
                ax.set_xlabel("Model", fontsize=9)
                ax.set_ylabel("Individual ID (↑ = most deviant)", fontsize=9)

                plt.tight_layout()
                h_str = str(h).replace(".", "p")
                hmap_dir = "../overleaf/images/rmse_contrib"
                os.makedirs(hmap_dir, exist_ok=True)
                hmap_path = f"{hmap_dir}/heatmap_{ds}_h{h_str}_c{ctx}.pdf"
                plt.savefig(hmap_path, dpi=300, bbox_inches="tight")
                plt.close("all")
                print(f"  Saved deviation heatmap → {hmap_path}")

    print("Individual RMSE Contribution Analysis complete.")

    if not args.rmse_only:
        # Build full_ceg.csv from individual per-run ceg.csv files
        print("Building full_ceg.csv from individual ceg.csv files...")
        ceg_pattern = re.compile(r"\.\./([\\w-]+)/h-([^/]+?)hr/context-([^/]+?)hr/ceg\.csv")
        ceg_files = glob("../*/*/context-*/ceg.csv")
        ceg_parts = []
        for path in ceg_files:
            m = ceg_pattern.match(path)
            if not m:
                continue
            dataset_key, horizon_key, context_key = m.group(1), m.group(2), m.group(3)
            try:
                df = pd.read_csv(path, on_bad_lines="skip")
            except Exception as e:
                print(f"  Skipping {path}: {e}")
                continue
            def _fix_horizon(val):
                val = str(val)
                val = re.split(r"hr/", val)[0]
                m2 = re.match(r"^([^h]+?)hr$", val)
                if m2:
                    return m2.group(1)
                return val
            if "horizon" in df.columns:
                df["horizon"] = df["horizon"].apply(_fix_horizon)
                bad_mask = df["horizon"].str.contains(r"[/]", regex=True, na=False)
                if bad_mask.any():
                    print(f"  Dropping {bad_mask.sum()} malformed horizon rows from {path}")
                    df = df[~bad_mask]
            else:
                df["horizon"] = horizon_key
            df["dataset"] = dataset_key
            df["context"] = context_key
            ceg_parts.append(df)
        if ceg_parts:
            full_ceg = pd.concat(ceg_parts, ignore_index=True)
            bad_rows = full_ceg["horizon"].str.contains(r"[/]", regex=True, na=False)
            if bad_rows.any():
                print(f"  Dropping {bad_rows.sum()} remaining malformed horizon rows from full_ceg")
                full_ceg = full_ceg[~bad_rows]
            full_ceg.to_csv("../full_ceg.csv", index=False)
            print(f"  Written ../full_ceg.csv with {len(full_ceg)} rows from {len(ceg_parts)} files.")
        else:
            print("  Warning: no ceg.csv files found; ../full_ceg.csv not updated.")

        # Clarke Error Grid Analysis
        print("Running Clarke Error Grid Analysis...")
        ceg = pd.read_csv("../full_ceg.csv")
        ceg.id = ceg.id.astype(str).str.pad(width=3,side="left",fillchar="0")
        ceg.loc[ceg.horizon == "half","horizon"] = "0.5"

        if "model" in ceg.columns:
            ceg = ceg.loc[~ceg.model.isin(["HBML (c)", "HBML (d)"])]
            ceg.loc[ceg.model == "HBML-SFHDF", "model"] = "HBML"
            setting_cols = ["dataset","horizon","context","model"]
        else:
            setting_cols = ["dataset","horizon","context"]

        ceg["A+B"] = ceg.A + ceg.B
        ceg["total"] = ceg.A + ceg.B + ceg.C + ceg.D1 + ceg.D2 + ceg.E1 + ceg.E2


        div_cols = ["A","B","C","D1","D2","E1","E2","A+B"]
        pct_cols = [col + "pct" for col in div_cols]

        ceg.loc[:,pct_cols] = 100*ceg.loc[:,div_cols].div(ceg.total,axis=0).values
        ceg = ceg.loc[ceg["A+Bpct"] > 50]

        max_ceg = ceg.loc[ceg.groupby(setting_cols + ["id"]).A.idxmax()].drop(["eta","id"],axis=1,errors='ignore').groupby(setting_cols).mean(numeric_only=True).reset_index()
        max_ceg.loc[:,pct_cols + setting_cols] = np.round(max_ceg.loc[:,pct_cols + setting_cols],2)

        max_ceg = max_ceg.sort_values(by=setting_cols, key=lambda x: x.map(dataset_order | horizon_order | context_order | model_order)).drop(div_cols + ["total"], axis=1, errors='ignore')
        max_ceg["dataset"] = max_ceg["dataset"].map(dict(weinstock="Weinstock 2016", cgmacros="CGMacros"))
        max_ceg["d_freq"] = max_ceg.groupby('dataset')['dataset'].transform('count')
        
        if "model" in max_ceg.columns:
            max_ceg["h_freq"] = max_ceg.groupby(['dataset','horizon'])['horizon'].transform('count')
            max_ceg["c_freq"] = max_ceg.groupby(['dataset','horizon','context'])['context'].transform('count')
            
            max_ceg["htab"] = np.where(max_ceg.horizon.shift() == max_ceg.horizon, "", "\\multirow[t]{"+max_ceg['h_freq'].astype(str)+"}{*}{"+max_ceg['horizon'].astype(str)+"}")
            max_ceg["dtab"] = np.where(max_ceg.dataset.shift() == max_ceg.dataset, "", "\\multirow[t]{"+max_ceg['d_freq'].astype(str)+"}{*}{"+max_ceg['dataset'].astype(str)+"}")
            max_ceg["ctab"] = np.where(np.logical_and(max_ceg.context.shift() == max_ceg.context, max_ceg.horizon.shift() == max_ceg.horizon), "", "\\multirow[t]{"+max_ceg['c_freq'].astype(str)+"}{*}{"+max_ceg['context'].astype(str)+"}")
            
            max_ceg["endings"] = "\\\\"
            mask_diff_context = max_ceg["ctab"].shift(-1) != ""
            mask_same_dataset = max_ceg.dataset.shift(-1) == max_ceg.dataset
            mask_same_horizon = max_ceg.horizon.shift(-1) == max_ceg.horizon
            
            max_ceg.loc[mask_diff_context & mask_same_horizon & mask_same_dataset, "endings"] = "\\\\\\cmidrule{3-12}"
            max_ceg.loc[~mask_same_horizon & mask_same_dataset, "endings"] = "\\\\\\cmidrule{2-12}"
            max_ceg.loc[~mask_same_dataset & max_ceg.dataset.shift(-1).notna(), "endings"] = "\\\\\\cmidrule{1-12}"
            
        else:
            max_ceg["h_freq"] = max_ceg.groupby('horizon')['horizon'].transform('count')
            max_ceg["htab"] = np.where(max_ceg.horizon.shift() == max_ceg.horizon, "", "\\multirow[t]{"+max_ceg['h_freq'].astype(str)+"}{*}{"+max_ceg['horizon'].astype(str)+"}")
            max_ceg["dtab"] = np.where(max_ceg.dataset.shift() == max_ceg.dataset, "", "\\multirow[t]{"+max_ceg['d_freq'].astype(str)+"}{*}{"+max_ceg['dataset'].astype(str)+"}")
            max_ceg["endings"] = np.where(np.logical_and(max_ceg["htab"].shift(-1) != "", max_ceg.dataset.shift(-1) == max_ceg.dataset), "\\\\\\cmidrule{2-9}","\\\\\\")
            max_ceg["endings"] = np.where(np.logical_and(max_ceg.dataset.shift(-1) != max_ceg.dataset, max_ceg.dataset.shift(-1).notna()), "\\\\\\cmidrule{1-9}", max_ceg["endings"])
            
        max_ceg.to_csv("../overleaf/tables/max_ceg.csv", index=False)
        print("Saved original CEG table.")

        # ── HBML-only CEG comparison table ────────────────────────────────────
        print("Building HBML-only CEG comparison table...")
        # Abbreviation map: model name → short label used in the table
        _HBML_ABBREV = {
            "HBML":               "HBML-FS",    # Fixed Share (fast-decreasing alpha)
            "HBML-AE (clip-loss)":"HBML-CL",   # Clipped-Loss adaptive eta
            "HBML-AE (sqrt)":     "HBML-SQ",   # Sqrt-t adaptive eta
            "HBML-AE (ema0.2)":  "HBML-E2",   # EMA gamma=0.2
            "HBML-AE (ema0.5)":  "HBML-E5",   # EMA gamma=0.5
            "HBML-AE (ema0.8)":  "HBML-E8",   # EMA gamma=0.8
            "HBML-AE (ema1)":    "HBML-E10",  # EMA gamma=1
            "HBML-VS":           "HBML-VS",   # Variable Share
            "HBML-AH":           "HBML-AH",   # AdaHedge
            "HBML-SFH":          "HBML-SFH",  # Scale-Free Hedge
            "HBML-SFHDF":        "HBML-SFHDF", # Scale-Free Hedge with Decay Forgetting
        }
        if "model" in ceg.columns:
            hbml_ceg = ceg[ceg["model"].isin(_HBML_ABBREV.keys())].copy()
            hbml_ceg["model"] = hbml_ceg["model"].map(_HBML_ABBREV)
            hbml_setting_cols = ["dataset", "horizon", "context", "model"]

            hbml_max_ceg = (
                hbml_ceg
                .loc[hbml_ceg.groupby(hbml_setting_cols + ["id"]).A.idxmax()]
                .drop(["eta", "id"], axis=1, errors="ignore")
                .groupby(hbml_setting_cols)
                .mean(numeric_only=True)
                .reset_index()
            )
            hbml_max_ceg[pct_cols] = np.round(hbml_max_ceg[pct_cols], 2)
            _hbml_model_order = {v: i for i, v in enumerate(_HBML_ABBREV.values())}
            hbml_max_ceg = hbml_max_ceg.sort_values(
                by=hbml_setting_cols,
                key=lambda x: x.map(dataset_order | horizon_order | context_order | _hbml_model_order)
            ).drop(div_cols + ["total"], axis=1, errors="ignore")

            hbml_max_ceg["dataset"] = hbml_max_ceg["dataset"].map(
                dict(weinstock="Weinstock 2016", cgmacros="CGMacros")
            )
            hbml_max_ceg["d_freq"] = hbml_max_ceg.groupby("dataset")["dataset"].transform("count")
            hbml_max_ceg["h_freq"] = hbml_max_ceg.groupby(["dataset", "horizon"])["horizon"].transform("count")
            hbml_max_ceg["c_freq"] = hbml_max_ceg.groupby(["dataset", "horizon", "context"])["context"].transform("count")
            hbml_max_ceg["dtab"] = np.where(
                hbml_max_ceg.dataset.shift() == hbml_max_ceg.dataset, "",
                "\\multirow[t]{" + hbml_max_ceg["d_freq"].astype(str) + "}{*}{" + hbml_max_ceg["dataset"].astype(str) + "}"
            )
            hbml_max_ceg["htab"] = np.where(
                hbml_max_ceg.horizon.shift() == hbml_max_ceg.horizon, "",
                "\\multirow[t]{" + hbml_max_ceg["h_freq"].astype(str) + "}{*}{" + hbml_max_ceg["horizon"].astype(str) + "}"
            )
            hbml_max_ceg["ctab"] = np.where(
                np.logical_and(
                    hbml_max_ceg.context.shift() == hbml_max_ceg.context,
                    hbml_max_ceg.horizon.shift() == hbml_max_ceg.horizon,
                ), "",
                "\\multirow[t]{" + hbml_max_ceg["c_freq"].astype(str) + "}{*}{" + hbml_max_ceg["context"].astype(str) + "}"
            )
            hbml_max_ceg["endings"] = "\\\\"
            _mask_dc = hbml_max_ceg["ctab"].shift(-1) != ""
            _mask_sh = hbml_max_ceg.horizon.shift(-1) == hbml_max_ceg.horizon
            _mask_sd = hbml_max_ceg.dataset.shift(-1) == hbml_max_ceg.dataset
            hbml_max_ceg.loc[_mask_dc & _mask_sh & _mask_sd, "endings"] = "\\\\\\cmidrule{3-12}"
            hbml_max_ceg.loc[~_mask_sh & _mask_sd, "endings"] = "\\\\\\cmidrule{2-12}"
            hbml_max_ceg.loc[~_mask_sd & hbml_max_ceg.dataset.shift(-1).notna(), "endings"] = "\\\\\\cmidrule{1-12}"

            hbml_max_ceg.to_csv("../overleaf/tables/hbml_ceg_comparison.csv", index=False)
            print("Saved HBML-only CEG comparison table → ../overleaf/tables/hbml_ceg_comparison.csv")
        else:
            print("  Skipping HBML CEG comparison: 'model' column not found in full_ceg.csv")

        # CGMacros Diabetic Status Analysis
        print("Running CGMacros Diabetic Status Analysis...")
        diab_order = {"T2D":0, "pre-T2D":1, "non-T2D":2}
        diabetic_map = {}
        for status, key in [("T2D", "diabetic_ids"), ("pre-T2D", "prediabetic_ids"), ("non-T2D", "nondiabetic_ids")]:
            for pat_id in GlobalValues.CGMacros.get(key, []):
                diabetic_map[pat_id] = status

        def get_diabetic_status(patient_id):
            return diabetic_map.get(str(patient_id).zfill(3), "Unknown")

        cgmacros_rmse = min_rmse[min_rmse.dataset == "cgmacros"].copy()
        cgmacros_rmse.loc[:, "status"] = cgmacros_rmse["id"].apply(get_diabetic_status)

        diab_rmse = cgmacros_rmse.groupby(["status", "horizon", "context", "model"]).rmse.apply(
            lambda x: pd.Series(
                {_stat_col: np.round((x.median() if args.median else x.mean()), 2),
                 _disp_col: np.round((x.quantile(0.75)-x.quantile(0.25) if args.median else x.std(ddof=0)), 2)},
                index=[_stat_col, _disp_col]
            )
        )
        diab_rmse = diab_rmse.reset_index().pivot(index=["status", "horizon", "context", "model"], columns=["level_4"], values="rmse").reset_index()
        diab_rmse = diab_rmse.groupby(["status", "horizon", "context"], group_keys=False).apply(format_rmse)

        diab_pivot = diab_rmse.pivot(index=["horizon", "model", "status"], columns=["context"], values=[_s_col, _d_col])
        diab_pivot = diab_pivot.sort_values(by=["status", "horizon", "model"], key=lambda x: x.map(diab_order | horizon_order | model_order)).reset_index()
        diab_pivot.columns = diab_pivot.columns.map(_flatten_col)
        _d_other = [c for c in diab_pivot.columns if not c.startswith(_stat_col + ".") and not c.startswith(_disp_col + ".")]
        _d_stat  = [c for c in diab_pivot.columns if c.startswith(_stat_col + ".")]
        _d_disp  = [c for c in diab_pivot.columns if c.startswith(_disp_col + ".")]
        diab_pivot = diab_pivot[_d_other + _d_stat + _d_disp]

        for status in ["T2D", "pre-T2D", "non-T2D"]:
            subset = diab_pivot[diab_pivot["status"] == status].copy()
            if not subset.empty:
                subset["h_freq"] = subset.groupby('horizon')['horizon'].transform('count')
                subset["htab"] = np.where(subset.horizon.shift() == subset.horizon, "", "\\multirow[t]{"+subset['h_freq'].astype(str)+"}{*}{"+subset['horizon'].astype(str)+"}")
                subset["endings"] = np.where(subset["htab"].shift(-1) != "", "\\\\\\cmidrule{1-6}", "\\\\\\")
                if len(subset) > 0:
                    subset.iloc[-1, subset.columns.get_loc("endings")] = "\\\\\\"
                subset.to_csv(f"../overleaf/tables/rmse_{_stat_col}_cgmacros_{status}.csv", index=False)
                print(f"Saved Diabetic RMSE table for {status}.")

        cgmacros_ceg = ceg[ceg.dataset == "cgmacros"].copy()
        cgmacros_ceg.loc[:, "status"] = cgmacros_ceg["id"].apply(get_diabetic_status)
        
        diab_setting_cols = ["status", "horizon", "context"]
        if "model" in cgmacros_ceg.columns:
            diab_setting_cols.append("model")
            
        diab_max_ceg = cgmacros_ceg.loc[cgmacros_ceg.groupby(diab_setting_cols + ["id"]).A.idxmax()].drop(["eta", "id", "dataset"], axis=1, errors='ignore').groupby(diab_setting_cols).mean(numeric_only=True).reset_index()
        diab_max_ceg.loc[:, pct_cols] = np.round(diab_max_ceg.loc[:, pct_cols], 2)
        diab_max_ceg = diab_max_ceg.sort_values(by=diab_setting_cols, key=lambda x: x.map(diab_order | horizon_order | context_order | model_order)).drop(div_cols + ["total"], axis=1, errors='ignore')

        for status in ["T2D", "pre-T2D", "non-T2D"]:
            subset = diab_max_ceg[diab_max_ceg["status"] == status].copy()
            if not subset.empty:
                if "model" in subset.columns:
                    subset["h_freq"] = subset.groupby('horizon')['horizon'].transform('count')
                    subset["c_freq"] = subset.groupby(['horizon','context'])['context'].transform('count')
                    
                    subset["htab"] = np.where(subset.horizon.shift() == subset.horizon, "", "\\multirow[t]{"+subset['h_freq'].astype(str)+"}{*}{"+subset['horizon'].astype(str)+"}")
                    subset["ctab"] = np.where(np.logical_and(subset.context.shift() == subset.context, subset.horizon.shift() == subset.horizon), "", "\\multirow[t]{"+subset['c_freq'].astype(str)+"}{*}{"+subset['context'].astype(str)+"}")
                    
                    subset["endings"] = "\\\\"
                    mask_diff_context = subset["ctab"].shift(-1) != ""
                    mask_same_horizon = subset.horizon.shift(-1) == subset.horizon
                    
                    subset.loc[mask_diff_context & mask_same_horizon, "endings"] = "\\\\\\cmidrule{2-10}"
                    subset.loc[~mask_same_horizon & subset.horizon.shift(-1).notna(), "endings"] = "\\\\\\cmidrule{1-10}"
                else:
                    subset["h_freq"] = subset.groupby('horizon')['horizon'].transform('count')
                    subset["htab"] = np.where(subset.horizon.shift() == subset.horizon, "", "\\multirow[t]{"+subset['h_freq'].astype(str)+"}{*}{"+subset['horizon'].astype(str)+"}")
                    subset["endings"] = np.where(subset["htab"].shift(-1) != "", "\\\\\\cmidrule{2-8}", "\\\\\\")
                
                subset.to_csv(f"../overleaf/tables/ceg_cgmacros_{status}.csv", index=False)
                print(f"Saved Original Diabetic CEG table for {status}.")

        # Adaptive Regret
        print("Running Adaptive Regret Analysis...")
        def _eta_str(e):
            f = float(e)
            return str(int(f)) if f == int(f) else str(f).replace('.', 'p')
        combinations = list(product(rmse.dataset.unique(), rmse.horizon.unique(), rmse.context.unique(), rmse.eta.apply(_eta_str).unique()))
        omit_models = ["Autoformer"]

        for d,h,c,e in combinations:
            files = list(set(glob(f"../{d}/h-{h}hr/context-{c}hr/regrets/*_fullregrets_eta{e}.csv")) - set(corrupted_paths))
            if not files:
                continue
            shapes = [pd.read_csv(path).shape for path in files]
            max_len = max([shape[0] for shape in shapes])
            max_col = max([shape[1] for shape in shapes])
            padded_arrays = []
            for path in files:
                df = pd.read_csv(path)
                cols_to_drop = list(set(df.columns) & set(omit_models))
                df.drop(cols_to_drop,axis=1,inplace=True)
                arr, cols = df.to_numpy(), df.columns
                padded_arrays.append(np.pad(arr, ((0,max_len - len(arr)), (0, max_col - arr.shape[1])), mode="constant", constant_values=np.nan))
            stacked_arr = np.stack(padded_arrays)
            avg_regrets = pd.DataFrame(np.nanmean(stacked_arr, axis=0), columns=cols)
            c_vals = {col: GlobalValues.color_params[col] if not col.startswith("HBML") else GlobalValues.color_params["HBML"] for col in avg_regrets}
            ls_vals = {col: GlobalValues.linestyle_params[col] if col.startswith("HBML") else "-" for col in avg_regrets}
            extent = (100, max_len - 100, 0, np.max(avg_regrets.to_numpy()))
            xmin, xmax, ymin, ymax = extent
            xmin_z, xmax_z = 0.8 * xmax, xmax
            if "HBML-SFHDF" in avg_regrets.columns:
                ymin_z, ymax_z = .8 * avg_regrets.loc[xmin_z:xmax_z, "HBML-SFHDF"].min(), 1.2 * avg_regrets.loc[xmin_z:xmax_z, "HBML-SFHDF"].max()
            else:
                ymin_z, ymax_z = ymin, ymax
            plt.close("all")
            fig,ax = plt.subplots(figsize=(12,8))
            axins = ax.inset_axes([0.025, .5, .4, .45], xlim=(xmin_z, xmax_z), ylim=(ymin_z, ymax_z), xticks=[], yticks=[])
            for col in avg_regrets.columns:
                if col in ["HBML (c)", "HBML (d)"]:
                    continue
                label = "HBML" if col.startswith("HBML") else col
                ax.plot(avg_regrets.loc[xmin:xmax,col], c=c_vals[col], ls=ls_vals[col], alpha=1-0.3*(not col.startswith("HBML")), label=label, rasterized=True)
                axins.plot(avg_regrets.loc[xmin:xmax,col], c=c_vals[col], ls=ls_vals[col], alpha=1-0.3*(not col.startswith("HBML")), label=label, rasterized=True)
            ax.indicate_inset_zoom(axins, edgecolor="black")
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=13)
            ax.set_title(make_plot_title(d, h), fontsize=16)
            ax.set_xlim((xmin,xmax)); ax.set_ylim((ymin,ymax))
            ax.set_xlabel("Time Elapsed (min)", fontsize=14); ax.set_ylabel("Regret Over Time ([mg/dL]/min)", fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            regret_dir = f"../overleaf/images/regret/{d}"
            os.makedirs(regret_dir, exist_ok=True)
            h_str_r = str(h).replace(".","p").replace("half","0p5")
            c_str_r = str(c).replace("full","full")
            plt.savefig(f"{regret_dir}/zoom_h{h_str_r}_c{c_str_r}_eta{e}.pdf", dpi=400, bbox_inches="tight")
            plt.close("all")
            fig,ax = plt.subplots(figsize=(12,8))
            for col in avg_regrets.columns:
                if col in ["HBML (c)", "HBML (d)"]:
                    continue
                label = "HBML" if col.startswith("HBML") else col
                ax.plot(avg_regrets[col], c=c_vals[col], ls=ls_vals[col], alpha=1-0.3*(not col.startswith("HBML")), label=label, rasterized=True)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.set_title(make_plot_title(d, h))
            ax.set_xlim((xmin,xmax)); ax.set_ylim((ymin,ymax))
            ax.set_xlabel("Time Elapsed (min)"); ax.set_ylabel("Regret Over Time ([mg/dL]/min)")
            plt.savefig(f"{regret_dir}/h{h_str_r}_c{c_str_r}_eta{e}.pdf", dpi=400, bbox_inches="tight")
        print("Adaptive Regret Analysis complete.")

        # Weight Plot and other visualizations
        print("Running Extra Visualizations...")
        try:
            targets = utils.load_results_from_csv("../weinstock/data/110.csv")
            forecasts = utils.load_results_from_csv("../weinstock/h-5hr/context-fullhr/forecasts/110_forecasts.csv")
            losses = utils.load_results_from_csv("../weinstock/h-5hr/context-fullhr/losses/110_losses.csv")
            settings = utils.load_sim_settings("../weinstock/h-5hr/context-fullhr/settings/110.json")
            # Weight plot using Scale-Free Hedge with Decay Forgetting (HBML-SFHDF)
            from ExpMethods.simulate import scale_free_hedge_df_forecast
            _,_, W = scale_free_hedge_df_forecast(
                forecasts, losses,
                save_weights=True, forecast_type="mean", **settings
            )
            methods_dir = "../overleaf/images/methods"
            os.makedirs(methods_dir, exist_ok=True)
            viz.plot_weights(W, names=forecasts.keys(), show=False, save=True, path=f"{methods_dir}/expert_weights_plot.pdf")
            
            # --- Rolling RMSE Plot ---
            try:
                full_losses = pd.read_csv("../weinstock/h-5hr/context-fullhr/losses/110_fulllosses_advanced.csv")
                rolling_window = 144 # 12 hours * 12 (5-min intervals)
                rolling_rmse = np.sqrt(full_losses.rolling(window=rolling_window).mean())
                
                plt.figure(figsize=(12, 6))
                for col in ["NODE", "XGBoost", "ARIMA", "HBML-SFHDF"]:
                    if col in rolling_rmse.columns:
                        c = GlobalValues.color_params.get(col, None)
                        if "HBML" in col: c = GlobalValues.color_params.get("HBML", "red")
                        ls = GlobalValues.linestyle_params.get(col, "-")
                        plt.plot(rolling_rmse[col], label=col, color=c, linestyle=ls, linewidth=2)
                
                plt.title("Rolling 12-Hour RMSE (Patient 110)")
                plt.xlabel("Time Step (5-min intervals)")
                plt.ylabel("Rolling RMSE (mg/dL)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{methods_dir}/rolling_rmse_110.pdf", dpi=300)
                plt.close()
                print(f"  Saved Rolling RMSE plot → {methods_dir}/rolling_rmse_110.pdf")
            except Exception as e:
                print(f"  Skipping Rolling RMSE plot due to: {e}")
                
            clarke_dir = "../overleaf/images/clarke/weinstock"
            os.makedirs(clarke_dir, exist_ok=True)
            preds = pd.read_csv("../weinstock/h-halfhr/context-6hr/forecasts/103_fullforecasts_advanced.csv")["HBML-SFHDF"]
            ref = pd.read_csv("../weinstock/data/103.csv")["Libre.GL"]
            viz.clarke_error_grid(ref, preds, show=False,
                                  save_file=f"{clarke_dir}/h0p5_c6_sfhdf_103.pdf")
            print(f"  Saved CEG scatter → {clarke_dir}/h0p5_c6_sfhdf_103.pdf")
            print("Visualizations created.")
        except Exception as e:
            print(f"Skipping extra visualizations due to: {e}")

    print("generate_results.py execution finished successfully.")

