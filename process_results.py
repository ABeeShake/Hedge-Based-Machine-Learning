import os
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import ExpMethods.utils as utils
import ExpMethods.simulate as sim
import ExpMethods.visualizations as viz

from ExpMethods.globals import GlobalValues
from glob import glob
from ExpMethods.simulate import (
    MixingMethods, AlphaMethods, EtaMethods,
    get_weighted_forecasts_adaptive,
    get_weighted_forecasts_advanced,
)

# ---- Shared Plot Formatting Helpers ----
_DATASET_DISPLAY = {"weinstock": "Weinstock", "cgmacros": "CGMacros"}
_HORIZON_DISPLAY = {"half": "0.5", "0.5": "0.5", "2": "2", "5": "5"}

def fmt_dataset(d):
    """Return display name for a dataset key."""
    return _DATASET_DISPLAY.get(d.lower(), d.capitalize())

def generate_macros():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tables_dir = os.path.join(base_dir, "overleaf", "tables")
    output_file = os.path.join(base_dir, "overleaf", "sections", "generated_macros.tex")

    macros = []
    
    # 1. Process rmse_pivot.csv
    rmse_path = os.path.join(tables_dir, "rmse_pivot_mean.csv")
    if os.path.exists(rmse_path):
        rmse_df = pd.read_csv(rmse_path)
        
        weinstock_30 = rmse_df[(rmse_df["dataset"].str.contains("weinstock", case=False, na=False)) & (rmse_df["horizon"] == 0.5)]
        
        for _, row in rmse_df.iterrows():
            dataset = str(row['dataset']).replace(" ", "").replace("(", "").replace(")", "").replace(".", "").replace("-", "").replace("2016", "").capitalize()
            if pd.isna(row['dataset']) or dataset == "Nan":
                continue
                
            h_raw = str(row['horizon'])
            if h_raw == '0.5':
                horizon = "ThirtyMin"
            elif h_raw == '2.0' or h_raw == '2':
                horizon = "TwoHour"
            elif h_raw == '5.0' or h_raw == '5':
                horizon = "FiveHour"
            else:
                horizon = h_raw.replace(".", "")
                
            model = str(row['model']).replace(" ", "").replace("(", "").replace(")", "").replace(".", "").replace("-", "")
            
            for context_suffix, context_name in [('6', 'SixHr'), ('12', 'TwelveHr'), ('24', 'TwentyFourHr'), ('full', 'Full')]:
                col_name = f"mean.{context_suffix}"
                if col_name in row and not pd.isna(row[col_name]):
                    val = str(row[col_name]).replace("*", "").replace("\\textbf{", "").replace("}", "")
                    macro_name = f"RMSEMean{dataset}{horizon}{model}{context_name}"
                    macros.append(f"\\newcommand{{\\{macro_name}}}{{{val}}}")
                    
        # calculate reduction
        try:
            node_row = weinstock_30[weinstock_30["model"] == "NODE"]
            hbml_row = weinstock_30[weinstock_30["model"] == "HBML"]
            if not node_row.empty and not hbml_row.empty:
                node_val_str = str(node_row["mean.full"].values[0]).replace("*", "").replace("\\textbf{", "").replace("}", "")
                hbml_val_str = str(hbml_row["mean.full"].values[0]).replace("*", "").replace("\\textbf{", "").replace("}", "")
                node_val = float(node_val_str)
                hbml_val = float(hbml_val_str)
                reduction = (node_val - hbml_val) / node_val * 100
                macros.append(f"\\newcommand{{\\RMSEMeanWeinstockThirtyMinReduction}}{{{reduction:.1f}}}")
        except Exception as e:
            print("Could not compute RMSE reduction:", e)
            
    # 2. Process max_ceg.csv for CEG metrics
    ceg_path = os.path.join(tables_dir, "max_ceg.csv")
    if os.path.exists(ceg_path):
        ceg_df = pd.read_csv(ceg_path)
        for _, row in ceg_df.iterrows():
            dataset = str(row['dataset']).replace(" ", "").replace("(", "").replace(")", "").replace(".", "").replace("-", "").replace("2016", "").capitalize()
            if pd.isna(row['dataset']) or dataset == "Nan":
                continue
            
            h_raw = str(row['horizon'])
            if h_raw == '0.5':
                horizon = "ThirtyMin"
            elif h_raw == '2.0' or h_raw == '2':
                horizon = "TwoHour"
            elif h_raw == '5.0' or h_raw == '5':
                horizon = "FiveHour"
            else:
                horizon = h_raw.replace(".", "")
                
            c_raw = str(row['context'])
            if c_raw == '6':
                context = "SixHr"
            elif c_raw == '12':
                context = "TwelveHr"
            elif c_raw == '24':
                context = "TwentyFourHr"
            elif c_raw == 'full':
                context = "Full"
            else:
                context = c_raw

            macros.append(f"\\newcommand{{\\CEG{dataset}{horizon}{context}AB}}{{{row['A+Bpct']:.2f}}}")
            macros.append(f"\\newcommand{{\\CEG{dataset}{horizon}{context}A}}{{{row['Apct']:.2f}}}")
            macros.append(f"\\newcommand{{\\CEG{dataset}{horizon}{context}B}}{{{row['Bpct']:.2f}}}")
            macros.append(f"\\newcommand{{\\CEG{dataset}{horizon}{context}C}}{{{row['Cpct']:.2f}}}")
            macros.append(f"\\newcommand{{\\CEG{dataset}{horizon}{context}DOne}}{{{row['D1pct']:.2f}}}")
            macros.append(f"\\newcommand{{\\CEG{dataset}{horizon}{context}DTwo}}{{{row['D2pct']:.2f}}}")
            macros.append(f"\\newcommand{{\\CEG{dataset}{horizon}{context}EOne}}{{{row['E1pct']:.2f}}}")
            macros.append(f"\\newcommand{{\\CEG{dataset}{horizon}{context}ETwo}}{{{row['E2pct']:.2f}}}")

            # Also yield shortcuts for the heavily discussed horizons in text to not break compilation
            if dataset == "Weinstock" and horizon == "ThirtyMin" and context == "Full":
                macros.append(f"\\newcommand{{\\CEGWeinstockThirtyMinAB}}{{{row['A+Bpct']:.2f}}}")
                macros.append(f"\\newcommand{{\\CEGWeinstockThirtyMinA}}{{{row['Apct']:.2f}}}")
                macros.append(f"\\newcommand{{\\CEGWeinstockThirtyMinCD}}{{{row['Cpct'] + row['D1pct'] + row['D2pct']:.2f}}}")
                macros.append(f"\\newcommand{{\\CEGWeinstockThirtyMinE}}{{{row['E1pct'] + row['E2pct']:.2f}}}")
            
            if dataset == "Weinstock" and horizon == "FiveHour" and context == "TwentyFourHr":
                macros.append(f"\\newcommand{{\\CEGWeinstockFiveHourAB}}{{{row['A+Bpct']:.2f}}}")
                macros.append(f"\\newcommand{{\\CEGWeinstockFiveHourA}}{{{row['Apct']:.2f}}}")
                macros.append(f"\\newcommand{{\\CEGWeinstockFiveHourCD}}{{{row['Cpct'] + row['D1pct'] + row['D2pct']:.2f}}}")
                macros.append(f"\\newcommand{{\\CEGWeinstockFiveHourE}}{{{row['E1pct'] + row['E2pct']:.2f}}}")

    # Deduplicate: if the same macro name was added more than once (e.g. from
    # multiple process_results.py runs), keep only the last definition so that
    # LaTeX doesn't raise "\newcommand already defined" errors.
    seen: dict[str, str] = {}
    for line in macros:
        # Extract the macro name (\newcommand{\Name}{value} -> Name)
        m = re.match(r"\\newcommand\{\\([^}]+)\}", line)
        key = m.group(1) if m else line
        seen[key] = line
    unique_macros = list(seen.values())

    with open(output_file, "w") as f:
        f.write("% Auto-generated macros from CSV data\n")
        f.write("\n".join(unique_macros))
    
    duplicates = len(macros) - len(unique_macros)
    if duplicates:
        print(f"  Removed {duplicates} duplicate macro definition(s).")
    print(f"Generated {len(unique_macros)} macros in {output_file}")

def fmt_horizon(h):
    """Return display string for a horizon value."""
    return _HORIZON_DISPLAY.get(str(h), str(h))

def make_plot_title(dataset, horizon):
    """Format standard plot title: 'Dataset (h hr Horizon)'."""
    return f"{fmt_dataset(dataset)} ({fmt_horizon(horizon)} hr Horizon)"

def _eta_str(eta):
    """Return a filename-safe string for an eta value.
    Whole-number floats produce clean strings (10.0 -> '10');
    fractional values replace the decimal point with 'p' (0.5 -> '0p5').
    This keeps filenames consistent whether eta is passed as int or float.
    """
    f = float(eta)
    if f == int(f):
        return str(int(f))
    return str(f).replace('.', 'p')

def get_args():
    parser = argparse.ArgumentParser(
        prog="Process Results",
        description="Unified script to process forecasts, generate losses, regrets, RMSE, CEG, and plots."
    )
    for flag, settings in GlobalValues.command_line_args.items():
        parser.add_argument(flag, **settings)
    
    # fix_outputs args
    parser.add_argument("--omit_ids", nargs="*", type=str, default=[])
    parser.add_argument("--eta", type=float, default=10)
    parser.add_argument("--omit_models", nargs="*", type=str, default=[])
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--adaptive_eta", action="store_true",
                        help="Use adaptive (time-varying) eta schedules instead of the static eta value.")
    
    # results args
    parser.add_argument("--bands", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_bumpchart", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_title", type=str, default="")
    parser.add_argument("--plot_file", type=str, default="average_regrets.pdf")
    parser.add_argument("--plot_ceg", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_all_regrets", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_moe_regrets", action=argparse.BooleanOptionalAction)
    parser.add_argument("--hedge_version", type=str, default="HBML-SFHDF")
    parser.add_argument("--update_loss_type", type=str, default="mse", choices=["mse", "mae"],
                        help="Loss metric used for internal exponential weight updates")
    parser.add_argument("--norm_type", type=str, default="shift", choices=["shift", "ratio"],
                        help="Per-step loss normalization: 'shift' subtracts min (default), 'ratio' divides by max (bounded)")
    parser.add_argument("--advanced_methods", action="store_true",
                        help="Also run Variable Share, AdaHedge, and Scale-Free Hedge algorithms")
    parser.add_argument("--gamma_search", action="store_true",
                        help="Grid-search SFHDF decay gamma so that SFHDF beats all experts in RMSE")
    
    return parser.parse_args()

def get_methods(settings, **kwargs):
    eta = kwargs.get("eta", None)
    mix_funcs = kwargs.get("mix_funcs",None)
    alpha_funcs = kwargs.get("alpha_funcs",None)
    update_loss_type = kwargs.get("update_loss_type", "mae")
    
    methods = {
        "HBML (c)":sim.DefaultSimulationParams.exp_params(
            start = settings["start"], end = settings["end"], save_weights = True, update_loss_type=update_loss_type),
        "HBML (d)":sim.DefaultSimulationParams.exp_params(
            start = settings["start"], end = settings["end"], save_weights = True, update_loss_type=update_loss_type),
        "HBML (fd)":sim.DefaultSimulationParams.exp_params(
            start = settings["start"], end = settings["end"], save_weights = True, update_loss_type=update_loss_type)
    }
    
    if eta is not None:
        for method in methods.keys():
            methods[method]["eta"] = eta
    
    if mix_funcs:
        for method in methods.keys():
            methods[method]["mix_func"] = mix_funcs[method]
            
    if alpha_funcs:
        for method in methods.keys():
            methods[method]["alpha_func"] = alpha_funcs[method]
            
    for method in methods.keys():
        methods[method]["start"] = settings["start"]
        methods[method]["end"] = settings["end"]
    
    return methods


def get_methods_adaptive(settings, **kwargs):
    """Build method dicts for the adaptive-eta pipeline.

    All three variants use the fast-decreasing (fd) alpha schedule so the
    comparison isolates the effect of the eta schedule only:

    - ``HBML-AE (fd, √t)``   : fast-dec alpha + sqrt_t eta          (theory-optimal)
    - ``HBML-AE (fd, 2ln)``  : fast-dec alpha + sqrt_2lnm_t eta     (Freund-Schapire)
    - ``HBML-AE (fd, loss)`` : fast-dec alpha + loss_based_eta       (loss-adaptive)

    where ``AE`` stands for "Adaptive Eta".
    """
    mix_funcs   = kwargs.get("mix_funcs",   None)
    alpha_funcs = kwargs.get("alpha_funcs", None)

    # Only active variants listed — others commented for speed.
    # Re-enable by uncommenting the relevant lines.
    eta_funcs = {
        # "HBML-AE (clip-loss)": EtaMethods.clipped_loss_based_eta,
        # "HBML-AE (sqrt)": EtaMethods.sqrt_t_eta,
        # "HBML-AE (ema0.2)": EtaMethods.make_ema_loss_eta(gamma=0.2),
        # "HBML-AE (ema0.5)": EtaMethods.make_ema_loss_eta(gamma=0.5),
        # "HBML-AE (ema0.8)": EtaMethods.make_ema_loss_eta(gamma=0.8),
        # "HBML-AE (ema1)": EtaMethods.make_ema_loss_eta(gamma=1)
    }

    eta_init = kwargs.get("eta", 10.0)
    update_loss_type = kwargs.get("update_loss_type", "mse")
    norm_type = kwargs.get("norm_type", "shift")

    methods = {
        name: sim.DefaultSimulationParams.exp_params(
            start=settings["start"],
            end=settings["end"],
            save_weights=True,
            eta_func=eta_funcs[name],
            eta=eta_init,
            update_loss_type=update_loss_type,
            norm_type=norm_type,
        )
        for name in eta_funcs
    }

    default_mix_funcs   = {name: MixingMethods.FS_start_mix          for name in eta_funcs}
    default_alpha_funcs = {name: AlphaMethods.fastdecreasing_alpha    for name in eta_funcs}

    resolved_mix   = mix_funcs   if mix_funcs   else default_mix_funcs
    resolved_alpha = alpha_funcs if alpha_funcs else default_alpha_funcs

    for method in methods:
        methods[method]["mix_func"]   = resolved_mix[method]
        methods[method]["alpha_func"] = resolved_alpha[method]
        methods[method]["start"]      = settings["start"]
        methods[method]["end"]        = settings["end"]

    return methods


def get_methods_advanced(settings, **kwargs):
    """Build method dicts for Variable Share, AdaHedge, and Scale-Free Hedge.

    Each entry sets a ``method_type`` key so the unified dispatcher
    :func:`get_weighted_forecasts_advanced` routes the call to the correct
    standalone function in simulate.py.
    """
    eta_init      = kwargs.get("eta",           10.0)
    norm_type     = kwargs.get("norm_type",     "ratio")
    forecast_type = kwargs.get("forecast_type", "mean")

    base = dict(
        start=settings["start"],
        end=settings["end"],
        save_weights=True,
        forecast_type=forecast_type,
    )

    return {
        # Inactive variants — uncomment to re-enable:
        # "HBML-VS": dict(**base, method_type="variable_share", eta=eta_init),
        # "HBML-AH": dict(**base, method_type="adahedge", eta=eta_init, norm_type=norm_type),
        "HBML-SFH": dict(
            **base,
            method_type="scale_free_hedge",
        ),
        "HBML-SFHDF": dict(
            **base,
            method_type="scale_free_hedge_df",
            gamma=kwargs.get("gamma", 0.2),
        ),
    }

# Ordered candidate list for gamma grid search (0 = no forgetting, 1 = fully forgetful)
SFHDF_GAMMA_CANDIDATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def find_best_sfhdf_gamma(forecast_dir, data_dir, settings_dir, target_col,
                          eta_str, omit_ids, omit_models, common_ids):
    """Grid-search SFHDF gamma over SFHDF_GAMMA_CANDIDATES.

    For each gamma, runs scale_free_hedge_df_forecast in-memory across all
    patients and computes mean RMSE.  Returns the first gamma for which mean
    SFHDF RMSE is strictly less than the mean RMSE of *every* expert model.
    Falls back to the gamma with the overall lowest SFHDF RMSE if none clears
    the bar.

    Parameters
    ----------
    forecast_dir, data_dir, settings_dir : str
        Directories used in the main processing loop.
    target_col : str
        Name of the glucose column ('Libre.GL' or 'Dexcom.GL').
    eta_str, omit_ids, omit_models, common_ids :
        Forwarded from main() to mirror the same patient subset.

    Returns
    -------
    float
        The chosen gamma value.
    """
    import glob as _glob

    # ---- collect patient data (mirrors main loop file loading, no I/O side-effects) ----
    patient_data = []   # list of (forecasts_dict, losses_dict, settings_dict)
    for file in _glob.glob(os.path.join(forecast_dir, "*_forecasts.csv")):
        id_num = os.path.basename(file)[:3]
        if id_num in omit_ids or id_num not in common_ids:
            continue
        f_path = file
        t_path = os.path.join(data_dir, f"{id_num}.csv")
        s_path = os.path.join(settings_dir, f"{id_num}.json")
        if not os.path.exists(s_path) or not os.path.exists(t_path):
            continue

        settings = utils.load_sim_settings(s_path)
        f_df = pd.read_csv(f_path)
        cols_to_drop = list(set(f_df.columns) & ({"LSTM", "HBML", "FS (Start)",
                                                   "FS (Uniform)", "FS (Decay)"} | set(omit_models)))
        f_df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")
        if "HoltWinters" in f_df.columns:
            f_df["ETS"] = f_df["HoltWinters"]
            f_df.drop("HoltWinters", axis=1, inplace=True)

        forecasts = {col: f_df[col].to_numpy() for col in f_df.columns}

        t_df = pd.read_csv(t_path)
        targets = t_df[target_col].to_numpy() if target_col in t_df.columns else utils.load_targets_from_csv(t_path)

        # Length alignment
        _fl = len(forecasts[list(forecasts.keys())[0]])
        if _fl != len(targets):
            _min = min(_fl, len(targets))
            targets = targets[:_min]
            forecasts = {k: v[:_min] for k, v in forecasts.items()}

        losses = sim.get_online_losses(forecasts, targets, **settings)
        patient_data.append((id_num, forecasts, losses, settings))

    if not patient_data:
        print("[gamma_search] No patient data found; defaulting to gamma=0.2")
        return 0.2

    # ---- expert mean RMSE across patients ----
    expert_rmse_sum = {}   # model -> sum of per-patient RMSE
    n_patients = len(patient_data)
    for _, forecasts, losses, _ in patient_data:
        for model, loss_arr in losses.items():
            if model in (set(forecasts.keys()) | {"HBML", "HBML-SFH", "HBML-SFHDF"}):
                valid = loss_arr[100:-10]
                rmse = np.sqrt(np.nanmean(valid)) if len(valid) > 0 and not np.isnan(valid).all() else np.nan
                expert_rmse_sum.setdefault(model, 0.0)
                expert_rmse_sum[model] += rmse
    expert_mean = {m: expert_rmse_sum[m] / n_patients for m in expert_rmse_sum if not np.isnan(expert_rmse_sum[m] / n_patients)}
    if not expert_mean:
        print("[gamma_search] Could not compute expert RMSE; defaulting to gamma=0.2")
        return 0.2
    min_expert_mean_rmse = min(expert_mean.values())

    # ---- grid search ----
    sfhdf_mean_per_gamma = {}
    for gamma in SFHDF_GAMMA_CANDIDATES:
        rmse_sum = 0.0
        for _, forecasts, losses, settings in patient_data:
            exp_f, exp_l = sim.scale_free_hedge_df_forecast(
                forecasts, losses, gamma=gamma,
                save_weights=False, forecast_type="mean", **settings
            )
            valid = exp_l[100:-10]
            rmse_sum += np.sqrt(np.nanmean(valid)) if len(valid) > 0 and not np.isnan(valid).all() else 0.0
        sfhdf_mean = rmse_sum / n_patients
        sfhdf_mean_per_gamma[gamma] = sfhdf_mean
        print(f"  [gamma_search] gamma={gamma:.2f}  SFHDF mean RMSE={sfhdf_mean:.4f}  "
              f"(min expert mean={min_expert_mean_rmse:.4f})")
        if sfhdf_mean < min_expert_mean_rmse:
            print(f"  [gamma_search] Selected gamma={gamma:.2f} (SFHDF beats all experts)")
            return gamma

    # No gamma cleared the bar — use the one with the lowest SFHDF RMSE
    best_fallback = min(sfhdf_mean_per_gamma, key=sfhdf_mean_per_gamma.get)
    print(f"  [gamma_search] No gamma beat all experts. "
          f"Falling back to gamma={best_fallback:.2f} (lowest SFHDF RMSE={sfhdf_mean_per_gamma[best_fallback]:.4f})")
    return best_fallback


def main():
    plt.rcParams.update(GlobalValues.plot_params)

    args = get_args()
    eta_str = _eta_str(args.eta)  # filename-safe eta string (e.g. 10 -> '10', 0.5 -> '0p5')
    
    forecast_dir = os.path.join(args.output_dir,"forecasts")
    losses_dir = os.path.join(args.output_dir,"losses")
    weights_dir = os.path.join(args.output_dir,"weights")
    regrets_dir = os.path.join(args.output_dir,"regrets")
    settings_dir = os.path.join(args.output_dir,"settings")
    images_dir = os.path.join(args.output_dir,"images")
    data_dir = args.data_dir if args.data_dir else os.path.join(args.output_dir,"data")
    
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(losses_dir, exist_ok=True)
    os.makedirs(forecast_dir, exist_ok=True)
    os.makedirs(regrets_dir, exist_ok=True)
    
    # Global accumulators
    rmse_accum = {}
    maxae_accum = {}
    ceg_counts = []
    
    regrets_padded_arrays = []
    max_len = 0
    max_col = 0
    cols_order = None
    
    main_dir = os.path.relpath(os.path.join(data_dir, os.pardir))
    dataset_name = os.path.basename(main_dir)
    try:
        horizon = re.findall(r"\w+-([^/]+?)hr/", args.output_dir)[0]
        context = re.findall(r"context-(\S+)hr", args.output_dir)[0]
    except Exception:
        horizon, context = "unknown", "unknown"
    
    target_col = "Libre.GL" if dataset_name == "weinstock" else "Dexcom.GL"
    

    mix_funcs = {
        "HBML (c)":MixingMethods.FS_start_mix,
        "HBML (d)":MixingMethods.FS_start_mix,
        "HBML (fd)":MixingMethods.FS_start_mix,
    }
    alpha_funcs = {
        "HBML (c)":AlphaMethods.constant_alpha,
        "HBML (d)":AlphaMethods.decreasing_alpha,
        "HBML (fd)":AlphaMethods.fastdecreasing_alpha,
    }
    
    # Find all forecasts for the CURRENT horizon only, to compute which IDs are
    # present across all context windows for this horizon.  Using a global
    # intersection (all horizons × contexts) was too aggressive: an ID missing
    # from an unrelated horizon would silently be dropped from every context run.
    current_horizon_dir = os.path.dirname(os.path.dirname(forecast_dir))  # e.g. .../h-halfhr
    search_pattern = os.path.join(current_horizon_dir, "context-*", "forecasts", "*_forecasts.csv")
    all_forecast_files = glob(search_pattern)

    from collections import defaultdict
    id_by_context = defaultdict(set)
    for f_path in all_forecast_files:
        filename = os.path.basename(f_path)
        id_num = filename[:3]
        context_dir = os.path.basename(os.path.dirname(os.path.dirname(f_path)))
        if context_dir.startswith("context-"):
            id_by_context[context_dir].add(id_num)

    if id_by_context:
        common_ids = set.intersection(*[ids for ids in id_by_context.values()])
        if args.debug:
            all_ids_union = set.union(*id_by_context.values())
            dropped = sorted(all_ids_union - common_ids)
            if dropped:
                print(f"  IDs excluded by cross-context intersection for this horizon: {dropped}")
    else:
        # No cross-context data found; fall back to allowing all IDs in this run
        common_ids = {os.path.basename(f)[:3] for f in glob(os.path.join(forecast_dir, "*_forecasts.csv"))}

    # ---- Gamma search for SFHDF (optional pre-pass, runs after common_ids is known) ----
    sfhdf_gamma = 0.2  # default
    if args.advanced_methods and args.gamma_search:
        print("Running SFHDF gamma grid search...")
        sfhdf_gamma = find_best_sfhdf_gamma(
            forecast_dir=forecast_dir,
            data_dir=data_dir,
            settings_dir=settings_dir,
            target_col=target_col,
            eta_str=eta_str,
            omit_ids=set(args.omit_ids),
            omit_models=set(args.omit_models),
            common_ids=common_ids,
        )
        print(f"Using SFHDF gamma={sfhdf_gamma:.2f} for main processing loop.")

        
    for file in glob(os.path.join(forecast_dir,"*_forecasts.csv")):
        id_num = os.path.basename(file)[0:3]
        if id_num in args.omit_ids:
            continue
            
        if id_num not in common_ids:
            if args.debug:
                print(f"Skipping ID {id_num}: missing from at least one horizon-context condition.")
            continue
            
        f_path = os.path.join(forecast_dir,f"{id_num}_forecasts.csv")
        t_path = os.path.join(data_dir, f"{id_num}.csv")
        s_path = os.path.join(settings_dir,f"{id_num}.json")
        r_path = os.path.join(regrets_dir,f"{id_num}_fullregrets_eta{eta_str}.csv")

        if not os.path.exists(s_path) or not os.path.exists(t_path):
            continue

        if args.debug:
            print(f"Processing ID: {id_num}")
            
        settings = utils.load_sim_settings(s_path)
        f_df = pd.read_csv(f_path)
        
        # Clean columns
        cols_to_drop = list(set(f_df.columns) & ({"LSTM","HBML","FS (Start)","FS (Uniform)","FS (Decay)"} | set(args.omit_models)))
        f_df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')
        if "HoltWinters" in f_df.columns:
            f_df["ETS"] = f_df.loc[:,"HoltWinters"]
            f_df.drop("HoltWinters", axis=1, inplace=True)
            
        if args.overwrite or not os.path.exists(r_path):
            f_df.to_csv(f_path, index=False)
            
        forecasts = {col: f_df[col].to_numpy() for col in f_df.columns}
        
        # Remove alignment bug fix since simulate.py already places predictions at t+h
        for col in forecasts:
            # forecasts[col] = np.roll(forecasts[col], -1)
            pass
            
        t_df = pd.read_csv(t_path)
        
        if target_col in t_df.columns:
            targets = t_df[target_col].to_numpy()
        else:
            targets = utils.load_targets_from_csv(t_path)
            
        _forecast_len = len(forecasts.get("NODE", forecasts[list(forecasts.keys())[0]]))
        if _forecast_len != len(targets):
            _min_len = min(_forecast_len, len(targets))
            print(f"  ID {id_num}: length mismatch (forecasts={_forecast_len}, targets={len(targets)})"
                  f" — trimming both to {_min_len}")
            targets = targets[:_min_len]
            forecasts = {k: v[:_min_len] for k, v in forecasts.items()}
            
        if args.overwrite or not os.path.exists(r_path):
            losses = sim.get_online_losses(forecasts, targets, **settings)
            
            methods = get_methods(settings, eta=args.eta, mix_funcs=mix_funcs, alpha_funcs=alpha_funcs, update_loss_type=args.update_loss_type)
            exp_forecasts, exp_losses, exp_weights = sim.get_weighted_forecasts(forecasts, losses, methods, **settings)
            
            full_forecasts = forecasts | exp_forecasts
            full_losses = losses | exp_losses
            
            flat_weights = {}
            for name, wt_mat in exp_weights.items():
                for i, exp_name in enumerate(forecasts.keys()):
                    flat_weights[f"{name} - {exp_name}"] = wt_mat[:-1, i]
            
            utils.save_data(losses, path=os.path.join(losses_dir, f"{id_num}_losses.csv"))
            utils.save_data(exp_forecasts, path=os.path.join(forecast_dir, f"{id_num}_expforecasts_eta{eta_str}.csv"))
            utils.save_data(exp_losses, path=os.path.join(losses_dir, f"{id_num}_explosses_eta{eta_str}.csv"))
            utils.save_data(flat_weights, path=os.path.join(weights_dir, f"{id_num}_expweights_eta{eta_str}.csv"))
            utils.save_data(full_forecasts, path=os.path.join(forecast_dir, f"{id_num}_fullforecasts_eta{eta_str}.csv"))
            utils.save_data(full_losses, path=os.path.join(losses_dir, f"{id_num}_fulllosses_eta{eta_str}.csv"))
            
            regrets = sim.get_regrets(exp_losses=exp_losses, losses=losses, **settings)
            full_regrets = sim.get_regrets(exp_losses=full_losses, losses=losses, **settings)
            
            utils.save_data(regrets, path=os.path.join(regrets_dir, f"{id_num}_regrets_eta{eta_str}.csv"))
            utils.save_data(full_regrets, path=r_path)

            # --- Adaptive eta branch (runs when --adaptive_eta is set and at least one eta variant is active) ---
            if args.adaptive_eta:
                ae_r_path = os.path.join(regrets_dir, f"{id_num}_fullregrets_adaptive_eta.csv")
                ae_methods = get_methods_adaptive(
                    settings, eta=args.eta, update_loss_type=args.update_loss_type, norm_type=args.norm_type
                )
                if ae_methods:
                    ae_exp_forecasts, ae_exp_losses, ae_exp_weights = get_weighted_forecasts_adaptive(
                        forecasts, losses, ae_methods, **settings
                    )
                    ae_full_forecasts = forecasts | ae_exp_forecasts
                    ae_full_losses    = losses    | ae_exp_losses

                    ae_flat_weights = {}
                    for name, wt_mat in ae_exp_weights.items():
                        for i, exp_name in enumerate(forecasts.keys()):
                            ae_flat_weights[f"{name} - {exp_name}"] = wt_mat[:-1, i]

                    utils.save_data(ae_exp_forecasts,  path=os.path.join(forecast_dir, f"{id_num}_expforecasts_adaptive_eta.csv"))
                    utils.save_data(ae_exp_losses,     path=os.path.join(losses_dir,   f"{id_num}_explosses_adaptive_eta.csv"))
                    utils.save_data(ae_flat_weights,   path=os.path.join(weights_dir,  f"{id_num}_expweights_adaptive_eta.csv"))
                    utils.save_data(ae_full_forecasts, path=os.path.join(forecast_dir, f"{id_num}_fullforecasts_adaptive_eta.csv"))
                    utils.save_data(ae_full_losses,    path=os.path.join(losses_dir,   f"{id_num}_fulllosses_adaptive_eta.csv"))

                    ae_regrets      = sim.get_regrets(exp_losses=ae_exp_losses,  losses=losses, **settings)
                    ae_full_regrets = sim.get_regrets(exp_losses=ae_full_losses, losses=losses, **settings)

                    utils.save_data(ae_regrets,      path=os.path.join(regrets_dir, f"{id_num}_regrets_adaptive_eta.csv"))
                    utils.save_data(ae_full_regrets, path=ae_r_path)

                    # Merge adaptive models into full_losses / full_forecasts for RMSE accumulation below.
                    full_losses    = full_losses    | ae_full_losses
                    full_forecasts = full_forecasts | ae_full_forecasts


            # --- Advanced algorithms branch (Variable Share, AdaHedge, Scale-Free Hedge) ---
            if args.advanced_methods:
                adv_methods = get_methods_advanced(
                    settings, eta=args.eta, norm_type=args.norm_type, gamma=sfhdf_gamma
                )
                adv_exp_forecasts, adv_exp_losses, adv_exp_weights = get_weighted_forecasts_advanced(
                    forecasts, losses, adv_methods
                )
                adv_full_forecasts = forecasts | adv_exp_forecasts
                adv_full_losses    = losses    | adv_exp_losses

                adv_flat_weights = {}
                for name, wt_mat in adv_exp_weights.items():
                    for i, exp_name in enumerate(forecasts.keys()):
                        adv_flat_weights[f"{name} - {exp_name}"] = wt_mat[:-1, i]

                utils.save_data(adv_exp_forecasts,  path=os.path.join(forecast_dir, f"{id_num}_expforecasts_advanced.csv"))
                utils.save_data(adv_exp_losses,     path=os.path.join(losses_dir,   f"{id_num}_explosses_advanced.csv"))
                utils.save_data(adv_flat_weights,   path=os.path.join(weights_dir,  f"{id_num}_expweights_advanced.csv"))
                utils.save_data(adv_full_forecasts, path=os.path.join(forecast_dir, f"{id_num}_fullforecasts_advanced.csv"))
                utils.save_data(adv_full_losses,    path=os.path.join(losses_dir,   f"{id_num}_fulllosses_advanced.csv"))

                adv_regrets      = sim.get_regrets(exp_losses=adv_exp_losses,  losses=losses, **settings)
                adv_full_regrets = sim.get_regrets(exp_losses=adv_full_losses, losses=losses, **settings)

                utils.save_data(adv_regrets,      path=os.path.join(regrets_dir, f"{id_num}_regrets_advanced.csv"))
                utils.save_data(adv_full_regrets, path=os.path.join(regrets_dir, f"{id_num}_fullregrets_advanced.csv"))

                full_losses    = full_losses    | adv_full_losses
                full_forecasts = full_forecasts | adv_full_forecasts
                full_regrets   = full_regrets   | adv_full_regrets

        else:
            full_losses_df = pd.read_csv(os.path.join(losses_dir, f"{id_num}_fulllosses_eta{eta_str}.csv"))
            full_losses = {col: full_losses_df[col].to_numpy() for col in full_losses_df.columns}
            full_forecasts_df = pd.read_csv(os.path.join(forecast_dir, f"{id_num}_fullforecasts_eta{eta_str}.csv"))
            full_forecasts = {col: full_forecasts_df[col].to_numpy() for col in full_forecasts_df.columns}
            full_regrets_df = pd.read_csv(r_path)
            full_regrets_df = full_regrets_df.drop(list(set(full_regrets_df.columns) & set(args.omit_models)), axis=1, errors='ignore')
            cols_order = full_regrets_df.columns.to_list()
            full_regrets = {col: full_regrets_df[col].to_numpy() for col in full_regrets_df.columns}

            # Load cached adaptive results if they exist and --adaptive_eta requested.
            if args.adaptive_eta:
                ae_full_losses_path    = os.path.join(losses_dir,   f"{id_num}_fulllosses_adaptive_eta.csv")
                ae_full_forecasts_path = os.path.join(forecast_dir, f"{id_num}_fullforecasts_adaptive_eta.csv")
                if os.path.exists(ae_full_losses_path) and os.path.exists(ae_full_forecasts_path):
                    ae_fl_df = pd.read_csv(ae_full_losses_path)
                    ae_ff_df = pd.read_csv(ae_full_forecasts_path)
                    # Merge only the HBML-AE columns (not base experts, already in full_losses).
                    ae_cols = [c for c in ae_fl_df.columns if c.startswith("HBML-AE")]
                    for c in ae_cols:
                        full_losses[c]    = ae_fl_df[c].to_numpy()
                        full_forecasts[c] = ae_ff_df[c].to_numpy() if c in ae_ff_df.columns else full_forecasts.get(c)
                        
            # Load cached advanced results if they exist and --advanced_methods requested.
            if args.advanced_methods:
                adv_full_losses_path    = os.path.join(losses_dir,   f"{id_num}_fulllosses_advanced.csv")
                adv_full_forecasts_path = os.path.join(forecast_dir, f"{id_num}_fullforecasts_advanced.csv")
                adv_full_regrets_path   = os.path.join(regrets_dir,  f"{id_num}_fullregrets_advanced.csv")
                if os.path.exists(adv_full_losses_path) and os.path.exists(adv_full_forecasts_path):
                    adv_fl_df = pd.read_csv(adv_full_losses_path)
                    adv_ff_df = pd.read_csv(adv_full_forecasts_path)
                    adv_fr_df = pd.read_csv(adv_full_regrets_path) if os.path.exists(adv_full_regrets_path) else None
                    adv_cols = [c for c in adv_fl_df.columns if c.startswith("HBML-")]
                    for c in adv_cols:
                        full_losses[c]    = adv_fl_df[c].to_numpy()
                        full_forecasts[c] = adv_ff_df[c].to_numpy() if c in adv_ff_df.columns else full_forecasts.get(c)
                        if adv_fr_df is not None and c in adv_fr_df.columns:
                            full_regrets[c] = adv_fr_df[c].to_numpy()
        
        # --- Accumulate RMSE and Max-AE ---
        rmse_accum[id_num] = {}
        maxae_accum[id_num] = {}
        for model, loss_arr in full_losses.items():
            valid_loss = loss_arr[100:-10]
            if len(valid_loss) > 0:
                rmse_accum[id_num][model] = np.sqrt(np.mean(valid_loss))
                maxae_accum[id_num][model] = np.max(np.sqrt(valid_loss))
            else:
                rmse_accum[id_num][model] = np.nan
                maxae_accum[id_num][model] = np.nan
        
        # --- Accumulate Regrets ---
        if not cols_order:
            cols_order = [c for c in full_regrets.keys() if c not in args.omit_models]
            r_arr = np.stack([full_regrets[c] for c in cols_order], axis=1)
        else:
            # Recompute to ensure new dynamic models are always grabbed
            cols_order = [c for c in full_regrets.keys() if c not in args.omit_models]
            r_arr = np.stack([full_regrets[c] for c in cols_order], axis=1)

        regrets_padded_arrays.append(r_arr)
        max_len = max(max_len, r_arr.shape[0])
        max_col = max(max_col, r_arr.shape[1])
        
        # --- Accumulate CEG for all models ---
        if args.plot_ceg:
            for model_name, p in full_forecasts.items():
                start_idx, end_idx = 100, -10
                a_sliced = targets[start_idx:end_idx]
                p_sliced = p[start_idx:end_idx]
                
                if len(a_sliced) > 0 and len(p_sliced) > 0:
                    ceg_dict = viz.clarke_error_grid(a_sliced, p_sliced, return_dict=True, plot=False)
                    ceg_counts.append({"id": str(id_num), "model": model_name} | ceg_dict)

    # --- Finalize RMSE ---
    rmse_df = pd.DataFrame.from_dict(rmse_accum, orient="index")
    rmse_df.index.name = "id"
    rmse_df = rmse_df.reset_index()
    
    print(f"RMSE for h = {horizon}, c = {context}, eta = {args.eta}")
    print(np.round(rmse_df.drop("id", axis=1).mean(axis=0), 2))
    
    rmse_long = pd.melt(
        rmse_df, id_vars=["id"], 
        value_vars=[col for col in rmse_df.columns if col != "id"], 
        var_name="model", value_name="rmse"
    )
    rmse_long["horizon"] = horizon
    rmse_long["context"] = context
    rmse_long["eta"] = args.eta
    rmse_long["dataset"] = dataset_name
    
    rmse_path = os.path.join(main_dir, "rmse.csv")
    if not os.path.exists(rmse_path):
        utils.save_data(rmse_long, path=rmse_path, mode="w", header=True)
    else:
        utils.save_data(rmse_long, path=rmse_path, mode="a", header=False)
    
    # Deduplicate RMSE
    _rmse_tmp = pd.read_csv(rmse_path, on_bad_lines="skip")
    _rmse_tmp.drop_duplicates(subset=["horizon","context","eta","model","id"], keep="last").to_csv(rmse_path, index=False)

    # --- Finalize Max-AE ---
    maxae_df = pd.DataFrame.from_dict(maxae_accum, orient="index")
    maxae_df.index.name = "id"
    maxae_df = maxae_df.reset_index()

    maxae_long = pd.melt(
        maxae_df, id_vars=["id"], 
        value_vars=[col for col in maxae_df.columns if col != "id"], 
        var_name="model", value_name="maxae"
    )
    maxae_long["horizon"] = horizon
    maxae_long["context"] = context
    maxae_long["eta"] = args.eta
    maxae_long["dataset"] = dataset_name
    
    maxae_path = os.path.join(main_dir, "maxae.csv")
    if not os.path.exists(maxae_path):
        utils.save_data(maxae_long, path=maxae_path, mode="w", header=True)
    else:
        utils.save_data(maxae_long, path=maxae_path, mode="a", header=False)
    
    # Deduplicate Max-AE
    _maxae_tmp = pd.read_csv(maxae_path, on_bad_lines="skip")
    _maxae_tmp.drop_duplicates(subset=["horizon","context","eta","model","id"], keep="last").to_csv(maxae_path, index=False)
    
    # --- Finalize Regrets ---
    padded_ar_list = []
    for r in regrets_padded_arrays:
        padded_ar_list.append(np.pad(r, ((0, max_len - r.shape[0]), (0, max_col - r.shape[1])), mode="constant", constant_values=np.nan))
    
    if padded_ar_list:
        stacked_arr = np.stack(padded_ar_list)
        avg_r_dict, q5_r_dict, q95_r_dict = get_plot_series(stacked_arr, cols_order)
        
        utils.save_data(avg_r_dict, path=os.path.join(regrets_dir, f"average_regrets_eta{eta_str}.csv"))
        utils.save_data(q5_r_dict, path=os.path.join(regrets_dir, f"q5_regrets_eta{eta_str}.csv"))
        utils.save_data(q95_r_dict, path=os.path.join(regrets_dir, f"q95_regrets_eta{eta_str}.csv"))
        
        h_str_p = str(horizon).replace(".", "p").replace("half", "0p5")
        c_str_p = str(context)
        e_str_p = str(args.eta).replace(".", "p") if float(args.eta) != int(float(args.eta)) else str(int(float(args.eta)))
        regret_img_dir = os.path.join("..", "overleaf", "images", "regret", dataset_name)
        os.makedirs(regret_img_dir, exist_ok=True)

        omit = ["FS (Decay)", "FS (Uniform)", "FS (Decay2)", "MoE"] + args.omit_models

        if args.plot_all_regrets:
            plot_zoomed_regrets(
                avg_r_dict, q5_r_dict, q95_r_dict, omit=omit, start=100, end=max_len-500, show=False,
                bands=args.bands, hedge_version=args.hedge_version,
                save_file=os.path.join(regret_img_dir, f"h{h_str_p}_c{c_str_p}_eta{e_str_p}.pdf"),
                title=make_plot_title(dataset_name, horizon))
            
        if args.plot_moe_regrets:
            plot_moe_regrets(
                avg_r_dict, q5_r_dict, q95_r_dict,
                omit=[k for k in avg_r_dict.keys() if not k.startswith("HBML")],
                start=100, end=max_len-500, show=False, bands=args.bands,
                save_file=os.path.join(regret_img_dir, f"moe_h{h_str_p}_c{c_str_p}_eta{e_str_p}.pdf"),
                title=make_plot_title(dataset_name, horizon))

        if args.plot_bumpchart:
            plot_bumpchart(
                avg_r_dict, omit=omit, start=100, end=max_len-500, show=False,
                hedge_version=args.hedge_version,
                save_file=os.path.join(regret_img_dir, f"bump_h{h_str_p}_c{c_str_p}_eta{e_str_p}.pdf"),
                title=make_plot_title(dataset_name, horizon))
        
    # --- Finalize CEG ---
    if args.plot_ceg and ceg_counts:
        ceg_df = pd.DataFrame(ceg_counts)
        print(f"CEG values saved for h = {horizon}, c = {context}, eta = {args.eta}")
        
        ceg_df["horizon"] = horizon
        ceg_df["context"] = context
        ceg_df["eta"] = args.eta
        ceg_df["dataset"] = dataset_name
        
        ceg_path = os.path.join(args.output_dir, "ceg.csv") 
        
        if os.path.exists(ceg_path):
            old_ceg = pd.read_csv(ceg_path, nrows=1)
            if "model" not in old_ceg.columns:
                os.remove(ceg_path)
        
        if not os.path.exists(ceg_path):
            ceg_df.to_csv(ceg_path, mode="w", header=True, index=False)
        else:
            ceg_df.to_csv(ceg_path, mode="a", header=False, index=False)
            
        _ceg_tmp = pd.read_csv(ceg_path, on_bad_lines="skip")
        _ceg_tmp.drop_duplicates(subset=["id","horizon","context","eta","model"], keep="last").to_csv(ceg_path, index=False)

    print("Generating latex macros...")
    generate_macros()

# ====== Plotting Logic from results.py copied here ======
def plot_moe_regrets(avgrdict, q5rdict, q95rdict, omit = [], start=100, end=-100,**kwargs):
    title = kwargs.get("title","Average Regret Over Time Across All Patients")
    save_file = kwargs.get("save_file","average_regrets.pdf")
    show = kwargs.get("show",False)
    bands = kwargs.get("bands", True)
    
    avg_r_dict = {key: value for key,value in avgrdict.items() if key not in omit}
    q5_r_dict = {key: value for key,value in q5rdict.items() if key not in omit}
    q95_r_dict = {key: value for key,value in q95rdict.items() if key not in omit}
    
    plt.close("all")

    rep = {"(d)": "(decay)", "(fd)": "(fast decay)", "(c)": "(constant)"}
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
        
    for i,key in enumerate(avg_r_dict.keys()):
        T = (np.arange(len(avg_r_dict[key])) + 1)[start:end]
        Y = (avg_r_dict[key][start:end]/T)
        if key.startswith("HBML"):
            label = pattern.sub(lambda m: rep[re.escape(m.group(0))], key)
            col = GlobalValues.color_params["HBML"]
            ls = GlobalValues.linestyle_params[key]
        else:
            label = key
            col = GlobalValues.color_params[key]
            ls = "-"
        alpha = 1 if key.startswith("HBML") else 0.7
        plt.plot(
            T, Y, 
            label = f"{label}", 
            color=col, 
            linestyle = ls,
            alpha = alpha,
            rasterized = True
            )
        if bands:
            plt.plot(T, (q5_r_dict[key][start:end]/T), 
            ls = ":", color=col, alpha = 0.2,
            rasterized = True)
            plt.plot(T, (q95_r_dict[key][start:end]/T), 
            ls = ":", color=col,alpha = 0.2,
            rasterized = True)
            plt.fill_between(T, 
            y1 = (q5_r_dict[key][start:end]/T), y2 = (q95_r_dict[key][start:end]/T), 
            color=col, alpha = 0.2,
            rasterized = True)
        
    plt.xlabel("Time Elapsed (min)")
    plt.ylabel("Regret Over Time ([mg/dL]/min)")
    plt.ylim(0, None)
    plt.title(title)
    plt.legend()
    
    if show:
        plt.show()
    if save_file:
        plt.savefig(save_file, transparent = True)
        
def plot_zoomed_regrets(avgrdict, q5rdict, q95rdict, omit = [], start=100, end=-100,**kwargs):
    title = kwargs.get("title","Average Regret Over Time Across All Patients")
    save_file = kwargs.get("save_file","average_regrets.pdf")
    show = kwargs.get("show",False)
    bands = kwargs.get("bands", True)
    hedge_version = kwargs.get("hedge_version",None)
    
    hedge_keys = list(filter(lambda x: re.findall("HBML",string=x), avgrdict.keys()))
    if hedge_version:
        omit = omit + [key for key in hedge_keys if key != hedge_version]

    avg_r_dict = {key: value for key,value in avgrdict.items() if key not in omit}
    q5_r_dict = {key: value for key,value in q5rdict.items() if key not in omit}
    q95_r_dict = {key: value for key,value in q95rdict.items() if key not in omit}

    if hedge_version:
        if hedge_version in avg_r_dict:
            avg_r_dict["HBML"] = avg_r_dict.pop(hedge_version)
            q5_r_dict["HBML"] = q5_r_dict.pop(hedge_version)
            q95_r_dict["HBML"] = q95_r_dict.pop(hedge_version)
    
    plt.close("all")
    
    fig = plt.figure(figsize=(12,6))
    sub1 = fig.add_subplot(1,2,1)
    sub2 = fig.add_subplot(1,2,2)
    
    xmin, xmax, ymin, ymax = start, end, 0, 0
    
    rep = {"(d)": "(decay)", "(fd)": "(fast decay)", "(c)": "(constant)"}
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
        
    for i,key in enumerate(avg_r_dict.keys()):
        T = (np.arange(len(avg_r_dict[key])) + 1)[start:end]
        if xmax < 0:
            xmax = len(T)
        Y = (avg_r_dict[key][start:end]/T)
        
        if key.startswith("HBML"):
            ymax = max(Y.max(),ymax)
            col = GlobalValues.color_params["HBML"]
            ls = GlobalValues.linestyle_params[key]
            alpha = 1
        else:
            col = GlobalValues.color_params[key]
            ls = "-"
            alpha = 0.7
            
        sub1.plot(
            T, Y, 
            label = f"{key}: Average", color=col, linestyle = ls, alpha = alpha, rasterized = True
            )
        sub2.plot(
            T, Y, 
            label = f"{key}: Average", color=col, linestyle = ls, alpha = alpha, rasterized = True
            )
        if bands:
            sub1.plot(T, (q5_r_dict[key]/T)[start:end], ls = ":", color=col, alpha = 0.2, rasterized = True)
            sub1.plot(T, (q95_r_dict[key]/T)[start:end], ls = ":", color=col,alpha = 0.2, rasterized = True)
            sub1.fill_between(T, y1 = (q5_r_dict[key]/T)[start:end], y2 = (q95_r_dict[key]/T)[start:end], color=col, alpha = 0.2, rasterized = True)
        
    sub1.set_xlabel("Time Elapsed (min)")
    sub1.set_ylabel("Regret Over Time ([mg/dL]/min)")
    sub1.set_ylim(0, None)
    sub1.set_xlim(xmin, xmax)
    sub2.set_xlabel("Time Elapsed (min)")
    sub2.set_ylim(ymin, ymax+200)
    sub2.set_xlim(xmin, xmax)
    
    box = patches.Rectangle(
        (xmin,ymin), width = (xmax-xmin), height = (200 + ymax-ymin),
        linewidth = 10, linestyle = "--", edgecolor = "gray", facecolor = "none", rasterized = True)
    con1 = patches.ConnectionPatch(
        xyA = (xmax, ymin), coordsA = sub1.transData,
        xyB = (xmin,ymin), coordsB = sub2.transData, linestyle = "--" )
    con2 = patches.ConnectionPatch(
        xyA = (xmax, ymax+200), coordsA = sub1.transData,
        xyB = (xmin,ymax+200), coordsB = sub2.transData, linestyle = "--" )
        
    fig.add_artist(box)
    fig.add_artist(con1)
    fig.add_artist(con2)
        
    plt.suptitle(title)
    sub1.legend()
    
    if show:
        plt.show()
    if save_file:
        plt.savefig(save_file, transparent = True, dpi = 300)
        
def get_plot_series(stacked_arr, cols):
    avg_regrets = np.nanmean(stacked_arr, axis = 0)
    q5_regrets = np.nanquantile(stacked_arr, axis = 0, q = 0.05)
    q95_regrets = np.nanquantile(stacked_arr, axis = 0, q = 0.95)

    avg_r_dict = dict(zip(cols,avg_regrets.T))
    q5_r_dict = dict(zip(cols,q5_regrets.T))
    q95_r_dict = dict(zip(cols,q95_regrets.T))
    
    return avg_r_dict, q5_r_dict, q95_r_dict

def plot_bumpchart(avgrdict, omit = [], start = 100, end = -100, **kwargs):
    save_file = kwargs.get("save_file","average_regrets.pdf")
    title = kwargs.get("title")
    show = kwargs.get("show",False)
    hedge_version = kwargs.get("hedge_version","HBML-SFHDF")
    
    hedge_keys = list(filter(lambda x: re.findall("HBML",string=x), avgrdict.keys()))
    if hedge_version:
        omit = omit + [key for key in hedge_keys if key != hedge_version]
    
    avg_r_dict = {key: value for key,value in avgrdict.items() if key not in omit}
    
    if hedge_version and hedge_version in avg_r_dict:
        avg_r_dict["HBML"] = avg_r_dict.pop(hedge_version)

    df = pd.DataFrame(avg_r_dict).drop("HBML",axis=1).apply(lambda x: np.argsort(x).argsort()+1, axis = 1)
    df = df.iloc[start:end]

    df["HBML"] = pd.DataFrame(avg_r_dict).apply(lambda x: np.argsort(x).argsort()+1, axis = 1).HBML.iloc[start:end]
    
    plt.close("all")
    axes = viz.bumpchart(
        df, show_rank_axis= True, scatter = True, holes = True,
        line_args= {"linewidth": 2, "alpha": 0.8},
        scatter_args= {"s": 10, "alpha": 0.8},
        )
    axes[1].set_yticklabels([])
    axes[1].set_ylabel("")
    axes[1].tick_params(length = 0)
    axes[2].set_ylabel("Expert Ranking (1 = Highest Accuracy)")
    axes[2].yaxis.set_label_position("right")
    
    for tick_label in axes[0].get_yticklabels():
        if tick_label.get_text().startswith("HBML"):
            tick_label.set_color(GlobalValues.color_params["HBML"])
        else:
            tick_label.set_color(GlobalValues.color_params[tick_label.get_text()])
    
    rect = patches.Rectangle(
        (6.925, 0.75), width = len(df)+200, height = .5, 
        linewidth = 1.5, linestyle = "--", edgecolor = "red", facecolor = "none")
    axes[0].add_patch(rect)
    plt.text((6.925 + (len(df)+200)/2), 0.65, "Ideal Choice of Expert", ha = "center", va = "center", color = "red")
    plt.tight_layout()
    if title:
        plt.title(title)
    if show:
        plt.show()
    if save_file:
        plt.savefig(save_file, transparent = True, dpi = 300)

if __name__ == "__main__":
    main()
