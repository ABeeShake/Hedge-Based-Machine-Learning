#!/usr/bin/env python3
"""evaluate_external_cohort.py -- Self-contained Engine v2.3 evaluation on an external CGM cohort.

Reads cached forecasts from Outputs/<dataset>/h-*hr/context-*hr/forecasts/ and produces
self-contained participant/summary RMSE tables, paired statistical tests (Wilcoxon + BCa CI +
participant bootstrap TOST + BH FDR), weight dynamics, mechanism ablations (including
interleaved (h+1)-learner delayed reduction and equal-context ARIMA/ETS/NHITS sub-pool), and
clinical hypoglycaemia alert metrics inside Outputs/external/<dataset>/.

Never modifies the primary Weinstock + CGMacros manuscript tables in overleaf/tables/.

Usage:
  python3 evaluate_external_cohort.py --dataset ohiot1dm
"""
import argparse
import csv
import glob
import os
import sys
from collections import Counter

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ExpMethods import aggregators as agg  # noqa: E402
from ExpMethods import evaluation as ev  # noqa: E402
from ExpMethods.globals import GlobalValues as G  # noqa: E402
import calculate_statistics as cstat  # noqa: E402
import evaluate_clinical_and_weights as ecw  # noqa: E402
import analyze_pool_mechanism as apm  # noqa: E402

START = int(G.burn_in_steps)
MIN_SCORED = int(G.min_scored_points)
HORIZONS = [("half", "0.5", 6), ("2", "2", 24), ("5", "5", 60)]
CONTEXTS = ["6", "12", "24", "full"]


def max_ae(loss_col, mask):
    """Maximum absolute error (mg/dL) over scored targets."""
    vals = np.asarray(loss_col, dtype=float)[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.sqrt(np.max(vals))) if len(vals) > 0 else float("nan")


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_diabetes_types(dataset: str, root: str = BASE_DIR):
    """Map participant pid -> Parkes grid diabetes type (2 if manifest flags T2D, else 1)."""
    types = {}
    manifest = os.path.join(root, dataset, f"{dataset}_manifest.csv")
    if not os.path.exists(manifest):
        return types
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            pid = (r.get("pid") or "").strip()
            diag = (r.get("diagnosis") or "").strip().lower()
            types[pid] = 2 if ("t2d" in diag or diag == "2") else 1
    return types


def evaluate_cohort(dataset: str, outputs_dir: str, out_dir: str, n_boot_stat: int = 10000, n_boot_mech: int = 2000):
    os.makedirs(out_dir, exist_ok=True)
    diabetes_types = load_diabetes_types(dataset, root=ev.ROOT)
    rmse_rows = []
    rmse_summary_rows = []
    stat_all_rows = []
    stat_best_rows = []
    weight_rows = []
    hypo_rows = []
    sweep_rows = []
    sweep_pp_rows = []
    mech_rows = []

    for h_str, h_disp, h_steps in HORIZONS:
        for ctx in CONTEXTS:
            fc_dir = os.path.join(outputs_dir, dataset, f"h-{h_str}hr", f"context-{ctx}hr", "forecasts")
            files = ev.forecast_files(os.path.dirname(fc_dir)) if os.path.isdir(fc_dir) else []
            if not files:
                continue

            p_res = []
            for fc_path in files:
                pid = os.path.basename(fc_path)[:3]
                res = ecw.run_participant(dataset, pid, fc_path, h_steps, root=ev.ROOT)
                if not res:
                    continue
                f_mat = res["F"]
                y = res["y"]
                cols = res["cols"]
                mask = res["scored_mask"]
                l_mat = agg.online_losses(f_mat, y, START, h_steps)
                loss_hb = (res["pred_primary"] - y) ** 2
                _, loss_inter = agg.run_interleaved_hedge(ecw.PRIMARY_HBML, f_mat, l_mat, y, start=START, horizon=h_steps)
                short_idx = [k for k, c in enumerate(cols) if c in apm.SHORTCTX_EXPERTS]
                if len(short_idx) >= 2:
                    F_sc = f_mat[:, short_idx]
                    L_sc = l_mat[:, short_idx]
                    _, loss_sc = agg.run_named(ecw.PRIMARY_HBML, F_sc, L_sc, y, start=START, horizon=h_steps)
                    rmse_sc = ev.rmse(loss_sc, mask)
                else:
                    rmse_sc = float("nan")
                nxn_idx = [k for k, c in enumerate(cols) if c not in apm.NO_XGB_NODE_EXCLUDE]
                if len(nxn_idx) >= 2:
                    F_nxn = f_mat[:, nxn_idx]
                    L_nxn = l_mat[:, nxn_idx]
                    _, loss_nxn = agg.run_named(ecw.PRIMARY_HBML, F_nxn, L_nxn, y, start=START, horizon=h_steps)
                    rmse_nxn = ev.rmse(loss_nxn, mask)
                else:
                    rmse_nxn = float("nan")
                rmse_uni = ev.rmse((f_mat.mean(axis=1) - y) ** 2, mask)

                res["interleaved_rmse"] = ev.rmse(loss_inter, mask)
                res["shortctx_rmse"] = rmse_sc
                res["no_xgb_node_rmse"] = rmse_nxn
                res["uniform_rmse"] = rmse_uni
                res["hbml_maxae"] = max_ae(loss_hb, mask)
                res["exp_maxaes"] = {cols[k]: max_ae(l_mat[:, k], mask) for k in range(len(cols))}
                p_res.append(res)

            if not p_res:
                continue

            n_p = len(p_res)
            cols = p_res[0]["cols"]
            exp_means = {c: float(np.mean([r["exp_rmses"][c] for r in p_res])) for c in cols}
            best_exp = min(exp_means, key=exp_means.get)

            for r in p_res:
                row = {
                    "dataset": dataset,
                    "horizon": h_disp,
                    "context": ctx,
                    "id": r["pid"],
                    "n_scored": r["n_scored"],
                    "HBML": r["hbml_rmse"],
                    "HBML_SFHDF": r["alt_rmse"] if ecw.PRIMARY_HBML != "HBML-SFHDF" else r["hbml_rmse"],
                    "HBML_Interleaved": r["interleaved_rmse"],
                    "HBML_ShortCtx": r["shortctx_rmse"],
                    "HBML_NoXGBNODE": r["no_xgb_node_rmse"],
                    "FTL": r["ftl_rmse"],
                    "Uniform": r["uniform_rmse"],
                    "HBML_MaxAE": r["hbml_maxae"],
                }
                for c in cols:
                    row[c] = r["exp_rmses"][c]
                rmse_rows.append(row)

            # Summary row + mechanism summary
            hb_arr = np.array([r["hbml_rmse"] for r in p_res], dtype=float)
            s_row = {
                "dataset": dataset,
                "horizon": h_disp,
                "context": ctx,
                "n": n_p,
                "best_expert": best_exp,
                "HBML_mean": float(np.mean(hb_arr)),
                "HBML_sd": float(np.std(hb_arr, ddof=1)) if n_p > 1 else float("nan"),
                "BestExpert_mean": exp_means[best_exp],
                "Diff_BestMinusHBML": exp_means[best_exp] - float(np.mean(hb_arr)),
                "HBML_MaxAE_mean": float(np.mean([r["hbml_maxae"] for r in p_res])),
                "BestExpert_MaxAE_mean": float(np.mean([r["exp_maxaes"][best_exp] for r in p_res])),
                "HBML_Interleaved_mean": float(np.mean([r["interleaved_rmse"] for r in p_res])),
                "HBML_ShortCtx_mean": float(np.nanmean([r["shortctx_rmse"] for r in p_res])),
                "HBML_NoXGBNODE_mean": float(np.nanmean([r["no_xgb_node_rmse"] for r in p_res])),
                "FTL_mean": float(np.mean([r["ftl_rmse"] for r in p_res])),
                "Uniform_mean": float(np.mean([r["uniform_rmse"] for r in p_res])),
            }
            for c in cols:
                s_row[f"{c}_mean"] = exp_means[c]
            rmse_summary_rows.append(s_row)

            # Mechanism comparison summary vs HBML
            variants = {
                "HBML": hb_arr,
                "HBML-interleaved": np.array([r["interleaved_rmse"] for r in p_res], dtype=float),
                "HBML-no-XGB-NODE": np.array([r["no_xgb_node_rmse"] for r in p_res], dtype=float),
                "HBML-shortctx-pool": np.array([r["shortctx_rmse"] for r in p_res], dtype=float),
                "FTL": np.array([r["ftl_rmse"] for r in p_res], dtype=float),
                "Uniform": np.array([r["uniform_rmse"] for r in p_res], dtype=float),
            }
            for vname, varr in variants.items():
                valid = np.isfinite(varr) & np.isfinite(hb_arr)
                if not np.any(valid):
                    continue
                diff = varr[valid] - hb_arr[valid]
                lo, hi = apm.boot_ci(diff, n_boot=n_boot_mech)
                mech_rows.append({
                    "dataset": dataset, "horizon": h_disp, "context": ctx, "variant": vname,
                    "n": int(valid.sum()), "mean_rmse": float(np.mean(varr[valid])),
                    "diff_vs_hbml": float(np.mean(diff)), "diff_ci95_low": lo, "diff_ci95_high": hi,
                })

            # Paired statistical tests across all experts (reusing calculate_statistics.paired_stats)
            cond_stat_rows = []
            for c in cols:
                e_arr = np.array([r["exp_rmses"][c] for r in p_res], dtype=float)
                res_st = cstat.paired_stats(e_arr, hb_arr, n_boot=n_boot_stat)
                res_st.update({
                    "dataset": dataset,
                    "horizon": float(h_disp),
                    "context": ctx,
                    "expert": c,
                    "is_best_expert": (c == best_exp),
                    "source": "external_raw",
                })
                cond_stat_rows.append(res_st)
                if c == best_exp:
                    stat_best_rows.append(res_st)
            stat_all_rows.extend(cond_stat_rows)

            # Weights summary
            dom_counts = Counter(r["dom_expert"] for r in p_res)
            w_row = {
                "dataset": dataset, "horizon": h_disp, "context": ctx, "n": n_p,
                "hbml_rmse": float(np.mean(hb_arr)),
                "mean_max_weight": float(np.mean([r["mean_max_weight"] for r in p_res])),
                "dom_pct_ARIMA_ETS": 100.0 * (dom_counts.get("ARIMA", 0) + dom_counts.get("ETS", 0)) / n_p,
            }
            for c in ecw.ALL_EXPERTS:
                w_row[f"mean_w_{c}"] = float(np.mean([r["mean_weights"].get(c, 0.0) for r in p_res])) if c in cols else 0.0
                w_row[f"dom_pct_{c}"] = 100.0 * dom_counts.get(c, 0) / n_p
            weight_rows.append(w_row)

            # Clinical evaluation (full-history primary setting + 30-min sweep)
            if ctx == "full":
                best_k = cols.index(best_exp)
                specs = (
                    ("HBML", lambda r: r["pred_primary"]),
                    (f"BestExpert ({best_exp})", lambda r, bk=best_k: r["F"][:, bk]),
                    ("Persistence", lambda r, hs=h_steps: np.r_[np.zeros(hs), r["y"][:-hs]]),
                )
                for m_label, get_pred in specs:
                    ep_tot = det_tot = det_any_tot = tp_tot = fp_tot = fn_tot = fc_tot = tc_tot = 0
                    chance_tot = days_tot = days_cal_tot = 0.0
                    leads_det, leads_all, pa_list, pab_list = [], [], [], []
                    cens_tot = noeval_tot = n_p_ep = 0
                    det_w = Counter()
                    for r in p_res:
                        dtype = diabetes_types.get(r["pid"], 1)
                        ev_r = ecw.evaluate_hypo_trajectory(r["y"], get_pred(r), r["scored_mask"], h_steps, diabetes_type=dtype)
                        if not ev_r:
                            continue
                        cens_tot += ev_r["n_episodes_left_censored"]
                        noeval_tot += ev_r["n_episodes_not_evaluable"]
                        n_p_ep += int(ev_r["n_episodes"] > 0)
                        for w in ecw.WARNING_MINUTES:
                            det_w[w] += ev_r[f"n_detected_w{w}"]
                        ep_tot += ev_r["n_episodes"]
                        det_tot += ev_r["n_detected"]
                        det_any_tot += ev_r["n_detected_any"]
                        chance_tot += ev_r["chance_detected_sum"]
                        tp_tot += ev_r["tp"]
                        fp_tot += ev_r["fp"]
                        fn_tot += ev_r["fn"]
                        days_tot += ev_r["days"]
                        days_cal_tot += ev_r["days_calendar"]
                        fc_tot += ev_r["false_clusters"]
                        tc_tot += ev_r["true_clusters"]
                        leads_det.extend(ev_r["lead_times_det"])
                        leads_all.extend(ev_r["lead_times_all"])
                        pa_list.append(ev_r["parkes_a_pct"])
                        pab_list.append(ev_r["parkes_ab_pct"])
                    ep_sens = 100.0 * det_tot / ep_tot if ep_tot > 0 else float("nan")
                    ch_sens = 100.0 * chance_tot / ep_tot if ep_tot > 0 else float("nan")
                    pt_sens = 100.0 * tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else float("nan")
                    pt_ppv = 100.0 * tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else float("nan")
                    cl_tot = tc_tot + fc_tot
                    hypo_rows.append({
                        "dataset": dataset, "horizon": h_disp, "context": ctx,
                        "model": m_label.split(" ")[0], "best_expert_name": best_exp,
                        "n_participants": n_p, "n_participants_with_episodes": n_p_ep,
                        "n_episodes": ep_tot, "n_episodes_left_censored": cens_tot,
                        "n_episodes_not_evaluable": noeval_tot, "n_detected": det_tot,
                        "episode_sensitivity_pct": ep_sens,
                        **{f"sensitivity_warn{w}min_pct": (100.0 * det_w[w] / ep_tot if ep_tot else float("nan"))
                           for w in ecw.WARNING_MINUTES},
                        "episode_sensitivity_any_pct": 100.0 * det_any_tot / ep_tot if ep_tot > 0 else float("nan"),
                        "chance_sensitivity_pct": ch_sens,
                        "above_chance_pp": (ep_sens - ch_sens) if ep_tot > 0 else float("nan"),
                        "mean_lead_time_min": sum(leads_all) / len(leads_all) if leads_all else 0.0,
                        "mean_lead_time_detected_min": sum(leads_det) / len(leads_det) if leads_det else 0.0,
                        "point_sensitivity_pct": pt_sens, "point_ppv_pct": pt_ppv,
                        "point_f1_pct": (2 * pt_sens * pt_ppv / (pt_sens + pt_ppv)) if (pt_sens + pt_ppv) > 0 else float("nan"),
                        "n_alert_clusters": cl_tot, "true_alert_clusters": tc_tot,
                        "cluster_ppv_pct": 100.0 * tc_tot / cl_tot if cl_tot > 0 else float("nan"),
                        "false_alerts_per_day": fc_tot / max(days_tot, 1e-6),
                        "false_alerts_per_calendar_day": fc_tot / max(days_cal_tot, 1e-6),
                        "parkes_a_pct": sum(pa_list) / len(pa_list),
                        "parkes_ab_pct": sum(pab_list) / len(pab_list),
                    })

                if h_str == "half":
                    for thr in ecw.SWEEP_THRESHOLDS:
                        for m_label, get_pred in specs:
                            ep_tot = det_tot = fc_tot = tc_tot = 0
                            chance_tot = days_tot = 0.0
                            det_w = Counter()
                            for r in p_res:
                                ev_r = ecw.evaluate_hypo_trajectory(
                                    r["y"], get_pred(r), r["scored_mask"], h_steps,
                                    thresh=thr, true_thresh=70.0, compute_parkes=False,
                                )
                                if not ev_r:
                                    continue
                                sweep_pp_rows.append(dict(
                                    model=m_label.split(" ")[0], threshold_mgdl=thr, pid=r["pid"],
                                    n_episodes=ev_r["n_episodes"], false_clusters=ev_r["false_clusters"],
                                    days=ev_r["days"],
                                    **{f"n_detected_w{w}": ev_r[f"n_detected_w{w}"] for w in ecw.WARNING_MINUTES},
                                ))
                                for w in ecw.WARNING_MINUTES:
                                    det_w[w] += ev_r[f"n_detected_w{w}"]
                                ep_tot += ev_r["n_episodes"]
                                det_tot += ev_r["n_detected"]
                                chance_tot += ev_r["chance_detected_sum"]
                                days_tot += ev_r["days"]
                                fc_tot += ev_r["false_clusters"]
                                tc_tot += ev_r["true_clusters"]
                            ep_sens = 100.0 * det_tot / ep_tot if ep_tot > 0 else float("nan")
                            ch_sens = 100.0 * chance_tot / ep_tot if ep_tot > 0 else float("nan")
                            cl_tot = tc_tot + fc_tot
                            sweep_rows.append({
                                "dataset": dataset, "horizon": h_disp, "context": ctx,
                                "threshold_mgdl": thr, "model": m_label.split(" ")[0],
                                "best_expert_name": best_exp, "n_participants": n_p,
                                "n_episodes": ep_tot, "n_detected": det_tot,
                                "episode_sensitivity_pct": ep_sens,
                                "chance_sensitivity_pct": ch_sens,
                                "above_chance_pp": (ep_sens - ch_sens) if ep_tot > 0 else float("nan"),
                                "cluster_ppv_pct": 100.0 * tc_tot / cl_tot if cl_tot > 0 else float("nan"),
                                "false_alerts_per_day": fc_tot / max(days_tot, 1e-6),
                            })

    if not rmse_summary_rows:
        raise SystemExit(f"No locked engine-{ecw.ENGINE_VERSION} forecasts found for dataset={dataset} under {outputs_dir}")

    # Apply Benjamini-Hochberg FDR across statistical families
    if stat_best_rows:
        adj = cstat.bh_adjust([r["wilcoxon_p"] for r in stat_best_rows])
        for r, p in zip(stat_best_rows, adj):
            r["wilcoxon_p_bh"] = p
    if stat_all_rows:
        adj_all = cstat.bh_adjust([r["wilcoxon_p"] for r in stat_all_rows])
        for r, p in zip(stat_all_rows, adj_all):
            r["wilcoxon_p_bh"] = p

    write_csv(os.path.join(out_dir, "rmse_participant.csv"), rmse_rows)
    write_csv(os.path.join(out_dir, "rmse_summary.csv"), rmse_summary_rows)
    write_csv(os.path.join(out_dir, "statistical_tests_best_expert.csv"), stat_best_rows)
    write_csv(os.path.join(out_dir, "statistical_tests_all_experts.csv"), stat_all_rows)
    write_csv(os.path.join(out_dir, "weights_summary.csv"), weight_rows)
    write_csv(os.path.join(out_dir, "clinical_hypo.csv"), hypo_rows)
    write_csv(os.path.join(out_dir, "clinical_hypo_sweep.csv"), sweep_rows)
    write_csv(os.path.join(out_dir, "pool_mechanism_summary.csv"), mech_rows)

    if sweep_pp_rows:
        mfar = ecw.matched_far_comparison(sweep_pp_rows)
        write_csv(os.path.join(out_dir, "clinical_hypo_matched_far.csv"), mfar)

    # Emit Markdown report
    lines = [
        f"# External Cohort Evaluation: `{dataset}` (Engine v{ecw.ENGINE_VERSION}, {ecw.PRIMARY_HBML})",
        "",
        f"Evaluated conditions: **{len(rmse_summary_rows)}**.",
        "",
        "## RMSE & Equivalence vs Best Single Expert",
        "| Horizon | Context | N | HBML RMSE | Best Expert | Best RMSE | Diff (Best - HBML) | TOST p (1.0 mg/dL) | Interleaved RMSE | ShortCtx-Pool RMSE |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    st_lookup = {(str(r["horizon"]).replace(".0", "") if r["horizon"] != 0.5 else "0.5", r["context"]): r
                 for r in stat_best_rows}
    for s in rmse_summary_rows:
        st = st_lookup.get((s["horizon"], s["context"]), {})
        tp_val = st.get("tost_boot_p", float("nan"))
        lines.append(
            f"| {s['horizon']} hr | {s['context']} | {s['n']} | {s['HBML_mean']:.2f} | "
            f"{s['best_expert']} | {s['BestExpert_mean']:.2f} | {s['Diff_BestMinusHBML']:+.2f} | "
            f"{tp_val:.4g} | {s['HBML_Interleaved_mean']:.2f} | {s['HBML_ShortCtx_mean']:.2f} |"
        )
    if hypo_rows:
        lines += [
            "",
            "## Pre-Onset Hypoglycaemia Detection (<70 mg/dL, Full History)",
            "| Horizon | Model | N | Episodes | Pre-onset Sens (%) | Above Chance (pp) | Cluster PPV (%) | FAR (scored/day) | FAR (cal/day) | Parkes A+B (%) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for h in hypo_rows:
            lines.append(
                f"| {h['horizon']} hr | {h['model']} | {h['n_participants']} | {h['n_episodes']} | "
                f"{h['episode_sensitivity_pct']:.1f} | {h['above_chance_pp']:+.1f} | "
                f"{h['cluster_ppv_pct']:.1f} | {h['false_alerts_per_day']:.2f} | "
                f"{h['false_alerts_per_calendar_day']:.2f} | {h['parkes_ab_pct']:.1f} |"
            )

    with open(os.path.join(out_dir, "external_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote external cohort report and CSVs to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Engine v2.3 on an external cohort (e.g. ohiot1dm).")
    parser.add_argument("--dataset", required=True, help="External dataset directory name (e.g. ohiot1dm).")
    parser.add_argument("--outputs-dir", default=os.path.join(BASE_DIR, "Outputs"))
    parser.add_argument("--out-dir", default=None, help="Output directory (default: Outputs/external/<dataset>).")
    parser.add_argument("--n-boot-stat", type=int, default=10000)
    parser.add_argument("--n-boot-mech", type=int, default=2000)
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.outputs_dir, "external", args.dataset)
    evaluate_cohort(args.dataset, args.outputs_dir, out_dir,
                    n_boot_stat=args.n_boot_stat, n_boot_mech=args.n_boot_mech)


if __name__ == "__main__":
    main()
