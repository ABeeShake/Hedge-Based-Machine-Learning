#!/usr/bin/env python3
"""evaluate_clinical_and_weights.py
===============================================================================
Computes three sets of manuscript tables and diagnostics directly from the saved
forecast trajectories (`Outputs/<dataset>/h-*/context-*/forecasts/*_forecasts.csv`)
and raw glucose recordings (`<dataset>/*.csv`):

1. Weight & Aggregator Mechanism Diagnostics (`overleaf/tables/weight_diagnostics.csv`)
   - Mean maximum expert weight (`mean_max_weight`) of HBML-SFHDF over the evaluation window
   - Mean weight of each constituent expert (`mean_w_<expert>`)
   - Share of participants where each expert (and ARIMA/ETS combined) has the largest
     weight at the end of the recording (`dom_pct_<expert>`, `dom_pct_ARIMA_ETS`)
   - RMSE of Follow-the-Leader (`ftl_rmse`) and Orabona-Pal sqrt-scale-free Hedge
     (`sfh_sqrt_rmse`), plus 7-expert pool RMSE (`hbml_pool7_rmse`) and clipped-forecast
     RMSEs (`clipped_hbml_rmse`, `clipped_best_expert_rmse`, `clipped_<expert>_rmse`).
   - Also updates `fs_comparison/summary.csv` with `ftl` and `sfh_sqrt` rows so
     `generate_results_macros.py` includes Follow-the-Leader and Orabona-Pal SFH-Sqrt
     in `overleaf/tables/table_aggregator_ablation.tex`.

2. Event-Level Hypoglycemia Detection & Parkes Error Grid (`overleaf/tables/hypoglycemia_events.csv`
   and `overleaf/tables/table_hypoglycemia_events.tex`)
   - Evaluates HBML, the best constituent expert, and Persistence across horizons (0.5, 2, 5 hr)
     in the full-history configuration on both cohorts:
     * True hypoglycemic episodes (< 70 mg/dL for >= 15 min [>= 3 consecutive slots],
       separated by >= 30 min [>= 6 slots] >= 70 mg/dL)
     * Episode detection sensitivity (%), mean lead time (min), point-wise sensitivity (%),
       precision/PPV (%), F1 (%), false-alert clusters per participant-day, and
       Parkes Error Grid Zone A (%) and Zone A+B (%).

3. Regime-Stratified Accuracy & Winner Frequency Table (`overleaf/tables/regime_summary_table.csv`
   and `overleaf/tables/table_regime_summary.tex`)
   - Classifies scored timesteps into the 6 operational regimes (`STABLE`, `RISING`,
     `FALLING`, `HIGHVAR`, `RECOVERY`, `HYPO_TRANS`) and reports timestep share (%),
     HBML RMSE, best single expert RMSE, and most frequently winning expert per regime.

Pure Python stdlib implementation (`csv`, `math`, `statistics`) so it runs in < 10 seconds
both locally (`/usr/bin/python3`) and on the VM (`run_on_vm.sh`), and automatically adapts
between Engine v1 (legacy forecasts) and Engine v2.1 (`.meta.json` 5-minute grid forecasts).
===============================================================================
"""

import csv
import glob
import json
import math
import os
import statistics
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(BASE_DIR, "overleaf", "tables")
OUTPUTS_DIR = os.path.join(BASE_DIR, "Outputs")
FS_DIR = os.path.join(BASE_DIR, "fs_comparison")

HORIZONS = [("half", "0.5", 6), ("2", "2", 24), ("5", "5", 60)]
CONTEXTS = ["6", "12", "24", "full"]
BASE_EXPERTS = ["ARIMA", "ETS", "NHITS", "NODE", "XGBoost"]
ALL_EXPERTS = ["ARIMA", "ETS", "NHITS", "NODE", "XGBoost", "Persistence", "RunningMean"]
EPS = 1e-8


def is_v2_forecast(fc_path):
    meta_path = fc_path[:-4] + ".meta.json"
    if not os.path.exists(meta_path):
        return False
    try:
        meta = json.load(open(meta_path))
        return meta.get("settings", {}).get("engine_version") == "2.1"
    except Exception:
        return False


_RAW_CACHE = {}


def load_raw_glucose(dataset, pid):
    """Return list of floats for participant pid in dataset (cached)."""
    key = (dataset, pid)
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]
    path = os.path.join(BASE_DIR, dataset, f"{pid}.csv")
    if not os.path.exists(path):
        _RAW_CACHE[key] = None
        return None
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        col_name = "Libre.GL" if "Libre.GL" in header else "Dexcom.GL"
        col_idx = header.index(col_name)
        vals = []
        for line in f:
            parts = line.strip().split(",")
            if len(parts) > col_idx:
                try:
                    v = float(parts[col_idx])
                    if math.isfinite(v):
                        vals.append(v)
                except ValueError:
                    pass
    _RAW_CACHE[key] = vals
    return vals


def load_targets_for_file(dataset, pid, v2_mode):
    if v2_mode:
        try:
            from ExpMethods import evaluation as ev
            t = ev.load_targets(dataset, pid, root=BASE_DIR)
            return [float(x) if (x is not None and math.isfinite(float(x))) else float("nan") for x in t]
        except Exception:
            pass
    return load_raw_glucose(dataset, pid)


def load_forecasts_csv(fc_path):
    with open(fc_path, "r") as f:
        header = f.readline().strip().split(",")
        cols = []
        col_idxs = []
        for idx, c in enumerate(header):
            c_clean = "ETS" if c == "HoltWinters" else c
            if c_clean in ALL_EXPERTS:
                cols.append(c_clean)
                col_idxs.append(idx)
        K = len(cols)
        data = {c: [] for c in cols}
        col_lists = [data[c] for c in cols]
        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= col_idxs[-1]:
                continue
            for k in range(K):
                try:
                    col_lists[k].append(float(parts[col_idxs[k]]))
                except ValueError:
                    col_lists[k].append(float("nan"))
    return cols, data


def run_aggregators_single(cols, f_data, y, start, h, eval_start, eval_tail, gamma=0.2, clip_bounds=None):
    """Run HBML-SFHDF, FTL, and SFH-Sqrt on a single participant trajectory."""
    K = len(cols)
    n = min(len(y), len(f_data[cols[0]]))
    if n <= eval_start + eval_tail + 10:
        return None

    # Build matrix F[t][k]
    lo_b, hi_b = clip_bounds if clip_bounds else (-float("inf"), float("inf"))
    F = []
    for t in range(n):
        row = []
        for c in cols:
            v = f_data[c][t]
            if clip_bounds and math.isfinite(v) and v != 0.0:
                v = min(max(v, lo_b), hi_b)
            row.append(v)
        F.append(row)

    # Scorable rows: finite y[t] and all F[t][k] finite & non-zero
    scorable = [False] * n
    for t in range(n):
        yt = y[t]
        if math.isfinite(yt):
            ok = True
            for k in range(K):
                fk = F[t][k]
                if not (math.isfinite(fk) and fk != 0.0):
                    ok = False
                    break
            scorable[t] = ok

    # Eval window mask
    end_idx = n - eval_tail
    scored_mask = [scorable[t] and (eval_start <= t < end_idx) for t in range(n)]
    n_scored = sum(1 for x in scored_mask if x)
    if n_scored < 10:
        return None

    # Online losses L[t][k]
    L_mat = [[0.0] * K for _ in range(n)]
    for t in range(start + h, n):
        if scorable[t]:
            yt = y[t]
            F_t = F[t]
            L_mat[t] = [(F_t[k] - yt) ** 2 for k in range(K)]

    # State for 1) HBML-SFHDF, 2) FTL, 3) SFH-Sqrt
    cum = [0.0] * K
    L_ema = EPS
    sum_sq = 0.0
    lnK = max(math.log(max(K, 2)), 1e-12)

    w_sfhdf = [1.0 / K] * K
    w_ftl = [1.0 / K] * K
    w_sqrt = [1.0 / K] * K

    pred_sfhdf = [0.0] * n
    se_sfhdf = [float("nan")] * n
    se_ftl = [float("nan")] * n
    se_sqrt = [float("nan")] * n

    max_w_sum = 0.0
    w_sums = [0.0] * K
    w_count = 0

    for g in range(start, n):
        lg = L_mat[g]
        if any(x != 0.0 for x in lg):
            l_top = max(lg)
            L_ema = (1.0 - gamma) * L_ema + gamma * l_top
            L_eff = max(L_ema, EPS)

            sum_sq += l_top * l_top
            L_sq_eff = max(math.sqrt(sum_sq / lnK), EPS)

            for k in range(K):
                cum[k] += lg[k]

            c_min = min(cum)
            # 1) HBML-SFHDF
            exps = [math.exp(min(0.0, -(cum[k] - c_min) / L_eff)) for k in range(K)]
            s_e = sum(exps)
            w_sfhdf = [x / s_e for x in exps] if s_e > 1e-12 else [1.0 / K] * K

            # 2) FTL
            leaders = [1.0 if abs(cum[k] - c_min) <= 1e-12 * max(1.0, abs(c_min)) else 0.0 for k in range(K)]
            s_l = sum(leaders)
            w_ftl = [x / s_l for x in leaders] if s_l > 0 else [1.0 / K] * K

            # 3) SFH-Sqrt (Orabona-Pal)
            exps_sq = [math.exp(min(0.0, -(cum[k] - c_min) / L_sq_eff)) for k in range(K)]
            s_sq = sum(exps_sq)
            w_sqrt = [x / s_sq for x in exps_sq] if s_sq > 1e-12 else [1.0 / K] * K

        t_target = g + h
        if t_target < n:
            if scored_mask[t_target]:
                max_w_sum += max(w_sfhdf)
                for k in range(K):
                    w_sums[k] += w_sfhdf[k]
                w_count += 1
            if scorable[t_target]:
                yt = y[t_target]
                Ft = F[t_target]
                p1 = sum(w_sfhdf[k] * Ft[k] for k in range(K))
                p2 = sum(w_ftl[k] * Ft[k] for k in range(K))
                p3 = sum(w_sqrt[k] * Ft[k] for k in range(K))
                pred_sfhdf[t_target] = p1
                se_sfhdf[t_target] = (p1 - yt) ** 2
                se_ftl[t_target] = (p2 - yt) ** 2
                se_sqrt[t_target] = (p3 - yt) ** 2

    def calc_rmse(se_arr):
        vals = [se_arr[t] for t in range(n) if scored_mask[t] and math.isfinite(se_arr[t])]
        return math.sqrt(sum(vals) / len(vals)) if vals else float("nan")

    exp_rmses = {}
    for k, c in enumerate(cols):
        vals = [L_mat[t][k] for t in range(n) if scored_mask[t]]
        exp_rmses[c] = math.sqrt(sum(vals) / len(vals)) if vals else float("nan")

    dom_idx = max(range(K), key=lambda k: w_sfhdf[k])
    return {
        "n": n,
        "scored_mask": scored_mask,
        "y": y,
        "F": F,
        "cols": cols,
        "pred_sfhdf": pred_sfhdf,
        "hbml_rmse": calc_rmse(se_sfhdf),
        "ftl_rmse": calc_rmse(se_ftl),
        "sfh_sqrt_rmse": calc_rmse(se_sqrt),
        "exp_rmses": exp_rmses,
        "mean_max_weight": max_w_sum / max(w_count, 1),
        "mean_weights": {cols[k]: w_sums[k] / max(w_count, 1) for k in range(K)},
        "dom_expert": cols[dom_idx],
    }


def parkes_type1_ab(r, p):
    """Classify (reference r, predicted p) in Parkes (Consensus Type 1) Error Grid.
    Returns ('A', 'B', or 'CDE').
    Based on Parkes et al. (2000) / Pfutzner et al. (2013) polygon boundaries for Type 1 diabetes.
    """
    if not (math.isfinite(r) and math.isfinite(p)):
        return "unknown"
    # Zone A: within 20% for r >= 75, or within 15 mg/dL for r < 75 (or both <= 50)
    if (r <= 50 and p <= 50) or abs(p - r) <= max(15.0, 0.20 * r):
        return "A"
    # Outside Zone B (Zones C, D, E):
    # Upper C/D/E boundary: (r=30, p=110), (r=50, p=130), (r=70, p=180), (r=120, p=260), (r=260, p=550)
    if r <= 30 and p >= 110:
        return "CDE"
    if 30 < r <= 50 and p >= 110 + (r - 30) * (130 - 110) / 20.0:
        return "CDE"
    if 50 < r <= 70 and p >= 130 + (r - 50) * (180 - 130) / 20.0:
        return "CDE"
    if 70 < r <= 120 and p >= 180 + (r - 70) * (260 - 180) / 50.0:
        return "CDE"
    if r > 120 and p >= 260 + (r - 120) * (550 - 260) / 140.0:
        return "CDE"
    # Lower C/D/E boundary: (r=120, p=50), (r=150, p=65), (r=250, p=130), (r=550, p=250)
    if 120 <= r <= 150 and p <= 50 + (r - 120) * (65 - 50) / 30.0:
        return "CDE"
    if 150 < r <= 250 and p <= 65 + (r - 150) * (130 - 65) / 100.0:
        return "CDE"
    if r > 250 and p <= 130 + (r - 250) * (250 - 130) / 300.0:
        return "CDE"
    return "B"


def evaluate_hypo_trajectory(y, pred, scored_mask, h, thresh=70.0, min_ep_slots=3, min_gap_slots=6):
    """Evaluate episode-level and point-level hypoglycemia (<70 mg/dL) detection on one participant."""
    idx = [t for t, ok in enumerate(scored_mask) if ok]
    if not idx:
        return None
    n_scored = len(idx)
    days = n_scored / 288.0

    # Point-wise metrics & Parkes error grid on scored indices
    tp = fp = fn = tn = 0
    parkes_a = parkes_ab = 0
    for t in idx:
        yt, pt = y[t], pred[t]
        z = parkes_type1_ab(yt, pt)
        if z == "A":
            parkes_a += 1
            parkes_ab += 1
        elif z == "B":
            parkes_ab += 1
        is_hypo_true = yt < thresh
        is_hypo_pred = pt < thresh
        if is_hypo_true and is_hypo_pred:
            tp += 1
        elif not is_hypo_true and is_hypo_pred:
            fp += 1
        elif is_hypo_true and not is_hypo_pred:
            fn += 1
        else:
            tn += 1

    # Identify contiguous true hypoglycemic episodes (< 70 mg/dL for >= min_ep_slots, merged if gap < min_gap_slots)
    raw_runs = []
    cur_start = None
    cur_end = None
    for t in idx:
        if y[t] < thresh:
            if cur_start is None:
                cur_start = t
                cur_end = t
            elif t - cur_end <= 2:
                cur_end = t
            else:
                raw_runs.append([cur_start, cur_end])
                cur_start, cur_end = t, t
    if cur_start is not None:
        raw_runs.append([cur_start, cur_end])

    # Merge runs separated by < min_gap_slots
    merged = []
    for st, en in raw_runs:
        if merged and (st - merged[-1][1] < min_gap_slots):
            merged[-1][1] = en
        else:
            merged.append([st, en])

    episodes = [(st, en) for st, en in merged if (en - st + 1) >= min_ep_slots]

    detected = 0
    lead_times_min = []
    for st, en in episodes:
        # A forecast targeting slot t in [st, en] was issued at t - h.
        first_alert = None
        for t in range(st, en + 1):
            if scored_mask[t] and pred[t] < thresh:
                first_alert = t
                break
        if first_alert is not None:
            detected += 1
            # Alert was issued at (first_alert - h), so lead time before episode onset `st` is:
            lead_slots = max(0, h - (first_alert - st))
            lead_times_min.append(lead_slots * 5.0)

    # False alert clusters: contiguous runs of pred[t] < thresh where NO true y < thresh occurs within +/- 2 slots (10 min) of target
    pred_runs = []
    p_start = None
    p_end = None
    for t in idx:
        if pred[t] < thresh:
            if p_start is None:
                p_start = t
                p_end = t
            elif t - p_end <= min_gap_slots:
                p_end = t
            else:
                pred_runs.append((p_start, p_end))
                p_start, p_end = t, t
    if p_start is not None:
        pred_runs.append((p_start, p_end))

    false_clusters = 0
    n_tot = len(y)
    for st, en in pred_runs:
        w_lo = max(0, st - 2)
        w_hi = min(n_tot - 1, en + 2)
        any_true = any(scored_mask[tau] and y[tau] < thresh for tau in range(w_lo, w_hi + 1))
        if not any_true:
            false_clusters += 1

    return {
        "n_scored": n_scored,
        "days": days,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_episodes": len(episodes),
        "n_detected": detected,
        "lead_times": lead_times_min,
        "false_clusters": false_clusters,
        "parkes_a_pct": 100.0 * parkes_a / n_scored,
        "parkes_ab_pct": 100.0 * parkes_ab / n_scored,
    }


_REGIME_CACHE = {}


def classify_regimes_stdlib(dataset, pid, y):
    """Classify timesteps into the 6 operational regimes of compute_regimes.py (cached per participant)."""
    key = (dataset, pid, len(y))
    if key in _REGIME_CACHE:
        return _REGIME_CACHE[key]
    n = len(y)
    regimes = ["STABLE"] * n
    for t in range(n):
        gt = y[t]
        if not math.isfinite(gt):
            regimes[t] = "UNKNOWN"
            continue
        # 6-step (30-min) OLS slope (mg/dL per 5-min step)
        w_s = [ (i, y[t - 5 + i]) for i in range(6) if 0 <= t - 5 + i < n and math.isfinite(y[t - 5 + i]) ]
        if len(w_s) >= 3:
            xm = sum(p[0] for p in w_s) / len(w_s)
            ym = sum(p[1] for p in w_s) / len(w_s)
            den = sum((p[0] - xm) ** 2 for p in w_s)
            slope = sum((p[0] - xm) * (p[1] - ym) for p in w_s) / den if den > 0 else 0.0
        else:
            slope = float("nan")

        # 12-step (60-min) rolling sample variance
        w_v = [y[tau] for tau in range(max(0, t - 11), t + 1) if math.isfinite(y[tau])]
        if len(w_v) >= 6:
            vm = sum(w_v) / len(w_v)
            var = sum((x - vm) ** 2 for x in w_v) / (len(w_v) - 1)
        else:
            var = float("nan")

        reg = "STABLE"
        if math.isfinite(slope) and slope < -1.5:
            reg = "FALLING"
        if math.isfinite(slope) and slope > 1.5:
            reg = "RISING"
        if math.isfinite(var) and var >= 100.0:
            reg = "HIGHVAR"
        if math.isfinite(slope) and ((50.0 < gt < 70.0 and slope > 0.0) or (180.0 < gt < 200.0 and slope < 0.0)):
            reg = "RECOVERY"
        if gt < 80.0 and math.isfinite(slope) and slope < -1.0:
            reg = "HYPO_TRANS"
        regimes[t] = reg
    _REGIME_CACHE[key] = regimes
    return regimes


def wilcoxon_approx_p(diffs):
    d = [x for x in diffs if math.isfinite(x) and abs(x) > 1e-12]
    n = len(d)
    if n < 6:
        return float("nan")
    abs_d = sorted((abs(x), i) for i, x in enumerate(d))
    ranks = [0.0] * n
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j < n and abs(abs_d[j][0] - abs_d[i][0]) <= 1e-12:
            j += 1
        avg_r = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[abs_d[k][1]] = avg_r
        t_cnt = j - i
        tie_term += (t_cnt ** 3 - t_cnt)
        i = j
    w_pos = sum(ranks[idx] for idx in range(n) if d[idx] > 0)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return 1.0
    z = (w_pos - mu) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2.0))


def main():
    import sys as _sys
    only_full = "--full-only" in _sys.argv
    os.makedirs(TABLES_DIR, exist_ok=True)
    # Load reference HBML means from rmse_pivot_mean.csv so we detect which local Outputs match the tables
    ref_means = {}
    piv_path = os.path.join(TABLES_DIR, "rmse_pivot_mean.csv")
    if os.path.exists(piv_path):
        with open(piv_path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("model") == "HBML":
                    ds = r["dataset"].strip().lower()
                    hz = "half" if r["horizon"].strip() in ("0.5", "half") else r["horizon"].strip().replace(".0", "")
                    for c in CONTEXTS:
                        s = (r.get(f"mean.{c}") or "").replace("\\textbf{", "").replace("}", "").strip()
                        try:
                            ref_means[(ds, hz, c)] = float(s)
                        except ValueError:
                            pass

    weight_rows = []
    hypo_rows = []
    regime_rows = []
    new_fs_rows = []

    for ds in ("weinstock", "cgmacros"):
        for h_str, h_disp, h_steps in HORIZONS:
            for ctx in (["full"] if only_full else CONTEXTS):
                fc_dir = os.path.join(OUTPUTS_DIR, ds, f"h-{h_str}hr", f"context-{ctx}hr", "forecasts")
                files = sorted(f for f in glob.glob(os.path.join(fc_dir, "*_forecasts.csv"))
                               if os.path.basename(f)[3:] == "_forecasts.csv")
                if not files:
                    continue
                v2_mode = is_v2_forecast(files[0])
                start = 288 if v2_mode else 21
                eval_start = 348 if v2_mode else 100
                eval_tail = 10

                p_res = []
                p_res_clip = []
                p_res_pool7 = []
                for f_path in files:
                    pid = os.path.basename(f_path)[:3]
                    y = load_targets_for_file(ds, pid, v2_mode)
                    if not y:
                        continue
                    cols, f_data = load_forecasts_csv(f_path)
                    if len(cols) < 2:
                        continue
                    res = run_aggregators_single(cols, f_data, y, start, h_steps, eval_start, eval_tail)
                    if not res:
                        continue
                    res["pid"] = pid
                    p_res.append(res)

                    if ctx == "full":
                        # Also evaluate clipped [40, 400] and 7-expert pool (adding Persistence and RunningMean if absent)
                        res_c = run_aggregators_single(cols, f_data, y, start, h_steps, eval_start, eval_tail,
                                                       clip_bounds=(40.0, 400.0))
                        if res_c:
                            p_res_clip.append(res_c)
                        if "Persistence" not in cols:
                            n_y = len(y)
                            pers = [0.0] * n_y
                            clim = [0.0] * n_y
                            s_y = 0.0
                            c_y = 0
                            run_m = [0.0] * n_y
                            for t_i in range(n_y):
                                if math.isfinite(y[t_i]):
                                    s_y += y[t_i]
                                    c_y += 1
                                run_m[t_i] = s_y / max(c_y, 1)
                            for t_i in range(h_steps, n_y):
                                pers[t_i] = y[t_i - h_steps]
                                clim[t_i] = run_m[t_i - h_steps]
                            cols7 = cols + ["Persistence", "RunningMean"]
                            f_data7 = dict(f_data)
                            f_data7["Persistence"] = pers
                            f_data7["RunningMean"] = clim
                            res7 = run_aggregators_single(cols7, f_data7, y, start, h_steps, eval_start, eval_tail)
                            if res7:
                                p_res_pool7.append(res7)
                        else:
                            p_res_pool7.append(res)

                if not p_res:
                    continue

                n_p = len(p_res)
                hb_mean = sum(r["hbml_rmse"] for r in p_res) / n_p
                ref_m = ref_means.get((ds, h_str, ctx))
                matches = (ref_m is None) or (abs(hb_mean - ref_m) <= 0.01 * ref_m)
                print(f"Processed {ds} h={h_str} c={ctx}: n={n_p}, HBML={hb_mean:.2f}, matches={matches}", flush=True)

                cols = p_res[0]["cols"]
                exp_means = {c: sum(r["exp_rmses"][c] for r in p_res) / n_p for c in cols}
                best_exp = min(exp_means, key=exp_means.get)
                best_exp_mean = exp_means[best_exp]

                ftl_mean = sum(r["ftl_rmse"] for r in p_res) / n_p
                sqrt_mean = sum(r["sfh_sqrt_rmse"] for r in p_res) / n_p
                maxw_mean = sum(r["mean_max_weight"] for r in p_res) / n_p

                dom_counts = Counter(r["dom_expert"] for r in p_res)
                dom_arima_ets = dom_counts.get("ARIMA", 0) + dom_counts.get("ETS", 0)

                w_row = {
                    "dataset": ds,
                    "horizon": h_disp,
                    "context": ctx,
                    "n": n_p,
                    "matches_tables": "yes" if matches else "no",
                    "hbml_rmse": hb_mean,
                    "ftl_rmse": ftl_mean,
                    "sfh_sqrt_rmse": sqrt_mean,
                    "hbml_pool7_rmse": (sum(r["hbml_rmse"] for r in p_res_pool7) / len(p_res_pool7)) if p_res_pool7 else "",
                    "clipped_hbml_rmse": (sum(r["hbml_rmse"] for r in p_res_clip) / len(p_res_clip)) if p_res_clip else "",
                    "mean_max_weight": maxw_mean,
                    "dom_count_ARIMA_ETS": dom_arima_ets,
                    "dom_pct_ARIMA_ETS": 100.0 * dom_arima_ets / n_p,
                }
                for c in ALL_EXPERTS:
                    w_row[f"mean_w_{c}"] = (sum(r["mean_weights"].get(c, 0.0) for r in p_res) / n_p) if c in cols else 0.0
                    w_row[f"dom_pct_{c}"] = 100.0 * dom_counts.get(c, 0) / n_p
                    w_row[f"clipped_{c}_rmse"] = (
                        sum(r["exp_rmses"][c] for r in p_res_clip) / len(p_res_clip)
                        if (p_res_clip and c in cols) else ""
                    )
                weight_rows.append(w_row)

                # If this condition matches the tables, record FTL and SFH-Sqrt rows for fs_comparison/summary.csv
                if matches:
                    for vname, vkey in (("ftl", "ftl_rmse"), ("sfh_sqrt", "sfh_sqrt_rmse")):
                        v_vals = [r[vkey] for r in p_res]
                        h_vals = [r["hbml_rmse"] for r in p_res]
                        b_vals = [r["exp_rmses"][best_exp] for r in p_res]
                        d_cur = [h_vals[i] - v_vals[i] for i in range(n_p)]
                        d_best = [b_vals[i] - v_vals[i] for i in range(n_p)]
                        v_mean = sum(v_vals) / n_p
                        new_fs_rows.append({
                            "dataset": ds,
                            "horizon": h_str,
                            "context": ctx,
                            "n": n_p,
                            "variant": vname,
                            "mean_rmse": v_mean,
                            "median_rmse": statistics.median(v_vals),
                            "best_expert": best_exp,
                            "best_expert_mean_rmse": best_exp_mean,
                            "delta_vs_current": sum(d_cur) / n_p,
                            "pct_change_vs_current": 100.0 * (v_mean - hb_mean) / hb_mean,
                            "pct_participants_better_than_current": 100.0 * sum(1 for x in d_cur if x > 1e-9) / n_p,
                            "wilcoxon_p_vs_current": wilcoxon_approx_p(d_cur),
                            "delta_vs_best_expert": sum(d_best) / n_p,
                            "pct_participants_better_than_best_expert": 100.0 * sum(1 for x in d_best if x > 1e-9) / n_p,
                            "matches_tables": "yes",
                        })

                # Event-level hypoglycemia evaluation on full-history conditions
                if ctx == "full" and matches:
                    best_k = cols.index(best_exp)
                    for m_label, get_pred in (
                        ("HBML", lambda r: r["pred_sfhdf"]),
                        (f"BestExpert ({best_exp})", lambda r, bk=best_k: [row[bk] for row in r["F"]]),
                        ("Persistence", lambda r, hs=h_steps: [0.0] * hs + r["y"][:-hs]),
                    ):
                        ep_tot = det_tot = tp_tot = fp_tot = fn_tot = 0
                        days_tot = 0.0
                        fc_tot = 0
                        leads_all = []
                        pa_list = []
                        pab_list = []
                        for r in p_res:
                            ev_r = evaluate_hypo_trajectory(r["y"], get_pred(r), r["scored_mask"], h_steps)
                            if not ev_r:
                                continue
                            ep_tot += ev_r["n_episodes"]
                            det_tot += ev_r["n_detected"]
                            tp_tot += ev_r["tp"]
                            fp_tot += ev_r["fp"]
                            fn_tot += ev_r["fn"]
                            days_tot += ev_r["days"]
                            fc_tot += ev_r["false_clusters"]
                            leads_all.extend(ev_r["lead_times"])
                            pa_list.append(ev_r["parkes_a_pct"])
                            pab_list.append(ev_r["parkes_ab_pct"])
                        ep_sens = 100.0 * det_tot / ep_tot if ep_tot > 0 else float("nan")
                        pt_sens = 100.0 * tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else float("nan")
                        pt_ppv = 100.0 * tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else float("nan")
                        pt_f1 = (2 * pt_sens * pt_ppv / (pt_sens + pt_ppv)) if (pt_sens + pt_ppv) > 0 else float("nan")
                        far_day = fc_tot / max(days_tot, 1e-6)
                        lead_mean = sum(leads_all) / len(leads_all) if leads_all else 0.0
                        hypo_rows.append({
                            "dataset": ds,
                            "horizon": h_disp,
                            "context": ctx,
                            "model": m_label.split(" ")[0],
                            "best_expert_name": best_exp,
                            "n_participants": n_p,
                            "n_episodes": ep_tot,
                            "n_detected": det_tot,
                            "episode_sensitivity_pct": ep_sens,
                            "mean_lead_time_min": lead_mean,
                            "point_sensitivity_pct": pt_sens,
                            "point_ppv_pct": pt_ppv,
                            "point_f1_pct": pt_f1,
                            "false_alerts_per_day": far_day,
                            "parkes_a_pct": sum(pa_list) / len(pa_list),
                            "parkes_ab_pct": sum(pab_list) / len(pab_list),
                        })

                # Regime-stratified summary on 30-min full history
                if ctx == "full" and h_str == "half" and matches:
                    reg_order = ["STABLE", "RISING", "FALLING", "HIGHVAR", "RECOVERY", "HYPO_TRANS"]
                    reg_se_hbml = defaultdict(list)
                    reg_se_exp = {c: defaultdict(list) for c in cols}
                    reg_win_counts = {reg: Counter() for reg in reg_order}
                    reg_step_counts = Counter()
                    tot_scored = 0

                    for r in p_res:
                        y_r = r["y"]
                        regs = classify_regimes_stdlib(ds, r["pid"], y_r)
                        mask_r = r["scored_mask"]
                        p_hb = r["pred_sfhdf"]
                        F_r = r["F"]
                        # Per-participant regime SEs
                        p_se_hb = defaultdict(list)
                        p_se_ex = {c: defaultdict(list) for c in cols}
                        for t_i, ok in enumerate(mask_r):
                            if not ok:
                                continue
                            rg = regs[t_i]
                            if rg not in reg_win_counts:
                                continue
                            tot_scored += 1
                            reg_step_counts[rg] += 1
                            yt = y_r[t_i]
                            ehb = (p_hb[t_i] - yt) ** 2
                            p_se_hb[rg].append(ehb)
                            ex_errs = {}
                            for k_i, c in enumerate(cols):
                                e_k = (F_r[t_i][k_i] - yt) ** 2
                                p_se_ex[c][rg].append(e_k)
                                ex_errs[c] = e_k
                            ex_errs["HBML"] = ehb
                            min_e = min(ex_errs.values())
                            tied = [m for m, val in ex_errs.items() if abs(val - min_e) <= 1e-6]
                            for m in tied:
                                reg_win_counts[rg][m] += 1.0 / len(tied)

                        for rg in reg_order:
                            if p_se_hb[rg]:
                                reg_se_hbml[rg].append(math.sqrt(sum(p_se_hb[rg]) / len(p_se_hb[rg])))
                                for c in cols:
                                    reg_se_exp[c][rg].append(math.sqrt(sum(p_se_ex[c][rg]) / len(p_se_ex[c][rg])))

                    for rg in reg_order:
                        if not reg_se_hbml[rg]:
                            continue
                        hb_r_mean = sum(reg_se_hbml[rg]) / len(reg_se_hbml[rg])
                        exp_r_means = {c: sum(reg_se_exp[c][rg]) / len(reg_se_exp[c][rg]) for c in cols}
                        best_r_exp = min(exp_r_means, key=exp_r_means.get)
                        best_r_val = exp_r_means[best_r_exp]
                        # Top winning constituent expert by timestep frequency
                        exp_wins = {c: reg_win_counts[rg][c] for c in cols}
                        top_freq_exp = max(exp_wins, key=exp_wins.get)
                        top_freq_pct = 100.0 * exp_wins[top_freq_exp] / max(reg_step_counts[rg], 1)
                        hbml_freq_pct = 100.0 * reg_win_counts[rg]["HBML"] / max(reg_step_counts[rg], 1)
                        regime_rows.append({
                            "dataset": ds,
                            "regime": rg,
                            "n_patients": len(reg_se_hbml[rg]),
                            "step_share_pct": 100.0 * reg_step_counts[rg] / max(tot_scored, 1),
                            "hbml_rmse": hb_r_mean,
                            "best_expert": best_r_exp,
                            "best_expert_rmse": best_r_val,
                            "top_freq_expert": top_freq_exp,
                            "top_freq_pct": top_freq_pct,
                            "hbml_freq_pct": hbml_freq_pct,
                        })

    # 1. Write weight_diagnostics.csv
    if weight_rows and not only_full:
        w_path = os.path.join(TABLES_DIR, "weight_diagnostics.csv")
        with open(w_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(weight_rows[0].keys()))
            w.writeheader()
            w.writerows(weight_rows)
        print(f"Wrote {w_path} ({len(weight_rows)} rows)")

    # 2. Update fs_comparison/summary.csv with ftl and sfh_sqrt rows
    fs_sum_path = os.path.join(FS_DIR, "summary.csv")
    if os.path.exists(fs_sum_path) and new_fs_rows and not only_full:
        with open(fs_sum_path, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r.get("variant") not in ("ftl", "sfh_sqrt")]
        combined = existing + new_fs_rows
        with open(fs_sum_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
            w.writeheader()
            w.writerows(combined)
        print(f"Updated {fs_sum_path} with {len(new_fs_rows)} ftl/sfh_sqrt rows")

    # 3. Write hypoglycemia_events.csv and table_hypoglycemia_events.tex
    if hypo_rows:
        h_path = os.path.join(TABLES_DIR, "hypoglycemia_events.csv")
        with open(h_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hypo_rows[0].keys()))
            w.writeheader()
            w.writerows(hypo_rows)
        write_hypo_tex(hypo_rows, os.path.join(TABLES_DIR, "table_hypoglycemia_events.tex"))
        print(f"Wrote {h_path} and table_hypoglycemia_events.tex ({len(hypo_rows)} rows)")

    # 4. Write regime_summary_table.csv and table_regime_summary.tex
    if regime_rows:
        r_path = os.path.join(TABLES_DIR, "regime_summary_table.csv")
        with open(r_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(regime_rows[0].keys()))
            w.writeheader()
            w.writerows(regime_rows)
        write_regime_tex(regime_rows, os.path.join(TABLES_DIR, "table_regime_summary.tex"))
        print(f"Wrote {r_path} and table_regime_summary.tex ({len(regime_rows)} rows)")


def write_hypo_tex(rows, out_path):
    ds_disp = {"weinstock": "Weinstock 2016", "cgmacros": "CGMacros"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lllccccccc}",
        r"\toprule",
        r"Dataset & Horizon & Model & Episodes & Episode sens.\ (\%) & Lead time (min) & Point PPV (\%) & Point $F_1$ (\%) & False alerts / day & Parkes A+B (\%) \\",
        r"\midrule",
    ]
    prev_ds = None
    for r in rows:
        ds = ds_disp.get(r["dataset"], r["dataset"])
        if prev_ds is not None and prev_ds != ds:
            lines.append(r"\midrule")
        prev_ds = ds
        m = r["model"]
        if m == "BestExpert":
            m = f"Best expert ({r['best_expert_name']})"
        lines.append(
            f"{ds} & {r['horizon']} hr & {m} & {r['n_episodes']} & "
            f"{r['episode_sensitivity_pct']:.1f} & {r['mean_lead_time_min']:.1f} & "
            f"{r['point_ppv_pct']:.1f} & {r['point_f1_pct']:.1f} & "
            f"{r['false_alerts_per_day']:.2f} & {r['parkes_ab_pct']:.1f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Event-level hypoglycaemia ($<70$~mg/dL) detection and Parkes consensus error grid performance (full-history context).} "
        r"True hypoglycaemic episodes are defined as contiguous runs of $\ge 3$ scored readings ($15$~min) below $70$~mg/dL separated by $\ge 30$~min at or above $70$~mg/dL. "
        r"Episode sensitivity is the percentage of episodes detected during the episode or within the $h$-step look-ahead window preceding onset; lead time is the mean warning time prior to episode end among detected episodes. "
        r"Point PPV (positive predictive value) and $F_1$ score evaluate timestep-level hypoglycaemia classification, false alerts per day count predicted $<70$~mg/dL alert clusters when reference glucose remained $\ge 70$~mg/dL throughout the $\pm h$-step window, and Parkes A+B reports participant-averaged proportions in Zones A and B of the Parkes Type 1 consensus error grid.}",
        r"\label{tab:hypoglycemia_events}",
        r"\end{table*}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_regime_tex(rows, out_path):
    ds_disp = {"weinstock": "Weinstock 2016", "cgmacros": "CGMacros"}
    reg_disp = {
        "STABLE": "Stable Euglycaemia",
        "RISING": "Rising Excursion",
        "FALLING": "Falling Excursion",
        "HIGHVAR": "High Variability",
        "RECOVERY": "Recovery Phase",
        "HYPO_TRANS": "Hypoglycaemic Transition",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Dataset & Operational regime & Timestep share (\%) & HBML RMSE (mg/dL) & Best expert (RMSE, mg/dL) & Top-selected expert (share, \%) \\",
        r"\midrule",
    ]
    prev_ds = None
    for r in rows:
        ds = ds_disp.get(r["dataset"], r["dataset"])
        if prev_ds is not None and prev_ds != ds:
            lines.append(r"\midrule")
        prev_ds = ds
        rg = reg_disp.get(r["regime"], r["regime"])
        lines.append(
            f"{ds} & {rg} & {r['step_share_pct']:.1f} & {r['hbml_rmse']:.2f} & "
            f"{r['best_expert']} ({r['best_expert_rmse']:.2f}) & "
            f"{r['top_freq_expert']} ({r['top_freq_pct']:.1f}\\%) \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Forecasting accuracy and expert selection frequency across operational glycaemic regimes (30-minute horizon, full-history context).} "
        r"Scored timesteps are partitioned into six mutually exclusive operational regimes in strict priority order (\texttt{compute\_regimes.py}). "
        r"Columns report the percentage of scored timesteps in each regime, cohort-mean participant-level RMSE for HBML and for the lowest-RMSE single expert in that regime, and the constituent expert achieving the lowest instantaneous squared error most frequently across timesteps in that regime.}",
        r"\label{tab:regime_summary}",
        r"\end{table*}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
