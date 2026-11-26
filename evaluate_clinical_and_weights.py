#!/usr/bin/env python3
"""evaluate_clinical_and_weights.py
===============================================================================
Computes weight diagnostics, event-level pre-onset hypoglycaemia detection,
Parkes (Type 1 / Type 2) consensus error grid metrics, and regime-stratified
summaries directly from saved simulation trajectories and raw CGM series using
the canonical ``ExpMethods.aggregators``, ``ExpMethods.evaluation``, and
``ExpMethods.globals`` modules.

Outputs (all in ``overleaf/tables/``):
  - ``weight_diagnostics.csv``
  - ``hypoglycemia_events.csv`` & ``table_hypoglycemia_events.tex``
  - ``hypoglycemia_threshold_sweep.csv`` (70-90 mg/dL alert thresholds, every detector)
  - ``hypoglycemia_matched_far.csv`` (sensitivity at matched false-alert rates and minimum
    warning times 0/10/15/20 min, participant-clustered bootstrap 95% intervals)
  - ``regime_summary_table.csv`` & ``table_regime_summary.tex``
===============================================================================
"""

import csv
import glob
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ExpMethods import aggregators as agg  # noqa: E402
from ExpMethods import evaluation as ev  # noqa: E402
from ExpMethods.globals import GlobalValues as G  # noqa: E402
from fast_engine.simulation import ENGINE_VERSION  # noqa: E402
# Set HBML_EVAL_ENGINE=2.2 to run these analyses on the cached v2.2 forecasts before the v2.3 re-run.
ENGINE_VERSION = os.environ.get("HBML_EVAL_ENGINE", ENGINE_VERSION)

TABLES_DIR = os.path.join(BASE_DIR, "overleaf", "tables")
OUTPUTS_DIR = os.path.join(BASE_DIR, "Outputs")

HORIZONS = [("half", "0.5", 6), ("2", "2", 24), ("5", "5", 60)]
CONTEXTS = ["6", "12", "24", "full"]
ALL_EXPERTS = list(G.experts)
PRIMARY_HBML = G.primary_hbml
START = int(G.burn_in_steps)
MIN_SCORED = int(getattr(G, "min_scored_points", 288))
T2D_IDS = set(G.CGMacros.get("diabetic_ids", []))

# Piecewise linear zone boundaries from Pfutzner et al. (2013), J Diabetes Sci Technol 7(5):1275-1281, Table 1
_PARKES_BOUNDS = {
    1: {
        "B_lower": [(50, 0), (50, 30), (170, 145), (385, 300), (550, 450)],
        "B_upper": [(0, 50), (30, 50), (140, 170), (280, 380), (430, 550)],
        "C_lower": [(120, 0), (120, 30), (260, 130), (550, 250)],
        "C_upper": [(0, 60), (30, 60), (50, 80), (70, 110), (260, 550)],
        "D_lower": [(250, 0), (250, 40), (410, 110), (550, 160)],
        "D_upper": [(0, 100), (25, 100), (50, 125), (80, 215), (125, 550)],
        "E_upper": [(0, 150), (35, 155), (50, 550)],
    },
    2: {
        "B_lower": [(50, 0), (50, 30), (90, 80), (330, 230), (550, 450)],
        "B_upper": [(0, 50), (30, 50), (230, 330), (440, 550)],
        "C_lower": [(90, 0), (260, 130), (550, 250)],
        "C_upper": [(0, 60), (30, 60), (280, 550)],
        "D_lower": [(250, 0), (250, 40), (410, 110), (550, 160)],
        "D_upper": [(0, 80), (25, 80), (35, 90), (125, 550)],
        "E_upper": [(0, 200), (35, 200), (50, 550)],
    },
}


def _interp_y(pts, r, default_y):
    """Evaluate y(r) along a piecewise linear polyline; return default_y outside its x-domain."""
    if r < pts[0][0]:
        return default_y
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x1 == x0:
            continue
        if x0 <= r <= x1:
            return y0 + (r - x0) * (y1 - y0) / float(x1 - x0)
    return float(pts[-1][1])


def parkes_error_grid_zone(r, p, diabetes_type=1):
    """Classify (reference r, predicted p) into ('A', 'B', 'C', 'D', 'E') using Pfutzner et al. (2013)."""
    if not (math.isfinite(r) and math.isfinite(p)):
        return "unknown"
    b = _PARKES_BOUNDS[2 if diabetes_type == 2 else 1]

    # Upper side (p >= r)
    if p >= r:
        if r <= b["B_upper"][-1][0]:
            b_u = _interp_y(b["B_upper"], r, 50.0)
            if p <= b_u:
                return "A"
        else:
            return "A"
        if r <= b["C_upper"][-1][0] and p <= _interp_y(b["C_upper"], r, b["C_upper"][0][1]):
            return "B"
        if r > b["C_upper"][-1][0]:
            return "B"
        if r <= b["D_upper"][-1][0] and p <= _interp_y(b["D_upper"], r, b["D_upper"][0][1]):
            return "C"
        if r > b["D_upper"][-1][0]:
            return "C"
        if r <= b["E_upper"][-1][0] and p <= _interp_y(b["E_upper"], r, b["E_upper"][0][1]):
            return "D"
        if r > b["E_upper"][-1][0]:
            return "D"
        return "E"

    # Lower side (p < r)
    b_l = _interp_y(b["B_lower"], r, 0.0) if r >= 50.0 else 0.0
    if p >= b_l:
        return "A"
    c_x0 = b["C_lower"][0][0]
    if r < c_x0 or p >= _interp_y(b["C_lower"], r, 0.0):
        return "B"
    d_x0 = b["D_lower"][0][0]
    if r < d_x0 or p >= _interp_y(b["D_lower"], r, 0.0):
        return "C"
    return "D"


def load_retained_ids():
    """Return {(dataset, horizon_disp, context): set_of_ids} from rmse_participant.csv."""
    p = os.path.join(TABLES_DIR, "rmse_participant.csv")
    out = defaultdict(set)
    if not os.path.exists(p):
        return out
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            ds = r["dataset"].strip().lower()
            hz = "0.5" if str(r["horizon"]).strip() in ("half", "0.5") else str(r["horizon"]).strip().replace(".0", "")
            ctx = str(r["context"]).strip()
            out[(ds, hz, ctx)].add(str(r["id"]).strip().zfill(3))
    return out


def load_reference_means_and_counts(retained_ids):
    """Return {(dataset, horizon_disp, context): (mean_hbml, n)} from rmse_pivot_mean.csv."""
    refs = {}
    piv_path = os.path.join(TABLES_DIR, "rmse_pivot_mean.csv")
    if not os.path.exists(piv_path):
        return refs
    with open(piv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("model") != "HBML":
                continue
            ds = r["dataset"].strip().lower()
            hz = "0.5" if str(r["horizon"]).strip() in ("half", "0.5") else str(r["horizon"]).strip().replace(".0", "")
            for c in CONTEXTS:
                s = (r.get(f"mean.{c}") or "").replace("\\textbf{", "").replace("}", "").strip()
                try:
                    n_expected = len(retained_ids.get((ds, hz, c), ())) or None
                    refs[(ds, hz, c)] = (float(s), n_expected)
                except ValueError:
                    pass
    return refs


def load_forecasts_matrix(fc_path):
    """Read raw expert forecast matrix matching GlobalValues.experts order."""
    with open(fc_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        cols = []
        src_col = {}
        for c in ALL_EXPERTS:
            if c in fieldnames:
                cols.append(c)
                src_col[c] = c
            elif c == "ETS" and "HoltWinters" in fieldnames:
                cols.append(c)
                src_col[c] = "HoltWinters"
        rows = []
        for r in reader:
            rows.append([float(r[src_col[c]]) if r[src_col[c]] != "" else float("nan") for c in cols])
    return cols, np.asarray(rows, dtype=float)


def run_participant(dataset, pid, fc_path, h_steps, root=None):
    """Evaluate primary HBML and comparator aggregators using ExpMethods.aggregators and scored_mask."""
    meta = ev.read_meta(fc_path)
    if not meta or meta.get("settings", {}).get("engine_version") != ENGINE_VERSION:
        return None
    y = ev.load_targets(dataset, pid, root=(BASE_DIR if root is None else root))
    if y is None:
        return None
    cols, f_mat = load_forecasts_matrix(fc_path)
    if len(cols) < 2:
        return None
    n = min(len(y), len(f_mat))
    if n <= int(G.eval_start) + int(G.eval_tail) + 10:
        return None
    y = y[:n]
    f_mat = f_mat[:n]

    mask = ev.scored_mask(f_mat, y)
    n_scored = int(mask.sum())
    if n_scored < MIN_SCORED:
        return None

    l_mat = agg.online_losses(f_mat, y, START, h_steps)
    pred_prim, loss_prim, W = agg.run_named(PRIMARY_HBML, f_mat, l_mat, y, start=START, horizon=h_steps, return_weights=True)
    _, loss_ftl = agg.run_named("FTL", f_mat, l_mat, y, start=START, horizon=h_steps)
    alt_name = "HBML-SFHDF" if PRIMARY_HBML != "HBML-SFHDF" else "HBML-SFH-Sqrt"
    _, loss_alt = agg.run_named(alt_name, f_mat, l_mat, y, start=START, horizon=h_steps)

    # Weight distribution applied to scored targets t (weights[g + 1] forecasts target t = g + h)
    target_indices = np.flatnonzero(mask)
    origin_indices = np.clip(target_indices - h_steps + 1, 0, len(W) - 1)
    W_eval = W[origin_indices]
    mean_max_w = float(np.mean(np.max(W_eval, axis=1)))
    mean_w = {cols[k]: float(np.mean(W_eval[:, k])) for k in range(len(cols))}
    dom_idx = int(np.argmax(W[-1]))

    exp_rmses = {cols[k]: ev.rmse(l_mat[:, k], mask) for k in range(len(cols))}
    return {
        "pid": pid,
        "y": y,
        "F": f_mat,
        "cols": cols,
        "scored_mask": mask,
        "n_scored": n_scored,
        "pred_primary": pred_prim,
        "hbml_rmse": ev.rmse(loss_prim, mask),
        "ftl_rmse": ev.rmse(loss_ftl, mask),
        "alt_rmse": ev.rmse(loss_alt, mask),
        "exp_rmses": exp_rmses,
        "mean_max_weight": mean_max_w,
        "mean_weights": mean_w,
        "dom_expert": cols[dom_idx],
    }


WARNING_MINUTES = (0, 10, 15, 20)


def find_hypo_episodes(y, true_thresh=70.0, min_ep_slots=3, min_gap_slots=6, join_slots=2,
                       onset_lookback_slots=6):
    """True hypoglycaemic episodes on the full observed trajectory (every real reading, scored or not).

    Readings < true_thresh are joined into runs across at most `join_slots` - 1 missing slots; runs
    closer than `min_gap_slots` (30 min) are merged; runs with >= `min_ep_slots` readings below the
    threshold (15 min) are episodes. Returns (episodes, n_left_censored): an episode is left-censored,
    and excluded, when no real reading >= true_thresh lies in the `onset_lookback_slots` (30 min)
    before its first low reading, because its onset time is then unknown (recording start, or onset
    inside a gap).
    """
    y = np.asarray(y, dtype=float)
    obs = np.flatnonzero(~np.isnan(y))
    low = obs[y[obs] < true_thresh]
    runs = []
    for t in low:
        t = int(t)
        if runs and t - runs[-1][1] <= join_slots:
            runs[-1][1] = t
            runs[-1][2] += 1
        else:
            runs.append([t, t, 1])
    merged = []
    for st, en, k in runs:
        if merged and st - merged[-1][1] < min_gap_slots:
            merged[-1][1] = en
            merged[-1][2] += k
        else:
            merged.append([st, en, k])
    episodes, n_cens = [], 0
    for st, en, k in merged:
        if k < min_ep_slots:
            continue
        pre = y[max(0, st - onset_lookback_slots):st]
        if not np.any(pre[~np.isnan(pre)] >= true_thresh):
            n_cens += 1
            continue
        episodes.append((st, en))
    return episodes, n_cens


def evaluate_hypo_trajectory(y, pred, scored_mask, h, diabetes_type=1,
                             thresh=70.0, true_thresh=70.0, min_ep_slots=3, min_gap_slots=6,
                             n_chance_shifts=200, compute_parkes=True, warning_minutes=WARNING_MINUTES):
    """Pre-onset episode hypoglycaemia detection, warning time, chance baseline, point metrics, Parkes.

    Episodes come from the full observed trajectory (find_hypo_episodes; left-censored episodes
    excluded). An episode is *evaluable* when at least one forecast targeting it was issued before
    onset (a scored target t in [st, en] with origin t - h < st); only evaluable episodes enter the
    denominator. Alerts are scored forecasts below `thresh`. One-to-one matching: an episode is
    detected at most once, by its earliest pre-onset alert; with minimum warning time W minutes it is
    detected only if that alert's origin t - h is <= st - W/5 slots (W = 0: strictly before onset).
    False alerts: alert clusters (alerts within `min_gap_slots` = 30 min merged, i.e. a 30-min
    refractory period) with no real reading < true_thresh within +/- 2 slots. Rates use scored days
    (scored targets / 288).

    Chance baseline: the participant's alert sequence over scored targets is circularly shifted by
    up to `n_chance_shifts` evenly spaced lags in [h + 12, n - h - 12] scored positions (wrapping
    around the end of the record), and the mean pre-onset hit rate over lags is the chance sensitivity.
    """
    idx = np.flatnonzero(scored_mask)
    if len(idx) == 0:
        return None
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n_scored = len(idx)
    days = n_scored / 288.0

    tp = fp = fn = tn = 0
    parkes_a = parkes_ab = 0
    for t in idx:
        yt, pt = float(y[t]), float(pred[t])
        if compute_parkes:
            z = parkes_error_grid_zone(yt, pt, diabetes_type=diabetes_type)
            if z == "A":
                parkes_a += 1
                parkes_ab += 1
            elif z == "B":
                parkes_ab += 1
        is_hypo_true = yt < true_thresh
        is_hypo_pred = pt < thresh
        if is_hypo_true and is_hypo_pred:
            tp += 1
        elif not is_hypo_true and is_hypo_pred:
            fp += 1
        elif is_hypo_true and not is_hypo_pred:
            fn += 1
        else:
            tn += 1

    all_eps, n_cens = find_hypo_episodes(y, true_thresh, min_ep_slots, min_gap_slots)
    idx_pos = {int(slot): pos for pos, slot in enumerate(idx)}
    episodes, ep_pre_positions = [], []
    for st, en in all_eps:
        pre_pos = [idx_pos[t] for t in range(st, min(en, st + h - 1) + 1) if t in idx_pos and t - h < st]
        if pre_pos:
            episodes.append((st, en))
            ep_pre_positions.append(pre_pos)
    n_not_evaluable = len(all_eps) - len(episodes)

    detected_pre = detected_any = 0
    detected_w = {w: 0 for w in warning_minutes}
    lead_times_det_min, lead_times_all_min = [], []
    for st, en in episodes:
        first_pre = None
        any_hit = False
        for t in range(st, en + 1):
            if scored_mask[t] and pred[t] < thresh:
                any_hit = True
                if (t - h) < st and first_pre is None:
                    first_pre = t
        if any_hit:
            detected_any += 1
        # earliest alert issued before onset (largest warning) among targets in the episode
        pre_alert_origins = [t - h for t in range(st, en + 1)
                             if scored_mask[t] and (t - h) < st and pred[t] < thresh]
        if first_pre is not None:
            detected_pre += 1
            lt = float(st - min(pre_alert_origins)) * 5.0
            lead_times_det_min.append(lt)
            lead_times_all_min.append(lt)
            for w in warning_minutes:
                if lt >= max(w, 5.0):
                    detected_w[w] += 1
        else:
            lead_times_all_min.append(0.0)

    chance_detected_sum = 0.0
    if episodes and n_chance_shifts > 0 and n_scored > 1:
        alert_vec = (pred[idx] < thresh)
        min_sep = h + 12
        lo = min_sep if n_scored > 2 * min_sep + 5 else 1
        hi = n_scored - min_sep if n_scored > 2 * min_sep + 5 else n_scored - 1
        if hi >= lo:
            shifts = np.unique(np.linspace(lo, hi, min(n_chance_shifts, hi - lo + 1), dtype=int))
        else:
            shifts = np.array([1], dtype=int)
        for pre_pos in ep_pre_positions:
            p_arr = np.asarray(pre_pos, dtype=int)[:, None]
            hits = alert_vec[(p_arr - shifts[None, :]) % n_scored].any(axis=0)
            chance_detected_sum += float(np.mean(hits))

    pred_runs = []
    p_start = p_end = None
    for t in idx:
        if pred[t] < thresh:
            if p_start is None:
                p_start = p_end = int(t)
            elif int(t) - p_end <= min_gap_slots:
                p_end = int(t)
            else:
                pred_runs.append((p_start, p_end))
                p_start = p_end = int(t)
    if p_start is not None:
        pred_runs.append((p_start, p_end))
    false_clusters = 0
    true_clusters = 0
    n_tot = len(y)
    for st, en in pred_runs:
        win = y[max(0, st - 2):min(n_tot, en + 3)]
        if np.any(win[~np.isnan(win)] < true_thresh):
            true_clusters += 1
        else:
            false_clusters += 1

    out = {
        "n_scored": n_scored,
        "days": days,
        "days_calendar": len(y) / 288.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_episodes": len(episodes),
        "n_episodes_left_censored": n_cens,
        "n_episodes_not_evaluable": n_not_evaluable,
        "n_detected": detected_pre,
        "n_detected_any": detected_any,
        "chance_detected_sum": chance_detected_sum,
        "lead_times_det": lead_times_det_min,
        "lead_times_all": lead_times_all_min,
        "n_alert_clusters": len(pred_runs),
        "true_clusters": true_clusters,
        "false_clusters": false_clusters,
        "parkes_a_pct": (100.0 * parkes_a / n_scored) if compute_parkes else float("nan"),
        "parkes_ab_pct": (100.0 * parkes_ab / n_scored) if compute_parkes else float("nan"),
    }
    for w in warning_minutes:
        out[f"n_detected_w{w}"] = detected_w[w]
    return out


def matched_far_comparison(pp_rows, far_targets=(0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0),
                           warning_minutes=WARNING_MINUTES, n_boot=2000, seed=7,
                           reference="HBML", comparators=("Persistence", "BestExpert")):
    """Sensitivity at matched false-alert rates with participant-clustered bootstrap intervals.

    pp_rows: one dict per (model, threshold, participant) with n_episodes, n_detected_w{W},
    false_clusters and days. For each model the pooled curve over thresholds (false alerts per
    scored day, pre-onset sensitivity) is linearly interpolated at each target rate; targets outside
    a model's swept range are NaN (no extrapolation). Participants are resampled with replacement
    (the same resample for every model), giving 95% percentile intervals for each sensitivity and
    for the paired difference reference - comparator.
    """
    models = sorted({r["model"] for r in pp_rows})
    thrs = sorted({float(r["threshold_mgdl"]) for r in pp_rows})
    pids = sorted({r["pid"] for r in pp_rows})
    P = {p: i for i, p in enumerate(pids)}
    keys = ["n_episodes", "false_clusters", "days"] + [f"n_detected_w{w}" for w in warning_minutes]
    A = {m: {k: np.zeros((len(thrs), len(pids))) for k in keys} for m in models}
    for r in pp_rows:
        ti, pi = thrs.index(float(r["threshold_mgdl"])), P[r["pid"]]
        for k in keys:
            A[r["model"]][k][ti, pi] = float(r[k])

    def curve(m, w, wts):
        ep = A[m]["n_episodes"] @ wts
        det = A[m][f"n_detected_w{w}"] @ wts
        far = (A[m]["false_clusters"] @ wts) / np.maximum(A[m]["days"] @ wts, 1e-9)
        sens = np.where(ep > 0, 100.0 * det / np.maximum(ep, 1e-9), np.nan)
        o = np.argsort(far, kind="stable")
        far, sens = far[o], sens[o]
        tgt = np.asarray(far_targets, dtype=float)
        v = np.interp(tgt, far, sens)
        v[(tgt < far[0]) | (tgt > far[-1])] = np.nan
        return v

    rng = np.random.default_rng(seed)
    ones = np.ones(len(pids))
    boots = [np.bincount(rng.integers(0, len(pids), len(pids)), minlength=len(pids)).astype(float)
             for _ in range(n_boot)]
    rows = []
    n_with_ep = int((A[models[0]]["n_episodes"].max(axis=0) > 0).sum()) if models else 0
    for w in warning_minutes:
        est = {m: curve(m, w, ones) for m in models}
        bs = {m: np.array([curve(m, w, b) for b in boots]) for m in models}
        for j, f in enumerate(far_targets):
            for m in models:
                lo, hi = np.nanpercentile(bs[m][:, j], [2.5, 97.5]) if np.isfinite(est[m][j]) else (np.nan, np.nan)
                row = dict(min_warning_min=w, false_alerts_per_day=f, model=m, sensitivity_pct=est[m][j],
                           ci95_low=lo, ci95_high=hi, n_participants=len(pids),
                           n_participants_with_episodes=n_with_ep)
                if m != reference and reference in est:
                    d = est[reference][j] - est[m][j]
                    db = bs[reference][:, j] - bs[m][:, j]
                    ok = np.isfinite(db)
                    row.update(diff_vs_reference_pp=d,
                               diff_ci95_low=np.percentile(db[ok], 2.5) if ok.sum() > 10 and np.isfinite(d) else np.nan,
                               diff_ci95_high=np.percentile(db[ok], 97.5) if ok.sum() > 10 and np.isfinite(d) else np.nan)
                rows.append(row)
    return rows


_REGIME_CACHE = {}


def classify_regimes_stdlib(dataset, pid, y):
    """Classify timesteps into the 7 operational regimes matching compute_regimes.py."""
    key = (dataset, pid, len(y))
    if key in _REGIME_CACHE:
        return _REGIME_CACHE[key]
    n = len(y)
    regimes = ["STABLE"] * n
    for t in range(n):
        gt = float(y[t])
        if not math.isfinite(gt):
            regimes[t] = "UNKNOWN"
            continue
        w_s = [(i, float(y[t - 5 + i])) for i in range(6) if 0 <= t - 5 + i < n and math.isfinite(y[t - 5 + i])]
        if len(w_s) >= 3:
            xm = sum(p[0] for p in w_s) / len(w_s)
            ym = sum(p[1] for p in w_s) / len(w_s)
            den = sum((p[0] - xm) ** 2 for p in w_s)
            slope = sum((p[0] - xm) * (p[1] - ym) for p in w_s) / den if den > 0 else 0.0
        else:
            slope = float("nan")
        w_v = [float(y[tau]) for tau in range(max(0, t - 11), t + 1) if math.isfinite(y[tau])]
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
        if reg == "STABLE" and not (70.0 <= gt <= 180.0):
            reg = "STABLE_OOR"   # stable but outside 70-180 mg/dL: not euglycaemia
        regimes[t] = reg
    _REGIME_CACHE[key] = regimes
    return regimes


SWEEP_THRESHOLDS = [70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 85.0, 88.0, 90.0]


def _plot_threshold_sweep(sweep_rows, out_pdf):
    """Plot pre-onset sensitivity and above-chance gain vs false alerts/day across thresholds 70--90 mg/dL."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping sweep plot ({exc})")
        return
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    styles = {
        "HBML": dict(color="#d62728", marker="o", lw=2.2, label="HBML (SFH-Sqrt)"),
        "BestExpert": dict(color="#2ca02c", marker="s", lw=2.0, ls="--", label="Best single expert (ARIMA)"),
        "Persistence": dict(color="#7f7f7f", marker="^", lw=1.8, ls=":", label="Persistence"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for model_key, st in styles.items():
        sub = [r for r in sweep_rows if r["model"] == model_key]
        if not sub:
            continue
        sub.sort(key=lambda r: float(r["threshold_mgdl"]))
        x = [float(r["false_alerts_per_day"]) for r in sub]
        y_sens = [float(r["episode_sensitivity_pct"]) for r in sub]
        y_above = [float(r["above_chance_pp"]) for r in sub]
        axes[0].plot(x, y_sens, color=st["color"], marker=st["marker"], lw=st["lw"], ls=st.get("ls", "-"),
                     markersize=5, label=st["label"])
        axes[1].plot(x, y_above, color=st["color"], marker=st["marker"], lw=st["lw"], ls=st.get("ls", "-"),
                     markersize=5, label=st["label"])
        # Annotate nominal 70 mg/dL and 90 mg/dL endpoints
        axes[0].annotate(f"{int(sub[0]['threshold_mgdl'])}", (x[0], y_sens[0]),
                         textcoords="offset points", xytext=(4, -10), fontsize=8, color=st["color"])
        axes[0].annotate(f"{int(sub[-1]['threshold_mgdl'])}", (x[-1], y_sens[-1]),
                         textcoords="offset points", xytext=(4, 2), fontsize=8, color=st["color"])
    axes[0].set_xlabel("False alerts per participant-day")
    axes[0].set_ylabel("Pre-onset episode sensitivity (%)")
    axes[0].set_title("(a) Sensitivity vs. false-alert operating curve")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].axhline(0.0, color="black", lw=0.8, ls="--")
    axes[1].set_xlabel("False alerts per participant-day")
    axes[1].set_ylabel("Sensitivity above time-shifted chance (pp)")
    axes[1].set_title("(b) Pre-onset gain over chance baseline")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    hypo_only = "--hypo-only" in sys.argv
    only_full = ("--full-only" in sys.argv) or hypo_only
    no_plots = "--no-plots" in sys.argv
    retained_ids = load_retained_ids()
    refs = load_reference_means_and_counts(retained_ids)

    weight_rows = []
    hypo_rows = []
    sweep_rows = []
    sweep_pp_rows = []
    regime_rows = []

    for ds in ("weinstock", "cgmacros"):
        for h_str, h_disp, h_steps in HORIZONS:
            for ctx in (["full"] if only_full else CONTEXTS):
                fc_dir = os.path.join(OUTPUTS_DIR, ds, f"h-{h_str}hr", f"context-{ctx}hr", "forecasts")
                keep = retained_ids.get((ds, h_disp, ctx))
                files = sorted(f for f in glob.glob(os.path.join(fc_dir, "*_forecasts.csv"))
                               if os.path.basename(f)[3:] == "_forecasts.csv"
                               and (not keep or os.path.basename(f)[:3] in keep))
                if not files:
                    continue

                p_res = []
                for f_path in files:
                    pid = os.path.basename(f_path)[:3]
                    res = run_participant(ds, pid, f_path, h_steps)
                    if res:
                        p_res.append(res)
                if not p_res:
                    continue

                n_p = len(p_res)
                hb_mean = sum(r["hbml_rmse"] for r in p_res) / n_p
                ref_entry = refs.get((ds, h_disp, ctx))
                if ref_entry is not None:
                    ref_m, ref_n = ref_entry
                    matches = (abs(hb_mean - ref_m) <= 0.01) and (ref_n is None or n_p == ref_n)
                else:
                    matches = True
                print(f"Processed {ds} h={h_str} c={ctx}: n={n_p}, {PRIMARY_HBML}={hb_mean:.2f}, matches={matches}", flush=True)

                cols = p_res[0]["cols"]
                exp_means = {c: sum(r["exp_rmses"][c] for r in p_res) / n_p for c in cols}
                best_exp = min(exp_means, key=exp_means.get)

                ftl_mean = sum(r["ftl_rmse"] for r in p_res) / n_p
                alt_mean = sum(r["alt_rmse"] for r in p_res) / n_p
                maxw_mean = sum(r["mean_max_weight"] for r in p_res) / n_p
                dom_counts = Counter(r["dom_expert"] for r in p_res)
                dom_arima_ets = dom_counts.get("ARIMA", 0) + dom_counts.get("ETS", 0)

                w_row = {
                    "dataset": ds,
                    "horizon": h_disp,
                    "context": ctx,
                    "n": n_p,
                    "matches_tables": "yes" if matches else "no",
                    "primary_aggregator": PRIMARY_HBML,
                    "hbml_rmse": hb_mean,
                    "ftl_rmse": ftl_mean,
                    "sfhdf_rmse": alt_mean if PRIMARY_HBML != "HBML-SFHDF" else hb_mean,
                    "sfh_sqrt_rmse": hb_mean if PRIMARY_HBML == "HBML-SFH-Sqrt" else alt_mean,
                    "mean_max_weight": maxw_mean,
                    "dom_count_ARIMA_ETS": dom_arima_ets,
                    "dom_pct_ARIMA_ETS": 100.0 * dom_arima_ets / n_p,
                }
                for c in ALL_EXPERTS:
                    w_row[f"mean_w_{c}"] = (sum(r["mean_weights"].get(c, 0.0) for r in p_res) / n_p) if c in cols else 0.0
                    w_row[f"dom_pct_{c}"] = 100.0 * dom_counts.get(c, 0) / n_p
                weight_rows.append(w_row)

                if ctx == "full" and matches:
                    best_k = cols.index(best_exp)
                    model_specs = (
                        ("HBML", lambda r: r["pred_primary"]),
                        (f"BestExpert ({best_exp})", lambda r, bk=best_k: r["F"][:, bk]),
                        ("Persistence", lambda r, hs=h_steps: np.r_[np.zeros(hs), r["y"][:-hs]]),
                    )
                    for m_label, get_pred in model_specs:
                        ep_tot = det_tot = det_any_tot = tp_tot = fp_tot = fn_tot = 0
                        chance_tot = 0.0
                        days_tot = 0.0
                        days_cal_tot = 0.0
                        fc_tot = 0
                        tc_tot = 0
                        leads_det = []
                        leads_all = []
                        pa_list = []
                        pab_list = []
                        cens_tot = noeval_tot = n_p_ep = 0
                        det_w_tot = Counter()
                        for r in p_res:
                            dtype = 2 if (ds == "cgmacros" and r["pid"] in T2D_IDS) else 1
                            ev_r = evaluate_hypo_trajectory(r["y"], get_pred(r), r["scored_mask"], h_steps, diabetes_type=dtype)
                            if not ev_r:
                                continue
                            cens_tot += ev_r["n_episodes_left_censored"]
                            noeval_tot += ev_r["n_episodes_not_evaluable"]
                            n_p_ep += int(ev_r["n_episodes"] > 0)
                            for w in WARNING_MINUTES:
                                det_w_tot[w] += ev_r[f"n_detected_w{w}"]
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
                        chance_sens = 100.0 * chance_tot / ep_tot if ep_tot > 0 else float("nan")
                        above_chance = (ep_sens - chance_sens) if (ep_tot > 0) else float("nan")
                        ep_sens_any = 100.0 * det_any_tot / ep_tot if ep_tot > 0 else float("nan")
                        pt_sens = 100.0 * tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else float("nan")
                        pt_ppv = 100.0 * tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else float("nan")
                        pt_f1 = (2 * pt_sens * pt_ppv / (pt_sens + pt_ppv)) if (pt_sens + pt_ppv) > 0 else float("nan")
                        far_day = fc_tot / max(days_tot, 1e-6)
                        far_cal_day = fc_tot / max(days_cal_tot, 1e-6)
                        cl_tot = tc_tot + fc_tot
                        cluster_ppv = 100.0 * tc_tot / cl_tot if cl_tot > 0 else float("nan")
                        lead_all_mean = sum(leads_all) / len(leads_all) if leads_all else 0.0
                        lead_det_mean = sum(leads_det) / len(leads_det) if leads_det else 0.0
                        hypo_rows.append({
                            "dataset": ds,
                            "horizon": h_disp,
                            "context": ctx,
                            "model": m_label.split(" ")[0],
                            "best_expert_name": best_exp,
                            "n_participants": n_p,
                            "n_participants_with_episodes": n_p_ep,
                            "n_episodes": ep_tot,
                            "n_episodes_left_censored": cens_tot,
                            "n_episodes_not_evaluable": noeval_tot,
                            "n_detected": det_tot,
                            "episode_sensitivity_pct": ep_sens,
                            **{f"sensitivity_warn{w}min_pct": (100.0 * det_w_tot[w] / ep_tot if ep_tot else float("nan"))
                               for w in WARNING_MINUTES},
                            "episode_sensitivity_any_pct": ep_sens_any,
                            "chance_sensitivity_pct": chance_sens,
                            "above_chance_pp": above_chance,
                            "mean_lead_time_min": lead_all_mean,
                            "mean_lead_time_detected_min": lead_det_mean,
                            "point_sensitivity_pct": pt_sens,
                            "point_ppv_pct": pt_ppv,
                            "point_f1_pct": pt_f1,
                            "n_alert_clusters": cl_tot,
                            "true_alert_clusters": tc_tot,
                            "cluster_ppv_pct": cluster_ppv,
                            "false_alerts_per_day": far_day,
                            "false_alerts_per_calendar_day": far_cal_day,
                            "parkes_a_pct": sum(pa_list) / len(pa_list),
                            "parkes_ab_pct": sum(pab_list) / len(pab_list),
                        })

                    if ds == "weinstock" and h_str == "half":
                        for thr in SWEEP_THRESHOLDS:
                            for m_label, get_pred in model_specs:
                                ep_tot = det_tot = tp_tot = fp_tot = fn_tot = 0
                                chance_tot = days_tot = 0.0
                                fc_tot = 0
                                det_w_tot = Counter()
                                for r in p_res:
                                    ev_r = evaluate_hypo_trajectory(
                                        r["y"], get_pred(r), r["scored_mask"], h_steps,
                                        thresh=thr, true_thresh=70.0, compute_parkes=False,
                                    )
                                    if not ev_r:
                                        continue
                                    sweep_pp_rows.append(dict(
                                        model=m_label.split(" ")[0], threshold_mgdl=thr, pid=r["pid"],
                                        n_episodes=ev_r["n_episodes"], false_clusters=ev_r["false_clusters"],
                                        days=ev_r["days"],
                                        **{f"n_detected_w{w}": ev_r[f"n_detected_w{w}"] for w in WARNING_MINUTES}))
                                    for w in WARNING_MINUTES:
                                        det_w_tot[w] += ev_r[f"n_detected_w{w}"]
                                    ep_tot += ev_r["n_episodes"]
                                    det_tot += ev_r["n_detected"]
                                    chance_tot += ev_r["chance_detected_sum"]
                                    tp_tot += ev_r["tp"]
                                    fp_tot += ev_r["fp"]
                                    fn_tot += ev_r["fn"]
                                    days_tot += ev_r["days"]
                                    fc_tot += ev_r["false_clusters"]
                                ep_sens = 100.0 * det_tot / ep_tot if ep_tot > 0 else float("nan")
                                ch_sens = 100.0 * chance_tot / ep_tot if ep_tot > 0 else float("nan")
                                pt_sens = 100.0 * tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else float("nan")
                                pt_ppv = 100.0 * tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else float("nan")
                                pt_f1 = (2 * pt_sens * pt_ppv / (pt_sens + pt_ppv)) if (pt_sens + pt_ppv) > 0 else float("nan")
                                sweep_rows.append({
                                    "dataset": ds,
                                    "horizon": h_disp,
                                    "context": ctx,
                                    "threshold_mgdl": thr,
                                    "model": m_label.split(" ")[0],
                                    "best_expert_name": best_exp,
                                    "n_participants": n_p,
                                    "n_episodes": ep_tot,
                                    "n_detected": det_tot,
                                    "episode_sensitivity_pct": ep_sens,
                                    **{f"sensitivity_warn{w}min_pct": (100.0 * det_w_tot[w] / ep_tot if ep_tot else float("nan"))
                                       for w in WARNING_MINUTES},
                                    "chance_sensitivity_pct": ch_sens,
                                    "above_chance_pp": ep_sens - ch_sens,
                                    "false_alerts_per_day": fc_tot / max(days_tot, 1e-6),
                                    "point_ppv_pct": pt_ppv,
                                    "point_f1_pct": pt_f1,
                                })

                if ctx == "full" and h_str == "half" and matches and not hypo_only:
                    reg_order = ["STABLE", "STABLE_OOR", "RISING", "FALLING", "HIGHVAR", "RECOVERY", "HYPO_TRANS"]
                    reg_se_hbml = defaultdict(list)
                    reg_se_exp = {c: defaultdict(list) for c in cols}
                    reg_win_counts = {reg: Counter() for reg in reg_order}
                    reg_step_counts = Counter()
                    tot_scored = 0

                    for r in p_res:
                        y_r = r["y"]
                        regs = classify_regimes_stdlib(ds, r["pid"], y_r)
                        mask_r = r["scored_mask"]
                        p_hb = r["pred_primary"]
                        F_r = r["F"]
                        p_se_hb = defaultdict(list)
                        p_se_ex = {c: defaultdict(list) for c in cols}
                        for t_i in np.flatnonzero(mask_r):
                            rg = regs[t_i]
                            if rg not in reg_win_counts:
                                continue
                            tot_scored += 1
                            reg_step_counts[rg] += 1
                            yt = float(y_r[t_i])
                            ehb = (float(p_hb[t_i]) - yt) ** 2
                            p_se_hb[rg].append(ehb)
                            ex_errs = {}
                            for k_i, c in enumerate(cols):
                                e_k = (float(F_r[t_i, k_i]) - yt) ** 2
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
                        exp_wins = {c: reg_win_counts[rg][c] for c in cols}
                        top_freq_exp = max(exp_wins, key=exp_wins.get)
                        top_freq_pct = 100.0 * exp_wins[top_freq_exp] / max(reg_step_counts[rg], 1)
                        hbml_freq_pct = 100.0 * reg_win_counts[rg]["HBML"] / max(reg_step_counts[rg], 1)
                        regime_rows.append({
                            "dataset": ds,
                            "regime": rg,
                            "n_patients": len(reg_se_hbml[rg]),
                            "n_cohort": n_p,
                            "step_share_pct": 100.0 * reg_step_counts[rg] / max(tot_scored, 1),
                            "hbml_rmse": hb_r_mean,
                            "best_expert": best_r_exp,
                            "best_expert_rmse": best_r_val,
                            "top_freq_expert": top_freq_exp,
                            "top_freq_pct": top_freq_pct,
                            "hbml_freq_pct": hbml_freq_pct,
                        })

    if weight_rows and not only_full:
        w_path = os.path.join(TABLES_DIR, "weight_diagnostics.csv")
        with open(w_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(weight_rows[0].keys()))
            w.writeheader()
            w.writerows(weight_rows)
        print(f"Wrote {w_path} ({len(weight_rows)} rows)")

    if hypo_rows:
        h_path = os.path.join(TABLES_DIR, "hypoglycemia_events.csv")
        with open(h_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hypo_rows[0].keys()))
            w.writeheader()
            w.writerows(hypo_rows)
        main_rows = [r for r in hypo_rows if str(r["horizon"]) in ("0.5", "half")]
        supp_rows = [r for r in hypo_rows if str(r["horizon"]) not in ("0.5", "half")]
        write_hypo_tex(main_rows, os.path.join(TABLES_DIR, "table_hypoglycemia_events.tex"), supplement=False)
        write_hypo_tex(supp_rows, os.path.join(TABLES_DIR, "table_hypoglycemia_events_supp.tex"), supplement=True)
        print(f"Wrote {h_path}, table_hypoglycemia_events.tex ({len(main_rows)}), and table_hypoglycemia_events_supp.tex ({len(supp_rows)})")

    if sweep_rows:
        s_path = os.path.join(TABLES_DIR, "hypoglycemia_threshold_sweep.csv")
        with open(s_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            w.writeheader()
            w.writerows(sweep_rows)
        print(f"Wrote {s_path} ({len(sweep_rows)} rows)")
        if sweep_pp_rows:
            mf_rows = matched_far_comparison(sweep_pp_rows)
            mf_path = os.path.join(TABLES_DIR, "hypoglycemia_matched_far.csv")
            with open(mf_path, "w", newline="") as f:
                fields = sorted({k for r in mf_rows for k in r}, key=lambda k: list(mf_rows[-1]).index(k)
                                if k in mf_rows[-1] else 99)
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(mf_rows)
            print(f"Wrote {mf_path} ({len(mf_rows)} rows; participant-clustered bootstrap)")
        if not no_plots:
            _plot_threshold_sweep(sweep_rows, os.path.join(BASE_DIR, "overleaf", "images", "clinical", "fig_hypo_threshold_sweep.pdf"))

    if regime_rows:
        r_path = os.path.join(TABLES_DIR, "regime_summary_table.csv")
        with open(r_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(regime_rows[0].keys()))
            w.writeheader()
            w.writerows(regime_rows)
        write_regime_tex(regime_rows, os.path.join(TABLES_DIR, "table_regime_summary.tex"))
        print(f"Wrote {r_path} and table_regime_summary.tex ({len(regime_rows)} rows)")


def write_hypo_tex(rows, out_path, supplement=False):
    ds_disp = {"weinstock": "Weinstock 2016", "cgmacros": "CGMacros"}
    has_chance = bool(rows) and ("chance_sensitivity_pct" in rows[0])
    has_cluster = bool(rows) and ("cluster_ppv_pct" in rows[0])
    n_num_cols = 9 + (2 if has_chance else 0) + (2 if has_cluster else 0)
    cols_spec = "lll" + ("c" * n_num_cols)
    chance_hdr = " & Chance sens.\\ (\\%) & Above chance (pp)" if has_chance else ""
    cluster_hdr = " & Cluster PPV (\\%)" if has_cluster else ""
    cal_hdr = " & FAR / cal.\\ day" if has_cluster else ""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        f"\\begin{{tabular}}{{{cols_spec}}}",
        r"\toprule",
        f"Dataset & Horizon & Model & $N$ & Episodes & Pre-onset sens.\\ (\\%){chance_hdr} & Lead (all, min) & Lead (det., min) & Point PPV (\\%) & Point $F_1$ (\\%){cluster_hdr} & False alerts / day{cal_hdr} & Parkes A+B (\\%) \\\\",
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
        def _fmt1(v):
            if v is None or str(v).strip() == "":
                return "---"
            fv = float(v)
            return "---" if math.isnan(fv) else f"{fv:.1f}"

        def _fmts1(v):
            if v is None or str(v).strip() == "":
                return "---"
            fv = float(v)
            if math.isnan(fv):
                return "---"
            if abs(round(fv, 1)) < 1e-9:
                return "+0.0"
            return f"{fv:+.1f}"
        ch_cells = f" & {_fmt1(r.get('chance_sensitivity_pct'))} & {_fmts1(r.get('above_chance_pp'))}" if has_chance else ""
        cl_cell = f" & {_fmt1(r.get('cluster_ppv_pct'))}" if has_cluster else ""
        cal_val = r.get("false_alerts_per_calendar_day")
        cal_cell = (f" & {float(cal_val):.2f}" if (has_cluster and cal_val not in (None, "")) else (" & ---" if has_cluster else ""))
        lines.append(
            f"{ds} & {r['horizon']} hr & {m} & {r['n_participants']} & {r['n_episodes']} & "
            f"{_fmt1(r['episode_sensitivity_pct'])}{ch_cells} & {_fmt1(r['mean_lead_time_min'])} & "
            f"{_fmt1(r['mean_lead_time_detected_min'])} & "
            f"{_fmt1(r['point_ppv_pct'])} & {_fmt1(r['point_f1_pct'])}{cl_cell} & "
            f"{float(r['false_alerts_per_day']):.2f}{cal_cell} & {_fmt1(r['parkes_ab_pct'])} \\\\"
        )
    if supplement:
        cap = (
            r"\caption{\textbf{Descriptive 2-hour and 5-hour pre-onset hypoglycemia ($<70$~mg/dL) detection metrics and Parkes consensus error grid performance (full-history context).} "
            r"Descriptive long-horizon extension of Table~\ref{tab:hypoglycemia_events}. Notice that reactive persistence alerts targeting 2--5~hours ahead reflect recurrent low readings hours before a subsequent episode rather than active predictive anticipation. "
            r"Metrics follow Table~\ref{tab:hypoglycemia_events}.}"
        )
        lbl = r"\label{tab:hypoglycemia_events_supp}"
    else:
        cap = (
            r"\caption{\textbf{30-minute pre-onset hypoglycemia ($<70$~mg/dL) episode detection and Parkes consensus error grid performance (full-history context).} "
            r"True hypoglycemic episodes are runs of $\ge 3$ real readings ($15$~min) below $70$~mg/dL on the full observed trajectory, merged when $<30$~min apart, across the $N$ retained participants; "
            r"left-censored episodes (no reading $\ge 70$~mg/dL in the preceding 30~min) and episodes with no forecast issued before onset are excluded from ``Episodes''. "
            r"Pre-onset episode sensitivity counts episodes detected (one-to-one, by the earliest alert) by a forecast below 70~mg/dL issued strictly before onset ($t_{\text{issue}} = t - h < t_{\text{start}}$). "
            + (r"``Chance sens.'' and ``Above chance'' report expected pre-onset sensitivity under circular time-shifted alerts within each participant and the net gain over chance in percentage points. " if has_chance else "")
            + r"``Lead (all, min)'' reports the unconditional mean warning time prior to episode onset across all episodes (counting missed episodes as $0$~min), whereas ``Lead (det., min)'' averages over detected episodes. "
            r"Point PPV and $F_1$ evaluate timestep-level hypoglycemia classification; Cluster PPV reports the percentage of 30-min merged alert clusters coinciding with a real reading $<70$~mg/dL within $\pm 10$~min; false alerts per scored participant-day (and per elapsed calendar day, FAR / cal.\ day) count uncorroborated alert clusters; "
            r"and Parkes A+B reports participant-averaged proportions in Zones A and B using the Parkes Type 2 grid for CGMacros participants with type 2 diabetes and the Type 1 grid otherwise. (Descriptive 2- and 5-hour metrics appear in Supplementary Table~\ref{tab:hypoglycemia_events_supp}.)}"
        )
        lbl = r"\label{tab:hypoglycemia_events}"
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", cap, lbl, r"\end{table*}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_regime_tex(rows, out_path):
    ds_disp = {"weinstock": "Weinstock 2016", "cgmacros": "CGMacros"}
    reg_disp = {
        "STABLE": "Stable Euglycemia",
        "STABLE_OOR": "Stable Out-of-Range",
        "RISING": "Rising Excursion",
        "FALLING": "Falling Excursion",
        "HIGHVAR": "High Variability",
        "RECOVERY": "Recovery Phase",
        "HYPO_TRANS": "Hypoglycemic Transition",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Dataset & Operational regime & $N$ & Timestep share (\%) & HBML RMSE (mg/dL) & Best expert (RMSE, mg/dL) & Most frequent instantaneous winner (share, \%) \\",
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
            f"{ds} & {rg} & {r['n_patients']} & {r['step_share_pct']:.1f} & {r['hbml_rmse']:.2f} & "
            f"{r['best_expert']} ({r['best_expert_rmse']:.2f}) & "
            f"{r['top_freq_expert']} ({r['top_freq_pct']:.1f}\\%) \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{\textbf{Forecasting accuracy and expert selection frequency across operational glycemic regimes (30-minute horizon, full-history context).} "
        r"Scored timesteps are partitioned into seven mutually exclusive operational regimes in priority order (slow-moving glucose outside 70--180~mg/dL is Stable Out-of-Range, not Stable Euglycemia). "
        r"Columns report the number of participants $N$ contributing timesteps in each regime, the percentage of scored timesteps in each regime, cohort-mean participant-level RMSE for HBML and for the lowest-RMSE single expert in that regime, and the most frequent instantaneous winner: the constituent expert with the lowest instantaneous squared error at the largest share of timesteps in that regime (a property of the experts, not of HBML's weights).}",
        r"\label{tab:regime_summary}",
        r"\end{table*}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
