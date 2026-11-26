"""ExpMethods/evaluation.py -- shared helpers for scoring saved forecasts (NumPy/pandas only).

Forecast files have one row per 5-minute grid slot (ExpMethods/timegrid.py). Targets come from
load_targets (NaN where the slot has no real reading) and every score uses scored_mask, so all
models are compared on the same real readings."""
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from ExpMethods.globals import GlobalValues as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HORIZON_STEPS = {"half": 6, "2": 24, "5": 60}
HORIZON_LABEL = {"half": "0.5", "2": "2", "5": "5"}


def scored_mask(f_mat, targets):
    """Rows that enter every score: inside GlobalValues.eval_slice(), a real reading in the slot and an
    issued forecast (ExpMethods.aggregators.scorable). Use the expert forecast matrix, so every model of
    a participant is scored on the same rows."""
    from ExpMethods.aggregators import scorable
    ok = scorable(f_mat, targets)
    win = np.zeros(len(ok), dtype=bool)
    win[G.eval_slice()] = True
    return ok & win


def rmse(sq_err, mask=None):
    """RMSE over `mask` (from scored_mask); without a mask, over the finite rows of the eval window."""
    v = np.asarray(sq_err, dtype=float)
    v = v[mask[: len(v)]] if mask is not None else v[G.eval_slice()]
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v))) if len(v) else float("nan")


def _participant_csv(dataset, pid, root=ROOT):
    for d in (os.path.join(root, dataset), os.path.join(root, "Outputs", dataset, "data")):
        p = os.path.join(d, f"{pid}.csv")
        if os.path.exists(p):
            return p
    return None


_GRIDS = {}


def load_grid(dataset, pid, root=ROOT):
    """The participant's readings on the 5-minute grid (same grid the engine used); None if missing."""
    key = (dataset, str(pid), root, G.max_gap_min)
    if key not in _GRIDS:
        from ExpMethods.timegrid import Grid
        p = _participant_csv(dataset, pid, root)
        try:
            _GRIDS[key] = None if p is None else Grid.from_csv(p, max_gap_min=G.max_gap_min,
                                                               step_min=G.grid_step_min)
        except OSError:
            _GRIDS[key] = None
        if len(_GRIDS) > 64:
            _GRIDS.pop(next(iter(_GRIDS)))
    return _GRIDS[key]


def targets_from_csv(csv_path):
    """load_targets for an explicit participant file."""
    from ExpMethods.timegrid import Grid
    return Grid.from_csv(csv_path, max_gap_min=G.max_gap_min, step_min=G.grid_step_min).raw


def load_targets(dataset, pid, root=ROOT):
    """Scoring targets aligned with the forecast rows: the real reading in each 5-min slot, NaN where
    there is none (short-gap fills and long gaps are never targets)."""
    g = load_grid(dataset, pid, root)
    return None if g is None else g.raw.copy()


def participant_files(dataset, root=ROOT):
    files = sorted(glob.glob(os.path.join(root, dataset, "*.csv")))
    return [
        f for f in files
        if os.path.basename(f) != f"{dataset}.csv" and not os.path.basename(f).endswith("_manifest.csv")
    ]


def forecast_files(cond_dir, suffix="_forecasts.csv"):
    return sorted(f for f in glob.glob(os.path.join(cond_dir, "forecasts", f"*{suffix}"))
                  if os.path.basename(f)[3:] == suffix)


def read_meta(fc_path):
    try:
        return json.load(open(fc_path[: -len(".csv")] + ".meta.json"))
    except (OSError, ValueError):
        return None


def wilcoxon_p(d):
    """Two-sided Wilcoxon signed-rank p-value (SciPy if available, else normal approximation)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) < 6:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        a = pd.Series(np.abs(d))
        r = a.rank().to_numpy()
        n = len(d)
        ties = a.value_counts().to_numpy()
        var = n * (n + 1) * (2 * n + 1) / 24 - (ties ** 3 - ties).sum() / 48
        z = (r[d > 0].sum() - n * (n + 1) / 4) / math.sqrt(var)
        return float(math.erfc(abs(z) / math.sqrt(2)))


def paired_summary(a, b):
    """Compare per-participant RMSE arrays a (reference) and b; positive delta = b lower."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    return dict(delta_mean=float(np.mean(d)), rel_pct=float(100 * (a.mean() - b.mean()) / a.mean()),
                pct_improved=float(100 * np.mean(d > 1e-3)), wilcoxon_p=wilcoxon_p(d))


def tex_p(p):
    if p is None or p != p:
        return "---"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"
