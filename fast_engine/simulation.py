"""fast_engine/simulation.py -- walk-forward simulation on the 5-minute grid (engine v2.1).

Each participant's readings are put on a 5-minute grid (ExpMethods/timegrid.py). A forecast is
issued from slot t only if
  * t holds a real reading,
  * the burn-in is over (the participant's first `burn_in` real readings, i.e. 24 h of data), and
  * t is at least `min_segment_history` slots into its segment (1 h after a long gap ends).
Each expert receives the grid inputs y[0..t] (short gaps filled, long gaps bridged) and the
mask of slots outside long gaps, and forecasts y[t+h], i.e. h * 5 minutes of clock time ahead.
The forecast is stored at row t+h of <id>_forecasts.csv (one row per grid slot); rows without an
issued forecast are 0, the same convention as the burn-in rows. Only rows whose slot holds a real
reading are ever scored (ExpMethods/evaluation.py). Forecasts are clipped to the sensor range;
if an expert raises, the step falls back to persistence and the failure is counted.

Each participant also gets <id>_forecasts.meta.json with the engine version, protocol settings,
grid summary and per-expert timings; run_simulation_chunk.py treats a forecast file as complete
only if its metadata matches the current engine and settings.
"""
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .models import build_experts

ENGINE_VERSION = "2.1"


def _defaults():
    from ExpMethods.globals import GlobalValues as G
    return G


def load_grid(csv_path, max_gap_min=None):
    """The participant's readings on the 5-minute grid (ExpMethods.timegrid.Grid)."""
    from ExpMethods.timegrid import Grid
    G = _defaults()
    return Grid.from_csv(csv_path, max_gap_min=G.max_gap_min if max_gap_min is None else max_gap_min,
                         step_min=G.grid_step_min)


def load_series(csv_path):
    """Grid model inputs (short gaps filled, long gaps bridged). Kept for scripts that need a plain array."""
    return load_grid(csv_path).inputs


class FastRollingSimulation:
    def __init__(self, horizon=6, context_len=-1, burn_in=None, retrain_interval=None,
                 end_step=None, experts: Optional[List[str]] = None, clip=None, seed=0,
                 start_step=None, min_segment_history=None, max_gap_min=None):
        G = _defaults()
        self.horizon = int(horizon)
        self.context_len = None if (context_len is None or context_len <= 0) else int(context_len)
        # start_step is accepted for backward compatibility and means the same as burn_in.
        self.burn_in = int(burn_in if burn_in is not None else (start_step if start_step is not None
                                                                  else G.burn_in_steps))
        self.retrain_interval = int(retrain_interval or G.retrain_interval)
        self.end_step = end_step
        self.experts = list(experts or G.experts)
        self.clip = G.forecast_clip if clip is None else clip
        self.seed = seed
        self.min_segment_history = int(G.min_segment_history if min_segment_history is None
                                       else min_segment_history)
        self.max_gap_min = float(G.max_gap_min if max_gap_min is None else max_gap_min)
        self.grid_step_min = float(G.grid_step_min)

    def settings(self):
        return dict(engine_version=ENGINE_VERSION, horizon=self.horizon, context_len=self.context_len,
                    burn_in=self.burn_in, retrain_interval=self.retrain_interval, experts=self.experts,
                    clip=list(self.clip) if self.clip else None, seed=self.seed,
                    grid_step_min=self.grid_step_min, max_gap_min=self.max_gap_min,
                    min_segment_history=self.min_segment_history)

    def _as_grid(self, X):
        from ExpMethods.timegrid import Grid
        if isinstance(X, Grid):
            return X
        # a plain array is taken to be already on the grid (NaN = no reading)
        return Grid.from_values(np.asarray(X, dtype=float).ravel(), max_gap_min=self.max_gap_min,
                                step_min=self.grid_step_min)

    def run_patient(self, X, verbose=False, log_interval=500):
        grid = self._as_grid(X)
        y, valid = grid.inputs, grid.valid
        T, h = len(grid), self.horizon
        origins = grid.origins(self.burn_in, h, self.min_segment_history, self.end_step)
        models = build_experts(self.experts, h, self.context_len, self.retrain_interval, self.seed)
        fc = {n: np.zeros(T) for n in models}
        call_s = {n: 0.0 for n in models}
        n_fail = {n: 0 for n in models}
        errors = {}
        t_start = time.perf_counter()
        for i, t in enumerate(origins):
            hist, val = y[: t + 1], valid[: t + 1]
            for n, m in models.items():
                t0 = time.perf_counter()
                try:
                    p = m.forecast(hist, val)
                    if not np.isfinite(p):
                        raise FloatingPointError("non-finite forecast")
                except Exception as e:  # keep the run alive; record and fall back to persistence
                    n_fail[n] += 1
                    errors.setdefault(n, repr(e)[:300])
                    p = hist[-1]
                call_s[n] += time.perf_counter() - t0
                if self.clip:
                    p = min(max(p, self.clip[0]), self.clip[1])
                fc[n][t + h] = p
            if verbose and (i + 1) % log_interval == 0:
                print(f"  origin {i + 1}/{len(origins)} (slot {t}/{T}, {time.perf_counter() - t_start:.1f}s)",
                      flush=True)
        n_steps = len(origins)
        timings = {}
        for n, m in models.items():
            fit = np.asarray(m.fit_seconds, dtype=float)
            timings[n] = dict(n_steps=n_steps, n_fits=int(len(fit)),
                              fit_s_total=float(fit.sum()), fit_s_mean=float(fit.mean()) if len(fit) else 0.0,
                              fit_s_sd=float(fit.std(ddof=1)) if len(fit) > 1 else 0.0,
                              # per-step forecast cost excluding refits
                              forecast_s_mean=float((call_s[n] - fit.sum()) / max(n_steps, 1)),
                              n_fallback=int(n_fail[n]), first_error=errors.get(n))
        first = grid.first_origin(self.burn_in)
        all_obs = grid.observed[first:max(T - h, first)]
        grid_info = grid.summary() | dict(
            first_origin=int(first), n_origins=int(n_steps),
            # real readings after the burn-in that could have been an origin, and how many were skipped
            n_candidate_origins=int(all_obs.sum()), n_skipped_warmup=int(all_obs.sum() - n_steps))
        return dict(forecasts=fc, timings=timings, total_time_s=time.perf_counter() - t_start,
                    steps_completed=n_steps, settings=self.settings(), grid=grid_info)


def meta_path(fc_path):
    return fc_path[: -len(".csv")] + ".meta.json"


def is_complete(fc_path, expected: Dict):
    """True if fc_path exists and its metadata matches every key in `expected`."""
    mp = meta_path(fc_path)
    if not (os.path.exists(fc_path) and os.path.exists(mp)):
        return False
    try:
        meta = json.load(open(mp))
    except (OSError, ValueError):
        return False
    s = meta.get("settings", {})
    return all(s.get(k) == v for k, v in expected.items())


def _patient_id(csv_path):
    base = os.path.basename(csv_path)
    pid = base[: base.find(".csv")]
    return pid[pid.find("-") + 1: pid.find("-") + 4] if "-" in pid else pid


def _worker_run_patient(task):
    csv_path, kw, output_dir, skip_existing = task
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    pid = _patient_id(csv_path)
    sim = FastRollingSimulation(**kw)
    fc_dir = os.path.join(output_dir, "forecasts")
    fc_path = os.path.join(fc_dir, f"{pid}_forecasts.csv")
    if skip_existing and is_complete(fc_path, sim.settings()):
        return pid, 0.0
    res = sim.run_patient(load_grid(csv_path, sim.max_gap_min))
    os.makedirs(fc_dir, exist_ok=True)
    pd.DataFrame(res["forecasts"]).to_csv(fc_path + ".tmp", index=False)
    os.replace(fc_path + ".tmp", fc_path)
    with open(meta_path(fc_path), "w") as f:
        json.dump(dict(id=pid, settings=res["settings"], grid=res["grid"], timings=res["timings"],
                       total_time_s=res["total_time_s"], steps_completed=res["steps_completed"]), f, indent=1)
    bad = {n: t["n_fallback"] for n, t in res["timings"].items() if t["n_fallback"]}
    print(f"done {pid} in {res['total_time_s']:.1f}s" + (f"  fallbacks: {bad}" if bad else ""), flush=True)
    return pid, res["total_time_s"]


def run_cohort(csv_paths, horizon=6, context_len=-1, output_dir=None, n_workers=4, skip_existing=True,
               burn_in=None, retrain_interval=None, experts=None, end_step=None, seed=0, start_step=None):
    """Simulate every participant in csv_paths in parallel; returns {id: seconds}."""
    from concurrent.futures import ProcessPoolExecutor
    kw = dict(horizon=horizon, context_len=context_len, burn_in=burn_in, retrain_interval=retrain_interval,
              experts=experts, end_step=end_step, seed=seed, start_step=start_step)
    tasks = [(p, kw, output_dir, skip_existing) for p in csv_paths]
    if n_workers <= 1:
        return dict(_worker_run_patient(t) for t in tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        return dict(ex.map(_worker_run_patient, tasks))
