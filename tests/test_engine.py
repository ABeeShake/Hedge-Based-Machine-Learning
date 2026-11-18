#!/usr/bin/env python3
"""tests/test_engine.py -- checks for the v2.1 simulation engine (5-minute grid).

Run before any long simulation:   python3 tests/test_engine.py
Core tests need only NumPy/pandas. The smoke test of the full expert pool runs when torch,
xgboost, statsforecast and neuralforecast are installed (i.e. on the VM) and is skipped otherwise.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ExpMethods.globals import GlobalValues as G  # noqa: E402
from fast_engine import features as ft  # noqa: E402
from fast_engine import models as fm  # noqa: E402
from fast_engine.simulation import FastRollingSimulation  # noqa: E402
from ExpMethods.timegrid import Grid, snap  # noqa: E402
from ExpMethods import aggregators as agg  # noqa: E402
from ExpMethods import evaluation as ev  # noqa: E402

FIRST = 287   # first origin = slot of the 288th real reading (index 287 on a gap-free series)


def series(T=900, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    return 140 + 40 * np.sin(2 * np.pi * t / 288) + np.cumsum(rng.normal(0, 2, T)) * 0.3


class CountingExpert(fm.BaseExpert):
    """Records every history length it sees and every refit time; forecasts y(t)."""
    name = "Counting"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen, self.fits = [], []

    def fit(self, history):
        self.fits.append(len(history) - 1)

    def predict(self, history):
        self.seen.append(len(history) - 1)
        return history[-1]


def test_alignment_and_burn_in():
    y = series()
    for h in (6, 24, 60):
        sim = FastRollingSimulation(horizon=h, context_len=-1, burn_in=288, experts=["Persistence", "RunningMean"],
                                    clip=None)
        fc = sim.run_patient(y)["forecasts"]
        rows = np.arange(FIRST + h, len(y))
        assert np.allclose(fc["Persistence"][rows], y[rows - h]), "persistence must equal y[t] stored at t+h"
        means = np.cumsum(y) / np.arange(1, len(y) + 1)
        assert np.allclose(fc["RunningMean"][rows], means[rows - h])
        assert np.all(fc["Persistence"][: FIRST + h] == 0), "no forecasts before the burn-in"


def test_no_lookahead_and_refit_schedule():
    y = series()
    fm.EXPERT_CLASSES["Counting"] = lambda **kw: CountingExpert(**kw)
    try:
        sim = FastRollingSimulation(horizon=6, burn_in=288, retrain_interval=100, experts=["Counting"], clip=None)
        models = fm.build_experts(["Counting"], 6, None, 100)
        m = models["Counting"]
        for t in range(288, len(y) - 6):
            m.forecast(y[: t + 1])
        assert m.seen == list(range(288, len(y) - 6)), "expert must see y[0..t] exactly"
        assert m.fits == list(range(288, len(y) - 6, 100)), f"refits at burn-in then every 100 steps: {m.fits[:5]}"
        # perturbing the future must not change any forecast
        a = sim.run_patient(y)["forecasts"]["Counting"]
        y2 = y.copy(); y2[600:] += 500
        b = sim.run_patient(y2)["forecasts"]["Counting"]
        assert np.array_equal(a[: 600 + 6], b[: 600 + 6]), "forecasts up to row t+h may only use y[0..t]"
        assert m.fits[0] == 288
    finally:
        fm.EXPERT_CLASSES.pop("Counting", None)


def test_clipping():
    y = series(); y[400:420] = 900
    fc = FastRollingSimulation(horizon=6, burn_in=288, experts=["Persistence"]).run_patient(y)["forecasts"]
    assert fc["Persistence"].max() <= G.forecast_clip[1]


def test_chronological_val_split_and_protocol_constants():
    assert G.primary_hbml == "HBML-SFH-Sqrt"
    assert G.min_scored_points == 288
    assert G.equiv_margin_mgdl == 1.0
    pool = np.arange(0, 300)
    tr, va = fm.chronological_val_split(pool, horizon=24)
    assert len(va) >= 5 and np.all(tr + 24 < va[0]), "train targets must stay strictly before validation origins"
    # Direct Parkes Type 1 & Type 2 sanity checks (Pfutzner et al. 2013)
    from evaluate_clinical_and_weights import parkes_error_grid_zone
    assert parkes_error_grid_zone(100.0, 105.0, 1) == "A"
    assert parkes_error_grid_zone(100.0, 105.0, 2) == "A"
    assert parkes_error_grid_zone(40.0, 350.0, 1) == "E"
    assert parkes_error_grid_zone(40.0, 350.0, 2) == "E"


def test_feature_causality():
    y = series(600)
    e = np.array([10, 300, 500])
    X = ft.xgb_features(y, e)
    y2 = y.copy(); y2[501:] = -1e6
    assert np.allclose(X, ft.xgb_features(y2, e)), "features at e may not read y[e+1:]"
    Xt, Yt = ft.xgb_training_set(y, 400, 24)
    valid = np.ones(600, bool); valid[100:150] = False
    e = ft.eligible_ends(valid, 400, 24)
    assert not np.any(~valid[e]) and not np.any(~valid[e + 24])
    e = ft.eligible_ends(valid, 400, 24, need_path=True)
    assert all(valid[i:i + 25].all() for i in e) and 75 in e and 76 not in e and 150 in e
    assert len(Xt) == 400 - 24 + 1 and np.allclose(Yt, y[np.arange(377) + 24] - y[np.arange(377)])
    Xn, Yn, base = ft.neural_windows(y, np.array([100, 200]), 72, 6)
    assert Xn.shape == (2, 72) and Yn.shape == (2, 6) and np.allclose(Xn[:, -1], 0)
    assert np.allclose(Yn[0] * ft.NEURAL_SCALE + base[0], y[101:107])


# ---------------------------------------------------------------- 5-minute grid
def gappy(T=900, seed=1):
    """Series on the grid with a short gap (4 missing slots) and a long gap (60 missing slots)."""
    y = series(T, seed)
    y[400:404] = np.nan          # 25 min between readings -> filled
    y[500:560] = np.nan          # 5 h -> new segment
    return y


def test_snap_and_fill():
    t0 = np.datetime64("2020-01-01T00:00:00")
    mins = np.array([0, 5.2, 9.7, 15.1, 15.4, 40, 45, 120, 125])     # jitter, a duplicate, gaps
    ts = t0 + (mins * 60).astype("timedelta64[s]")
    g = np.arange(len(mins), dtype=float) + 100
    slot, gs, nd = snap(ts, g)
    assert list(slot) == [0, 1, 2, 3, 8, 9, 24, 25] and nd == 1 and gs[3] == 103, (slot, gs)
    grid = Grid(slot, gs, t0, max_gap_min=30)
    assert grid.observed.sum() == 8 and len(grid) == 26
    assert grid.valid[4:8].all() and np.allclose(grid.inputs[3:9], np.linspace(103, 105, 6))  # 25 min: filled
    assert not grid.valid[10:24].any() and grid.segment[24] == 1 and grid.segment_start[25] == 24
    assert np.isnan(grid.raw[4:8]).all(), "filled slots are never targets"


def test_grid_origins_and_scoring():
    y = gappy()
    h = 6
    sim = FastRollingSimulation(horizon=h, burn_in=288, experts=["Persistence"], clip=None)
    res = sim.run_patient(y)
    fc = res["forecasts"]["Persistence"]
    issued = np.flatnonzero(fc != 0) - h            # origins
    assert np.all(np.isfinite(y[issued])), "origins must hold a real reading"
    assert not np.any((issued >= 400) & (issued < 404)), "no origin inside a short gap"
    assert not np.any((issued >= 500) & (issued < 560 + G.min_segment_history - 1)), "warm-up after a long gap"
    assert 560 + G.min_segment_history - 1 in issued
    assert res["grid"]["n_segments"] == 2 and res["grid"]["n_origins"] == len(issued)
    # scoring: real reading at t+h and an issued forecast
    F = fc[:, None]
    mask = ev.scored_mask(F, y)
    assert not np.any(mask & np.isnan(y)) and not np.any(mask & (fc == 0))
    L = agg.online_losses(F, y, 288, h)
    assert np.all(L[~agg.scorable(F, y)] == 0), "no feedback without a real reading or a forecast"
    _, loss = agg.run_hedge(np.column_stack([F, F + 1]), np.column_stack([L, L]), y, start=288, horizon=h)
    assert np.all(np.isnan(loss[~agg.scorable(F, y) & (np.arange(len(y)) >= 288 + h)]))
    assert np.isfinite(ev.rmse(loss, mask))


def test_grid_no_lookahead():
    """Changing a reading at slot k may only change forecasts issued from origins >= k, including
    origins whose input window contains the filled / bridged slots before k."""
    y = gappy()
    fm.EXPERT_CLASSES["Counting"] = lambda **kw: CountingExpert(**kw)
    try:
        for k in (404, 560, 700):             # end of the short gap, end of the long gap, ordinary slot
            sim = FastRollingSimulation(horizon=6, burn_in=288, experts=["Persistence", "RunningMean"], clip=None)
            a = sim.run_patient(y)["forecasts"]
            y2 = y.copy(); y2[k] += 300
            b = sim.run_patient(y2)["forecasts"]
            for n in a:
                assert np.array_equal(a[n][: k + 6], b[n][: k + 6]), (n, k)
    finally:
        fm.EXPERT_CLASSES.pop("Counting", None)


def test_real_file_if_present():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weinstock", "001.csv")
    try:
        g = Grid.from_csv(p, max_gap_min=G.max_gap_min)
    except OSError:
        print("  skipped real-file test (weinstock/001.csv not available)")
        return
    res = FastRollingSimulation(horizon=24, experts=["Persistence", "RunningMean"]).run_patient(g)
    fc = res["forecasts"]["Persistence"]
    org = np.flatnonzero(fc != 0) - 24
    assert np.all(g.observed[org]) and np.allclose(fc[org + 24], np.clip(g.raw[org], *G.forecast_clip))
    y = ev.targets_from_csv(p)
    assert len(y) == len(fc)
    print(f"  001: {res['grid']['n_slots']} slots, {res['grid']['n_segments']} segments, "
          f"{res['grid']['n_origins']} origins ({res['grid']['n_skipped_warmup']} skipped for warm-up), "
          f"{int(ev.scored_mask(fc[:, None], y).sum())} scored rows")


def test_full_pool_smoke():
    try:
        import torch, xgboost, statsforecast, neuralforecast  # noqa: F401
    except ImportError as e:
        print(f"  skipped full-pool smoke test ({e.name} not installed)")
        return
    y = series(480)
    y[330:333] = np.nan                       # short gap
    y[380:420] = np.nan                       # long gap: inputs bridged, warm-up, training skips it
    sim = FastRollingSimulation(horizon=6, context_len=72, burn_in=288, retrain_interval=60)
    res = sim.run_patient(y)
    rows = np.flatnonzero(res["forecasts"]["Persistence"] != 0)
    for n, f in res["forecasts"].items():
        assert np.all(np.isfinite(f[rows])) and np.all(f[rows] >= 40) and np.all(f[rows] <= 400), n
        assert res["timings"][n]["n_fallback"] == 0, (n, res["timings"][n]["first_error"])
    sc = rows[np.isfinite(y[rows])]
    rmse = {n: float(np.sqrt(np.mean((f[sc] - y[sc]) ** 2))) for n, f in res["forecasts"].items()}
    print("  smoke RMSE:", {k: round(v, 1) for k, v in rmse.items()})
    print("  fit seconds:", {k: round(v["fit_s_mean"], 2) for k, v in res["timings"].items()})


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
