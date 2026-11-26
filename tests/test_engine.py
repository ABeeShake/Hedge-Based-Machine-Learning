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
    # too little history for a non-overlapping split: no validation set, never overlapping targets
    tr, va = fm.chronological_val_split(np.arange(0, 40), horizon=24)
    assert len(va) == 0 and len(tr) == 40
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
    # short-gap fills are never training targets
    observed = np.ones(600, bool); observed[300:303] = False
    e = ft.eligible_ends(None, 400, 24, observed=observed)
    assert not np.any(np.isin(e + 24, [300, 301, 302])) and 275 in e and 276 not in e
    e = ft.eligible_ends(valid, 400, 24, need_path=True, observed=observed)
    assert np.all(observed[e + 24])
    M = ft.target_mask(observed, np.array([290, 350]), 24)
    assert M.shape == (2, 24) and M[0].sum() == 21 and M[1].all()
    Xo, Yo = ft.xgb_training_set(y, 400, 24, observed=observed)
    assert len(Xo) == len(Xt) - 3
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


def test_hypo_chance_and_threshold_sweep():
    import evaluate_clinical_and_weights as ecw
    n = 400
    y = np.full(n, 140.0)
    y[200:206] = 62.0   # 30-min clinical hypoglycaemic episode at slots 200..205
    mask = np.ones(n, dtype=bool)
    mask[:20] = False
    # Forecast predicts 73 mg/dL strictly pre-onset (slot 200, issued 194 = 200 - 6 < 200)
    pred = np.full(n, 140.0)
    pred[200] = 73.0
    r70 = ecw.evaluate_hypo_trajectory(y, pred, mask, h=6, thresh=70.0, true_thresh=70.0, n_chance_shifts=50)
    r75 = ecw.evaluate_hypo_trajectory(y, pred, mask, h=6, thresh=75.0, true_thresh=70.0, n_chance_shifts=50)
    assert r70["n_episodes"] == 1 and r70["n_detected"] == 0
    assert r75["n_episodes"] == 1 and r75["n_detected"] == 1
    assert r75["chance_detected_sum"] < 0.05
    assert r75["n_detected_w0"] == 1 and r75["n_detected_w20"] == 1   # target 200 issued at 194: 30 min warning
    pred2 = np.full(n, 140.0); pred2[200] = 65.0                         # issued at 194 -> 30 min before onset
    r = ecw.evaluate_hypo_trajectory(y, pred2, mask, h=6, n_chance_shifts=0)
    assert r["lead_times_det"] == [30.0] and r["n_detected_w20"] == 1
    pred3 = np.full(n, 140.0); pred3[205] = 65.0                         # issued at 199 -> 5 min before onset
    r = ecw.evaluate_hypo_trajectory(y, pred3, mask, h=6, n_chance_shifts=0)
    assert r["n_detected_w0"] == 1 and r["n_detected_w10"] == 0
    # episodes use every real reading (not just scored slots) and drop left-censored onsets
    y2 = y.copy(); y2[300:306] = 60.0; y2[294:300] = np.nan             # onset inside a gap: censored
    eps, cens = ecw.find_hypo_episodes(y2)
    assert eps == [(200, 205)] and cens == 1
    m2 = mask.copy(); m2[200:203] = False                                # unscored low readings still define onset
    eps, _ = ecw.find_hypo_episodes(y)
    assert eps == [(200, 205)]
    r = ecw.evaluate_hypo_trajectory(y, pred3, m2, h=6, n_chance_shifts=0)
    assert r["n_episodes"] == 1 and r["n_detected"] == 1


def test_matched_far_bootstrap():
    import evaluate_clinical_and_weights as ecw
    rows = []
    for m, base in (("HBML", 0.5), ("Persistence", 0.7)):
        for thr, far in ((70, 0.2), (80, 0.6), (90, 1.2)):
            for pid in range(20):
                det = int(round(10 * min(1.0, base + 0.2 * (thr - 70) / 10)))
                rows.append(dict(model=m, threshold_mgdl=thr, pid=f"{pid:03d}", n_episodes=10,
                                 false_clusters=far * 10, days=10.0,
                                 **{f"n_detected_w{w}": det for w in ecw.WARNING_MINUTES}))
    out = ecw.matched_far_comparison(rows, far_targets=(0.1, 0.4), n_boot=50)
    p = [r for r in out if r["model"] == "Persistence" and r["min_warning_min"] == 0]
    assert np.isnan(p[0]["sensitivity_pct"]), "no extrapolation outside the swept range"
    assert abs(p[1]["diff_vs_reference_pp"] - (-20.0)) < 1e-6



def test_simplex_ls_fixed_weights():
    from analyze_pool_mechanism import simplex_ls
    rng = np.random.default_rng(0)
    y = rng.normal(120, 30, 3000)
    F = np.column_stack([y + rng.normal(0, 5, 3000), y + rng.normal(0, 10, 3000), y + 30.0])
    w = simplex_ls(F, y)
    assert abs(w.sum() - 1) < 1e-9 and np.all(w >= 0) and w[2] < 0.05 and w[0] > w[1]


def test_interleaved_hedge_and_mechanism_variants():
    from analyze_pool_mechanism import participant_variants
    rng = np.random.default_rng(12)
    T = 450
    h = 6
    start = 288
    y = series(T, seed=2)
    F = np.zeros((T, len(G.experts)), dtype=float)
    for j in range(len(G.experts)):
        F[start + h:, j] = y[start + h:] + rng.normal(0, 4.0 + 3.0 * j, T - start - h)
    L = agg.online_losses(F, y, start, h)
    pred_i, loss_i = agg.run_interleaved_hedge(G.primary_hbml, F, L, y, start=start, horizon=h)
    # Causality check: modifying targets >= k must not change forecasts strictly before k + h
    y2 = y.copy()
    y2[360:] += 400.0
    L2 = agg.online_losses(F, y2, start, h)
    pred_i2, _ = agg.run_interleaved_hedge(G.primary_hbml, F, L2, y2, start=start, horizon=h)
    assert np.allclose(pred_i[:360 + h], pred_i2[:360 + h])
    assert np.isfinite(ev.rmse(loss_i, ev.scored_mask(F, y)))
    mask = ev.scored_mask(F, y)
    res = participant_variants(F, y, list(G.experts), mask, h)
    for key in ("HBML", "HBML-interleaved", "Uniform", "HBML-no-reference", "HBML-shortctx-pool", "Expert-ARIMA"):
        assert key in res and np.isfinite(res[key]), key


def test_cluster_ppv_and_calendar_far():
    import evaluate_clinical_and_weights as ecw
    n = 400
    y = np.full(n, 130.0)
    y[220:226] = 62.0  # true hypo cluster around 220..225
    mask = np.ones(n, dtype=bool)
    mask[:30] = False
    pred = np.full(n, 130.0)
    pred[219] = 66.0   # true alert cluster (within +-2 slots of y[220])
    pred[330] = 66.0   # false alert cluster
    r = ecw.evaluate_hypo_trajectory(y, pred, mask, h=6, n_chance_shifts=0)
    assert abs(r["days_calendar"] - n / 288.0) < 1e-9
    assert r["n_alert_clusters"] == 2
    assert r["true_clusters"] == 1
    assert r["false_clusters"] == 1


def test_ohiot1dm_preparation_and_registry():
    import tempfile
    import prepare_ohiot1dm as prep
    from run_simulation_chunk import build_condition_registry
    reg_default = build_condition_registry()
    assert len(reg_default) == 24
    reg_ohio = build_condition_registry(["ohiot1dm"])
    assert len(reg_ohio) == 12 and all(c["dataset"] == "ohiot1dm" for c in reg_ohio)
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = os.path.join(tmp, "raw", "2020")
        out_dir = os.path.join(tmp, "ohiot1dm")
        os.makedirs(raw_dir, exist_ok=True)
        with open(os.path.join(raw_dir, "540-ws-training.xml"), "w") as f:
            f.write("<patient id='540'><glucose_level>"
                    "<event ts='01-01-2020 00:00:00' value='110'/>"
                    "<event ts='01-01-2020 00:05:00' value='115'/>"
                    "</glucose_level></patient>")
        with open(os.path.join(raw_dir, "540-ws-testing.xml"), "w") as f:
            f.write("<patient id='540'><glucose_level>"
                    "<event ts='01-01-2020 00:10:00' value='120'/>"
                    "</glucose_level></patient>")
        rows = prep.prepare_ohiot1dm(os.path.join(tmp, "raw"), out_dir, split="combined")
        assert len(rows) == 1 and rows[0]["pid"] == "001" and rows[0]["source_id"] == "540"
        assert rows[0]["cohort_year"] == "2020" and rows[0]["n_points"] == 3
        pts = prep.parse_ohio_csv(os.path.join(out_dir, "001.csv"))
        assert [v for _, v in pts] == [110.0, 115.0, 120.0]


def test_glucobench_preparation():
    import io
    import tempfile
    import zipfile
    import prepare_glucobench as pgb
    from run_simulation_chunk import build_condition_registry
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "raw_data.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("raw_data/hall.csv",
                        "id,time,gl,diagnosis\n"
                        "sub-b,2014-02-01 00:05:00,112.0,2\n"
                        "sub-a,2014-02-01 00:00:00,95.0,0\n"
                        "sub-a,2014-02-01 00:05:00,,0\n")
            zf.writestr("raw_data/dubosson.csv",
                        "id,time,gl\n"
                        "2.0,2014-10-01 19:14:00,185.4\n")
        res = pgb.prepare_glucobench(["hall", "dubosson"], zip_path=zip_path, output_root=tmp)
        assert len(res["hall"]) == 2 and res["hall"][0]["diagnosis"] == "normoglycemic"
        assert res["hall"][1]["diagnosis"] == "T2D" and res["hall"][1]["pid"] == "002"
        assert len(res["dubosson"]) == 1 and res["dubosson"][0]["diagnosis"] == "T1D"
        assert len(build_condition_registry(["hall"])) == 12


def test_validation_and_external_scripts_synthetic():
    import csv
    import json
    import tempfile
    import evaluate_external_cohort as ext
    import validate_engine_outputs as val
    with tempfile.TemporaryDirectory() as tmp:
        ds = "mockcohort"
        ds_dir = os.path.join(tmp, ds)
        os.makedirs(ds_dir, exist_ok=True)
        n_rows = 700
        t0 = np.datetime64("2021-01-01T00:00:00")
        y_vals = 135.0 + 25.0 * np.sin(np.arange(n_rows) * 0.05)
        y_vals[400:406] = 63.0
        for pid in ("001", "002", "003", "004", "005", "006"):
            with open(os.path.join(ds_dir, f"{pid}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "Timestamp", "Dexcom.GL"])
                for i in range(n_rows):
                    ts = (t0 + np.timedelta64(5 * i, "m")).astype(str).replace("T", " ")
                    w.writerow([pid, ts, f"{y_vals[i]:.1f}"])
        fc_dir = os.path.join(tmp, "Outputs", ds, "h-halfhr", "context-fullhr", "forecasts")
        os.makedirs(fc_dir, exist_ok=True)
        for pid_idx, pid in enumerate(("001", "002", "003", "004", "005", "006")):
            fc_path = os.path.join(fc_dir, f"{pid}_forecasts.csv")
            with open(fc_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(list(G.experts))
                scale = 1.0 + 0.15 * pid_idx
                for i in range(n_rows):
                    vals = [0.0 if i < 294 else round(float(y_vals[i] + scale * (k + 1)), 2) for k in range(len(G.experts))]
                    w.writerow(vals)
            from fast_engine.simulation import ENGINE_VERSION
            meta = {
                "settings": {"engine_version": ENGINE_VERSION},
                "timings": {"NHITS": {"n_fits": 3, "n_fallback": 0, "fit_windows_train_median": 120.0,
                                      "fit_windows_train_min": 80, "fit_windows_val_median": 24.0,
                                      "n_fits_without_early_stopping": 0}},
            }
            with open(fc_path.replace(".csv", ".meta.json"), "w") as mf:
                json.dump(meta, mf)
        orig_base_val, orig_base_ext, orig_root_ev = val.BASE_DIR, ext.BASE_DIR, ev.ROOT
        try:
            fit_rows = val.summarize_fit_windows([ds], os.path.join(tmp, "Outputs"))
            assert len(fit_rows) == 1 and fit_rows[0]["total_refits"] == 18
            import ExpMethods.timegrid as tg
            if hasattr(tg.pd, "read_csv"):
                val.BASE_DIR = tmp
                ext.BASE_DIR = tmp
                ev.ROOT = tmp
                val_out = os.path.join(tmp, "Outputs", "validation")
                ext_out = os.path.join(tmp, "Outputs", "external", ds)
                val.run_validation([ds], os.path.join(tmp, "Outputs"), val_out, n_boot=50)
                ext.evaluate_cohort(ds, os.path.join(tmp, "Outputs"), ext_out, n_boot_stat=50, n_boot_mech=50)
                for fn in ("validation_fit_windows.csv", "validation_cohort_sensitivity.csv", "validation_gap_burden.csv"):
                    assert os.path.exists(os.path.join(val_out, fn)), fn
                for fn in ("rmse_summary.csv", "statistical_tests_best_expert.csv", "clinical_hypo.csv",
                           "pool_mechanism_summary.csv", "external_report.md"):
                    assert os.path.exists(os.path.join(ext_out, fn)), fn
        finally:
            val.BASE_DIR, ext.BASE_DIR, ev.ROOT = orig_base_val, orig_base_ext, orig_root_ev
    # Constant/zero paired differences must not trigger scipy division-by-zero or moment warnings
    import warnings
    import calculate_statistics as cstat
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        st = cstat.paired_stats(np.full(8, 10.0), np.full(8, 10.0), n_boot=50)
        assert np.isnan(st["ttest_p"]) and np.isnan(st["shapiro_p"])
    scipy_warns = [w for w in caught if issubclass(w.category, (RuntimeWarning, UserWarning))
                   and ("moment calculation" in str(w.message) or "divide by zero" in str(w.message).lower()
                        or "range zero" in str(w.message).lower())]
    assert not scipy_warns, [str(w.message) for w in scipy_warns]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
