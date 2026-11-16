"""ExpMethods/timegrid.py -- put raw CGM readings on a regular 5-minute grid.

  1. snap     each reading advances round(interval / 5 min) slots from the previous one (so clock
              drift and sensor restarts cannot create spurious gaps); a reading < 2.5 min after
              the previous one is dropped.
  2. reindex  every slot from the first to the last reading; `observed` marks real readings.
  3. fill     if consecutive real readings are at most `max_gap_min` apart, the slots between them
              are linearly interpolated.
  4. segment  a longer gap ends a segment (`segment` = -1 inside it).

Model inputs use `inputs`: real readings, short gaps filled, long gaps bridged by a straight line
between the readings on either side. Every filled value depends only on readings at or before the
end of its gap, and forecasts are only issued from slots with a real reading, so an expert at slot
t never sees anything after t. Only real readings are scored (`raw` is NaN everywhere else).

Used by the simulation engine (fast_engine/simulation.py) and every scoring script
(ExpMethods/evaluation.py), so forecasts and targets always share the same rows.
"""
import numpy as np
import pandas as pd

STEP_MIN = 5.0


def read_raw(csv_path):
    """(timestamps as datetime64[ns], glucose) from a participant file (Weinstock / CGMacros layout)."""
    df = pd.read_csv(csv_path)
    col = "Libre.GL" if "Libre.GL" in df.columns else "Dexcom.GL"
    ts = pd.to_datetime(df["Timestamp"]).to_numpy()
    g = df[col].to_numpy(dtype=float)
    ok = np.isfinite(g) & ~pd.isna(ts)
    ts, g = ts[ok], g[ok]
    order = np.argsort(ts, kind="stable")
    return ts[order], g[order]


def snap(ts, g, step_min=STEP_MIN, return_keep=False):
    """Assign readings to slots. Returns (slot indices, glucose, number of readings dropped)
    [, indices of the kept readings].

    Each reading advances round(interval / 5 min) slots from the previous kept reading, so a
    sensor whose clock runs at 5.0x min (or restarts at an arbitrary second after a gap) never
    produces spurious collisions or one-slot gaps, as snapping to a fixed absolute grid would.
    A reading less than 2.5 min after the previous one shares its slot and is dropped.
    """
    mins = (ts - ts[0]) / np.timedelta64(1, "s") / 60.0
    keep = [0]
    slot = [0]
    for i in range(1, len(mins)):
        k = int(np.rint((mins[i] - mins[keep[-1]]) / step_min))
        if k <= 0:
            continue
        keep.append(i)
        slot.append(slot[-1] + k)
    keep = np.asarray(keep)
    out = (np.asarray(slot, dtype=np.int64), np.asarray(g)[keep], int(len(mins) - len(keep)))
    return out + (keep,) if return_keep else out


def _interp_runs(values, known):
    """Linearly interpolate the unknown slots of `values` from the known ones (interior gaps only)."""
    idx = np.arange(len(values))
    out = values.copy()
    out[~known] = np.interp(idx[~known], idx[known], values[known])
    return out


class Grid:
    """A participant's series on the 5-minute grid.

    raw              real reading per slot, NaN where there is none (the scoring targets)
    observed         True where a real reading sits in the slot
    inputs           model inputs: raw + short gaps filled + long gaps bridged (no NaN)
    valid            True outside long gaps (real or short-filled); training targets must be valid
    segment          segment id per slot (-1 inside long gaps)
    segment_start    first slot of each slot's segment (-1 inside long gaps)
    times            nominal datetime64 of each slot (first reading + k * 5 min)
    reading_slots /  slot and original timestamp of every kept reading (from_csv only), for mapping
    reading_times    timestamp-indexed data (e.g. regime labels) onto the grid with to_grid()
    """

    def __init__(self, slot, g, t0=None, max_gap_min=30.0, step_min=STEP_MIN, n_duplicates=0):
        slot = np.asarray(slot, dtype=np.int64)
        T = int(slot[-1]) + 1
        self.step_min = float(step_min)
        self.max_gap_min = float(max_gap_min)
        # a gap of k missing slots spans (k + 1) * step minutes between real readings
        self.max_fill_slots = int(round(max_gap_min / step_min)) - 1
        self.n_duplicates = int(n_duplicates)
        raw = np.full(T, np.nan)
        raw[slot] = g
        self.raw = raw
        self.observed = np.isfinite(raw)
        self.inputs = _interp_runs(raw, self.observed)
        # gap length (missing slots) between consecutive real readings
        obs_idx = np.flatnonzero(self.observed)
        run = np.diff(obs_idx) - 1
        long_gap = run > self.max_fill_slots
        valid = np.ones(T, dtype=bool)
        for a, r in zip(obs_idx[:-1][long_gap], run[long_gap]):
            valid[a + 1: a + 1 + r] = False
        self.valid = valid
        new_seg = np.zeros(T, dtype=bool)
        new_seg[0] = True
        new_seg[obs_idx[1:][long_gap]] = True
        seg = np.cumsum(new_seg) - 1
        seg[~valid] = -1
        self.segment = seg
        starts = np.flatnonzero(new_seg)
        st = starts[np.clip(np.searchsorted(starts, np.arange(T), side="right") - 1, 0, None)]
        st[~valid] = -1
        self.segment_start = st
        self.times = None if t0 is None else t0 + (np.arange(T) * step_min * 60).astype("timedelta64[s]")

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_csv(cls, csv_path, max_gap_min=30.0, step_min=STEP_MIN):
        ts, g = read_raw(csv_path)
        slot, gs, nd, keep = snap(ts, g, step_min, return_keep=True)
        grid = cls(slot, gs, ts[0], max_gap_min, step_min, nd)
        grid.reading_slots, grid.reading_times = slot, ts[keep]
        return grid

    @classmethod
    def from_values(cls, values, max_gap_min=30.0, step_min=STEP_MIN):
        """Grid from values already on the grid (NaN = no reading). Used by tests and synthetic data."""
        v = np.asarray(values, dtype=float)
        idx = np.flatnonzero(np.isfinite(v))
        v = v[: idx[-1] + 1]
        g = cls(idx, v[idx], None, max_gap_min, step_min)
        if idx[0] > 0:          # leading missing slots: treat as a long gap
            g.valid[: idx[0]] = False
            g.segment[: idx[0]] = -1
            g.segment_start[: idx[0]] = -1
        return g

    # ------------------------------------------------------------------ helpers
    def to_grid(self, by_timestamp):
        """Place a Series indexed by the raw reading timestamps (e.g. regime labels) on the grid rows;
        slots without a reading get NaN / None. Only for grids built with from_csv."""
        vals = pd.Series(by_timestamp)
        vals.index = pd.to_datetime(vals.index)
        vals = vals[~vals.index.duplicated()]
        out = np.full(len(self), None if vals.dtype == object else np.nan, dtype=vals.dtype if vals.dtype == object else float)
        out[self.reading_slots] = vals.reindex(pd.to_datetime(self.reading_times)).to_numpy()
        return out

    def __len__(self):
        return len(self.raw)

    @property
    def filled(self):
        return self.valid & ~self.observed

    def segment_bounds(self):
        """(start, end_exclusive) of each segment."""
        s = self.segment
        ids = np.flatnonzero(np.diff(np.r_[-2, s, -2]) != 0)
        return [(int(a), int(b)) for a, b in zip(ids[:-1], ids[1:]) if s[a] >= 0]

    def first_origin(self, burn_in_obs):
        """Slot of the `burn_in_obs`-th real reading (the first forecast origin)."""
        obs = np.flatnonzero(self.observed)
        return int(obs[min(burn_in_obs, len(obs)) - 1]) if len(obs) else len(self)

    def origins(self, burn_in_obs, horizon, min_segment_history=0, end=None):
        """Forecast origins: real reading, past the burn-in, at least `min_segment_history` slots
        into its segment, and target slot inside the grid."""
        T = len(self)
        end = T - horizon if end is None else min(end, T - horizon)
        t = np.arange(self.first_origin(burn_in_obs), max(end, 0))
        if len(t) == 0:
            return t
        ok = self.observed[t] & (t - self.segment_start[t] + 1 >= min_segment_history)
        return t[ok]

    def summary(self):
        b = self.segment_bounds()
        return dict(n_slots=len(self), n_observed=int(self.observed.sum()),
                    n_filled=int(self.filled.sum()), n_long_gap=int((~self.valid).sum()),
                    n_segments=len(b), n_duplicates=self.n_duplicates)
