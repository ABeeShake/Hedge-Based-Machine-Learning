"""fast_engine/features.py -- causal feature and window builders (NumPy only).

Every builder takes the series ``y`` (the 5-minute grid inputs, no NaN) and the indices ``ends``
of the last observation available to a forecast; nothing after ``ends`` is ever read.
``valid`` (optional) marks slots outside long gaps; training samples whose target window touches
a long gap are dropped (see ``eligible_ends``). Kept free of torch /
xgboost imports so it can be unit-tested anywhere.
"""
import numpy as np

# Glucose deltas are divided by this before entering the neural experts (mg/dL).
NEURAL_SCALE = 50.0
# Lags (in 5-min steps) used by XGBoost: 5 min ... 3 h.
XGB_LAGS = (1, 2, 3, 4, 5, 6, 9, 12, 18, 24, 36)
XGB_WINDOW = max(XGB_LAGS) + 1


def lag_matrix(y, ends, width):
    """Rows y[e-width+1 .. e] (oldest -> newest) for each e in ends, left-padded with y[0]."""
    y = np.asarray(y, dtype=float)
    ends = np.asarray(ends, dtype=int)
    idx = ends[:, None] - np.arange(width - 1, -1, -1)[None, :]
    return y[np.clip(idx, 0, None)]


def xgb_features(y, ends):
    """Level, multi-scale deltas and rolling statistics at each end index (no exogenous inputs)."""
    W = lag_matrix(y, ends, XGB_WINDOW)
    last = W[:, -1]
    cols = [last]
    cols += [last - W[:, -1 - j] for j in XGB_LAGS]
    for w in (6, 12, 36):
        cols.append(W[:, -w:].mean(axis=1) - last)
        cols.append(W[:, -w:].std(axis=1))
    return np.column_stack(cols)


def eligible_ends(valid, t, h, need_path=False):
    """Training end points e in [0, t-h] whose target is known by time t and not inside a long gap.

    need_path=False: valid[e] and valid[e+h]   (direct h-step target, XGBoost)
    need_path=True : valid[e .. e+h] all       (whole trajectory is a target, N-HITS / NODE)
    """
    if t - h < 0:
        return np.zeros(0, dtype=int)
    ends = np.arange(0, t - h + 1)
    if valid is None:
        return ends
    v = np.asarray(valid, dtype=bool)[: t + 1]
    if not need_path:
        return ends[v[ends] & v[ends + h]]
    bad = np.r_[0, np.cumsum(~v)]
    return ends[bad[ends + h + 1] - bad[ends] == 0]


def xgb_training_set(y, t, h, valid=None):
    """Rows with end e in eligible_ends(valid, t, h); target = y[e+h] - y[e]."""
    ends = eligible_ends(valid, t, h)
    if len(ends) == 0:
        return None, None
    y = np.asarray(y, dtype=float)
    return xgb_features(y, ends), y[ends + h] - y[ends]


def neural_windows(y, ends, input_size, h=None):
    """Normalised input windows (and h-step targets) relative to the last input value."""
    y = np.asarray(y, dtype=float)
    X = lag_matrix(y, ends, input_size)
    base = X[:, -1:]
    Xn = (X - base) / NEURAL_SCALE
    if h is None:
        return Xn, base[:, 0]
    Y = y[np.asarray(ends)[:, None] + np.arange(1, h + 1)[None, :]]
    return Xn, (Y - base) / NEURAL_SCALE, base[:, 0]
