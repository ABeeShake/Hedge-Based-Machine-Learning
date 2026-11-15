"""ExpMethods/aggregators.py -- dependency-light online expert aggregators.

This module needs only NumPy, so the same code runs in the SLURM pipeline, in
``compare_fixed_share.py`` and in the unit tests. ``ExpMethods/simulate.py``
exposes these functions through its usual ``(forecasts, losses, **kwargs)``
interface.

Timing convention (shared with the rest of the code base)
---------------------------------------------------------
``f_mat[t, k]`` is expert k's forecast *for* time t, issued at time t - h.
``l_mat[t, k]`` is its squared error, which becomes known at time t.
At step g the aggregator first learns from ``l_mat[g]`` and then issues the
combined forecast for ``g + h``. No information later than g is used.

One engine, ``run_hedge``, covers the variants below.

=============================  ===========  ============  =========
variant                        update       scale         alpha
=============================  ===========  ============  =========
HBML-SFHDF (results to date)   cumulative   ema           0
Algorithm 1 in the manuscript  recursive    decay_max     1/n^2
=============================  ===========  ============  =========

``update``
    ``cumulative``: w ~ exp(-sum_s loss_s / L_t)  (whole history rescaled by today's L)
    ``recursive`` : v ~ p_t * exp(-(loss_t - min loss_t) / L_t)
``scale``  (L_t, the loss scale; gamma is the forgetting rate)
    ``ema``         : L_t = (1 - gamma) L_{t-1} + gamma * max_k loss_t
    ``decay_max``   : L_t = max((1 - gamma) L_{t-1}, max_k loss_t)
    ``running_max`` : L_t = max(L_{t-1}, max_k loss_t)
``alpha``  (Fixed-Share mixing, p_{t+1} = (1 - a) v' + a / K)
    0 / None  : no sharing
    float     : constant sharing rate
    "d"       : a_n = 1 / (n + 1)        (n = number of updates so far)
    "fd"      : a_n = 1 / (n + 1)^2
"""
import numpy as np

EPS = 1e-8
UPDATES = ("cumulative", "recursive")
SCALES = ("ema", "decay_max", "running_max")


def online_losses(f_mat, targets, start, horizon):
    """Squared errors with the warm-up rows zeroed (mirrors simulate.get_online_losses)."""
    f_mat = np.asarray(f_mat, dtype=float)
    y = np.asarray(targets, dtype=float).reshape(-1, 1)
    l_mat = (f_mat - y) ** 2
    l_mat[: start + horizon] = 0.0
    return l_mat


def alpha_value(alpha, n):
    """Sharing rate applied after the n-th weight update (n >= 1)."""
    if alpha is None:
        return 0.0
    if isinstance(alpha, str):
        if alpha == "fd":
            return 1.0 / (n + 1) ** 2
        if alpha == "d":
            return 1.0 / (n + 1)
        if alpha in ("0", "none", ""):
            return 0.0
        return float(alpha)
    if callable(alpha):
        return float(alpha(n))
    return float(alpha)


def run_hedge(f_mat, l_mat, targets=None, *, start, horizon, update="cumulative",
              scale="ema", gamma=0.2, alpha=0.0, return_weights=False):
    """Run one aggregator over a single participant.

    Returns ``(forecasts, losses)`` or ``(forecasts, losses, weights)`` where
    ``weights[g + 1]`` is the distribution in force after step g.
    """
    if update not in UPDATES:
        raise ValueError(f"update must be one of {UPDATES}")
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    f_mat = np.asarray(f_mat, dtype=float)
    l_mat = np.asarray(l_mat, dtype=float)
    T, K = l_mat.shape
    y = None if targets is None else np.asarray(targets, dtype=float).ravel()
    gamma = float(gamma)

    cum = np.zeros(K)
    L = EPS
    w = np.ones(K) / K
    W = np.ones((T + 1, K)) / K if return_weights else None
    out_f = np.zeros(T)
    out_l = np.zeros(T)
    n_upd = 0

    for g in range(start, T):
        l = l_mat[g]
        if not (l == 0).all():
            nan_mask = np.isnan(l)
            if nan_mask.any():
                fill = np.nanmax(l) if not nan_mask.all() else 0.0
                l_safe = np.where(nan_mask, fill, l)
            else:
                l_safe = l
            l_top = float(l_safe.max())
            if scale == "ema":
                L = (1.0 - gamma) * L + gamma * l_top
            elif scale == "decay_max":
                L = max((1.0 - gamma) * L, l_top)
            else:
                L = max(L, l_top)
            L_eff = max(L, EPS)

            if update == "cumulative":
                cum = cum + l_safe
                log_w = -cum / L_eff
            else:
                with np.errstate(divide="ignore"):
                    log_w = np.log(w) - (l_safe - l_safe.min()) / L_eff
            if nan_mask.any():
                log_w = np.where(nan_mask, -np.inf, log_w)
            finite = np.isfinite(log_w)
            if finite.any():
                v = np.exp(log_w - log_w[finite].max())
                v[~finite] = 0.0
                s = v.sum()
                v = v / s if s > 1e-12 else np.ones(K) / K
            else:
                v = np.ones(K) / K

            n_upd += 1
            a = alpha_value(alpha, n_upd)
            w = (1.0 - a) * v + a / K if a > 0.0 else v

        if return_weights:
            W[g + 1] = w
        t_target = g + horizon
        if t_target < T:
            pred = np.nansum(w * f_mat[t_target])
            out_f[t_target] = pred
            if y is not None and t_target < len(y):
                out_l[t_target] = (pred - y[t_target]) ** 2
            else:
                out_l[t_target] = np.nansum(w * l_mat[t_target])

    if return_weights:
        return out_f, out_l, W
    return out_f, out_l


# Named configurations. PRIMARY_SPECS keys are the model names written to disk.
SPECS = {
    "HBML-SFHDF": dict(update="cumulative", scale="ema", gamma=0.2, alpha=0.0),
    "HBML-FS": dict(update="recursive", scale="decay_max", gamma=0.2, alpha="fd"),
}


def run_named(name, f_mat, l_mat, targets=None, **kwargs):
    spec = dict(SPECS[name])
    spec.update({k: v for k, v in kwargs.items() if k in ("gamma", "alpha", "update", "scale")})
    return run_hedge(f_mat, l_mat, targets, start=kwargs["start"], horizon=kwargs["horizon"],
                     return_weights=kwargs.get("return_weights", False), **spec)
