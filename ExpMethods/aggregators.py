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
UPDATES = ("cumulative", "recursive", "ftl")
SCALES = ("ema", "decay_max", "running_max", "sqrt_sum_sq")
LOSS_FUNCTIONS = ("squared", "mae", "huber15", "huber25", "pinball20", "asym_hypo")


def compute_point_loss(pred, target, loss_fn="squared"):
    """Elementwise forecasting loss on real target slots (pred and target broadcast cleanly)."""
    if loss_fn == "squared":
        return (pred - target) ** 2
    if loss_fn == "mae":
        return np.abs(pred - target)
    if loss_fn in ("huber15", "huber25"):
        delta = 15.0 if loss_fn == "huber15" else 25.0
        err = np.abs(pred - target)
        return np.where(err <= delta, 0.5 * err ** 2, delta * (err - 0.5 * delta))
    if loss_fn == "pinball20":
        tau = 0.20
        diff = target - pred
        return np.where(diff >= 0.0, tau * diff, (tau - 1.0) * diff)
    if loss_fn == "asym_hypo":
        sq = (pred - target) ** 2
        return np.where((target < 80.0) & (pred > target), 4.0 * sq, sq)
    raise ValueError(f"Unknown loss_fn {loss_fn!r}; expected one of {LOSS_FUNCTIONS}")


def _point_loss_deriv(pred, target, loss_fn="squared"):
    """Derivative (or subgradient) d/d(pred) of compute_point_loss(pred, target, loss_fn)."""
    r = pred - target
    if loss_fn == "squared":
        return 2.0 * r
    if loss_fn == "mae":
        return np.sign(r)
    if loss_fn in ("huber15", "huber25"):
        delta = 15.0 if loss_fn == "huber15" else 25.0
        return np.clip(r, -delta, delta)
    if loss_fn == "pinball20":
        tau = 0.20
        return np.where(r >= 0.0, 1.0 - tau, -tau)
    if loss_fn == "asym_hypo":
        w = np.where((target < 80.0) & (r > 0.0), 4.0, 1.0)
        return 2.0 * w * r
    raise ValueError(f"Unknown loss_fn {loss_fn!r}; expected one of {LOSS_FUNCTIONS}")


def scorable(f_mat, targets):
    """Rows with an issued forecast (every expert column non-zero; unissued rows are 0) and a real
    reading (finite target). Only these rows produce feedback or enter any score."""
    f_mat = np.asarray(f_mat, dtype=float)
    if f_mat.ndim == 1:
        f_mat = f_mat[:, None]
    y = np.asarray(targets, dtype=float).ravel()
    n = min(len(f_mat), len(y))
    ok = np.zeros(len(f_mat), dtype=bool)
    f = f_mat[:n]
    ok[:n] = np.isfinite(y[:n]) & np.all(np.isfinite(f) & (f != 0), axis=1)
    return ok


def online_losses(f_mat, targets, start, horizon, loss_fn="squared"):
    """Online feedback losses; zero (= no feedback, aggregators skip the update) on warm-up and
    non-scorable rows. Defaults to squared error, mirroring simulate.get_online_losses."""
    f_mat = np.asarray(f_mat, dtype=float)
    y = np.asarray(targets, dtype=float).reshape(-1, 1)
    with np.errstate(invalid="ignore"):
        l_mat = compute_point_loss(f_mat, y, loss_fn=loss_fn)
    l_mat[: start + horizon] = 0.0
    l_mat[~scorable(f_mat, y)] = 0.0
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
    sum_sq = 0.0
    lnK = max(float(np.log(max(K, 2))), 1e-12)
    w = np.ones(K) / K
    W = np.ones((T + 1, K)) / K if return_weights else None
    out_f = np.zeros(T)
    out_l = np.zeros(T)
    ok = scorable(f_mat, y) if y is not None else np.ones(T, dtype=bool)
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
            elif scale == "sqrt_sum_sq":
                sum_sq += l_top * l_top
                L = float(np.sqrt(sum_sq / lnK))
            else:
                L = max(L, l_top)
            L_eff = max(L, EPS)

            if update == "ftl":
                cum = cum + l_safe
                c_masked = np.where(nan_mask, np.inf, cum)
                best_val = np.min(c_masked)
                leaders = np.isclose(c_masked, best_val, rtol=1e-12, atol=1e-12)
                v = leaders.astype(float) / max(int(leaders.sum()), 1)
            elif update == "cumulative":
                cum = cum + l_safe
                log_w = -cum / L_eff
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
                # rows that are not scorable get NaN, so they can never enter an RMSE by accident
                out_l[t_target] = (pred - y[t_target]) ** 2 if ok[t_target] else np.nan
            else:
                out_l[t_target] = np.nansum(w * l_mat[t_target])

    if return_weights:
        return out_f, out_l, W
    return out_f, out_l


# Named configurations. PRIMARY_SPECS keys are the model names written to disk.
SPECS = {
    "HBML-SFHDF": dict(update="cumulative", scale="ema", gamma=0.2, alpha=0.0),
    "HBML-FS": dict(update="recursive", scale="decay_max", gamma=0.2, alpha="fd"),
    "HBML-SFH-Sqrt": dict(update="cumulative", scale="sqrt_sum_sq", gamma=0.0, alpha=0.0),
    "FTL": dict(update="ftl", scale="running_max", gamma=0.0, alpha=0.0),
}


def run_named(name, f_mat, l_mat, targets=None, **kwargs):
    spec = dict(SPECS[name])
    spec.update({k: v for k, v in kwargs.items() if k in ("gamma", "alpha", "update", "scale")})
    return run_hedge(f_mat, l_mat, targets, start=kwargs["start"], horizon=kwargs["horizon"],
                     return_weights=kwargs.get("return_weights", False), **spec)


def _run_interleaved(runner_fn, f_mat, l_mat, targets, *, start, horizon, return_weights=False):
    """Shared (horizon + 1) modulo-strided non-delayed interleaved wrapper."""
    f_mat = np.asarray(f_mat, dtype=float)
    l_mat = np.asarray(l_mat, dtype=float)
    y = None if targets is None else np.asarray(targets, dtype=float).ravel()
    T, K = f_mat.shape
    stride = int(horizon) + 1
    out_f = np.zeros(T, dtype=float)
    ok = scorable(f_mat, y) if y is not None else np.ones(T, dtype=bool)
    out_l = np.zeros(T, dtype=float)
    out_l[int(start) + int(horizon):] = np.nan
    W = np.ones((T + 1, K), dtype=float) / K if return_weights else None

    for r in range(stride):
        idx_r = np.arange(int(start) + int(horizon) + r, T, stride)
        if len(idx_r) == 0:
            continue
        f_pad = np.vstack([np.zeros((1, K), dtype=float), f_mat[idx_r]])
        l_pad = np.vstack([np.zeros((1, K), dtype=float), l_mat[idx_r]])
        y_pad = np.r_[np.nan, y[idx_r]] if y is not None else None
        res = runner_fn(f_pad, l_pad, y_pad)
        if return_weights:
            f_sub, l_sub, w_sub = res
            W[idx_r + 1] = w_sub[1:len(idx_r) + 1]
        else:
            f_sub, l_sub = res
        out_f[idx_r] = f_sub[1:]
        if y is not None:
            valid = ok[idx_r]
            out_l[idx_r[valid]] = l_sub[1:][valid]
        else:
            out_l[idx_r] = l_sub[1:]

    if return_weights:
        return out_f, out_l, W
    return out_f, out_l


def run_interleaved_hedge(name, f_mat, l_mat, targets, *, start, horizon, return_weights=False, **kwargs):
    """Run (horizon + 1) independent interleaved non-delayed Hedge learners across slot offsets."""
    return _run_interleaved(
        lambda f_sub, l_sub, y_sub: run_named(name, f_sub, l_sub, y_sub, start=0, horizon=1,
                                              return_weights=return_weights, **kwargs),
        f_mat, l_mat, targets, start=start, horizon=horizon, return_weights=return_weights,
    )


def run_mixture_eg(f_mat, l_mat, targets, *, start, horizon, loss_fn="squared",
                   prior_mix=0.0, return_weights=False):
    """Scale-free Exponentiated Gradient optimizing ensemble mixture loss l(w^T f_t, y_t) directly.

    Uses causal delayed gradient feedback g_t = d/dw l(w^T f_t, y_t) = psi'(pred_retro - y_t) * f_t,
    normalized via Orabona-Pal adaptive range scale L_t = sqrt(sum_s (max g_s - min g_s)^2 / ln K).
    Because predictions are shifted around pred_retro, the rule is both shift- and unit-scale invariant.
    Returns squared error in ``out_l`` for uniform RMSE evaluation alongside online predictions.
    """
    f_mat = np.asarray(f_mat, dtype=float)
    y = np.asarray(targets, dtype=float).ravel()
    T, K = f_mat.shape
    lnK = max(float(np.log(max(K, 2))), 1e-12)
    w = np.ones(K) / K
    cum_g = np.zeros(K)
    sum_sq = 0.0
    W = np.ones((T + 1, K)) / K if return_weights else None
    out_f = np.zeros(T)
    out_l = np.zeros(T)
    ok = scorable(f_mat, y)

    for g in range(start, T):
        if g >= start + horizon and g < len(y) and ok[g]:
            fg = f_mat[g]
            fin = np.isfinite(fg)
            if fin.any():
                w_fin = np.where(fin, w, 0.0)
                ws = float(w_fin.sum())
                w_eff = (w_fin / ws) if ws > 1e-12 else (fin.astype(float) / fin.sum())
                pred_retro = float(np.nansum(w_eff * fg))
                fg_safe = np.where(fin, fg, pred_retro)
                d_loss = float(_point_loss_deriv(pred_retro, y[g], loss_fn=loss_fn))
                grad = d_loss * (fg_safe - pred_retro)
                osc = float(grad.max() - grad.min())
                sum_sq += osc * osc
                L_eff = max(float(np.sqrt(sum_sq / lnK)), EPS)
                cum_g = cum_g + grad
                log_w = np.where(fin, -cum_g / L_eff, -np.inf)
                log_w_fin = log_w[fin]
                v = np.zeros(K)
                v[fin] = np.exp(log_w_fin - log_w_fin.max())
                s = float(v.sum())
                v = (v / s) if s > 1e-12 else (fin.astype(float) / fin.sum())
                w = (1.0 - prior_mix) * v + prior_mix / K if prior_mix > 0.0 else v

        if return_weights:
            W[g + 1] = w
        t_target = g + horizon
        if t_target < T:
            pred = float(np.nansum(w * f_mat[t_target]))
            out_f[t_target] = pred
            if t_target < len(y):
                out_l[t_target] = (pred - y[t_target]) ** 2 if ok[t_target] else np.nan

    if return_weights:
        return out_f, out_l, W
    return out_f, out_l


def run_interleaved_mixture_eg(f_mat, l_mat, targets, *, start, horizon, loss_fn="squared",
                               prior_mix=0.0, return_weights=False):
    """Run (horizon + 1) independent interleaved sub-learners of ``run_mixture_eg``."""
    return _run_interleaved(
        lambda f_sub, l_sub, y_sub: run_mixture_eg(
            f_sub, l_sub, y_sub, start=0, horizon=1, loss_fn=loss_fn,
            prior_mix=prior_mix, return_weights=return_weights,
        ),
        f_mat, l_mat, targets, start=start, horizon=horizon, return_weights=return_weights,
    )
