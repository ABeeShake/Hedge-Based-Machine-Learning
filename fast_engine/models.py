"""fast_engine/models.py -- online experts for the walk-forward simulation (engine v2).

Common protocol (see ``BaseExpert``)
-----------------------------------
``forecast(history, valid)`` receives the grid inputs up to the origin, ``y[0..t]`` (5-min slots;
short gaps filled, long gaps bridged; see ExpMethods/timegrid.py) and the mask of slots outside long
gaps, the mask of slots holding a real reading, and returns the forecast of ``y[t+h]``. Training
windows lie outside long gaps, and every training target is a real reading: the h-step end point
must be observed and trajectory losses are masked to observed slots, so short-gap fills are used
only as inputs. Experts with parameters are (re)fitted on ``y[0..t]`` the first time
they are called (the end of the burn-in) and again every ``retrain_interval`` steps; between
refits they only condition on new data. Nothing after ``t`` is ever used.

``context_len`` (None = full history) is the look-back window an expert conditions on:
ARIMA/ETS are fitted to it and N-HITS uses it as its input length. Learned experts (NODE,
N-HITS, XGBoost) are trained on all readings observed up to the refit time.

Heavy dependencies (torch, xgboost, statsforecast, neuralforecast) are imported lazily so
the cheap experts and the tests run without them.
"""
import os
import time
from typing import Optional

import numpy as np

from . import features as ft


def resolve_device(device=None):
    import torch
    if device is not None:
        return device
    if os.environ.get("FAST_ENGINE_DEVICE", "cpu").strip().lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BaseExpert:
    name = "base"
    needs_fit = True

    def __init__(self, horizon, context_len=None, retrain_interval=288, seed=0):
        self.horizon = int(horizon)
        self.context_len = None if (context_len is None or context_len <= 0) else int(context_len)
        self.retrain_interval = int(retrain_interval)
        self.seed = seed
        self.last_fit_t = None
        self.fit_seconds = []
        self.valid = None
        self.observed = None
        self.fit_log = []    # one dict per refit: window counts and whether early stopping ran

    def lookback(self, history):
        return history if self.context_len is None else history[-self.context_len:]

    def fit(self, history):  # pragma: no cover - overridden
        raise NotImplementedError

    def predict(self, history):  # pragma: no cover - overridden
        raise NotImplementedError

    def forecast(self, history, valid=None, observed=None):
        history = np.asarray(history, dtype=float).ravel()
        self.valid = None if valid is None else np.asarray(valid, dtype=bool)[: len(history)]
        self.observed = None if observed is None else np.asarray(observed, dtype=bool)[: len(history)]
        t = len(history) - 1
        if self.needs_fit and (self.last_fit_t is None or t - self.last_fit_t >= self.retrain_interval):
            t0 = time.perf_counter()
            self.fit(history)
            self.fit_seconds.append(time.perf_counter() - t0)
            self.last_fit_t = t
        return float(self.predict(history))


class Persistence(BaseExpert):
    """y_hat(t+h) = y(t)."""
    name = "Persistence"
    needs_fit = False

    def predict(self, history):
        return history[-1]


class RunningMean(BaseExpert):
    """y_hat(t+h) = mean of the real readings in y(0..t) (ignores context_len and gap fills)."""
    name = "RunningMean"
    needs_fit = False

    def predict(self, history):
        mask = self.observed if self.observed is not None else self.valid
        return history.mean() if mask is None or not mask.any() else history[mask].mean()


class StatsForecaster(BaseExpert):
    """AutoARIMA / AutoETS: model selection on the look-back window at each refit,
    Kalman-filter state update on the current look-back window at every step."""

    def __init__(self, model_type="arima", **kw):
        super().__init__(**kw)
        self.model_type = model_type.lower()
        self.name = "ARIMA" if self.model_type == "arima" else "ETS"
        self.state = None

    def _auto(self):
        from statsforecast.models import AutoARIMA, AutoETS
        return AutoARIMA() if self.model_type == "arima" else AutoETS()

    def fit(self, history):
        m = self._auto()
        m.fit(y=self.lookback(history))
        self.state = m.model_

    def predict(self, history):
        from statsforecast.models import forward_arima, forward_ets, forecast_arima, forecast_ets
        y = self.lookback(history)
        fwd, fc = (forward_arima, forecast_arima) if self.model_type == "arima" else (forward_ets, forecast_ets)
        try:
            state = fwd(self.state, y)
        except Exception:
            self.fit(history)          # numerical failure: refit on the current window
            state = self.state
        return fc(state, h=self.horizon)["mean"][-1]


class XGBoostExpert(BaseExpert):
    """Direct h-step gradient-boosted trees on lagged levels, deltas and rolling statistics
    (fast_engine.features.xgb_features); predicts the change y(t+h) - y(t)."""
    name = "XGBoost"

    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=4, **kw):
        super().__init__(**kw)
        self.n_estimators = n_estimators
        self.params = {"max_depth": max_depth, "eta": learning_rate, "objective": "reg:squarederror",
                       "tree_method": "hist", "subsample": 0.8, "colsample_bytree": 0.8,
                       "lambda": 1.0, "alpha": 0.0, "verbosity": 0, "nthread": 1, "seed": self.seed}
        self.booster = None

    def fit(self, history):
        import xgboost as xgb
        X, Y = ft.xgb_training_set(history, len(history) - 1, self.horizon, self.valid, self.observed)
        self.fit_log.append(dict(t=len(history) - 1, n_train=0 if X is None else len(X), n_val=0,
                                 early_stopping=False))
        self.booster = None if X is None or len(X) < 20 else xgb.train(
            self.params, xgb.DMatrix(X, label=Y), num_boost_round=self.n_estimators)

    def predict(self, history):
        if self.booster is None:
            return history[-1]
        import xgboost as xgb
        x = ft.xgb_features(history, np.array([len(history) - 1]))
        return history[-1] + float(self.booster.predict(xgb.DMatrix(x))[0])


def chronological_val_split(pool, horizon, min_train=16, min_val=5):
    """Split sorted eligible end indices into chronological train and held-out validation pools.

    The last 20% of windows (at least `min_val`) are held out. Training windows are restricted to
    ends whose whole target path finishes strictly before the first validation origin
    (e + horizon < val_ends[0]), so no training target overlaps a validation window. If fewer than
    `min_train` such windows exist, no validation set is returned and the caller trains for a
    short fixed budget without early stopping; targets are never allowed to overlap.
    """
    pool = np.asarray(pool, dtype=int)
    empty = np.zeros(0, dtype=int)
    if len(pool) < min_train + min_val:
        return pool, empty
    n_val = max(min_val, len(pool) // 5)
    val_ends = pool[-n_val:]
    strict_train = pool[pool + horizon < val_ends[0]]
    if len(strict_train) < min_train:
        return pool, empty
    return strict_train, val_ends


def _masked_mse(torch, pred, target, mask):
    """Mean squared error over observed target slots only."""
    m = mask.to(pred.dtype)
    return ((pred - target) ** 2 * m).sum() / m.sum().clamp_min(1.0)


def _fit_early_stopping(torch, net, opt, n_steps, val_loss_fn, step_fn,
                        eval_every=20, patience=4, max_grad_norm=1.0):
    """Run up to `n_steps` gradient updates with clipping and validation early stopping."""
    has_val = val_loss_fn is not None
    if has_val:
        net.eval()
        with torch.no_grad():
            best_val = float(val_loss_fn())
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        stale = 0
    net.train()
    step = 0
    for step in range(1, n_steps + 1):
        loss = step_fn()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        opt.step()
        if has_val and (step % eval_every == 0 or step == n_steps):
            net.eval()
            with torch.no_grad():
                v = float(val_loss_fn())
            if np.isfinite(v) and v < best_val * (1.0 - 1e-4):
                best_val = v
                best_state = {k: val.detach().cpu().clone() for k, val in net.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
            net.train()
    if has_val:
        net.load_state_dict({k: v.to(next(net.parameters()).device) for k, v in best_state.items()})
    net.eval()
    return step


class NODEExpert(BaseExpert):
    """Neural ODE on standardised glucose integrated over normalized unit interval [0, 1].
    Each refit runs up to `fit_steps` Adam steps (first fit: `first_fit_steps`) on mini-batches of
    h-step trajectories with gradient clipping and early stopping on a chronological 20% validation
    split, warm-starting from the previous best checkpoint."""
    name = "NODE"

    def __init__(self, hidden_dim=64, lr=1e-3, weight_decay=1e-3, batch_size=32,
                 first_fit_steps=300, fit_steps=100, device=None, **kw):
        super().__init__(**kw)
        import torch
        import ExpMethods.models as m
        torch.manual_seed(self.seed)
        self.torch = torch
        self.device = resolve_device(device)
        params = m.DefaultModelParams.node_params(horizon=self.horizon, hidden_dim=hidden_dim)
        params["sensitivity"] = "autograd"   # direct backprop: far faster than adjoint for this small field
        self.net = m.NODE(**params).to(self.device)
        # Integrate over normalized tau in [0, 1] so DepthCat time injection remains O(1) across h in {6, 24, 60}.
        self.t_span = torch.linspace(0.0, 1.0, self.horizon + 1, device=self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size, self.first_fit_steps, self.fit_steps = batch_size, first_fit_steps, fit_steps
        self.rng = np.random.default_rng(self.seed)
        self.mu, self.sd = 0.0, 1.0

    def _traj_loss(self, z, ends, offs):
        torch = self.torch
        s = torch.as_tensor(ends, dtype=torch.long, device=self.device)
        x0 = z[s].unsqueeze(-1)
        target = z[s[:, None] + offs]
        mask = torch.as_tensor(ft.target_mask(self.observed, ends, self.horizon), device=self.device)
        traj = self.net(x0, self.t_span)[1]
        return _masked_mse(torch, traj[1:, :, 0].T, target, mask)

    def fit(self, history):
        torch = self.torch
        t, h = len(history) - 1, self.horizon
        pool = ft.eligible_ends(self.valid, t, h, need_path=True, observed=self.observed)
        if len(pool) == 0:
            self.fit_log.append(dict(t=t, n_train=0, n_val=0, early_stopping=False, steps=0))
            return
        train_ends, val_ends = chronological_val_split(pool, h)
        mask = self.observed if self.observed is not None else self.valid
        ref = history if mask is None else history[mask]
        self.mu, self.sd = float(ref.mean()), float(max(ref.std(), 1.0))
        z = torch.tensor((history - self.mu) / self.sd, dtype=torch.float32, device=self.device)
        n_steps = self.first_fit_steps if not self.fit_seconds else self.fit_steps
        if len(val_ends) == 0:
            n_steps = min(n_steps, 50)
        offs = torch.arange(1, h + 1, device=self.device)
        val_sub = val_ends if len(val_ends) <= 256 else val_ends[-256:]
        val_fn = (lambda: float(self._traj_loss(z, val_sub, offs))) if len(val_ends) > 0 else None
        step_fn = lambda: self._traj_loss(z, self.rng.choice(train_ends, self.batch_size), offs)
        steps = _fit_early_stopping(torch, self.net, self.opt, n_steps, val_fn, step_fn)
        self.fit_log.append(dict(t=t, n_train=len(train_ends), n_val=len(val_ends),
                                 early_stopping=len(val_ends) > 0, steps=steps))

    def predict(self, history):
        torch = self.torch
        with torch.no_grad():
            x = torch.tensor([[(history[-1] - self.mu) / self.sd]], dtype=torch.float32, device=self.device)
            return float(self.net(x, self.t_span)[1][-1, 0, 0]) * self.sd + self.mu


class NHITSExpert(BaseExpert):
    """N-HITS (NeuralForecast module) trained with Adam (coupled L2 weight decay), gradient clipping, and
    early stopping on the chronologically held-out final 20% of observed windows up to slot t.
    Warm-started at each refit. Input length = context_len (120 steps for full history)."""
    name = "NHITS"

    def __init__(self, mlp_width=64, lr=5e-4, weight_decay=1e-3, batch_size=64,
                 first_fit_steps=400, fit_steps=120, device=None, **kw):
        super().__init__(**kw)
        import torch
        from neuralforecast.models import NHITS
        torch.manual_seed(self.seed)
        self.torch = torch
        self.device = resolve_device(device)
        self.input_size = self.context_len or 120
        self.net = NHITS(h=self.horizon, input_size=self.input_size, max_steps=1,
                         mlp_units=3 * [[mlp_width, mlp_width]]).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size, self.first_fit_steps, self.fit_steps = batch_size, first_fit_steps, fit_steps
        self.rng = np.random.default_rng(self.seed)

    def _run(self, Xn):
        torch = self.torch
        x = torch.as_tensor(Xn, dtype=torch.float32, device=self.device).unsqueeze(-1)   # (B, L, 1)
        batch = {"insample_y": x, "insample_mask": torch.ones_like(x),
                 "futr_exog": None, "hist_exog": None, "stat_exog": None}
        out = self.net(batch)
        return out.reshape(out.shape[0], self.horizon, -1)[:, :, 0]                     # (B, h)

    def fit(self, history):
        torch = self.torch
        t, h = len(history) - 1, self.horizon
        pool = ft.eligible_ends(self.valid, t, h, need_path=True, observed=self.observed)
        if len(pool) == 0:
            self.fit_log.append(dict(t=t, n_train=0, n_val=0, early_stopping=False, steps=0))
            return
        train_ends, val_ends = chronological_val_split(pool, h)
        n_steps = self.first_fit_steps if not self.fit_seconds else self.fit_steps
        if len(val_ends) == 0:
            n_steps = min(n_steps, 40)
            val_fn = None
        else:
            val_sub = val_ends if len(val_ends) <= 256 else val_ends[-256:]
            Xv, Yv, _ = ft.neural_windows(history, val_sub, self.input_size, h)
            Yv_t = torch.as_tensor(Yv, dtype=torch.float32, device=self.device)
            Mv_t = torch.as_tensor(ft.target_mask(self.observed, val_sub, h), device=self.device)
            val_fn = lambda: float(_masked_mse(torch, self._run(Xv), Yv_t, Mv_t))

        def step_fn():
            ends = self.rng.choice(train_ends, self.batch_size)
            Xn, Yn, _ = ft.neural_windows(history, ends, self.input_size, h)
            M = torch.as_tensor(ft.target_mask(self.observed, ends, h), device=self.device)
            return _masked_mse(torch, self._run(Xn),
                               torch.as_tensor(Yn, dtype=torch.float32, device=self.device), M)

        steps = _fit_early_stopping(torch, self.net, self.opt, n_steps, val_fn, step_fn)
        self.fit_log.append(dict(t=t, n_train=len(train_ends), n_val=len(val_ends),
                                 early_stopping=len(val_ends) > 0, steps=steps))

    def predict(self, history):
        Xn, base = ft.neural_windows(history, np.array([len(history) - 1]), self.input_size)
        with self.torch.no_grad():
            return float(self._run(Xn)[0, -1]) * ft.NEURAL_SCALE + float(base[0])


EXPERT_CLASSES = {
    "ARIMA": lambda **kw: StatsForecaster(model_type="arima", **kw),
    "ETS": lambda **kw: StatsForecaster(model_type="ets", **kw),
    "NHITS": lambda **kw: NHITSExpert(**kw),
    "NODE": lambda **kw: NODEExpert(**kw),
    "XGBoost": lambda **kw: XGBoostExpert(**kw),
    "Persistence": lambda **kw: Persistence(**kw),
    "RunningMean": lambda **kw: RunningMean(**kw),
}


def build_experts(names, horizon, context_len, retrain_interval, seed=0):
    return {n: EXPERT_CLASSES[n](horizon=horizon, context_len=context_len,
                                 retrain_interval=retrain_interval, seed=seed) for n in names}
