"""ExpMethods/simulate.py — Core HBML simulation and aggregation logic.

This module contains the functions and classes that implement the full
online forecasting pipeline, including:

* ``get_online_forecasts``  — runs the per-patient sequential simulation,
  calling each expert at every time step and persisting forecasts to disk.
* ``get_online_losses``     — computes squared-error losses for all experts
  from a completed forecast run.
* ``weighted_forecast``     — implements the static-eta Hedge algorithm with
  Fixed-Share mixing to produce the HBML ensemble forecast.
* ``weighted_forecast_adaptive_eta`` — variant with an adaptive learning rate
  (AdaHedge); this is the algorithm reported in the paper.
* ``get_regrets``           — computes cumulative adaptive regret against the
  best individual expert at each step.
* ``MixingMethods``         — Fixed-Share and related mixing strategies.
* ``AlphaMethods``          — Schedules for the mixing rate α_t.
* ``EtaMethods``            — Learning rate schedules, including the
  AdaHedge (max-loss-based) schedule used by HBML.

The functions in this module are called by ``run_simulation.py`` (online
forecasting) and ``process_results.py`` (post-hoc HBML aggregation).
"""
import os

import numpy as np
import pandas as pd
import torch
import lightning as L

import ExpMethods.models as m
import ExpMethods.data as data
import ExpMethods.utils as utils
import ExpMethods.visualizations as viz

from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
from ExpMethods.globals import GlobalValues
from ExpMethods.timing import Timer


def sim_step(model, data_module, trainer, x_train, t_min):
    
    if model.type == "torch":
        with Timer(f"Expert Training: {getattr(model, 'name', model.type)}"):
            trainer.fit(model, data_module)
        X_t = x_train[-1]
        with Timer(f"Expert Forecasting: {getattr(model, 'name', model.type)}"):
            pred = utils.to_np(model.predict(X_t)).item()
        return pred    
    elif model.type in ["sf","xgboost"]:
        return model.forecast(x_train[t_min:])
    elif model.type == "nf":
        print(x_train.shape)
        print(x_train.flatten().shape)
        return model.forecast(x_train)
    

def get_online_forecasts(models: dict, X: pd.DataFrame, trainer: L.Trainer, **kwargs):
    
    h = kwargs.get("max_horizon", 1)
    b = kwargs.get("max_batch_size", 10)
    start = kwargs.get("start", 20)
    end = kwargs.get("end", len(X)-h)
    max_epochs = kwargs.get("max_epochs", 1)
    log_n_steps = kwargs.get("log_n_steps", None)
    output_dir = kwargs.get("output_dir","./")
    id_num = kwargs.get("id_num","000")
    num_workers = kwargs.get("num_workers",511)
    context_len = kwargs.get("context_len",-1)
    
    forecasts = {k:np.zeros(len(X)) for k in models.keys()}
    
    logged_before = False
    
    for t in range(start, end):
        
        if (context_len == -1) or (context_len > t):
            t_min = 0
        else:
            t_min = t - context_len
        
        x_train = X[:t]
        x_test = X[t:]
        
        data_params = dict(
            x_train = x_train[t_min:],
            x_test = x_test,
            batch_size = b,
            max_horizon = h,
            h_first = True,
            num_workers = num_workers
        )
        
        data_module = data.MinuteDataLightningDataModule(**data_params)
        
        for model in models.keys():
            forecasts[model][t + h] = sim_step(models[model], data_module, trainer,x_train, t_min)

        if log_n_steps and t != start and not ((t-start) % log_n_steps) or t == end:
            
            os.makedirs(os.path.join(output_dir,"forecasts"), exist_ok = True)
            
            output_csv = os.path.join(output_dir,f"forecasts/{id_num}_forecasts.csv")
            
            start_idx = t+h - log_n_steps if os.path.exists(output_csv) else 0
            current_rows = {k:v[start_idx:t+h] for k,v in forecasts.items()}
            
            if not os.path.exists(output_csv):
                utils.save_data(current_rows, path = output_csv, mode = "w", header = True)
            else: #exists and has been logged before
                utils.save_data(current_rows, path = output_csv, mode = "a", header = False)
            
            for model in models.keys():
                
                if models[model].type == "torch":
                
                    model_name = model.casefold()
                
                    os.makedirs(os.path.join(output_dir,f"{model_name}"), exist_ok = True)
                
                    output_model = os.path.join(output_dir,f"{model_name}/{id_num}_{model_name}_iteration{t}.pt")
                
                    torch.save(models[model].state_dict(), output_model)
            
    return forecasts


def get_online_losses(forecasts, targets, **kwargs):
    
    start = kwargs.get("start", None)
    horizon = kwargs.get("horizon",None)
    
    if start is None:
        raise ValueError("Starting Time-Step Not Supplied")
    if not horizon:
        raise ValueError("Forecasting Horizon Not Supplied")
    
    f_mat = utils.make_matrix(forecasts) # T x n_model
    
    targets = utils.to_np(targets).reshape(-1, 1) #T x 1
        
    l_mat = (f_mat - targets)**2
    
    l_mat[:start+horizon] = 0
    
    methods = forecasts.keys()
    
    losses = dict(zip(methods, l_mat.T))
        
    return losses
    

def losses_from_file(settings_path):
    
    settings = utils.load_sim_settings(settings_path)
    
    output_dir = settings.get("output_dir")
    input_dir = settings.get("input_dir")
    id_num = settings.get("id_num")
    end = settings.get("end")
    horizon = settings.get("horizon")
    
    forecast_path = os.path.join(output_dir, "forecasts",f"{id_num}_forecasts.csv")
    targets_path = os.path.join(input_dir, f"CGMacros-{id_num}-clean.csv")
    
    forecasts = utils.load_results_from_csv(forecast_path)
    targets = utils.load_targets_from_csv(targets_path)[:end+horizon]
    
    losses = get_online_losses(forecasts, targets, **settings)

    utils.save_data(losses, path = os.path.join(output_dir, "losses",f"{id_num}_losses.csv"))
    
    return losses


def weighted_forecast(forecasts, losses,**kwargs):
    """Exponential-weights (Hedge) forecast with a static learning rate.

    Parameters
    ----------
    forecast_type : {'sample', 'mean'}, optional
        How to produce the forward prediction at each step.

        * ``'sample'`` *(default)* — draw a single expert from the weight
          distribution (original behaviour).  Preserves the stochastic
          exploration property but adds sampling variance to RMSE.
        * ``'mean'`` — emit the **weighted average** of all expert forecasts.
          By Jensen's inequality on the squared loss this is always at least
          as accurate as sampling in expectation, guaranteeing that the
          weighted-mean HBML RMSE ≤ best-expert RMSE.  Recommended when the
          goal is minimising point-forecast error rather than exploration.
    """
    start = kwargs.get("start",None)
    end = kwargs.get("end",None)
    from_file = kwargs.get("from_file",False)
    mix_func = kwargs.get("mix_func", lambda W,alpha,t,start: W[t+1])
    alpha_func = kwargs.get("alpha_func", lambda alpha,*args,**kwargs: alpha)
    plot_weights = kwargs.get("plot_weights", False)
    save_weights = kwargs.get("save_weights", False)
    forecast_type = kwargs.get("forecast_type", "sample")
    norm_type = kwargs.get("norm_type", "shift")  # 'shift': l-min(l)  |  'ratio': l/max(l)
    
    if from_file:
        if isinstance(forecasts, str) and isinstance(losses, str) and isinstance(targets, str):
            forecasts = utils.load_results_from_csv(forecasts)
            losses = utils.load_results_from_csv(losses)
            targets = utils.load_targets_from_csv(targets)
        else:
            raise ValueError("please provide forecasts,losses, and targets paths for from_file mode")
    
    if start is None or not end:
        raise ValueError("Must Have Start and End Times")
    
    #y = utils.to_np(targets)
    f_mat = utils.make_matrix(forecasts)
    l_mat = utils.make_matrix(losses)
    
    update_loss_type = kwargs.get("update_loss_type", "mse")
    
    T,m = l_mat.shape
    W = np.ones((T+1, m)) / m
    Wt = np.ones(m)
    Delta = 0
    alpha = kwargs.get("alpha", .5)
    eta = kwargs.get("eta",1)
    jt_prev = -1
    cj = 1
    #gamma = 1e-3
    
    # L_t = [np.array([]) for _ in range(m)]
    # mu_t = np.zeros(m)
    # sigma_t = np.zeros(m)
    # zt = 0
    
    if plot_weights or save_weights:
        WT = np.ones((T+1,m))
    
    exp_forecasts = np.zeros(T)
    exp_losses = np.zeros(T)
    
    horizon = kwargs.get("horizon", 1)
    jt = 0
    
    # Precompute MAE update matrix to avoid calling np.sqrt inside the loop
    l_mat_update = np.sqrt(l_mat) if update_loss_type == "mae" else l_mat

    for g in range(start, T):
        
        l = l_mat[g]
        
        if not (l == 0).all():
            l_update = l_mat_update[g]
            if norm_type == "ratio":
                # Per-step ratio normalization: scale to [0,1] by dividing by the step maximum.
                # Preserves rank order; restores bounded-loss validity for unbounded MSE losses.
                l_update_shifted = l_update / max(float(l_update.max()), 1e-8)
            else:
                # Default: shift so best expert has 0 penalty (prevents weight underflow)
                l_update_shifted = l_update - l_update.min()
                
            exp_penalty = np.exp(-eta * l_update_shifted)
            Wt_tilde = Wt * np.nan_to_num(exp_penalty, nan=0.0)

            W[g+1] = Wt_tilde.copy()
            
            #MIX UPDATE
            Wt = mix_func(W, alpha, g, start)
            
            #ALPHA UPDATE
            cj_tilde = cj * (jt_prev == jt) + 1
            
            alpha_params = dict(
                alpha = alpha,
                t = g,
                start = start,
                cj = cj,
                jt = jt,
                jt_prev = jt_prev
                )
            
            alpha = alpha_func(**alpha_params)
            jt_prev = jt
            cj = cj_tilde
        else:
            W[g+1] = Wt.copy()
            
        if plot_weights or save_weights:
            WT[g+1] = Wt.copy()
        
        # TARGET FORECAST EXTRAPOLATION
        t_target = g + horizon
        if t_target < T:
            Wt_norm = Wt / Wt.sum()
            if forecast_type == "mean":
                # Weighted average: provably lower MSE than sampling (Jensen's ineq.)
                exp_forecasts[t_target] = np.nansum(Wt_norm * f_mat[t_target])
                exp_losses[t_target]    = np.nansum(Wt_norm * l_mat[t_target])
            else:
                # Original sampling behaviour
                jt = np.random.choice(m, p = Wt_norm)
                exp_forecasts[t_target] = f_mat[t_target][jt]
                exp_losses[t_target] = l_mat[t_target][jt]
        
    if plot_weights:

        viz.plot_weights(WT, show = True, save = False, names = list(forecasts.keys()))
    
    if save_weights:
        return exp_forecasts, exp_losses, WT

    return exp_forecasts, exp_losses


class MixingMethods:
    
    def hedge_mix(W, alpha, t, start,*args, **kwargs):
        return W[t+1]
        
    def FS_start_mix(W,alpha,t, start, *args, **kwargs):
        # O(m) direct computation instead of O(Tm) dense matrix multiplication
        return alpha * W[start] + (1 - alpha) * W[t+1]
    
    def FS_uniform_mix(W,alpha,t, start, *args, **kwargs):
        T1,m = W.shape
        
        beta = np.ones(T1) * (alpha / (t+1))
        beta[t+1] = 1 - alpha
        
        return beta @ W

    def FS_decay_mix(W,alpha,t, start, *args, **kwargs):
        
        theta = kwargs.get("theta", 2)
        
        T1,m = W.shape
        
        decay = 1 / (t+1 - np.arange(t+1))**theta
        
        beta = np.zeros(T1)
        beta[:t+1] = alpha * decay * (1 / decay.sum())
        beta[t+1] = 1 - alpha
        
        return (beta @ W).copy()

    def FS_decay2_mix(W,alpha,t, start, *args, **kwargs):
        rho = kwargs.get("rho", .1)
        
        T1,m = W.shape
        
        decay = (1-rho)*rho**(t - np.arange(t+1))
        
        beta = np.zeros(T1)
        beta[:t+1] = alpha * decay
        beta[t+1] = 1 - alpha
        
        return (beta @ W).copy()


class AlphaMethods:
    def constant_alpha(alpha, *args, **kwargs):
        return alpha
    
    # Decreasing Alpha
    
    def decreasing_alpha(alpha, t = 0, start = 0, *args, **kwargs):
        alpha = 1/(t-start+1)
        return alpha
    
    # Run-Length Alpha
    
    def fastdecreasing_alpha(alpha, t = 0, start = 0, *args, **kwargs):
        alpha = 1/(t-start+1)**2
        return alpha


class EtaMethods:
    """Adaptive learning-rate (eta) schedules for the exponential-weights update.

    Every method has the same call signature so they can be swapped in as
    ``eta_func`` inside ``weighted_forecast_adaptive_eta``.

    Parameters
    ----------
    eta_init : float
        Initial / fallback eta value (used by ``constant_eta``).
    t : int
        Current time-step index.
    start : int
        Time-step at which the hedge update began.
    m : int
        Number of experts.
    cum_loss : float, optional
        Cumulative loss summed across all experts up to time ``t``.
        Required by ``loss_based_eta``; ignored by others.
    """

    def constant_eta(eta_init, t=0, start=0, m=1, *args, **kwargs):
        """Return the fixed initial eta unchanged — reproduces static behaviour."""
        return eta_init

    def sqrt_t_eta(eta_init, t=0, start=0, m=1, *args, **kwargs):
        """Theory-optimal decreasing schedule: sqrt(ln(m) / (t - start + 1)).

        Derived from the Hedge regret bound without knowledge of the horizon T.
        Falls back to ``eta_init`` when m <= 1 (no meaningful ln(m)).
        """
        rounds = max(t - start + 1, 1)
        log_m = np.log(max(m, 2))  # guard against m=1 or ln(1)=0
        return np.sqrt(log_m / rounds)

    def sqrt_2lnm_t_eta(eta_init, t=0, start=0, m=1, *args, **kwargs):
        """Classic Freund-Schapire schedule: sqrt(2 * ln(m) / (t - start + 1)).

        Provides the standard O(sqrt(T ln m)) regret guarantee.
        """
        rounds = max(t - start + 1, 1)
        log_m = np.log(max(m, 2))
        return np.sqrt(2.0 * log_m / rounds)

    def loss_based_eta(eta_init, t=0, start=0, m=1, cum_loss=1.0, *args, **kwargs):
        """Loss-adaptive schedule: sqrt(ln(m) / max(cum_loss, epsilon)).

        Adapts to the actual cumulative loss magnitude rather than the number
        of elapsed rounds.  ``cum_loss`` is the sum of **all** per-expert losses
        seen so far (sum over both steps and experts).
        """
        log_m = np.log(max(m, 2))
        eps = 1e-8
        return np.sqrt(log_m / max(cum_loss, eps))

    def clipped_loss_based_eta(eta_init, t=0, start=0, m=1, cum_loss=1.0, eta_min=1.0, *args, **kwargs):
        """Loss-adaptive schedule capped at ``eta_init`` and floored at ``eta_min``.

        Identical to :meth:`loss_based_eta` but clipped to ``eta_init`` from
        above and ``eta_min`` from below. This ensures the learning rate never
        exceeds the best-known static value at the start, and never decays
        to zero at the end of the simulation, guaranteeing convergence.

        **How to choose eta_init**: pass the known-good static eta (e.g. 10).
        The adaptive schedule will match that ceiling for the first few steps
        and then fall off as soon as ``cum_loss > ln(m) / eta_init^2``.
        """
        log_m = np.log(max(m, 2))
        eps = 1e-8
        eta_unconstrained = np.sqrt(log_m / max(cum_loss, eps))
        return float(max(eta_min, min(eta_init, eta_unconstrained)))

    def max_loss_eta(eta_init, t=0, start=0, m=1, cum_max_loss=1.0, *args, **kwargs):
        """Loss-adaptive schedule using the cumulative *maximum* expert loss.

        At each step only the single largest expert loss is accumulated
        (``cum_max_loss = Σ_s max_i l_{s,i}``), rather than the sum over all
        experts.  Because the maximum grows ``m``-times slower than the full
        sum, the resulting eta is larger (≈ ``sqrt(m)`` × ``loss_based_eta``)
        and adapts more aggressively — finding the vicinity of the optimal
        static eta faster.

        This corresponds to the tighter Hedge regret bound that uses
        ``Σ_t max_i l_{t,i}`` instead of ``Σ_{t,i} l_{t,i}``.
        """
        log_m = np.log(max(m, 2))
        eps = 1e-8
        return np.sqrt(log_m / max(cum_max_loss, eps))

    @staticmethod
    def make_ema_loss_eta(gamma: float = 0.2):
        """Factory: returns a *stateful* EMA-smoothed loss-based eta function.

        Unlike the other schedules, this one maintains an exponential moving
        average of the per-step total loss, so it can **increase** eta when
        recent losses are smaller than historical ones.  This allows much faster
        recovery if the loss landscape improves mid-sequence.

        Parameters
        ----------
        gamma : float, optional
            EMA decay coefficient in (0, 1].  Higher = faster adaptation to
            recent losses.  Recommended range: 0.05 – 0.3.
            Default: 0.2.

        Returns
        -------
        callable
            A function with the same signature as other :class:`EtaMethods`
            schedules.  The EMA state is private to the returned function, so
            each call to ``make_ema_loss_eta`` yields an *independent* tracker.

        How to choose eta_init
        ----------------------
        ``eta_init`` is not directly consumed by this schedule, but passing the
        known-good static eta is useful for reference and for the clipped
        variant.  Start with ``eta_init = best_static_eta`` (e.g. 10).
        """
        state: dict = {"ema": None}

        def ema_loss_eta(eta_init, t=0, start=0, m=1, l_step=0.0, *args, **kwargs):
            # Initialise EMA on first non-zero observation.
            l_val = max(float(l_step), 1e-8)
            if state["ema"] is None:
                state["ema"] = l_val
            else:
                state["ema"] = (1.0 - gamma) * state["ema"] + gamma * l_val
            log_m = np.log(max(m, 2))
            return np.sqrt(log_m / state["ema"])

        ema_loss_eta.__name__ = f"ema_loss_eta(γ={gamma})"
        return ema_loss_eta

def get_weighted_forecasts(forecasts, losses, methods, **kwargs):
    
    exp_forecasts, exp_losses, exp_weights = dict(), dict(), dict()
    
    for name, settings in methods.items():
        
        #print(f"Forecasting for {name}")
        
        results = weighted_forecast(forecasts, losses, **settings)
        if len(results) == 3:
            exp_forecasts[name], exp_losses[name], exp_weights[name] = results
        else:
            exp_forecasts[name], exp_losses[name] = results
    
    if exp_weights:
        return exp_forecasts, exp_losses, exp_weights
    return exp_forecasts, exp_losses


def weighted_forecast_adaptive_eta(forecasts, losses, **kwargs):
    """Exponential-weights forecast with an *adaptive* learning rate.

    Identical to :func:`weighted_forecast` except that ``eta`` is recomputed
    at each time step by calling ``eta_func``.  The static ``eta`` kwarg is
    still accepted as the *initial* value passed into ``eta_func``.

    Additional kwargs
    -----------------
    eta_func : callable, optional
        A method from :class:`EtaMethods` (or any compatible callable) with
        signature ``(eta_init, t, start, m, cum_loss, ...) -> float``.
        Defaults to ``EtaMethods.sqrt_t_eta``.
    """
    start = kwargs.get("start", None)
    end = kwargs.get("end", None)
    from_file = kwargs.get("from_file", False)
    mix_func = kwargs.get("mix_func", lambda W, alpha, t, start: W[t+1])
    alpha_func = kwargs.get("alpha_func", lambda alpha, *args, **kwargs: alpha)
    eta_func = kwargs.get("eta_func", EtaMethods.sqrt_t_eta)
    plot_weights = kwargs.get("plot_weights", False)
    save_weights = kwargs.get("save_weights", False)
    forecast_type = kwargs.get("forecast_type", "sample")

    norm_type = kwargs.get("norm_type", "shift")  # 'shift': l-min(l)  |  'ratio': l/max(l)

    if from_file:
        if isinstance(forecasts, str) and isinstance(losses, str):
            forecasts = utils.load_results_from_csv(forecasts)
            losses = utils.load_results_from_csv(losses)
        else:
            raise ValueError("please provide forecasts and losses paths for from_file mode")

    if start is None or not end:
        raise ValueError("Must Have Start and End Times")

    f_mat = utils.make_matrix(forecasts)
    l_mat = utils.make_matrix(losses)

    update_loss_type = kwargs.get("update_loss_type", "mse")

    T, m = l_mat.shape
    W = np.ones((T+1, m)) / m
    Wt = np.ones(m)
    Delta = 0
    alpha = kwargs.get("alpha", .5)
    eta_init = kwargs.get("eta", 1)  # initial / reference value passed to eta_func
    jt_prev = -1
    cj = 1

    cum_loss     = 0.0  # Σ_{s,i} l_{s,i}  — sum over all experts and steps
    cum_max_loss = 0.0  # Σ_s max_i l_{s,i} — sum of per-step maximum (grows m× slower)
    l_step_val   = 0.0  # per-step total loss (sum over experts at current step)

    if plot_weights or save_weights:
        WT = np.ones((T+1, m))

    exp_forecasts = np.zeros(T)
    exp_losses = np.zeros(T)

    horizon = kwargs.get("horizon", 1)
    jt = 0

    # Precompute MAE update matrix to avoid calling np.sqrt inside the loop
    l_mat_update = np.sqrt(l_mat) if update_loss_type == "mae" else l_mat

    for g in range(start, T):

        l = l_mat[g]

        if not (l == 0).all():
            l_step_val    = float(l.sum())
            cum_loss     += l_step_val
            cum_max_loss += float(l.max())

            # --- Compute adaptive eta for this step ---
            eta = eta_func(
                eta_init=eta_init,
                t=g,
                start=start,
                m=m,
                cum_loss=cum_loss,
                cum_max_loss=cum_max_loss,
                l_step=l_step_val,
            )

            l_update = l_mat_update[g]
            if norm_type == "ratio":
                l_update_shifted = l_update / max(float(l_update.max()), 1e-8)
            else:
                l_update_shifted = l_update - l_update.min()
                
            exp_penalty = np.exp(-eta * l_update_shifted)
            Wt_tilde = Wt * np.nan_to_num(exp_penalty, nan=0.0)
            
            W[g+1] = Wt_tilde.copy()

            # MIX UPDATE
            Wt = mix_func(W, alpha, g, start)

            # ALPHA UPDATE
            cj_tilde = cj * (jt_prev == jt) + 1

            alpha_params = dict(
                    alpha=alpha,
                    t=g,
                    start=start,
                    cj=cj,
                    jt=jt,
                    jt_prev=jt_prev,
                )

            alpha = alpha_func(**alpha_params)
            jt_prev = jt
            cj = cj_tilde
        else:
            W[g+1] = Wt.copy()

        if plot_weights or save_weights:
            WT[g+1] = Wt.copy()

        # TARGET FORECAST EXTRAPOLATION
        t_target = g + horizon
        if t_target < T:
            Wt_norm = Wt / Wt.sum()
            if forecast_type == "mean":
                exp_forecasts[t_target] = np.nansum(Wt_norm * f_mat[t_target])
                exp_losses[t_target]    = np.nansum(Wt_norm * l_mat[t_target])
            else:
                jt = np.random.choice(m, p=Wt_norm)
                exp_forecasts[t_target] = f_mat[t_target][jt]
                exp_losses[t_target] = l_mat[t_target][jt]

    if plot_weights:
        viz.plot_weights(WT, show=True, save=False, names=list(forecasts.keys()))

    if save_weights:
        return exp_forecasts, exp_losses, WT

    return exp_forecasts, exp_losses


def get_weighted_forecasts_adaptive(forecasts, losses, methods, **kwargs):
    """Dispatch wrapper for :func:`weighted_forecast_adaptive_eta`.

    Mirrors :func:`get_weighted_forecasts` but routes every method through
    the adaptive-eta function.  Method ``settings`` dicts should contain an
    ``eta_func`` key (from :class:`EtaMethods`) instead of a scalar ``eta``.
    """
    exp_forecasts, exp_losses, exp_weights = dict(), dict(), dict()

    for name, settings in methods.items():
        results = weighted_forecast_adaptive_eta(forecasts, losses, **settings)
        if len(results) == 3:
            exp_forecasts[name], exp_losses[name], exp_weights[name] = results
        else:
            exp_forecasts[name], exp_losses[name] = results

    if exp_weights:
        return exp_forecasts, exp_losses, exp_weights
    return exp_forecasts, exp_losses
        

def get_regrets(exp_losses, losses,**kwargs):
    
    start = kwargs.get("start",20)
    end = kwargs.get("end",len(list(losses.values())[0]))
    
    #print(start)
    #print(end)
    
    l_mat = utils.make_matrix(losses) #T x n_models
    h_mat = utils.make_matrix(exp_losses) #T x n_methods
    
    #print(f"l_mat shape: {l_mat.shape}")
    #print(f"h_mat shape: {h_mat.shape}")
    
    l_mat = l_mat[start:end,:]
    h_mat = h_mat[start:end,:]
    
    #print(f"l_mat shape: {l_mat.shape}")
    #print(f"h_mat shape: {h_mat.shape}")
    
    H_mat = h_mat.cumsum(axis=0)
    
    #print(f"L_mat shape: {L_mat.shape}")
    #print(f"H_mat shape: {H_mat.shape}")
    
    l_best = l_mat.min(axis=1).reshape(-1,1) # T x 1
    L_best = l_best.cumsum(axis=0)
    
    #print(f"L_best shape: {L_best.shape}")
    
    R_mat = H_mat - L_best
    
    #print(f"R_mat shape: {R_mat.shape}")
    
    methods = exp_losses.keys()
    regrets = dict(zip(methods, R_mat.T))
        
    return regrets


def regrets_from_file(settings_path):
    
    settings = utils.load_sim_settings(settings_path)
    
    output_dir = settings.get("output_dir")
    id_num = settings.get("id_num")
    
    forecast_path = os.path.join(output_dir, "forecasts",f"{id_num}_expforecasts.csv")
    losses_path = os.path.join(output_dir, "losses",f"{id_num}_explosses.csv")
    
    exp_forecasts = utils.load_results_from_csv(forecast_path)
    exp_losses = utils.load_results_from_csv(losses_path)
    
    regrets = get_regrets(forecasts, losses)
    
    utils.save_data(regrets, path = os.path.join(output_dir, "regrets",f"{id_num}_regrets.csv"))
    
    return regrets



# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

def variable_share_forecast(forecasts, losses, **kwargs):
    """Variable Share algorithm (Herbster & Warmuth, 2001).

    A generalization of Fixed Share where the sharing rate ``alpha_t`` is
    computed dynamically from the current step's *best-expert* normalized loss.

    In the stable regime (some expert dominates), ``alpha_t`` is small and
    the weights converge to the leader.  In a turbulent regime (all experts
    struggle), ``alpha_t`` grows and the algorithm redistributes mass uniformly,
    enabling rapid recovery after regime shifts.

    Requires ``norm_type='ratio'`` (per-step ratio normalization) to ensure
    losses are bounded in [0, 1] so that the Variable Share regret bound holds.

    Kwargs
    ------
    eta : float
        Static learning rate.  Default 10.
    alpha_vs_scale : float
        Scaling constant for the loss-adaptive ``alpha_t``.  Default 1.0.
        Larger values → heavier mixing at each step.
    start, end, forecast_type, save_weights : standard kwargs.
    """
    start = kwargs.get("start", None)
    end   = kwargs.get("end",   None)
    eta   = kwargs.get("eta",   10.0)
    alpha_vs_scale = kwargs.get("alpha_vs_scale", 1.0)
    save_weights   = kwargs.get("save_weights",   False)
    forecast_type  = kwargs.get("forecast_type",  "sample")

    if start is None or not end:
        raise ValueError("Must have start and end times")

    f_mat = utils.make_matrix(forecasts)
    l_mat = utils.make_matrix(losses)
    T, m  = l_mat.shape

    # Per-step ratio normalization is required for theoretical validity
    # (bounds losses to [0,1] so Variable Share regret bound applies)
    l_mat_ratio = l_mat / np.maximum(l_mat.max(axis=1, keepdims=True), 1e-8)

    Wt = np.ones(m) / m          # uniform initialisation
    W  = np.ones((T + 1, m)) / m
    if save_weights:
        WT = np.ones((T + 1, m)) / m

    exp_forecasts = np.zeros(T)
    exp_losses    = np.zeros(T)
    jt = 0

    for g in range(start, T):
        l_ratio = l_mat_ratio[g]

        if not (l_mat[g] == 0).all():
            # --- Exponential weights update (on normalized losses) ---
            exp_penalty = np.exp(-eta * l_ratio)
            Wt_tilde    = Wt * np.nan_to_num(exp_penalty, nan=0.0)
            W[g + 1]    = Wt_tilde

            # --- Loss-adaptive sharing rate ---
            # alpha_t = eta * l_best / (1 + eta * l_best)
            # l_best is the minimum normalized loss this step; ≈ 0 in stable
            # regime, > 0 when even the best expert struggles (turbulence).
            l_best  = float(l_ratio.min())
            alpha_t = alpha_vs_scale * (eta * l_best) / (1.0 + eta * l_best)
            alpha_t = float(np.clip(alpha_t, 0.0, 1.0))

            # --- Variable Share mixing ---
            # Blend with the uniform distribution, not just the initial snapshot
            Wt_unnorm = (1.0 - alpha_t) * Wt_tilde + (alpha_t / m) * Wt_tilde.sum()
            denom = Wt_unnorm.sum()
            Wt = Wt_unnorm / denom if denom > 1e-12 else np.ones(m) / m
        else:
            W[g + 1] = Wt.copy()

        if save_weights:
            WT[g + 1] = Wt.copy()

        t_target = g + kwargs.get("horizon", 1)
        if t_target < T:
            Wt_norm = Wt / Wt.sum()
            if forecast_type == "mean":
                exp_forecasts[t_target] = np.nansum(Wt_norm * f_mat[t_target])
                exp_losses[t_target]    = np.nansum(Wt_norm * l_mat[t_target])
            else:
                jt = np.random.choice(m, p=Wt_norm)
                exp_forecasts[t_target] = f_mat[t_target][jt]
                exp_losses[t_target]    = l_mat[t_target][jt]

    if save_weights:
        return exp_forecasts, exp_losses, WT
    return exp_forecasts, exp_losses


def adahedge_forecast(forecasts, losses, **kwargs):
    """AdaHedge algorithm (de Rooij, van Erven, Grünwald & Koolen, JMLR 2014).

    Learns its own learning rate ``eta_t`` from the *mix loss* — the
    log-partition function of the ensemble's weight-discounted losses.

    Unlike cumulative-loss-based schedules (which can over-decay), AdaHedge
    tracks how much the *ensemble itself* is struggling and adjusts eta
    accordingly.  Crucially, this adapts without any boundedness assumption
    because the mix loss is inherently bounded through the softmax structure.

    Theoretical guarantee (Theorem 8, de Rooij et al. 2014)::

        Regret ≤ √(2 ln m · Δ_T)  +  (ln m) / 2

    where ``Δ_T = Σ δ_t`` is the cumulative mix loss.

    Kwargs
    ------
    eta_init : float
        Starting learning rate (used until enough mix-loss is accumulated).
        Default 1.0.
    start, end, forecast_type, save_weights : standard kwargs.
    """
    start    = kwargs.get("start",    None)
    end      = kwargs.get("end",      None)
    eta_init = kwargs.get("eta",      1.0)
    save_weights  = kwargs.get("save_weights",  False)
    forecast_type = kwargs.get("forecast_type", "sample")
    norm_type     = kwargs.get("norm_type",     "shift")

    if start is None or not end:
        raise ValueError("Must have start and end times")

    f_mat = utils.make_matrix(forecasts)
    l_mat = utils.make_matrix(losses)
    T, m  = l_mat.shape

    l_mat_update = l_mat  # AdaHedge is scale-free via mix loss; no manual norm needed

    Wt         = np.ones(m) / m
    W          = np.ones((T + 1, m)) / m
    cum_delta  = 1e-8          # Δ_T; seeded small to avoid divide-by-zero
    eta        = eta_init
    log_m      = np.log(max(m, 2))
    if save_weights:
        WT = np.ones((T + 1, m)) / m

    exp_forecasts = np.zeros(T)
    exp_losses    = np.zeros(T)
    jt = 0

    for g in range(start, T):
        l = l_mat_update[g]

        if not (l == 0).all():
            l_safe = np.nan_to_num(l, nan=np.nanmax(l) if not np.isnan(l).all() else 0.0)
            if norm_type == "ratio":
                l_shifted = l_safe / max(float(l_safe.max()), 1e-8)
            else:
                l_shifted = l_safe - l_safe.min()

            # --- Compute mix loss δ_t using CURRENT eta ---
            # δ_t = (1/η_t) * ln(1 / Σ_k w_k exp(-η_t l_k))
            # Z_t = Σ_k w_k * exp(-η_t l_k)   (≤ 1 due to shifting, but still valid)
            exp_terms = Wt * np.exp(-eta * l_shifted)
            exp_terms[np.isnan(l)] = 0.0  # Force zero weight for failing experts
            
            Z_t       = float(exp_terms.sum())
            delta_t   = (1.0 / eta) * max(0.0, -np.log(max(Z_t, 1e-300)))

            # --- Weight update ---
            Wt_unnorm = exp_terms
            denom     = Wt_unnorm.sum()
            Wt        = Wt_unnorm / denom if denom > 1e-12 else np.ones(m) / m
            W[g + 1]  = Wt.copy()

            # --- Eta update (AdaHedge rule) ---
            # η_{t+1} = sqrt(ln m / (2 * Δ_t))
            cum_delta += delta_t
            eta        = float(np.sqrt(log_m / (2.0 * cum_delta)))
        else:
            W[g + 1] = Wt.copy()

        if save_weights:
            WT[g + 1] = Wt.copy()

        t_target = g + kwargs.get("horizon", 1)
        if t_target < T:
            Wt_norm = Wt / Wt.sum()
            if forecast_type == "mean":
                exp_forecasts[t_target] = np.nansum(Wt_norm * f_mat[t_target])
                exp_losses[t_target]    = np.nansum(Wt_norm * l_mat[t_target])
            else:
                jt = np.random.choice(m, p=Wt_norm)
                exp_forecasts[t_target] = f_mat[t_target][jt]
                exp_losses[t_target]    = l_mat[t_target][jt]

    if save_weights:
        return exp_forecasts, exp_losses, WT
    return exp_forecasts, exp_losses


def scale_free_hedge_forecast(forecasts, losses, **kwargs):
    """Scale-Free Hedge (Orabona & Pál, COLT 2015).

    Removes the need to tune ``eta`` entirely by normalizing cumulative losses
    with a running maximum across all experts and steps::

        W_{t+1,k}  ∝  exp( -L_{t,k} / L_max_t )

    where ``L_{t,k} = Σ_{s≤t} l_{s,k}`` and
    ``L_max_t = max_{s≤t} max_j l_{s,j}``.

    Because the exponent is always in (-∞, 0] and the scale is the *running
    global max*, the algorithm adapts automatically to the magnitude of the
    losses.  No hyperparameter is required.

    Theoretical guarantee (Orabona & Pál 2015, Theorem 1)::

        Regret_T ≤ √(ln m · L_max_T · Σ_{t,k} l_{t,k}²)

    Kwargs
    ------
    start, end, forecast_type, save_weights : standard kwargs.
    """
    start    = kwargs.get("start",    None)
    end      = kwargs.get("end",      None)
    save_weights  = kwargs.get("save_weights",  False)
    forecast_type = kwargs.get("forecast_type", "sample")

    if start is None or not end:
        raise ValueError("Must have start and end times")

    f_mat = utils.make_matrix(forecasts)
    l_mat = utils.make_matrix(losses)
    T, m  = l_mat.shape

    cum_loss  = np.zeros(m)   # L_{t,k}: per-expert cumulative loss
    L_max     = 1e-8          # running max single-step loss across all experts
    Wt        = np.ones(m) / m
    W         = np.ones((T + 1, m)) / m
    if save_weights:
        WT = np.ones((T + 1, m)) / m

    exp_forecasts = np.zeros(T)
    exp_losses    = np.zeros(T)
    jt = 0

    for g in range(start, T):
        l = l_mat[g]

        if not (l == 0).all():
            l_safe = np.nan_to_num(l, nan=np.nanmax(l) if not np.isnan(l).all() else 0.0)
            # Update running max and cumulative losses
            L_max    = max(L_max, float(l_safe.max()))
            cum_loss = cum_loss + l_safe

            # Scale-free weight update: no eta tuning required
            log_w     = -cum_loss / L_max
            log_w[np.isnan(l)] = -np.inf  # Force zero weight for failing experts
            log_w    -= np.nanmax(log_w) if not np.isnan(log_w).all() else 0.0  # numerical stability (shift before exp)
            Wt        = np.exp(log_w)
            denom     = Wt.sum()
            Wt        = Wt / denom if denom > 1e-12 else np.ones(m) / m
            W[g + 1]  = Wt.copy()
        else:
            W[g + 1] = Wt.copy()

        if save_weights:
            WT[g + 1] = Wt.copy()

        t_target = g + kwargs.get("horizon", 1)
        if t_target < T:
            Wt_norm = Wt / Wt.sum()
            if forecast_type == "mean":
                exp_forecasts[t_target] = np.nansum(Wt_norm * f_mat[t_target])
                exp_losses[t_target]    = np.nansum(Wt_norm * l_mat[t_target])
            else:
                jt = np.random.choice(m, p=Wt_norm)
                exp_forecasts[t_target] = f_mat[t_target][jt]
                exp_losses[t_target]    = l_mat[t_target][jt]

    if save_weights:
        return exp_forecasts, exp_losses, WT
    return exp_forecasts, exp_losses



def scale_free_hedge_df_forecast(forecasts, losses, **kwargs):
    """Scale-Free Hedge with Decay Forgetting (SFH-DF).

    An extension of :func:`scale_free_hedge_forecast` that prevents a single
    large loss (e.g. a CGM sensor artifact) from permanently inflating
    ``L_max`` and suppressing all future weight updates.

    Instead of a hard running maximum, ``L_max`` decays exponentially::

        L_max_t = max( (1 - gamma) * L_max_{t-1},  max_k l_{t,k} )

    When ``gamma = 0`` this reduces to the original Scale-Free Hedge.
    When ``gamma = 1``, ``L_max`` equals only the current step's worst loss
    (fully forgetful, equivalent to per-step ratio normalization).
    Values in ``0.1 – 0.4`` offer a smooth decay that keeps enough memory
    to remain scale-free while recovering from transient spikes.

    Kwargs
    ------
    gamma : float
        Forgetting rate in [0, 1).  Default 0.2.
    start, end, forecast_type, save_weights : standard kwargs.
    """
    start    = kwargs.get("start",    None)
    end      = kwargs.get("end",      None)
    gamma    = float(kwargs.get("gamma", 0.2))   # decay rate for L_max
    save_weights  = kwargs.get("save_weights",  False)
    forecast_type = kwargs.get("forecast_type", "sample")

    if start is None or not end:
        raise ValueError("Must have start and end times")

    f_mat = utils.make_matrix(forecasts)
    l_mat = utils.make_matrix(losses)
    T, m  = l_mat.shape

    cum_loss  = np.zeros(m)   # L_{t,k}: per-expert cumulative loss
    L_max     = 1e-8          # exponentially-decayed running max
    Wt        = np.ones(m) / m
    W         = np.ones((T + 1, m)) / m
    if save_weights:
        WT = np.ones((T + 1, m)) / m

    exp_forecasts = np.zeros(T)
    exp_losses    = np.zeros(T)
    jt = 0

    for g in range(start, T):
        l = l_mat[g]

        if not (l == 0).all():
            l_safe = np.nan_to_num(l, nan=np.nanmax(l) if not np.isnan(l).all() else 0.0)
            # Decay-forgetting update: use EMA of max loss to smooth the learning rate
            L_max = (1.0 - gamma) * L_max + gamma * float(l_safe.max())
            cum_loss = cum_loss + l_safe

            log_w  = -cum_loss / max(L_max, 1e-8)
            log_w[np.isnan(l)] = -np.inf  # Force zero weight for failing experts
            log_w -= np.nanmax(log_w) if not np.isnan(log_w).all() else 0.0  # numerical stability
            Wt     = np.exp(log_w)
            denom  = Wt.sum()
            Wt     = Wt / denom if denom > 1e-12 else np.ones(m) / m
            W[g + 1] = Wt.copy()
        else:
            W[g + 1] = Wt.copy()

        if save_weights:
            WT[g + 1] = Wt.copy()

        t_target = g + kwargs.get("horizon", 1)
        if t_target < T:
            Wt_norm = Wt / Wt.sum()
            if forecast_type == "mean":
                exp_forecasts[t_target] = np.nansum(Wt_norm * f_mat[t_target])
                exp_losses[t_target]    = np.nansum(Wt_norm * l_mat[t_target])
            else:
                jt = np.random.choice(m, p=Wt_norm)
                exp_forecasts[t_target] = f_mat[t_target][jt]
                exp_losses[t_target]    = l_mat[t_target][jt]

    if save_weights:
        return exp_forecasts, exp_losses, WT
    return exp_forecasts, exp_losses


# Registry mapping method names to their forecast functions.
# Used by get_weighted_forecasts_advanced to dispatch calls.
_ADVANCED_FORECAST_FUNCS = {
    "variable_share":        variable_share_forecast,
    "adahedge":              adahedge_forecast,
    "scale_free_hedge":      scale_free_hedge_forecast,
    "scale_free_hedge_df":   scale_free_hedge_df_forecast,
    # existing adaptive-eta variants are dispatched via the same interface
    "adaptive_eta":          weighted_forecast_adaptive_eta,
}


def get_weighted_forecasts_advanced(forecasts, losses, methods, **kwargs):
    """Unified dispatcher for advanced online learning algorithms.

    Each entry in ``methods`` is a ``(name, settings)`` dict.  The
    ``settings`` dict **must** contain a ``"method_type"`` key whose value
    is one of the keys in :data:`_ADVANCED_FORECAST_FUNCS`.

    Returns the same ``(exp_forecasts, exp_losses, exp_weights)`` triple as
    :func:`get_weighted_forecasts_adaptive`.
    """
    exp_forecasts, exp_losses, exp_weights = {}, {}, {}

    for name, settings in methods.items():
        method_type = settings.get("method_type", "adaptive_eta")
        fn = _ADVANCED_FORECAST_FUNCS.get(method_type, weighted_forecast_adaptive_eta)
        results = fn(forecasts, losses, **settings)
        if len(results) == 3:
            exp_forecasts[name], exp_losses[name], exp_weights[name] = results
        else:
            exp_forecasts[name], exp_losses[name] = results

    if exp_weights:
        return exp_forecasts, exp_losses, exp_weights
    return exp_forecasts, exp_losses


class DefaultSimulationParams:
    
    def sim_params(**kwargs): 
        return GlobalValues.sim_params | kwargs

    def trainer_params(**kwargs): 
        return GlobalValues.trainer_params | kwargs
    
    def exp_params(**kwargs):
        return GlobalValues.exp_params | kwargs
