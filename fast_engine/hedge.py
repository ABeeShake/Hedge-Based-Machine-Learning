"""fast_engine/hedge.py — Vectorized Online Hedge Aggregation Engine.

Decouples Hedge ensemble weighting from expert forecasting.
Evaluates Fixed-Share, AdaHedge, and Scale-Free updates across thousands of
timesteps in milliseconds.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class FastOnlineHedge:
    """Vectorized Scale-Free Fixed-Share & AdaHedge Aggregator."""
    
    def __init__(
        self,
        horizon: int = 6,
        start_step: int = 20,
        alpha: float = 0.5,
        eta: float = 1.0,
        adaptive_eta: bool = False,
        gamma: float = 0.0,  # Discount / forgetting factor
        norm_type: str = "shift",  # 'shift' or 'ratio'
        forecast_type: str = "mean",  # 'mean' (weighted average) or 'sample'
    ):
        self.horizon = horizon
        self.start_step = start_step
        self.alpha = alpha
        self.eta = eta
        self.adaptive_eta = adaptive_eta
        self.gamma = gamma
        self.norm_type = norm_type
        self.forecast_type = forecast_type
        
    def evaluate(
        self,
        forecast_matrix: np.ndarray,  # Shape (T, K)
        targets: np.ndarray,          # Shape (T,)
        expert_names: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Run vectorized online ensemble aggregation over pre-computed expert forecasts.
        
        Args:
            forecast_matrix: (T, K) array of expert predictions.
            targets: (T,) array of ground-truth glucose values.
            expert_names: List of K model names.
            
        Returns:
            Dictionary containing:
                - 'ensemble_forecasts': (T,) array of ensemble predictions.
                - 'ensemble_losses': (T,) array of squared errors.
                - 'weights': (T+1, K) weight trajectory matrix.
                - 'expert_losses': (T, K) array of squared errors for each expert.
                - 'regrets': (T, K) cumulative adaptive regret array.
        """
        T, K = forecast_matrix.shape
        targets = np.asarray(targets, dtype=float).flatten()
        
        # Expert squared errors
        loss_matrix = (forecast_matrix - targets[:, None]) ** 2
        # Zero out losses before horizon extrapolation is active
        loss_matrix[: self.start_step + self.horizon] = 0.0
        
        # State variables
        W = np.ones((T + 1, K)) / K
        w_current = np.ones(K) / K
        ensemble_forecasts = np.zeros(T)
        ensemble_losses = np.zeros(T)
        
        cum_expert_loss = np.zeros(K)
        cum_ensemble_loss = 0.0
        regrets = np.zeros((T, K))
        
        for t in range(self.start_step, T):
            losses_t = loss_matrix[t]
            
            # Update weights if non-zero feedback is present
            if not np.all(losses_t == 0.0):
                if self.norm_type == "ratio":
                    max_l = max(float(np.max(losses_t)), 1e-8)
                    loss_norm = losses_t / max_l
                else:
                    loss_norm = losses_t - np.min(losses_t)
                    
                # Learning rate
                curr_eta = self.eta
                if self.adaptive_eta:
                    # AdaHedge schedule: eta_t = sqrt(ln K / max(1e-4, Delta_t))
                    max_loss = np.max(loss_norm)
                    curr_eta = np.sqrt(np.log(K) / max(max_loss, 1e-4))
                    
                # Exponential penalty update
                penalty = np.exp(-curr_eta * loss_norm)
                if self.gamma > 0.0:
                    # Discount historical weights
                    w_current = w_current * (1.0 - self.gamma)
                    
                w_tilde = w_current * penalty
                w_tilde = np.nan_to_num(w_tilde, nan=0.0)
                
                # Fixed-share mixing update: (1 - alpha) * w + alpha * uniform
                if np.sum(w_tilde) > 0:
                    w_tilde_norm = w_tilde / np.sum(w_tilde)
                else:
                    w_tilde_norm = np.ones(K) / K
                    
                w_current = (1.0 - self.alpha) * w_tilde_norm + self.alpha * (1.0 / K)
                W[t + 1] = w_current
            else:
                W[t + 1] = w_current
                
            # Ensemble forecast at horizon t + h
            t_target = t + self.horizon
            if t_target < T:
                w_norm = w_current / np.sum(w_current)
                if self.forecast_type == "mean":
                    pred = np.sum(w_norm * forecast_matrix[t_target])
                else:
                    chosen_idx = np.random.choice(K, p=w_norm)
                    pred = forecast_matrix[t_target, chosen_idx]
                    
                ensemble_forecasts[t_target] = pred
                loss_val = (pred - targets[t_target]) ** 2
                ensemble_losses[t_target] = loss_val
                
                # Regret tracking
                cum_ensemble_loss += loss_val
                cum_expert_loss += loss_matrix[t_target]
                regrets[t_target] = cum_ensemble_loss - cum_expert_loss
                
        return {
            "ensemble_forecasts": ensemble_forecasts,
            "ensemble_losses": ensemble_losses,
            "weights": W,
            "expert_losses": loss_matrix,
            "regrets": regrets,
        }
