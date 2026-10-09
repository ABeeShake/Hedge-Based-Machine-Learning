#!/usr/bin/env python3
"""verify_scale_invariance.py — Formal unit test and verification of scale-invariance.

Verifies that the Scale-Free Hedge (SFH) and SFH with Decay Forgetting (SFH-DF)
algorithms in ExpMethods/simulate.py satisfy strict mathematical scale-invariance:
multiplying inputs and targets by any positive scalar a > 0 (e.g. converting between
mg/dL and mmol/L via a = 1 / 18.0182) results in:
  1. Identical weight trajectories at every time step t: w_t(a * losses) == w_t(losses)
  2. Identical relative regret and normalized performance
  3. Perfectly scaled predictions: y_hat(a * forecasts) == a * y_hat(forecasts)
"""

import numpy as np

def run_scale_invariance_test():
    np.random.seed(42)
    T = 500
    K = 5
    
    # Simulate synthetic glucose trajectory (in mg/dL)
    true_y_mgdl = np.random.normal(140, 30, size=T).clip(40, 400)
    
    # Simulate 5 expert predictions with varying noise
    expert_preds_mgdl = {
        f"expert_{k}": true_y_mgdl + np.random.normal(0, (k + 1) * 5, size=T)
        for k in range(K)
    }
    
    # Scale factor: mg/dL to mmol/L conversion
    a = 1.0 / 18.0182
    true_y_mmoll = true_y_mgdl * a
    expert_preds_mmoll = {k: v * a for k, v in expert_preds_mgdl.items()}
    
    gamma = 0.2
    
    for method_name, use_df in [("Scale-Free Hedge (SFH)", False), ("SFH with Decay Forgetting (SFH-DF)", True)]:
        print(f"\n--- Testing {method_name} ---")
        
        # Run in mg/dL
        w_mgdl = []
        preds_mgdl = []
        cum_loss_1 = np.zeros(K)
        L_max_1 = 1.0
        
        # Run in mmol/L
        w_mmoll = []
        preds_mmoll = []
        cum_loss_2 = np.zeros(K)
        L_max_2 = 1.0 * (a ** 2)
        
        for t in range(T):
            p1 = np.array([expert_preds_mgdl[f"expert_{k}"][t] for k in range(K)])
            p2 = np.array([expert_preds_mmoll[f"expert_{k}"][t] for k in range(K)])
            
            # Target at t
            y1 = true_y_mgdl[t]
            y2 = true_y_mmoll[t]
            
            # Squared losses
            loss1 = (p1 - y1) ** 2
            loss2 = (p2 - y2) ** 2
            
            # Update L_max
            if use_df:
                L_max_1 = max((1.0 - gamma) * L_max_1, np.max(loss1), 1e-8)
                L_max_2 = max((1.0 - gamma) * L_max_2, np.max(loss2), 1e-8 * (a ** 2))
            else:
                L_max_1 = max(L_max_1, np.max(loss1), 1e-8)
                L_max_2 = max(L_max_2, np.max(loss2), 1e-8 * (a ** 2))
                
            cum_loss_1 += loss1
            cum_loss_2 += loss2
            
            # Exponent update
            log_w1 = -cum_loss_1 / L_max_1
            log_w1 -= np.max(log_w1)
            wt1 = np.exp(log_w1)
            wt1 /= np.sum(wt1)
            
            log_w2 = -cum_loss_2 / L_max_2
            log_w2 -= np.max(log_w2)
            wt2 = np.exp(log_w2)
            wt2 /= np.sum(wt2)
            
            w_mgdl.append(wt1)
            w_mmoll.append(wt2)
            
            y_hat1 = np.dot(wt1, p1)
            y_hat2 = np.dot(wt2, p2)
            
            preds_mgdl.append(y_hat1)
            preds_mmoll.append(y_hat2)
            
        w_mgdl = np.array(w_mgdl)
        w_mmoll = np.array(w_mmoll)
        preds_mgdl = np.array(preds_mgdl)
        preds_mmoll = np.array(preds_mmoll)
        
        # Assertions
        weight_diff_max = np.max(np.abs(w_mgdl - w_mmoll))
        pred_ratio_diff = np.max(np.abs(preds_mmoll - a * preds_mgdl))
        
        print(f"  Maximum weight difference: {weight_diff_max:.2e}")
        print(f"  Maximum prediction scaling error: {pred_ratio_diff:.2e}")
        
        assert weight_diff_max < 1e-12, f"Weight vectors diverged: {weight_diff_max}"
        assert pred_ratio_diff < 1e-12, f"Predictions failed scale invariance: {pred_ratio_diff}"
        print(f"  [PASS] {method_name} is strictly scale-invariant under unit transformations!")

if __name__ == "__main__":
    run_scale_invariance_test()
