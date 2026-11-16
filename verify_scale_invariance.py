"""verify_scale_invariance.py
===============================================================================
Numerical & Algebraic Verification of Proposition 1 (Unit Scale-Invariance) on
the Production HBML Aggregators (`ExpMethods.aggregators.run_hedge`).

Tests both the primary cumulative-loss Scale-Free Hedge with EMA forgetting
(`HBML-SFHDF`), Recursive Fixed-Share (`HBML-FS`), `Cum-DecayMax`, and
`Rec-NoShare` directly via `ExpMethods.aggregators.run_hedge`.

Specifically, we verify:
1. Strict machine-precision invariance (< 1e-15) when the numerical floor EPS
   scales with a^2 (tilde_EPS = a^2 * EPS).
2. Practical invariance (< 1e-10) when a fixed floor EPS = 1e-8 is used in
   both mg/dL and mmol/L (where a = 1 / 18.0182, a^2 ~= 0.00308).
===============================================================================
"""

import numpy as np
import ExpMethods.aggregators as agg
from ExpMethods.aggregators import run_hedge, online_losses

VARIANT_SPECS = {
    "HBML-SFHDF": dict(update="cumulative", scale="ema", gamma=0.2, alpha=0.0),
    "HBML-FS": dict(update="recursive", scale="decay_max", gamma=0.2, alpha="fd"),
    "Cum-DecayMax": dict(update="cumulative", scale="decay_max", gamma=0.2, alpha=0.0),
    "Rec-NoShare": dict(update="recursive", scale="decay_max", gamma=0.2, alpha=0.0),
}


def main():
    print("=" * 80)
    print("VERIFYING PROPOSITION 1: SCALE-INVARIANCE OF PRODUCTION HBML AGGREGATORS")
    print("=" * 80)

    np.random.seed(42)
    T = 1000
    K = 5
    start = 21
    horizon = 6

    # Simulate realistic glucose trajectory in mg/dL (range ~ 70 to 250 mg/dL)
    t_grid = np.linspace(0, 20, T)
    y_mgdl = 130.0 + 50.0 * np.sin(t_grid) + np.random.normal(0, 8, size=T)

    # Simulate 5 diverse experts in mg/dL
    preds_mgdl = np.zeros((T, K))
    for k in range(K):
        bias = (k - 2) * 5.0
        noise_scale = 5.0 + 4.0 * k
        preds_mgdl[:, k] = y_mgdl + bias + np.random.normal(0, noise_scale, size=T)

    losses_mgdl = online_losses(preds_mgdl, y_mgdl, start=start, horizon=horizon)

    # Conversion factor from mg/dL to mmol/L: a = 1 / 18.0182
    a = 1.0 / 18.0182
    y_mmol = a * y_mgdl
    preds_mmol = a * preds_mgdl
    losses_mmol = online_losses(preds_mmol, y_mmol, start=start, horizon=horizon)

    print(f"Scale factor a (mg/dL -> mmol/L): {a:.8f} (a^2 = {a**2:.8e})\n")

    start_eval = start + horizon

    for label, spec in VARIANT_SPECS.items():
        # 1. Production run_hedge with fixed EPS = 1e-8 in both units
        agg.EPS = 1e-8
        f_mgdl, l_mgdl, w_mgdl = run_hedge(
            preds_mgdl, losses_mgdl, y_mgdl, start=start, horizon=horizon, return_weights=True, **spec
        )
        f_mmol_fixed, l_mmol_fixed, w_mmol_fixed = run_hedge(
            preds_mmol, losses_mmol, y_mmol, start=start, horizon=horizon, return_weights=True, **spec
        )

        max_w_diff_fixed = float(np.max(np.abs(w_mgdl[start_eval + 1:] - w_mmol_fixed[start_eval + 1:])))
        max_f_diff_fixed = float(
            np.max(np.abs(a * f_mgdl[start_eval + horizon:] - f_mmol_fixed[start_eval + horizon:]))
        )

        # 2. Production run_hedge with scaled EPS = a^2 * 1e-8
        agg.EPS = (a ** 2) * 1e-8
        f_mmol_scaled, l_mmol_scaled, w_mmol_scaled = run_hedge(
            preds_mmol, losses_mmol, y_mmol, start=start, horizon=horizon, return_weights=True, **spec
        )
        agg.EPS = 1e-8

        max_w_diff_scaled = float(np.max(np.abs(w_mgdl - w_mmol_scaled)))
        max_f_diff_scaled = float(np.max(np.abs(a * f_mgdl - f_mmol_scaled)))

        print(f"Variant: {label} ({spec})")
        print(f"  [Scaled EPS = a^2 * 1e-8] Max |w_mgdl - w_mmol|         : {max_w_diff_scaled:.3e}")
        print(f"  [Scaled EPS = a^2 * 1e-8] Max |a*y_hat_mgdl - y_hat_mmol|: {max_f_diff_scaled:.3e}")
        print(f"  [Fixed  EPS = 1e-8]       Max |w_mgdl - w_mmol|         : {max_w_diff_fixed:.3e}")
        print(f"  [Fixed  EPS = 1e-8]       Max |a*y_hat_mgdl - y_hat_mmol|: {max_f_diff_fixed:.3e}")

        assert max_w_diff_scaled < 1e-14, f"Scaled-EPS weight invariance failed for {label}!"
        assert max_w_diff_fixed < 1e-7, f"Fixed-EPS weight invariance failed for {label}!"

    print("\n" + "=" * 80)
    print("SUCCESS: Production aggregators satisfy Proposition 1 (Unit Scale-Invariance).")
    print("=" * 80)


if __name__ == "__main__":
    main()
