#!/usr/bin/env python3
"""evaluate_loss_experts.py -- Evaluate retrained alternative-loss experts saved in Outputs_loss/<loss>/.

Compares expert-retrained HBML runs (using both standard squared Hedge loss and the matching online
Hedge loss, plus Mixture-EG) against baseline Outputs/ (MSE experts) and Persistence on RMSE, MAE,
TOST equivalence, and 30-min matched-FAR hypoglycemia sensitivity.

Usage:
  python3 evaluate_loss_experts.py --expert-loss huber20 --datasets weinstock,cgmacros,hall,dubosson
"""
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import evaluate_loss_variants as elv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-loss", required=True, choices=["huber20", "pinball20", "asym_hypo"])
    ap.add_argument("--datasets", default="weinstock,cgmacros,hall,dubosson")
    ap.add_argument("--outputs-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--full-only", action="store_true")
    a = ap.parse_args()

    out_root = a.outputs_root or os.path.join(BASE_DIR, "Outputs_loss", a.expert_loss)
    out_dir = a.out_dir or os.path.join(BASE_DIR, "loss_results", f"experts_{a.expert_loss}")
    datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]
    elv.run_loss_variants(datasets, out_root, out_dir, n_boot=a.n_boot, full_only=a.full_only)


if __name__ == "__main__":
    main()
