"""run_paper_fast.py — Master High-Performance Reproduction Runner.

Executes all 24 experimental configurations in the HBML paper using the redesigned
high-performance engine (fast_engine), then seamlessly executes post-processing,
figure generation, and LaTeX macro exports.

Usage:
------
    # Run full paper reproduction with auto-detected workers:
    python3 run_paper_fast.py

    # Quick test on 1 patient:
    python3 run_paper_fast.py --debug

    # Customize datasets or horizons:
    python3 run_paper_fast.py --datasets cgmacros --horizons half 2 --n_workers 8
"""

import os
import sys
import time
import glob
import json
import argparse
import subprocess
from typing import List, Dict, Any

import pandas as pd
import numpy as np

from fast_engine import FastRollingSimulation, run_cohort
from ExpMethods.globals import GlobalValues


HORIZON_MAP = {
    "half": 6,
    "0.5": 6,
    "2": 24,
    "5": 60,
}

CONTEXT_MAP = {
    72: "6",
    144: "12",
    288: "24",
    -1: "full",
}


def get_patient_id(path: str) -> str:
    base = os.path.basename(path)
    if "-" in base:
        id_num = base[base.find("-") + 1 : base.find("-") + 4]
    else:
        id_num = os.path.splitext(base)[0]
    return id_num


def run_stage1_simulations(
    datasets: List[str],
    horizons: List[str],
    contexts: List[int],
    output_dir: str,
    n_workers: int,
    debug: bool = False,
):
    print("=" * 80)
    print("STAGE 1: HIGH-PERFORMANCE WALK-FORWARD ONLINE SIMULATIONS")
    print("=" * 80)
    print(f"Datasets:        {datasets}")
    print(f"Horizons:        {horizons}")
    print(f"Context lengths: {contexts}")
    print(f"Worker threads:  {n_workers}")
    print(f"Debug mode:      {debug}")
    print("-" * 80)

    total_conditions = len(datasets) * len(horizons) * len(contexts)
    condition_idx = 0
    t_stage1_start = time.perf_counter()

    for ds in datasets:
        input_dir = os.path.join(".", ds)
        csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
        
        if not csv_files:
            print(f"Warning: No CSV files found in {input_dir}. Skipping dataset.")
            continue
            
        if debug:
            csv_files = csv_files[:1]

        for h_str in horizons:
            h_num = HORIZON_MAP.get(str(h_str).lower(), 6)

            for ctx in contexts:
                condition_idx += 1
                c_str = CONTEXT_MAP.get(ctx, str(ctx))
                cond_dir = os.path.join(output_dir, ds, f"h-{h_str}hr", f"context-{c_str}hr")
                settings_dir = os.path.join(cond_dir, "settings")
                forecasts_dir = os.path.join(cond_dir, "forecasts")

                os.makedirs(settings_dir, exist_ok=True)
                os.makedirs(forecasts_dir, exist_ok=True)

                print(f"\n[{condition_idx}/{total_conditions}] Running {ds} | Horizon: {h_str}hr ({h_num} steps) | Context: {c_str}hr ({ctx} steps)")

                # Create settings JSON for each patient (required by downstream process_results.py)
                for csv_path in csv_files:
                    p_id = get_patient_id(csv_path)
                    raw_df = pd.read_csv(csv_path)
                    t_end = len(raw_df) - h_num
                    if debug:
                        t_end = min(t_end, 70)

                    settings = {
                        "start": GlobalValues.burn_in_steps,
                        "log_n_steps": 500,
                        "horizon": h_num,
                        "batch_size": 32,
                        "epochs": 10,
                        "output_dir": os.path.abspath(cond_dir),
                        "num_workers": 0,
                        "context_len": ctx if ctx != -1 else len(raw_df),
                        "id_num": p_id,
                        "end": t_end,
                        "input_dir": os.path.abspath(input_dir),
                        "model_dir": os.path.abspath(cond_dir),
                    }
                    with open(os.path.join(settings_dir, f"{p_id}.json"), "w") as f:
                        json.dump(settings, f)

                # Execute cohort simulation
                t0 = time.perf_counter()
                results = run_cohort(
                    csv_paths=csv_files,
                    horizon=h_num,
                    context_len=ctx,
                    end_step=GlobalValues.burn_in_steps + 80 if debug else None,
                    n_workers=n_workers,
                    output_dir=cond_dir,
                )
                dt = time.perf_counter() - t0
                print(f"  ✓ Finished {len(csv_files)} participants in {dt:.2f} s ({dt/max(len(csv_files),1):.2f} s/patient)")

    total_stage1_time = time.perf_counter() - t_stage1_start
    print("\n" + "=" * 80)
    print(f"STAGE 1 COMPLETE: All {total_conditions} conditions finished in {total_stage1_time:.2f} s ({total_stage1_time/60:.2f} min)")
    print("=" * 80)


def run_command(cmd: str, desc: str):
    print(f"\n>>> Running {desc}...")
    print(f"    Command: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"Error running {desc} (exit code {res.returncode})")
    else:
        print(f"✓ {desc} completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Run HBML fast paper reproduction pipeline.")
    parser.add_argument("--datasets", nargs="+", default=["cgmacros", "weinstock"], help="Datasets to simulate")
    parser.add_argument("--horizons", nargs="+", default=["half", "2", "5"], help="Horizons to simulate")
    parser.add_argument("--contexts", nargs="+", type=int, default=[72, 144, 288, -1], help="Context lengths in 5-min steps (-1 for full)")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--n_workers", type=int, default=os.cpu_count() or 4, help="Number of worker threads")
    parser.add_argument("--debug", action="store_true", help="Quick debug run on 1 participant and 70 steps")
    parser.add_argument("--skip_stage1", action="store_true", help="Skip Stage 1 simulation and run downstream processing only")
    args = parser.parse_args()

    if not args.skip_stage1:
        run_stage1_simulations(
            datasets=args.datasets,
            horizons=args.horizons,
            contexts=args.contexts,
            output_dir=args.output_dir,
            n_workers=args.n_workers,
            debug=args.debug,
        )

    py = sys.executable
    run_command(f"{py} process_results.py", "Stage 2: Process simulation results")
    run_command(f"{py} calculate_statistics.py", "Stage 3: Paired statistical tests")
    run_command(f"{py} calculate_baselines.py", "Stage 4: Naive baseline comparisons")
    run_command(f"{py} compute_regimes.py", "Stage 5: Physiological regime classification")
    run_command(f"{py} generate_figures.py", "Stage 6: Figure generation")
    run_command(f"{py} generate_results_macros.py --strict", "Stage 7: LaTeX macro generation")
    run_command(f"{py} check_macros.py", "Stage 8: Manuscript macro audit")


if __name__ == "__main__":
    main()
