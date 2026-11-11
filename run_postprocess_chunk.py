"""run_postprocess_chunk.py — Chunk-Executable Post-Processing for SLURM Batch Jobs.

Runs Stage 2 post-processing (HBML weights, regrets, Clarke Error Grids)
on individual experimental conditions (0-23) or all completed conditions.

Usage Examples:
---------------
  # Post-process a single condition by array index (0-23):
  python3 run_postprocess_chunk.py --condition_idx 0

  # Post-process on SLURM (auto-detects SLURM_ARRAY_TASK_ID):
  sbatch --array=0-23 slurm/submit_postprocess_array.slurm

  # Post-process all conditions that have completed simulations:
  python3 run_postprocess_chunk.py --all_completed
"""

import os
import sys
import glob
import argparse
import subprocess

from run_simulation_chunk import build_condition_registry, is_patient_complete, get_patient_id


def process_condition(cond, overwrite=False, plot_ceg=True):
    cond_dir = cond["output_dir"]
    ds = cond["dataset"]
    
    if not os.path.exists(cond_dir):
        print(f"[Skip] Directory does not exist: {cond_dir}")
        return False
        
    print("=" * 80)
    print(f"Post-processing condition [{cond['condition_idx']}]: {ds} | Horizon: {cond['horizon_str']}hr | Context: {cond['context_str']}hr")
    print(f"Directory: {cond_dir}")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        "process_results.py",
        "--output_dir", cond_dir,
        "--data_dir", f"./{ds}",
        "--adaptive_eta",
        "--update_loss_type", "mse",
        "--norm_type", "ratio",
        "--advanced_methods",
    ]
    if overwrite:
        cmd.append("--overwrite")
    if plot_ceg:
        cmd.append("--plot_ceg")
        
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print(f"✓ Condition [{cond['condition_idx']}] post-processing completed successfully.")
        return True
    else:
        print(f"✗ Condition [{cond['condition_idx']}] post-processing failed with exit code {res.returncode}.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run post-processing in chunks for SLURM batch jobs.")
    parser.add_argument("--condition_idx", type=int, default=None, help="Condition index (0-23) to process. Auto-detects SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--all_completed", action="store_true", default=False, help="Process all conditions where all simulations are finished.")
    parser.add_argument("--all", action="store_true", default=False, help="Process all 24 conditions.")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing post-processed files.")
    parser.add_argument("--no_plot_ceg", action="store_false", dest="plot_ceg", default=True, help="Disable CEG plotting.")
    args = parser.parse_args()

    registry = build_condition_registry()

    cond_idx = args.condition_idx
    if cond_idx is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        try:
            cond_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
            print(f"[SLURM] Detected SLURM_ARRAY_TASK_ID={cond_idx}")
        except ValueError:
            pass

    if cond_idx is not None:
        if 0 <= cond_idx < len(registry):
            process_condition(registry[cond_idx], overwrite=args.overwrite, plot_ceg=args.plot_ceg)
        else:
            print(f"[Error] Invalid condition index {cond_idx}.")
            sys.exit(1)
    elif args.all_completed:
        for cond in registry:
            ds = cond["dataset"]
            csv_files = [f for f in sorted(glob.glob(os.path.join(ds, "*.csv"))) if os.path.basename(f) != f"{ds}.csv"]
            complete = all(is_patient_complete(cond["output_dir"], get_patient_id(f)) for f in csv_files)
            if complete and csv_files:
                process_condition(cond, overwrite=args.overwrite, plot_ceg=args.plot_ceg)
            else:
                print(f"[Pending] Condition [{cond['condition_idx']}] not yet complete, skipping post-processing.")
    elif args.all:
        for cond in registry:
            process_condition(cond, overwrite=args.overwrite, plot_ceg=args.plot_ceg)
    else:
        print("Please specify --condition_idx <0-23>, --all_completed, or --all.")


if __name__ == "__main__":
    main()
