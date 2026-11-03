"""run_simulation_chunk.py — Chunk-Executable Simulation Runner for SLURM Batch Jobs.

Enables fine-grained, parallel execution of online forecasting simulations across:
  1. Experimental condition indices (0 to 23, ideal for SLURM job arrays).
  2. Subsets/chunks of participant cohorts (e.g. chunks of 10 or 20 patients).
  3. Safe resource allocation (auto-detects SLURM_CPUS_PER_TASK, enforces thread limits).
  4. Automatic resumption (skips already computed patient forecasts).

Usage Examples:
---------------
  # Run a single condition by array index (0-23):
  python3 run_simulation_chunk.py --condition_idx 0

  # Run on SLURM (auto-detects SLURM_ARRAY_TASK_ID and SLURM_CPUS_PER_TASK):
  sbatch --array=0-23 slurm/submit_simulation_array.slurm

  # Run a patient chunk within a condition:
  python3 run_simulation_chunk.py --condition_idx 12 --chunk_size 20 --patient_chunk 0

  # Check completion progress across all conditions:
  python3 run_simulation_chunk.py --list_conditions

  # Quick debug verification on 1 patient:
  python3 run_simulation_chunk.py --condition_idx 0 --debug
"""

import os
import sys
import time
import glob
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fast_engine import FastRollingSimulation, run_cohort


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

DATASETS = ["cgmacros", "weinstock"]
HORIZONS = ["half", "2", "5"]
CONTEXTS = [72, 144, 288, -1]


def build_condition_registry() -> List[Dict[str, Any]]:
    """Build the canonical 24 experimental condition list."""
    registry = []
    idx = 0
    for ds in DATASETS:
        for h_str in HORIZONS:
            h_num = HORIZON_MAP[h_str]
            for ctx in CONTEXTS:
                c_str = CONTEXT_MAP[ctx]
                registry.append({
                    "condition_idx": idx,
                    "dataset": ds,
                    "horizon_str": h_str,
                    "horizon_num": h_num,
                    "context_val": ctx,
                    "context_str": c_str,
                    "output_dir": os.path.join("Outputs", ds, f"h-{h_str}hr", f"context-{c_str}hr"),
                })
                idx += 1
    return registry


def parse_conditions_spec(spec: str) -> List[int]:
    """Parse condition spec like '12-23', '0,1,2', '12', 'all', or '0-3,12-15' into list of ints.
    Preserves the specified order so callers can prioritize specific conditions."""
    if not spec or spec.strip().lower() == "all":
        return list(range(24))
    indices = []
    seen = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            s_str, e_str = part.split("-", 1)
            start_i = int(s_str.strip())
            end_i = int(e_str.strip())
            step = 1 if end_i >= start_i else -1
            for i in range(start_i, end_i + step, step):
                if i not in seen:
                    indices.append(i)
                    seen.add(i)
        else:
            val = int(part)
            if val not in seen:
                indices.append(val)
                seen.add(val)
    return indices


def build_grid_tasks(
    conditions_spec: str = "0-23",
    chunk_size: int = 20,
    registry: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build flat list of (condition, patient_chunk) tasks across specified conditions."""
    if registry is None:
        registry = build_condition_registry()
    cond_indices = parse_conditions_spec(conditions_spec)
    grid_tasks = []
    task_counter = 0

    for c_idx in cond_indices:
        if c_idx < 0 or c_idx >= len(registry):
            continue
        cond = registry[c_idx]
        ds = cond["dataset"]
        csv_files = sorted(glob.glob(os.path.join(ds, "*.csv")))
        total_patients = len(csv_files)
        if total_patients == 0:
            continue
        num_chunks = max(1, (total_patients + chunk_size - 1) // chunk_size)
        for chunk_idx in range(num_chunks):
            start_p = chunk_idx * chunk_size
            end_p = min(start_p + chunk_size, total_patients)
            grid_tasks.append({
                "task_id": task_counter,
                "condition_idx": c_idx,
                "patient_chunk": chunk_idx,
                "chunk_size": chunk_size,
                "dataset": ds,
                "horizon_str": cond["horizon_str"],
                "context_str": cond["context_str"],
                "start_patient": start_p,
                "end_patient": end_p,
                "num_patients": end_p - start_p,
                "total_patients": total_patients,
                "num_chunks": num_chunks,
            })
            task_counter += 1

    return grid_tasks



def get_patient_id(path: str) -> str:
    """Extract standard patient ID from CSV filename."""
    base = os.path.basename(path)
    if "-" in base:
        return base[base.find("-") + 1 : base.find("-") + 4]
    return os.path.splitext(base)[0]


# Timestamp cutoff: Only files created on or after Oct 15, 2026 18:47 UTC were generated
# by the new high-performance engine (fast_engine). Older files are from legacy simulations.
FAST_ENGINE_CUTOFF = 1792090000.0


def is_patient_complete(cond_dir: str, p_id: str) -> bool:
    """Check if the forecast file for a patient exists and was produced by the new fast engine."""
    fc_path = os.path.join(cond_dir, "forecasts", f"{p_id}_forecasts.csv")
    if not (os.path.exists(fc_path) and os.path.getsize(fc_path) > 500):
        return False
    # Reject files from legacy/old simulations (prior to Oct 15 2026 21:00 UTC)
    if os.path.getmtime(fc_path) < FAST_ENGINE_CUTOFF:
        return False
    return True


def list_conditions(registry: List[Dict[str, Any]]):
    """Print status and completion stats for all 24 conditions."""
    print("=" * 95)
    print(f"{'Idx':<4} {'Dataset':<10} {'Horizon':<10} {'Context':<10} {'Total':<7} {'Completed':<11} {'Pending':<9} {'Status':<12}")
    print("=" * 95)
    
    total_all_patients = 0
    total_all_completed = 0
    
    for cond in registry:
        ds = cond["dataset"]
        csv_files = sorted(glob.glob(os.path.join(ds, "*.csv")))
        total_p = len(csv_files)
        completed_p = 0
        for f in csv_files:
            p_id = get_patient_id(f)
            if is_patient_complete(cond["output_dir"], p_id):
                completed_p += 1
                
        pending_p = total_p - completed_p
        status = "DONE" if (total_p > 0 and pending_p == 0) else ("IN PROGRESS" if completed_p > 0 else "PENDING")
        
        total_all_patients += total_p
        total_all_completed += completed_p
        
        print(f"{cond['condition_idx']:<4} {ds:<10} {cond['horizon_str'] + 'hr':<10} {cond['context_str'] + 'hr':<10} {total_p:<7} {completed_p:<11} {pending_p:<9} {status:<12}")
        
    print("=" * 95)
    pct = (total_all_completed / max(total_all_patients, 1)) * 100.0
    print(f"Grand Total: {total_all_completed}/{total_all_patients} patient simulations complete ({pct:.1f}%)\n")


def run_condition(
    cond: Dict[str, Any],
    chunk_size: Optional[int] = None,
    patient_chunk: Optional[int] = None,
    patient_start: Optional[int] = None,
    patient_end: Optional[int] = None,
    patient_ids: Optional[List[str]] = None,
    n_workers: int = 4,
    skip_existing: bool = True,
    debug: bool = False,
):
    """Run simulation for a single experimental condition with optional patient chunking."""
    ds = cond["dataset"]
    h_str = cond["horizon_str"]
    h_num = cond["horizon_num"]
    ctx = cond["context_val"]
    c_str = cond["context_str"]
    cond_dir = cond["output_dir"]
    
    csv_files = sorted(glob.glob(os.path.join(ds, "*.csv")))
    if not csv_files:
        print(f"[Error] No CSV files found in {ds}/")
        return
        
    total_cohort_size = len(csv_files)
    
    # Filter by specific patient IDs if provided
    if patient_ids:
        csv_files = [f for f in csv_files if get_patient_id(f) in patient_ids]
        
    # Apply chunking if specified
    if chunk_size is not None and patient_chunk is not None:
        start_idx = patient_chunk * chunk_size
        end_idx = min(start_idx + chunk_size, len(csv_files))
        csv_files = csv_files[start_idx:end_idx]
    elif patient_start is not None or patient_end is not None:
        s = patient_start if patient_start is not None else 0
        e = patient_end if patient_end is not None else len(csv_files)
        csv_files = csv_files[s:e]
        
    if debug:
        csv_files = csv_files[:1]
        
    settings_dir = os.path.join(cond_dir, "settings")
    forecasts_dir = os.path.join(cond_dir, "forecasts")
    os.makedirs(settings_dir, exist_ok=True)
    os.makedirs(forecasts_dir, exist_ok=True)
    
    print("-" * 80)
    print(f"Condition [{cond['condition_idx']}]: {ds} | Horizon: {h_str}hr ({h_num} steps) | Context: {c_str}hr ({ctx} steps)")
    print(f"Target directory: {cond_dir}")
    print(f"Selected cohort:  {len(csv_files)} of {total_cohort_size} participants")
    print(f"Workers:          {n_workers}")
    print(f"Skip existing:    {skip_existing}")
    print(f"Debug mode:       {debug}")
    print("-" * 80)
    
    # 1. Write settings JSONs (needed for downstream process_results.py)
    for csv_path in csv_files:
        p_id = get_patient_id(csv_path)
        raw_df = pd.read_csv(csv_path)
        t_end = len(raw_df) - h_num
        if debug:
            t_end = min(t_end, 70)
            
        settings = {
            "start": 21,
            "log_n_steps": 500,
            "horizon": h_num,
            "batch_size": 32,
            "epochs": 10,
            "output_dir": os.path.abspath(cond_dir),
            "num_workers": 0,
            "context_len": ctx if ctx != -1 else len(raw_df),
            "id_num": p_id,
            "end": t_end,
            "input_dir": os.path.abspath(ds),
            "model_dir": os.path.abspath(cond_dir),
        }
        with open(os.path.join(settings_dir, f"{p_id}.json"), "w") as f:
            json.dump(settings, f, indent=2)

    # 2. Filter out already completed if skip_existing
    pending_files = []
    skipped_count = 0
    for f in csv_files:
        p_id = get_patient_id(f)
        if skip_existing and is_patient_complete(cond_dir, p_id):
            skipped_count += 1
        else:
            pending_files.append(f)
            
    print(f"Summary: {skipped_count} already completed, {len(pending_files)} pending.")
    
    if not pending_files:
        print(f"✓ All {len(csv_files)} participants in this chunk are already completed. Nothing to do.")
        return

    # 3. Execute cohort simulation
    t0 = time.perf_counter()
    results = run_cohort(
        csv_paths=pending_files,
        horizon=h_num,
        context_len=ctx,
        start_step=21,
        end_step=70 if debug else None,
        n_workers=n_workers,
        output_dir=cond_dir,
        skip_existing=skip_existing,
    )
    dt = time.perf_counter() - t0
    print(f"✓ Finished {len(pending_files)} participants in {dt:.2f} s ({dt/max(len(pending_files),1):.2f} s/patient)")


def get_default_workers() -> int:
    """Determine safe worker count, prioritizing SLURM allocation."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        try:
            return max(1, int(os.environ["SLURM_CPUS_PER_TASK"]))
        except ValueError:
            pass
    # Fallback to safe conservative default (4 workers) to avoid starving shared machines
    return 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HBML online forecasting simulations in chunks for SLURM batch jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Condition selection
    parser.add_argument(
        "--condition_idx",
        type=int,
        default=None,
        help="Condition index (0-23). Defaults to SLURM_ARRAY_TASK_ID environment variable if set.",
    )
    parser.add_argument("--dataset", type=str, choices=["cgmacros", "weinstock", "all"], default=None, help="Filter by dataset.")
    parser.add_argument("--horizon", type=str, choices=["half", "2", "5", "all"], default=None, help="Filter by horizon.")
    parser.add_argument("--context", type=int, choices=[72, 144, 288, -1], default=None, help="Filter by context length.")
    
    # Patient chunking
    parser.add_argument("--patient_chunk", type=int, default=None, help="Chunk index (0-based) when dividing cohort into chunks.")
    parser.add_argument("--chunk_size", type=int, default=None, help="Number of patients per chunk.")
    parser.add_argument("--patient_start", type=int, default=None, help="Start patient slice index.")
    parser.add_argument("--patient_end", type=int, default=None, help="End patient slice index.")
    parser.add_argument("--patient_ids", type=str, default=None, help="Comma-separated patient IDs to run (e.g. '001,002,003').")

    # 2D Grid Execution (Condition Array + Patient Chunks)
    parser.add_argument(
        "--conditions",
        type=str,
        default=os.environ.get("CONDITIONS", None),
        help="Condition range specification for unified grid (e.g. '12-23', '0-23', '0,4,12'). Auto-detects $CONDITIONS.",
    )
    parser.add_argument(
        "--grid_task_id",
        type=int,
        default=None,
        help="Task index in unified 2D grid of (condition, patient_chunk). Auto-detects SLURM_ARRAY_TASK_ID if --conditions or $CONDITIONS is active.",
    )
    parser.add_argument(
        "--count_grid_tasks",
        action="store_true",
        default=False,
        help="Print total number of tasks in the unified 2D grid and exit (used by submit scripts).",
    )
    parser.add_argument(
        "--list_grid",
        action="store_true",
        default=False,
        help="Print tabular summary of tasks in the unified 2D grid and exit.",
    )
    
    # Computational controls
    parser.add_argument(
        "--n_workers",
        type=int,
        default=get_default_workers(),
        help="Number of parallel worker processes. Auto-detects SLURM_CPUS_PER_TASK if unset.",
    )
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip patients with complete forecast files.")
    parser.add_argument("--force_overwrite", action="store_false", dest="skip_existing", help="Force overwrite of existing forecast files.")
    parser.add_argument("--debug", action="store_true", default=False, help="Quick test mode (1 patient, 70 steps).")
    parser.add_argument("--list_conditions", action="store_true", default=False, help="Print status of all 24 conditions and exit.")
    
    return parser.parse_args()


def main():
    args = parse_args()
    registry = build_condition_registry()
    
    if args.list_conditions:
        list_conditions(registry)
        return

    # Check for Grid Mode queries
    chunk_sz = args.chunk_size if args.chunk_size is not None else (int(os.environ["CHUNK_SIZE"]) if "CHUNK_SIZE" in os.environ else 20)
    cond_spec = args.conditions if args.conditions is not None else os.environ.get("CONDITIONS")

    if args.count_grid_tasks:
        grid = build_grid_tasks(conditions_spec=cond_spec or "0-23", chunk_size=chunk_sz, registry=registry)
        print(len(grid))
        return

    if args.list_grid:
        grid = build_grid_tasks(conditions_spec=cond_spec or "0-23", chunk_size=chunk_sz, registry=registry)
        print("=" * 105)
        print(f"Unified 2D Grid: {len(grid)} total tasks for conditions [{cond_spec or '0-23'}] with chunk size {chunk_sz}")
        print("=" * 105)
        print(f"{'Task':<6} {'Cond':<5} {'Dataset':<10} {'Horizon':<8} {'Context':<8} {'Chunk':<12} {'Patient Slice':<16} {'Count':<6}")
        print("-" * 105)
        for t in grid:
            p_slice = f"[{t['start_patient']}:{t['end_patient']}] of {t['total_patients']}"
            chunk_info = f"#{t['patient_chunk']} (of {t['num_chunks']})"
            print(f"{t['task_id']:<6} {t['condition_idx']:<5} {t['dataset']:<10} {t['horizon_str']+'hr':<8} {t['context_str']+'hr':<8} {chunk_info:<12} {p_slice:<16} {t['num_patients']:<6}")
        print("=" * 105)
        return

    # Check for Grid Execution Mode:
    # Triggered if --grid_task_id is given OR (conditions is specified/exported AND SLURM_ARRAY_TASK_ID is present AND condition_idx is not explicitly given)
    grid_task_id = args.grid_task_id
    if grid_task_id is None and cond_spec is not None and args.condition_idx is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        try:
            grid_task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
            print(f"[SLURM] Detected Grid Task ID from SLURM_ARRAY_TASK_ID={grid_task_id} (Conditions: {cond_spec}, Chunk Size: {chunk_sz})")
        except ValueError:
            pass

    if grid_task_id is not None:
        grid = build_grid_tasks(conditions_spec=cond_spec or "0-23", chunk_size=chunk_sz, registry=registry)
        if 0 <= grid_task_id < len(grid):
            task_info = grid[grid_task_id]
            print(f"==============================================================================")
            print(f"Executing Grid Task [{grid_task_id}/{len(grid)-1}]:")
            print(f"  Condition:       {task_info['condition_idx']} ({task_info['dataset']})")
            print(f"  Horizon/Context: {task_info['horizon_str']}hr / {task_info['context_str']}hr")
            print(f"  Patient Chunk:   #{task_info['patient_chunk']} (Patients {task_info['start_patient']}..{task_info['end_patient']-1} of {task_info['total_patients']})")
            print(f"==============================================================================")
            run_condition(
                cond=registry[task_info["condition_idx"]],
                chunk_size=task_info["chunk_size"],
                patient_chunk=task_info["patient_chunk"],
                n_workers=args.n_workers,
                skip_existing=args.skip_existing,
                debug=args.debug,
            )
            return
        else:
            print(f"[Error] Invalid grid task ID {grid_task_id}. Must be between 0 and {len(grid)-1}.")
            sys.exit(1)

    # Check for SLURM_ARRAY_TASK_ID if condition_idx is not explicitly set (Standard Condition Array Mode)
    cond_idx = args.condition_idx
    if cond_idx is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        try:
            cond_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
            print(f"[SLURM] Detected SLURM_ARRAY_TASK_ID={cond_idx}")
        except ValueError:
            pass

    # Select conditions to run
    if cond_idx is not None:
        if 0 <= cond_idx < len(registry):
            target_conditions = [registry[cond_idx]]
        else:
            print(f"[Error] Invalid condition index {cond_idx}. Must be between 0 and {len(registry)-1}.")
            sys.exit(1)
    else:
        # Filter based on dataset, horizon, context
        target_conditions = registry
        if args.dataset and args.dataset != "all":
            target_conditions = [c for c in target_conditions if c["dataset"] == args.dataset]
        if args.horizon and args.horizon != "all":
            target_conditions = [c for c in target_conditions if c["horizon_str"] == args.horizon]
        if args.context is not None:
            target_conditions = [c for c in target_conditions if c["context_val"] == args.context]

    patient_ids_list = [p.strip() for p in args.patient_ids.split(",")] if args.patient_ids else None

    print(f"Executing {len(target_conditions)} condition(s)...")
    for cond in target_conditions:
        run_condition(
            cond=cond,
            chunk_size=args.chunk_size,
            patient_chunk=args.patient_chunk,
            patient_start=args.patient_start,
            patient_end=args.patient_end,
            patient_ids=patient_ids_list,
            n_workers=args.n_workers,
            skip_existing=args.skip_existing,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
