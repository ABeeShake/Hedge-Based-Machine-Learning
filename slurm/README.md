# SLURM Execution Framework for HBML Online Forecasting

This directory contains modular SLURM batch scripts designed to run simulations, post-processing, and figure generation in **controlled chunks** on shared clusters.

---

## Key Features

1. **Neighbor-Friendly Concurrency**:
   - Throttles job arrays (e.g. `%4`), limiting total active tasks so resources are never monopolized.
   - Restricts OpenMP / MKL / BLAS threads (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`) to avoid process-level thread explosions.
2. **Auto-Resumption (`--skip_existing`)**:
   - Any patient whose forecast file (`Outputs/.../forecasts/<id>_forecasts.csv`) is already completed is instantly skipped.
   - If a job hits a time limit, resubmitting continues immediately from where it stopped with zero wasted work.
3. **Multi-Level Chunking**:
   - **Condition-level**: 24 experimental configurations (indices 0 to 23).
   - **Patient-level**: Large datasets (e.g. Weinstock with 201 patients) can be chunked into arbitrary batch sizes (e.g. 20 patients per chunk).

---

## Quick Reference Commands

### 1. Check Current Progress & Status Table
```bash
python3 run_simulation_chunk.py --list_conditions
```
Prints a table of all 24 conditions with total, completed, pending counts, and status (`DONE`, `IN PROGRESS`, `PENDING`).

---

### 2. Submit the Full Automated Pipeline (Stage 1 -> Stage 2 -> Stages 3 & 4)
```bash
bash slurm/submit_pipeline.sh
```
Uses SLURM job dependencies (`--dependency=afterok:<job_id>`) to automatically:
1. Run simulations in array (`#SBATCH --array=0-23%4`)
2. Run post-processing array once simulations finish
3. Generate figures and LaTeX macros once post-processing finishes

---

### 3. Submit Simulations Only (Job Array 0-23)
```bash
sbatch slurm/submit_simulation_array.slurm
```
To run only a specific range of conditions (e.g., conditions 4 through 10, running at most 2 simultaneously):
```bash
sbatch --array=4-10%2 slurm/submit_simulation_array.slurm
```

---

### 4. Submit Single Condition
```bash
# Run condition 4 (e.g., the 1 pending patient in cgmacros 2hr context-6hr):
sbatch --export=CONDITION_IDX=4 slurm/submit_single_condition.slurm
```

---

### 5. Unified 2D Grid (Condition Array + Patient Chunks) — Recommended for Weinstock
Run across a range of conditions (e.g. 12-23) while splitting large cohorts into small, time-safe patient chunks (e.g. 20 patients per chunk):

```bash
# Helper script (automatically calculates total tasks and array bounds):
bash slurm/submit_grid.sh --conditions 12-23 --chunk_size 20 --throttle 4

# Positional shorthand (Conditions, ChunkSize, Throttle):
bash slurm/submit_grid.sh 12-23 20 4

# Preview task mappings and sbatch command without submitting:
bash slurm/submit_grid.sh --conditions 12-23 --chunk_size 20 --dry-run

# Or submit directly via sbatch:
sbatch --array=0-131%4 --export=ALL,CONDITIONS=12-23,CHUNK_SIZE=20 slurm/submit_grid_simulation_array.slurm
```

---

### 6. Single-Condition Patient Chunking
```bash
# Run condition 12 only in chunks of 20 patients:
sbatch --export=CONDITION_IDX=12,CHUNK_SIZE=20 slurm/submit_patient_array.slurm
```

---

### 7. Post-Processing
```bash
# Post-process all conditions via SLURM array:
sbatch slurm/submit_postprocess_array.slurm

# Or post-process only conditions that have 100% completed:
python3 run_postprocess_chunk.py --all_completed
```

---

## SLURM Monitoring Commands

| Task | Command |
|---|---|
| View active jobs | `squeue -u $USER` |
| View cluster node partition | `sinfo` |
| Cancel a running job | `scancel <JOB_ID>` |
| Cancel all your jobs | `scancel -u $USER` |
| View live job logs | `tail -f slurm_logs/sim_<JOB_ID>_<ARRAY_ID>.out` |
