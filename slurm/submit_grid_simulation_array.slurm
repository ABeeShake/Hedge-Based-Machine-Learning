#!/bin/bash
#SBATCH --job-name=hbml_grid
#SBATCH --partition=LocalQ
#SBATCH --array=0-131%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/grid_%A_%a.out
#SBATCH --error=slurm_logs/grid_%A_%a.err

# ==============================================================================
# SLURM Job Array: Unified 2D Grid Simulation (Conditions × Patient Chunks)
#
# Maps each array task to a unique (condition_idx, patient_chunk) pair.
# Allows submitting an array across conditions (e.g. 12-23) while also breaking
# large cohorts into patient chunks (e.g. 20 patients per chunk).
#
# Defaults:
#   CONDITIONS: "12-23" (Weinstock cohort: 12 conditions)
#   CHUNK_SIZE: 20      (201 patients / 20 = 11 chunks per condition)
#   Total Tasks: 12 * 11 = 132 tasks (Array: 0-131)
#   Concurrency: %4 (at most 4 chunk tasks active, 32 CPU cores total)
#
# Submit via helper script (automatically calculates exact array bounds):
#   bash slurm/submit_grid.sh --conditions 12-23 --chunk_size 20 --throttle 4
#
# Or submit directly via sbatch:
#   sbatch --array=0-131%4 --export=ALL,CONDITIONS="12-23",CHUNK_SIZE=20 slurm/submit_grid_simulation_array.slurm
# ==============================================================================

set -euo pipefail

mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"

# Prevent OpenMP / MKL / BLAS thread explosion across worker processes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run multi-worker simulation purely on CPU cores (avoids GPU lock contention across workers)
export CUDA_VISIBLE_DEVICES=""
export FAST_ENGINE_DEVICE="cpu"

PYTHON_BIN="./hedge-env/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

CONDITIONS="${CONDITIONS:-12-23}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"

echo "=============================================================================="
echo "SLURM Job ID:       $SLURM_JOB_ID"
echo "Array Task ID:      $SLURM_ARRAY_TASK_ID (Grid Task #$SLURM_ARRAY_TASK_ID)"
echo "Condition Spec:     $CONDITIONS"
echo "Patient Chunk Size: $CHUNK_SIZE"
echo "Node:               $SLURMD_NODENAME"
echo "CPUs allocated:     $SLURM_CPUS_PER_TASK"
echo "Python interpreter: $PYTHON_BIN"
echo "Start time:         $(date)"
echo "=============================================================================="

$PYTHON_BIN run_simulation_chunk.py \
    --conditions "$CONDITIONS" \
    --chunk_size "$CHUNK_SIZE" \
    --grid_task_id "$SLURM_ARRAY_TASK_ID" \
    --n_workers "$SLURM_CPUS_PER_TASK" \
    --skip_existing

echo "=============================================================================="
echo "Grid Task $SLURM_ARRAY_TASK_ID completed at $(date)"
echo "=============================================================================="
