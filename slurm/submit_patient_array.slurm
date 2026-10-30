#!/bin/bash
#SBATCH --job-name=hbml_pts
#SBATCH --partition=LocalQ
#SBATCH --array=0-10%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/patient_chunk_%A_%a.out
#SBATCH --error=slurm_logs/patient_chunk_%A_%a.err

# ==============================================================================
# SLURM Job Array: Patient-Level Chunking within a Single Condition
#
# Useful for large cohorts (e.g. Weinstock with 201 patients) broken into chunks
# (e.g., 20 patients per chunk -> 11 chunks: array 0-10).
#
# Usage:
#   sbatch --export=CONDITION_IDX=12,CHUNK_SIZE=20 slurm/submit_patient_array.slurm
# ==============================================================================

set -euo pipefail

mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON_BIN="./hedge-env/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

CONDITION_IDX="${CONDITION_IDX:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"

echo "=============================================================================="
echo "SLURM Job ID:       $SLURM_JOB_ID"
echo "Array Task ID:      $SLURM_ARRAY_TASK_ID (Patient Chunk #$SLURM_ARRAY_TASK_ID)"
echo "Condition Index:    $CONDITION_IDX"
echo "Chunk Size:         $CHUNK_SIZE patients"
echo "CPUs allocated:     $SLURM_CPUS_PER_TASK"
echo "Start time:         $(date)"
echo "=============================================================================="

$PYTHON_BIN run_simulation_chunk.py \
    --condition_idx "$CONDITION_IDX" \
    --chunk_size "$CHUNK_SIZE" \
    --patient_chunk "$SLURM_ARRAY_TASK_ID" \
    --n_workers "$SLURM_CPUS_PER_TASK" \
    --skip_existing

echo "=============================================================================="
echo "Patient chunk $SLURM_ARRAY_TASK_ID completed at $(date)"
echo "=============================================================================="
