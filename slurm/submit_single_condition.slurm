#!/bin/bash
#SBATCH --job-name=hbml_single
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/single_cond_%j.out
#SBATCH --error=slurm_logs/single_cond_%j.err

# ==============================================================================
# SLURM Batch Script: Single Experimental Condition Execution
#
# Usage:
#   sbatch --export=CONDITION_IDX=4 slurm/submit_single_condition.slurm
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

echo "=============================================================================="
echo "SLURM Job ID:       $SLURM_JOB_ID"
echo "Condition Index:    $CONDITION_IDX"
echo "CPUs allocated:     $SLURM_CPUS_PER_TASK"
echo "Start time:         $(date)"
echo "=============================================================================="

$PYTHON_BIN run_simulation_chunk.py \
    --condition_idx "$CONDITION_IDX" \
    --n_workers "$SLURM_CPUS_PER_TASK" \
    --skip_existing

echo "=============================================================================="
echo "Job finished at $(date)"
echo "=============================================================================="
