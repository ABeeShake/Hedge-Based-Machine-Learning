#!/bin/bash
#SBATCH --job-name=hbml_sim
#SBATCH --partition=LocalQ
#SBATCH --array=0-23%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/sim_%A_%a.out
#SBATCH --error=slurm_logs/sim_%A_%a.err

# ==============================================================================
# SLURM Job Array: Online Forecasting Simulation across 24 Experimental Conditions
#
# Runs 24 conditions (indices 0 to 23).
# The '%4' throttles execution to at most 4 concurrent conditions at any time,
# using at most 4 * 8 = 32 CPU cores total, avoiding starving other cluster users.
#
# Submit command:
#   sbatch slurm/submit_simulation_array.slurm
#
# To run a subset of conditions (e.g., conditions 0 to 4):
#   sbatch --array=0-4%2 slurm/submit_simulation_array.slurm
# ==============================================================================

set -euo pipefail

# Create log directory if it does not exist
mkdir -p slurm_logs

# Move to workspace directory
cd "$SLURM_SUBMIT_DIR"

# Prevent OpenMP / MKL / BLAS thread explosion across worker processes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Use virtual environment python
PYTHON_BIN="./hedge-env/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "=============================================================================="
echo "SLURM Job ID:       $SLURM_JOB_ID"
echo "Array Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Node:               $SLURMD_NODENAME"
echo "CPUs allocated:     $SLURM_CPUS_PER_TASK"
echo "Python interpreter: $PYTHON_BIN"
echo "Start time:         $(date)"
echo "=============================================================================="

$PYTHON_BIN run_simulation_chunk.py \
    --condition_idx "$SLURM_ARRAY_TASK_ID" \
    --n_workers "$SLURM_CPUS_PER_TASK" \
    --skip_existing

echo "=============================================================================="
echo "Condition $SLURM_ARRAY_TASK_ID completed at $(date)"
echo "=============================================================================="
