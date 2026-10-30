#!/bin/bash
#SBATCH --job-name=hbml_post
#SBATCH --partition=LocalQ
#SBATCH --array=0-23%6
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/post_%A_%a.out
#SBATCH --error=slurm_logs/post_%A_%a.err

# ==============================================================================
# SLURM Job Array: Post-Processing across 24 Conditions
#
# Computes online Hedge weights, regrets, and Clarke Error Grids.
#
# Usage:
#   sbatch slurm/submit_postprocess_array.slurm
# ==============================================================================

set -euo pipefail

mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="./hedge-env/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "=============================================================================="
echo "SLURM Job ID:       $SLURM_JOB_ID"
echo "Array Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Start time:         $(date)"
echo "=============================================================================="

$PYTHON_BIN run_postprocess_chunk.py --condition_idx "$SLURM_ARRAY_TASK_ID"

echo "=============================================================================="
echo "Post-processing task $SLURM_ARRAY_TASK_ID completed at $(date)"
echo "=============================================================================="
