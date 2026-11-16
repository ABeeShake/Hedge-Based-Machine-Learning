#!/bin/bash
# Population-model secondary analysis: one array task per (dataset, horizon), within-cohort folds.
#   sbatch --array=0-5 slurm/secondary_population.slurm
#   SCHEME=cross sbatch --array=0-5 slurm/secondary_population.slurm     # cross-cohort training
#SBATCH --job-name=hbml_popmodels
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/popmodels_%A_%a.out
#SBATCH --error=slurm_logs/popmodels_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p slurm_logs
export CUDA_VISIBLE_DEVICES="" FAST_ENGINE_DEVICE="cpu"
PY=./hedge-env/bin/python3; [ -x "$PY" ] || PY=python3
DATASETS=(weinstock weinstock weinstock cgmacros cgmacros cgmacros)
HORIZONS=(half 2 5 half 2 5)
i=${SLURM_ARRAY_TASK_ID}
"$PY" secondary_population_models.py --stage train --scheme "${SCHEME:-within}" \
    --dataset "${DATASETS[$i]}" --horizon "${HORIZONS[$i]}" --threads "$SLURM_CPUS_PER_TASK"
