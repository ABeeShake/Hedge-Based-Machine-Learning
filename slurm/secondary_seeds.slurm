#!/bin/bash
# Multi-seed stochastic stability secondary analysis: one array task per (dataset, seed).
#   sbatch --array=0-5 slurm/secondary_seeds.slurm
#SBATCH --job-name=hbml_seeds
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/seeds_%A_%a.out
#SBATCH --error=slurm_logs/seeds_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p slurm_logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" FAST_ENGINE_DEVICE="cpu"
PY=./hedge-env/bin/python3; [ -x "$PY" ] || PY=python3
DATASETS=(weinstock weinstock weinstock cgmacros cgmacros cgmacros)
SEEDS=(0 1 2 0 1 2)
i=${SLURM_ARRAY_TASK_ID}
"$PY" secondary_seed_stability.py --stage simulate --dataset "${DATASETS[$i]}" \
    --seeds "${SEEDS[$i]}" --n_workers "$SLURM_CPUS_PER_TASK"

