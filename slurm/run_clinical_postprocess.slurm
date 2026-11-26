#!/bin/bash
#SBATCH --job-name=hbml_hypo_post
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/hypo_post_%j.out
#SBATCH --error=slurm_logs/hypo_post_%j.err

set -euo pipefail
mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"

PY="./hedge-env/bin/python3"
[ -x "$PY" ] || PY="python3"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export FAST_ENGINE_DEVICE="cpu"

echo "=== Re-analysing saved forecasts for A3 (chance baseline + threshold sweep) ==="
"$PY" evaluate_clinical_and_weights.py --hypo-only
"$PY" generate_results_macros.py --strict
"$PY" check_macros.py
echo "=== Finished A3 post-processing at $(date) ==="

