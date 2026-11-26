#!/bin/bash
#SBATCH --job-name=hbml_loss_epost
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/loss_epost_%j.out
#SBATCH --error=slurm_logs/loss_epost_%j.err

set -euo pipefail
mkdir -p slurm_logs loss_results
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

PY="./hedge-env/bin/python3"
[ -x "$PY" ] || PY="python3"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export FAST_ENGINE_DEVICE="cpu"

"$PY" evaluate_loss_experts.py \
  --expert-loss "${EXPERT_LOSS:-huber20}" \
  --datasets "${DATASETS:-weinstock,cgmacros,hall,dubosson}" \
  --n-boot "${N_BOOT:-2000}"

echo "=== Alternative base-expert loss evaluation complete at $(date) ==="
