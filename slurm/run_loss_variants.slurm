#!/bin/bash
#SBATCH --job-name=hbml_loss_var
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/loss_var_%j.out
#SBATCH --error=slurm_logs/loss_var_%j.err

# ==============================================================================
# Fast cached-output simulation of alternative online Hedge loss functions
# (MAE, Huber15/25, Pinball-20%, AsymHypo) & direct Mixture-EG learners across
# all 48 configurations (Weinstock, CGMacros, Hall 2018, Dubosson D1NAMO).
# Zero base-expert retraining required.
# ==============================================================================

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

"$PY" evaluate_loss_variants.py \
  --datasets "${DATASETS:-weinstock,cgmacros,hall,dubosson}" \
  --outputs-root "${OUTPUTS_ROOT:-Outputs}" \
  --out-dir "${OUT_DIR:-loss_results}" \
  --n-boot "${N_BOOT:-2000}"

echo "=== Alternative loss & Mixture-EG audit complete at $(date) ==="
