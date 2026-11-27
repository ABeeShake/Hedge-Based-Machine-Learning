#!/bin/bash
#SBATCH --job-name=hbml_all_post
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/all_post_%j.out
#SBATCH --error=slurm_logs/all_post_%j.err

# ==============================================================================
# Consolidated zero-retraining cached post-processing (with canonical START=288):
#   1. Validation & M2-A mechanism audits (weinstock, cgmacros)
#   2. External cohort evaluations (hall, dubosson -- including M2-A no-XGB-NODE)
#   3. Cached alternative-loss & Mixture-EG variants (all 4 cohorts)
#   4. Expert-retrained loss evaluations (asym_hypo, huber20, pinball20)
# ==============================================================================

set -euo pipefail
mkdir -p slurm_logs validation_results loss_results
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

PY="./hedge-env/bin/python3"
[ -x "$PY" ] || PY="python3"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export FAST_ENGINE_DEVICE="cpu"

echo "=== [1/4] Validation & M2-A pool mechanism (weinstock, cgmacros) ==="
"$PY" validate_engine_outputs.py --datasets weinstock,cgmacros --n-boot 2000
cp -f Outputs/validation/* validation_results/
"$PY" analyze_pool_mechanism.py --datasets weinstock,cgmacros --n_boot 5000

echo "=== [2/4] External cohort post-processing (hall, dubosson) ==="
for DS in hall dubosson; do
  "$PY" evaluate_external_cohort.py --dataset "$DS"
  mkdir -p "external_results/${DS}"
  cp -f "Outputs/external/${DS}/"* "external_results/${DS}/"
done

echo "=== [3/4] Cached alternative online loss & Mixture-EG variants ==="
"$PY" evaluate_loss_variants.py --datasets weinstock,cgmacros,hall,dubosson --outputs-root Outputs --out-dir loss_results --n-boot 2000

echo "=== [4/4] Expert-retrained alternative losses ==="
for EL in asym_hypo huber20 pinball20; do
  if [ -d "Outputs_loss/${EL}" ]; then
    "$PY" evaluate_loss_experts.py --expert-loss "$EL" --datasets weinstock,cgmacros,hall,dubosson --n-boot 2000
  fi
done

echo "=== All consolidated cached post-processing complete at $(date) ==="
