#!/bin/bash
#SBATCH --job-name=hbml_val_mech
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/val_mech_%j.out
#SBATCH --error=slurm_logs/val_mech_%j.err

# ==============================================================================
# Zero-retraining validation & extended mechanism audits on cached Outputs/:
#   1. validate_engine_outputs.py  (fit windows, N=200/195 sensitivity, gap tertiles)
#   2. analyze_pool_mechanism.py   (interleaved (h+1)-learner, short-context pool, LOO)
#   3. evaluate_clinical_and_weights.py --hypo-only  (cluster PPV + calendar-day FAR)
# ==============================================================================

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

echo "=== [1/3] Running cached-output validation & missing-data audit ==="
"$PY" validate_engine_outputs.py --datasets weinstock,cgmacros --n-boot 2000

echo "=== [2/3] Running extended mechanism & context-asymmetry ablations ==="
"$PY" analyze_pool_mechanism.py --datasets weinstock,cgmacros --n_boot 5000

echo "=== [3/3] Re-running clinical hypoglycaemia metrics (cluster PPV & calendar FAR) ==="
"$PY" evaluate_clinical_and_weights.py --hypo-only

echo "=== All zero-retraining audits complete at $(date) ==="
