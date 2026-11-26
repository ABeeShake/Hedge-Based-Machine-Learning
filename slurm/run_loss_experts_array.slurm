#!/bin/bash
#SBATCH --job-name=hbml_loss_exp
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/loss_exp_%A_%a.out
#SBATCH --error=slurm_logs/loss_exp_%A_%a.err

# ==============================================================================
# Retrain XGBoost / NODE / N-HITS under an alternative training loss into
# isolated Outputs_loss/<EXPERT_LOSS>/ (never overwrites baseline Outputs/).
#
# Example submission (full-history 30-min, 2-hr, 5-hr for Hall or Weinstock):
#   sbatch --export=ALL,DATASET=hall,EXPERT_LOSS=pinball20,CONDITIONS=3,7,11,CHUNK_SIZE=15 --array=0-11 slurm/run_loss_experts_array.slurm
# ==============================================================================

set -euo pipefail
mkdir -p slurm_logs
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

PY="./hedge-env/bin/python3"
[ -x "$PY" ] || PY="python3"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export FAST_ENGINE_DEVICE="cpu"

"$PY" run_simulation_chunk.py \
  --dataset "${DATASET:-weinstock}" \
  --conditions "${CONDITIONS:-3,7,11}" \
  --chunk_size "${CHUNK_SIZE:-20}" \
  --grid_task_id "${SLURM_ARRAY_TASK_ID}" \
  --n_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --expert_loss "${EXPERT_LOSS:-huber20}"
