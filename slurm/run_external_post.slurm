#!/bin/bash
#SBATCH --job-name=hbml_ext_post
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/ext_post_%j.out
#SBATCH --error=slurm_logs/ext_post_%j.err

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

DATASET="${DATASET:-hall}"
echo "=== [1/2] Post-processing $DATASET conditions ==="
"$PY" run_postprocess_chunk.py --dataset "$DATASET" --all --no_plot_ceg

echo "=== [2/2] Running self-contained external cohort report for $DATASET ==="
"$PY" evaluate_external_cohort.py --dataset "$DATASET"

echo "=== Finished $DATASET external evaluation at $(date) ==="
