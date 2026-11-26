#!/bin/bash
#SBATCH --job-name=hbml_ext_sim
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/ext_sim_%A_%a.out
#SBATCH --error=slurm_logs/ext_sim_%A_%a.err

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

"$PY" run_simulation_chunk.py \
  --dataset "${DATASET:-hall}" \
  --conditions "${CONDITIONS:-0-11}" \
  --chunk_size "${CHUNK_SIZE:-6}" \
  --grid_task_id "${SLURM_ARRAY_TASK_ID}" \
  --n_workers "${SLURM_CPUS_PER_TASK:-4}"
