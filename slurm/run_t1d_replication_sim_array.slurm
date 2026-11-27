#!/bin/bash
#SBATCH --job-name=hbml_t1d_sim
#SBATCH --partition=LocalQ
#SBATCH --array=0-647%32
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/t1d_sim_%A_%a.out
#SBATCH --error=slurm_logs/t1d_sim_%A_%a.err

# ==============================================================================
# Pre-registered T1D replication (reviews/preregistration_T1D_replication.md):
# locked Engine v2.3 forecasts for azt1d, ohiot1dm, brist1d and brist1d_15min.
#
# One array task = one (dataset, condition, participant): CHUNK_SIZE=1, so every task
# writes a single participant's forecast file plus its .meta.json.
#   azt1d 23 x 12 = 276 | ohiot1dm 12 x 12 = 144 | brist1d 14 x 12 = 168 | brist1d_15min 5 x 12 = 60
#   Total 648 tasks (array 0-647), at most 32 running at once. One participant per task means one
#   worker process, hence 1 CPU per task (the engine parallelises across participants only).
#
# Resuming: completed participants are skipped (--skip_existing checks the .meta.json against
# the current engine settings). If tasks hit the time limit or fail, resubmit the same command;
# only unfinished participants are recomputed. To rerun only specific tasks:
#   sbatch --array=17,203,590 slurm/run_t1d_replication_sim_array.slurm
#
# Submit (simulation, then post-processing once every task has succeeded):
#   SIM=$(sbatch --parsable slurm/run_t1d_replication_sim_array.slurm)
#   sbatch --dependency=afterok:$SIM slurm/run_t1d_replication_post.slurm
#
# The first task to start verifies the Parquet checksums, converts the engine CSVs if they are
# missing (prepare_parquet_cohorts.py) and runs tests/test_engine.py, under a file lock; the other
# tasks wait for it. A failed checksum or test stops every task.
# ==============================================================================

set -euo pipefail
mkdir -p slurm_logs
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export FAST_ENGINE_DEVICE="cpu"

PY="./hedge-env/bin/python3"
[ -x "$PY" ] || PY="python3"

DATASETS=(azt1d ohiot1dm brist1d brist1d_15min)
CHUNK_SIZE=1
TASK="${SLURM_ARRAY_TASK_ID:?run this script with sbatch --array}"

# --- one-time data check / conversion (serialised with flock) -------------------
(
    flock -x 9
    sha256sum -c --quiet - <<'EOF'
e1a2c53d1a4dda7a972aae501e73daf2c802d0c6f3fea84ad27e05c0a32f0df0  AZT1D.parquet
a41b443d3c963befcaef74ff5e32c87b721e974fdb5ea725483f90ad277c4acf  OhioT1DM.parquet
d7e93bf1d61b823a6cd4fa89f0ee3b5d1e74aac7e4f052530d2e51b0d60757ed  BrisT1D.parquet
EOF
    missing=0
    for ds in "${DATASETS[@]}"; do
        [ -f "$ds/${ds}_manifest.csv" ] || missing=1
    done
    if [ "$missing" -eq 1 ]; then
        READER="pandas"
        "$PY" -c "import pandas, pyarrow" 2>/dev/null || READER="pure"
        echo "Converting Parquet cohorts (reader: $READER)..."
        "$PY" prepare_parquet_cohorts.py --reader "$READER"
    fi
    marker="slurm_logs/.t1d_tests_passed_${SLURM_ARRAY_JOB_ID:-manual}"
    if [ ! -f "$marker" ]; then
        echo "Running core engine unit tests..."
        "$PY" tests/test_engine.py
        touch "$marker"
    fi
) 9>slurm_logs/.t1d_prepare.lock

# --- map the global array index to (dataset, local grid task) -------------------
offset=0
DATASET=""
LOCAL=""
for ds in "${DATASETS[@]}"; do
    n=$("$PY" run_simulation_chunk.py --dataset "$ds" --conditions 0-11 --chunk_size "$CHUNK_SIZE" --count_grid_tasks)
    if [ "$TASK" -lt $((offset + n)) ]; then
        DATASET="$ds"
        LOCAL=$((TASK - offset))
        break
    fi
    offset=$((offset + n))
done
if [ -z "$DATASET" ]; then
    echo "Array index $TASK is beyond the $offset grid tasks of ${DATASETS[*]}; nothing to do."
    exit 0
fi

echo "=============================================================================="
echo "SLURM Job ID:       ${SLURM_JOB_ID:-}  (array ${SLURM_ARRAY_JOB_ID:-}_${TASK})"
echo "Dataset:            $DATASET  (local grid task $LOCAL)"
echo "Node:               ${SLURMD_NODENAME:-}"
echo "CPUs allocated:     ${SLURM_CPUS_PER_TASK:-1}"
echo "Python interpreter: $PY"
echo "Start time:         $(date)"
echo "=============================================================================="

"$PY" run_simulation_chunk.py \
    --dataset "$DATASET" \
    --conditions 0-11 \
    --chunk_size "$CHUNK_SIZE" \
    --grid_task_id "$LOCAL" \
    --n_workers "${SLURM_CPUS_PER_TASK:-1}" \
    --skip_existing

echo "=============================================================================="
echo "Task $TASK ($DATASET #$LOCAL) completed at $(date)"
echo "=============================================================================="
