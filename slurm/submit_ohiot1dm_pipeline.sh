#!/bin/bash
# Submit locked Engine v2.3 external evaluation on OhioT1DM via SLURM.
#
# Usage:
#   bash slurm/submit_ohiot1dm_pipeline.sh [RAW_OHIO_DIR]
#
# If RAW_OHIO_DIR is provided (or $OHIO_RAW_DIR is set), prepares ohiot1dm/*.csv first.
# Otherwise verifies ohiot1dm/*.csv already exists, runs fast core tests, submits the
# 12-condition x participant-chunk SLURM simulation array, and chains post-processing.

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p slurm_logs

PY="./hedge-env/bin/python3"
[ -x "$PY" ] || PY="python3"

DATASET="ohiot1dm"
RAW_DIR="${1:-${OHIO_RAW_DIR:-}}"
CONDITIONS="${CONDITIONS:-0-11}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"

if [ -n "$RAW_DIR" ]; then
    echo "Preparing $DATASET from $RAW_DIR..."
    "$PY" prepare_ohiot1dm.py --input-dir "$RAW_DIR" --output-dir "$DATASET"
fi

NUM_CSV=$(ls -1 "$DATASET"/*.csv 2>/dev/null | grep -cv "${DATASET}_manifest.csv" || true)
if [ "$NUM_CSV" -eq 0 ]; then
    echo "Error: no participant CSVs in $DATASET/. Pass raw OhioT1DM XML/CSV directory as first arg."
    exit 1
fi
echo "Found $NUM_CSV participant CSVs in $DATASET/."

echo "Running core engine unit tests..."
"$PY" tests/test_engine.py

TOTAL_TASKS=$("$PY" run_simulation_chunk.py --dataset "$DATASET" --conditions "$CONDITIONS" --chunk_size "$CHUNK_SIZE" --count_grid_tasks)
if [ "$TOTAL_TASKS" -le 0 ]; then
    echo "Error: zero grid tasks computed."
    exit 1
fi
LAST_TASK=$((TOTAL_TASKS - 1))
echo "Submitting $TOTAL_TASKS grid tasks (0-${LAST_TASK}%16) for $DATASET..."

SIM_JOB=$(sbatch --parsable \
  --array="0-${LAST_TASK}%16" \
  --export="ALL,DATASET=${DATASET},CONDITIONS=${CONDITIONS},CHUNK_SIZE=${CHUNK_SIZE}" \
  slurm/run_external_sim_array.slurm)
echo "Submitted simulation array: $SIM_JOB"

POST_JOB=$(sbatch --parsable \
  --dependency="afterok:${SIM_JOB}" \
  --export="ALL,DATASET=${DATASET}" \
  slurm/run_external_post.slurm)
echo "Submitted dependent post-processing & report job: $POST_JOB"
