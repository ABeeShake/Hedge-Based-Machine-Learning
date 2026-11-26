#!/bin/bash
# Prepare and submit locked Engine v2.3 external evaluation on any GlucoBench dataset via SLURM.
#
# Usage:
#   bash slurm/submit_external_pipeline.sh [hall|dubosson|ohiot1dm]
#
# Default dataset is `hall` (Hall et al. 2018, PLOS Biology, N=57 subjects, CC BY 4.0,
# directly downloadable from IrinaStatsLab/GlucoBench raw_data.zip).

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p slurm_logs

PY="./hedge-env/bin/python3"
[ -x "$PY" ] || PY="python3"

DATASET="${1:-hall}"
CONDITIONS="${CONDITIONS:-0-11}"
CHUNK_SIZE="${CHUNK_SIZE:-6}"

NUM_CSV=$(ls -1 "$DATASET"/*.csv 2>/dev/null | grep -cv "_manifest.csv" || true)
if [ "$NUM_CSV" -eq 0 ]; then
    if [ "$DATASET" = "hall" ] || [ "$DATASET" = "dubosson" ]; then
        echo "Preparing GlucoBench dataset '$DATASET'..."
        "$PY" prepare_glucobench.py --dataset "$DATASET"
        NUM_CSV=$(ls -1 "$DATASET"/*.csv 2>/dev/null | grep -cv "_manifest.csv" || true)
    else
        echo "Error: no participant CSVs in $DATASET/. Prepare data before submitting."
        exit 1
    fi
fi
echo "Found $NUM_CSV participant CSVs in $DATASET/."

echo "Running core engine unit tests..."
"$PY" tests/test_engine.py

TOTAL_TASKS=$("$PY" run_simulation_chunk.py --dataset "$DATASET" --conditions "$CONDITIONS" --chunk_size "$CHUNK_SIZE" --count_grid_tasks)
if [ "$TOTAL_TASKS" -le 0 ]; then
    echo "Error: zero grid tasks computed for $DATASET."
    exit 1
fi
LAST_TASK=$((TOTAL_TASKS - 1))
echo "Submitting $TOTAL_TASKS grid tasks (0-${LAST_TASK}%16) for $DATASET..."

SIM_JOB=$(sbatch --parsable \
  --array="0-${LAST_TASK}%16" \
  --export="ALL,DATASET=${DATASET},CONDITIONS=${CONDITIONS},CHUNK_SIZE=${CHUNK_SIZE}" \
  slurm/run_ohiot1dm_sim_array.slurm)
echo "Submitted simulation array: $SIM_JOB"

POST_JOB=$(sbatch --parsable \
  --dependency="afterok:${SIM_JOB}" \
  --export="ALL,DATASET=${DATASET}" \
  slurm/run_ohiot1dm_post.slurm)
echo "Submitted dependent post-processing & report job: $POST_JOB"
