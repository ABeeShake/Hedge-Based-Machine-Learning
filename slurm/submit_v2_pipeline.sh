#!/bin/bash
# ==============================================================================
# slurm/submit_v2_pipeline.sh — Master SLURM Submitter for Engine v2.1 Pipeline
#
# Submits all VM jobs as SLURM batch / array jobs with automatic dependencies:
#   Stage 0: Archive legacy v1 outputs (`archive_v1_outputs.sh`) and run unit tests
#   Stage 1: 2D Grid Array Job across all 24 conditions (`slurm/submit_grid_simulation_array.slurm`)
#   Stage 2: Secondary analysis SLURM array jobs (run in parallel with / after Stage 1):
#            - `slurm/secondary_population.slurm` (within-cohort 5-fold + cross-cohort)
#            - `slurm/secondary_cadence.slurm`    (6 h, 24 h, 48 h refit intervals)
#            - `slurm/secondary_seeds.slurm`      (3-seed stochastic stability)
#   Stage 3: Post-processing, tables, figures, and macro verification (`slurm/run_on_vm.slurm`),
#            automatically chained via `--dependency=afterok` once Stages 1 and 2 finish.
#
# Usage:
#   bash slurm/submit_v2_pipeline.sh
#   bash slurm/submit_v2_pipeline.sh --chunk_size 20 --throttle 4
#   bash slurm/submit_v2_pipeline.sh --dry-run
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PY="./hedge-env/bin/python3"
[ -x "$PY" ] || PY="python3"

CONDITIONS="0-23"
CHUNK_SIZE="20"
THROTTLE="4"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conditions) CONDITIONS="$2"; shift 2 ;;
        --chunk_size|-c) CHUNK_SIZE="$2"; shift 2 ;;
        --throttle|-t) THROTTLE="${2#%}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p slurm_logs

TOTAL_TASKS=$("$PY" run_simulation_chunk.py --conditions "$CONDITIONS" --chunk_size "$CHUNK_SIZE" --count_grid_tasks | tr -d '\r')
MAX_IDX=$((TOTAL_TASKS - 1))
ARRAY_SPEC="0-${MAX_IDX}%${THROTTLE}"

echo "=============================================================================="
echo "HBML Engine v2.1 Full SLURM Pipeline"
echo "=============================================================================="
echo "1. Archive legacy v1 outputs: bash archive_v1_outputs.sh"
echo "2. Main 24-condition grid:    sbatch --array=${ARRAY_SPEC} slurm/submit_grid_simulation_array.slurm"
echo "3. Population models (within): sbatch --array=0-5 slurm/secondary_population.slurm"
echo "4. Population models (cross):  SCHEME=cross sbatch --array=0-5 slurm/secondary_population.slurm"
echo "5. Refit cadence sensitivity:  sbatch --array=0-5 slurm/secondary_cadence.slurm"
echo "6. Seed stability (3 seeds):   sbatch --array=0-5 slurm/secondary_seeds.slurm"
echo "7. Post-processing & macros:   sbatch --dependency=afterok:<ids> slurm/run_on_vm.slurm"
echo "=============================================================================="

if [ "$DRY_RUN" = true ]; then
    echo "[Dry-Run] Exiting without submitting jobs."
    exit 0
fi

echo ">>> Running engine unit tests and archiving legacy v1 outputs..."
"$PY" tests/test_engine.py
bash archive_v1_outputs.sh

export CONDITIONS CHUNK_SIZE
GRID_OUT=$(sbatch --array="${ARRAY_SPEC}" --export=ALL slurm/submit_grid_simulation_array.slurm)
GRID_ID=$(echo "$GRID_OUT" | grep -oE '[0-9]+' | head -n 1)
echo "Submitted Main Grid Array:             Job $GRID_ID"

POP_W_OUT=$(sbatch --array=0-5 --export=ALL,SCHEME=within slurm/secondary_population.slurm)
POP_W_ID=$(echo "$POP_W_OUT" | grep -oE '[0-9]+' | head -n 1)
echo "Submitted Population Models (within):  Job $POP_W_ID"

POP_C_OUT=$(sbatch --array=0-5 --export=ALL,SCHEME=cross slurm/secondary_population.slurm)
POP_C_ID=$(echo "$POP_C_OUT" | grep -oE '[0-9]+' | head -n 1)
echo "Submitted Population Models (cross):   Job $POP_C_ID"

CAD_OUT=$(sbatch --array=0-5 slurm/secondary_cadence.slurm)
CAD_ID=$(echo "$CAD_OUT" | grep -oE '[0-9]+' | head -n 1)
echo "Submitted Refit Cadence Sensitivity:   Job $CAD_ID"

SEED_OUT=$(sbatch --array=0-5 slurm/secondary_seeds.slurm)
SEED_ID=$(echo "$SEED_OUT" | grep -oE '[0-9]+' | head -n 1)
echo "Submitted Multi-Seed Stability:        Job $SEED_ID"

DEP_LIST="${GRID_ID}:${POP_W_ID}:${POP_C_ID}:${CAD_ID}:${SEED_ID}"
POST_OUT=$(sbatch --dependency="afterok:${DEP_LIST}" slurm/run_on_vm.slurm)
POST_ID=$(echo "$POST_OUT" | grep -oE '[0-9]+' | head -n 1)
echo "Submitted Post-Processing & Macros:    Job $POST_ID (afterok:${DEP_LIST})"
echo "=============================================================================="

