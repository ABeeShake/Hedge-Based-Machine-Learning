#!/bin/bash
# ==============================================================================
# slurm/submit_grid.sh — Unified 2D Grid Submitter (Conditions × Patient Chunks)
#
# Automatically calculates the exact SLURM array range required for any
# combination of conditions and patient chunk size, then submits the array job.
#
# Usage:
#   # Submit Weinstock conditions (12-23) in chunks of 20 patients:
#   bash slurm/submit_grid.sh --conditions 12-23 --chunk_size 20 --throttle 4
#
#   # Positional shorthand (Conditions, ChunkSize, Throttle):
#   bash slurm/submit_grid.sh 12-23 20 4
#
#   # Preview mapping and sbatch command without submitting:
#   bash slurm/submit_grid.sh --conditions 12-23 --chunk_size 20 --dry-run
#
#   # List task details table:
#   bash slurm/submit_grid.sh --conditions 12-23 --chunk_size 20 --list
# ==============================================================================

set -euo pipefail

# Move to workspace root
cd "$(dirname "$0")/.."

PYTHON_BIN="./hedge-env/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

CONDITIONS="12-23"
CHUNK_SIZE="20"
THROTTLE="4"
DRY_RUN=false
LIST_ONLY=false

usage() {
    echo "Usage: $0 [OPTIONS] [CONDITIONS] [CHUNK_SIZE] [THROTTLE]"
    echo ""
    echo "Options:"
    echo "  --conditions <spec>   Condition range/list (e.g. '12-23', '0-23', '0,4,12'). [Default: 12-23]"
    echo "  --chunk_size <n>      Number of patients per chunk. [Default: 20]"
    echo "  --throttle <n>        Max concurrent SLURM tasks (%n). [Default: 4]"
    echo "  --dry-run             Display plan and exact sbatch command without submitting."
    echo "  --list                Display full grid task-to-chunk mapping table and exit."
    echo "  -h, --help            Show this help message."
    echo ""
    echo "Examples:"
    echo "  $0 --conditions 12-23 --chunk_size 20 --throttle 4"
    echo "  $0 12-23 20 4"
    echo "  $0 --conditions 0-23 --chunk_size 20 --dry-run"
    exit 0
}

# Parse named arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conditions)
            CONDITIONS="$2"
            shift 2
            ;;
        --chunk_size|-c)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --throttle|-t)
            THROTTLE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Fallback to positional arguments if provided
if [ ${#POSITIONAL_ARGS[@]} -ge 1 ]; then
    CONDITIONS="${POSITIONAL_ARGS[0]}"
fi
if [ ${#POSITIONAL_ARGS[@]} -ge 2 ]; then
    CHUNK_SIZE="${POSITIONAL_ARGS[1]}"
fi
if [ ${#POSITIONAL_ARGS[@]} -ge 3 ]; then
    THROTTLE="${POSITIONAL_ARGS[2]}"
fi

# Clean throttle parameter (strip leading '%')
THROTTLE="${THROTTLE#%}"

# If --list was requested, print the full table and exit
if [ "$LIST_ONLY" = true ]; then
    $PYTHON_BIN run_simulation_chunk.py --conditions "$CONDITIONS" --chunk_size "$CHUNK_SIZE" --list_grid
    exit 0
fi

echo "=============================================================================="
echo "HBML Unified 2D Grid SLURM Submitter"
echo "=============================================================================="
echo "Target Conditions:  $CONDITIONS"
echo "Patient Chunk Size: $CHUNK_SIZE patients"
echo "Concurrency Limit:  %$THROTTLE (max $THROTTLE active tasks, $((THROTTLE * 8)) CPU cores)"
echo "------------------------------------------------------------------------------"

# Calculate total grid tasks via python helper
TOTAL_TASKS=$($PYTHON_BIN run_simulation_chunk.py --conditions "$CONDITIONS" --chunk_size "$CHUNK_SIZE" --count_grid_tasks | tr -d '\r')

if [ -z "$TOTAL_TASKS" ] || [ "$TOTAL_TASKS" -le 0 ]; then
    echo "[Error] No tasks found for condition spec: '$CONDITIONS'"
    exit 1
fi

MAX_IDX=$((TOTAL_TASKS - 1))
ARRAY_SPEC="0-${MAX_IDX}%${THROTTLE}"

echo "Calculated Grid:    $TOTAL_TASKS tasks total (SLURM array range: 0-$MAX_IDX)"
echo "Array Specification: --array=$ARRAY_SPEC"
echo "=============================================================================="

SBATCH_CMD="sbatch --array=\"${ARRAY_SPEC}\" --export=ALL,CONDITIONS=\"${CONDITIONS}\",CHUNK_SIZE=\"${CHUNK_SIZE}\" slurm/submit_grid_simulation_array.slurm"

if [ "$DRY_RUN" = true ]; then
    echo "[Dry-Run Mode] No job submitted. Proposed command:"
    echo ""
    echo "  $SBATCH_CMD"
    echo ""
    echo "To inspect task mappings, run:"
    echo "  bash slurm/submit_grid.sh --conditions $CONDITIONS --chunk_size $CHUNK_SIZE --list"
    exit 0
fi

# Submit job via sbatch
mkdir -p slurm_logs
JOB_OUTPUT=$(sbatch --array="${ARRAY_SPEC}" --export=ALL,CONDITIONS="${CONDITIONS}",CHUNK_SIZE="${CHUNK_SIZE}" slurm/submit_grid_simulation_array.slurm)
echo "Result: $JOB_OUTPUT"

# Extract Job ID
JOB_ID=$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+' | head -n 1)

echo "------------------------------------------------------------------------------"
echo "Monitoring & Management:"
echo "  Check queue:            squeue -u \$USER"
echo "  Check active job:       squeue -j $JOB_ID"
echo "  Check condition table:  python3 run_simulation_chunk.py --list_conditions"
echo "  Cancel job array:       scancel $JOB_ID"
echo "=============================================================================="
