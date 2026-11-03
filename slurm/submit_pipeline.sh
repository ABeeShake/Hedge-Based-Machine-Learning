#!/bin/bash
# ==============================================================================
# slurm/submit_pipeline.sh — Master Pipeline Submitter with SLURM Job Dependencies
#
# Submits the full HBML pipeline to SLURM in structured batches:
#   Stage 1: Simulation array jobs (0-23 throttled to %4 concurrent)
#   Stage 2: Post-processing array jobs (triggered only after Stage 1 finishes)
#   Stage 3 & 4: Paper figures and LaTeX macros (triggered after Stage 2 finishes)
#
# Usage:
#   bash slurm/submit_pipeline.sh
#   bash slurm/submit_pipeline.sh --sim-only
#   bash slurm/submit_pipeline.sh --post-only
# ==============================================================================

set -euo pipefail

mkdir -p slurm_logs

SIM_ONLY=false
POST_ONLY=false

for arg in "$@"; do
    case $arg in
        --sim-only)  SIM_ONLY=true ;;
        --post-only) POST_ONLY=true ;;
    esac
done

if [ "$POST_ONLY" = true ]; then
    echo "Submitting Stage 2 (Post-processing Array)..."
    POST_JOB=$(sbatch --parsable slurm/submit_postprocess_array.slurm)
    echo "Submitted Post-processing Array: Job ID $POST_JOB"
    exit 0
fi

echo "=============================================================================="
echo "Submitting Full HBML SLURM Pipeline"
echo "=============================================================================="

# 1. Submit Stage 1 Simulation Array
echo ">>> Submitting Stage 1 (Simulation Array)..."
SIM_JOB=$(sbatch --parsable slurm/submit_simulation_array.slurm)
echo "    Submitted Simulation Array: Job ID $SIM_JOB"

if [ "$SIM_ONLY" = true ]; then
    echo "Pipeline submission complete (simulation only requested)."
    exit 0
fi

# 2. Submit Stage 2 Post-Processing Array with dependency on Stage 1
echo ">>> Submitting Stage 2 (Post-processing Array with dependency afterok:$SIM_JOB)..."
POST_JOB=$(sbatch --parsable --dependency=afterok:$SIM_JOB slurm/submit_postprocess_array.slurm)
echo "    Submitted Post-processing Array: Job ID $POST_JOB"

# 3. Submit Stage 3 & 4 Figures and LaTeX Macros
echo ">>> Submitting Stages 3 & 4 (Figures and Macros with dependency afterok:$POST_JOB)..."
cat << 'EOF' > slurm/generate_all_figures.slurm
#!/bin/bash
#SBATCH --job-name=hbml_figs
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/figures_%j.out
#SBATCH --error=slurm_logs/figures_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="./hedge-env/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "Generating paper figures, statistical tests, baselines, and LaTeX macros..."
$PYTHON_BIN generate_figures.py
$PYTHON_BIN calculate_statistics.py
$PYTHON_BIN calculate_baselines.py
$PYTHON_BIN plot_and_format_volatility.py
$PYTHON_BIN plot_regime_analysis.py
$PYTHON_BIN plot_subgroups.py
$PYTHON_BIN plot_weights.py
bash copy_overleaf_images.sh
$PYTHON_BIN generate_results_macros.py
echo "All figures, tables, and macros generated successfully!"
EOF

FIG_JOB=$(sbatch --parsable --dependency=afterok:$POST_JOB slurm/generate_all_figures.slurm)
echo "    Submitted Figures & Macros: Job ID $FIG_JOB"

echo "=============================================================================="
echo "Pipeline submitted successfully!"
echo "  Stage 1 (Simulations): Job ID $SIM_JOB"
echo "  Stage 2 (Postprocess): Job ID $POST_JOB (depends on $SIM_JOB)"
echo "  Stage 3 (Figures):     Job ID $FIG_JOB  (depends on $POST_JOB)"
echo ""
echo "Monitor queue with:      squeue -u $USER"
echo "Check condition status:  python3 run_simulation_chunk.py --list_conditions"
echo "=============================================================================="
