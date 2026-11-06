#!/bin/bash
# reproduce.sh — End-to-end reproduction of all HBML paper results.
#
# This script orchestrates the full pipeline in four stages:
#
#   Stage 1: Online simulation (generate per-patient expert forecasts)
#   Stage 2: Post-processing (compute HBML weights, regrets, Clarke Error Grids)
#   Stage 3: Figure generation (produce all paper figures as PDFs)
#   Stage 4: LaTeX macro generation (write \newcommand values used in the paper)
#
# ─── Quick-start (GPU recommended) ───────────────────────────────────────────
#
#   pip install -r requirements.txt
#   bash reproduce.sh
#
# ─── Data setup ──────────────────────────────────────────────────────────────
#
#   CGMacros: Download from PhysioNet (https://physionet.org/content/cgmacros/)
#             and place per-patient CSVs in ./cgmacros/
#
#   Weinstock 2016: Available from the study authors upon request. Place
#             per-patient CSVs in ./weinstock/
#
#   If you only want to reproduce figures from pre-computed results (and skip
#   the multi-hour simulation), skip Stage 1 and Stage 2; the aggregated
#   CSV files in overleaf/tables/ and the root-level rmse.csv / maxae.csv
#   are included in this repository.
#
# ─── Expected runtime ────────────────────────────────────────────────────────
#
#   Stage 1 (simulation): ~12-48 hrs per dataset on GPU; longer on CPU.
#   Stage 2 (processing): ~15-30 min per dataset.
#   Stage 3 + 4 (figures/macros): < 5 min.
#
# Usage: bash reproduce.sh [--skip-sim] [--skip-process]

set -e

SKIP_SIM=false
SKIP_PROCESS=false

PUSH_OVERLEAF=false

for arg in "$@"; do
    case $arg in
        --skip-sim)       SKIP_SIM=true ;;
        --skip-process)   SKIP_PROCESS=true ;;
        --push-overleaf)  PUSH_OVERLEAF=true ;;
    esac
done

mkdir -p output_logs Outputs

# ─── Stage 1: Simulation ──────────────────────────────────────────────────────
if [ "$SKIP_SIM" = false ]; then
    echo "========================================================================"
    echo "Stage 1: Running online simulations (CGMacros + Weinstock 2016)"
    echo "========================================================================"

    echo "--- CGMacros ---"
    bash cgmacros-horizons.sh

    echo "--- Weinstock 2016 ---"
    bash weinstock-horizons.sh

else
    echo "[Skipping Stage 1: simulation]"
fi

# ─── Stage 2: Post-processing ─────────────────────────────────────────────────
if [ "$SKIP_PROCESS" = false ]; then
    echo "========================================================================"
    echo "Stage 2: Post-processing (HBML aggregation + Clarke Error Grids)"
    echo "========================================================================"
    bash run_all_process_results.sh
else
    echo "[Skipping Stage 2: post-processing]"
fi

# ─── Stage 3: Statistical Tests, Baselines & Figure Generation ────────────────
echo "========================================================================"
echo "Stage 3: Statistical tests, baseline comparators, and paper figures"
echo "========================================================================"
python3 calculate_statistics.py || true
python3 calculate_baselines.py || true
python3 generate_figures.py || true
python3 plot_and_format_volatility.py || true
python3 plot_regime_analysis.py || true
python3 plot_subgroups.py || true
python3 plot_weights.py || true
bash copy_overleaf_images.sh || true

# ─── Stage 4: LaTeX macros ────────────────────────────────────────────────────
echo "========================================================================"
echo "Stage 4: Generating LaTeX result macros"
echo "========================================================================"
python3 generate_results_macros.py

# ─── Optional: Direct Overleaf Push ───────────────────────────────────────────
if [ "$PUSH_OVERLEAF" = true ] && [ -d "overleaf/.git" ]; then
    echo "========================================================================"
    echo "Pushing updated figures, tables, and macros directly to Overleaf..."
    echo "========================================================================"
    git -C overleaf add tables/ images/ sections/ macros/ 2>/dev/null || git -C overleaf add -A
    git -C overleaf commit -m "Update paper figures, tables, and macros from simulation results [$(date '+%Y-%m-%d %H:%M')]" || true
    git -C overleaf push origin main
    echo "✓ Pushed to Overleaf successfully."
fi

echo "========================================================================"
echo "Reproduction complete. Figures are in overleaf/images/."
echo "LaTeX macros are in overleaf/sections/generated_macros.tex."
echo "========================================================================"
