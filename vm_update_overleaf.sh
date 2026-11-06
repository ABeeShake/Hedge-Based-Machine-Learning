#!/bin/bash
# ==============================================================================
# vm_update_overleaf.sh — Direct VM-to-Overleaf Results Pipeline
#
# Runs entirely on the remote VM. Post-processes simulation outputs, computes
# participant-level statistics, baseline comparators, paper figures, and
# LaTeX macros, and then pushes all updated artifacts directly to Overleaf
# (via GitHub origin integration).
#
# Eliminates the need to download any simulation results, logs, or project files
# to your local computer!
#
# Usage (on the VM):
#   bash vm_update_overleaf.sh [OPTIONS]
#
# Options:
#   --skip-process    Skip Stage 1 batch post-processing (use if already run or using SLURM)
#   --skip-stats      Skip Stage 2 statistical tests and baselines
#   --skip-figures    Skip Stage 3 paper figure generation
#   --skip-macros     Skip Stage 4 LaTeX macros generation
#   --no-push         Commit changes to overleaf repo without git pushing
#   --help, -h        Show this help message
#
# Example:
#   # Full end-to-end processing & direct Overleaf push:
#   bash vm_update_overleaf.sh
#
#   # Post-processing was already completed via SLURM; only update figures, macros & Overleaf:
#   bash vm_update_overleaf.sh --skip-process
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_PROCESS=false
SKIP_STATS=false
SKIP_FIGURES=false
SKIP_MACROS=false
NO_PUSH=false

OVERLEAF_REMOTE_URL="${OVERLEAF_GIT_REMOTE:-https://github.com/ABeeShake/-Submission-Template-Hedging-Powered-Predictions.git}"

for arg in "$@"; do
    case $arg in
        --skip-process) SKIP_PROCESS=true ;;
        --skip-stats)   SKIP_STATS=true ;;
        --skip-figures) SKIP_FIGURES=true ;;
        --skip-macros)  SKIP_MACROS=true ;;
        --no-push)      NO_PUSH=true ;;
        --help|-h)
            grep -E "^# " "$0" | cut -c 3-
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Run 'bash $0 --help' for usage."
            exit 1
            ;;
    esac
done

echo "=============================================================================="
echo "          DIRECT VM-TO-OVERLEAF RESULTS PIPELINE                              "
echo "=============================================================================="
echo "Workspace Directory: $SCRIPT_DIR"
echo "Skip Postprocess:    $SKIP_PROCESS"
echo "Skip Statistics:     $SKIP_STATS"
echo "Skip Figures:        $SKIP_FIGURES"
echo "Skip LaTeX Macros:   $SKIP_MACROS"
echo "Git Push to Remote:  $([ "$NO_PUSH" = true ] && echo 'No (commit only)' || echo 'Yes (push to origin main)')"
echo "=============================================================================="

# --- Step 0: Python Environment Discovery ---
if [ -f "./hedge-env/bin/python3" ]; then
    PYTHON_EXE="./hedge-env/bin/python3"
elif [ -f "./hedge-env/bin/python" ]; then
    PYTHON_EXE="./hedge-env/bin/python"
elif [ -f "./venv/bin/python" ]; then
    PYTHON_EXE="./venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_EXE="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python3)"
else
    echo "Error: No Python 3 interpreter found."
    exit 1
fi

export PYTHON="$PYTHON_EXE"
echo ">>> Detected Python: $PYTHON_EXE ($("$PYTHON_EXE" --version))"

# --- Step 0B: Ensure Overleaf Git Repository is Configured on VM ---
echo ""
echo ">>> Verifying Overleaf git repository..."
if [ ! -d "overleaf" ]; then
    echo "Cloning Overleaf git repository from $OVERLEAF_REMOTE_URL..."
    git clone "$OVERLEAF_REMOTE_URL" overleaf
elif [ ! -d "overleaf/.git" ]; then
    echo "Initializing git tracking in existing overleaf/ directory..."
    git -C overleaf init -b main
    git -C overleaf remote add origin "$OVERLEAF_REMOTE_URL" 2>/dev/null || git -C overleaf remote set-url origin "$OVERLEAF_REMOTE_URL"
    git -C overleaf fetch origin main || true
    git -C overleaf reset --mixed origin/main || true
fi

# Ensure latest changes from Overleaf are pulled (prevents non-fast-forward push rejections)
if [ -d "overleaf/.git" ]; then
    echo "Syncing latest commits from Overleaf origin main..."
    git -C overleaf pull --rebase origin main 2>/dev/null || echo "Note: pull --rebase skipped (local commits or clean tree)."
fi


# --- Stage 1: Batch Post-Processing Simulation Outputs ---
if [ "$SKIP_PROCESS" = false ]; then
    echo ""
    echo "=============================================================================="
    echo "STAGE 1: POST-PROCESSING SIMULATION OUTPUTS ON VM"
    echo "=============================================================================="
    if [ -d "./Outputs" ]; then
        bash run_all_process_results.sh ./Outputs
        echo "✓ Post-processing finished."
    else
        echo "Note: ./Outputs directory not found. Skipping batch post-processing."
    fi
else
    echo ""
    echo "[Skipping Stage 1: Post-processing already complete or skipped]"
fi


# --- Stage 2: Statistical Tests, Baselines, and Volatility ---
if [ "$SKIP_STATS" = false ]; then
    echo ""
    echo "=============================================================================="
    echo "STAGE 2: COMPUTING STATISTICAL TESTS & COMPARATOR BASELINES"
    echo "=============================================================================="

    if [ -f "calculate_statistics.py" ]; then
        echo ">>> Running calculate_statistics.py (Wilcoxon signed-rank & paired t-tests)..."
        "$PYTHON_EXE" calculate_statistics.py || echo "Warning: calculate_statistics.py completed with partial data."
    fi

    if [ -f "calculate_baselines.py" ]; then
        echo ">>> Running calculate_baselines.py (Uniform ensemble & Persistence baselines)..."
        "$PYTHON_EXE" calculate_baselines.py || echo "Warning: calculate_baselines.py completed with partial data."
    fi

    if [ -f "plot_and_format_volatility.py" ] && [ -f "volatility_results.csv" ]; then
        echo ">>> Running plot_and_format_volatility.py..."
        "$PYTHON_EXE" plot_and_format_volatility.py || true
    fi
    echo "✓ Statistics and baselines updated."
else
    echo ""
    echo "[Skipping Stage 2: Statistics & Baselines]"
fi


# --- Stage 3: Paper Figure Generation ---
if [ "$SKIP_FIGURES" = false ]; then
    echo ""
    echo "=============================================================================="
    echo "STAGE 3: GENERATING PAPER FIGURES"
    echo "=============================================================================="

    if [ -f "generate_figures.py" ]; then
        echo ">>> Running generate_figures.py (RMSE tables, Max-AE tables, error heatmaps)..."
        "$PYTHON_EXE" generate_figures.py || echo "Warning: generate_figures.py completed with partial settings."
    fi

    if [ -f "plot_subgroups.py" ]; then
        echo ">>> Running plot_subgroups.py (diabetic subgroup analysis)..."
        "$PYTHON_EXE" plot_subgroups.py || true
    fi

    if [ -f "plot_regime_analysis.py" ]; then
        echo ">>> Running plot_regime_analysis.py..."
        "$PYTHON_EXE" plot_regime_analysis.py || true
    fi

    if [ -f "plot_weights.py" ]; then
        SAMPLE_WEIGHTS=$(find Outputs -name "*expweights_eta10.csv" 2>/dev/null | head -n 1)
        if [ -n "$SAMPLE_WEIGHTS" ]; then
            echo ">>> Running plot_weights.py using sample: $SAMPLE_WEIGHTS..."
            "$PYTHON_EXE" plot_weights.py --file "$SAMPLE_WEIGHTS" --output overleaf/images/methods/expert_weights_plot.pdf || true
        else
            echo ">>> Running plot_weights.py with auto-discovery..."
            "$PYTHON_EXE" plot_weights.py --output overleaf/images/methods/expert_weights_plot.pdf || true
        fi
    fi

    if [ -f "copy_overleaf_images.sh" ]; then
        echo ">>> Organizing Clarke error grids and regret curves into overleaf/images/..."
        bash copy_overleaf_images.sh
    fi
    echo "✓ Figures generated and copied to overleaf/images/."
else
    echo ""
    echo "[Skipping Stage 3: Figure generation]"
fi


# --- Stage 4: LaTeX Results Macros Generation ---
if [ "$SKIP_MACROS" = false ]; then
    echo ""
    echo "=============================================================================="
    echo "STAGE 4: GENERATING LATEX RESULTS MACROS"
    echo "=============================================================================="
    if [ -f "generate_results_macros.py" ]; then
        "$PYTHON_EXE" generate_results_macros.py
        echo "✓ Generated macros in overleaf/sections/generated_macros.tex."
    fi
else
    echo ""
    echo "[Skipping Stage 4: LaTeX macro generation]"
fi


# --- Stage 5: Commit and Push Directly to Overleaf ---
echo ""
echo "=============================================================================="
echo "STAGE 5: PUSHING UPDATED RESULTS DIRECTLY TO OVERLEAF"
echo "=============================================================================="

if [ -d "overleaf/.git" ]; then
    echo "Checking Overleaf git status:"
    git -C overleaf status --short

    CHANGES=$(git -C overleaf status --porcelain)
    if [ -n "$CHANGES" ]; then
        echo ""
        echo "Staging updated paper tables, figures, and macros..."
        git -C overleaf add tables/ images/ sections/ macros/ 2>/dev/null || git -C overleaf add -A

        COMMIT_MSG="Update paper figures, tables, and macros from VM simulation results [$(date '+%Y-%m-%d %H:%M')]"
        git -C overleaf commit -m "$COMMIT_MSG"
        echo "✓ Committed changes to overleaf git repository."

        if [ "$NO_PUSH" = false ]; then
            echo ""
            echo "Pushing commits directly from VM to Overleaf (origin main)..."
            git -C overleaf push origin main
            echo ""
            echo "=============================================================================="
            echo "✓ SUCCESS! Results, figures, and macros pushed directly to Overleaf."
            echo "  No local downloads were needed."
            echo "=============================================================================="
        else
            echo "Note: Changes committed locally in overleaf/ repository (--no-push requested)."
        fi
    else
        echo "No changes detected in overleaf/ repository. Overleaf is already up to date!"
    fi
else
    echo "Warning: overleaf/.git directory not found. Cannot push directly to Overleaf."
fi
