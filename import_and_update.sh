#!/bin/bash
# ==============================================================================
# import_and_update.sh
#
# Master script to import results and updated code from the remote GPU VM,
# execute the full analysis pipeline locally (handling partial experimental
# settings gracefully), and update the Overleaf repository.
#
# Usage:
#   bash import_and_update.sh [USER@HOST] [REMOTE_DIR] [OPTIONS]
#
# Defaults:
#   Host:       abhishekdevarajan@jsb3.stat.ucla.edu (or $VM_HOST)
#   Remote Dir: ~/HedgeProject (or $VM_DIR)
#
# Options:
#   --skip-sync       Skip remote SSH/rsync (use when results already imported)
#   --skip-process    Skip batch post-processing of simulation outputs
#   --skip-figures    Skip figure generation
#   --push            Automatically git push changes to Overleaf
#   --help            Show this help message
# ==============================================================================

set -e

# --- Configuration & Defaults ---
REMOTE_HOST="${VM_HOST:-abhishekdevarajan@jsb3.stat.ucla.edu}"
REMOTE_DIR="${VM_DIR:-~/HedgeProject}"

SKIP_SYNC=false
SKIP_PROCESS=false
SKIP_FIGURES=false
AUTO_PUSH=false
REMOTE_ONLY=false

# Positional and flag parsing
for arg in "$@"; do
    case $arg in
        --remote-only|--remote) REMOTE_ONLY=true ;;
        --skip-sync)     SKIP_SYNC=true ;;
        --skip-process)  SKIP_PROCESS=true ;;
        --skip-figures)  SKIP_FIGURES=true ;;
        --push)          AUTO_PUSH=true ;;
        --help|-h)
            grep -E "^# " "$0" | cut -c 3-
            exit 0
            ;;
        *@*)
            REMOTE_HOST="$arg"
            ;;
        ~/*|/*)
            REMOTE_DIR="$arg"
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$REMOTE_ONLY" = true ]; then
    echo "========================================================================"
    echo "          DIRECT REMOTE VM-TO-OVERLEAF EXECUTION                        "
    echo "========================================================================"
    echo "Triggering direct pipeline on remote host: $REMOTE_HOST:$REMOTE_DIR"
    echo "No files will be downloaded to your local computer."
    echo "========================================================================"
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR && bash vm_update_overleaf.sh"
    echo "========================================================================"
    echo "✓ Remote Overleaf pipeline completed directly on VM."
    echo "========================================================================"
    exit 0
fi

echo "========================================================================"
echo "          HBML SIMULATION IMPORT & OVERLEAF UPDATE PIPELINE             "
echo "========================================================================"
echo "Local Directory:   $SCRIPT_DIR"
echo "Remote Source:     $REMOTE_HOST:$REMOTE_DIR"
echo "Skip Remote Sync:  $SKIP_SYNC"
echo "Skip Processing:   $SKIP_PROCESS"
echo "Skip Figures:      $SKIP_FIGURES"
echo "Auto-Push Overleaf: $AUTO_PUSH"
echo "========================================================================"

# --- Step 1: Remote Sync via SSH Connection Multiplexing ---
if [ "$SKIP_SYNC" = false ]; then
    echo ""
    echo "========================================================================"
    echo "STAGE 1: IMPORTING UPDATED CODE & SIMULATION RESULTS FROM VM"
    echo "========================================================================"
    echo "Establishing multiplexed SSH connection to $REMOTE_HOST..."
    echo "(You may be prompted for your SSH password ONCE below)"
    echo "------------------------------------------------------------------------"

    CONTROL_SOCKET="/tmp/ssh-mux-%r@%h:%p"
    SSH_MUX_OPTS="-o ControlMaster=auto -o ControlPath=${CONTROL_SOCKET} -o ControlPersist=10m"

    # Trap to cleanly close SSH ControlMaster on exit/interrupt
    cleanup_ssh() {
        if ssh -O check -o ControlPath="${CONTROL_SOCKET}" "$REMOTE_HOST" 2>/dev/null; then
            echo "Closing SSH multiplex connection..."
            ssh -O exit -o ControlPath="${CONTROL_SOCKET}" "$REMOTE_HOST" 2>/dev/null || true
        fi
    }
    trap cleanup_ssh EXIT INT TERM

    # Ensure remote directory exists and test connection (single password prompt here)
    ssh ${SSH_MUX_OPTS} "$REMOTE_HOST" "test -d $REMOTE_DIR" || {
        echo "Error: Directory $REMOTE_DIR does not exist on $REMOTE_HOST."
        exit 1
    }

    echo "✓ SSH connection established."

    # 1A: Pull updated code files (user notes remote code files have been updated)
    echo ""
    echo ">>> Step 1A: Pulling updated code files and scripts from VM..."
    rsync -av \
        -e "ssh ${SSH_MUX_OPTS}" \
        --exclude '.git/' \
        --exclude 'venv/' \
        --exclude '.venv/' \
        --exclude 'hedge-env/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude '.DS_Store' \
        --exclude '*.log' \
        --exclude 'output-data_weinstock_stdout.log' \
        --exclude 'Outputs/' \
        --exclude 'Outputs_debug/' \
        --exclude 'cgmacros/data/' \
        --exclude 'weinstock/full_data/' \
        --exclude 'lightning_logs/' \
        --exclude 'scratch/' \
        "$REMOTE_HOST:$REMOTE_DIR/" ./

    echo "✓ Code files synchronized."

    # 1B: Pull simulation results and metrics
    echo ""
    echo ">>> Step 1B: Pulling simulation outputs and metric files from VM..."
    mkdir -p Outputs overleaf/images overleaf/tables overleaf/sections images

    RSYNC_RESULTS_FILTER=(
        # Raw outputs from running simulations
        --include='Outputs/'
        --include='Outputs/***'
        --exclude='Outputs/**/node/'
        --exclude='Outputs/**/*.pt'
        --exclude='Outputs/**/__pycache__/'
        # Summary CSV metrics
        --include='rmse.csv'
        --include='maxae.csv'
        --include='volatility_results.csv'
        --include='output_timings.csv'
        # Overleaf tables, macros, and images if generated remotely
        --include='overleaf/'
        --include='overleaf/tables/***'
        --include='overleaf/images/***'
        --include='overleaf/sections/***'
        --include='overleaf/macros/***'
        --include='images/***'
        --exclude='*'
    )

    rsync -avz --progress \
        -e "ssh ${SSH_MUX_OPTS}" \
        "${RSYNC_RESULTS_FILTER[@]}" \
        "$REMOTE_HOST:$REMOTE_DIR/" ./

    echo "✓ Results synchronized."
else
    echo "[Skipping Stage 1: remote sync]"
fi


# --- Step 2: Python Environment Discovery ---
echo ""
echo "========================================================================"
echo "STAGE 2: DETECTING PYTHON ENVIRONMENT"
echo "========================================================================"

if [ -f "./hedge-env/bin/python" ]; then
    PYTHON_EXE="./hedge-env/bin/python"
elif [ -f "./venv/bin/python" ]; then
    PYTHON_EXE="./venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_EXE="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python3)"
else
    echo "Error: No Python 3 interpreter found."
    exit 1
fi

export PYTHON="$PYTHON_EXE"
echo "Using Python binary: $PYTHON_EXE ($($PYTHON_EXE --version))"


# --- Step 3: Run Post-Processing on Simulation Outputs ---
if [ "$SKIP_PROCESS" = false ]; then
    echo ""
    echo "========================================================================"
    echo "STAGE 3: BATCH POST-PROCESSING SIMULATION OUTPUTS"
    echo "========================================================================"
    echo "Note: Settings that have not completed or run yet will be skipped safely."
    echo "------------------------------------------------------------------------"
    
    bash run_all_process_results.sh ./Outputs
    echo "✓ Post-processing finished."
else
    echo "[Skipping Stage 3: post-processing]"
fi


# --- Step 4: Downstream Statistical Tests and Baselines ---
echo ""
echo "========================================================================"
echo "STAGE 4: COMPUTING STATISTICAL TESTS & COMPARATOR BASELINES"
echo "========================================================================"

# Paired participant-level Wilcoxon signed-rank & paired t-tests
if [ -f "calculate_statistics.py" ]; then
    echo ">>> Running calculate_statistics.py..."
    "$PYTHON_EXE" calculate_statistics.py || echo "Warning: calculate_statistics.py encountered partial data."
fi

# Empirical comparator baselines (Uniform ensemble & Persistence)
if [ -f "calculate_baselines.py" ]; then
    echo ">>> Running calculate_baselines.py..."
    "$PYTHON_EXE" calculate_baselines.py || echo "Warning: calculate_baselines.py encountered partial data."
fi

# Volatility summary
if [ -f "plot_and_format_volatility.py" ] && [ -f "volatility_results.csv" ]; then
    echo ">>> Formatting volatility results..."
    "$PYTHON_EXE" plot_and_format_volatility.py || true
fi


# --- Step 5: Paper Figures Generation ---
if [ "$SKIP_FIGURES" = false ]; then
    echo ""
    echo "========================================================================"
    echo "STAGE 5: GENERATING PAPER FIGURES"
    echo "========================================================================"

    if [ -f "generate_figures.py" ]; then
        echo ">>> Running generate_figures.py..."
        "$PYTHON_EXE" generate_figures.py || echo "Warning: generate_figures.py completed with warnings for unrun settings."
    fi

    if [ -f "plot_subgroups.py" ]; then
        echo ">>> Running plot_subgroups.py..."
        "$PYTHON_EXE" plot_subgroups.py || true
    fi

    # Weight plot if a valid weight file is present
    if [ -f "plot_weights.py" ]; then
        SAMPLE_WEIGHTS=$(find Outputs -name "*expweights_eta10.csv" 2>/dev/null | head -n 1)
        if [ -n "$SAMPLE_WEIGHTS" ]; then
            echo ">>> Plotting expert weights using sample: $SAMPLE_WEIGHTS..."
            "$PYTHON_EXE" plot_weights.py --file "$SAMPLE_WEIGHTS" --output overleaf/images/methods/expert_weights_plot.pdf || true
        fi
    fi

    # Copy and organize Clarke Error Grid & regret curve PDFs into overleaf/images/
    if [ -f "copy_overleaf_images.sh" ]; then
        echo ">>> Copying figures into overleaf/images/..."
        bash copy_overleaf_images.sh
    fi

    # Ensure any tables/images written by legacy scripts to ../overleaf are synced into ./overleaf
    if [ -d "../overleaf/tables" ]; then
        rsync -a ../overleaf/tables/ overleaf/tables/ 2>/dev/null || true
    fi
    if [ -d "../overleaf/images" ]; then
        rsync -a ../overleaf/images/ overleaf/images/ 2>/dev/null || true
    fi

    echo "✓ Figures generated and copied to overleaf/images/."
else
    echo "[Skipping Stage 5: figure generation]"
fi


# --- Step 6: LaTeX Macros Generation ---
echo ""
echo "========================================================================"
echo "STAGE 6: GENERATING LATEX RESULTS MACROS"
echo "========================================================================"
if [ -f "generate_results_macros.py" ]; then
    "$PYTHON_EXE" generate_results_macros.py
    echo "✓ Generated macros in overleaf/sections/generated_macros.tex."
fi


# --- Step 7: Update Overleaf Repository ---
echo ""
echo "========================================================================"
echo "STAGE 7: UPDATING OVERLEAF REPOSITORY"
echo "========================================================================"

if [ -d "overleaf/.git" ]; then
    cd overleaf

    echo "Checking Overleaf git status:"
    git status --short

    # Check if there are any changes in overleaf
    if [ -n "$(git status --porcelain)" ]; then
        echo ""
        echo "Staging updated tables, figures, and macros in Overleaf..."
        git add tables/ images/ sections/ macros/ 2>/dev/null || git add -A

        COMMIT_MSG="Update paper figures, tables, and macros from VM simulation results [$(date '+%Y-%m-%d %H:%M')]"
        git commit -m "$COMMIT_MSG"
        echo "✓ Committed changes locally to Overleaf repo:"
        echo "  '$COMMIT_MSG'"

        if [ "$AUTO_PUSH" = true ]; then
            echo ""
            echo "Pushing changes to remote Overleaf repository..."
            git push origin main
            echo "✓ Overleaf repository pushed successfully!"
        else
            echo ""
            echo "------------------------------------------------------------------------"
            echo "Changes are committed locally in overleaf/."
            echo "To push to Overleaf now, run:"
            echo "    cd overleaf && git push origin main"
            echo "(Or run this script with --push next time to push automatically)"
            echo "------------------------------------------------------------------------"
        fi
    else
        echo "No changes detected in overleaf/ repository (already up to date)."
    fi

    cd "$SCRIPT_DIR"
else
    echo "Note: overleaf/.git not found; skipping git commit for overleaf directory."
fi

echo ""
echo "========================================================================"
echo "✓ PIPELINE COMPLETE!"
echo "========================================================================"
echo "Summary of updated artifacts:"
echo "  • Results / Outputs:  ./Outputs/ and ./rmse.csv, ./maxae.csv"
echo "  • LaTeX Tables:       ./overleaf/tables/"
echo "  • Paper Figures:      ./overleaf/images/ (clarke/, regret/, subgroups/)"
echo "  • LaTeX Macros:       ./overleaf/sections/generated_macros.tex"
echo "========================================================================"
