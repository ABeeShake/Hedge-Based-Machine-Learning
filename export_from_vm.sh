#!/bin/bash
# export_from_vm.sh — Streamlined export of results, figures, and tables from the VM.
#
# Pulls all generated figures, LaTeX tables, macros, and summary metrics from
# the remote GPU VM directly into your local repository.
#
# Usage:
#   bash export_from_vm.sh [USER@HOST] [REMOTE_DIR] [--include-outputs]
#
# Defaults:
#   Host:       abhishekdevarajan@jsb3.stat.ucla.edu (or $VM_HOST)
#   Remote Dir: ~/HedgeProject (or $VM_DIR)
#
# Example:
#   bash export_from_vm.sh
#   bash export_from_vm.sh abhishekdevarajan@jsb3.stat.ucla.edu ~/HedgeProject --include-outputs

set -e

REMOTE_HOST="${1:-${VM_HOST:-abhishekdevarajan@jsb3.stat.ucla.edu}}"
REMOTE_DIR="${2:-${VM_DIR:-~/HedgeProject}}"
INCLUDE_OUTPUTS=false

for arg in "$@"; do
    case $arg in
        --include-outputs) INCLUDE_OUTPUTS=true ;;
    esac
done

echo "========================================================================"
echo "EXPORTING RESULTS FROM REMOTE VM"
echo "========================================================================"
echo "Remote Source:    $REMOTE_HOST:$REMOTE_DIR"
echo "Local Target:     $(pwd)"
echo "Include Outputs:  $INCLUDE_OUTPUTS"
echo "========================================================================"

# Ensure local directories exist
mkdir -p overleaf/images overleaf/tables overleaf/sections images

# --- Setup SSH Multiplexing (Single Password Entry) ---
CONTROL_SOCKET="/tmp/ssh-mux-%r@%h:%p"
SSH_MUX_OPTS="-o ControlMaster=auto -o ControlPath=${CONTROL_SOCKET} -o ControlPersist=10m"

cleanup_ssh() {
    if ssh -O check -o ControlPath="${CONTROL_SOCKET}" "$REMOTE_HOST" 2>/dev/null; then
        echo ""
        echo ">>> Closing SSH master connection..."
        ssh -O exit -o ControlPath="${CONTROL_SOCKET}" "$REMOTE_HOST" 2>/dev/null || true
    fi
}
trap cleanup_ssh EXIT INT TERM

echo ""
echo ">>> Establishing authenticated SSH session with $REMOTE_HOST..."
echo ">>> (You will only be prompted for your SSH password ONCE below)"
echo "------------------------------------------------------------------------"

# Authenticate once and establish the master socket
ssh ${SSH_MUX_OPTS} "$REMOTE_HOST" "test -d $REMOTE_DIR" || {
    echo "Error: Remote directory $REMOTE_DIR not found on $REMOTE_HOST."
    exit 1
}
echo "✓ SSH session authenticated and cached for multiplexing."
echo ""

echo ">>> [1/4] Syncing Overleaf figures and images..."
rsync -avz --progress -e "ssh ${SSH_MUX_OPTS}" "$REMOTE_HOST:$REMOTE_DIR/overleaf/images/" ./overleaf/images/

echo ">>> [2/4] Syncing Overleaf tables and LaTeX macros..."
rsync -avz --progress -e "ssh ${SSH_MUX_OPTS}" "$REMOTE_HOST:$REMOTE_DIR/overleaf/tables/" ./overleaf/tables/
rsync -avz --progress -e "ssh ${SSH_MUX_OPTS}" "$REMOTE_HOST:$REMOTE_DIR/overleaf/sections/generated_macros.tex" ./overleaf/sections/ || true
if ssh ${SSH_MUX_OPTS} "$REMOTE_HOST" "[ -d $REMOTE_DIR/overleaf/macros ]"; then
    mkdir -p overleaf/macros
    rsync -avz --progress -e "ssh ${SSH_MUX_OPTS}" "$REMOTE_HOST:$REMOTE_DIR/overleaf/macros/" ./overleaf/macros/
fi

echo ">>> [3/4] Syncing root figures and summary metrics CSVs..."
if ssh ${SSH_MUX_OPTS} "$REMOTE_HOST" "[ -d $REMOTE_DIR/images ]"; then
    rsync -avz --progress -e "ssh ${SSH_MUX_OPTS}" "$REMOTE_HOST:$REMOTE_DIR/images/" ./images/
fi

for f in rmse.csv maxae.csv volatility_results.csv output_timings.csv; do
    rsync -avz --progress -e "ssh ${SSH_MUX_OPTS}" "$REMOTE_HOST:$REMOTE_DIR/$f" ./ 2>/dev/null || true
done

if [ "$INCLUDE_OUTPUTS" = true ]; then
    echo ">>> [4/4] Syncing complete Outputs/ directory..."
    mkdir -p Outputs
    rsync -avz --progress \
        -e "ssh ${SSH_MUX_OPTS}" \
        --exclude 'node/' \
        --exclude '*.pt' \
        "$REMOTE_HOST:$REMOTE_DIR/Outputs/" ./Outputs/
else
    echo ">>> [4/4] Skipping raw Outputs/ (pass --include-outputs to also download raw per-step forecasts)."
fi

echo "========================================================================"
echo "✓ Export complete! All figures, tables, and macros updated locally."
echo "Your overleaf/ directory and figures are ready for inspection or git commit."
echo "========================================================================"
