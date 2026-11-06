#!/bin/bash
# sync_to_vm.sh — Streamlined upload of code and data to the remote GPU VM.
#
# Usage:
#   bash sync_to_vm.sh [USER@HOST] [REMOTE_DIR]
#
# Defaults:
#   Host:       abhishekdevarajan@jsb3.stat.ucla.edu (or $VM_HOST)
#   Remote Dir: ~/HedgeProject (or $VM_DIR)
#
# Example:
#   bash sync_to_vm.sh
#   bash sync_to_vm.sh myuser@myvm.cloud.google.com ~/work/HedgeProject

set -e

REMOTE_HOST="${1:-${VM_HOST:-abhishekdevarajan@jsb3.stat.ucla.edu}}"
REMOTE_DIR="${2:-${VM_DIR:-~/HedgeProject}}"

echo "========================================================================"
echo "SYNCING CODEBASE TO REMOTE VM"
echo "========================================================================"
echo "Target Host:      $REMOTE_HOST"
echo "Remote Directory: $REMOTE_DIR"
echo "========================================================================"

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

# Ensure remote target directory exists and establish the master socket
ssh ${SSH_MUX_OPTS} "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"
echo "✓ SSH session authenticated and cached for multiplexing."
echo ""

# Rsync codebase and datasets, excluding virtualenvs, caches, root git, Outputs, and logs
rsync -avz --progress \
    -e "ssh ${SSH_MUX_OPTS}" \
    --exclude 'hedge-env/' \
    --exclude 'venv/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '/.git/' \
    --exclude 'Outputs/' \
    --exclude 'Outputs_debug/' \
    --exclude '*.log' \
    --exclude 'output-data_weinstock_stdout.log' \
    --exclude 'lightning_logs/' \
    --exclude 'scratch/' \
    ./ "$REMOTE_HOST:$REMOTE_DIR/"

echo "========================================================================"
echo "✓ Sync complete! Codebase is ready on $REMOTE_HOST:$REMOTE_DIR"
echo ""
echo "Direct VM-to-Overleaf Workflow:"
echo "  To run simulations and push results directly to Overleaf:"
echo "    sbatch slurm/submit_pipeline.sh"
echo "  Or to update Overleaf directly from completed simulations on the VM:"
echo "    bash vm_update_overleaf.sh"
echo "========================================================================"
