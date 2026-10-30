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

# Ensure remote target directory exists
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

# Rsync codebase and datasets, excluding virtualenvs, caches, git, and multi-GB logs
rsync -avz --progress \
    --exclude 'hedge-env/' \
    --exclude 'venv/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '.git/' \
    --exclude '*.log' \
    --exclude 'output-data_weinstock_stdout.log' \
    --exclude 'lightning_logs/' \
    --exclude 'scratch/' \
    ./ "$REMOTE_HOST:$REMOTE_DIR/"

echo "========================================================================"
echo "✓ Sync complete! Codebase is ready on $REMOTE_HOST:$REMOTE_DIR"
echo "Next step: SSH to VM and run:"
echo "    cd $REMOTE_DIR && bash setup_vm_env.sh"
echo "========================================================================"
