sq#!/bin/bash
#SBATCH --job-name=hbml_run_vm
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/run_on_vm_%j.out
#SBATCH --error=slurm_logs/run_on_vm_%j.err

set -euo pipefail
mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export FAST_ENGINE_DEVICE="cpu"

echo "=============================================================================="
echo "SLURM Job ID:       $SLURM_JOB_ID"
echo "Allocated CPUs:     $SLURM_CPUS_PER_TASK"
echo "Start time:         $(date)"
echo "=============================================================================="

bash run_on_vm.sh

echo "=============================================================================="
echo "Completed run_on_vm.sh at $(date)"
echo "=============================================================================="

