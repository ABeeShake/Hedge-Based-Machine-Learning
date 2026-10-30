#!/bin/bash
# setup_vm_env.sh — One-shot environment setup and GPU verification on the VM.
#
# Creates a virtual environment, installs dependencies with CUDA acceleration,
# and verifies PyTorch/XGBoost GPU access.
#
# Run this on the VM:
#   bash setup_vm_env.sh

set -e

echo "========================================================================"
echo "SETTING UP HBML GPU ENVIRONMENT ON VM"
echo "========================================================================"

# Choose python3 executable
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.10+."
    exit 1
fi

echo "Python version: $($PYTHON_CMD --version)"

# Create virtual environment if it doesn't already exist
VENV_DIR="hedge-env"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in ./$VENV_DIR..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support if an NVIDIA GPU is detected
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected via nvidia-smi:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    echo "Installing PyTorch with CUDA..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121 || pip install torch
else
    echo "No nvidia-smi found. Installing standard PyTorch..."
    pip install torch
fi

# Install project dependencies
echo "Installing project dependencies from requirements.txt..."
pip install -r requirements.txt

# Run environment verification test
echo "========================================================================"
echo "VERIFYING ENVIRONMENT & ACCELERATION"
echo "========================================================================"
python -c "
import torch
print('PyTorch version:   ', torch.__version__)
print('CUDA Available:    ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Device Count: ', torch.cuda.device_count())
    print('Device Name:       ', torch.cuda.get_device_name(0))
else:
    print('MPS Available:     ', torch.backends.mps.is_available())

import statsforecast
print('StatsForecast:     ', statsforecast.__version__)
import xgboost
print('XGBoost:           ', xgboost.__version__)
import neuralforecast
print('NeuralForecast:    ', neuralforecast.__version__)

from fast_engine import FastRollingSimulation, FastOnlineHedge
print('Fast Engine:        Loaded successfully!')
"

echo "========================================================================"
echo "✓ VM Environment ready!"
echo "To run the complete paper simulation across all datasets & horizons:"
echo "    source hedge-env/bin/activate"
echo "    python run_paper_fast.py"
echo "========================================================================"
