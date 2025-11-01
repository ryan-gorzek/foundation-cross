#!/usr/bin/env bash
# Install environment for cross-species label transfer pipeline
# This is an updated version of install_scgpt_UGE.sh with additional dependencies

set -euo pipefail

echo "=========================================="
echo "Installing Cross-Species Transfer Environment"
echo "=========================================="

# Load environment modules (cluster-specific)
module load anaconda3
module load gcc/10.2.0
module load cmake/3.30.0

# Ensure conda is available
if command -v conda >/dev/null 2>&1; then
  : # ok
else
  echo "conda not found in PATH after 'module load anaconda3'." >&2
  exit 1
fi

# Activate conda in this shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create and activate environment
ENV_NAME="scgpt"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

echo ""
echo "Installing core dependencies..."
echo "----------------------------------------"

# Core pins
pip install --no-cache-dir "numpy==1.26.4" "h5py==3.10.0"

# PyTorch CUDA 12.1 wheels
pip install --no-cache-dir \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 torchtext==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121

# Arrow ecosystem
pip install --only-binary=:all: "pyarrow>=17,<19"

# TensorStore via conda-forge
conda install -y -c conda-forge tensorstore

# Jupyter/IPython stack
pip install "ipython>=8.12,<9" "traitlets>=5.9" "prompt-toolkit>=3.0"

# scGPT
pip install --no-build-isolation scgpt

# Fix NumPy version
conda remove -y numpy || true
conda install -y -c conda-forge "numpy==1.26.4" "libstdcxx-ng>=12" "libgcc-ng>=12"

# scvi-tools stack
pip install --no-cache-dir "jax[cpu]==0.4.33" "jaxlib==0.4.33" "ml_dtypes>=0.3.2"
pip install --no-cache-dir "anndata==0.10.8"

echo ""
echo "Installing additional pipeline dependencies..."
echo "----------------------------------------"

# Scanpy and visualization
pip install --no-cache-dir \
  "scanpy>=1.9.0" \
  "matplotlib>=3.4.0" \
  "seaborn>=0.11.0" \
  "pandas>=1.3.0" \
  "scipy>=1.7.0" \
  "scikit-learn>=1.0.0"

# Configuration and utilities
pip install --no-cache-dir \
  "pyyaml>=5.4" \
  "tqdm>=4.60.0"

# Optional: wandb for experiment tracking
pip install --no-cache-dir wandb

echo ""
echo "=========================================="
echo "Environment '$ENV_NAME' ready!"
echo ""
echo "To activate:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run an experiment:"
echo "  python scripts/run_experiment.py --config configs/experiments/mouse_to_opossum.yaml"
echo "=========================================="