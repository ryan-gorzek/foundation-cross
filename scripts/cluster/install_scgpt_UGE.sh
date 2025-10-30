#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Install scGPT on UCLA UGE-style clusters (tested with GTX1080Ti + CUDA 12.1 wheels)
# Creates a conda env `scgpt` with pinned dependencies that worked reliably.
# -----------------------------------------------------------------------------
# Notes:
# - If you're running on a login node, request an interactive GPU session first:
#     qrsh -l gpu,GTX1080Ti,cuda=1,h_data=24G,h_rt=01:00:00
# - The cluster must provide modules: anaconda3, gcc/10.2.0, cmake/3.30.0.
# - PyTorch wheels below are for CUDA 12.1. Adjust the index URL if using a different CUDA.
# -----------------------------------------------------------------------------

set -euo pipefail

# Load environment modules (cluster-specific)
module load anaconda3
module load gcc/10.2.0
module load cmake/3.30.0

# Ensure `conda` is available in this non-interactive shell
# (On many clusters, conda is set up by the anaconda3 module, but this is safe.)
if command -v conda >/dev/null 2>&1; then
  : # ok
else
  echo "conda not found in PATH after 'module load anaconda3'." >&2
  exit 1
fi

# Activate conda in this shell
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create and activate environment
if ! conda env list | awk '{print $1}' | grep -qx "scgpt"; then
  conda create -y -n scgpt python=3.11
fi
conda activate scgpt

# Core pins that avoid ABI issues
pip install --no-cache-dir "numpy==1.26.4" "h5py==3.10.0"

# PyTorch CUDA 12.1 wheels
pip install --no-cache-dir \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 torchtext==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121

# Arrow ecosystem (binary-only to avoid C++ toolchain mismatches)
pip install --only-binary=:all: "pyarrow>=17,<19"

# TensorStore via conda-forge (pulls in newer low-level libs cleanly)
conda install -y -c conda-forge tensorstore

# Jupyter/IPython stack pins compatible with this env
pip install "ipython>=8.12,<9" "traitlets>=5.9" "prompt-toolkit>=3.0"

# scGPT (build isolation off to respect already-pinned deps)
pip install --no-build-isolation scgpt

# TensorStore may have dragged in NumPy 2.x; enforce 1.26.x again
# Safe even if already correct.
conda remove -y numpy || true
conda install -y -c conda-forge "numpy==1.26.4" "libstdcxx-ng>=12" "libgcc-ng>=12"

# scvi-tools stack compatibility for scGPT usage patterns
pip install --no-cache-dir "jax[cpu]==0.4.33" "jaxlib==0.4.33" "ml_dtypes>=0.3.2"
pip install --no-cache-dir "anndata==0.10.8"

# Optional
pip install wandb

echo "scGPT environment ready. Activate with: conda activate scgpt"
