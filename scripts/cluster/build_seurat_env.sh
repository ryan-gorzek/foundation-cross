#!/usr/bin/env bash
# seurat_env_setup.sh
# Locale + runtime settings for the conda env "seurat_env". Safe to run multiple times.

set -euo pipefail

# Require active conda env
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "CONDA_PREFIX not set; activate your conda environment first." >&2
  exit 1
fi

# Optionally enforce env name
env_name="$(basename "$CONDA_PREFIX")"
if [ "$env_name" != "seurat_env" ]; then
  echo "Active conda env is '$env_name' (expected 'seurat_env'). Continuing anyway." >&2
fi

# Locale (host provides en_US.utf8)
unset LC_ALL LC_CTYPE LC_MESSAGES LC_COLLATE LC_TIME LC_MONETARY LC_PAPER LC_MEASUREMENT || true
export LANG=en_US.utf8
export LC_ALL=en_US.utf8

# Ensure conda’s runtimes are first (fixes GLIBCXX/CXXABI issues)
triplet_lib="$CONDA_PREFIX/x86_64-conda-linux-gnu/lib"
if [ -d "$triplet_lib" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$triplet_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Tame threaded BLAS/OpenMP on shared nodes
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Optional quick checks when invoked with --check
if [ "${1:-}" = "--check" ]; then
  echo "[check] LD_LIBRARY_PATH:"
  echo "  $LD_LIBRARY_PATH"
  echo "[check] libstdc++/libgcc in env:"
  find "$CONDA_PREFIX" -maxdepth 3 -name 'libstdc++.so*' -o -name 'libgcc_s.so*' | sed 's/^/  /'
  echo "[check] R links (expect ICU 75 and conda libstdc++):"
  ldd "$CONDA_PREFIX/lib/R/bin/exec/R" | egrep 'libstdc\+\+|libgcc_s|icu' || true
  echo "[check] R sessionInfo():"
  R --vanilla -q -e 'sessionInfo()' || true
fi
