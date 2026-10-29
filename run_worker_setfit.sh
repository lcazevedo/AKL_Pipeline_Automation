#!/bin/bash
# run_worker.sh
# -------------
# Wrapper script that activates the correct Python environment before
# running any AKL worker. This ensures the pipeline always uses the
# right dependencies regardless of the system's default Python.
#
# Usage (called by OpenWork skills):
#   ./run_worker.sh workers/step1_worker.py wos --input data/inputs/file.txt
#   ./run_worker.sh workers/step2_worker.py
#
# Setup for new users:
#   1. Edit CONDA_ENV below to match your environment name.
#   2. Edit CONDA_BASE if miniconda/anaconda is in a different path.
#   3. Run: chmod +x run_worker.sh

# ── Configuration — edit these two lines for your setup ──────────────────────
CONDA_ENV="akl_setfit"
CONDA_BASE="${HOME}/miniconda3"   # change to ~/anaconda3 if needed
# ─────────────────────────────────────────────────────────────────────────────

CONDA_INIT="${CONDA_BASE}/etc/profile.d/conda.sh"

if [ ! -f "$CONDA_INIT" ]; then
    CONDA_INIT="${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if [ ! -f "$CONDA_INIT" ]; then
    echo "ERROR: conda not found at ${CONDA_BASE} or ~/anaconda3"
    echo "Please edit CONDA_BASE in run_worker.sh."
    exit 1
fi

source "$CONDA_INIT"
conda activate "$CONDA_ENV" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "ERROR: Could not activate conda environment '${CONDA_ENV}'"
    echo "Please edit CONDA_ENV in run_worker.sh."
    echo "Available environments:"
    conda env list
    exit 1
fi

python "$@"