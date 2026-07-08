#!/bin/bash
# Autoresearch benchmark: BLOCK-BRIDGE flow matching (3-frame -> 3-frame).
# Emits METRIC name=value lines. Primary: rollout_ed (lower=better).
# Hyperparams read from .auto/run.env (written each iteration).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # required by torch deterministic algorithms
set -a
[ -f .auto/run.env ] && source .auto/run.env
set +a
python block_bridge_experiment.py 2>&1
