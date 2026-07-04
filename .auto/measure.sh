#!/bin/bash
# Autoresearch benchmark: 2D-video rollout stability for rectified-flow forecasters.
# Emits METRIC name=value lines. Primary: rollout_mmd (lower=better).
# Technique + hyperparams are read from .auto/run.env (written each iteration).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # required by torch deterministic algorithms
set -a
[ -f .auto/run.env ] && source .auto/run.env
set +a
python video_rollout_experiment.py 2>&1
