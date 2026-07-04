#!/bin/bash
# Autoresearch benchmark: 3D volumetric rollout stability for rectified-flow forecasters.
# Emits METRIC name=value lines. Primary: rollout_mmd (lower=better).
set -euo pipefail
cd "$(dirname "$0")/.."

# Technique + hyperparams come from env (set by the loop or measure.sh itself).
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python video_rollout_experiment.py 2>&1
