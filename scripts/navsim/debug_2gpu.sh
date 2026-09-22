#!/usr/bin/env bash
# First-version models, fixed single-node two-CUDA-GPU smoke budget.
# Run once from ClearML; do not wrap this bootstrap in torchrun.
set -euo pipefail
RFT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$RFT_REPO/configs/navsim/aliyun_paths.sh"
cd "$RFT_REPO"
exec "${RFT_PYTHON:-python3}" scripts/navsim/pipeline_16gpu.py "$@" \
  --nnodes 1 --gpus-per-node 2 --node-rank 0 --profile smoke
