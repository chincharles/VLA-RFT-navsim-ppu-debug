#!/usr/bin/env bash
# Run once per machine, not once per GPU. Never masks the platform's assigned GPUs.
set -euo pipefail
RFT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$RFT_REPO/configs/navsim/aliyun_paths.sh"
cd "$RFT_REPO"
exec "${RFT_PYTHON:-python3}" scripts/navsim/pipeline_16gpu.py "$@"
