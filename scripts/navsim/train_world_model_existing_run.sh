#!/usr/bin/env bash
set -euo pipefail

RUN="${1:?usage: $0 /absolute/path/to/existing-run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$RUN/envs/node-0/bin/python"
[[ -x "$PY" ]] || { echo "Missing run Python: $PY" >&2; exit 2; }
[[ -f "$RUN/config/wm.json" ]] || { echo "Missing WM config" >&2; exit 2; }
[[ -f "$RUN/data/train/manifest.json" ]] || { echo "Missing train manifest" >&2; exit 2; }
[[ -f "$RUN/data/train/stats.json" ]] || { echo "Missing action stats" >&2; exit 2; }

export RFT_WORK="$RUN"
export RFT_CACHE_ROOT="${RFT_CACHE_ROOT:-$(dirname "$RUN")/../cache-pytest-fix-mirror}"
export HF_HOME="${HF_HOME:-$RFT_CACHE_ROOT/huggingface}"
export PYTHONNOUSERSITE=1
export NAVSIM_ROOT="${NAVSIM_ROOT:-$REPO/vendor/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$RUN/navsim-exp}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="$REPO/configs/navsim/ppu_compat:$REPO:$REPO/train/verl:$REPO/train/verl/vla-adapter/openvla-oft:$REPO/vendor/navsim:${PYTHONPATH:-}"

OUT="${RFT_WM_OUTPUT:-$RUN/wm-200}"
STEPS="${RFT_WM_STEPS:-200}"
GPUS="${RFT_GPUS:-2}"
mkdir -p "$OUT"

exec torchrun --nproc_per_node="$GPUS" --master_port="${RFT_MASTER_PORT:-29521}" \
  -m navsim_rft.train --config "$RUN/config/wm.json" --output "$OUT" \
  --steps "$STEPS" --save-every "${RFT_SAVE_EVERY:-50}"

