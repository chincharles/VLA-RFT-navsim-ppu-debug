#!/usr/bin/env bash
set -euo pipefail
RUN="${1:?usage: $0 EXISTING_RUN OUTPUT_DIR [LIMIT] [DATA_ROOT] [NAVSIM_ROOT]}"
OUT="${2:?missing output directory}"
LIMIT="${3:-800}"
DATA="${4:-/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAVSIM="${5:-$REPO/vendor/navsim}"
PY="$RUN/envs/node-0/bin/python"
[[ -x "$PY" ]] || { echo "Missing existing run Python: $PY" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Output already exists: $OUT" >&2; exit 2; }
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/configs/navsim/ppu_compat:$REPO:$REPO/train/verl:$REPO/train/verl/vla-adapter/openvla-oft:$REPO/vendor/navsim:${PYTHONPATH:-}"
"$PY" -m navsim_rft.export_multiview \
  --root "$NAVSIM" --logs "$DATA/navsim_logs/trainval" \
  --sensors "$DATA/sensor_blobs/trainval" --split navtrain \
  --role train --limit "$LIMIT" --out "$OUT"
"$PY" "$REPO/scripts/navsim/check_multiview_data.py" --manifest "$OUT/manifest.json" --limit 8
