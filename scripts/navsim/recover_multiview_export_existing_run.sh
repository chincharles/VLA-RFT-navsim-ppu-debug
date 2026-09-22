#!/usr/bin/env bash
set -euo pipefail
RUN="${1:?usage: $0 EXISTING_RUN OUTPUT_DIR [DATA_ROOT] [NAVSIM_ROOT]}"
OUT="${2:?missing existing export directory}"
DATA="${3:-/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAVSIM="${4:-$REPO/vendor/navsim}"
PY="$RUN/envs/node-0/bin/python"
[[ -x "$PY" && -d "$OUT" ]] || { echo 'Missing existing run Python or export directory' >&2; exit 2; }
[[ ! -e "$OUT/manifest.json" && ! -e "$OUT/stats.json" ]] || { echo 'Manifest/statistics already exist; refusing overwrite' >&2; exit 2; }
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$DATA}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA/map}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/configs/navsim/ppu_compat:$REPO:$REPO/train/verl:$REPO/train/verl/vla-adapter/openvla-oft:$REPO/vendor/navsim:${PYTHONPATH:-}"
"$PY" -m navsim_rft.recover_multiview_manifest \
  --root "$NAVSIM" --logs "$DATA/navsim_logs/trainval" \
  --sensors "$DATA/sensor_blobs/trainval" --out "$OUT" --limit "${RFT_MULTIVIEW_LIMIT:-800}"
"$PY" "$REPO/scripts/navsim/check_multiview_data.py" --manifest "$OUT/manifest.json" --limit 8
