#!/usr/bin/env bash
# Resume the existing 800-scene, two-PPU experiment. No installation/download.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${1:-/mnt/cpfs-wlc-rdma-300t/navsim/vla-rft/runs/smoke2-20260922T042137Z-df2bc7}"
export RFT_RESUME=1
export RFT_CACHE_ROOT="${RFT_CACHE_ROOT:-/mnt/cpfs-wlc-rdma-300t/navsim/vla-rft/cache-pytest-fix-mirror}"
export RFT_MULTIVIEW_TRAIN="${RFT_MULTIVIEW_TRAIN:-/mnt/cpfs-wlc-rdma-300t/navsim/vla-rft/runs/multiview-800/data/train4cam}"
export RFT_TOKENIZER_INIT="${RFT_TOKENIZER_INIT:-$RUN/tokenizer-2000/pretrained-002000}"
export RFT_TOKENIZER_OUTPUT="${RFT_TOKENIZER_OUTPUT:-$RUN/tokenizer-4cam-4000-800-b16}"
export RFT_WM_OUTPUT="${RFT_WM_OUTPUT:-$RUN/wm-4cam-500-800-b16}"
export RFT_TOKENIZER_STEPS=4000 RFT_WM_STEPS=500 RFT_BATCH_SIZE_PER_GPU=16 RFT_GPUS=2
exec bash "$REPO/scripts/navsim/train_multiview_existing_run.sh" "$RUN"
