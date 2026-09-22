#!/usr/bin/env bash
set -euo pipefail
RUN="${1:?usage: $0 /absolute/path/to/existing-run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$RUN/envs/node-0/bin/python"
TRAIN="$RUN/data/train4cam"
TOK_INIT="${RFT_TOKENIZER_INIT:-$RUN/tokenizer-2000/pretrained-002000}"
TOK_OUT="${RFT_TOKENIZER_OUTPUT:-$RUN/tokenizer-4cam-4000}"
WM_OUT="${RFT_WM_OUTPUT:-$RUN/wm-4cam-500}"
GPUS="${RFT_GPUS:-2}"
[[ -x "$PY" && -f "$TRAIN/manifest.json" && -f "$TRAIN/stats.json" ]] || { echo 'Missing run environment or train4cam export' >&2; exit 2; }
[[ -d "$TOK_INIT" ]] || { echo "Missing tokenizer init: $TOK_INIT" >&2; exit 2; }
[[ ! -e "$TOK_OUT" && ! -e "$WM_OUT" ]] || { echo 'Output already exists; choose new RFT_TOKENIZER_OUTPUT/RFT_WM_OUTPUT' >&2; exit 2; }
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/configs/navsim/ppu_compat:$REPO:$REPO/train/verl:$REPO/train/verl/vla-adapter/openvla-oft:$REPO/vendor/navsim:${PYTHONPATH:-}"
export VLA_RFT_VGG16_PATH="${VLA_RFT_VGG16_PATH:-${RFT_CACHE_ROOT:-$(dirname "$RUN")/../cache-pytest-fix-mirror}/weights/vgg16-397923af.pth}"
export VLA_RFT_LPIPS_PATH="${VLA_RFT_LPIPS_PATH:-${RFT_CACHE_ROOT:-$(dirname "$RUN")/../cache-pytest-fix-mirror}/weights/vgg.pth}"
PORT="${RFT_MASTER_PORT:-29531}"
"$PY" -m torch.distributed.run --nproc_per_node="$GPUS" --master_port="$PORT" -m navsim_rft.train_multiview_tokenizer \
  --manifest "$TRAIN/manifest.json" --stats "$TRAIN/stats.json" --init "$TOK_INIT" --output "$TOK_OUT" \
  --steps "${RFT_TOKENIZER_STEPS:-4000}" --save-every 500
TOK="$TOK_OUT/pretrained-$(printf '%06d' "${RFT_TOKENIZER_STEPS:-4000}")"
"$PY" -m torch.distributed.run --nproc_per_node="$GPUS" --master_port="$PORT" -m navsim_rft.train_multiview_world \
  --manifest "$TRAIN/manifest.json" --stats "$TRAIN/stats.json" --tokenizer "$TOK" --output "$WM_OUT" \
  --steps "${RFT_WM_STEPS:-500}" --save-every 100
echo "tokenizer=$TOK"
echo "world=$WM_OUT/step-$(printf '%06d' "${RFT_WM_STEPS:-500}").pt"
