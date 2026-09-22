#!/usr/bin/env bash
set -euo pipefail
RUN="${1:?usage: $0 EXISTING_RUN MULTIVIEW_TRAIN_DIR WORLD_CHECKPOINT OUTPUT_DIR}"
TRAIN="${2:?missing multiview train directory}"
CKPT="${3:?missing world checkpoint}"
OUT="${4:?missing output directory}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$RUN/envs/node-0/bin/python"
TOKENIZER="${RFT_TOKENIZER_CHECKPOINT:-$RUN/tokenizer-4cam-4000/pretrained-004000}"
[[ -x "$PY" && -f "$TRAIN/manifest.json" && -f "$TRAIN/stats.json" && -f "$CKPT" && -d "$TOKENIZER" ]] || {
  echo 'Missing existing environment, multiview export, checkpoint, or tokenizer' >&2; exit 2;
}
[[ ! -e "$OUT" ]] || { echo "Output already exists: $OUT" >&2; exit 2; }
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/configs/navsim/ppu_compat:$REPO:$REPO/train/verl:$REPO/train/verl/vla-adapter/openvla-oft:$REPO/vendor/navsim:${PYTHONPATH:-}"
export VLA_RFT_VGG16_PATH="${VLA_RFT_VGG16_PATH:-${RFT_CACHE_ROOT:-$(dirname "$RUN")/../cache-pytest-fix-mirror}/weights/vgg16-397923af.pth}"
export VLA_RFT_LPIPS_PATH="${VLA_RFT_LPIPS_PATH:-${RFT_CACHE_ROOT:-$(dirname "$RUN")/../cache-pytest-fix-mirror}/weights/vgg.pth}"
exec "$PY" -m navsim_rft.check_multiview_world \
  --checkpoint "$CKPT" --tokenizer "$TOKENIZER" \
  --manifest "$TRAIN/manifest.json" --stats "$TRAIN/stats.json" \
  --output "$OUT" --limit "${RFT_QUALITY_LIMIT:-4}"
