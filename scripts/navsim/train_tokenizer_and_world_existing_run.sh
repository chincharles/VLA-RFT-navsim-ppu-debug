#!/usr/bin/env bash
set -euo pipefail

RUN="${1:?usage: $0 /absolute/path/to/existing-run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$RUN/envs/node-0/bin/python"
[[ -x "$PY" ]] || { echo "Missing run Python: $PY" >&2; exit 2; }
[[ -f "$RUN/data/train/manifest.json" && -f "$RUN/data/train/stats.json" ]] || { echo "Missing train data" >&2; exit 2; }
[[ -f "$RUN/config/wm.json" ]] || { echo "Missing WM config" >&2; exit 2; }

export RFT_WORK="$RUN"
export RFT_CACHE_ROOT="${RFT_CACHE_ROOT:-$(dirname "$RUN")/../cache-pytest-fix-mirror}"
export HF_HOME="${HF_HOME:-$RFT_CACHE_ROOT/huggingface}"
export VLA_RFT_VGG16_PATH="${VLA_RFT_VGG16_PATH:-$RFT_CACHE_ROOT/weights/vgg16-397923af.pth}"
export VLA_RFT_LPIPS_PATH="${VLA_RFT_LPIPS_PATH:-$RFT_CACHE_ROOT/weights/vgg.pth}"
export PYTHONNOUSERSITE=1
export NAVSIM_ROOT="${NAVSIM_ROOT:-$REPO/vendor/navsim}"
export PYTHONPATH="$REPO/configs/navsim/ppu_compat:$REPO:$REPO/train/verl:$REPO/train/verl/vla-adapter/openvla-oft:$REPO/vendor/navsim:${PYTHONPATH:-}"

GPUS="${RFT_GPUS:-2}"
TOK_STEPS="${RFT_TOKENIZER_STEPS:-2000}"
WM_STEPS="${RFT_WM_STEPS:-500}"
TOK_OUT="${RFT_TOKENIZER_OUTPUT:-$RUN/tokenizer-$TOK_STEPS}"
WM_OUT="${RFT_WM_OUTPUT:-$RUN/wm-$WM_STEPS-from-tokenizer}"
PORT="${RFT_MASTER_PORT:-29523}"

if [[ -e "$TOK_OUT" ]]; then echo "Tokenizer output exists: $TOK_OUT" >&2; exit 2; fi
if [[ -e "$WM_OUT" ]]; then echo "World-model output exists: $WM_OUT" >&2; exit 2; fi

echo "Training tokenizer for $TOK_STEPS steps: $TOK_OUT"
"$PY" -m torch.distributed.run --nproc_per_node="$GPUS" --master_port="$PORT" \
  -m navsim_rft.train_tokenizer \
  --manifest "$RUN/data/train/manifest.json" \
  --stats "$RUN/data/train/stats.json" \
  --output "$TOK_OUT" --steps "$TOK_STEPS" --save-every 500

TOK_PRETRAINED="$TOK_OUT/pretrained-$(printf '%06d' "$TOK_STEPS")"
[[ -d "$TOK_PRETRAINED" ]] || { echo "Missing exported tokenizer: $TOK_PRETRAINED" >&2; exit 2; }

WM_CONFIG="$RUN/wm-config-tokenizer-$TOK_STEPS.json"
"$PY" - "$RUN/config/wm.json" "$WM_CONFIG" "$TOK_PRETRAINED" <<'PY'
import json, sys
src, dst, tokenizer = sys.argv[1:]
cfg = json.load(open(src))
cfg['tokenizer'] = tokenizer
json.dump(cfg, open(dst, 'w'), indent=2)
PY

echo "Training world model for $WM_STEPS steps: $WM_OUT"
"$PY" -m torch.distributed.run --nproc_per_node="$GPUS" --master_port="$PORT" \
  -m navsim_rft.train --config "$WM_CONFIG" --output "$WM_OUT" \
  --steps "$WM_STEPS" --save-every 100

echo "Tokenizer: $TOK_PRETRAINED"
echo "World model: $WM_OUT/step-$(printf '%06d' "$WM_STEPS").pt"
