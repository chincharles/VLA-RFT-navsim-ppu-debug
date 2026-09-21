#!/usr/bin/env bash
# Bounded 1-GPU real-data validation. No distributed job is launched implicitly.
set -euo pipefail
source scripts/navsim/env.sh
: "${CONFIG_DIR:?Generate configs first}"
: "${RUN_DIR:?Choose new independent output root}"
: "${VAL_MANIFEST:?Export held-out navtrain validation first}"
: "${ACTION_STATS:?Training-only stats path}"
mkdir -p "$RUN_DIR"
python -m navsim_rft.preflight --config "$CONFIG_DIR/rl_img.json" --output "$RUN_DIR/preflight.json"
python -m navsim_rft.train --config "$CONFIG_DIR/wm.json" --output "$RUN_DIR/wm" --steps 2
python -m navsim_rft.train --config "$CONFIG_DIR/wm.json" --output "$RUN_DIR/wm" --resume "$RUN_DIR/wm/step-000002.pt" --steps 1
python -m navsim_rft.quality --checkpoint "$RUN_DIR/wm/step-000003.pt" --manifest "$VAL_MANIFEST" --stats "$ACTION_STATS" --output "$RUN_DIR/wm_quality" --limit 2
python -m navsim_rft.train --config "$CONFIG_DIR/sft.json" --output "$RUN_DIR/sft" --steps 2
python -m navsim_rft.train --config "$CONFIG_DIR/sft.json" --output "$RUN_DIR/sft" --resume "$RUN_DIR/sft/step-000002.pt" --steps 1
# Validate the baseline via the official evaluator before RL. These paths point to trainval logs/blobs.
: "${TRAIN_LOGS:?Set trainval log path}"
: "${TRAIN_SENSORS:?Set trainval sensor path}"
: "${TRAIN_METRIC_CACHE:?Set training and validation scene metric cache}"
python -m navsim_rft.evaluate --navsim-root "$NAVSIM_ROOT" --checkpoint "$RUN_DIR/sft/step-000003.pt" --logs "$TRAIN_LOGS" --sensors "$TRAIN_SENSORS" --metric-cache "$TRAIN_METRIC_CACHE" --output "$RUN_DIR/eval_sft" --split navtrain --manifest "$VAL_MANIFEST"
python -m navsim_rft.train --config "$CONFIG_DIR/rl_img.json" --output "$RUN_DIR/rl_img" --steps 1 --init-policy "$RUN_DIR/sft/step-000003.pt" --wm-checkpoint "$RUN_DIR/wm/step-000003.pt"
python -m navsim_rft.train --config "$CONFIG_DIR/rl_drive.json" --output "$RUN_DIR/rl_drive" --steps 1 --init-policy "$RUN_DIR/sft/step-000003.pt"
for mode in rl_img rl_drive; do
 python -m navsim_rft.evaluate --navsim-root "$NAVSIM_ROOT" --checkpoint "$RUN_DIR/$mode/step-000001.pt" --logs "$TRAIN_LOGS" --sensors "$TRAIN_SENSORS" --metric-cache "$TRAIN_METRIC_CACHE" --output "$RUN_DIR/eval_$mode" --split navtrain --manifest "$VAL_MANIFEST"
done
