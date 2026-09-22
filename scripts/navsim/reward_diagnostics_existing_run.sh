#!/usr/bin/env bash
set -euo pipefail

# Re-run only reward analysis for an already completed smoke run.
# Usage: bash scripts/navsim/reward_diagnostics_existing_run.sh /path/to/run

RUN="${1:?usage: $0 /absolute/path/to/existing-run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$RUN/envs/node-0/bin/python"
ATTEMPT="$RUN/launches/initial"

[[ -x "$PY" ]] || { echo "Missing run Python: $PY" >&2; exit 2; }
[[ -f "$RUN/data/train/manifest.json" ]] || { echo "Missing train manifest" >&2; exit 2; }
[[ -f "$RUN/data/train/stats.json" ]] || { echo "Missing action stats" >&2; exit 2; }
[[ -f "$RUN/metric-cache/metadata.json" || -d "$RUN/metric-cache" ]] || { echo "Missing metric cache" >&2; exit 2; }
[[ -f "$RUN/sft/step-000002.pt" ]] || { echo "Missing SFT checkpoint" >&2; exit 2; }
[[ -f "$RUN/wm/step-000002.pt" ]] || { echo "Missing world-model checkpoint" >&2; exit 2; }

export RFT_WORK="$RUN"
export RFT_CACHE_ROOT="${RFT_CACHE_ROOT:-$(dirname "$RUN")/cache-pytest-fix-mirror}"
export HF_HOME="${HF_HOME:-$RFT_CACHE_ROOT/huggingface}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/configs/navsim/ppu_compat:$REPO:$REPO/train/verl:$REPO/train/verl/vla-adapter/openvla-oft:$REPO/vendor/navsim:${PYTHONPATH:-}"
export NAVSIM_ROOT="${NAVSIM_ROOT:-$REPO/vendor/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$RUN/navsim-exp}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"

OUT="$ATTEMPT/reward-analysis-diagnostics"
if [[ -e "$OUT" ]]; then
  OUT="$OUT-$(date -u +%Y%m%dT%H%M%SZ)"
fi

exec "$PY" -m navsim_rft.analyze_rewards \
  --policy "$RUN/sft/step-000002.pt" \
  --world "$RUN/wm/step-000002.pt" \
  --manifest "$RUN/data/train/manifest.json" \
  --stats "$RUN/data/train/stats.json" \
  --metric-cache "$RUN/metric-cache" \
  --output "$OUT" \
  --limit "${RFT_REWARD_LIMIT:-8}" \
  --candidates "${RFT_REWARD_CANDIDATES:-4}"

