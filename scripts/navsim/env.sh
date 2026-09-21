#!/usr/bin/env bash
# Source from the migration repository root. User supplies server paths.
set -euo pipefail
export PYTHONPATH="$PWD:$PWD/train/verl:$PWD/train/verl/vla-adapter/openvla-oft:${NAVSIM_ROOT:?Set NAVSIM_ROOT to pinned official checkout}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/outputs/hf-cache}"
export TOKENIZERS_PARALLELISM=false
