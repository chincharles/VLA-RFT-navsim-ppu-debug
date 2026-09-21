#!/usr/bin/env bash
# Use the PPU image's Python 3.12, preserving its torch/torchvision/SDK.
set -euo pipefail
RFT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$RFT_REPO/scripts/navsim/debug_2gpu.sh" "$@" --runtime ppu
