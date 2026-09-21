#!/usr/bin/env bash
# Run only inside a NEW Python 3.10 virtualenv. Does not overwrite existing envs.
set -euo pipefail
if [[ "${RFT_RUNTIME:-}" == ppu ]] || command -v ppu-smi >/dev/null 2>&1; then
  printf 'PPU detected: use debug_2ppu.sh or train_16gpu.sh --runtime ppu; preserve vendor torch.\n' >&2
  exit 2
fi
: "${VIRTUAL_ENV:?Activate a dedicated Python 3.10 virtualenv}"
: "${NAVSIM_ROOT:?Set path to NAVSIM v1.1 checkout}"
python -c 'import sys; assert sys.version_info[:2] == (3,10), "Require Python 3.10"'
test "$(git -C "$NAVSIM_ROOT" rev-parse HEAD)" = 0811876c274e8b058ab2be9b3dcd4d37bd23f177
python -m pip install -r configs/navsim/requirements-server.txt
# Replace only the incompatible upstream version pins, keep official simulator dependencies.
python - "$NAVSIM_ROOT" <<'PY'
import os,sys
from pathlib import Path
blocked={'torch','torchvision','numpy','hydra-core','scipy','pandas'}
lines=Path(sys.argv[1],'requirements.txt').read_text().splitlines()
lines=[s for s in lines if not any(s.startswith(k+'==') or s.startswith(k+'>=') for k in blocked)]
out=Path(os.environ.get('RFT_REQUIREMENTS_OUTPUT','outputs/navsim-requirements.txt'))
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text('\n'.join(lines)+'\n')
PY
python -m pip install -r "${RFT_REQUIREMENTS_OUTPUT:-outputs/navsim-requirements.txt}" -c configs/navsim/requirements-server.txt
# NAVSIM is imported via PYTHONPATH; do not install its conflicting torch-2.0 metadata.
python -m pip check
