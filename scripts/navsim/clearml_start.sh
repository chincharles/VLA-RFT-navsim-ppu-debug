#!/usr/bin/env bash
# Run once as a ClearML task entry point, after ClearML has cloned this repository.
# No ClearML API credentials or SDK are required by this script.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'HELP'
ClearML bounded real-data startup (Linux, Python 3.10, one allocated CUDA GPU).
Required environment:
  OPENSCENE_DATA_ROOT  Mounted v1 data root containing navsim_logs and sensor_blobs
  NUPLAN_MAPS_ROOT     Mounted nuPlan maps root
  RFT_OUTPUT_ROOT     Persistent output parent; creates a fresh run per launch
  RFT_CACHE_ROOT      Persistent pip/Hugging Face/reward-weight cache
Optional environment:
  RFT_PYTHON          Python 3.10 executable (default python3)
  NAVSIM_ROOT         Existing exact v1.1 checkout; otherwise fetched into this run
  VLM_DIR             Existing compatible local HF checkpoint, instead of download
  RFT_VLM_REPO        Public initialization candidate (default VLA-Adapter/LIBERO-Object)
  RFT_VLM_REVISION    Requested HF revision; resolved and recorded as immutable SHA
  TRAIN_LOGS, TRAIN_SENSORS  Override the default data-root/trainval paths

Creates 16 train + 4 validation clips, tokenizer 20 steps, WM/SFT 2+1 steps,
image-reward and driving-reward RL one step each, and official validation PDMS.
This is a smoke run, not converged training. Never wrap this entry in torchrun.
HELP
  exit 0
fi
if (( $# )); then
  printf 'Unexpected arguments; use --help or configure environment variables.\n' >&2
  exit 2
fi

: "${OPENSCENE_DATA_ROOT:?Set the mounted NAVSIM v1 data root}"
: "${NUPLAN_MAPS_ROOT:?Set the mounted nuPlan maps root}"
: "${RFT_OUTPUT_ROOT:?Set a persistent output parent directory}"
: "${RFT_CACHE_ROOT:?Set a persistent download/cache directory}"
export OPENSCENE_DATA_ROOT NUPLAN_MAPS_ROOT RFT_OUTPUT_ROOT RFT_CACHE_ROOT

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
RFT_PYTHON="${RFT_PYTHON:-python3}"
command -v "$RFT_PYTHON" >/dev/null
command -v git >/dev/null
"$RFT_PYTHON" - <<'PY'
import os, sys
from pathlib import Path
if sys.platform != 'linux' or sys.version_info[:2] != (3, 10):
    raise SystemExit('Select a Linux Python 3.10 image/runtime; ClearML does not install Python for this entry.')
for name in ('OPENSCENE_DATA_ROOT', 'NUPLAN_MAPS_ROOT', 'RFT_OUTPUT_ROOT', 'RFT_CACHE_ROOT'):
    if not Path(os.environ[name]).is_absolute():
        raise SystemExit(f'{name} must be an absolute mounted path')
for name in ('OPENSCENE_DATA_ROOT', 'NUPLAN_MAPS_ROOT'):
    if not Path(os.environ[name]).is_dir():
        raise SystemExit(f'Missing mount: {name}')
if int(os.environ.get('WORLD_SIZE', '1')) != 1 or int(os.environ.get('RANK', '0')) != 0 or int(os.environ.get('LOCAL_RANK', '0')) != 0:
    raise SystemExit('Launch one bootstrap process. Do not use torchrun/multi-worker launch for this smoke task.')
if os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'):
    raise SystemExit('ClearML has not assigned a visible CUDA GPU')
PY

export TRAIN_LOGS="${TRAIN_LOGS:-$OPENSCENE_DATA_ROOT/navsim_logs/trainval}"
export TRAIN_SENSORS="${TRAIN_SENSORS:-$OPENSCENE_DATA_ROOT/sensor_blobs/trainval}"
test -d "$TRAIN_LOGS" || { printf 'Missing TRAIN_LOGS: %s\n' "$TRAIN_LOGS" >&2; exit 2; }
test -d "$TRAIN_SENSORS" || { printf 'Missing TRAIN_SENSORS: %s\n' "$TRAIN_SENSORS" >&2; exit 2; }
mkdir -p "$RFT_OUTPUT_ROOT" "$RFT_CACHE_ROOT"
export RFT_WORK
RFT_WORK="$(mktemp -d "$RFT_OUTPUT_ROOT/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
exec > >(tee "$RFT_WORK/launcher.log") 2>&1
stage=initialization
finish() {
  local status=$?
  trap - EXIT
  printf '{"exit_code":%d,"last_stage":"%s"}\n' "$status" "$stage" > "$RFT_WORK/launcher-status.json"
  printf '\nClearML launcher exit=%s stage=%s output=%s\n' "$status" "$stage" "$RFT_WORK"
  exit "$status"
}
trap finish EXIT
phase() { stage="$1"; printf '\n[%s] %s\n' "$(date -u +%FT%TZ)" "$stage"; }
printf 'Run directory: %s\nRepository: %s\n' "$RFT_WORK" "$REPO_ROOT"

# Select the first assigned device, preserving UUID/MIG/physical-index identities.
# With no CUDA mask, 0 means the first device already visible inside this container.
# Masking the rest also keeps checkpoint CUDA RNG handling on this single device.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
# Keep network credentials; avoid inherited Python package paths.
unset PYTHONPATH PYTHONHOME
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$RFT_CACHE_ROOT/pip"
export HF_HOME="$RFT_CACHE_ROOT/huggingface"
export TORCH_HOME="$RFT_CACHE_ROOT/torch"
export TOKENIZERS_PARALLELISM=false
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT="$RFT_WORK/navsim-exp"
export TRAIN_MANIFEST="$RFT_WORK/data/train16/manifest.json"
export VAL_MANIFEST="$RFT_WORK/data/val4/manifest.json"
export ACTION_STATS="$RFT_WORK/data/train16/stats.json"
export TRAIN_METRIC_CACHE="$RFT_WORK/metric-cache"
export CONFIG_DIR="$RFT_WORK/config"
export RUN_DIR="$RFT_WORK/run"
mkdir -p "$NAVSIM_EXP_ROOT"

phase navsim_checkout
readonly NAVSIM_PIN=0811876c274e8b058ab2be9b3dcd4d37bd23f177
if [[ -z "${NAVSIM_ROOT:-}" ]]; then
  export NAVSIM_ROOT="$RFT_WORK/deps/navsim"
  git init -q "$NAVSIM_ROOT"
  git -C "$NAVSIM_ROOT" remote add origin https://github.com/autonomousvision/navsim.git
  git -C "$NAVSIM_ROOT" fetch --depth 1 origin "$NAVSIM_PIN"
  git -C "$NAVSIM_ROOT" checkout --detach FETCH_HEAD
fi
export NAVSIM_ROOT
test "$(git -C "$NAVSIM_ROOT" rev-parse HEAD)" = "$NAVSIM_PIN"
# Never reset an externally supplied checkout or silently use edited evaluator code.
test -z "$(git -C "$NAVSIM_ROOT" status --porcelain --untracked-files=no)" || {
  printf 'NAVSIM_ROOT contains tracked changes. Supply a clean pinned checkout.\n' >&2; exit 2;
}

phase install_environment
"$RFT_PYTHON" -m venv "$RFT_WORK/venv"
source "$RFT_WORK/venv/bin/activate"
bash scripts/navsim/install_server.sh
source scripts/navsim/env.sh
python -m pip freeze > "$RFT_WORK/requirements-resolved.txt"

phase cuda_check
python - <<'PY'
import json, os, subprocess
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'CUDA unavailable; check ClearML GPU assignment and driver/image'
x = torch.randn(512, 512, device='cuda:0')
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all(), 'CUDA matrix computation failed'
record = dict(torch=torch.__version__, cuda=torch.version.cuda,
              gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
              visible_device_count=torch.cuda.device_count(), selected_device='cuda:0',
              seed=42, train_scenes=16, validation_scenes=4,
              tokenizer_steps=20, wm_steps=3, sft_steps=3, image_rl_steps=1, driving_rl_steps=1,
              data_root=os.environ['OPENSCENE_DATA_ROOT'], maps_root=os.environ['NUPLAN_MAPS_ROOT'],
              navsim_commit=subprocess.check_output(['git', '-C', os.environ['NAVSIM_ROOT'], 'rev-parse', 'HEAD'], text=True).strip())
try:
    record['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
except subprocess.CalledProcessError:
    record['code_commit'] = None
Path(os.environ['RFT_WORK'], 'environment.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
PY

phase export_train
python -m navsim_rft.cli export --root "$NAVSIM_ROOT" --logs "$TRAIN_LOGS" --sensors "$TRAIN_SENSORS" \
  --split navtrain --role train --limit 16 --out "$RFT_WORK/data/train16"
phase export_validation
python -m navsim_rft.cli export --root "$NAVSIM_ROOT" --logs "$TRAIN_LOGS" --sensors "$TRAIN_SENSORS" \
  --split navtrain --role val --limit 4 --out "$RFT_WORK/data/val4" --exclude "$TRAIN_MANIFEST"

phase download_weights
python scripts/navsim/clearml_weights.py
export VLM_DIR VLA_RFT_VGG16_PATH VLA_RFT_LPIPS_PATH
VLM_DIR="$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["RFT_WORK"],"weights.json")))["vlm"])')"
VLA_RFT_VGG16_PATH="$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["RFT_WORK"],"weights.json")))["vgg"])')"
VLA_RFT_LPIPS_PATH="$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["RFT_WORK"],"weights.json")))["lpips"])')"

# Check the VLM before spending GPU time on tokenizer/WM training.
phase configure
python scripts/navsim/configure.py --navsim-root "$NAVSIM_ROOT" --train-manifest "$TRAIN_MANIFEST" \
  --stats "$ACTION_STATS" --vlm "$VLM_DIR" --tokenizer "$RFT_WORK/tokenizer/pretrained-000020" \
  --metric-cache "$TRAIN_METRIC_CACHE" --output "$CONFIG_DIR"
phase verify_vlm
python scripts/navsim/check_vlm.py --config "$CONFIG_DIR/sft.json" --output "$RFT_WORK/vlm-check.json"

phase train_tokenizer
python -m navsim_rft.train_tokenizer --manifest "$TRAIN_MANIFEST" --stats "$ACTION_STATS" \
  --output "$RFT_WORK/tokenizer" --steps 20 --seed 42
phase metric_cache
python scripts/navsim/cache_subset.py --navsim-root "$NAVSIM_ROOT" --logs "$TRAIN_LOGS" \
  --manifests "$TRAIN_MANIFEST" "$VAL_MANIFEST" --cache "$TRAIN_METRIC_CACHE"
phase minimal_train_and_evaluate
bash scripts/navsim/minimal_server.sh
phase completed_smoke
