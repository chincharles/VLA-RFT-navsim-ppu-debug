#!/usr/bin/env bash
# Explicitly opt in. Replicated data parallelism, not a claim of FSDP-equivalent scaling.
set -euo pipefail
source scripts/navsim/env.sh
: "${CONFIG:?Set one generated config}"
: "${OUTPUT:?Set independent output path}"
: "${STEPS:?Set an explicit step budget}"
: "${NPROC_PER_NODE:?Set GPUs per node}"
torchrun --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" --nproc_per_node="$NPROC_PER_NODE" \
 --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29500}" \
 -m navsim_rft.train --config "$CONFIG" --output "$OUTPUT" --steps "$STEPS" "$@"
