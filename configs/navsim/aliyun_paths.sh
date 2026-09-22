#!/usr/bin/env bash
# User-provided Alibaba Cloud CPFS layout. Explicit environment values take priority.
# Source this file; it only exports paths and never creates or changes directories.
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/map}"
export TRAIN_LOGS="${TRAIN_LOGS:-$OPENSCENE_DATA_ROOT/navsim_logs/trainval}"
export TRAIN_SENSORS="${TRAIN_SENSORS:-$OPENSCENE_DATA_ROOT/sensor_blobs/trainval}"
export TEST_LOGS="${TEST_LOGS:-$OPENSCENE_DATA_ROOT/navsim_logs/test}"
export TEST_SENSORS="${TEST_SENSORS:-$OPENSCENE_DATA_ROOT/sensor_blobs/test}"
export MINI_LOGS="${MINI_LOGS:-$OPENSCENE_DATA_ROOT/navsim_logs/mini}"
export MINI_SENSORS="${MINI_SENSORS:-$OPENSCENE_DATA_ROOT/sensor_blobs/mini}"
# Independent output/cache defaults alongside the dataset, each launch creates a new run.
export RFT_OUTPUT_ROOT="${RFT_OUTPUT_ROOT:-/mnt/cpfs-wlc-rdma-300t/navsim/vla-rft/runs}"
export RFT_CACHE_ROOT="${RFT_CACHE_ROOT:-/mnt/cpfs-wlc-rdma-300t/navsim/vla-rft/cache}"
