#!/usr/bin/env bash
# Record an evaluation bag for Dev 2 (sim now, real camera later).
# Records sensor INPUTS only (+ ground truth if present); excludes /tf so the stack
# regenerates odom->base_link and map->odom on replay.
#
#   bash record_eval_bag.sh eval_bags/sim_loop1
# Override topics via env: IMAGE_TOPIC, CAMERA_INFO_TOPIC, WHEEL_ODOM_TOPIC, GT_TOPIC
set -euo pipefail

OUT="${1:?usage: record_eval_bag.sh <output_dir>}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/camera/image_raw}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/camera/camera_info}"
WHEEL_ODOM_TOPIC="${WHEEL_ODOM_TOPIC:-/wheel/odom}"
GT_TOPIC="${GT_TOPIC:-/ground_truth/odom}"   # Dev 5 sim; TBD

exec ros2 bag record -o "$OUT" \
  "$IMAGE_TOPIC" "$CAMERA_INFO_TOPIC" "$WHEEL_ODOM_TOPIC" "$GT_TOPIC" \
  /tf_static /clock
