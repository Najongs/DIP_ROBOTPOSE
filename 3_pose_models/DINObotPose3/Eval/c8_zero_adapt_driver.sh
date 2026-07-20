#!/usr/bin/env bash
# C8 critical experiment (2026-07-20 critic debate): zero-real-adaptation + RC.
# Synthetic angle (angle_occaug_light_20260704) + synthetic rot (rot_crop_occaug_20260704)
# — i.e. the azure deployed heads — applied to ALL cameras, per-camera RC as deployed.
# azure needs no run: its deployed config IS zero-adaptation (RC off) = 0.7945.
# Question this answers: does the pipeline beat RoboPEPP 0.780 with NO per-camera self-training?
set -uo pipefail
cd "$(dirname "$0")"
GPU=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd   # A6000 (largest free VRAM; all GPUs shared)
export PATH=/home/najo/.conda/envs/dino/bin:$PATH
for CAM in realsense kinect orb; do
  bash ablation_run.sh zero_adapt "$CAM" "$GPU"
done
echo "=== C8 zero_adapt results ==="
grep zero_adapt ablation_logs/results.tsv
