#!/bin/bash
# RoboPEPP-protocol occlusion robustness bench (arXiv:2411.17662): black rect/circle masks covering
# {0,10,20,30,40}% of the GT-keypoint RoI on panda_synth_test_photo, deterministic per (frame,ratio).
# Runs the full deployable pipeline (auto bbox + crop + solve), then nvdr+SAM render-compare on top.
# Usage: bash occlusion_bench.sh <ratio> [gpu] [frames] [val_dir]
# Their curve (Fig.6): RoboPEPP 79.5/73/60/47/35.1, HPE 57/50.5/40.5/32/28.2, RoboPose 54/42/28/21/14.5
set -eu
cd "$(dirname "$(readlink -f "$0")")"
R="$1"; GPU="${2:-1}"; MAXF="${3:-200}"
VAL="${4:-../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_photo}"
# POSE_EXTRA / RC_EXTRA: pass occlusion-robustness options, e.g.
#   POSE_EXTRA="--cov-pnp --prior-adaptive 0.02"  RC_EXTRA="--occl-robust-w 0.2"
# LABEL: suffix to keep A/B logs apart.
POSE_EXTRA="${POSE_EXTRA:-}"; RC_EXTRA="${RC_EXTRA:-}"; LABEL="${LABEL:-base}"
TAG=$(basename "$VAL")_r${R}_cleanhead
export CUDA_VISIBLE_DEVICES=$GPU CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.2}

mkdir -p rc_dumps occl_logs
python selfbbox_eval.py \
  --stage1-detector ../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth \
  --stage1-angle ../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth \
  --stage1-rot ../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth \
  --crop-detector ../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth \
  --crop-angle ../TRAIN/outputs_angle/angle_crop_20260605_174740/best_angle_head.pth \
  --rot-head ../TRAIN/outputs_rotation/rot_crop_20260606_022535/best_rot_head.pth \
  --bbox-from-solved --bbox-guard --cov-pnp --dark-decode --occlude-ratio "$R" $POSE_EXTRA \
  --val-dir "$VAL" --max-frames "$MAXF" \
  --dump-npz rc_dumps/occl_${TAG}.npz > occl_logs/${TAG}_pose.log 2>&1
P=$(grep -aoE 'ADD-AUC@100mm: [0-9.]+' occl_logs/${TAG}_pose.log | tail -1 | grep -oE '[0-9.]+$')

python rc_refine_from_dump.py \
  --dump rc_dumps/occl_${TAG}.npz --val-dir "$VAL" \
  --sam-checkpoint ../weights_sam/sam_vit_b_01ec64.pth \
  --render-h 448 --batch-size 4 --occlude-ratio "$R" $RC_EXTRA > occl_logs/${TAG}_rc.log 2>&1
RC=$(grep -aoE 'render-compare ADD-AUC@100mm [0-9.]+' occl_logs/${TAG}_rc.log | tail -1 | grep -oE '[0-9.]+$')

echo "OCCL_BENCH ${TAG}  pose=${P:-FAIL}  +RC=${RC:-FAIL}"
