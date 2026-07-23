#!/usr/bin/env bash
# Minimal real-camera detector eval to dump raw 2D (kp2d_full, gtkp2d) for apples-to-apples
# vs synth. Same shared crop-detector as synth+deployment. No oracle-angle (deployed crop), no RC.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."   # -> Eval/
export CUDA_VISIBLE_DEVICES=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
PY=/home/najo/.conda/envs/dino/bin/python
DET=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth
CROPDET=../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
ST=../TRAIN/outputs_selftrain
DATA=../Dataset/Converted_dataset/DREAM_real
OUT=_debate_tmp
mkdir -p "$OUT"

run() {
  local CAM="$1" VAL="$2" HEAD="$3" ROT="$4"
  echo "=== $CAM START $(date +%T) ==="
  $PY selfbbox_eval.py --stage1-detector $DET --stage1-angle $S1ANG --stage1-rot $S1ROT \
    --crop-detector $CROPDET --crop-angle "$HEAD" --rot-head "$ROT" \
    --bbox-from-solved --bbox-guard --cov-pnp --dark-decode \
    --frac-range 0.7 1.0 --max-frames 400 --val-dir "$VAL" \
    --dump-npz $OUT/real2d_${CAM}.npz > $OUT/real2d_${CAM}.log 2>&1
  echo "=== $CAM DONE $(date +%T) rc=$? ==="
}

run realsense $DATA/panda-3cam_realsense $ST/realsense_lightstack_20260705_003546/best_selftrain_head.pth $ST/realsense_lightstack_20260705_003546/best_selftrain_rot.pth
run kinect    $DATA/panda-3cam_kinect360  $ST/kinect_lightstack_20260705_003552/best_selftrain_head.pth    $ST/kinect_lightstack_20260705_003552/best_selftrain_rot.pth
run orb       $DATA/panda-orb             $ST/orb_lightstack_20260705_003549/best_selftrain_head.pth        $ST/orb_lightstack_20260705_003549/best_selftrain_rot.pth
run azure     $DATA/panda-3cam_azure      ../TRAIN/outputs_angle/angle_occaug_light_20260704_015400/best_angle_head.pth ../TRAIN/outputs_rotation/rot_crop_occaug_20260704_002102/best_rot_head.pth
echo "ALL DONE $(date +%T)"
