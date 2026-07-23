#!/bin/bash
# Occlusion-robustness sweep for the occ-aug-washout A/B (paper contribution iii counterfactual).
# Runs selfbbox_eval pose stage (deployed pipeline, no RC) at RoboPEPP occlusion ratios 0/0.2/0.4
# on panda_synth_test_photo. Swaps ONLY the crop angle+rot head (A=deployed lightstack vs B=pure).
# Usage: bash sweep.sh <label> <crop_angle_head> <rot_head> [frames]
set -eu
cd "$(dirname "$(readlink -f "$0")")"
LABEL="$1"; CROP_ANGLE="$2"; ROT_HEAD="$3"; MAXF="${4:-200}"
D=/home/najo/NAS/DIP/3_pose_models/DINObotPose3
PY=/home/najo/.conda/envs/dino/bin/python
GPU=GPU-70a2a406-5d77-3533-e8ba-0c9d338f4a11
VAL=$D/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_photo
export CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline
mkdir -p logs
echo "==== SWEEP $LABEL  head=$CROP_ANGLE ===="
for R in 0.0 0.2 0.4; do
  LOG=logs/${LABEL}_r${R}.log
  $PY $D/Eval/selfbbox_eval.py \
    --stage1-detector $D/TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth \
    --stage1-angle    $D/TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth \
    --stage1-rot      $D/TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth \
    --crop-detector   $D/TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth \
    --crop-angle      "$CROP_ANGLE" \
    --rot-head        "$ROT_HEAD" \
    --bbox-from-solved --bbox-guard --cov-pnp --dark-decode --occlude-ratio $R \
    --val-dir "$VAL" --max-frames "$MAXF" > "$LOG" 2>&1
  A=$(grep -aoE 'ADD-AUC@100mm: [0-9.]+' "$LOG" | tail -1 | grep -oE '[0-9.]+$')
  echo "OCCL $LABEL  ratio=$R  ADD-AUC@100mm=${A:-FAIL}"
done
