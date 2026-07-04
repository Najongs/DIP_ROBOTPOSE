#!/bin/bash
# Phase 1: FR3 zero-shot transfer of the Panda pipeline (FR3 kinematics == Panda).
# Uses generic synth-trained crop heads (FR3 cameras != DREAM cameras, so NOT the per-DREAM-cam self-train heads).
set -u
cd "$(dirname "$0")"
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate dino
export CUDA_VISIBLE_DEVICES=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
DET=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth
CROPDET=../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
CROPANG=../TRAIN/outputs_angle/angle_crop_20260605_174740/best_angle_head.pth   # generic synth crop angle
CROPROT=../TRAIN/outputs_rotation/rot_crop_20260606_022535/best_rot_head.pth     # generic synth crop rot
FR3=/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset/franka_research3_to_DREAM_modified
NF=${1:-800}
mkdir -p fr3_logs
python selfbbox_eval.py --stage1-detector $DET --stage1-angle $S1ANG --stage1-rot $S1ROT \
  --crop-detector $CROPDET --crop-angle $CROPANG --rot-head $CROPROT \
  --bbox-from-solved --bbox-guard --dark-decode --cov-pnp \
  --max-frames $NF --val-dir $FR3 --dump-npz rc_dumps/fr3_zeroshot.npz \
  > fr3_logs/zeroshot_base.log 2>&1
echo "[FR3 zero-shot base] $(grep -oE 'ADD-AUC@100mm: [0-9.]+' fr3_logs/zeroshot_base.log | tail -1)"
echo FR3_ZS_DONE > fr3_logs/DONE
