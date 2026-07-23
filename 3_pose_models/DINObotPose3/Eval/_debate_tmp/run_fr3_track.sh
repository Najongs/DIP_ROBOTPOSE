#!/bin/bash
set -u
cd /home/najo/NAS/DIP/3_pose_models/DINObotPose3/Eval
export CUDA_VISIBLE_DEVICES=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
PY=/home/najo/.conda/envs/dino/bin/python
CD=/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset
L=_debate_tmp
FR3DET=../TRAIN/outputs_fr3/crop_det_20260704_133515/best_heatmap.pth
FR3ANG=../TRAIN/outputs_fr3/angle/best_angle_head.pth
FR3ROT=../TRAIN/outputs_fr3/rot/best_rot_head.pth
echo "[a] tracker seed=head"
$PY fr3_track_eval.py --detector $FR3DET --rot-head $FR3ROT --angle-head $FR3ANG \
  --val-dir $CD/fr3_val --seed head > $L/fr3_track_head.log 2>&1
echo "[b] tracker seed=gt"
$PY fr3_track_eval.py --detector $FR3DET --rot-head $FR3ROT --angle-head $FR3ANG \
  --val-dir $CD/fr3_val --seed gt > $L/fr3_track_gt.log 2>&1
echo "FR3_TRACK_DONE" > $L/FR3_TRACK_DONE
