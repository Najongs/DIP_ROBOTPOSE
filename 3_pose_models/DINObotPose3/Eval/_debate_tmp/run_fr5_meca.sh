#!/bin/bash
set -u
cd /home/najo/NAS/DIP/3_pose_models/DINObotPose3/Eval
export CUDA_VISIBLE_DEVICES=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
PY=/home/najo/.conda/envs/dino/bin/python
CD=/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset
L=_debate_tmp
FR5D=../TRAIN/outputs_fr5/detector_20260706_131423/best_heatmap.pth
FR5A=../TRAIN/outputs_fr5/angle_20260706_153946/best_angle_head.pth
FR5R=../TRAIN/outputs_fr5/rot_20260706_142231/best_rot_head.pth
MD=../TRAIN/outputs_meca500/real_detector_20260706_122304/best_heatmap.pth
MA=../TRAIN/outputs_meca500/real_angle_20260706_132024/best_angle_head.pth
MR=../TRAIN/outputs_meca500/real_rot_20260706_125349/best_rot_head.pth

echo "[1/4] FR5 head-direct FULL"
$PY meca_add_eval.py --robot fr5 --head-direct --detector $FR5D --angle-head $FR5A --rot-head $FR5R \
  --val-dir $CD/fr5_val --max-frames 1500 --batch-size 32 > $L/fr5_headdirect.log 2>&1
echo "[2/4] FR5 solver FULL"
$PY meca_add_eval.py --robot fr5 --detector $FR5D --angle-head $FR5A --rot-head $FR5R \
  --val-dir $CD/fr5_val --max-frames 1500 --batch-size 32 > $L/fr5_solver.log 2>&1
echo "[3/4] MECA head-direct FULL"
$PY meca_add_eval.py --robot meca500 --head-direct --detector $MD --angle-head $MA --rot-head $MR \
  --val-dir $CD/meca_real_val --max-frames 1500 --batch-size 32 > $L/meca_headdirect.log 2>&1
echo "[4/4] MECA solver FULL"
$PY meca_add_eval.py --robot meca500 --detector $MD --angle-head $MA --rot-head $MR \
  --val-dir $CD/meca_real_val --max-frames 1500 --batch-size 32 > $L/meca_solver.log 2>&1
echo "FR5_MECA_ALL_DONE" > $L/FR5_MECA_DONE
