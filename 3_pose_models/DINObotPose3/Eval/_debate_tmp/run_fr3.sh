#!/bin/bash
set -u
cd /home/najo/NAS/DIP/3_pose_models/DINObotPose3/Eval
export CUDA_VISIBLE_DEVICES=GPU-c3d180c2-f92c-fe13-b7b8-337247e36a33
PY=/home/najo/.conda/envs/dino/bin/python
CD=/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset
L=_debate_tmp
STAGE1=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth
CROPDET=../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
CROPANG=../TRAIN/outputs_angle/angle_crop_20260605_174740/best_angle_head.pth
CROPROT=../TRAIN/outputs_rotation/rot_crop_20260606_022535/best_rot_head.pth
FR3DET=../TRAIN/outputs_fr3/crop_det_20260704_133515/best_heatmap.pth
FR3ANG=../TRAIN/outputs_fr3/angle/best_angle_head.pth
FR3ROT=../TRAIN/outputs_fr3/rot/best_rot_head.pth
FR3ZS=$CD/franka_research3_to_DREAM_modified

echo "[1/4] FR3 zero-shot (Panda synth heads) FULL 800"
$PY selfbbox_eval.py --stage1-detector $STAGE1 --stage1-angle $S1ANG --stage1-rot $S1ROT \
  --crop-detector $CROPDET --crop-angle $CROPANG --rot-head $CROPROT \
  --bbox-from-solved --bbox-guard --dark-decode --cov-pnp \
  --max-frames 800 --val-dir $FR3ZS > $L/fr3_zeroshot.log 2>&1
echo "[2/4] FR3 trained single-frame FULL 1442"
$PY selfbbox_eval.py --stage1-detector $STAGE1 --stage1-angle $S1ANG --stage1-rot $S1ROT \
  --crop-detector $FR3DET --crop-angle $FR3ANG --rot-head $FR3ROT \
  --bbox-from-solved --bbox-guard --dark-decode --cov-pnp \
  --max-frames 1442 --val-dir $CD/fr3_val > $L/fr3_trained.log 2>&1
echo "[3/4] FR3 tracker seed=head"
$PY fr3_track_eval.py --detector $FR3DET --rot-head $FR3ROT --angle-head $FR3ANG \
  --val-dir $CD/fr3_val --seed head > $L/fr3_track_head.log 2>&1
echo "[4/4] FR3 tracker seed=gt"
$PY fr3_track_eval.py --detector $FR3DET --rot-head $FR3ROT --angle-head $FR3ANG \
  --val-dir $CD/fr3_val --seed gt > $L/fr3_track_gt.log 2>&1
echo "FR3_ALL_DONE" > $L/FR3_DONE
