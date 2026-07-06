#!/bin/bash
set -u
cd /home/najo/NAS/DIP-multirobot/3_pose_models/DINObotPose3/TRAIN
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate dino
export CUDA_VISIBLE_DEVICES=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
export WANDB_MODE=offline
CD=/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset
KPN=link0,link1,link2,link3,link4,link5,link6
# wait for the real detector
while [ ! -f outputs_meca500/REAL_DET_DONE ]; do sleep 60; done
DET=$(ls -t outputs_meca500/real_detector_*/best_heatmap.pth | head -1)
echo "using real detector: $DET"
# --- real rot head (the SOTA basin-pin; FR3 real rot head hit 0.21 deg) ---
ROUT=./outputs_meca500/real_rot_$(date +%Y%m%d_%H%M%S); mkdir -p "$ROUT"
python3 train_rotation.py \
  --detector-ckpt "$DET" --train-dir "$CD/meca_real_train" --val-dir "$CD/meca_real_val" \
  --keypoint-names "$KPN" --fk-robot meca500 --output-dir "$ROUT" \
  --crop-to-robot --crop-margin 1.5 --epochs 30 --batch-size 32 --lr 1e-3 --t-weight 1.0 --num-workers 12 \
  --wandb-project meca500-real-rot > "$ROUT/train.log" 2>&1
echo "ROT done -> $ROUT"
# --- real angle head (observable J0-J2; wrist stays mean by observability) ---
AOUT=./outputs_meca500/real_angle_$(date +%Y%m%d_%H%M%S); mkdir -p "$AOUT"
python3 train_angle.py \
  --detector-ckpt "$DET" --train-dir "$CD/meca_real_train" --val-dir "$CD/meca_real_val" \
  --keypoint-names "$KPN" --output-dir "$AOUT" \
  --crop-to-robot --crop-margin 1.5 --fk-weight 0 --reproj-weight 0 \
  --head-type mlp --epochs 40 --batch-size 32 --lr 1e-3 --num-workers 12 \
  --wandb-project meca500-real-angle > "$AOUT/train.log" 2>&1
echo "ANGLE done -> $AOUT"
echo "$ROUT $AOUT" > outputs_meca500/REAL_HEADS_DONE
