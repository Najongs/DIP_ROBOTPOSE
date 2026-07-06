#!/bin/bash
set -u
cd /home/najo/NAS/DIP-multirobot/3_pose_models/DINObotPose3/TRAIN
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate dino
export CUDA_VISIBLE_DEVICES=GPU-70a2a406-5d77-3533-e8ba-0c9d338f4a11
export WANDB_MODE=offline
CD=/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset
AOUT=./outputs_fr3/angle_rand_$(date +%Y%m%d_%H%M%S); mkdir -p "$AOUT"
python3 train_angle.py \
  --detector-ckpt ./outputs_fr3/crop_det_20260704_133515/best_heatmap.pth \
  --train-dir "$CD/fr3_rand_train" --val-dir "$CD/fr3_rand_val" \
  --output-dir "$AOUT" \
  --crop-to-robot --crop-margin 1.5 --fk-weight 0 --reproj-weight 0 \
  --head-type mlp --epochs 25 --batch-size 16 --lr 1e-3 --num-workers 10 \
  --wandb-project fr3-angle-rand > "$AOUT/train.log" 2>&1
echo "$AOUT" > outputs_fr3/ANGLE_RAND_DONE
