#!/bin/bash
set -u
cd /home/najo/NAS/DIP-multirobot/3_pose_models/DINObotPose3/TRAIN
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate dino
export CUDA_VISIBLE_DEVICES=GPU-7ff6997b-14c1-9283-5119-251c9c899b8e
export WANDB_MODE=offline
CD=/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset
KPN=link0,link1,link2,link3,link4,link5,link6
# --- 1) detector (warm-start Panda crop det, real FR5) ---
DOUT=./outputs_fr5/detector_$(date +%Y%m%d_%H%M%S); mkdir -p "$DOUT"
python3 train_heatmap.py \
  --data-dir "$CD/fr5_train" --val-dir "$CD/fr5_val" \
  --checkpoint ./outputs_heatmap/crop_20260605_010622/best_heatmap.pth \
  --keypoint-names "$KPN" --output-dir "$DOUT" \
  --image-size 512 --heatmap-size 512 --crop-to-robot --crop-margin 1.5 \
  --unfreeze-blocks 4 --aug-level strong --occlusion-prob 0.0 --fda-prob 0.0 \
  --epochs 20 --batch-size 16 --num-workers 10 \
  --learning-rate 2e-4 --backbone-lr 2e-5 --min-lr 1e-7 \
  --wandb-project fr5-detector > "$DOUT/train.log" 2>&1
echo "FR5_DET_DONE $DOUT" > outputs_fr5/FR5_DET_DONE
DET="$DOUT/best_heatmap.pth"
# --- 2) rot head (Fr5 FK) ---
ROUT=./outputs_fr5/rot_$(date +%Y%m%d_%H%M%S); mkdir -p "$ROUT"
python3 train_rotation.py \
  --detector-ckpt "$DET" --train-dir "$CD/fr5_train" --val-dir "$CD/fr5_val" \
  --keypoint-names "$KPN" --fk-robot fr5 --output-dir "$ROUT" \
  --crop-to-robot --crop-margin 1.5 --epochs 25 --batch-size 16 --lr 1e-3 --t-weight 1.0 --num-workers 10 \
  --wandb-project fr5-rot > "$ROUT/train.log" 2>&1
# --- 3) angle head (observable joints) ---
AOUT=./outputs_fr5/angle_$(date +%Y%m%d_%H%M%S); mkdir -p "$AOUT"
python3 train_angle.py \
  --detector-ckpt "$DET" --train-dir "$CD/fr5_train" --val-dir "$CD/fr5_val" \
  --keypoint-names "$KPN" --output-dir "$AOUT" \
  --crop-to-robot --crop-margin 1.5 --fk-weight 0 --reproj-weight 0 \
  --head-type mlp --epochs 30 --batch-size 16 --lr 1e-3 --num-workers 10 \
  --wandb-project fr5-angle > "$AOUT/train.log" 2>&1
echo "$ROUT $AOUT" > outputs_fr5/FR5_HEADS_DONE
