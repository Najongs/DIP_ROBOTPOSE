#!/bin/bash
set -u
cd "$(dirname "$0")"
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate dino
export CUDA_VISIBLE_DEVICES=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
export WANDB_MODE=offline
DATA=/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset
DET=./outputs_fr3/crop_det_20260704_133515/best_heatmap.pth
OUT=./outputs_fr3/angle_tf; mkdir -p "$OUT"
python3 train_angle.py --detector-ckpt "$DET" --train-dir "$DATA/fr3_train" --val-dir "$DATA/fr3_val" \
  --output-dir "$OUT" --crop-to-robot --crop-margin 1.5 --fk-weight 10.0 \
  --head-type transformer --epochs 50 --batch-size 32 --lr 1e-3 --min-lr 1e-6 --num-workers 12 \
  > "$OUT/train.log" 2>&1
echo "FR3_ANGLE_TF_DONE" > outputs_fr3/ANGLE_TF_DONE
