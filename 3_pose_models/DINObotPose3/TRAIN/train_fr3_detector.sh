#!/bin/bash
# Phase 1: FR3 crop detector finetune (warm-start Panda crop detector, supervised on FR3 real ArUco labels).
# FR3 kinematics/keypoints == Panda -> no code changes; just FR3 data + Panda-recipe crop training.
set -u
cd "$(dirname "$0")"
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate dino
export CUDA_VISIBLE_DEVICES=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
export WANDB_MODE=offline
TS=$(date +%Y%m%d_%H%M%S)
OUT=./outputs_fr3/crop_det_${TS}; mkdir -p "$OUT"
DATA=/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset
PRETRAIN=./outputs_heatmap/crop_20260605_010622/best_heatmap.pth
python3 train_heatmap.py \
  --data-dir "$DATA/fr3_train" \
  --val-dir "$DATA/fr3_val" \
  --checkpoint "$PRETRAIN" \
  --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
  --output-dir "$OUT" \
  --image-size 512 --heatmap-size 512 \
  --crop-to-robot --crop-margin 1.5 \
  --unfreeze-blocks 4 \
  --aug-level strong \
  --occlusion-prob 0.0 --fda-prob 0.0 \
  --epochs 20 \
  --batch-size 32 \
  --num-workers 12 \
  --learning-rate 2e-4 --backbone-lr 2e-5 --min-lr 1e-7 --weight-decay 1e-5 \
  --wandb-project "fr3-detector" --wandb-run-name "crop_${TS}" \
  > "$OUT/train.log" 2>&1
echo "FR3_DET_DONE $OUT" > outputs_fr3/DET_DONE
echo "best: $(grep -iE 'best|saved' $OUT/train.log | tail -3)"
