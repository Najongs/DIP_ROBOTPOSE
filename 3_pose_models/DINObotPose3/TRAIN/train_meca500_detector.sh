#!/bin/bash
set -u
cd "$(dirname "$0")"
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate dino
export CUDA_VISIBLE_DEVICES=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
export WANDB_MODE=offline
SYN=/home/najo/NAS/DIP/datasets/meca500_synth
OUT=./outputs_meca500/detector_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
python3 train_heatmap.py \
  --data-dir "$SYN/train" --val-dir "$SYN/val" \
  --checkpoint ./outputs_heatmap/crop_20260605_010622/best_heatmap.pth \
  --keypoint-names link0,link1,link2,link3,link4,link5,link6 \
  --output-dir "$OUT" \
  --image-size 512 --heatmap-size 512 --crop-to-robot --crop-margin 1.5 \
  --unfreeze-blocks 4 --aug-level strong --occlusion-prob 0.0 --fda-prob 0.0 \
  --epochs 20 --batch-size 32 --num-workers 12 \
  --learning-rate 2e-4 --backbone-lr 2e-5 --min-lr 1e-7 \
  --wandb-project meca500-detector > "$OUT/train.log" 2>&1
echo "MECA_DET_DONE $OUT" > outputs_meca500/DET_DONE
