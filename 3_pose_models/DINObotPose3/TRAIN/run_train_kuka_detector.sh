#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_IDS="${GPU_IDS:-0,1,2,3,4}"
NUM_GPUS="${NUM_GPUS:-$(tr ',' '\n' <<< "$GPU_IDS" | wc -l)}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-./outputs_heatmap/kuka_dream_detector_${STAMP}}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export WANDB_MODE="${WANDB_MODE:-offline}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" train_heatmap.py \
  --data-dir ../../../datasets/synthetic/kuka_synth_train_dr \
  --val-dir ../../../datasets/synthetic/kuka_synth_test_dr \
  --checkpoint ./outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth \
  --keypoint-names iiwa7_link_1,iiwa7_link_2,iiwa7_link_3,iiwa7_link_4,iiwa7_link_5,iiwa7_link_6,iiwa7_link_7 \
  --output-dir "$OUT" \
  --image-size 512 \
  --heatmap-size 512 \
  --crop-to-robot \
  --crop-margin 1.5 \
  --unfreeze-blocks 2 \
  --aug-level strong \
  --occlusion-prob 0.3 \
  --occlusion-size 0.18 \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --learning-rate "${LR:-2e-4}" \
  --backbone-lr "${BACKBONE_LR:-2e-5}" \
  --min-lr 1e-7 \
  --wandb-project dinov3-dream-kuka-detector \
  --wandb-run-name "kuka_dream_detector_${STAMP}"
