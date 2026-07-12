#!/usr/bin/env bash
# KUKA iiwa7 angle head — frozen KUKA detector + trainable angle head.
# Predicts joint_1..6 (joint_7 fixed=0; it does not move any link_1..7 origin), iiwa7 FK
# consistency loss. Single GPU (head-only). Select GPU by UUID (integer indices scrambled).
#   Usage: GPU=GPU-<uuid> bash run_train_kuka_angle.sh [detector_ckpt]
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:?set GPU=GPU-<uuid> (nvidia-smi -L to list)}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DET="${1:-./outputs_heatmap/baxter_left_dream_detector_20260710_152926/best_heatmap.pth}"
OUT="${OUT:-./outputs_angle/baxter_angle_${STAMP}}"
export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "$OUT"

python3 train_angle.py \
  --detector-ckpt "$DET" \
  --train-dir ../../../datasets/synthetic/baxter_synth_train_dr \
  --val-dir   ../../../datasets/synthetic/baxter_synth_test_dr \
  --keypoint-names left_s0,left_s1,left_e0,left_e1,left_w0,left_w1,left_w2 \
  --fk-robot baxter \
  --angle-joint-names left_s0,left_s1,left_e0,left_e1,left_w0,left_w1,left_w2 \
  --crop-to-robot --crop-margin 1.5 \
  --output-dir "$OUT" \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --image-size 512 --batch-size "${BATCH_SIZE:-32}" \
  --epochs "${EPOCHS:-60}" --lr 1e-3 --min-lr 1e-6 --weight-decay 1e-4 \
  --fk-weight 10.0 --num-workers "${NUM_WORKERS:-8}" \
  --use-wandb --wandb-project dinov3-baxter-angle --wandb-run-name "baxter_angle_${STAMP}" \
  2>&1 | tee "$OUT/train.log"
