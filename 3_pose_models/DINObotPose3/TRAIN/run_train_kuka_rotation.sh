#!/usr/bin/env bash
# KUKA iiwa7 rotation head — predict robot->camera 6D rotation (+translation) to seed the
# kinematic solver R_init. GT R,t built by Kabsch of iiwa7 FK(gt angles) onto camera-frame
# GT keypoints. Frozen KUKA detector; only rot_head trains. Single GPU (UUID).
#   Usage: GPU=GPU-<uuid> bash run_train_kuka_rotation.sh [detector_ckpt]
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:?set GPU=GPU-<uuid> (nvidia-smi -L to list)}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DET="${1:-./outputs_heatmap/kuka_dream_detector_20260709_183119/best_heatmap.pth}"
OUT="${OUT:-./outputs_rotation/kuka_rot_${STAMP}}"
export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "$OUT"

python3 train_rotation.py \
  --detector-ckpt "$DET" \
  --train-dir ../../../datasets/synthetic/kuka_synth_train_dr \
  --val-dir   ../../../datasets/synthetic/kuka_synth_test_dr \
  --keypoint-names iiwa7_link_1,iiwa7_link_2,iiwa7_link_3,iiwa7_link_4,iiwa7_link_5,iiwa7_link_6,iiwa7_link_7 \
  --fk-robot kuka \
  --angle-joint-names iiwa7_joint_1,iiwa7_joint_2,iiwa7_joint_3,iiwa7_joint_4,iiwa7_joint_5,iiwa7_joint_6,iiwa7_joint_7 \
  --crop-to-robot --crop-margin 1.5 \
  --output-dir "$OUT" \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --image-size 512 --batch-size "${BATCH_SIZE:-32}" \
  --epochs "${EPOCHS:-30}" --lr 1e-3 --min-lr 1e-6 --weight-decay 1e-4 --num-workers "${NUM_WORKERS:-8}" \
  --t-weight 50.0 \
  --use-wandb --wandb-project dinov3-kuka-rotation --wandb-run-name "kuka_rot_${STAMP}" \
  2>&1 | tee "$OUT/train.log"
