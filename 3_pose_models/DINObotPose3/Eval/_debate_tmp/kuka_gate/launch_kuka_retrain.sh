#!/usr/bin/env bash
# KUKA iiwa7 detector RETRAIN — attacks the diffuse-heatmap tail (gate verdict: detector-limited,
# correct-mode ABSENT, swaps are LOW-conf diffuse not confident-wrong).
# Changes vs deployed kuka_dream_detector_20260709_183119 (val 2D-AUC 0.735):
#   (1) warm-start from that detector (keep the 0.735 floor)
#   (2) unfreeze 4 backbone blocks (was 2) — capacity to encode link-identity / positional context
#       so near-identical iiwa cylinders stop collapsing to diffuse heatmaps
#   (3) KUKA-aligned joint-weights — deployed used the legacy Panda U-shape [2.5,1.5,1.3,1.0,1.3,1.5,2.5]
#       which UP-weights KUKA's BEST link (L6, 17.5% cata) and UNDER-weights its WORST (L2, 28.9%).
#       New weights ~ per-link catastrophic rate: [1.6,1.65,1.9,1.65,1.2,1.4,1.15].
#   everything else identical (crop_to_robot 1.5, aug strong, occ 0.3/0.18, 512).
set -euo pipefail
cd "$(dirname "$0")/../../../TRAIN"
GPU="${GPU:?set GPU=GPU-<uuid>}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DET="./outputs_heatmap/kuka_dream_detector_20260709_183119/best_heatmap.pth"
OUT="./outputs_heatmap/kuka_detector_retrain_${STAMP}"
export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_MODE=offline
mkdir -p "$OUT"

/home/najo/.conda/envs/dino/bin/python train_heatmap.py \
  --data-dir ../../../datasets/synthetic/kuka_synth_train_dr \
  --val-dir  ../../../datasets/synthetic/kuka_synth_test_dr \
  --checkpoint "$DET" \
  --keypoint-names iiwa7_link_1,iiwa7_link_2,iiwa7_link_3,iiwa7_link_4,iiwa7_link_5,iiwa7_link_6,iiwa7_link_7 \
  --joint-weights 1.6,1.65,1.9,1.65,1.2,1.4,1.15 \
  --output-dir "$OUT" \
  --image-size 512 --heatmap-size 512 \
  --crop-to-robot --crop-margin 1.5 \
  --unfreeze-blocks 4 \
  --aug-level strong --occlusion-prob 0.3 --occlusion-size 0.18 \
  --epochs 10 --batch-size 16 --num-workers 8 \
  --learning-rate 2e-4 --backbone-lr 2e-5 --min-lr 1e-7 --weight-decay 1e-4 \
  --amp \
  --wandb-project dinov3-dream-kuka-detector \
  --wandb-run-name "kuka_detector_retrain_${STAMP}" \
  2>&1 | tee "$OUT/train.log"
