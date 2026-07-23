#!/usr/bin/env bash
# Baxter WHOLE-BODY (17-keypoint) heatmap detector — matches the DREAM baxter benchmark
# (torso + both arms). Warm-start the backbone from the baxter-LEFT detector (baxter-adapted);
# the 17-channel heatmap_predictor + refine convs reinit (shape mismatch, filtered on load).
set -euo pipefail
cd "$(dirname "$0")"

GPU_IDS="${GPU_IDS:-GPU-7ff6997b-14c1-9283-5119-251c9c899b8e,GPU-c3d180c2-f92c-fe13-b7b8-337247e36a33,GPU-70a2a406-5d77-3533-e8ba-0c9d338f4a11,GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd,GPU-1cdd7bc8-7c5a-dc09-d4fc-5dcfe92e104d}"
NUM_GPUS="${NUM_GPUS:-5}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-./outputs_heatmap/baxter_fullbody_detector_${STAMP}}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export WANDB_MODE="${WANDB_MODE:-offline}"

KPS="torso_t0,left_s0,left_s1,left_e0,left_e1,left_w0,left_w1,left_w2,left_hand,right_s0,right_s1,right_e0,right_e1,right_w0,right_w1,right_w2,right_hand"

PY="${PY:-/home/najo/.conda/envs/dino/bin/python}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" train_heatmap.py \
  --data-dir ../../../datasets/synthetic/baxter_synth_train_dr \
  --val-dir ../../../datasets/synthetic/baxter_synth_test_dr \
  --checkpoint ./outputs_heatmap/baxter_left_dream_detector_20260710_152926/best_heatmap.pth \
  --keypoint-names "$KPS" \
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
  --wandb-project dinov3-dream-baxter-fullbody-detector \
  --wandb-run-name "baxter_fullbody_detector_${STAMP}"
echo "OUT=$OUT"
