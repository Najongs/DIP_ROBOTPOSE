#!/usr/bin/env bash
# Whole-body Baxter angle head (12 joints) + rotation head, on a frozen 17-kp detector.
# Runs both in PARALLEL on two GPUs (heads are small MLPs on frozen features -> fast).
#   Usage: DET=<best_heatmap.pth> GPU_A=GPU-<uuid> GPU_B=GPU-<uuid> bash run_train_baxter_fullbody_heads.sh
set -euo pipefail
cd "$(dirname "$0")"

DET="${DET:?set DET=path/to/best_heatmap.pth}"
GPU_A="${GPU_A:?set GPU_A=GPU-<uuid>}"
GPU_B="${GPU_B:?set GPU_B=GPU-<uuid>}"
PY="${PY:-/home/najo/.conda/envs/dino/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TRAIN=../../../datasets/synthetic/baxter_synth_train_dr
VAL=../../../datasets/synthetic/baxter_synth_test_dr
MAXTRAIN="${MAXTRAIN:-40000}"
export WANDB_MODE="${WANDB_MODE:-offline}"

AOUT="./outputs_angle/baxter_fb_angle_${STAMP}"
ROUT="./outputs_rotation/baxter_fb_rot_${STAMP}"

CUDA_VISIBLE_DEVICES="$GPU_A" "$PY" train_angle_fullbody.py \
  --detector-ckpt "$DET" --train-dir "$TRAIN" --val-dir "$VAL" \
  --crop-to-robot --crop-margin 1.5 --image-size 512 --batch-size 32 \
  --epochs "${AEPOCHS:-25}" --lr 1e-3 --fk-weight 10.0 \
  --max-train "$MAXTRAIN" --max-val 1500 \
  --output-dir "$AOUT" --use-wandb --wandb-project dinov3-baxter-fullbody-angle \
  --wandb-run-name "baxter_fb_angle_${STAMP}" > "/tmp/baxter_fb_angle.log" 2>&1 &
APID=$!
echo "angle head training PID $APID -> $AOUT"

CUDA_VISIBLE_DEVICES="$GPU_B" "$PY" train_rotation_fullbody.py \
  --detector-ckpt "$DET" --train-dir "$TRAIN" --val-dir "$VAL" \
  --crop-to-robot --crop-margin 1.5 --image-size 512 --batch-size 32 \
  --epochs "${REPOCHS:-20}" --lr 1e-3 --t-weight 50.0 \
  --max-train "$MAXTRAIN" --max-val 1500 \
  --output-dir "$ROUT" --use-wandb --wandb-project dinov3-baxter-fullbody-rotation \
  --wandb-run-name "baxter_fb_rot_${STAMP}" > "/tmp/baxter_fb_rot.log" 2>&1 &
RPID=$!
echo "rot head training PID $RPID -> $ROUT"

echo "AOUT=$AOUT"; echo "ROUT=$ROUT"
wait $APID $RPID
echo "both heads done"
