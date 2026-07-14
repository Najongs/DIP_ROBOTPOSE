#!/usr/bin/env bash
# Parallel variant of the SigLIP2 pose-cascade: angle and rot heads are INDEPENDENT (rot reuses
# AnglePredictor with with_rotation, does not consume the angle head), so train them concurrently on
# two separate GPUs to halve wall-clock. Both freeze the siglip backbone+crop-detector; only the head
# trains. Matches the DINOv3 crop recipe (angle 50ep / rot 30ep). Each writes its own done-marker.
set -u
cd "$(dirname "$0")"
MODEL=google/siglip2-base-patch16-512
DET=outputs_heatmap/siglip_crop_ddp_20260715_014111/best_heatmap.pth
TRAIN_DIR=../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr
VAL_DIR=../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr
SCRATCH=/tmp/claude-1002/-home-najo-NAS-DIP/5aafbd5b-1895-41b2-90ed-8d6e9438b7dd/scratchpad

mapfile -t U < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
# angle -> U[1], rot -> U[2] (GPU4/idx? both free). Override via $1 $2 (uuid indices).
GA=${1:-1}; GR=${2:-2}
TS=$(date +%Y%m%d_%H%M%S)
AOUT="./outputs_angle/siglip_angle_crop_${TS}"; mkdir -p "$AOUT"
ROUT="./outputs_rotation/siglip_rot_crop_${TS}"; mkdir -p "$ROUT"
echo "$AOUT" > "$SCRATCH/siglip_angle_out.txt"
echo "$ROUT" > "$SCRATCH/siglip_rot_out.txt"
rm -f "$SCRATCH/siglip_angle_done.txt" "$SCRATCH/siglip_rot_done.txt"
echo "angle->gpu_uuidx$GA (${U[$GA]})  rot->gpu_uuidx$GR (${U[$GR]})  det=$DET"

( export CUDA_VISIBLE_DEVICES="${U[$GA]}"
  python3 train_angle.py --detector-ckpt "$DET" \
    --train-dir "$TRAIN_DIR" --val-dir "$VAL_DIR" --output-dir "$AOUT" \
    --model-name "$MODEL" --head-type mlp --image-size 512 --batch-size 32 --epochs 50 \
    --lr 1e-3 --min-lr 1e-6 --weight-decay 1e-4 --fk-weight 10.0 \
    --crop-to-robot --crop-margin 1.5 --num-workers 8 > "$AOUT/train.log" 2>&1
  echo "ANGLE DONE $AOUT" > "$SCRATCH/siglip_angle_done.txt" ) &

( export CUDA_VISIBLE_DEVICES="${U[$GR]}"
  python3 train_rotation.py --detector-ckpt "$DET" \
    --train-dir "$TRAIN_DIR" --val-dir "$VAL_DIR" --output-dir "$ROUT" \
    --model-name "$MODEL" --image-size 512 --batch-size 32 --epochs 30 \
    --lr 1e-3 --min-lr 1e-6 --weight-decay 1e-4 --num-workers 8 \
    --t-weight 50.0 --crop-to-robot --crop-margin 1.5 > "$ROUT/train.log" 2>&1
  echo "ROT DONE $ROUT" > "$SCRATCH/siglip_rot_done.txt" ) &

wait
echo "SIGLIP POSE PARALLEL COMPLETE" > "$SCRATCH/siglip_cascade_done.txt"
