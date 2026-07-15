#!/usr/bin/env bash
# Pose-level backbone comparison (§4.10 / Table 12): SigLIP2 crop-angle + crop-rot heads on top of
# the SigLIP2 crop-detector, mirroring the DINOv3 crop config (run_train_angle_crop.sh /
# run_train_rotation.sh) — only the backbone is swapped. Backbone+detector are FROZEN; only the
# small head trains, so this is fast and single-GPU. Runs angle then rot sequentially on one free GPU.
set -u
cd "$(dirname "$0")"
MODEL=google/siglip2-base-patch16-512
DET=outputs_heatmap/siglip_crop_ddp_20260715_014111/best_heatmap.pth
TRAIN_DIR=../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr
VAL_DIR=../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr
SCRATCH=/tmp/claude-1002/-home-najo-NAS-DIP/5aafbd5b-1895-41b2-90ed-8d6e9438b7dd/scratchpad

# Pick a free GPU by UUID (integer indices are scrambled); default to the one with least mem used.
mapfile -t U < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
mapfile -t M < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
FREE_IDX=0; MIN=999999
for i in "${!M[@]}"; do if [ "${M[$i]}" -lt "$MIN" ]; then MIN=${M[$i]}; FREE_IDX=$i; fi; done
export CUDA_VISIBLE_DEVICES="${U[$FREE_IDX]}"
echo "Using GPU idx=$FREE_IDX (uuid ${U[$FREE_IDX]}, ${M[$FREE_IDX]}MiB used)  detector=$DET"

TS=$(date +%Y%m%d_%H%M%S)
AOUT="./outputs_angle/siglip_angle_crop_${TS}"; mkdir -p "$AOUT"
ROUT="./outputs_rotation/siglip_rot_crop_${TS}"; mkdir -p "$ROUT"
echo "$AOUT" > "$SCRATCH/siglip_angle_out.txt"
echo "$ROUT" > "$SCRATCH/siglip_rot_out.txt"

echo "=== [1/2] SigLIP2 crop-angle head ==="
python3 train_angle.py \
    --detector-ckpt "$DET" \
    --train-dir "$TRAIN_DIR" --val-dir "$VAL_DIR" \
    --output-dir "$AOUT" \
    --model-name "$MODEL" \
    --head-type mlp --image-size 512 --batch-size 32 --epochs 50 \
    --lr 1e-3 --min-lr 1e-6 --weight-decay 1e-4 --fk-weight 10.0 \
    --crop-to-robot --crop-margin 1.5 --num-workers 8 \
    > "$AOUT/train.log" 2>&1
echo "angle done -> $AOUT"

echo "=== [2/2] SigLIP2 crop-rot head ==="
python3 train_rotation.py \
    --detector-ckpt "$DET" \
    --train-dir "$TRAIN_DIR" --val-dir "$VAL_DIR" \
    --output-dir "$ROUT" \
    --model-name "$MODEL" \
    --image-size 512 --batch-size 32 --epochs 30 \
    --lr 1e-3 --min-lr 1e-6 --weight-decay 1e-4 --num-workers 8 \
    --t-weight 50.0 --crop-to-robot --crop-margin 1.5 \
    > "$ROUT/train.log" 2>&1
echo "rot done -> $ROUT"
echo "SIGLIP POSE CASCADE COMPLETE" > "$SCRATCH/siglip_cascade_done.txt"
