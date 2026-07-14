#!/usr/bin/env bash
# SigLIP2 crop-detector for the pose-level backbone comparison (matches DINOv3 crop-detector config,
# only swapping the backbone). 4-GPU DDP, batch 8/gpu = 32 effective (== DINOv3 config).
set -u
cd "$(dirname "$0")"
mapfile -t U < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
TS=$(date +%Y%m%d_%H%M%S)
OUT="./outputs_heatmap/siglip_crop_ddp_${TS}"; mkdir -p "$OUT"
echo "$OUT" > /tmp/claude-1002/-home-najo-NAS-DIP/5aafbd5b-1895-41b2-90ed-8d6e9438b7dd/scratchpad/siglip_crop_out.txt
WARM=outputs_heatmap/siglip2_unfrozen_20260602_184024/best_heatmap.pth
TRAIN_DIR=../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr
VAL_DIR=../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr
export CUDA_VISIBLE_DEVICES="${U[0]},${U[1]},${U[2]},${U[3]}"
echo "OUT=$OUT  GPUS=$CUDA_VISIBLE_DEVICES"
torchrun --standalone --nnodes=1 --nproc_per_node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    train_heatmap.py \
    --data-dir "$TRAIN_DIR" --val-dir "$VAL_DIR" \
    --checkpoint "$WARM" \
    --model-name "google/siglip2-base-patch16-512" \
    --output-dir "$OUT" \
    --image-size 512 --heatmap-size 512 --crop-to-robot --crop-margin 1.5 \
    --unfreeze-blocks 4 --aug-level strong --occlusion-prob 0.0 --fda-prob 0.0 \
    --epochs 20 --batch-size 8 --num-workers 8 \
    --learning-rate 2e-4 --backbone-lr 2e-5 --min-lr 1e-7 --weight-decay 1e-5 \
    > "$OUT/train.log" 2>&1
