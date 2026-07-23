#!/usr/bin/env bash
# §4.10 backbone ablation — pretraining vs architecture (Table 12, detection level).
# Trains a stage1 full-frame detector (val on real-azure -> AUC) with a chosen backbone/regime,
# matching the DINOv3/SigLIP2 stage1 recipe; only the backbone (and pretrained-ness) changes.
#   run_backbone_ablation.sh <job> "<gpu_uuid1,gpu_uuid2,...>"
# job:
#   random-frozen    random-init ViT, FROZEN (head-only)  -> parameter-matched control (~collapse)
#   random-unfrozen  random-init ViT, full from-scratch    -> the decisive cell (long schedule)
#   sup-frozen       google/vit supervised, frozen (head-only)
#   sup-unfrozen     google/vit supervised, unfreeze last-4
#   dino-frozen-sanity  DINOv3 frozen -> harness sanity (should reproduce ~0.80)
set -u
cd "$(dirname "$0")"
JOB="${1:?job}"; GPUS="${2:?comma-separated GPU UUIDs}"
NPROC=$(awk -F, '{print NF}' <<<"$GPUS")
export CUDA_VISIBLE_DEVICES="$GPUS"
SCRATCH=/tmp/claude-1002/-home-najo-NAS-DIP/5aafbd5b-1895-41b2-90ed-8d6e9438b7dd/scratchpad
TS=$(date +%Y%m%d_%H%M%S)

TRAIN_DIR=../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr
VAL_DIR=../Dataset/Converted_dataset/DREAM_real/panda-3cam_azure
VIT=google/vit-base-patch16-384
DINO=facebook/dinov3-vitb16-pretrain-lvd1689m

case "$JOB" in
  random-frozen)      MODEL=$VIT;  EXTRA="--random-init --unfreeze-blocks 0";  EP=20; BS=10; LR="--learning-rate 2e-4"; WD=1e-5 ;;
  random-unfrozen)    MODEL=$VIT;  EXTRA="--random-init --unfreeze-blocks 12"; EP=80; BS=8;  LR="--learning-rate 2e-4 --backbone-lr 2e-4"; WD=0.05 ;;
  sup-frozen)         MODEL=$VIT;  EXTRA="--unfreeze-blocks 0";  EP=40; BS=10; LR="--learning-rate 2e-4"; WD=1e-5 ;;
  sup-unfrozen)       MODEL=$VIT;  EXTRA="--unfreeze-blocks 4";  EP=40; BS=8;  LR="--learning-rate 2e-4 --backbone-lr 2e-5"; WD=1e-5 ;;
  dino-frozen-sanity) MODEL=$DINO; EXTRA="--unfreeze-blocks 0";  EP=20; BS=10; LR="--learning-rate 2e-4"; WD=1e-5 ;;
  *) echo "unknown job: $JOB"; exit 1 ;;
esac

OUT="./outputs_heatmap/bbabl_${JOB}_${TS}"; mkdir -p "$OUT"
echo "$OUT" > "$SCRATCH/bbabl_${JOB}_out.txt"
echo "JOB=$JOB MODEL=$MODEL NPROC=$NPROC BS=$BS(/gpu) EP=$EP WD=$WD OUT=$OUT GPUS=$GPUS"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    train_heatmap.py \
    --data-dir "$TRAIN_DIR" --val-dir "$VAL_DIR" \
    --model-name "$MODEL" $EXTRA \
    --image-size 512 --heatmap-size 512 \
    --aug-level strong --occlusion-prob 0.0 --fda-prob 0.0 \
    $LR --min-lr 1e-7 --weight-decay "$WD" \
    --epochs "$EP" --batch-size "$BS" --num-workers 8 \
    --output-dir "$OUT" \
    --wandb-project dinov3-backbone-ablation --wandb-run-name "${JOB}_${TS}" \
    > "$OUT/train.log" 2>&1
echo "BBABL $JOB DONE exit=$?" > "$SCRATCH/bbabl_${JOB}_done.txt"
