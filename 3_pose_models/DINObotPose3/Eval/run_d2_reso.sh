#!/usr/bin/env bash
# D2: RC render-resolution sweep on RealSense (predicted, held-out dump). Accuracy vs render-h.
set -u; cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="$1"
DUMP=rc_dumps_abl/full_realsense.npz
VAL=../Dataset/Converted_dataset/DREAM_real/panda-3cam_realsense
SAM=../weights_sam/sam_vit_b_01ec64.pth
for H in 224 320 448 512; do
  t0=$SECONDS
  python rc_refine_from_dump.py --dump $DUMP --val-dir $VAL --sam-checkpoint $SAM \
    --render-h $H --max-frames 1000 > d2_logs/reso_${H}.log 2>&1
  A=$(grep -haoE 'render-compare ADD-AUC@100mm[: ]+[0-9]+\.[0-9]+' d2_logs/reso_${H}.log | tail -1 | grep -oE '[0-9.]+$')
  printf 'render-h %s\tADD-AUC=%s\twall=%ss\n' "$H" "${A:-FAIL}" "$((SECONDS-t0))" | tee -a d2_logs/results.tsv
done
