#!/usr/bin/env bash
# G1 keypoint-localization-noise sensitivity (DREAM-style PnP robustness). Inject Gaussian 2D noise
# (px std) into the decoded keypoints before the solver and sweep sigma; run with cov-PnP ON vs OFF
# to test whether anisotropic whitening absorbs added noise. RealSense held-out 1000, base-only
# (no RC, isolates solver/PnP). Two variants run in parallel on two GPUs.
#   g1_kpjitter.sh <cov|nocov> <gpu_uuid>
set -uo pipefail
cd "$(dirname "$0")"
VARIANT="${1:?cov|nocov}"; export CUDA_VISIBLE_DEVICES="${2:?gpu uuid}"
NF=1000
DET=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth
CROPDET=../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
SELFHEAD=../TRAIN/outputs_selftrain/realsense_lightstack_20260705_003546/best_selftrain_head.pth
SELFROT=../TRAIN/outputs_selftrain/realsense_lightstack_20260705_003546/best_selftrain_rot.pth
DATA=../Dataset/Converted_dataset/DREAM_real/panda-3cam_realsense
RES=ablation_logs/g1_kpjitter; mkdir -p "$RES"
COV="--cov-pnp"; [ "$VARIANT" = "nocov" ] && COV=""
OUT="$RES/results_${VARIANT}.tsv"; : > "$OUT"
for S in 0 1 2 4 8; do
  L="$RES/${VARIANT}_j${S}.log"
  python selfbbox_eval.py --stage1-detector $DET --stage1-angle $S1ANG --stage1-rot $S1ROT \
    --crop-detector $CROPDET --crop-angle $SELFHEAD --rot-head $SELFROT \
    --bbox-from-solved --bbox-guard $COV --dark-decode --conf-gate 0.05 \
    --frac-range 0.7 1.0 --max-frames $NF --val-dir $DATA --kp-jitter $S > "$L" 2>&1
  A=$(grep -a "ADD-AUC@100mm" "$L" | tail -1 | sed -E 's/.*ADD-AUC@100mm: ([0-9.]+).*/\1/')
  echo -e "jitter\t${S}\t${VARIANT}\tADD-AUC=${A}" | tee -a "$OUT"
done
echo "G1 ${VARIANT} DONE" > /tmp/claude-1002/-home-najo-NAS-DIP/5aafbd5b-1895-41b2-90ed-8d6e9438b7dd/scratchpad/g1_${VARIANT}_done.txt
