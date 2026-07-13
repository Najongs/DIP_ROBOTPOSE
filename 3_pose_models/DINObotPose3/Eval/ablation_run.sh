#!/usr/bin/env bash
# Paper ablation campaign — run ONE (ablation, camera) cell on a given GPU.
#   ablation_run.sh <ablation> <camera> <gpu_uuid>
#   ablation ∈ full|no_cov|no_dark|no_rot|no_rc|gt_bbox|no_confgate|no_occaug
#   camera   ∈ realsense|kinect|orb|azure
# Deployed locked config (@1000, lightstack heads, per-camera RC) from FINAL_MODEL.md.
set -uo pipefail
cd "$(dirname "$0")"
ABL="${1:?ablation}"; CAM="${2:?camera}"; export CUDA_VISIBLE_DEVICES="${3:?gpu uuid}"
NF=1000
DET=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth
CROPDET=../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
ST=../TRAIN/outputs_selftrain
DATA=../Dataset/Converted_dataset/DREAM_real
SAM=../weights_sam/sam_vit_b_01ec64.pth
CLEANHEAD=../TRAIN/outputs_angle/angle_crop_20260605_174740/best_angle_head.pth   # occ-aug+self-train OFF (B7)
RESDIR=ablation_logs; mkdir -p "$RESDIR" rc_dumps_abl

case "$CAM" in
  realsense) VAL=$DATA/panda-3cam_realsense; HEAD=$ST/realsense_lightstack_20260705_003546/best_selftrain_head.pth; ROT=$ST/realsense_lightstack_20260705_003546/best_selftrain_rot.pth; RH=448; RC=1; FRAC="--frac-range 0.7 1.0";;
  kinect)    VAL=$DATA/panda-3cam_kinect360; HEAD=$ST/kinect_lightstack_20260705_003552/best_selftrain_head.pth; ROT=$ST/kinect_lightstack_20260705_003552/best_selftrain_rot.pth; RH=448; RC=1; FRAC="--frac-range 0.7 1.0";;
  orb)       VAL=$DATA/panda-orb;            HEAD=$ST/orb_lightstack_20260705_003549/best_selftrain_head.pth;    ROT=$ST/orb_lightstack_20260705_003549/best_selftrain_rot.pth;    RH=512; RC=1; FRAC="--frac-range 0.7 1.0";;
  azure)     VAL=$DATA/panda-3cam_azure;     HEAD=../TRAIN/outputs_angle/angle_occaug_light_20260704_015400/best_angle_head.pth; ROT=../TRAIN/outputs_rotation/rot_crop_occaug_20260704_002102/best_rot_head.pth; RH=448; RC=0; FRAC="";;
  *) echo "unknown camera $CAM"; exit 1;;
esac

BASE="--bbox-from-solved --bbox-guard --cov-pnp --dark-decode"
ROTFLAG="--rot-head $ROT"
RUN_RC=$RC
case "$ABL" in
  full)        ;;
  no_cov)      BASE="--bbox-from-solved --bbox-guard --dark-decode";;
  no_dark)     BASE="--bbox-from-solved --bbox-guard --cov-pnp";;
  no_rot)      ROTFLAG="";;
  no_rc)       RUN_RC=0;;
  gt_bbox)     BASE="--oracle-bbox --cov-pnp --dark-decode";;
  no_confgate) BASE="$BASE --conf-gate 0";;
  no_occaug)   HEAD=$CLEANHEAD;;
  *) echo "unknown ablation $ABL"; exit 1;;
esac

DUMP=rc_dumps_abl/${ABL}_${CAM}.npz
LOG=$RESDIR/${ABL}_${CAM}
echo "[$(date +%H:%M:%S)] START $ABL/$CAM gpu=$CUDA_VISIBLE_DEVICES" >> $RESDIR/run.log
python selfbbox_eval.py --stage1-detector $DET --stage1-angle $S1ANG --stage1-rot $S1ROT \
  --crop-detector $CROPDET --crop-angle $HEAD $ROTFLAG \
  $BASE $FRAC --max-frames $NF --val-dir $VAL --dump-npz $DUMP > ${LOG}_base.log 2>&1
B=$(grep -haoE 'ADD-AUC@100mm[: ]+[0-9]+\.[0-9]+' ${LOG}_base.log | tail -1 | grep -oE '[0-9]+\.[0-9]+$')
F="$B"
if [ "$RUN_RC" = "1" ]; then
  python rc_refine_from_dump.py --dump $DUMP --val-dir $VAL --sam-checkpoint $SAM --render-h $RH --max-frames $NF > ${LOG}_rc.log 2>&1
  F=$(grep -haoE 'render-compare ADD-AUC@100mm[: ]+[0-9]+\.[0-9]+' ${LOG}_rc.log | tail -1 | grep -oE '[0-9]+\.[0-9]+$')
fi
printf '%s\t%s\tbase=[%s]\tfinal=[%s]\n' "$ABL" "$CAM" "$B" "$F" | tee -a $RESDIR/results.tsv
echo "[$(date +%H:%M:%S)] DONE $ABL/$CAM  final=$F" >> $RESDIR/run.log
