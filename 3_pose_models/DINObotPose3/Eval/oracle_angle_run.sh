#!/usr/bin/env bash
# Known-joint ceiling: deployed pipeline with GT joint angles (theta frozen, solve only R,t).
#   oracle_angle_run.sh <camera> <gpu_uuid>
# Same deployed per-camera config as ablation_run.sh 'full' + --oracle-angle. base + RC.
set -uo pipefail
cd "$(dirname "$0")"
CAM="${1:?camera}"; export CUDA_VISIBLE_DEVICES="${2:?gpu uuid}"
NF=1000
DET=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth
CROPDET=../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
ST=../TRAIN/outputs_selftrain
DATA=../Dataset/Converted_dataset/DREAM_real
SAM=../weights_sam/sam_vit_b_01ec64.pth
RES=ablation_logs/oracle_angle; mkdir -p "$RES" rc_dumps_oa

case "$CAM" in
  realsense) VAL=$DATA/panda-3cam_realsense; HEAD=$ST/realsense_lightstack_20260705_003546/best_selftrain_head.pth; ROT=$ST/realsense_lightstack_20260705_003546/best_selftrain_rot.pth; RH=448; RC=1; FRAC="--frac-range 0.7 1.0";;
  kinect)    VAL=$DATA/panda-3cam_kinect360; HEAD=$ST/kinect_lightstack_20260705_003552/best_selftrain_head.pth; ROT=$ST/kinect_lightstack_20260705_003552/best_selftrain_rot.pth; RH=448; RC=1; FRAC="--frac-range 0.7 1.0";;
  orb)       VAL=$DATA/panda-orb;            HEAD=$ST/orb_lightstack_20260705_003549/best_selftrain_head.pth;    ROT=$ST/orb_lightstack_20260705_003549/best_selftrain_rot.pth;    RH=512; RC=1; FRAC="--frac-range 0.7 1.0";;
  azure)     VAL=$DATA/panda-3cam_azure;     HEAD=../TRAIN/outputs_angle/angle_occaug_light_20260704_015400/best_angle_head.pth; ROT=../TRAIN/outputs_rotation/rot_crop_occaug_20260704_002102/best_rot_head.pth; RH=448; RC=0; FRAC="";;
  *) echo "unknown camera $CAM"; exit 1;;
esac

DUMP=rc_dumps_oa/${CAM}.npz; L=$RES/${CAM}
python selfbbox_eval.py --stage1-detector $DET --stage1-angle $S1ANG --stage1-rot $S1ROT \
  --crop-detector $CROPDET --crop-angle $HEAD --rot-head $ROT \
  --bbox-from-solved --bbox-guard --cov-pnp --dark-decode --oracle-angle \
  $FRAC --max-frames $NF --val-dir $VAL --dump-npz $DUMP > ${L}_base.log 2>&1
B=$(grep -haoE 'ADD-AUC@100mm[: ]+[0-9]+\.[0-9]+' ${L}_base.log | tail -1 | grep -oE '[0-9]+\.[0-9]+$')
F="$B"
if [ "$RC" = "1" ]; then
  python rc_refine_from_dump.py --dump $DUMP --val-dir $VAL --sam-checkpoint $SAM --render-h $RH --max-frames $NF > ${L}_rc.log 2>&1
  F=$(grep -haoE 'render-compare ADD-AUC@100mm[: ]+[0-9]+\.[0-9]+' ${L}_rc.log | tail -1 | grep -oE '[0-9]+\.[0-9]+$')
fi
printf 'oracle-angle\t%s\tbase=[%s]\tfinal=[%s]\n' "$CAM" "$B" "$F" | tee -a $RES/results.tsv
