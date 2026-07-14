#!/usr/bin/env bash
# C (cumulative build-up waterfall) + G2 (conf-gate sensitivity) on the RealSense held-out set.
#   buildup_g2.sh <cell> <gpu_uuid>
#   cell ∈  c0_base c1_dark c2_cov c3_rot c4_selfaug c5_rc   (waterfall, each ADDS one lever)
#          | g_gate0 g_gate10 g_gate20                        (conf-gate {0,.10,.20}; .05=deployed)
# RealSense, held-out --frac-range 0.7 1.0 --max-frames 1000. base-only (RC only for c5).
set -uo pipefail
cd "$(dirname "$0")"
CELL="${1:?cell}"; export CUDA_VISIBLE_DEVICES="${2:?gpu uuid}"
NF=1000
DET=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth
CROPDET=../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
STROT=../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth   # stage1 rot for waterfall
CLEANHEAD=../TRAIN/outputs_angle/angle_crop_20260605_174740/best_angle_head.pth   # occ-aug OFF
ST=../TRAIN/outputs_selftrain
SELFHEAD=$ST/realsense_lightstack_20260705_003546/best_selftrain_head.pth
SELFROT=$ST/realsense_lightstack_20260705_003546/best_selftrain_rot.pth
DATA=../Dataset/Converted_dataset/DREAM_real/panda-3cam_realsense
SAM=../weights_sam/sam_vit_b_01ec64.pth
FRAC="--frac-range 0.7 1.0"; RH=448
RES=ablation_logs/buildup; mkdir -p "$RES" rc_dumps_bu

# defaults: full deployed knobs (used by g_* cells)
HEAD=$SELFHEAD; ROTFLAG="--rot-head $SELFROT"; KNOB="--cov-pnp --dark-decode"; GATE="--conf-gate 0.05"; RC=0
case "$CELL" in
  c0_base)     HEAD=$CLEANHEAD; ROTFLAG="";                 KNOB="";                        ;;
  c1_dark)     HEAD=$CLEANHEAD; ROTFLAG="";                 KNOB="--dark-decode";           ;;
  c2_cov)      HEAD=$CLEANHEAD; ROTFLAG="";                 KNOB="--cov-pnp --dark-decode"; ;;
  c3_rot)      HEAD=$CLEANHEAD; ROTFLAG="--rot-head $STROT";KNOB="--cov-pnp --dark-decode"; ;;
  c4_selfaug)  HEAD=$SELFHEAD;  ROTFLAG="--rot-head $SELFROT";KNOB="--cov-pnp --dark-decode";;
  c5_rc)       HEAD=$SELFHEAD;  ROTFLAG="--rot-head $SELFROT";KNOB="--cov-pnp --dark-decode"; RC=1;;
  g_gate0)     GATE="--conf-gate 0";;
  g_gate10)    GATE="--conf-gate 0.10";;
  g_gate20)    GATE="--conf-gate 0.20";;
  *) echo "unknown cell $CELL"; exit 1;;
esac

DUMP=rc_dumps_bu/${CELL}.npz
L=$RES/${CELL}
python selfbbox_eval.py --stage1-detector $DET --stage1-angle $S1ANG --stage1-rot $S1ROT \
  --crop-detector $CROPDET --crop-angle $HEAD $ROTFLAG \
  --bbox-from-solved --bbox-guard $KNOB $GATE $FRAC --max-frames $NF \
  --val-dir $DATA --dump-npz $DUMP > ${L}_base.log 2>&1
B=$(grep -haoE 'ADD-AUC@100mm[: ]+[0-9]+\.[0-9]+' ${L}_base.log | tail -1 | grep -oE '[0-9]+\.[0-9]+$')
F="$B"
if [ "$RC" = "1" ]; then
  python rc_refine_from_dump.py --dump $DUMP --val-dir $DATA --sam-checkpoint $SAM \
    --render-h $RH --max-frames $NF > ${L}_rc.log 2>&1
  F=$(grep -haoE 'render-compare ADD-AUC@100mm[: ]+[0-9]+\.[0-9]+' ${L}_rc.log | tail -1 | grep -oE '[0-9]+\.[0-9]+$')
fi
printf '%s\tbase=[%s]\tfinal=[%s]\n' "$CELL" "$B" "$F" | tee -a $RES/results.tsv
