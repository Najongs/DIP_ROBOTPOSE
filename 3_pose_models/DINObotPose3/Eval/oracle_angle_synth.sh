#!/usr/bin/env bash
# Synthetic known-joint ceiling: same clean synth-native pipeline as the deployed synth number
# (occlusion_bench r0), run BOTH predicted-angle and --oracle-angle (GT theta, solve only R,t),
# base + RC, on panda_synth_test_{dr,photo}. Diagnoses whether the synth gap is angle or depth,
# and fills the '--' DR/Photo cells of Table 1's oracle-angle (italic) row.
#   oracle_angle_synth.sh <dr|photo> <gpu_uuid>
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
SPLIT="${1:?dr|photo}"; export CUDA_VISIBLE_DEVICES="${2:?gpu uuid}"
NF=1000
DET=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth
CROPDET=../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
CROPANG=../TRAIN/outputs_angle/angle_crop_20260605_174740/best_angle_head.pth
ROT=../TRAIN/outputs_rotation/rot_crop_20260606_022535/best_rot_head.pth
SAM=../weights_sam/sam_vit_b_01ec64.pth
VAL=../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_${SPLIT}
RES=ablation_logs/oracle_angle_synth; mkdir -p "$RES" rc_dumps_oas
SCRATCH=/tmp/claude-1002/-home-najo-NAS-DIP/5aafbd5b-1895-41b2-90ed-8d6e9438b7dd/scratchpad

run_one () {  # $1=mode(pred|oracle)  $2=extra-flag
  local mode="$1" extra="$2" L="$RES/${SPLIT}_${1}" DUMP="rc_dumps_oas/${SPLIT}_${1}.npz"
  python selfbbox_eval.py --stage1-detector $DET --stage1-angle $S1ANG --stage1-rot $S1ROT \
    --crop-detector $CROPDET --crop-angle $CROPANG --rot-head $ROT \
    --bbox-from-solved --bbox-guard --cov-pnp --dark-decode $extra \
    --max-frames $NF --val-dir $VAL --dump-npz $DUMP > ${L}_base.log 2>&1
  local B=$(grep -haoE 'ADD-AUC@100mm[: ]+[0-9]+\.[0-9]+' ${L}_base.log | tail -1 | grep -oE '[0-9]+\.[0-9]+$')
  python rc_refine_from_dump.py --dump $DUMP --val-dir $VAL --sam-checkpoint $SAM \
    --render-h 448 --batch-size 4 --max-frames $NF > ${L}_rc.log 2>&1
  local F=$(grep -haoE 'render-compare ADD-AUC@100mm[ :]+[0-9]+\.[0-9]+' ${L}_rc.log | tail -1 | grep -oE '[0-9]+\.[0-9]+$')
  printf '%s\t%s\tbase=[%s]\tRC=[%s]\n' "$SPLIT" "$mode" "${B:-FAIL}" "${F:-FAIL}" | tee -a $RES/results.tsv
}

run_one pred   ""
run_one oracle "--oracle-angle"
echo "ORACLE_ANGLE_SYNTH ${SPLIT} DONE" > "$SCRATCH/oas_${SPLIT}_done.txt"
