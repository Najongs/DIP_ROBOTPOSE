#!/usr/bin/env bash
# Fairness audit sweep. Lanes A/B/C/D each pinned to one GPU UUID (set by caller via
# CUDA_VISIBLE_DEVICES). Jobs run sequentially within a lane.
set -u
cd "$(dirname "$0")/../.."          # -> Eval/
PY=/home/najo/.conda/envs/dino/bin/python
T=../TRAIN
KDET=$T/outputs_heatmap/kuka_dream_detector_20260709_183119/best_heatmap.pth
KANG=$T/outputs_angle/kuka_angle_20260712_060212/best_angle_head.pth
KROT=$T/outputs_rotation/kuka_rot_20260712_060214/best_rot_head.pth
BDET=$T/outputs_heatmap/baxter_left_dream_detector_20260710_152926/best_heatmap.pth
BANG=$T/outputs_angle/baxter_angle_20260713_074831/best_angle_head.pth
BROT=$T/outputs_rotation/baxter_rot_20260713_074833/best_rot_head.pth
KDATA=../../../datasets/synthetic/kuka_synth_test_dr
BDATA=../../../datasets/synthetic/baxter_synth_test_dr
# panda deployed realsense
S1DET=$T/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=$T/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=$T/outputs_rotation/rot_20260604_162336/best_rot_head.pth
CDET=$T/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
RSANG=$T/outputs_selftrain/realsense_lightstack_20260705_003546/best_selftrain_head.pth
RSROT=$T/outputs_selftrain/realsense_lightstack_20260705_003546/best_selftrain_rot.pth
LTANG=$T/outputs_angle/angle_occaug_light_20260704_015400/best_angle_head.pth
LTROT=$T/outputs_rotation/rot_crop_occaug_20260704_002102/best_rot_head.pth
RS=Dataset/Converted_dataset/DREAM_real/panda-3cam_realsense
W=_debate_tmp/kuka_gate
LOG=$W/audit_logs; mkdir -p $LOG
NF=6000; BS=32

run_kuka() { local seed=$1 rot=$2 tag=$3; local extra=""; [ "$rot" = 1 ] && extra="--rot-head $KROT"
  $PY $W/seed_run.py $seed kuka_add_eval.py --detector $KDET --angle-head $KANG $extra \
    --val-dir $KDATA --max-frames $NF --batch-size $BS 2>&1 | grep -av "it/s\]\|it\]" > $LOG/$tag.log; echo "DONE $tag"; }
run_baxter() { local seed=$1 rot=$2 tag=$3; local extra=""; [ "$rot" = 1 ] && extra="--rot-head $BROT"
  $PY $W/seed_run.py $seed baxter_add_eval.py --detector $BDET --angle-head $BANG $extra \
    --val-dir $BDATA --max-frames $NF --batch-size $BS 2>&1 | grep -av "it/s\]\|it\]" > $LOG/$tag.log; echo "DONE $tag"; }
# panda selfbbox base (pre-RC). args: angle rot frac tag
run_panda() { local ang=$1 rot=$2 frac=$3 tag=$4
  $PY selfbbox_eval.py --stage1-detector $S1DET --stage1-angle $S1ANG --stage1-rot $S1ROT \
    --crop-detector $CDET --crop-angle $ang --rot-head $rot \
    --bbox-from-solved --bbox-guard --cov-pnp --dark-decode \
    $frac --max-frames 1000 --batch-size 16 --val-dir $RS 2>&1 | grep -av "it/s\]\|it\]" > $LOG/$tag.log; echo "DONE $tag"; }

case "$1" in
  A) run_kuka 0 0 k_norot_s0; run_kuka 1 0 k_norot_s1; run_kuka 2 0 k_norot_s2 ;;
  B) run_kuka 3 0 k_norot_s3; run_kuka 4 0 k_norot_s4; run_kuka 0 1 k_rot_s0 ;;
  C) run_kuka 42 1 k_rot_s42; run_baxter 0 1 b_rot_s0; run_baxter 42 1 b_rot_s42 ;;
  D) run_baxter 0 0 b_norot_s0
     run_panda $RSANG $RSROT "--frac-range 0.7 1.0" panda_rs_deployed_heldout30
     run_panda $RSANG $RSROT "" panda_rs_deployed_full
     run_panda $LTANG $LTROT "--frac-range 0.7 1.0" panda_rs_light_heldout30
     run_panda $LTANG $LTROT "" panda_rs_light_full ;;
esac
echo "LANE $1 COMPLETE"
