#!/usr/bin/env bash
# Panda realsense held-out-30% vs full-sequence, base solver (pre-RC). Corrected dataset path.
set -u
cd "$(dirname "$0")/../.."          # -> Eval/
PY=/home/najo/.conda/envs/dino/bin/python
T=../TRAIN
S1DET=$T/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=$T/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=$T/outputs_rotation/rot_20260604_162336/best_rot_head.pth
CDET=$T/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
RSANG=$T/outputs_selftrain/realsense_lightstack_20260705_003546/best_selftrain_head.pth
RSROT=$T/outputs_selftrain/realsense_lightstack_20260705_003546/best_selftrain_rot.pth
LTANG=$T/outputs_angle/angle_occaug_light_20260704_015400/best_angle_head.pth
LTROT=$T/outputs_rotation/rot_crop_occaug_20260704_002102/best_rot_head.pth
RS=../Dataset/Converted_dataset/DREAM_real/panda-3cam_realsense
LOG=_debate_tmp/kuka_gate/audit_logs
run_panda() { local ang=$1 rot=$2 frac=$3 tag=$4
  $PY selfbbox_eval.py --stage1-detector $S1DET --stage1-angle $S1ANG --stage1-rot $S1ROT \
    --crop-detector $CDET --crop-angle $ang --rot-head $rot \
    --bbox-from-solved --bbox-guard --cov-pnp --dark-decode \
    $frac --max-frames 1000 --batch-size 16 --val-dir $RS 2>&1 | grep -av "it/s\]\|it\]" > $LOG/$tag.log; echo "DONE $tag"; }
run_panda $RSANG $RSROT "--frac-range 0.7 1.0" panda_rs_deployed_heldout30
run_panda $RSANG $RSROT "" panda_rs_deployed_full
run_panda $LTANG $LTROT "--frac-range 0.7 1.0" panda_rs_light_heldout30
run_panda $LTANG $LTROT "" panda_rs_light_full
echo "PANDA RERUN COMPLETE"
