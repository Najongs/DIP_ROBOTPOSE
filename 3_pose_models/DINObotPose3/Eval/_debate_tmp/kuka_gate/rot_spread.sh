#!/usr/bin/env bash
# Rot-head (paper-cell) run-to-run spread. The rot path passes R_init AND t_init, so pnp_init
# RANSAC output is discarded -> any variation across these runs is CUDA nondeterminism, not seed.
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
LOG=_debate_tmp/kuka_gate/audit_logs; NF=6000; BS=32
rk(){ $PY _debate_tmp/kuka_gate/seed_run.py $1 kuka_add_eval.py --detector $KDET --angle-head $KANG --rot-head $KROT --val-dir $KDATA --max-frames $NF --batch-size $BS 2>&1 | grep -av "it/s\]\|it\]" > $LOG/$2.log; echo "DONE $2"; }
rb(){ $PY _debate_tmp/kuka_gate/seed_run.py $1 baxter_add_eval.py --detector $BDET --angle-head $BANG --rot-head $BROT --val-dir $BDATA --max-frames $NF --batch-size $BS 2>&1 | grep -av "it/s\]\|it\]" > $LOG/$2.log; echo "DONE $2"; }
case "$1" in
  k) rk 0 k_rot_r0; rk 7 k_rot_r7 ;;
  b) rb 42 b_rot_r42; rb 7 b_rot_r7 ;;
esac
echo "ROT-SPREAD $1 COMPLETE"
