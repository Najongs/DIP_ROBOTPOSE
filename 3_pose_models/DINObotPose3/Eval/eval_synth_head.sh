#!/usr/bin/env bash
# Re-evaluate the synthetic Panda pipeline with a NEW crop-angle head and report whether the
# ~10% catastrophic tail shrank. Base solve only (no RC) — the tail is a base-solve failure.
#   eval_synth_head.sh <crop_angle_ckpt> <gpu_uuid> <tag>
# Baseline (deployed mlp head): DR base ADD-AUC 0.704, fail(>100mm) 10.7%.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
HEAD="${1:?crop-angle ckpt}"; export CUDA_VISIBLE_DEVICES="${2:?gpu}"; TAG="${3:?tag}"; HTYPE="${4:-mlp}"
DET=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth
CROPDET=../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
ROT=../TRAIN/outputs_rotation/rot_crop_20260606_022535/best_rot_head.pth
VAL=../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr
RES=ablation_logs/synth_head; mkdir -p "$RES" rc_dumps_head
DUMP=rc_dumps_head/${TAG}.npz
python selfbbox_eval.py --stage1-detector $DET --stage1-angle $S1ANG --stage1-rot $S1ROT \
  --crop-detector $CROPDET --crop-angle "$HEAD" --crop-head-type "$HTYPE" --rot-head $ROT \
  --bbox-from-solved --bbox-guard --cov-pnp --dark-decode \
  --max-frames 1000 --val-dir $VAL --dump-npz $DUMP > $RES/${TAG}.log 2>&1
A=$(grep -haoE 'ADD-AUC@100mm[: ]+[0-9]+\.[0-9]+' $RES/${TAG}.log | tail -1 | grep -oE '[0-9]+\.[0-9]+$')
# fail-rate + wrist analysis from the dump
python - "$DUMP" "$TAG" "$A" <<'PY'
import numpy as np, sys
d=np.load(sys.argv[1],allow_pickle=True); tag=sys.argv[2]; auc=sys.argv[3]
kp,gt=d['kp_cam'],d['gt3d']; add=np.linalg.norm(kp-gt,axis=2).mean(1); fail=add>0.1
print(f"[{tag}] ADD-AUC={auc}  median={np.median(add)*1000:.0f}mm  FAIL(>100mm)={fail.mean()*100:.1f}%  (baseline 0.704 / 10.7%)")
PY
echo "SYNTH_HEAD $TAG DONE" > /tmp/claude-1002/-home-najo-NAS-DIP/5aafbd5b-1895-41b2-90ed-8d6e9438b7dd/scratchpad/evalhead_${TAG}_done.txt
