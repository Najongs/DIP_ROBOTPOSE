#!/usr/bin/env bash
# DECISION GATE (2026-07-22): does a RoboPEPP-style FREEZE-theta + solve-6DOF path have life for us?
# Three angle-MAE points on the SAME 1000-frame Panda synth-DR pipeline as rc_dumps_gf:
#   A freeze @ mlpctrl head (~9.1 deg)   -> reproduce naive-freeze ~0.533 (full-set)
#   B freeze @ p1b head    (7.47 deg)    -> the NEW datapoint
#   C oracle GT angle (0 deg) + freeze   -> upper anchor (~0.899 good-frame / 0.861 full-set)
# Inference only. No train, no commit.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export CUDA_VISIBLE_DEVICES="${1:?gpu uuid}"
DET=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth
CROPDET=../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
ROT=../TRAIN/outputs_rotation/rot_crop_20260606_022535/best_rot_head.pth
VAL=../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr
MLP=../TRAIN/outputs_angle/mlpctrl_dr/best_angle_head.pth
P1B=../TRAIN/outputs_angle/p1b_resnet50/best_angle_head.pth
COMMON="--stage1-detector $DET --stage1-angle $S1ANG --stage1-rot $S1ROT --crop-detector $CROPDET --rot-head $ROT --bbox-from-solved --bbox-guard --cov-pnp --dark-decode --max-frames 1000 --val-dir $VAL"
PY=/home/najo/.conda/envs/dino/bin/python
mkdir -p rc_dumps_gf ablation_logs/freeze_gate
L=ablation_logs/freeze_gate

echo "=== A: freeze @ mlpctrl (mlp/dino, ~9.1deg) ==="
$PY selfbbox_eval.py $COMMON --crop-angle $MLP --crop-head-type mlp --freeze-head-theta \
  --dump-npz rc_dumps_gf/freeze_mlp.npz > $L/freeze_mlp.log 2>&1
echo "A done exit=$?"

echo "=== B: freeze @ p1b (ief/resnet50, 7.47deg) ==="
IEF_ITERS=4 $PY selfbbox_eval.py $COMMON --crop-angle $P1B --crop-head-type ief --angle-backbone resnet50 --freeze-head-theta \
  --dump-npz rc_dumps_gf/freeze_p1b.npz > $L/freeze_p1b.log 2>&1
echo "B done exit=$?"

echo "=== C: oracle GT angle (0deg) + freeze ==="
$PY selfbbox_eval.py $COMMON --crop-angle $MLP --crop-head-type mlp --oracle-angle \
  --dump-npz rc_dumps_gf/freeze_oracle.npz > $L/freeze_oracle.log 2>&1
echo "C done exit=$?"

echo "===== FREEZE-GATE RESULTS (full-set + good-frame CLEAN, JSON 640x480 off-frame) ====="
$PY - "$VAL" <<'PY'
import numpy as np, json, os, sys
val=sys.argv[1]; W,H=640,480; TRACK=['link0','link2','link3','link4','link6','link7','hand']
def anyoff(f):
    p=os.path.join(val,str(f)+'.json')
    if not os.path.exists(p): return False
    o=json.load(open(p))['objects'][0]
    kps={k['name'].replace('panda_',''):k.get('projected_location') for k in o.get('keypoints',[]) if 'name' in k}
    for t in TRACK:
        v=kps.get(t)
        if v and (v[0]<0 or v[0]>=W or v[1]<0 or v[1]>=H): return True
    return False
def auc(a): return float(np.clip(1-10*a,0,1).mean())
for tag,f in [('freeze_mlp(~9.1)','rc_dumps_gf/freeze_mlp.npz'),
              ('freeze_p1b(7.47)','rc_dumps_gf/freeze_p1b.npz'),
              ('oracle(0.0)','rc_dumps_gf/freeze_oracle.npz')]:
    d=np.load(f,allow_pickle=True); fid,kp,gt=d['fid'],d['kp_cam'],d['gt3d']
    add=np.linalg.norm(kp-gt,axis=2).mean(1)
    off=np.array([anyoff(x) for x in fid])
    print(f"[{tag:18s}] FULL={auc(add):.4f}  CLEAN(good,n={(~off).sum()})={auc(add[~off]):.4f}  median={np.median(add[~off])*1000:.1f}mm  fail>100mm={(add>0.1).mean()*100:.1f}%")
print("REF joint-solve: mlpctrl FULL 0.7128 / CLEAN 0.7706 ; p1b FULL 0.7113 / CLEAN 0.7672 ; deployed CLEAN 0.7884 ; oracle joint CLEAN 0.899")
PY
echo "FREEZE_GATE DONE"
