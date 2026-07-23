#!/usr/bin/env bash
# TRUE freeze-6DOF curve: freeze theta at GT + calibrated Gaussian noise (realized wrapped-abs MAE =
# target), solve only R,t. Answers convex-vs-linear: does freeze@3.8deg reach ~0.83 (closeable) or
# ~0.73 (concede)? Same 1000f Panda synth-DR pipeline as rc_dumps_gf. Inference only.
#   freeze_curve.sh <gpu_uuid> [seed]
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export CUDA_VISIBLE_DEVICES="${1:?gpu uuid}"; SEED="${2:-0}"
DET=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth
CROPDET=../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth
ROT=../TRAIN/outputs_rotation/rot_crop_20260606_022535/best_rot_head.pth
VAL=../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr
MLP=../TRAIN/outputs_angle/mlpctrl_dr/best_angle_head.pth
COMMON="--stage1-detector $DET --stage1-angle $S1ANG --stage1-rot $S1ROT --crop-detector $CROPDET --crop-angle $MLP --crop-head-type mlp --rot-head $ROT --bbox-from-solved --bbox-guard --cov-pnp --dark-decode --max-frames 1000 --val-dir $VAL --oracle-angle"
PY=/home/najo/.conda/envs/dino/bin/python
mkdir -p rc_dumps_gf ablation_logs/freeze_curve
L=ablation_logs/freeze_curve

for T in 0.0 1.5 2.4 3.0 3.8 5.0 6.0 7.47; do
  TAG="mae${T}_s${SEED}"
  echo "=== target MAE ${T} deg (seed ${SEED}) ==="
  $PY selfbbox_eval.py $COMMON --oracle-angle-noise-mae $T --oracle-noise-seed $SEED \
    --dump-npz rc_dumps_gf/fc_${TAG}.npz > $L/${TAG}.log 2>&1
  echo "  ${T} done exit=$?"
done

echo "===== FREEZE CURVE (realized MAE + good-frame CLEAN ADD-AUC) ====="
$PY - "$VAL" "$SEED" <<'PY'
import numpy as np, json, os, sys, glob, re
val, seed = sys.argv[1], sys.argv[2]
W,H=640,480; TRACK=['link0','link2','link3','link4','link6','link7','hand']
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
def realized_mae(tag):
    # parse "raw MLP" per-joint column from the eval log = |injected theta - gt| = realized MAE
    lg=f"ablation_logs/freeze_curve/{tag}.log"
    vals=[]
    for ln in open(lg,errors='ignore'):
        m=re.match(r'\s*J(\d)\s+([0-9.]+)\s+',ln)
        if m: vals.append(float(m.group(2)))
    return np.mean(vals) if vals else float('nan')
print(f"{'targetMAE':>9} {'realMAE':>8} {'CLEAN(good)':>12} {'FULL':>7} {'medianmm':>9}")
rows=[]
for T in ['0.0','1.5','2.4','3.0','3.8','5.0','6.0','7.47']:
    tag=f"mae{T}_s{seed}"; f=f"rc_dumps_gf/fc_{tag}.npz"
    if not os.path.exists(f): print(f"{T:>9} MISSING"); continue
    d=np.load(f,allow_pickle=True); fid,kp,gt3=d['fid'],d['kp_cam'],d['gt3d']
    add=np.linalg.norm(kp-gt3,axis=2).mean(1); off=np.array([anyoff(x) for x in fid])
    rm=realized_mae(tag); c=auc(add[~off])
    rows.append((float(T),rm,c))
    print(f"{T:>9} {rm:>8.2f} {c:>12.4f} {auc(add):>7.4f} {np.median(add[~off])*1000:>9.1f}")
print("\nANCHORS: 0deg oracle-freeze=0.887 (prior) ; 7.47deg p1b-freeze=0.584 (prior)")
print("TARGETS: joint-solve deployed=0.7884 ; RoboPEPP(full)=0.830")
# crossover with 0.788 by linear interp between bracketing points
xs=[r[1] for r in rows]; ys=[r[2] for r in rows]
for THRESH,name in [(0.7884,'joint-solve 0.788'),(0.830,'RoboPEPP 0.830')]:
    cx=None
    for i in range(len(rows)-1):
        y0,y1=ys[i],ys[i+1]
        if (y0-THRESH)*(y1-THRESH)<=0 and y0!=y1:
            x0,x1=xs[i],xs[i+1]; cx=x0+(THRESH-y0)*(x1-x0)/(y1-y0); break
    print(f"  freeze-6DOF crosses {name}: {'MAE '+format(cx,'.2f')+'deg' if cx is not None else 'NOT within measured range'}")
# convexity check: compare measured 3.8 to linear(0->7.47) prediction
def interp(x):
    return np.interp(x, xs, ys)
print(f"  linear(0->7.47) would predict 3.8deg={np.interp(3.8,[xs[0],xs[-1]],[ys[0],ys[-1]]):.3f} ; MEASURED 3.8deg={interp(3.8):.3f}  -> {'CONVEX (closeable-leaning)' if interp(3.8)>np.interp(3.8,[xs[0],xs[-1]],[ys[0],ys[-1]])+0.02 else 'LINEAR/CONCAVE (concede-leaning)'}")
PY
echo "FREEZE_CURVE DONE seed=$SEED"
