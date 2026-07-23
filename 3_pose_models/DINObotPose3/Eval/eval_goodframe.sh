#!/usr/bin/env bash
# Decision harness for the 2악장 angle experiments: run the deployed synth pipeline with a given
# crop-angle head and report ADD-AUC split into CLEAN (all-keypoints-in-frame) vs OFF-frame frames.
# The decision metric is CLEAN(=good)-frame ADD-AUC — baseline (deployed mlp) = 0.7884 on DR.
#   eval_goodframe.sh <crop_angle_ckpt> <head_type> <angle_backbone> <gpu_uuid> <tag> [ief_iters] [val_dir]
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
CANG="${1:?crop-angle ckpt}"; HT="${2:?head_type}"; AB="${3:?angle_backbone dino_frozen|resnet50}"
export CUDA_VISIBLE_DEVICES="${4:?gpu}"; TAG="${5:?tag}"; export IEF_ITERS="${6:-4}"
VAL="${7:-../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr}"
DET=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
S1ANG=../TRAIN/outputs_angle/angle_20260603_013948/best_angle_head.pth
S1ROT=../TRAIN/outputs_rotation/rot_20260604_162336/best_rot_head.pth
# crop detector: 기본값은 배포 체크포인트. 신규 detector 를 재는 경우에만 env 로 덮어쓴다
# (예: CROPDET=../TRAIN/outputs_heatmap/cropasp_a43/best_heatmap.pth ./eval_goodframe.sh ...).
CROPDET="${CROPDET:-../TRAIN/outputs_heatmap/crop_20260605_010622/best_heatmap.pth}"
ROT=../TRAIN/outputs_rotation/rot_crop_20260606_022535/best_rot_head.pth
RES=ablation_logs/goodframe; mkdir -p "$RES" rc_dumps_gf
DUMP=rc_dumps_gf/${TAG}.npz
AB_ARG=""; [ "$AB" = resnet50 ] && AB_ARG="--angle-backbone resnet50"
python selfbbox_eval.py --stage1-detector $DET --stage1-angle $S1ANG --stage1-rot $S1ROT \
  --crop-detector $CROPDET --crop-angle "$CANG" --crop-head-type "$HT" $AB_ARG --rot-head $ROT \
  --bbox-from-solved --bbox-guard --cov-pnp --dark-decode \
  --max-frames 1000 --val-dir "$VAL" --dump-npz $DUMP > $RES/${TAG}.log 2>&1
python - "$DUMP" "$TAG" "$VAL" <<'PY'
import numpy as np, json, os, sys
dump, tag, val = sys.argv[1], sys.argv[2], sys.argv[3]
d=np.load(dump,allow_pickle=True); fid,kp,gt=d['fid'],d['kp_cam'],d['gt3d']
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
off=np.array([anyoff(f) for f in fid]); add=np.linalg.norm(kp-gt,axis=2).mean(1)
def auc(a):
    ts=np.arange(0,0.1,1e-5); return float((a[None,:]<=ts[:,None]).mean(1).mean())
print(f"[{tag}] ALL={auc(add):.4f}  CLEAN(good,n={(~off).sum()})={auc(add[~off]):.4f}  OFF(n={off.sum()})={auc(add[off]):.4f}  fail={ (add>0.1).mean()*100:.1f}%")
print(f"       baseline(deployed mlp): ALL 0.704 / CLEAN 0.7884 / OFF 0.353  <- CLEAN is the decision metric")
PY
echo "GOODFRAME $TAG DONE" >> /tmp/claude-1002/-home-najo-NAS-DIP/5aafbd5b-1895-41b2-90ed-8d6e9438b7dd/scratchpad/gf_done.txt
