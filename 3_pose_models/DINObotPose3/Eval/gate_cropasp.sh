#!/usr/bin/env bash
# crop-aspect detector 재학습 게이트. 학습이 끝난 뒤 한 번 실행하면 세 지표를 전부 찍는다.
#   ./gate_cropasp.sh <detector_ckpt> <tag> [gpu_uuid]
# 예: ./gate_cropasp.sh ../TRAIN/outputs_heatmap/cropasp_a43/best_heatmap.pth cropasp_a43
#
# 기준선 (배포 crop detector crop_20260605_010622):
#   1) 검출기 2D px (mediocre 밴드, 원본 640x480, ORACLE bbox, chain A_deploy)
#        clean.med 1.78 / med.med 3.35 / med.p90 16.69      <- crop_chain_probe.py 실측
#        (배포 solved-bbox 기준 참고치: med.med 3.52 / med.p90 17.07)
#      성공 조건: chain A_deploy 가 기준선의 chain B_train(clean 1.14 / med 2.58) 쪽으로 이동.
#   2) clean(good) 프레임 ADD-AUC on synth DR : 0.7884
#      성공 조건: > 0.7884. 참고로 oracle-angle 천장은 0.8991.
#   3) mediocre 밴드 median ADD : 42.1mm  (밴드는 기준선 ADD 30~100mm 로 '고정'된 프레임 집합 —
#      새 모델로 밴드를 다시 정의하면 안 된다. 아래 파이썬이 mediocre_band.npz 의 fid 로 고정한다.)
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

DET="${1:?detector ckpt}"; TAG="${2:?tag}"
GPU="${3:-GPU-7ff6997b-14c1-9283-5119-251c9c899b8e}"
CANG=../TRAIN/outputs_angle/angle_cropjit_20260606_020835/best_angle_head.pth

echo "########## 1) 검출기 2D px (crop_chain_probe, oracle bbox) ##########"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 \
  /home/najo/.conda/envs/dino/bin/python crop_chain_probe.py \
    --detector "$DET" --out ablation_logs/crop_chain_${TAG}.npz 2>&1 \
  | grep -av "it/s\|it\]\|Warning\|warn\|Loading weights"

echo
echo "########## 2) clean(good) 프레임 ADD-AUC (배포 파이프라인, 새 detector) ##########"
CROPDET="$DET" ./eval_goodframe.sh "$CANG" mlp dino_frozen "$GPU" "$TAG"

echo
echo "########## 3) mediocre 밴드(기준선 고정) median ADD ##########"
/home/najo/.conda/envs/dino/bin/python - "rc_dumps_gf/${TAG}.npz" <<'PY'
import numpy as np, sys, os
dump = sys.argv[1]
if not os.path.exists(dump):
    print(f'  dump 없음: {dump} (2번 단계 실패?)'); sys.exit(1)
V = np.load(dump, allow_pickle=True)
D = np.load('ablation_logs/mediocre_band.npz', allow_pickle=True)
base_add = D['add'] * 1000
clean = D['n_offframe'] == 0
med = clean & (base_add >= 30) & (base_add < 100)     # 기준선으로 고정된 밴드
exc = clean & (base_add < 30)
pos = {str(f): i for i, f in enumerate([str(x) for x in V['fid']])}
order = np.array([pos[str(f)] for f in D['fid'] if str(f) in pos])
keep = np.array([str(f) in pos for f in D['fid']])
add = np.full(len(D['fid']), np.nan)
add[keep] = np.linalg.norm(V['kp_cam'][order] - V['gt3d'][order], axis=2).mean(1) * 1000
AUC = lambda a: float(np.clip(1 - 10 * a / 1000, 0, 1).mean())
for nm, m in [('excellent', exc & keep), ('mediocre', med & keep), ('clean', clean & keep)]:
    print(f'  {nm:10s} n={m.sum():4d}  median ADD={np.nanmedian(add[m]):6.1f}mm  AUC={AUC(add[m]):.4f}')
print(f'  기준선: excellent 앵커 / mediocre median 42.1mm / clean AUC 0.7884')
PY
