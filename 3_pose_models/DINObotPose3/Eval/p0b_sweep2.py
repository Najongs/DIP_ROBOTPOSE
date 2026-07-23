#!/usr/bin/env python
"""P0b 2단계: (a) 강한 anchor 범위, (b) 프레임 조건부 freeze/anchor (GT-free 게이트).

sweep1 결과: 모든 부분 freeze 가 손해(distal J5,J6 만 얼려도 clean-AUC 0.776->0.682),
anchor_w<=1 은 무영향. ALL-freeze 미니=0.5193 vs 배포 0.533 -> 미니솔버가 freeze 거동을 재현.
여기서는 "나쁜 2D 프레임에만" 걸면 이득이 나는지를 GT-free 게이트로 판정한다.
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'TRAIN'))
from model_v4 import panda_forward_kinematics  # noqa: E402
from mediocre_band_probe2 import R_to_d6, kabsch_Rt, KP, GT_DIR  # noqa: E402
from p0b_sweep import solve2, AUC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default=os.path.join(HERE, 'rc_dumps_oas', 'dr_pred_v3.npz'))
    ap.add_argument('--iters', type=int, default=400)
    args = ap.parse_args()

    V = np.load(args.dump, allow_pickle=True)
    D = np.load(os.path.join(HERE, 'ablation_logs', 'mediocre_band.npz'), allow_pickle=True)
    S = np.load(os.path.join(HERE, 'ablation_logs', 'p0b_sweep.npz'), allow_pickle=True)
    vf = [str(x) for x in V['fid']]; df = [str(x) for x in D['fid']]
    pos = {f: i for i, f in enumerate(vf)}; o = np.array([pos[f] for f in df])
    head_th = V['head_theta'][o].astype(float)
    gt3d = V['gt3d'][o].astype(float)
    conf_np = V['conf'][o].astype(float)
    det2d_o = S['det2d']; rp1 = S['rp1']
    clean, exc, med = S['clean'], S['exc'], S['med']
    add1 = S['add1']

    gtth = np.zeros((len(df), 7))
    for i, fid in enumerate(df):
        d = json.load(open(os.path.join(GT_DIR, f'{fid}.json')))
        j = {x['name'].split('/')[-1]: x['position'] for x in d['sim_state']['joints']}
        gtth[i] = [j[f'panda_joint{k}'] for k in range(1, 8)]
    fk_gt = panda_forward_kinematics(torch.tensor(gtth)).numpy()
    Rs, ts = [], []
    for k in range(len(df)):
        R, t = kabsch_Rt(fk_gt[k], gt3d[k]); Rs.append(R); ts.append(t)
    d6i = R_to_d6(torch.tensor(np.stack(Rs))); ti = torch.tensor(np.stack(ts))
    tt = torch.tensor(det2d_o); th0 = torch.tensor(head_th); cf = torch.tensor(conf_np)

    def report(tag, a, extra=''):
        print(f'  {tag:44s} clean-AUC={AUC(a[clean]):.4f}  exc={np.median(a[exc])*1000:6.1f}  '
              f'med={np.median(a[med])*1000:6.1f}  ALL={AUC(a):.4f} {extra}')

    print('=' * 104)
    print('기준선')
    print('=' * 104)
    report('pass-1 자유 (미니솔버 baseline)', add1)
    report('배포 실측', D['add'])

    print()
    print('=' * 104)
    print('(a) 강한 anchor 범위 — freeze(0.56) 와 자유(0.776) 사이에 내부 최적점이 있는가?')
    print('=' * 104)
    for w in [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        th, cam, rp = solve2(tt, th0.clone(), d6i.clone(), ti.clone(),
                             n_iter=args.iters, anchor_w=w, conf=cf)
        a = np.linalg.norm(cam.numpy() - gt3d, axis=2).mean(1)
        report(f'anchor_w={w:g}', a)

    # ---- GT-free 게이트 신호 ----
    diag = np.linalg.norm(det2d_o.max(1) - det2d_o.min(1), axis=1)   # 검출 스켈레톤 대각(px)
    mconf = conf_np.mean(1)
    gates = {
        'bbox 작음 (하위 50%)': diag < np.median(diag),
        'bbox 작음 (하위 25%)': diag < np.quantile(diag, .25),
        'pass-1 reproj 상위 25%': rp1 > np.quantile(rp1, .75),
        'pass-1 reproj 상위 10%': rp1 > np.quantile(rp1, .90),
        'conf 하위 25%': mconf < np.quantile(mconf, .25),
    }
    print()
    print('=' * 104)
    print('(b) 프레임 조건부 — 게이트된 프레임에서만 freeze/anchor 적용 (게이트는 GT-free)')
    print('=' * 104)
    print(f'  {"게이트":26s}{"적용률":>8s}{"게이트내 mediocre비율":>20s}')
    for gn, g in gates.items():
        print(f'  {gn:26s}{g.mean()*100:7.1f}%{med[g].mean()*100:19.1f}%')
    print()
    for gn, g in gates.items():
        for fn, js, aw in [('distal J5,J6 freeze', [4, 5], 0.0),
                           ('J4,J5,J6 freeze', [3, 4, 5], 0.0),
                           ('anchor_w=30', [], 30.0)]:
            fm = torch.zeros((len(df), 7), dtype=torch.bool)
            for j in js: fm[torch.tensor(g), j] = True
            aw_v = aw if aw > 0 else 0.0
            if aw > 0:
                # 게이트 밖은 anchor 0 -> 두 번 풀어서 합성
                th_a, cam_a, _ = solve2(tt, th0.clone(), d6i.clone(), ti.clone(),
                                        n_iter=args.iters, anchor_w=aw_v, conf=cf)
                a_g = np.linalg.norm(cam_a.numpy() - gt3d, axis=2).mean(1)
                a = np.where(g, a_g, add1)
            else:
                th, cam, _ = solve2(tt, th0.clone(), d6i.clone(), ti.clone(),
                                    n_iter=args.iters, freeze_mask=fm, conf=cf)
                a_g = np.linalg.norm(cam.numpy() - gt3d, axis=2).mean(1)
                a = np.where(g, a_g, add1)
            report(f'{gn} + {fn}', a)
    print()
    print('  판독: 어떤 조합도 baseline(clean-AUC/med)을 못 넘으면 P0b 전체가 반증.')


if __name__ == '__main__':
    main()
