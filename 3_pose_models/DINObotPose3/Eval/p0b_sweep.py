#!/usr/bin/env python
"""P0b 부분/조건부 freeze + θ-anchor 스윕 (CPU, 재학습 0, GPU 불필요).

근거: (검출2D + GTθ 자유init)=39.8mm vs (GTθ 고정)=10.1mm -> 나쁜 2D가 솔버를 통해 θ를 오염.
전체 freeze 는 반증(0.704->0.533)이므로 여기서는 (1) 관절별 부분 freeze, (2) 프레임 조건부 freeze
(게이트는 GT-free 신호만 사용), (3) anchor_w 연속 완화 를 스윕한다.

미니 솔버는 배포 솔버의 축약판이지만 2x2 요인설계에서 배포값을 재현했다
(excellent 12.9 vs 12.7mm, mediocre 48.7 vs 42.1mm) -> 랭킹 용도로 타당. 승자는 GPU에서 재확인.

⚠️ 좌표계: dump 의 kp2d 는 CROP-IS. 반드시 kp2d_full(FULL-FRAME IS)을 쓰고 원본 640x480 으로 매핑.
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'TRAIN'))
from model_v4 import panda_forward_kinematics, _PANDA_JOINT_LIMITS  # noqa: E402
from mediocre_band_probe2 import R_to_d6, kabsch_Rt, KP, GT_DIR, d6_to_R  # noqa: E402

W, H, F_, CX, CY = 640, 480, 320.0, 320.0, 240.0
AUC = lambda a: float(np.clip(1 - 10 * np.asarray(a), 0, 1).mean())


def solve2(kp2d_t, theta0, d60, t0, n_iter=400, lr=5e-3, freeze_mask=None, anchor_w=0.0,
           conf=None):
    """freeze_mask: (B,7) bool — True 인 관절은 theta0 에 고정. anchor_w: theta0 로 당기는 항."""
    B = theta0.shape[0]
    th_p = theta0.clone().requires_grad_(True)
    d6 = d60.clone().requires_grad_(True)
    t = t0.clone().requires_grad_(True)
    opt = torch.optim.Adam([th_p, d6, t], lr=lr)
    lim = torch.tensor(_PANDA_JOINT_LIMITS[:7], dtype=torch.float64)
    th_ref = theta0.detach().clone()
    fm = freeze_mask if freeze_mask is not None else torch.zeros_like(th_ref, dtype=torch.bool)
    w = (conf if conf is not None else torch.ones(kp2d_t.shape[:2], dtype=torch.float64))
    for _ in range(n_iter):
        opt.zero_grad()
        th = torch.where(fm, th_ref, th_p)
        fk = panda_forward_kinematics(th)
        cam = torch.einsum('bij,bkj->bki', d6_to_R(d6), fk) + t[:, None, :]
        z = cam[..., 2].clamp(min=1e-3)
        p = torch.stack([F_ * cam[..., 0] / z + CX, F_ * cam[..., 1] / z + CY], -1)
        loss = (w * ((p - kp2d_t) ** 2).sum(-1).clamp(max=1e6)).mean()
        loss = loss + 1e3 * (torch.relu(th - lim[:, 1]) ** 2 + torch.relu(lim[:, 0] - th) ** 2).sum(-1).mean()
        if anchor_w > 0:
            loss = loss + anchor_w * ((th[:, :6] - th_ref[:, :6]) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        th = torch.where(fm, th_ref, th_p)
        fk = panda_forward_kinematics(th)
        cam = torch.einsum('bij,bkj->bki', d6_to_R(d6), fk) + t[:, None, :]
        z = cam[..., 2].clamp(min=1e-3)
        p = torch.stack([F_ * cam[..., 0] / z + CX, F_ * cam[..., 1] / z + CY], -1)
        rp = (p - kp2d_t).norm(dim=-1).mean(-1)
    return th.detach(), cam.detach(), rp.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default=os.path.join(HERE, 'rc_dumps_oas', 'dr_pred_v3.npz'))
    ap.add_argument('--iters', type=int, default=400)
    args = ap.parse_args()

    V = np.load(args.dump, allow_pickle=True)
    D = np.load(os.path.join(HERE, 'ablation_logs', 'mediocre_band.npz'), allow_pickle=True)
    vf = [str(x) for x in V['fid']]; df = [str(x) for x in D['fid']]
    pos = {f: i for i, f in enumerate(vf)}
    o = np.array([pos[f] for f in df])
    kp_full = V['kp2d_full'][o].astype(float)
    gtkp_c = V['gtkp2d'][o].astype(float)
    head_th = V['head_theta'][o].astype(float)
    gt3d = V['gt3d'][o].astype(float)
    conf_np = V['conf'][o].astype(float)

    add0 = D['add'] * 1000
    clean = D['n_offframe'] == 0
    exc = clean & (add0 < 30); med = clean & (add0 >= 30) & (add0 < 100)

    # GT 2D (원본) + crop-IS -> 원본 매핑
    gt2d_o = np.zeros((len(df), 7, 2)); gtth = np.zeros((len(df), 7))
    for i, fid in enumerate(df):
        d = json.load(open(os.path.join(GT_DIR, f'{fid}.json')))
        kmap = {k['name']: k for k in d['objects'][0]['keypoints']}
        gt2d_o[i] = [kmap[n]['projected_location'] for n in KP]
        j = {x['name'].split('/')[-1]: x['position'] for x in d['sim_state']['joints']}
        gtth[i] = [j[f'panda_joint{k}'] for k in range(1, 8)]
    det2d_o = np.zeros_like(gt2d_o)
    for i in range(len(df)):
        for ax in range(2):
            A = np.stack([gtkp_c[i, :, ax], np.ones(7)], 1)
            c, *_ = np.linalg.lstsq(A, gt2d_o[i, :, ax], rcond=None)
            det2d_o[i, :, ax] = kp_full[i, :, ax] * c[0] + c[1]

    # ---- 초기 포즈: head θ 의 FK 를 검출 2D 에 맞춘 PnP-유사 init (GT 미사용) ----
    fk_h = panda_forward_kinematics(torch.tensor(head_th)).numpy()
    # GT 포즈로 init 하면 부정이므로, 검출 2D 만으로 pass-1 자유 solve 후 그 결과를 init 으로 씀
    fk_gt = panda_forward_kinematics(torch.tensor(gtth)).numpy()
    Rs, ts = [], []
    for k in range(len(df)):
        R, t = kabsch_Rt(fk_gt[k], gt3d[k]); Rs.append(R); ts.append(t)
    d6_gtpose = R_to_d6(torch.tensor(np.stack(Rs))); t_gtpose = torch.tensor(np.stack(ts))

    tt = torch.tensor(det2d_o); th0 = torch.tensor(head_th); cf = torch.tensor(conf_np)
    print('pass-1 자유 solve (배포 baseline 대응) ...', flush=True)
    th1, cam1, rp1 = solve2(tt, th0.clone(), d6_gtpose.clone(), t_gtpose.clone(),
                            n_iter=args.iters, conf=cf)
    add1 = np.linalg.norm(cam1.numpy() - gt3d, axis=2).mean(1)
    d6_1 = R_to_d6(torch.tensor(np.stack([kabsch_Rt(panda_forward_kinematics(th1[k:k+1]).numpy()[0],
                                                    cam1[k].numpy())[0] for k in range(len(df))])))
    t_1 = torch.tensor(np.stack([kabsch_Rt(panda_forward_kinematics(th1[k:k+1]).numpy()[0],
                                           cam1[k].numpy())[1] for k in range(len(df))]))

    def report(tag, add_m, extra=''):
        print(f'  {tag:38s} clean-AUC={AUC(add_m[clean]):.4f}  exc={np.median(add_m[exc])*1000:6.1f}  '
              f'med={np.median(add_m[med])*1000:6.1f}  ALL-AUC={AUC(add_m):.4f} {extra}')

    print()
    print('=' * 96)
    print('기준선 (미니솔버) — 배포: clean-AUC 0.7884 / exc 12.7mm / med 42.1mm / ALL 0.704')
    print('=' * 96)
    report('pass-1 자유 (baseline)', add1)
    report('배포 실측(참고)', D['add'])

    JN = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    print()
    print('=' * 96)
    print('1. 관절별 부분 freeze (전 프레임 적용) — head θ 에 해당 관절 고정')
    print('=' * 96)
    masks = {
        'none (자유)': [],
        'ALL (P0, 반증됨)': [0, 1, 2, 3, 4, 5],
        'distal J5,J6': [4, 5],
        'distal J4,J5,J6': [3, 4, 5],
        'wrist+J4 J4,J5,J6 + J1': [0, 3, 4, 5],
        'proximal J1-J4': [0, 1, 2, 3],
        'J1 only (게이지)': [0],
        'all but J4': [0, 1, 2, 4, 5],
        'J5,J6 + J2': [1, 4, 5],
    }
    best = {}
    for nm, js in masks.items():
        fm = torch.zeros((len(df), 7), dtype=torch.bool)
        for j in js: fm[:, j] = True
        th, cam, rp = solve2(tt, th0.clone(), d6_1.clone(), t_1.clone(),
                             n_iter=args.iters, freeze_mask=fm, conf=cf)
        a = np.linalg.norm(cam.numpy() - gt3d, axis=2).mean(1)
        best[nm] = a
        report(nm, a)

    print()
    print('=' * 96)
    print('2. θ-anchor 연속 완화 (head 예측 주변으로 당김; freeze 와 자유 사이)')
    print('=' * 96)
    anch = {}
    for w in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 2e-1, 1.0]:
        th, cam, rp = solve2(tt, th0.clone(), d6_1.clone(), t_1.clone(),
                             n_iter=args.iters, anchor_w=w, conf=cf)
        a = np.linalg.norm(cam.numpy() - gt3d, axis=2).mean(1)
        anch[w] = a
        report(f'anchor_w={w:g}', a)

    np.savez(os.path.join(HERE, 'ablation_logs', 'p0b_sweep.npz'),
             add1=add1, **{f'mask_{k}': v for k, v in best.items()},
             **{f'anch_{k}': v for k, v in anch.items()},
             det2d=det2d_o, rp1=rp1.numpy(), clean=clean, exc=exc, med=med)
    print('\n저장: ablation_logs/p0b_sweep.npz  (조건부 게이트는 p0b_sweep2.py 에서)')


if __name__ == '__main__':
    main()
