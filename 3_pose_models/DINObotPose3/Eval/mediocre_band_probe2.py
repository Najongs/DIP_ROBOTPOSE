#!/usr/bin/env python
"""mediocre 밴드 심층: (A) 2D-정합 분해, (B) 다변량 설명력, (C) scale->range 결합 메커니즘,
(D) 관측성 바닥 검정 = 완벽한 GT 2D로 풀었을 때 남는 오차. 전부 CPU.
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'TRAIN'))
from model_v4 import panda_forward_kinematics, _PANDA_JOINT_LIMITS  # noqa: E402

KP = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4',
      'panda_link6', 'panda_link7', 'panda_hand']
W, H, F, CX, CY = 640, 480, 320.0, 320.0, 240.0
GT_DIR = os.path.join(HERE, '..', 'Dataset', 'Converted_dataset',
                      'DREAM_to_DREAM_syn', 'panda_synth_test_dr')
AUC = lambda a: float(np.clip(1 - 10 * np.asarray(a), 0, 1).mean())


def d6_to_R(d6):
    a, b = d6[..., :3], d6[..., 3:]
    r1 = torch.nn.functional.normalize(a, dim=-1)
    b = b - (r1 * b).sum(-1, keepdim=True) * r1
    r2 = torch.nn.functional.normalize(b, dim=-1)
    r3 = torch.cross(r1, r2, dim=-1)
    return torch.stack([r1, r2, r3], dim=-1)


def R_to_d6(R):
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def kabsch_Rt(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1., 1., d]) @ U.T
    return R, qc - R @ pc


def solve(kp2d_t, theta0, d60, t0, n_iter=400, lr=5e-3, free_theta=True):
    """kp2d_t:(B,7,2) 목표 2D. theta0:(B,7) d60:(B,6) t0:(B,3). 재투영 최소화."""
    th = theta0.clone().requires_grad_(free_theta)
    d6 = d60.clone().requires_grad_(True)
    t = t0.clone().requires_grad_(True)
    params = [d6, t] + ([th] if free_theta else [])
    opt = torch.optim.Adam(params, lr=lr)
    lim = torch.tensor(_PANDA_JOINT_LIMITS[:7], dtype=torch.float64)
    for it in range(n_iter):
        opt.zero_grad()
        fk = panda_forward_kinematics(th)                       # (B,7,3) robot
        cam = torch.einsum('bij,bkj->bki', d6_to_R(d6), fk) + t[:, None, :]
        z = cam[..., 2].clamp(min=1e-3)
        u = F * cam[..., 0] / z + CX
        v = F * cam[..., 1] / z + CY
        p = torch.stack([u, v], -1)
        loss = ((p - kp2d_t) ** 2).sum(-1).clamp(max=1e6).mean()
        if free_theta:
            loss = loss + 1e3 * (torch.relu(th - lim[:, 1]) ** 2 + torch.relu(lim[:, 0] - th) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        fk = panda_forward_kinematics(th)
        cam = torch.einsum('bij,bkj->bki', d6_to_R(d6), fk) + t[:, None, :]
        z = cam[..., 2].clamp(min=1e-3)
        p = torch.stack([F * cam[..., 0] / z + CX, F * cam[..., 1] / z + CY], -1)
        px = (p - kp2d_t).norm(dim=-1).mean(-1)
    return th.detach(), cam.detach(), px.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat', default=os.path.join(HERE, 'ablation_logs', 'mediocre_band.npz'))
    ap.add_argument('--pred', default=os.path.join(HERE, 'rc_dumps_oas', 'dr_pred.npz'))
    ap.add_argument('--n-restart', type=int, default=6)
    ap.add_argument('--skip-floor', action='store_true')
    args = ap.parse_args()

    D = np.load(args.feat, allow_pickle=True)
    P = np.load(args.pred, allow_pickle=True)
    fids = [str(x) for x in D['fid']]
    add_mm = D['add'] * 1000
    clean = D['n_offframe'] == 0
    exc = clean & (add_mm < 30)
    med = clean & (add_mm >= 30) & (add_mm < 100)
    sel = exc | med

    gt2d = np.zeros((len(fids), 7, 2)); gtth = np.zeros((len(fids), 7))
    for i, fid in enumerate(fids):
        d = json.load(open(os.path.join(GT_DIR, f'{fid}.json')))
        kmap = {k['name']: k for k in d['objects'][0]['keypoints']}
        gt2d[i] = [kmap[n]['projected_location'] for n in KP]
        joints = {j['name'].split('/')[-1]: j['position'] for j in d['sim_state']['joints']}
        gtth[i] = [joints[f'panda_joint{k}'] for k in range(1, 8)]
    gt3d = P['gt3d'].astype(float)
    pred3d = P['kp_cam'].astype(float)
    prth = P['theta'].astype(float)

    print('=' * 78)
    print('A. mediocre 밴드의 2D-정합 분해 (예측 3D를 재투영한 px 오차 기준)')
    print('=' * 78)
    px = D['px_err_mean']
    bins = [(0, 1.5), (1.5, 3), (3, 6), (6, 1e9)]
    print(f'{"px_err bin":14s}{"n_med":>7s}{"%med":>7s}{"ADD med":>9s}{"radial":>8s}{"tangent":>8s}'
          f'{"shape":>8s}{"pose":>8s}{"aerrJ4":>8s}{"손실AUC":>9s}')
    for lo, hi in bins:
        m = med & (px >= lo) & (px < hi)
        if m.sum() == 0: continue
        lost = m.sum() / len(add_mm) - np.clip(1 - 10 * D['add'][m], 0, 1).sum() / len(add_mm)
        print(f'[{lo:4.1f},{hi if hi < 1e8 else 999:5.1f})  {m.sum():5d}{m.sum()/med.sum()*100:7.1f}'
              f'{np.median(add_mm[m]):9.1f}{np.median(D["err_radial"][m])*1000:8.1f}'
              f'{np.median(D["err_tangent"][m])*1000:8.1f}{np.median(D["add_shape"][m])*1000:8.1f}'
              f'{np.median(D["add_pose"][m])*1000:8.1f}{np.median(D["aerr_J4"][m]):8.2f}{lost:9.4f}')
    print(f'  참고 excellent: px_err median={np.median(px[exc]):.2f}, ADD median={np.median(add_mm[exc]):.1f}mm')

    print()
    print('=' * 78)
    print('C. scale->range 결합: "팔 겉보기 길이 오차" -> "시선방향 거리 오차"')
    print('=' * 78)
    # 예측 FK의 팔 펼침(link0->hand 3D 거리) vs GT
    ext_p = np.linalg.norm(pred3d[:, 6] - pred3d[:, 0], axis=1)
    ext_g = np.linalg.norm(gt3d[:, 6] - gt3d[:, 0], axis=1)
    rel_ext = (ext_p - ext_g) / ext_g
    # 부호있는 range 오차 (예측 centroid 깊이 - GT)
    dz = pred3d.mean(1)[:, 2] - gt3d.mean(1)[:, 2]
    for nm, m in [('excellent', exc), ('mediocre', med)]:
        r = np.corrcoef(rel_ext[m], dz[m])[0, 1]
        print(f'{nm:12s} corr(상대 팔길이오차, Δdepth) = {r:+.3f}   '
              f'|rel_ext| median={np.median(np.abs(rel_ext[m]))*100:5.2f}%   '
              f'|Δdepth| median={np.median(np.abs(dz[m]))*1000:6.1f}mm')
    # 예측 스케일 비율로 예측되는 깊이 변화: z_pred ≈ z_gt * (ext_p/ext_g)
    pred_dz = gt3d.mean(1)[:, 2] * rel_ext
    for nm, m in [('excellent', exc), ('mediocre', med)]:
        r = np.corrcoef(pred_dz[m], dz[m])[0, 1]
        expl = 1 - np.var(dz[m] - pred_dz[m]) / np.var(dz[m])
        print(f'{nm:12s} 스케일모델 z*rel_ext 로 Δdepth 설명: corr={r:+.3f}  R2={expl:+.3f}')

    print()
    print('=' * 78)
    print('B. 다변량 설명력 (입력측 관측가능 특징만, mediocre 판별 로지스틱)')
    print('=' * 78)
    feats = ['forearm_view_cos', 'upperarm_view_cos', 'arm_view_cos', 'fore_min', 'fore_dist',
             'bbox_diag_px', 'z_mean', 'kp2d_min_dist', 'selfocc_ratio', 'z_spread',
             'min_border_px', 'n_kp_occluded']
    X = np.stack([D[k][sel] for k in feats], 1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.hstack([X, np.ones((len(X), 1))])
    y = med[sel].astype(float)
    w = np.zeros(X.shape[1])
    for _ in range(3000):
        p = 1 / (1 + np.exp(-X @ w))
        w -= 0.5 * (X.T @ (p - y)) / len(X) * 10
    p = 1 / (1 + np.exp(-X @ w))
    o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    print(f'  입력측 특징만 사용한 mediocre 판별 AUROC = {(rk[y==1].sum()-n1*(n1+1)/2)/(n1*n0):.3f}')
    for k, wi in sorted(zip(feats, w[:-1]), key=lambda z: -abs(z[1])):
        print(f'    {k:22s} {wi:+.3f}')
    # 각도오차 포함 시
    feats2 = feats + ['aerr_J4', 'aerr_J2', 'aerr_J6', 'px_err_mean']
    X2 = np.stack([D[k][sel] for k in feats2], 1)
    X2 = (X2 - X2.mean(0)) / (X2.std(0) + 1e-9); X2 = np.hstack([X2, np.ones((len(X2), 1))])
    w2 = np.zeros(X2.shape[1])
    for _ in range(3000):
        p2 = 1 / (1 + np.exp(-X2 @ w2)); w2 -= 0.5 * (X2.T @ (p2 - y)) / len(X2) * 10
    p2 = 1 / (1 + np.exp(-X2 @ w2)); o = np.argsort(p2); rk = np.empty(len(p2)); rk[o] = np.arange(1, len(p2)+1)
    print(f'  +각도오차/재투영 포함 AUROC = {(rk[y==1].sum()-n1*(n1+1)/2)/(n1*n0):.3f}  (상한 참고)')

    if args.skip_floor:
        return

    print()
    print('=' * 78)
    print('D. 관측성 바닥 검정 — "완벽한 GT 2D"로 풀면 mediocre가 사라지는가?')
    print('=' * 78)
    idx = np.where(sel)[0]
    tt = torch.tensor(gt2d[idx], dtype=torch.float64)
    gtth_t = torch.tensor(gtth[idx], dtype=torch.float64)
    # GT 포즈 (Kabsch: FK(GTθ) -> gt3d)
    fk_gt = panda_forward_kinematics(gtth_t).numpy()
    Rs, ts = [], []
    for k in range(len(idx)):
        R, t = kabsch_Rt(fk_gt[k], gt3d[idx[k]])
        Rs.append(R); ts.append(t)
    Rs = np.stack(Rs); ts = np.stack(ts)
    d6gt = R_to_d6(torch.tensor(Rs))
    tgt = torch.tensor(ts)
    theta_pred_t = torch.tensor(prth[idx])

    rng = np.random.RandomState(0)
    best_add = np.full(len(idx), 1e9); best_px = np.full(len(idx), 1e9)
    worst_add_at_lowpx = np.zeros(len(idx))
    for r in range(args.n_restart):
        if r == 0:
            th0 = theta_pred_t.clone()                 # 배포 head 초기값
        elif r == 1:
            th0 = gtth_t.clone()                       # GT 초기값 (바닥 확인)
        else:
            th0 = gtth_t + torch.tensor(rng.uniform(-0.6, 0.6, size=gtth_t.shape))
        d60 = d6gt + torch.tensor(rng.normal(0, 0.05 if r else 0.0, size=d6gt.shape))
        t0 = tgt + torch.tensor(rng.normal(0, 0.05 if r else 0.0, size=tgt.shape))
        th, cam, px = solve(tt, th0, d60, t0)
        add = np.linalg.norm(cam.numpy() - gt3d[idx], axis=2).mean(1)
        pxn = px.numpy()
        upd = add < best_add
        best_add[upd] = add[upd]; best_px[upd] = pxn[upd]
        ok = pxn < 2.0
        worst_add_at_lowpx = np.where(ok & (add > worst_add_at_lowpx), add, worst_add_at_lowpx)
        print(f'  restart {r} ({"pred-init" if r==0 else "GT-init" if r==1 else "rand"}): '
              f'px median={np.median(pxn):6.2f}  ADD median={np.median(add)*1000:6.1f}mm')

    for nm, m in [('excellent', exc[idx.astype(int)] if False else np.isin(idx, np.where(exc)[0])),
                  ('mediocre', np.isin(idx, np.where(med)[0]))]:
        print(f'\n  [{nm}] n={m.sum()}')
        print(f'    현 파이프라인 ADD median = {np.median(add_mm[idx][m]):6.1f}mm   AUC={AUC(D["add"][idx][m]):.4f}')
        print(f'    완벽 GT-2D solve (최선 restart) ADD median = {np.median(best_add[m])*1000:6.1f}mm  '
              f'AUC={AUC(best_add[m]):.4f}')
        print(f'    -> GT-2D로도 >30mm 남는 비율 = {np.mean(best_add[m]*1000 > 30)*100:5.1f}%')
        print(f'    모호성(재투영<2px인데 ADD 큼): worst ADD median = '
              f'{np.median(worst_add_at_lowpx[m])*1000:6.1f}mm, '
              f'>30mm 비율 = {np.mean(worst_add_at_lowpx[m]*1000 > 30)*100:5.1f}%')

    np.savez(os.path.join(HERE, 'ablation_logs', 'mediocre_floor.npz'),
             idx=idx, best_add=best_add, best_px=best_px, worst_lowpx=worst_add_at_lowpx)


if __name__ == '__main__':
    main()
