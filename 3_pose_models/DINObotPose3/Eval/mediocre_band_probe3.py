#!/usr/bin/env python
"""mediocre 밴드 3단계: (B') 제대로 된 다변량 판별, (C') scale->range 재검토,
(E) 조건수 실험 = GT 2D에 σ px 노이즈를 넣고 재-solve -> dADD/dσ 프레임별 측정.
조건수가 mediocre를 설명하면 = 기하학적 ill-conditioning(관측성 성격), 아니면 = 각도추출 실패.
"""
import argparse, json, os, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'TRAIN'))
from model_v4 import panda_forward_kinematics  # noqa: E402
from mediocre_band_probe2 import solve, R_to_d6, kabsch_Rt, KP, GT_DIR  # noqa: E402

AUC = lambda a: float(np.clip(1 - 10 * np.asarray(a), 0, 1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigmas', default='0.5,1,2,3')
    ap.add_argument('--reps', type=int, default=3)
    args = ap.parse_args()

    D = np.load(os.path.join(HERE, 'ablation_logs', 'mediocre_band.npz'), allow_pickle=True)
    P = np.load(os.path.join(HERE, 'rc_dumps_oas', 'dr_pred.npz'), allow_pickle=True)
    fids = [str(x) for x in D['fid']]
    add_mm = D['add'] * 1000
    clean = D['n_offframe'] == 0
    exc = clean & (add_mm < 30); med = clean & (add_mm >= 30) & (add_mm < 100)
    sel = exc | med
    gt3d = P['gt3d'].astype(float); pred3d = P['kp_cam'].astype(float)

    gt2d = np.zeros((len(fids), 7, 2)); gtth = np.zeros((len(fids), 7))
    for i, fid in enumerate(fids):
        d = json.load(open(os.path.join(GT_DIR, f'{fid}.json')))
        kmap = {k['name']: k for k in d['objects'][0]['keypoints']}
        gt2d[i] = [kmap[n]['projected_location'] for n in KP]
        j = {x['name'].split('/')[-1]: x['position'] for x in d['sim_state']['joints']}
        gtth[i] = [j[f'panda_joint{k}'] for k in range(1, 8)]

    print('=' * 78)
    print("B'. 다변량 판별 (제대로) — mediocre vs excellent, clean 내")
    print('=' * 78)
    geo = ['forearm_view_cos', 'upperarm_view_cos', 'arm_view_cos', 'fore_min', 'fore_dist',
           'fore_prox', 'bbox_diag_px', 'bbox_area_px', 'z_mean', 'z_base', 'kp2d_min_dist',
           'selfocc_ratio', 'z_spread', 'min_border_px', 'n_kp_occluded', 'px_per_link',
           'vis_frac'] + [f'gtJ{i+1}' for i in range(6)]
    y = med[sel].astype(int)
    for nm, cols in [('기하/입력측만', geo),
                     ('+각도오차', geo + [f'aerr_J{i+1}' for i in range(6)]),
                     ('+2D오차', geo + ['px_err_mean', 'px_err_max', 'reproj_solver']),
                     ('전부', geo + [f'aerr_J{i+1}' for i in range(6)] + ['px_err_mean', 'px_err_max', 'reproj_solver'])]:
        X = np.stack([D[k][sel] for k in cols], 1)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        lr = LogisticRegression(max_iter=5000, C=1.0)
        a_lr = roc_auc_score(y, cross_val_predict(lr, Xs, y, cv=5, method='predict_proba')[:, 1])
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
        a_gb = roc_auc_score(y, cross_val_predict(gb, X, y, cv=5, method='predict_proba')[:, 1])
        print(f'  {nm:14s} 5-fold CV AUROC   logistic={a_lr:.3f}   GBM={a_gb:.3f}')
        if nm == '기하/입력측만':
            gb.fit(X, y)
            imp = sorted(zip(cols, gb.feature_importances_), key=lambda z: -z[1])
            print('    GBM 중요도 top10: ' + ', '.join(f'{k}={v:.3f}' for k, v in imp[:10]))

    print()
    print('=' * 78)
    print("C'. scale->range 재검토 (constellation 확산 기준)")
    print('=' * 78)
    def spread(X):
        return np.linalg.norm(X - X.mean(1, keepdims=True), axis=2).mean(1)
    sp_p, sp_g = spread(pred3d), spread(gt3d)
    rel = (sp_p - sp_g) / sp_g
    zp, zg = pred3d.mean(1)[:, 2], gt3d.mean(1)[:, 2]
    dz_rel = (zp - zg) / zg
    for nm, m in [('excellent', exc), ('mediocre', med)]:
        r = np.corrcoef(rel[m], dz_rel[m])[0, 1]
        sl = np.polyfit(rel[m], dz_rel[m], 1)[0]
        print(f'  {nm:12s} corr(Δ상대 constellation 확산, Δ상대 깊이)={r:+.3f}  기울기={sl:+.2f} '
              f'(완전 scale결합이면 +1.0)  |Δ확산| med={np.median(np.abs(rel[m]))*100:.2f}%  '
              f'|Δz/z| med={np.median(np.abs(dz_rel[m]))*100:.2f}%')
    print('  판독: 기울기≈+1 이면 "겉보기 크기 오차 -> 거리 오차" 결합이 성립')

    print()
    print('=' * 78)
    print('E. 조건수 실험 — GT 2D + σpx 노이즈로 재-solve (프레임 기하만의 민감도)')
    print('=' * 78)
    idx = np.where(sel)[0]
    fk_gt = panda_forward_kinematics(torch.tensor(gtth[idx])).numpy()
    Rs, ts = [], []
    for k in range(len(idx)):
        R, t = kabsch_Rt(fk_gt[k], gt3d[idx[k]]); Rs.append(R); ts.append(t)
    d6gt = R_to_d6(torch.tensor(np.stack(Rs))); tgt = torch.tensor(np.stack(ts))
    th_gt_t = torch.tensor(gtth[idx])
    rng = np.random.RandomState(0)
    sigmas = [float(s) for s in args.sigmas.split(',')]
    sens = {}
    print(f'{"σ(px)":>7s}{"exc ADD":>10s}{"med ADD":>10s}{"exc AUC":>10s}{"med AUC":>10s}')
    is_exc = np.isin(idx, np.where(exc)[0]); is_med = np.isin(idx, np.where(med)[0])
    for s in sigmas:
        accum = []
        for r in range(args.reps):
            noisy = torch.tensor(gt2d[idx] + rng.normal(0, s, size=gt2d[idx].shape))
            th, cam, px = solve(noisy, th_gt_t.clone(), d6gt.clone(), tgt.clone(), n_iter=300)
            accum.append(np.linalg.norm(cam.numpy() - gt3d[idx], axis=2).mean(1))
        a = np.mean(accum, 0)
        sens[s] = a
        print(f'{s:7.1f}{np.median(a[is_exc])*1000:10.1f}{np.median(a[is_med])*1000:10.1f}'
              f'{AUC(a[is_exc]):10.4f}{AUC(a[is_med]):10.4f}')
    print(f'  현 파이프라인:   exc {np.median(add_mm[idx][is_exc]):.1f}mm / '
          f'med {np.median(add_mm[idx][is_med]):.1f}mm')
    print('  판독: mediocre가 같은 σ에서 훨씬 큰 ADD를 내면 = 기하학적 ill-conditioning(프레임 탓)')
    print('        두 밴드가 같은 σ에서 비슷하면 = 프레임 기하는 무죄, 각도/2D 추출이 원인')

    # 프레임별 민감도 계수 (mm per px) 와 기하 요인 상관
    k_sens = (sens[sigmas[-1]] - sens[sigmas[0]]) / (sigmas[-1] - sigmas[0]) * 1000
    print(f'\n  프레임별 민감도 dADD/dσ (mm/px): exc median={np.median(k_sens[is_exc]):.1f}  '
          f'med median={np.median(k_sens[is_med]):.1f}')
    a = roc_auc_score(is_med.astype(int), k_sens)
    print(f'  민감도 단독 mediocre 판별 AUROC = {a:.3f}')
    print('  민감도와 기하 요인의 상관:')
    for k in ['forearm_view_cos', 'upperarm_view_cos', 'bbox_diag_px', 'z_mean',
              'fore_min', 'fore_dist', 'kp2d_min_dist', 'selfocc_ratio', 'z_spread']:
        print(f'    {k:22s} corr={np.corrcoef(D[k][idx], k_sens)[0,1]:+.3f}')
    np.savez(os.path.join(HERE, 'ablation_logs', 'mediocre_sens.npz'),
             idx=idx, k_sens=k_sens, **{f's{ss}': sens[ss] for ss in sigmas})


if __name__ == '__main__':
    main()
