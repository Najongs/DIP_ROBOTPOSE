#!/usr/bin/env python
"""mediocre 밴드 마무리: (F) 유효 2D-노이즈 σ 역산, (G) Kabsch-흡수 후 관절별 귀책,
(H) 기하 트리거 -> 특정 관절 오차 연결, (I) 시각화(최악 mediocre vs 최고 excellent).
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'TRAIN'))
from model_v4 import panda_forward_kinematics  # noqa: E402
from mediocre_band_probe2 import KP, GT_DIR, kabsch_Rt  # noqa: E402

KP_SHORT = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
AUC = lambda a: float(np.clip(1 - 10 * np.asarray(a), 0, 1).mean())
IMG_DIR = os.path.abspath(os.path.join(GT_DIR, '..', '..', '..', 'DREAM_syn', 'panda_synth_test_dr'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--viz-dir', default=os.path.join(HERE, 'mediocre_viz'))
    ap.add_argument('--n-viz', type=int, default=8)
    args = ap.parse_args()

    D = np.load(os.path.join(HERE, 'ablation_logs', 'mediocre_band.npz'), allow_pickle=True)
    P = np.load(os.path.join(HERE, 'rc_dumps_oas', 'dr_pred.npz'), allow_pickle=True)
    S = np.load(os.path.join(HERE, 'ablation_logs', 'mediocre_sens.npz'), allow_pickle=True)
    fids = [str(x) for x in D['fid']]
    add_mm = D['add'] * 1000
    clean = D['n_offframe'] == 0
    exc = clean & (add_mm < 30); med = clean & (add_mm >= 30) & (add_mm < 100)
    pred3d = P['kp_cam'].astype(float); gt3d = P['gt3d'].astype(float)
    prth = P['theta'].astype(float)

    gt2d = np.zeros((len(fids), 7, 2)); gtth = np.zeros((len(fids), 7))
    for i, fid in enumerate(fids):
        d = json.load(open(os.path.join(GT_DIR, f'{fid}.json')))
        kmap = {k['name']: k for k in d['objects'][0]['keypoints']}
        gt2d[i] = [kmap[n]['projected_location'] for n in KP]
        j = {x['name'].split('/')[-1]: x['position'] for x in d['sim_state']['joints']}
        gtth[i] = [j[f'panda_joint{k}'] for k in range(1, 8)]

    print('=' * 78)
    print('F. 유효 2D-노이즈 σ 역산 (E의 σ-ADD 곡선을 프레임별로 뒤집음)')
    print('=' * 78)
    idx = S['idx']; sig = np.array([0.5, 1.0, 2.0, 3.0])
    curves = np.stack([S[f's{s}'] for s in sig], 1)     # (n, 4) ADD(m)
    actual = D['add'][idx]
    eff = np.zeros(len(idx))
    for k in range(len(idx)):
        eff[k] = np.interp(actual[k], curves[k], sig, left=0.0, right=np.nan)
        if np.isnan(eff[k]):
            sl = (curves[k, -1] - curves[k, -2]) / (sig[-1] - sig[-2])
            eff[k] = sig[-1] + (actual[k] - curves[k, -1]) / max(sl, 1e-9)
    is_e = np.isin(idx, np.where(exc)[0]); is_m = np.isin(idx, np.where(med)[0])
    print(f'  유효 σ (프레임 기하를 감안한 등가 2D 노이즈):')
    print(f'    excellent median = {np.median(eff[is_e]):.2f} px')
    print(f'    mediocre  median = {np.median(eff[is_m]):.2f} px   (배율 {np.median(eff[is_m])/np.median(eff[is_e]):.2f}x)')
    print(f'  같은 σ에서의 기하 민감도 배율(=순수 기하 탓) = '
          f'{np.median(S["k_sens"][is_m])/np.median(S["k_sens"][is_e]):.2f}x')
    print(f'  실제 ADD 배율 = {np.median(add_mm[idx][is_m])/np.median(add_mm[idx][is_e]):.2f}x')
    print('  => ADD 배율 ≈ (기하 민감도 배율) x (유효 입력오차 배율). 두 몫의 분해가 곧 (b) vs (a).')

    print()
    print('=' * 78)
    print('G. Kabsch-흡수 후 관절별 귀책 (솔버가 rigid로 못 숨기는 순수 shape 오차, mm)')
    print('=' * 78)
    fk_gt = panda_forward_kinematics(torch.tensor(gtth)).numpy()
    res = {}
    for jj in range(6):
        t = gtth.copy(); t[:, jj] = prth[:, jj]
        fk = panda_forward_kinematics(torch.tensor(t)).numpy()
        r = np.zeros(len(fids))
        for k in range(len(fids)):
            R, tt = kabsch_Rt(fk[k], fk_gt[k])
            r[k] = np.linalg.norm((R @ fk[k].T).T + tt - fk_gt[k], axis=1).mean() * 1000
        res[jj] = r
    print(f'{"band":14s}' + ''.join(f'{f"J{j+1}":>9s}' for j in range(6)))
    for nm, m in [('excellent', exc), ('mediocre', med)]:
        print(f'{nm:14s}' + ''.join(f'{np.median(res[j][m]):9.2f}' for j in range(6)))
    print(f'{"Δ(med-exc)":14s}' + ''.join(
        f'{np.median(res[j][med])-np.median(res[j][exc]):9.2f}' for j in range(6)))
    tot = sum(np.median(res[j][med]) - np.median(res[j][exc]) for j in range(6))
    print(f'  Δ 합계 {tot:.2f}mm 중 관절별 점유율: ' + ', '.join(
        f'J{j+1}={100*(np.median(res[j][med])-np.median(res[j][exc]))/tot:.0f}%' for j in range(6)))

    print()
    print('=' * 78)
    print('H. 기하 트리거 -> 관절 오차 연결 (전 clean 프레임, spearman-ish 상관)')
    print('=' * 78)
    def rk(x):
        o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r
    trig = ['forearm_view_cos', 'upperarm_view_cos', 'bbox_diag_px', 'z_mean', 'selfocc_ratio',
            'fore_dist', 'kp2d_min_dist']
    print(f'{"trigger":22s}' + ''.join(f'{f"J{j+1}":>8s}' for j in range(6)) + f'{"px_err":>9s}')
    for t in trig:
        x = rk(D[t][clean])
        row = ''.join(f'{np.corrcoef(x, rk(D[f"aerr_J{j+1}"][clean]))[0,1]:+8.2f}' for j in range(6))
        row += f'{np.corrcoef(x, rk(D["px_err_mean"][clean]))[0,1]:+9.2f}'
        print(f'{t:22s}{row}')

    # 전완 시선정렬 상위/하위 분위별
    print('\n  forearm_view_cos 4분위별 (clean): mediocre비율 / J4오차 / J2오차 / px_err')
    v = D['forearm_view_cos'][clean]; qs = np.quantile(v, [0, .25, .5, .75, 1.0])
    for q in range(4):
        m0 = (v >= qs[q]) & (v <= qs[q + 1])
        sub = np.where(clean)[0][m0]
        mm = med[sub];
        print(f'    cos∈[{qs[q]:.2f},{qs[q+1]:.2f}]  mediocre={mm.mean()*100:5.1f}%  '
              f'J4={np.median(D["aerr_J4"][sub]):5.2f}°  J2={np.median(D["aerr_J2"][sub]):5.2f}°  '
              f'px={np.median(D["px_err_mean"][sub]):5.2f}  ADD={np.median(add_mm[sub]):5.1f}mm')

    # ---------- I. 시각화 ----------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('matplotlib 없음, 시각화 생략:', e); return
    os.makedirs(args.viz_dir, exist_ok=True)
    F_, CX, CY = 320.0, 320.0, 240.0
    proj = lambda X: np.stack([F_ * X[:, 0] / np.clip(X[:, 2], 1e-3, None) + CX,
                               F_ * X[:, 1] / np.clip(X[:, 2], 1e-3, None) + CY], 1)
    chain = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

    med_i = np.where(med)[0][np.argsort(-add_mm[med])][:args.n_viz]
    exc_i = np.where(exc)[0][np.argsort(add_mm[exc])][:args.n_viz]
    for tag, ids in [('mediocre_worst', med_i), ('excellent_best', exc_i)]:
        n = len(ids)
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 7.2))
        for ax, i in zip(np.array(axes).ravel(), ids):
            ip = os.path.join(IMG_DIR, f'{fids[i]}.rgb.jpg')
            if os.path.exists(ip):
                ax.imshow(plt.imread(ip))
            ax.set_xlim(0, 640); ax.set_ylim(480, 0)
            g, p = gt2d[i], proj(pred3d[i])
            for a, b in chain:
                ax.plot([g[a, 0], g[b, 0]], [g[a, 1], g[b, 1]], '-', c='lime', lw=2.0, alpha=.9)
                ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], '-', c='red', lw=1.6, alpha=.9)
            ax.scatter(g[:, 0], g[:, 1], c='lime', s=22, zorder=5, edgecolors='k', linewidths=.4)
            ax.scatter(p[:, 0], p[:, 1], c='red', s=22, zorder=5, marker='x')
            ax.set_title(f'{fids[i]}  ADD={add_mm[i]:.0f}mm\n'
                         f'J2={D["aerr_J2"][i]:.1f}° J4={D["aerr_J4"][i]:.1f}° '
                         f'fcos={D["forearm_view_cos"][i]:.2f} px={D["px_err_mean"][i]:.1f}',
                         fontsize=8)
            ax.axis('off')
        for ax in np.array(axes).ravel()[len(ids):]:
            ax.axis('off')
        fig.suptitle(f'{tag}: GT (green) vs predicted reprojection (red)', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.viz_dir, f'{tag}.png'), dpi=110)
        plt.close(fig)

    # 요약 산점도
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    axs[0].scatter(D['forearm_view_cos'][exc], add_mm[exc], s=7, alpha=.4, label='excellent')
    axs[0].scatter(D['forearm_view_cos'][med], add_mm[med], s=10, alpha=.7, c='r', label='mediocre')
    axs[0].set_xlabel('|cos(forearm, view axis)|  (1 = forearm along view axis)'); axs[0].set_ylabel('ADD (mm)')
    axs[0].set_yscale('log'); axs[0].legend(); axs[0].set_title('Forearm foreshortening')
    axs[1].scatter(D['aerr_J4'][exc], add_mm[exc], s=7, alpha=.4)
    axs[1].scatter(D['aerr_J4'][med], add_mm[med], s=10, alpha=.7, c='r')
    axs[1].set_xlabel('J4 angle error (deg)'); axs[1].set_ylabel('ADD (mm)')
    axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_title('J4 error vs ADD')
    axs[2].scatter(D['err_tangent'][exc] * 1000, D['err_radial'][exc] * 1000, s=7, alpha=.4)
    axs[2].scatter(D['err_tangent'][med] * 1000, D['err_radial'][med] * 1000, s=10, alpha=.7, c='r')
    axs[2].plot([0, 120], [0, 120], 'k--', lw=.8)
    axs[2].set_xlabel('tangential (image-plane) error, mm'); axs[2].set_ylabel('radial (along-ray) error, mm')
    axs[2].set_title('Error is almost entirely along-ray (range)')
    fig.tight_layout(); fig.savefig(os.path.join(args.viz_dir, 'factors.png'), dpi=110)
    print(f'\n시각화 저장: {args.viz_dir}')


if __name__ == '__main__':
    main()
