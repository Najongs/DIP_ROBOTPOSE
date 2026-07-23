#!/usr/bin/env python
"""Panda synth-DR "good-but-mediocre" 프레임 특성화 (오프라인, CPU 전용).

목적: clean(전 키포인트 in-frame) 프레임 중 ADD 30-100mm 중간대(mediocre)가
excellent(<30mm)와 무엇이 다른지 정량화하고, 각 요인이 (a) 고칠 수 있는 추출 실패인지
(b) 진짜 관측성 한계인지 판정한다.

입력: Eval/rc_dumps_oas/{dr_pred,dr_oracle}.npz + panda_synth_test_dr/*.json (GT)
출력: stdout 리포트 + (옵션) --viz-dir 에 오버레이 이미지
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'TRAIN'))
import torch
from model_v4 import panda_forward_kinematics  # noqa: E402

KP = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4',
      'panda_link6', 'panda_link7', 'panda_hand']
KP_SHORT = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
# 연결(운동학 체인 상 인접) — foreshortening 측정용
SEGS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
W, H = 640, 480
F = 320.0
CX, CY = 320.0, 240.0
GT_DIR = os.path.join(HERE, '..', 'Dataset', 'Converted_dataset',
                      'DREAM_to_DREAM_syn', 'panda_synth_test_dr')


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def kabsch(P, Q):
    """P,Q: (N,3). P를 Q에 맞추는 최적 rigid (R,t). 반환: 정렬된 P."""
    pc, qc = P.mean(0), Q.mean(0)
    Pc, Qc = P - pc, Q - qc
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return (R @ Pc.T).T + qc


def load_gt(fids):
    """GT json에서 2D/3D 키포인트, 관절각, distractor bbox 로드."""
    out = {}
    for fid in fids:
        p = os.path.join(GT_DIR, f'{fid}.json')
        d = json.load(open(p))
        obj = d['objects'][0]
        assert obj['class'] == 'panda'
        kmap = {k['name']: k for k in obj['keypoints']}
        kp2d = np.array([kmap[n]['projected_location'] for n in KP], float)
        kp3d = np.array([kmap[n]['location'] for n in KP], float) / 100.0  # cm -> m
        joints = {j['name'].split('/')[-1]: j['position'] for j in d['sim_state']['joints']}
        th = np.array([joints[f'panda_joint{i}'] for i in range(1, 8)], float)
        dis = []
        for o in d['objects'][1:]:
            bb = o['bounding_box']
            dis.append((bb['min'][0], bb['min'][1], bb['max'][0], bb['max'][1],
                        o['location'][2] / 100.0))
        out[fid] = dict(kp2d=kp2d, kp3d=kp3d, theta=th, distractors=np.array(dis, float),
                        img=d['meta']['image_path'])
    return out


def frame_features(pred3d, gt3d, gt2d, th_pred, th_gt, distractors, reproj):
    f = {}
    err = np.linalg.norm(pred3d - gt3d, axis=1)          # (7,) m
    f['add'] = err.mean()
    for i, n in enumerate(KP_SHORT):
        f[f'err_{n}'] = err[i]

    # --- ADD 분해: shape(관절각) vs pose(rigid) ---
    aligned = kabsch(pred3d, gt3d)
    f['add_shape'] = np.linalg.norm(aligned - gt3d, axis=1).mean()
    f['add_pose'] = f['add'] - f['add_shape']

    # --- 오차의 radial(시선방향=깊이) vs tangential(영상면) 분해 ---
    ray = gt3d / np.linalg.norm(gt3d, axis=1, keepdims=True)
    e = pred3d - gt3d
    rad = np.abs((e * ray).sum(1))
    tan = np.linalg.norm(e - (e * ray).sum(1, keepdims=True) * ray, axis=1)
    f['err_radial'] = rad.mean()
    f['err_tangent'] = tan.mean()

    # --- 예측 3D를 재투영해 GT 2D와 비교 (검출/솔버 2D 정합) ---
    zp = np.clip(pred3d[:, 2], 1e-3, None)
    p2d = np.stack([F * pred3d[:, 0] / zp + CX, F * pred3d[:, 1] / zp + CY], 1)
    f['px_err_mean'] = np.linalg.norm(p2d - gt2d, axis=1).mean()
    f['px_err_max'] = np.linalg.norm(p2d - gt2d, axis=1).max()
    f['reproj_solver'] = reproj

    # --- 관절각 오차 ---
    ae = np.abs(wrap(th_pred[:6] - th_gt[:6]))
    for i in range(6):
        f[f'aerr_J{i+1}'] = np.degrees(ae[i])
    f['aerr_mean'] = np.degrees(ae.mean())
    f['aerr_prox'] = np.degrees(ae[:4].mean())   # J1-J4
    f['aerr_dist'] = np.degrees(ae[4:].mean())   # J5-J6
    for i in range(6):
        f[f'gtJ{i+1}'] = np.degrees(th_gt[i])

    # --- 거리 / 크기 ---
    f['z_base'] = gt3d[0, 2]
    f['z_mean'] = gt3d[:, 2].mean()
    f['z_spread'] = gt3d[:, 2].max() - gt3d[:, 2].min()
    x0, y0 = gt2d.min(0)
    x1, y1 = gt2d.max(0)
    f['bbox_diag_px'] = float(np.hypot(x1 - x0, y1 - y0))
    f['bbox_area_px'] = float((x1 - x0) * (y1 - y0))
    # 화면 안쪽으로 clip한 실제 가시 bbox
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x1, W), min(y1, H)
    f['vis_frac'] = float(max(0, cx1 - cx0) * max(0, cy1 - cy0) /
                          max(1e-6, (x1 - x0) * (y1 - y0)))
    f['n_offframe'] = int(((gt2d[:, 0] < 0) | (gt2d[:, 0] >= W) |
                           (gt2d[:, 1] < 0) | (gt2d[:, 1] >= H)).sum())
    # 경계까지 최소거리 (음수면 화면 밖)
    f['min_border_px'] = float(np.minimum.reduce([gt2d[:, 0], gt2d[:, 1],
                                                  W - gt2d[:, 0], H - gt2d[:, 1]]).min())

    # --- foreshortening: 세그먼트별 (2D 길이) / (수직일 때 기대 2D 길이) ---
    ratios, l3ds, l2ds = [], [], []
    for a, b in SEGS:
        l3 = np.linalg.norm(gt3d[a] - gt3d[b])
        zm = 0.5 * (gt3d[a, 2] + gt3d[b, 2])
        exp2 = F * l3 / zm
        act2 = np.linalg.norm(gt2d[a] - gt2d[b])
        r = act2 / max(exp2, 1e-6)
        ratios.append(min(r, 1.0))
        l3ds.append(l3); l2ds.append(act2)
    ratios = np.array(ratios)
    f['fore_min'] = ratios.min()
    f['fore_mean'] = ratios.mean()
    f['fore_prox'] = ratios[:3].mean()     # link0..link4 (근위)
    f['fore_dist'] = ratios[3:].mean()     # link4..hand (원위)
    for i, (a, b) in enumerate(SEGS):
        f[f'fore_s{i}'] = ratios[i]
    # 링크당 픽셀 (겉보기 크기)
    f['px_per_link'] = float(np.sum(l2ds) / max(1e-6, np.sum(l3ds)) * 0.001)  # px per mm

    # --- 팔 축 vs 시선 정렬 ---
    v = gt3d[6] - gt3d[0]
    v = v / max(np.linalg.norm(v), 1e-9)
    c = gt3d.mean(0)
    c = c / max(np.linalg.norm(c), 1e-9)
    f['arm_view_cos'] = float(abs(np.dot(v, c)))
    fa = gt3d[4] - gt3d[2]  # 전완(link3->link6)
    fa = fa / max(np.linalg.norm(fa), 1e-9)
    f['forearm_view_cos'] = float(abs(np.dot(fa, c)))
    ua = gt3d[2] - gt3d[0]  # 상완(link0->link3)
    ua = ua / max(np.linalg.norm(ua), 1e-9)
    f['upperarm_view_cos'] = float(abs(np.dot(ua, c)))

    # --- 2D 혼잡도 (키포인트끼리 겹침 = 연관 모호) ---
    d2 = np.linalg.norm(gt2d[:, None, :] - gt2d[None, :, :], axis=2)
    iu = np.triu_indices(7, 1)
    f['kp2d_min_dist'] = float(d2[iu].min())
    # 3D로는 멀지만 2D로는 붙은 쌍 (진짜 자기가림 지표)
    d3 = np.linalg.norm(gt3d[:, None, :] - gt3d[None, :, :], axis=2)
    m = d3[iu] > 0.15
    f['selfocc_ratio'] = float((d2[iu][m] / (F * d3[iu][m] / gt3d[:, 2].mean())).min()) if m.any() else 1.0

    # --- distractor 가림 ---
    n_occ = 0
    if len(distractors):
        for i in range(7):
            u, v2 = gt2d[i]
            z = gt3d[i, 2]
            for (bx0, by0, bx1, by1, dz) in distractors:
                if bx0 <= u <= bx1 and by0 <= v2 <= by1 and dz < z:
                    n_occ += 1
                    break
    f['n_kp_occluded'] = n_occ
    return f


def auroc(x, y):
    """y=1(mediocre) 판별력. 0.5=무정보."""
    x = np.asarray(x, float); y = np.asarray(y, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    order = np.argsort(x)
    ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default=os.path.join(HERE, 'rc_dumps_oas', 'dr_pred.npz'))
    ap.add_argument('--oracle', default=os.path.join(HERE, 'rc_dumps_oas', 'dr_oracle.npz'))
    ap.add_argument('--out-npz', default=os.path.join(HERE, 'ablation_logs', 'mediocre_band.npz'))
    args = ap.parse_args()

    P = np.load(args.pred, allow_pickle=True)
    O = np.load(args.oracle, allow_pickle=True)
    fids = [str(x) for x in P['fid']]
    gt = load_gt(fids)

    rows = []
    for i, fid in enumerate(fids):
        g = gt[fid]
        r = frame_features(P['kp_cam'][i].astype(float), P['gt3d'][i].astype(float),
                           g['kp2d'], P['theta'][i].astype(float), g['theta'],
                           g['distractors'], float(P['reproj'][i]))
        # oracle-angle(GTθ, solved R,t) 대응 프레임 ADD
        eo = np.linalg.norm(O['kp_cam'][i].astype(float) - O['gt3d'][i].astype(float), axis=1)
        r['add_oracle'] = eo.mean()
        r['fid'] = fid
        rows.append(r)

    keys = [k for k in rows[0] if k != 'fid']
    D = {k: np.array([r[k] for r in rows], float) for k in keys}
    D_fid = np.array([r['fid'] for r in rows])

    np.savez(args.out_npz, fid=D_fid, **D)

    add_mm = D['add'] * 1000
    clean = D['n_offframe'] == 0
    auc = lambda a: np.clip(1 - 10 * a, 0, 1).mean()

    print('=' * 78)
    print('0. 밴드 정의 및 AUC 회계')
    print('=' * 78)
    print(f'전체 {len(add_mm)}f  AUC={auc(D["add"]):.4f}   oracle-angle AUC={auc(D["add_oracle"]):.4f}')
    print(f'clean(전 kp in-frame) {clean.sum()}f ({clean.mean()*100:.1f}%)  '
          f'AUC={auc(D["add"][clean]):.4f}  oracle={auc(D["add_oracle"][clean]):.4f}')
    exc = clean & (add_mm < 30)
    med = clean & (add_mm >= 30) & (add_mm < 100)
    tail = clean & (add_mm >= 100)
    for nm, m in [('excellent <30mm', exc), ('mediocre 30-100mm', med), ('clean-tail >100mm', tail)]:
        contrib = np.clip(1 - 10 * D['add'][m], 0, 1).sum() / len(add_mm)
        lost = m.sum() / len(add_mm) - contrib
        print(f'  {nm:20s} n={m.sum():4d} ({m.sum()/len(add_mm)*100:5.1f}%)  '
              f'median={np.median(add_mm[m]):6.1f}mm  AUC기여={contrib:.4f}  '
              f'상실AUC={lost:.4f}')
    print(f'\n  => mediocre 밴드가 잃는 AUC = {(med.sum()/len(add_mm) - np.clip(1-10*D["add"][med],0,1).sum()/len(add_mm)):.4f}')

    print()
    print('=' * 78)
    print('1. ADD 분해 (shape=관절각 / pose=rigid, radial=깊이 / tangent=영상면)')
    print('=' * 78)
    hdr = f'{"band":20s}{"ADD":>8s}{"shape":>8s}{"pose":>8s}{"radial":>8s}{"tangent":>8s}{"px_err":>8s}{"reproj":>8s}'
    print(hdr)
    for nm, m in [('excellent', exc), ('mediocre', med), ('clean-tail', tail)]:
        print(f'{nm:20s}' + ''.join(f'{np.median(D[k][m])*(1000 if k.startswith(("add","err")) else 1):>8.1f}'
              for k in ['add', 'add_shape', 'add_pose', 'err_radial', 'err_tangent',
                        'px_err_mean', 'reproj_solver']))
    print('  (단위: mm, px_err/reproj는 px; 전부 median)')

    print()
    print('=' * 78)
    print('2. 판별력 랭킹 (mediocre vs excellent, clean 프레임 내) — AUROC')
    print('=' * 78)
    y = med[clean | True]
    sel = exc | med
    cand = [k for k in keys if not k.startswith(('err_', 'add')) ]
    res = []
    for k in cand:
        a = auroc(D[k][sel], med[sel])
        res.append((abs(a - 0.5), a, k))
    res.sort(reverse=True)
    print(f'{"factor":22s}{"AUROC":>8s}{"|Δ|":>7s}{"exc_med":>10s}{"med_med":>10s}')
    for _, a, k in res[:34]:
        print(f'{k:22s}{a:8.3f}{abs(a-0.5):7.3f}{np.median(D[k][exc]):10.3f}{np.median(D[k][med]):10.3f}')

    print()
    print('=' * 78)
    print('3. 관절별 각도 오차 (deg, median) — 밴드별')
    print('=' * 78)
    print(f'{"band":14s}' + ''.join(f'{f"J{i+1}":>8s}' for i in range(6)) + f'{"prox":>8s}{"dist":>8s}')
    for nm, m in [('excellent', exc), ('mediocre', med), ('clean-tail', tail), ('ALL clean', clean)]:
        print(f'{nm:14s}' + ''.join(f'{np.median(D[f"aerr_J{i+1}"][m]):8.2f}' for i in range(6))
              + f'{np.median(D["aerr_prox"][m]):8.2f}{np.median(D["aerr_dist"][m]):8.2f}')
    print('  RoboPEPP Tab.3 DR:  J1 4.9  J2 2.3  J3 2.7  J4 2.2  J5 4.9  J6 5.4')

    print()
    print('=' * 78)
    print('4. 관절별 오차 -> ADD 기여 (FK 반사실: GTθ에서 관절 j만 예측값으로 교체)')
    print('=' * 78)
    th_gt = np.stack([gt[f]['theta'] for f in fids])
    th_pr = P['theta'].astype(float)
    base = panda_forward_kinematics(torch.tensor(th_gt)).numpy()
    contrib = {}
    for j in range(6):
        t = th_gt.copy(); t[:, j] = th_pr[:, j]
        fk = panda_forward_kinematics(torch.tensor(t)).numpy()
        contrib[j] = np.linalg.norm(fk - base, axis=2).mean(1) * 1000
    t = th_pr.copy(); t[:, 6] = th_gt[:, 6]
    fk_all = panda_forward_kinematics(torch.tensor(t)).numpy()
    all_c = np.linalg.norm(fk_all - base, axis=2).mean(1) * 1000
    print(f'{"band":14s}' + ''.join(f'{f"J{j+1}":>8s}' for j in range(6)) + f'{"ALL":>9s}')
    for nm, m in [('excellent', exc), ('mediocre', med), ('clean-tail', tail)]:
        print(f'{nm:14s}' + ''.join(f'{np.median(contrib[j][m]):8.1f}' for j in range(6))
              + f'{np.median(all_c[m]):9.1f}')
    print('  (mm: 그 관절 하나만 틀렸을 때의 ADD. shape 오차의 관절별 배분)')

    print()
    print('=' * 78)
    print('5. 모호성 검정 — 예측이 2D는 맞추는데 3D가 틀리는가?')
    print('=' * 78)
    for nm, m in [('excellent', exc), ('mediocre', med), ('clean-tail', tail)]:
        px = D['px_err_mean'][m]
        print(f'{nm:12s} n={m.sum():4d}  pred재투영 px_err median={np.median(px):6.2f}  '
              f'<3px 비율={np.mean(px < 3)*100:5.1f}%  <5px={np.mean(px < 5)*100:5.1f}%  '
              f'radial/총={np.median(D["err_radial"][m]/np.maximum(D["add"][m],1e-9)):.2f}')
    print('  판독: mediocre가 2D는 맞추면서(px_err 작음) 3D만 틀리면 = 깊이/모호성(관측성)')
    print('        px_err가 크면 = 검출/솔버 추출 실패(고칠 수 있음)')

    print()
    print('=' * 78)
    print('6. GT 관절 configuration 의존성 (mediocre 비율, 관절값 4분위)')
    print('=' * 78)
    for j in range(6):
        v = D[f'gtJ{j+1}'][sel]
        yy = med[sel]
        qs = np.quantile(v, [0, .25, .5, .75, 1])
        line = f'J{j+1} '
        for q in range(4):
            mm = (v >= qs[q]) & (v <= qs[q + 1])
            line += f'  [{qs[q]:7.1f},{qs[q+1]:7.1f}]={yy[mm].mean()*100:5.1f}%'
        print(line)
    print(f'  (전체 mediocre 비율 = {med[sel].mean()*100:.1f}%)')

    print()
    print('=' * 78)
    print('7. 상위 요인 조합 — mediocre 비율 (오즈)')
    print('=' * 78)
    top = [k for _, _, k in res[:6]]
    for k in top:
        v = D[k][sel]
        thr = np.median(v)
        lo, hi = v < thr, v >= thr
        print(f'{k:22s} <{thr:8.3f}: {med[sel][lo].mean()*100:5.1f}%   '
              f'>={thr:8.3f}: {med[sel][hi].mean()*100:5.1f}%')

    print()
    print(f'특징 저장: {args.out_npz}')


if __name__ == '__main__':
    main()
