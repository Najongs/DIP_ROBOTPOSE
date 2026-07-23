#!/usr/bin/env python
"""등가 입력오차 3.65px의 귀속: 검출기 2D 오차 vs 각도head θ 오차.

2x2 요인설계 — {검출2D, GT2D} x {head θ init, GT θ init} 를 동일한 미니 솔버로 풀어
관측된 mediocre 42.1mm 를 어느 쪽이 재현하는지 본다. (iii)셀이 실측을 재현하면 귀속 신뢰.

전제: dump v3 에 kp2d_full(검출 2D, full-frame IS) / gtkp2d(GT 2D, full-frame IS) / head_theta.
⚠️ dump 의 kp2d 는 CROP-IS 공간이라 gtkp2d 와 직접 비교 불가 — 반드시 kp2d_full 을 쓸 것.
crop IS 프레임 -> 원본 640x480 프레임 변환은 gtkp2d <-> GT projected_location 로 축별 1차
회귀(크롭이 축정렬 scale+translate 이므로 정확)해서 구한다.
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'TRAIN'))
from model_v4 import panda_forward_kinematics  # noqa: E402
from mediocre_band_probe2 import solve, R_to_d6, kabsch_Rt, KP, GT_DIR  # noqa: E402

AUC = lambda a: float(np.clip(1 - 10 * np.asarray(a), 0, 1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default=os.path.join(HERE, 'rc_dumps_oas', 'dr_pred_v3.npz'))
    args = ap.parse_args()

    V = np.load(args.dump, allow_pickle=True)
    print('dump 키:', sorted(V.files))
    for k in ('kp2d_full', 'gtkp2d', 'head_theta'):
        assert k in V.files, f'{k} 없음 — dump v2 재생성 필요'
    D = np.load(os.path.join(HERE, 'ablation_logs', 'mediocre_band.npz'), allow_pickle=True)

    vf = [str(x) for x in V['fid']]
    df = [str(x) for x in D['fid']]
    pos = {f: i for i, f in enumerate(vf)}
    order = np.array([pos[f] for f in df])            # v2 -> 기존 특징 순서로 정렬
    kp2d_c = V['kp2d_full'][order].astype(float)   # full-frame IS (gtkp2d 와 동일계)
    gtkp_c = V['gtkp2d'][order].astype(float)
    head_th = V['head_theta'][order].astype(float)
    kp_cam_v2 = V['kp_cam'][order].astype(float)
    gt3d = V['gt3d'][order].astype(float)

    add = D['add'] * 1000
    clean = D['n_offframe'] == 0
    exc = clean & (add < 30); med = clean & (add >= 30) & (add < 100)

    # 검증: v2 재현성
    add_v2 = np.linalg.norm(kp_cam_v2 - gt3d, axis=2).mean(1) * 1000
    print(f'\n[검증] v2 dump ADD median={np.median(add_v2):.1f}mm  AUC={AUC(add_v2/1000):.4f}'
          f'   (원 dump {np.median(add):.1f}mm / {AUC(D["add"]):.4f})')

    # GT 2D (원본 프레임)
    gt2d_o = np.zeros((len(df), 7, 2)); gtth = np.zeros((len(df), 7))
    for i, fid in enumerate(df):
        d = json.load(open(os.path.join(GT_DIR, f'{fid}.json')))
        kmap = {k['name']: k for k in d['objects'][0]['keypoints']}
        gt2d_o[i] = [kmap[n]['projected_location'] for n in KP]
        j = {x['name'].split('/')[-1]: x['position'] for x in d['sim_state']['joints']}
        gtth[i] = [j[f'panda_joint{k}'] for k in range(1, 8)]

    # crop IS 프레임 -> 원본 프레임 (축별 1차)
    det2d_o = np.zeros_like(gt2d_o); fit_res = np.zeros(len(df))
    for i in range(len(df)):
        for ax in range(2):
            A = np.stack([gtkp_c[i, :, ax], np.ones(7)], 1)
            coef, *_ = np.linalg.lstsq(A, gt2d_o[i, :, ax], rcond=None)
            det2d_o[i, :, ax] = kp2d_c[i, :, ax] * coef[0] + coef[1]
            fit_res[i] = max(fit_res[i], np.abs(A @ coef - gt2d_o[i, :, ax]).max())
    print(f'[검증] crop->원본 좌표 역변환 최대 잔차 = {np.median(fit_res):.4f}px (median), '
          f'{fit_res.max():.4f}px (max)  <- 0에 가까워야 정당')

    print()
    print('=' * 80)
    print('1. 검출기 2D 오차 (원본 640x480 프레임 기준, px)')
    print('=' * 80)
    e2d = np.linalg.norm(det2d_o - gt2d_o, axis=2)
    print(f'{"band":12s}{"mean_kp":>10s}{"median_kp":>11s}{"max_kp":>9s}{"p90":>8s}')
    for nm, m in [('excellent', exc), ('mediocre', med), ('clean', clean)]:
        print(f'{nm:12s}{np.median(e2d[m].mean(1)):10.2f}{np.median(np.median(e2d[m],1)):11.2f}'
              f'{np.median(e2d[m].max(1)):9.2f}{np.percentile(e2d[m].mean(1),90):8.2f}')
    print(f'  배율(mediocre/excellent) = '
          f'{np.median(e2d[med].mean(1))/np.median(e2d[exc].mean(1)):.2f}x')

    print()
    print('=' * 80)
    print('2. solve 이전 head θ 오차 (deg)')
    print('=' * 80)
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    ah = np.degrees(np.abs(wrap(head_th[:, :6] - gtth[:, :6])))
    print(f'{"band":12s}' + ''.join(f'{f"J{j+1}":>8s}' for j in range(6)) + f'{"J1-J6":>9s}')
    for nm, m in [('excellent', exc), ('mediocre', med), ('clean', clean)]:
        print(f'{nm:12s}' + ''.join(f'{np.median(ah[m, j]):8.2f}' for j in range(6))
              + f'{ah[m].mean():9.2f}')
    print(f'  J4 배율(med/exc) = {np.median(ah[med,3])/np.median(ah[exc,3]):.2f}x   '
          f'전체 배율 = {np.median(ah[med].mean(1))/np.median(ah[exc].mean(1)):.2f}x')
    # solve 가 head θ 를 얼마나 고치는가
    ar = np.stack([D[f'aerr_J{j+1}'] for j in range(6)], 1)
    print(f'  head -> solve 후 변화 (median J1-J6 평균): '
          f'exc {np.median(ah[exc].mean(1)):.2f}° -> {np.median(ar[exc].mean(1)):.2f}°   '
          f'med {np.median(ah[med].mean(1)):.2f}° -> {np.median(ar[med].mean(1)):.2f}°')

    print()
    print('=' * 80)
    print('3. 2x2 요인설계 — {검출2D, GT2D} x {head θ, GT θ}  (동일 미니 솔버)')
    print('=' * 80)
    sel = exc | med
    idx = np.where(sel)[0]
    fk_gt = panda_forward_kinematics(torch.tensor(gtth[idx])).numpy()
    Rs, ts = [], []
    for k in range(len(idx)):
        R, t = kabsch_Rt(fk_gt[k], gt3d[idx[k]]); Rs.append(R); ts.append(t)
    d6gt = R_to_d6(torch.tensor(np.stack(Rs))); tgt = torch.tensor(np.stack(ts))
    is_e = np.isin(idx, np.where(exc)[0]); is_m = np.isin(idx, np.where(med)[0])

    cells = {}
    for tag, t2d, th0 in [
            ('(iv) GT2D  + GTθ   [바닥]', gt2d_o, gtth),
            ('(ii) GT2D  + headθ', gt2d_o, head_th),
            ('(i)  검출2D + GTθ', det2d_o, gtth),
            ('(iii)검출2D + headθ [실측 대응]', det2d_o, head_th)]:
        th, cam, px = solve(torch.tensor(t2d[idx]), torch.tensor(th0[idx]).clone(),
                            d6gt.clone(), tgt.clone(), n_iter=400)
        a = np.linalg.norm(cam.numpy() - gt3d[idx], axis=2).mean(1)
        cells[tag] = a
        print(f'  {tag:34s} exc={np.median(a[is_e])*1000:6.1f}mm  '
              f'med={np.median(a[is_m])*1000:6.1f}mm  AUC(med)={AUC(a[is_m]):.4f}')
    print(f'  {"실측(배포 파이프라인)":34s} exc={np.median(add[idx][is_e]):6.1f}mm  '
          f'med={np.median(add[idx][is_m]):6.1f}mm  AUC(med)={AUC(D["add"][idx][is_m]):.4f}')

    print()
    base = np.median(cells['(iv) GT2D  + GTθ   [바닥]'][is_m]) * 1000
    full = np.median(cells['(iii)검출2D + headθ [실측 대응]'][is_m]) * 1000
    d_det = np.median(cells['(i)  검출2D + GTθ'][is_m]) * 1000 - base
    d_head = np.median(cells['(ii) GT2D  + headθ'][is_m]) * 1000 - base
    print(f'  mediocre 밴드 귀속 (바닥 {base:.1f}mm 대비 증분):')
    print(f'    검출기 2D 단독 기여 = {d_det:6.1f}mm')
    print(f'    head θ  단독 기여 = {d_head:6.1f}mm')
    print(f'    둘 다             = {full - base:6.1f}mm  (단독합 {d_det+d_head:.1f}mm '
          f'-> {"초가법(상호작용 有)" if full-base > d_det+d_head*1.1 else "가법에 가까움"})')
    tot = max(d_det + d_head, 1e-9)
    print(f'    => 비율: 검출기 {100*d_det/tot:.0f}%  /  head θ {100*d_head/tot:.0f}%')

    np.savez(os.path.join(HERE, 'ablation_logs', 'mediocre_attrib.npz'),
             idx=idx, e2d=e2d, head_aerr=ah, **{f'cell{i}': v for i, v in enumerate(cells.values())})


if __name__ == '__main__':
    main()
