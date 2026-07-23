#!/usr/bin/env python
"""(1) J1 제외 MAE vs RoboPEPP 동일기준 비교, (2) 꼬리 대 중앙값 프레이밍 정량화.
RoboPEPP 기준 확인됨: panda J1-J6만 회귀(test.py:201 gt_joints[:,:6]), joint7=0으로 FK
(test.py:216), 키포인트 7개 동일(datasets/dream.py:138) -> 우리와 완전 동일 기준.
Table 3 = per-joint MEAN absolute error (criterion_l1(...).mean(dim=0)).
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'TRAIN'))

RP_DR = np.array([4.9, 2.3, 2.7, 2.2, 4.9, 5.4])      # RoboPEPP Tab.3 panda DR, per-joint MEAN deg
RP_PH = np.array([4.4, 2.0, 2.3, 1.9, 4.2, 4.5])      # Photo (참고, 평균 3.2)


def main():
    D = np.load(os.path.join(HERE, 'ablation_logs', 'mediocre_band.npz'), allow_pickle=True)
    add = D['add'] * 1000
    clean = D['n_offframe'] == 0
    exc = clean & (add < 30); med = clean & (add >= 30) & (add < 100)
    A = np.stack([D[f'aerr_J{j + 1}'] for j in range(6)], 1)     # (N,6) deg

    print('=' * 84)
    print('1. J1 제외 MAE — RoboPEPP와 동일 기준(J1-J6 per-joint MEAN, joint7=0, kp 7개)')
    print('=' * 84)
    print(f'{"set":24s}' + ''.join(f'{f"J{j+1}":>7s}' for j in range(6))
          + f'{"J1-J6":>9s}{"J2-J6":>9s}')
    rows = [('우리 ALL (1000f)', np.ones(len(add), bool)),
            ('우리 clean (859f)', clean),
            ('우리 excellent (673f)', exc),
            ('우리 mediocre (140f)', med)]
    for nm, m in rows:
        mu = A[m].mean(0)
        print(f'{nm:24s}' + ''.join(f'{x:7.2f}' for x in mu) + f'{mu.mean():9.2f}{mu[1:].mean():9.2f}')
    print(f'{"RoboPEPP DR (Tab.3)":24s}' + ''.join(f'{x:7.2f}' for x in RP_DR)
          + f'{RP_DR.mean():9.2f}{RP_DR[1:].mean():9.2f}')
    print()
    ours_all, ours_cl = A.mean(0), A[clean].mean(0)
    print(f'  격차(J1 포함): ALL {ours_all.mean():.2f}° vs {RP_DR.mean():.2f}° = {ours_all.mean()/RP_DR.mean():.2f}x'
          f'   |  clean {ours_cl.mean():.2f}° = {ours_cl.mean()/RP_DR.mean():.2f}x')
    print(f'  격차(J1 제외): ALL {ours_all[1:].mean():.2f}° vs {RP_DR[1:].mean():.2f}° = '
          f'{ours_all[1:].mean()/RP_DR[1:].mean():.2f}x'
          f'   |  clean {ours_cl[1:].mean():.2f}° = {ours_cl[1:].mean()/RP_DR[1:].mean():.2f}x')
    print('  => J1 제외는 격차를 거의 못 줄인다(우리 J1이 특별히 나쁜 게 아님). '
          '"J1 때문에 부풀려졌다"는 가설은 기각.')
    print('  => 단 J1 오차의 ADD 기여는 정확히 0.00mm(게이지 DOF, Kabsch 흡수) 이므로 '
          'MAE는 어느 쪽이든 ADD와 부분적으로만 연결된다.')

    print()
    print('=' * 84)
    print('2. 꼬리 대 중앙값 — 우리 각도오차 분포는 "이동"이 아니라 "무거운 꼬리"인가?')
    print('=' * 84)
    print(f'{"joint":8s}{"median":>9s}{"mean":>9s}{"mean/med":>10s}{"p90":>9s}{"p99":>9s}'
          f'{"RoboPEPP":>10s}{"med<=RP?":>10s}')
    for j in range(6):
        v = A[clean, j]
        print(f'J{j+1:<7d}{np.median(v):9.2f}{v.mean():9.2f}{v.mean()/np.median(v):10.2f}'
              f'{np.percentile(v,90):9.2f}{np.percentile(v,99):9.2f}{RP_DR[j]:10.2f}'
              f'{"YES" if np.median(v)<=RP_DR[j] else "no":>10s}')
    print('  (clean 프레임 기준. mean/med >> 1 이면 꼬리 지배)')

    print()
    print('  꼬리 절단 반사실 (clean, per-joint MEAN deg):')
    for q in [100, 95, 90, 75]:
        cl = np.stack([np.minimum(A[clean, j], np.percentile(A[clean, j], q)) for j in range(6)], 1)
        mu = cl.mean(0)
        print(f'    상위 {100-q:2d}% 프레임을 p{q}로 클립: J1-J6 평균 = {mu.mean():5.2f}°  '
              f'(J2-J6 {mu[1:].mean():5.2f}°)   RoboPEPP {RP_DR.mean():.2f}°')
    print('  => 소수 꼬리 프레임만 억제해도 RoboPEPP 수준에 도달하는지 확인')

    print()
    print('  기여 집중도 — 전체 MEAN MAE 중 최악 프레임들이 차지하는 몫 (clean):')
    for j in [1, 3, 4, 5]:
        v = A[clean, j]; o = np.sort(v)[::-1]
        for frac in [0.05, 0.10]:
            k = int(len(v) * frac)
            print(f'    J{j+1}: 최악 {frac*100:3.0f}% 프레임이 MAE의 {o[:k].sum()/v.sum()*100:4.1f}% 차지', end='')
        print()

    print()
    print('=' * 84)
    print('3. 꼬리 프레임 ∩ (작음·단축·mediocre) 겹침')
    print('=' * 84)
    small = D['bbox_diag_px'] < np.quantile(D['bbox_diag_px'][clean], .5)
    fore = D['forearm_view_cos'] > np.quantile(D['forearm_view_cos'][clean], .5)
    print(f'{"tail 정의":26s}{"n":>6s}{"작음%":>8s}{"단축%":>8s}{"둘다%":>8s}{"mediocre%":>11s}'
          f'{"ADD중앙":>9s}')
    base_small = small[clean].mean() * 100; base_fore = fore[clean].mean() * 100
    base_both = (small & fore)[clean].mean() * 100
    for nm, jj in [('J4 오차 상위10%', 3), ('J2 오차 상위10%', 1),
                   ('J5 오차 상위10%', 4), ('J6 오차 상위10%', 5)]:
        thr = np.percentile(A[clean, jj], 90)
        m = clean & (A[:, jj] >= thr)
        print(f'{nm:26s}{m.sum():6d}{small[m].mean()*100:8.1f}{fore[m].mean()*100:8.1f}'
              f'{(small&fore)[m].mean()*100:8.1f}{med[m].mean()*100:11.1f}{np.median(add[m]):9.1f}')
    m = clean & (A[:, [1, 3, 4, 5]].max(1) >= np.percentile(A[clean][:, [1, 3, 4, 5]].max(1), 90))
    print(f'{"J2/J4/J5/J6 최대 상위10%":26s}{m.sum():6d}{small[m].mean()*100:8.1f}{fore[m].mean()*100:8.1f}'
          f'{(small&fore)[m].mean()*100:8.1f}{med[m].mean()*100:11.1f}{np.median(add[m]):9.1f}')
    print(f'{"(clean 전체 기저율)":26s}{clean.sum():6d}{base_small:8.1f}{base_fore:8.1f}'
          f'{base_both:8.1f}{med[clean].mean()*100:11.1f}{np.median(add[clean]):9.1f}')
    print('  => 꼬리 프레임이 작음/단축에 얼마나 편중되는지 = 요인이 꼬리를 설명하는 정도')


if __name__ == '__main__':
    main()
