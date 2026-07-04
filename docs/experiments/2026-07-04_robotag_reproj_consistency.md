# 2026-07-04 — RoboTAG식 cross-dim reproj consistency angle head 🔄

## 배경
RoboTAG(2025-11, arXiv:2511.07717) 검증: auto-bbox 동일 프로토콜에서 우리 mean 79.9 > RoboTAG 74.0 (3/4 승). 유일한 패 = **azure(RoboTAG 83.1 vs 우리 79.2)**. RoboTAG azure 우위의 원천 = closed-loop 2D-3D 일관성 손실(`ℒalign=α₁‖p³−p²‖²+α₂‖κ₃−κ₂‖²+α₃‖κ₃−κ_fk‖²`) — 예측 3D를 2D-lift와 FK 양쪽에 정렬.

## 가설
우리 angle head는 sin/cos + **robot-frame** FK loss만 씀 → 카메라 프레임/2D 투영 일관성 없음. 데이터셋의 카메라 프레임 GT 키포인트로 GT 포즈를 구해 **FK(pred)를 GT 포즈로 재투영→GT 2D 정렬**하는 항 추가 = RoboTAG cross-dim 정렬의 충실한 이식(train_3d_v4가 썼으나 배포 head엔 없던 항). 근거리(azure)에서 작은 각도 오차가 2D를 크게 움직이므로 이 신호가 각도를 날카롭게 할 것으로 기대.

## 설계 (`train_angle.py --reproj-weight`)
- GT 포즈 Rg,tg = Kabsch(FK(gt_angles), keypoints_3d[camera])
- proj = project(Rg·FK(pred_angles)+tg, K), Huber(proj/S, gt2d/S) × valid_mask
- backbone/detector 동결, angle head만. warm-start = 배포 crop head. w∈{50,150}, 18ep.

## 검증 계획
1. 배포 파이프라인에서 새 angle head로 4-split ADD (특히 azure) — 기준 azure 0.7916(+DARK)
2. 전 카메라 do-no-harm

## 결과
(학습 완료 후 기입)
