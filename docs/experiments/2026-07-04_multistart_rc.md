# 2026-07-04 — 멀티스타트 RC + SAM-IoU basin 선택 🔄

## 가설
남은 실패(orb 잔여 −0.010, 40% 가림 붕괴)의 공통 근원 = **wrong basin**: init 포즈가 틀린 회전 분지에 있으면 보수적 refine이 탈출 불가. 6월 MCL 반증의 원인은 *학습 selector*였고, 지금은 **SAM-IoU라는 외부 증거 선택기**가 있으므로 멀티가설이 성립.

## 설계 (`rc_refine_from_dump.py --multi-start`)
- 가설 군: init 카메라 포즈의 **base-Z축 회전 섭동** δ∈{0,±30°,±60°} — 2D 키포인트가 구분 못 하는 게이지 방향(단축/base-yaw 모호성), 실루엣은 구분 가능
- 각 가설 독립 refine → 최종 SAM-IoU 최고 채택, 원 가설 우선 마진 0.01 (do-no-harm)

## 검증 매트릭스 (실행 중)
| 대상 | 기준 | 게이트 |
|---|---|---|
| 40% 가림 (synth_photo 200) | 0.328 | ≥ 0.351 (RoboPEPP) |
| orb held-out 200 | 0.7607 (+RC@512) | +0.01 |
| 클린 realsense 200 | 0.8158 (+RC@448) | do-no-harm |

## 결과
(실행 완료 후 기입)
