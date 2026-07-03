# 2026-07-04 — DARK sub-pixel 디코딩 (서베이 2R Idea 3) ✅ 채택

## 가설
orb −0.010 격차의 근본 = 작은/먼 로봇의 2D 키포인트 정밀도. DARK(arXiv:1910.06278)는 히트맵을 가우시안 변조 후 log-heatmap 1·2차 미분으로 argmax를 sub-pixel 보정 — 저해상도/작은 타깃에 특화, **학습 불필요**.

## 구현
`Eval/decode_util.py::dark_decode` (Taylor 보정, 1px 클램프, NaN 방어). `selfbbox_eval.py --dark-decode`가 soft-argmax 대체.

## 결과 — pose stage 매칭 A/B (300f, held-out)
| 카메라 | base(soft-argmax) | DARK | Δ | mean ADD |
|---|---|---|---|---|
| orb | 0.7196 | **0.7270** | **+0.0074** | 54.8→49.6mm |
| azure | 0.7948 | **0.7993** | **+0.0045** | 24.8→24.3mm |
| realsense | 0.7467 | **0.7517** | **+0.0050** | 25.5→24.8mm |

**전 카메라 양수 + do-no-harm.** mean ADD 감소 = 꼬리(작은/먼 프레임) 개선, DARK 저해상도 강점과 일치. cov-PnP에 이은 2번째 무료 채택 레버.

## 다음: DARK + RC 스택 → 최종 배포 수치 (진행 중)
