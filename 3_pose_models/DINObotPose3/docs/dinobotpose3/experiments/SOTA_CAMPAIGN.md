# SOTA 캠페인 추적 (2026-07-22~)

목표: DREAM 전 로봇(Panda/KUKA/Baxter) × 도메인(real/synth-DR/synth-Photo)에서 **경쟁 모델 대비 SOTA-근접** 모델. 로봇별 가중치·FK는 달라도 **알고리즘은 동일**. 진단 완료 후 논문.

> ⚠️ **목표치 원칙 (2026-07-22 사용자 교정)**: 성능 목표는 **경쟁 모델 대비 DR/Photo/real ADD-AUC**로 잡는다. oracle-angle(0.899 등)은 *진단용 내부 천장*이지 목표가 아니다. 아래 §1의 경쟁 수치가 실제 게이트.

## §1. 경쟁 대비 표준 (Protocol A, ADD-AUC, full test set) — reference 에이전트가 채움

| 로봇/도메인 | Ours (base) | Ours (RC/배포) | RoboPEPP | HoRoPose | RoboPose | RoboTAG | DREAM | 우리 위치 |
|---|---|---|---|---|---|---|---|---|
| Panda real (mean 4cam) | — | **0.804** | 0.780 | ? | ? | ? | ? | 우위(vs PEPP) |
| Panda synth DR | 0.704 | **0.769** | ? | ? | ? | ? | ? | **미확정 ← 핵심** |
| Panda synth Photo | 0.738 | **0.799** | ? | ? | ? | ? | ? | **미확정 ← 핵심** |
| KUKA synth DR | 0.690 | ~0.72 | 0.762 | 0.751 | 0.802 | ? | ? | **−0.07 열세** |
| Baxter synth DR | 0.713 | 0.713 | 0.344 | 0.588 | 0.327 | 0.588 | ? | **1위** |

(? = reference 에이전트가 논문/repo에서 채울 것. bbox·protocol 주의: Protocol A vs B 교차 금지, auto vs GT bbox 명시.)

## §2. 진행 실험 (이 캠페인) — 단장 유지

| 실험 | 목표 | 핵심 config | 게이트 지표 | baseline | 결과 | GPU | 상태 |
|---|---|---|---|---|---|---|---|
| cropasp_a43_res | Panda 2D 기하수정 | crop-aspect 1.333, unfreeze4, strong aug | good-frame ADD-AUC + distal tail율 | 0.788 / 6.48% | — | GPU0 | 🔬 학습중 |
| cropasp_a43_res_distal | +distal loss 가중 | 위 + joint-weights link6/hand 3.0 | 위 (A/B vs geometry-only) | 0.788 / 6.48% | — | GPU4 | 🔬 학습중(PID 3949992) |
| KUKA link-swap 조사/공략 | KUKA −0.07 격차 | 회수율 게이트→kinematic decode or detector retrain | KUKA ADD-AUC vs 0.690, tail 15.9% | 0.690 | — | GPU1/3 | 🔬 조사중 |
| Baxter rot (진행중) | Baxter rot head | train_rotation | geo deg | — | Ep~ | GPU2 | 🔬 학습중 |

## §3. 결정·피벗 로그

- **2026-07-22 각도 head 라인 종료**: P1b(resnet50, MAE 6.80°) good-frame 0.767 < mlp-control(8.41°) 0.7706 — 각도 MAE는 ADD로 비전이(게이지+솔버 워시아웃). 각도 학습 3종 정지.
- **2026-07-22 회수율 게이트: distal decoder 死**: 정답 위치 heatmap 응답 median 0.074(≪ clean 0.894), 회수율 7.2%≪40%. test-time 경로 완전 닫힘 → Panda synth 유일 레버=detector 재학습.
- **2026-07-22 목표치 교정(사용자)**: oracle 대비 아닌 경쟁 대비 DR/Photo로 목표 재설정 (§1 확보 후 Panda synth 우선순위 재판단).
