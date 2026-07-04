# DINObotPose3 — 세션 진행 오버뷰 (2026-07-03 ~ 07-04)

> DREAM 벤치마크 SOTA 달성과 그 이후 개선/탐색의 전체 요약. 세부는 각 실험 파일, 원본 일지는 `../../3_pose_models/DINObotPose3/EXPERIMENTS.md`, 확정 결론은 `SUMMARY.md`.

---

## 🏆 현재 배포 성적: mean 0.799 vs RoboPEPP 0.780 / RoboTAG 0.740

DREAM 4개 real split, **동일 프로토콜(predicted angles + 완전 자동 bbox)**:

| 카메라 | 우리(현재) | RoboPEPP | RoboTAG | 판정 |
|---|---|---|---|---|
| realsense | **0.821** | 0.805 | 0.783 | BEAT |
| kinect360 | **0.813** | 0.785 | 0.757 | BEAT |
| azure | **0.792** | 0.753 | 0.831 | RoboPEPP BEAT, RoboTAG에 −0.039 |
| orb | **0.771** | 0.775(GT)/0.344(auto) | 0.588 | RoboPEPP GT에 −0.004, auto/RoboTAG는 압승 |
| **mean(4)** | **0.799** | 0.780 | 0.740 | **+0.019 / +0.059** |

핵심: 완전 자동 bbox(bbox-from-solved)가 RoboPEPP/RoboTAG를 침몰시키는 orb auto-detection 붕괴를 해결 → 동일 프로토콜에서 우리가 최고. 유일 상대 열세는 azure(RoboTAG 0.831).

---

## 파이프라인 (배포 구성)

```
이미지 → DINOv3 검출기(frozen) → self-bbox(풀린 스켈레톤 투영) → roi_align crop
      → crop 검출기 heatmap → [DARK sub-pixel 디코드] → 2D 키포인트+conf
      → crop angle head(θ) + crop rot head(R_init)
      → 운동학 솔버(PnP init + 재투영 refine, [cov-PnP 이방성 가중], conf-gate)
      → [nvdiffrast 정밀 mesh 실루엣 + SAM 마스크 render-and-compare] (원거리 카메라만)
      → 관절각 θ + 카메라 포즈 (R,t)
```
backbone은 **frozen이 최적**(적응 계열 3회 반증). 카메라별 config: azure는 RC off(근거리), rs/kinect/orb는 RC on.

---

## 무엇이 성능을 올렸나 (채택된 레버)

| 레버 | 이득 | 비용 | 세부 |
|---|---|---|---|
| **nvdiffrast+SAM render-and-compare** | 07-03 SOTA 0.796 달성 (rs/kinect/azure BEAT) | 없음(테스트타임) | [render_compare](2026-07-03_render_compare_sota.md) |
| **cov-PnP** (히트맵 이방성 공분산 Mahalanobis) | 가림 +0.011@20%, do-no-harm | 없음 | [occlusion_track](2026-07-03_occlusion_track.md) |
| **DARK sub-pixel 디코딩** | 전 카메라 pose +0.005~0.017, orb 격차 −0.010→−0.004, mean 0.796→0.799 | 없음(추론) | [dark_decode](2026-07-04_dark_decode.md) |

세 레버 모두 **학습 불필요** — 테스트타임/솔버/디코드 레벨. 이게 이 파이프라인의 강점(frozen backbone의 sub-pixel 정밀도 보존).

---

## 가림 강건성 (RoboPEPP Fig.6 프로토콜 벤치)

`Eval/occlusion_bench.sh`로 RoboPEPP 가림 실험 재현. 우리(+RC) vs RoboPEPP:

| RoI 가림 | 0% | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| **우리** | 0.775 | 0.726 | **0.626** | **0.525** | 0.328 |
| RoboPEPP | 0.795 | 0.730 | 0.600 | 0.470 | 0.351 |

**20-30% 가림에서 승**, 열화 기울기 동일. 기존 스택(conf-gate+FK전파+RC)이 이미 RoboPEPP급 가림 강건성 보유.

---

## 무엇이 안 됐나 (반증/종료 — 재실험 금지)

| 방향 | 판정 | 이유 |
|---|---|---|
| 멀티스타트 RC + IoU 선택 | ❌ | 잔여 실패는 R-basin 아님(orb=2D, 40%=θ 상류붕괴). 진단 가치 큼 |
| **DINO feature-metric RC** (문헌 2R 1순위) | ❌ 종료 | 우리 실루엣 RC와 중복 — 실루엣이 depth 신호 이미 포화. azure 여지없음/rs +0.002/40% +0.005 tail만 |
| edge-NCC RC | ❌ | 판별력≠최적화가능성 (프로브 GT승, 목적함수론 발산 −0.10~0.18) |
| 768 crop 해상도 | ❌ | frozen 512-스택에 회귀(−0.14), detector 재학습 캐스케이드 전제 |
| 가림-로버스트 실루엣(render∧¬SAM 다운웨이트) | ❌ | depth 편향(−0.019), 명시적 가림체 세그멘테이션 필요 |
| 모집단 평균 prior / 학습 state prior | ❌ | synth 관절 독립·광분산 → 정답과 싸움(−0.09) |
| 백본 적응 전 계열(SSL/co-finetune/V-JEPA) | ❌ (6월+재확인) | sub-pixel 정밀도 파괴, V-JEPA 2.1 논문 독립 확인 |

교훈: 문헌 순위표가 아니라 **우리 파이프라인의 실제 병목**에 맞는 것을 찾는 게 핵심. 1순위(feature-metric)는 중복이었고, 순위 낮던 DARK(무료 디코드 픽스)가 실질 성과.

---

## 문헌 조사 (2라운드)

- **1R** (`../robot_pose_sota_survey.md` §1-5): DREAM SOTA 계보 검증(RoboPEPP 0.780 프론티어, PoseDiff 저자철회), 휴먼 포즈 가림 메커니즘, 월드모델 적용성.
- **2R** (§6): post-SOTA 신규 아이디어 5종. DINO feature-metric RC(종료), **DARK(채택)**, dense correspondence head(미착수), uncertainty head(미착수), featuremetric 키포인트 refine(미착수).
- **RoboTAG(2025-11) 검증**: auto-bbox 동일 프로토콜에서 우리 79.9 > 74.0. 유일 우위 azure(0.831)는 closed-loop 2D-3D depth 일관성 → 로드맵 ⑦로 실험 중.

---

## 🔄 진행 중 (5-GPU, 2026-07-04 시점)

| 실험 | 상태 | 판정 대기 |
|---|---|---|
| 가림-증강 angle head (occaug) | Ep17/20 | 가림 벤치 30-40% + 클린 do-no-harm |
| 경량 가림-증강 (light) | Ep14/20 | Pareto 개선 여부(클린 −0.009 축소) |
| 가림-증강 rot head | ✅ 완료(geo 3.27°) | angle과 짝지어 평가 |
| **RoboTAG reproj-consistency** w50/w150 | Ep1 | azure ADD 개선(기준 0.792) |

occaug head = 강건성/정확도 트레이드오프(Ep1 기준 +0.014/0.018 가림, −0.009 클린) → 별도 "가림 강건성 config" 포지션. 완료 후 최종 A/B.

---

## 남은 방향 (로드맵, `../robot_pose_next_directions.md`)

1. orb −0.004 완전 초월: detector 2D 개선(768 재학습 캐스케이드) 또는 render-h 576+
2. azure(RoboTAG 대비 유일 열세): reproj-consistency(실험 중) 또는 depth 브랜치
3. 논문용 full-split 재잠금(현재 held-out 800), main merge + GitHub/GPU 서버 동기화

---

## 인프라 주의

- **GPU 배치는 UUID로** — 이 머신은 정수 `CUDA_VISIBLE_DEVICES`가 물리 GPU와 뒤엉켜 매핑됨(0→물리3, 2→물리1). 유휴 GPU: `GPU-7ff6997b`, `GPU-70a2a406`.
- 백그라운드 학습은 `setsid`로 완전 분리(안 하면 런처 셸 종료 시 orphan).
- env `dino`(torch 2.10+cu128, nvdiffrast 설치됨). 체크포인트 로컬 `TRAIN/outputs_*` 미러.
