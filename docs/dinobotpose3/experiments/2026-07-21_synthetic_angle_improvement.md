# 2026-07-21 — Panda 합성(synthetic) 성능 개선 조사·시도

> 문제: `PAPER_OVERLEAF.tex` `tab:main`의 **Panda 합성 ADD-AUC 우리 74.2/76.9(DR/Photo)가 RoboPEPP 83.0/84.1·RoboTAG 82.5/84.3에 ~9-10점 열세.** 이 문서는 원인 진단 → 실패한 시도 → 확보한 이득 → 방법론 조사 → 현재 진행/계획을 정리한다. 관련: [2026-07-20_critic_debate.md](2026-07-20_critic_debate.md), EXPERIMENTS.md 2026-07-21 항목.

## 1. 진단 — 병목은 순수 "관절각 예측"

측정(`Eval/ablation_logs/oracle_angle_synth/`, `.../ceiling/`, synth DR base, cov-PnP+DARK, 1000f):

| 설정 | DR base | DR +RC | Photo base | Photo +RC |
|---|---|---|---|---|
| **예측(현재)** | 0.704 | 0.769 | 0.738 | 0.799 |
| **oracle-angle (전부 GT)** | 0.861 | 0.886 | 0.869 | 0.897 |
| oracle-except-J5 (J5만 예측) | **0.818** | — | — | — |
| oracle-except-J5+J6 | 0.754 | — | — | — |

- **병목 = 관절각.** GT 각도 주입 시 0.886/0.897로 **RoboPEPP(0.830/0.841) 상회** → 데이터·2D검출(~0.86 AUC)·bbox·깊이·가림 전부 정상, 파이프라인 상한이 SOTA 이상.
- **달성 상한(J5 write-off) = 0.818 base (+RC ~0.843).** J5(손목 roll)=관측성 천장(자기축 회전 불가시, ADD-benign). **J6(손목 pitch)=최대 회복 레버(0.064), 관측 가능** → 개선 타깃.
- 배제 근거: `--oracle-except`(신규 플래그)로 조인트별 상한 분해; `--oracle-bbox`(0.705≈0.704, bbox 무죄); self-occlusion 무죄(키포인트 100% 검출).

### 실패 모드 (rc_dumps_oas 프레임별 분석)
- **bimodal**: 90% 프레임 ~15mm(완벽), **10% 파국(>100mm), 근거리(0.6-0.9m)+90° 광각 집중.** 원거리는 ~0% 실패(실측과 반대).
- 실패 프레임: 손목 J4/J5/J6 각도 붕괴(26-38° vs OK 3-9°) + **reproj 90px**(OK 1.5px) = 솔버가 잘못된 basin, 검출 2D조차 못 맞춤.

## 2. 실패한 시도 — 재분배/라우팅 계열 전부 순이익 0

`eval_synth_head.sh`(신규, synth DR base + fail%>100mm), baseline 0.704/10.7%:

| 시도 | 결과 | 왜 |
|---|---|---|
| **min-reproj multi-start** (`--ms-local`) | DR 0.704→**0.692** (손해) | 손목 2D 미관측 → reproj 노이즈 과적합 |
| **RC-실루엣 손목 multi-start** (`rc_refine_wrist.py`) | fail프레임 0.033→0.038 (무효) | 손목이 실루엣서도 약함 |
| **focal 꼬리 재가중** (`--tail-gamma`, J6=1.0 mask) | 부진(전체 MAE 악화) | 재분배는 정보 안 더함 |
| **mlp_patch** (crop-matched, MAE 9.0° vs 10.9°) | ADD 0.706 / fail 10.7% (무효) | 평균 MAE↓가 flip 꼬리 못 고침 |
| **P3 MoE** (`mlp_mixsel`, 2-mode + appearance selector, 손목 MAE 최저 12.6°) | ADD 0.702 / fail 10.3% (노이즈) | **손목↑ but base J0 퇴화로 상쇄 = 제로섬** |

**결론: 가중치 재분배·표준헤드·MoE 라우팅 전부 무효.** 각도 MAE↔ADD/fail 탈동조 — 병목은 평균이 아니라 10% flip 꼬리이고, **하나의 pooled feature/loss budget을 공유하면 제로섬**(base-vs-wrist).

## 3. 확보한 이득 (유일한 실이득)

**cov-PnP + DARK를 합성 eval에도 일관 적용** (실측엔 쓰지만 논문 Table엔 누락됐던 것):
- DR 0.742→**0.769**, Photo 0.769→**0.799** (+0.027/+0.030). 같은 기법 일관적용이라 정당 → Table 1 갱신 대상.

## 4. 방법론 조사 (human+robot pose, WebSearch) — 정보를 *더하는* 헤드-레벨 기법

| 방법 | 메커니즘 | 제로섬 회피 | frozen 호환 |
|---|---|---|---|
| **HybrIK** (CVPR'21) | twist-swing: 관측 swing=키포인트에서 **해석적 동시 해**, roll(twist)만 전용 헤드 | ✅ 완전(공유 budget 없음) | 헤드/솔버(3D 키포인트 필요) |
| **PARE** (ICCV'21) | 관절별 **learned attention** → 자기 영역 + 보이는 이웃 문맥 | ✅ 관절별 독립경로 | ✅ DINOv3 패치토큰 |
| **ProHMR/RoboKeyGen** | multimodal 각도(flow/diffusion), mean 붕괴 안 함 | ✅ | ✅ |
| **RLE/keypoint-filtering** | 키포인트 불확실도 → 솔버가 붕괴 손목 downweight (RoboPEPP가 씀) | 중립 | ✅ |

**RoboPEPP 우위 규명**: masked embedding-predictive 사전학습(관절주변 패치 가림→임베딩 복원=**구조주입**) + eval-time keypoint filtering. **per-robot fine-tune은 관행(확인됨, 리더보드 confound)이나, 손목/가림 강점은 메커니즘.** RoboTAG는 반대로 공유모델(2D-3D topological consistency).

## 5. KUKA/Baxter 관련 (스코핑)

- **Baxter RC = 반증**(내 실행: 0.262→**0.009** 파국; SUMMARY.md에 기록돼 있던 것 — 약한 base+손목모양 모호성에 RC 발산). **KUKA/Baxter "RC로 40점" 가설 폐기.**
- 이들의 진짜 병목 = **약한 base 포즈**(검출 transfer-only + 데이터 적음). RC는 base 좋을 때만 통함(Panda).
- 단 렌더러 자산은 확보: `Eval/baxter_render.py`(작동), KUKA RC는 ~120-180 LOC(iiwa STL from RoboPEPP + URDF origins, NVDRSilhouette robot-agnostic) — 단 base 개선 없이는 실익 없음.

## 6. 현재 진행 / 계획

**사용자 결정: ① Panda 개선 먼저 → ② KUKA/Baxter base 포즈.**

**① 진행중 (2026-07-21 20:00~, GPU 5장):** PARE 계열 5종 (`--head-type pare`, synth DR, 40ep):
- pare(main) / pare_ctrl(matched mlp) / pare+occ-aug0.3 / pare+kp-drop0.15 / pare+fk30
- 검증: val MAE 매칭비교(base J0 유지 + 관측 J4/J6 개선?) → 유망 후보 `eval_synth_head.sh --crop-head-type pare`로 fail% 판정.
- 게이트: fail% < 10.7% AND base 미퇴화. 실패 시 → **HybrIK twist-swing 이식**.

**② 대기:** KUKA/Baxter base 포즈(검출+각도) 개선 — 큰 작업, 헤드룸 40점.

## 7. 신규 코드 (이 세션)
- `TRAIN/model_angle.py`: `AngleHeadMixSel`(P3), `AngleHeadPARE`(PARE).
- `TRAIN/train_angle.py`: `--tail-gamma`/`--joint-weights`/`--focal-warmup-epochs`/`--focal-clamp`(focal), `--n-mix`/`--selector-weight`/`--load-balance`(P3), head-type choices에 `mlp_mcl`/`mlp_mixsel`/`pare`.
- `Eval/selfbbox_eval.py`: `--oracle-except`(조인트별 상한), `--crop-head-type`/`--crop-n-mix`.
- `Eval/eval_synth_head.sh`(헤드 fail% 재평가), `Eval/oracle_angle_synth.sh`, `Eval/rc_refine_wrist.py`(반증됨).
- **주의(반증 목록 추가)**: focal/mlp_patch/P3-MoE/min-reproj-MS/RC-실루엣-손목-MS/Baxter-RC — SUMMARY REFUTED 반영 필요.

---

## 8. 진단 정정 (핵심) — 병목은 "손목 각도 관측 천장"이 아니라 **손목이 화면 밖(off-frame)**

사용자 통찰로 재분석(`rc_dumps_oas/dr_pred.npz` + 주석 projected_location):
- **실패 프레임: 로봇 키포인트의 35%(median)가 프레임 밖.** OK 프레임 0%.
- 조인트별: link0~4(몸통) 화면밖 0-2% / **link6 50%·link7 56%·hand 50%**(손목/EE). OK는 3-7%.
- **메커니즘**: 근거리+90° 광각 → 로봇이 커져 손목/hand가 프레임 밖으로 잘림 → 검출기가 **화면밖 키포인트를 confident하게 환각**(가장자리 clamp, 그래서 "found=7/7"이 오해였음) → 각도/솔버 오염 → 손목 각도 붕괴 → basin flip(reproj 90px).
- **정정**: 앞선 "J5-roll 관측성 천장"은 부차적. 진짜는 **부분 가시성(손목 화면밖)**. "키포인트는 (보이는 것만) 잘 찍는데 안 보이는 손목 각도는 못 맞춤".

## 9. 2-에이전트 토론 결론 (사람자세 truncation vs 로봇 운동학/불확실도)

**완전 수렴** (양측 개별 반증 후 합의):
- **confidence축 ≠ presence축**: RoboPEPP의 confidence ε-필터는 우리에겐 무효(우리 검출기는 화면밖을 confident하게 환각). **boundary/presence로 걸러야 함.**
- **drop-not-recover**: 화면밖 손목은 90° 크롭에서 복원 불가(복원=또 다른 환각) → **드롭하고, 로봇의 정확한 FK+관절한계가 나머지를 결정**. (사람자세 A도 로봇 정확 운동학 근거로 수긍.)
- **basin flip은 나쁜 키포인트가 원인**(잔여 DOF 모호성 아님) → 드롭만으로 flip 멈출 것. 학습된 prior는 안 보이는 q5-7 채우기용(2차, 필요시).

**Exp 1 (합의, training-free)**: `selfbbox_eval --edge-gate <px>`(화면밖 키포인트 conf=0 → 솔버 배제) + `--prior-adaptive`(손목 채움). 게이트: PASS=tail reproj<3px AND fail%<4% AND ADD≥79 / PARTIAL=reproj 고쳐졌으나 ADD 76-79 / FAIL=reproj>10px.
**Exp 2 (PARTIAL 시)**: (a) ProbPose presence-head(BCE, crop-aug off-frame 학습) vs (b) 학습된 조건부 손목 prior(VPoser/NRDF) — detection-미스드롭 vs fill-quality 판별.

**신규 코드**: `Eval/selfbbox_eval.py --edge-gate`(crop kp2d→frame 매핑, 경계 밖 conf=0). 🔄 Exp1(edge-gate 8 + prior-adaptive 0.02) 실행 중 → 결과가 진단·해법 동시 확정.
