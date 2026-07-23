# 2026-07-22 — 2악장 P0: 분리 solve (freeze-head-theta) — off-frame 손목 flip을 R,t에 국한

> **2악장 처방 실행 스레드.** [1악장 재검토](2026-07-22_gap_reexamination.md)에서 남은 RoboPEPP 격차의 진짜 출처를 *"우리 솔버가 off-frame 손목 키포인트로 전 팔을 잘못된 basin으로 끌고 가는 취약성(basin flip)"* 으로 규명했다. 이 문서는 그 처방인 **관절각 θ를 head 예측값에 고정하고 카메라 6DOF(R,t)만 푸는 분리 solve**(RoboPEPP식)의 설계·변형·결과를 추적한다.
>
> 구현: `Eval/selfbbox_eval.py --freeze-head-theta` (신규 플래그, L136/L366) → `solve_batch(freeze_theta=True, theta_init=head_pred)`. head θ를 고정하면 off-frame 손목 키포인트는 관절각을 표류시킬 수 없고 R,t에만 영향 → flip 원천 제거.
>
> 재현 데이터: `Eval/rc_dumps_oas/dr_pred.npz` (DR base, cov-PnP+DARK, 1000f, control AUC≈0.704). 아래 counterfactual 수치는 이 dump 직접 계산.

---

## TL;DR

- **1악장 확정**: off-frame 손목 "tail"은 격차가 **아니다**. DREAM ADD-AUC = `mean·max(0, 1−10·ADD_m)` 이므로 ADD>100mm 프레임의 기여는 **정확히 0** → tail을 버리거나 GT로 채워도 AUC 불변(oracle-presence 0.703≈0.704, gt-fill 손목 0.704, mean-fill 0.701 — 전부 검증됨). 단일축 처방(edge-gate drop, prior-fill, focal, mlp_patch, MoE, PARE) = net-zero, **REFUTED**.
- **진짜 근원 = basin flip**: 우리 솔버는 θ(7)+R(3)+t(3)=13DOF를 재투영으로 **공동 최적화** → 환각된 off-frame 키포인트가 손목뿐 아니라 base·R·t까지 잘못된 basin으로 끌고 감. RoboPEPP는 θ를 **회귀**하고 6DOF만 conf-filter BPnP로 풀어 손상을 손목에 **국한**.
- **counterfactual**: proximal 앵커 **단독** +0.006 / distal cap **단독** +0.011 (= 단일축이 net-zero였던 이유) vs **둘 동시** +0.033(≤150mm) ~ +0.063(≤80mm). 격차 전체가 이 축.
- **P0 = 분리 solve**로 둘을 동시에 달성 (θ 고정 → 손목각 flip 차단 = distal cap, R,t는 conf-filter kp만 = proximal 격리). 예측 +0.03~0.06.
- **결과 (🟡 부분 반증 + 방향전환)**: naive freeze는 DR 0.704→**0.533**, Photo 0.738→**0.561**로 **크게 악화** — θ 재투영 정제가 **good 프레임(89%)에 필수**(good AUC 0.788→0.582)이기 때문. freeze는 tail만 개선(tail median 183→135mm)하나 good 손실이 압도. flip-trigger 2-pass(reproj-gate τ=60px)로 tail에만 적용하면 **+0.009** (실전), oracle flip-trigger 상한 +0.027. **basin flip은 작은 레버로 확정.**
- **진짜 격차 = good 프레임 각도 정확도**: good 893장 base 0.788 vs **oracle-angle(GTθ) 0.899**(med 7.2mm) → good에서만 **+0.11 헤드룸**, RoboPEPP 0.83 초과. head 아키텍처 튜닝(mlp_patch 0.780 / MoE 0.773)은 base 미달 = 천장. **다음 = regressed 각도 정확도 자체를 올리기(IEF/iterative + pose-prior, P1).**

---

## (a) 1악장 결론 요약 — off-frame tail은 AUC 0 기여, 단일축은 전부 REFUTED

DREAM ADD-AUC의 프레임별 기여는 닫힌 형태다 (RoboPEPP `test.py:283-292`, robopose `dream_meters.py:50-62` 동일):

```
AUC = mean_frames  max(0, 1 − 10·ADD_frame)      (ADD in meters)
```

⇒ ADD_frame ≥ 0.1m 프레임의 기여 = **0**. DR dump 실측: tail(ADD>100mm) 비율 **10.7%**, 그 AUC 총기여 **0.0000**. 따라서 off-frame 손목을 복원하거나 그 프레임을 드롭해도 AUC는 한 점도 오르지 않는다.

**검증된 net-zero 처방 (REFUTED 목록)**:

| 처방 | AUC | 판정 |
|---|---|---|
| control (DR base) | 0.704 | baseline |
| oracle-presence (off-frame 손목 드롭) | 0.703 | net-zero |
| gt-fill 손목 J5,J6 (GT로 채움) | 0.704 | net-zero |
| mean-fill 손목 | 0.701 | net-zero |
| edge-gate drop / prior-fill / focal / mlp_patch / MoE / PARE (단일축) | ±0 | REFUTED |

**clean vs off-frame 재진단** (프레임 분류 기준): CLEAN 프레임(86%, 전부 in-frame) base ADD-AUC **0.762**, median **15mm**; OFF-frame 프레임(14%) **0.353**, median **79mm**. off-frame이 파국적으로 낮지만 §(a)의 산술상 그 손실은 AUC에 이미 0으로 반영 → 관건은 tail을 버리는 게 아니라 **파국(>100mm)을 graded(<100mm)로 전환**하는 것.

> 주의: "off-frame 프레임 14%"(손목이 물리적으로 화면 밖)와 "tail 10.7%"(ADD>100mm)는 다른 분류. off-frame 중 일부는 basin flip 없이 graded로 살아남고, tail은 대부분 off-frame이 flip을 촉발한 프레임.

---

## (b) 근원 = basin flip (메커니즘 + counterfactual)

### 메커니즘

- **우리** (`solve_pose_kinematic.py:218,274`): `solve_batch`가 θ(6d)+R(6d)+t(3)를 재투영으로 **공동 최적화**(L274-278 모두 `requires_grad`). off-frame 손목 키포인트는 `conf_gate=0.05`로 하드컷되지만 **해당 관절 θ는 여전히 free** → 재투영 제약이 사라진 손목 DOF가 표류하고, Adam이 결합 목적함수에서 R·t까지 잘못된 basin으로 끌고 감.
  - tail 지문: reproj median **90px**(good 1.5px), 손목각 **26–38°**, base link0 median **62.7mm**(good 17.5mm) — 손목만이 아니라 **전 팔이 밀림**(tail 중 base>50mm가 53%).
- **RoboPEPP** (`model.py:15-51`, `test.py:252-258`): θ를 JointNet IEF 4-step으로 **회귀만**, 포즈는 conf-filter 키포인트로 **BPnP 6DOF cTr만** solve. θ는 재투영이 못 건드림 → off-frame 손상이 J6 회귀오차(**5.4° DR / 4.8° Photo**)로 국한, 전 팔 flip 없음.

### counterfactual (DR dump, tail n=107)

proximal(link0–4)을 good-median으로 앵커 + distal(link6/7/hand)을 cap:

| 개입 | AUC | Δ |
|---|---|---|
| baseline | 0.7041 | — |
| proximal 앵커만 | 0.7095 | +0.006 |
| distal cap≤150mm만 | 0.7150 | +0.011 |
| proximal 앵커 + distal≤150mm | **0.7365** | **+0.033** |
| proximal 앵커 + distal≤120mm | **0.7493** | **+0.045** |
| proximal 앵커 + distal≤100mm | **0.7581** | **+0.054** |
| proximal 앵커 + distal≤80mm | **0.7669** | **+0.063** |

**판독**: 단일축(+0.006/+0.011)은 무효 = refuted 단독 시도들의 정량 재현. **둘 동시** +0.033~0.063 = 격차 전체가 이 축. 현실 목표는 distal이 mean-fill 수준(~120–150mm)에 머문다는 보수 가정 아래 **+0.03~0.045**.

---

## (c) 2악장 P0 설계 — freeze-head-theta 변형

θ를 head 예측에 고정 = counterfactual의 "distal cap"(손목각이 flip 대신 head prior≈mean에 머묾) + R,t만 conf-filter kp로 solve = "proximal 격리"를 **동시** 구현. `--freeze-head-theta` (전역 고정) + `--edge-gate`(off-frame kp를 R,t solve에서 추가 배제).

| 변형 | 플래그 | 의미 |
|---|---|---|
| **control** | (없음) | 공동 13DOF solve 재현. DR base 0.704 재현이 목표(sanity). |
| **freeze** | `--freeze-head-theta` | θ=head 고정, R,t 6DOF만 solve. 순수 분리 solve = RoboPEPP식. flip 원천 차단. |
| **freeze+edge-gate8** | `--freeze-head-theta --edge-gate 8` | 위 + 프레임 경계 8px 이내(off-frame·near-edge) 키포인트를 R,t solve에서 배제. 오염된 kp가 R,t마저 흔드는 것 방지. |
| **freeze+edge-gate-oracle** | `--freeze-head-theta --edge-gate` (oracle presence) | GT presence로 off-frame kp를 정확히 배제한 상한. edge-gate8과의 차 = presence 판정 오류가 남긴 여지. |
| **freeze (Photo)** | `--freeze-head-theta` (Photo split) | Photo split 일반화 확인. |

**의미 해석 축**:
- `freeze` > `control` 이면 basin flip이 실제 원인임을 확증 (θ 고정만으로 회수).
- `edge-gate8` 추가 이득 = off-frame kp의 R,t 오염분.
- `edge-gate-oracle` − `edge-gate8` = presence 판정을 완벽히 하면 남는 추가 여지(P0b/검출기 개선 EV).
- **do-no-harm 게이트**: clean 프레임(86%) AUC가 control 대비 퇴화하면 안 됨 (θ 재투영 정제 이득 상실 위험 — 그럴 경우 flip-triggered 2-pass로 후퇴).

**예측**: DR AUC 0.704 → **+0.03~0.06** (counterfactual 하한~상한). Photo도 유사 폭 기대.

---

## (d) 결과 — 🟡 naive freeze REFUTED (good 프레임 붕괴)

### d.1 전역 freeze 변형 (measured)

| 변형 | split | AUC | Δ vs baseline | 판독 |
|---|---|---|---|---|
| control (공동 13DOF 재현) | DR | **0.7040** | — | 재현 ✓ (baseline 0.7040) |
| **freeze** (θ=head, R,t만) | DR | **0.5329** | **−0.171** | 크게 악화 |
| freeze+edge-gate8 | DR | 0.5310 | −0.173 | edge-gate 무의미 |
| freeze+edge-gate-oracle | DR | 0.5324 | −0.172 | oracle presence도 무효 |
| **freeze** | Photo | **0.5610** | **−0.177** (baseline 0.738) | Photo도 동일하게 붕괴 |

**전역 freeze는 정반대 결과.** edge-gate는 8px든 oracle presence든 차이 없음(0.531~0.532) → off-frame kp의 R,t 오염은 부차, 진짜 손실은 **good 프레임의 θ 정제 상실**.

### d.2 good/tail 분해 (control 기준, DR)

| 그룹 | n | control | freeze | 판독 |
|---|---|---|---|---|
| **GOOD** | 893 | AUC **0.7884** (med 14.8mm) | 0.582 (med 33mm) | **85% 악화** — 공동최적화가 good에 필수 |
| **TAIL** | 107 | median 183mm | median **135mm** | freeze가 tail엔 73% 개선(36%<100mm) |

freeze는 예상대로 tail의 flip을 완화(183→135mm)하지만, DREAM AUC 산술상 tail은 어차피 ~0 기여(§a)이고 **good 893장이 0.788→0.582로 무너지면서 전체가 붕괴**. θ 재투영 정제는 good 프레임에서 head 예측을 실제로 개선하고 있었음 — 이를 통째로 끄면 손해.

### d.3 flip-trigger 2-pass (tail에만 freeze 적용)

good 손실을 피하려 **reproj 큰 프레임만** freeze로 재solve:

| 방식 | AUC | Δ | 비고 |
|---|---|---|---|
| oracle per-frame flip-trigger (상한) | 0.7314 | +0.027 | GT로 flip 프레임 정확 선별 시 상한 |
| **실전 reproj-게이트 (τ=60px)** | **0.7130** | **+0.0090** | flip 프레임을 reproj로 판정 |

**basin flip 레버의 실전 크기 = +0.009** (oracle 상한도 +0.027). counterfactual 예측(+0.03~0.06)에 못 미침 — 예측은 proximal이 완벽 앵커된다는 가정이었으나, freeze는 good에서 그 앵커 이점까지 잃고 tail 판정도 불완전. **작은 레버로 확정.**

---

## (d′) 후속 진단 — 진짜 격차는 good 프레임 각도 정확도

P0가 tail을 겨냥했으나 분해가 격차의 위치를 **good 프레임**으로 이동시켰다.

### d′.1 good-frame 헤드룸 = +0.11 (병목도 각도)

GOOD 893장에서 관절각을 GT로 바꾸면(oracle-angle):

| GOOD 893장 | AUC | median (mm) |
|---|---|---|
| base (head θ) | 0.7884 | 14.8 |
| **oracle-angle (GT θ)** | **0.8991** | **7.2** |

**good 프레임에서만 +0.11 헤드룸**, 게다가 **0.899 > RoboPEPP 0.83.** 즉 격차는 tail의 flip이 아니라 **good 프레임의 회귀 각도 정확도**에 있고, 그 상한은 RoboPEPP를 넘는다. 1악장 H2(good proximal 관절 열세)가 정량 확정됨.

### d′.2 head 아키텍처 재배치는 천장에 막힘

good-frame AUC로 head 변형 비교:

| head | good-frame AUC |
|---|---|
| base | **0.7884** |
| mlp_patch | 0.7802 |
| p3mixsel (MoE) | 0.7728 |

둘 다 base 미달 → **가중치 재배치·라우팅 계열로는 0.788 천장을 못 뚫음**(07-21 제로섬 결론과 일치). oracle 0.899까지의 +0.11은 **각도 값 자체의 정확도**를 올려야 회수 가능 — 아키텍처 셔플이 아니라 회귀 품질.

---

## 결론 (🟡 부분 반증 + 방향전환)

- **naive decouple(전역 freeze) = REFUTED.** θ를 head에 고정하면 tail flip은 완화되나 good 프레임(89%)의 재투영 정제를 통째로 잃어 DR 0.704→0.533, Photo 0.738→0.561로 붕괴. **공동최적화는 good에 필수.**
- **basin flip은 실전 +0.009짜리 작은 레버** (reproj-gate flip-trigger 2-pass; oracle 상한도 +0.027). counterfactual 예측 +0.03~0.06은 "proximal 완벽 앵커" 가정 아래서였고, freeze는 그 앵커 이점을 good에서 잃어 미달. tail은 여전히 겨냥 가치가 낮음(AUC 0 기여 산술 + 작은 회수).
- **진짜 격차 = good 프레임 각도 정확도.** good 893장 base 0.788 vs oracle-angle **0.899**(+0.11), 그리고 **0.899 > RoboPEPP 0.83** — 격차 전량이 이 축이고 상한은 경쟁자를 넘는다. head 아키텍처 재배치(mlp_patch·MoE)로는 0.788 천장 → **회귀된 각도 값 자체의 정확도**를 올려야 함.
- **다음 = IEF/iterative 회귀 + pose-prior (P1).** 아키텍처 셔플이 아니라 각도 회귀 품질. flip-trigger 2-pass(+0.009)는 do-no-harm이면 무료 보조 레버로 병행 가능.

---

## (e) 다음 계획 — P1 / P2 (각도 정확도 축으로 재초점)

### P1 — good 프레임 proximal 관절 정확도 [학습, 2차 레버]
진단 연결: 1악장 H2. good 프레임 J2–J4 오차 우리 ~3–5° vs RoboPEPP **1.7–2.5°**. frozen 백본 유지하며:
- **P1a — 별도 학습형 관절-네트워크**: 키포인트 검출기와 **분리된** 관절-각 네트워크(fine-tune DINO 사본 또는 소형 ViT/CNN, 입력=crop). **반증 경계**: 3회 반증된 것은 *키포인트를 먹이는 공유 백본*의 적응(sub-pixel 파괴)뿐 — 별도 네트워크는 키포인트 백본을 안 건드리므로 반증 대상 아님.
- **P1b — IEF iterative head**: 현 AngleHead를 4-step residual(RoboPEPP JointNet 이식)로. frozen feature 위 헤드 레벨. `TRAIN/model_angle.py`. PARE와 병렬 후보.
- 게이트: val 관절 MAE를 RoboPEPP Tab.3(J1–J6 4.9/2.3/2.7/2.2/4.9/5.4 DR)와 직접 대조. **base J0 미퇴화**(07-21 MoE 제로섬 교훈).

### P2 — 측정으로 H1/H2 배분 확정 [분석, 선행 권장]
- good vs tail **관절 MAE 분리** + **per-joint good-frame MAE**를 RoboPEPP Tab.3와 대조. good proximal이 이미 ~2°면 P1 폐기·전량 P0; ~4–5°면 P1 유효.
- P0 적용 후 dump 재분해로 tail이 실제 graded(<100mm)로 가는지, distal이 몇 mm에 멈추는지 실측(회수량 확정).
- **P0+P1 가산성**: tail 교정(+0.03~0.045)과 good proximal(+0.02~0.03)이 독립 축이면 합산으로 RoboPEPP 초과 가능 — 검증 대상.

---

## 부록 — 재현 커맨드/파일

- 구현: `Eval/selfbbox_eval.py:136,366` (`--freeze-head-theta`), `Eval/solve_pose_kinematic.py:218,274,366` (`solve_batch(freeze_theta=)`).
- 실행: `Eval/selfbbox_eval.py --freeze-head-theta [--edge-gate 8]` (변형별 플래그 §c 표).
- 분해: `Eval/rc_dumps_oas/dr_pred.npz` 로드 → per-keypoint L2, `AUC = mean(clip(1−10·ADD, 0, 1))`, tail=ADD>0.1m.
- 근원 진단 원본: [2026-07-22_gap_reexamination.md](2026-07-22_gap_reexamination.md) (research agent 소유 — 참조만).
