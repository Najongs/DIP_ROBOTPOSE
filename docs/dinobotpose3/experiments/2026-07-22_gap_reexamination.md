# 2026-07-22 — RoboPEPP 격차 재검토: "off-frame 손목은 공통 한계, 우리 차별점은 솔버의 취약성"

> **문제 재설정.** 지금까지 진단 사슬은 *"실패의 근원 = off-frame 손목 각도의 정보이론적 예측불가"* 로 수렴했고, 이것이 RoboPEPP를 포함한 **모든 단일이미지 방법의 공통 한계**임을 입증했다. 그렇다면 **RoboPEPP의 +0.06 우위는 off-frame 손목에서 올 수 없다.** 이 문서는 남은 격차의 진짜 출처를 dump 기반 분해 + 경쟁 코드/논문 인용으로 규명하고, frozen-backbone 호환·비(非)반증 실험을 우선순위화한다.
>
> 관련: [2026-07-21_synthetic_angle_improvement.md](2026-07-21_synthetic_angle_improvement.md)(진단 사슬), [2026-07-20_critic_debate.md](2026-07-20_critic_debate.md), [references/sota_survey.md](../references/sota_survey.md), [architecture/model.md](../architecture/model.md).
> 재현 데이터: `Eval/rc_dumps_oas/dr_pred.npz` (DR base, cov-PnP+DARK, 1000f, AUC=0.704). 아래 모든 수치는 이 dump에서 직접 계산.

---

## TL;DR (결론 먼저)

1. **AUC 산술로 tail은 이미 0을 기여한다.** ADD>100mm인 10.7% 프레임의 AUC 기여 = **정확히 0.0000**. 즉 off-frame 손목을 "복원"해도(불가능), 혹은 그 프레임을 버려도 AUC는 **한 점도 오르지 않는다.** ⇒ off-frame 손목은 격차의 원천이 **아니다** (재설정 확정).
2. **격차를 여는 유일한 두 경로**: (a) **good 프레임(89%)을 더 정확히**, (b) **tail 프레임을 <100mm의 graded 상태로 끌어올리기.** 손목 복원이 아니라 tail의 **파국(basin flip)을 graded로 전환**하는 것.
3. **우리 tail이 파국인 이유 = 솔버 취약성.** 우리 파이프라인은 (θ 7 + R 3 + t 3 = 13 DOF)를 재투영으로 **공동 최적화**한다 → 검출기가 환각한 off-frame 손목 키포인트가 **손목뿐 아니라 base·R·t까지 잘못된 basin으로 끌고 감**(tail 손목각 26–38°, tail reproj 90px, base link0 median 62.7mm도 밀림).
4. **RoboPEPP는 같은 off-frame을 10× 싸게 지불한다.** 관절각 θ는 **네트워크가 회귀**(JointNet IEF 4-step), 카메라 포즈 6DOF만 **confidence-filtered BPnP**로 푼다 → off-frame 손목의 손상이 **손목에만 국한**(RoboPEPP J6 오차 5.4°(DR)/4.8°(Photo), 우리 tail 손목 26–38°). tail이 파국이 아니라 graded.
5. **역산(dump counterfactual)**: proximal 앵커 + distal cap을 **동시** 적용하면 0.704 → **0.737(≤150mm) / 0.749(≤120mm) / 0.758(≤100mm) / 0.767(≤80mm)**. **+0.06 격차가 이 축에서 회수 가능.** 둘 중 하나만(앵커만 0.7095, cap만 0.715)은 무효 — 이것이 refuted `edge-gate drop`/`prior fill` 단독 시도가 net-zero였던 이유를 **정확히** 설명한다.
6. **처방(P0, test-time only, 비반증)**: **regress-θ-then-solve-6DOF** (RoboPEPP식 분리) 또는 **off-frame 관절만 부분 freeze**. 우리 솔버엔 all-or-nothing `freeze_theta`만 있음 — **부분 freeze가 미구현·미시도.**
7. **2차 레버(P1)**: good 프레임 proximal 관절 정확도(우리 추정 ~3–5° vs RoboPEPP J2–J4 **1.7–2.5°**). frozen 백본 유지하며 **키포인트 검출기와 분리된 별도 학습형 관절-네트워크**로 접근(반증 대상 아님 — 반증은 *공유* 백본 적응).

---

## 1. 재설정: off-frame은 공통 한계, tail은 이미 AUC 0을 기여

### 1.1 AUC 산술 (왜 tail 복원이 무의미한가)

DREAM ADD-AUC의 프레임별 기여는 **닫힌 형태**다. AUC를 0–0.1m, step 1e-5로 sweep하면 (RoboPEPP `test.py:283-292`, robopose `dream_meters.py:50-62` 동일):

```
AUC = mean_frames  max(0, 1 − 10·ADD_frame)      (ADD in meters)
```

즉 ADD_frame ≥ 0.1m인 프레임의 기여는 **0**. dump 실측:

| 분해 | 값 |
|---|---|
| overall AUC | 0.7041 (검증: dump 재계산 = test.py 산출 일치) |
| tail(ADD>100mm) 비율 | **10.7%** |
| **tail의 AUC 총기여** | **0.0000** |
| good(89.3%) 총기여 | 0.7041 (평균 프레임 가중 0.789 → good 평균 ADD ~21mm, median 14.8mm) |

**함의**: off-frame 손목을 완벽 복원해도 그 프레임의 다른 오차가 남아 ADD가 여전히 >100mm면 기여는 0 → AUC 불변. 진단 사슬 point 4의 *"oracle-presence(off-frame 드롭) = 0.703, 변화 없음"* 이 여기서 산술적으로 필연임이 드러난다. **손목은 격차가 아니다.**

### 1.2 그럼 격차는 어디서? — dump counterfactual

tail 프레임(n=107)의 per-keypoint median 오차(mm): `link0 62.7 · link2 70.1 · link3 122.6 · link4 126.7 · link6 285 · link7 368 · hand 402`. **base link0도 62.7mm 밀려 있음**(good 17.5mm) — 순수 손목 문제가 아니라 **전 팔이 밀린 basin flip**(tail 중 base>50mm가 53%). 반증했던 `oracle-presence`가 왜 무효였는지 설명: base가 이미 틀렸으므로 손목만 드롭해도 소용없다.

AUC를 0.704에서 얼마나 올릴 수 있나 (proximal=link0–4를 good-median으로 앵커 + distal=link6/7/hand를 cap):

| 개입 | AUC | Δ |
|---|---|---|
| baseline | 0.7041 | — |
| **proximal 앵커만** (distal 그대로) | 0.7095 | +0.006 |
| **distal cap≤150mm만** (앵커 없음) | 0.7150 | +0.011 |
| proximal 앵커 + distal≤150mm | **0.7365** | **+0.033** |
| proximal 앵커 + distal≤120mm | **0.7493** | **+0.045** |
| proximal 앵커 + distal≤100mm | **0.7581** | **+0.054** |
| proximal 앵커 + distal≤80mm | **0.7669** | **+0.063** |

**핵심 판독**:
- **둘 중 하나만으로는 무효**(+0.006/+0.011). refuted `edge-gate drop`(proximal 앵커에 준함)·`prior-adaptive fill`(distal cap에 준함)이 **각각 단독**이라 net-zero였던 것 — dump가 이를 정량 재현.
- **둘을 동시**에 하면 +0.033~+0.063으로 **격차 전체가 이 축**. 관건은 tail을 **파국→graded**로 바꾸되 (i) base/proximal 재투영 solve를 오염원(off-frame kp)에서 **격리**하고 (ii) 손목각을 flip 대신 prior/회귀값(≈mean, ~5–6°)에 **고정**하는 것.
- distal을 mean-fill 수준(~120–150mm, off-frame 손목의 실현가능 최선)까지만 눌러도 **+0.033~+0.045** 회수. 80mm까지 눌러야만 격차 전부지만, 그건 손목 복원이 필요 → 비현실. **현실 목표는 +0.03~0.045.**

---

## 2. 진짜 격차의 가설 (증거순 랭킹)

### H1 (주): 우리 솔버가 off-frame 손목의 손상을 basin flip으로 **증폭**한다 — RoboPEPP는 안 한다
**증거 강도: 강 (dump + 코드 인용 양쪽).**
- 우리: `solve_pose_kinematic.py::solve_batch` (L218) — θ(6)+R(6d)+t(3)를 재투영으로 **공동 최적화**(L274-278: `p,d6,t` 모두 `requires_grad`). off-frame 손목 키포인트는 `conf_gate=0.05`로 하드컷되지만(L283-), **해당 관절의 θ는 여전히 free** → 재투영 제약이 사라진 손목 DOF가 자유롭게 표류하고, Adam이 결합 목적함수에서 R·t까지 잘못된 basin으로 끌고 감. tail reproj median 90px(good 1.5px)·손목각 26–38°가 이 flip의 지문.
- RoboPEPP: `models/model.py::JointNet`(L15-51) — θ를 **회귀만**(IEF, `n_iter=4`, init=0에서 residual). 포즈는 `test.py:252-258` — heatmap confidence로 키포인트 필터(`thresh` DR 0.325/Photo 0.35, 4개 미만이면 0.025씩 완화) 후 **`BPnP_m3d`로 6DOF cTr만** 푼다. **θ는 재투영이 못 건드림** → off-frame 손목의 오차가 J6 회귀오차(5.4°/4.8°)로 국한, 전 팔 flip 없음.
- 결과: 같은 off-frame 입력에 우리 tail 손목 26–38°(파국) vs RoboPEPP J6 ~5°(graded). **격차의 dominant 성분**(회수 +0.03~0.05).

### H2 (2차): good 프레임 proximal 관절 회귀 정확도 열세
**증거 강도: 중 (경쟁 논문 수치 확정 + 우리 간접 추정).**
- 경쟁 관절오차(도, cross-confirmed RoboPEPP Tab.3 ↔ RoboTAG Tab.2): **RoboPEPP DR 3.8/Photo 3.2**(J2–J4 **1.7–2.5**, J6 5.4/4.8), **RoboTAG DR 3.6/Photo 3.3**(J6 **6.2–6.3**, 오히려 손목은 더 나쁨), HoRoPose 4.4–4.5. 우리 synth ~7–11°(tail 포함).
- **주목**: RoboTAG의 J6(손목)이 RoboPEPP보다 나쁜데도 경쟁력 유지 → **손목각은 우위의 자리가 아님.** 우위는 **proximal(J2–J4) ~2°.** 이는 §1의 분해와 정확히 일치(tail=0, good이 전부).
- 우리 good 프레임 ADD 17–21mm 중 2D/PnP floor ~12mm(oracle-angle 상한 0.861 역산), 잔여 ~5–9mm가 proximal 관절오차(~3–5° 추정) → RoboPEPP good ~7–8mm(proximal ~2°)와의 차. 회수 +0.02~0.03. **단 이 proximal 추정은 미측정 — P2로 확정 필요.**

### H3 (기각/부차): crop이 off-frame을 더 잡아준다
**증거 강도: 약.** RoboPEPP bbox는 `image_proc.py::get_extended_bbox`(L541)에서 데이터셋별 **소폭 픽셀 패딩**(panda DR = `wmin-40,hmin-30,wmax+10,hmax+10`) 후 **이미지 경계 [0,640]×[0,480]로 clip**(dream_ssl.py:250). 물리적으로 화면 밖 내용은 clip되어 **off-frame은 여전히 off-frame.** (논문 본문은 "100px 확장/curriculum λ 0→120px"을 주장 — eval 코드의 modest 패딩과 불일치, 정직히 플래그. 어느 쪽이든 clip 때문에 격차 원천 아님.) 우리 bbox-from-solved와 본질 동일. **부차.**

### H4 (검토): PnP 자체가 취약성의 매개 — RoboTAG는 PnP를 아예 건너뜀
**증거 강도: 중, 시사적.** RoboTAG(§4.2)는 *"RoboPEPP/HoRoPose가 키포인트 가림/off-frame에서 PnP가 노이즈를 증폭해 약해진다"* 를 명시하고 **3D 브랜치(DepthAnything-V2 init)에서 R·θ를 직접 회귀, PnP 생략**을 셀링포인트로 삼음(35ms 1-pass). 우리 H1과 같은 통찰의 극단판. 우리는 PnP를 버릴 순 없지만(frozen kp 파이프라인의 강점), **PnP의 입력을 오염에서 격리**(H1의 P0)가 같은 효과.

---

## 3. 경쟁자별 차별점 (코드/논문 인용)

| 항목 | **우리** | **RoboPEPP** (CVPR'25, 2411.17662) | **RoboTAG** (2511.07717) | **HoRoPose** (ECCV'24, 2402.05655) |
|---|---|---|---|---|
| 관절각 θ | 회귀 **후 재투영 공동최적화** (`solve_pose_kinematic.py:218,274`) | **회귀만**, IEF 4-step (`model.py:15-51`) | 3D 브랜치 **직접 회귀** | 직접 회귀 (JointNet) |
| 카메라 포즈 | θ와 **공동** 13DOF 재투영 refine | **분리**: conf-filter kp → **BPnP 6DOF만** (`test.py:252-258`) | **PnP 없음** (3D 직접) | RotationNet+DepthNet 직접 |
| off-frame kp | conf_gate 하드컷 but θ free → **flip** | conf<ε 필터, 4개 미만이면 ε−0.025 반복 | 3D consistency가 흡수 | 명시적 처리 없음 |
| 백본 | **frozen DINOv3** (적응 3회 반증) | **I-JEPA 관절-마스킹 사전학습 + per-robot finetune** | DepthAnything-V2 + 2D-3D align loss | ResNet 직접학습 |
| Panda 관절오차 (DR/Photo) | ~7–11° (tail 포함) | **3.8 / 3.2** (J2–J4 1.7–2.5) | 3.6 / 3.3 (J6 6.2) | 4.4 / 4.5 |
| Panda ADD-AUC (DR/Photo) | 74.2/76.9 (+RC 76.9/79.9) | **84.1** / 80.5 | 82.5 / **84.3** | 82.7 / 82.0 |

**메트릭 공정성 재확인 (진단 point 6 보강)**: robopose `dream_meters.py:45`·RoboPEPP `test.py:272`·우리 모두 **ADD를 전 키포인트(off-frame 포함) 평균**. **단 DREAM 원본은 다르다**: `DREAM/dream/analysis.py` — PCK(2D)는 off-frame GT를 **제외**(frustum 안만), ADD(3D)는 *"all keypoints were considered"* 로 **포함**하되 PnP는 검출된(=in-frame) 키포인트만 쓰고 sample은 ≥4 in-frame으로 게이팅. 즉 **우리가 쓰는 robopose 메트릭이 off-frame을 가장 무겁게 벌한다** — 이것이 tail을 파국으로 만드는 메트릭 측 배경(그러나 §1대로 tail은 어차피 0 기여, 관건은 flip→graded 전환).

---

## 4. 우선순위 실험 (frozen-backbone 호환 · 비반증 · 진단 사슬 연결)

> 게이트 공통: (i) clean(good 프레임) do-no-harm, (ii) tail 비율↓ 또는 tail ADD를 <100mm graded로, (iii) synth DR base AUC +0.03 이상. 평가: `Eval/eval_synth_head.sh`(fail%>100mm) + dump 재분해.

### P0 — regress-θ-then-solve-6DOF (분리 solve) [test-time only, 최고 EV]
진단 연결: H1·§1.2. RoboPEPP 메커니즘의 직접 이식.
- **P0a (즉시 가능)**: `solve_batch(freeze_theta=True, theta_init=head_pred)` — 현재 `freeze_theta`는 oracle(GT θ)용으로만 쓰였으나, **head 예측 θ로 고정**하고 R,t 6DOF만 conf+presence-filter 키포인트로 풀면 = RoboPEPP식. flip 원천 제거. **위험**: good 프레임의 θ 재투영 정제 이득 상실 → clean 회귀 가능. 따라서 **flip-triggered 2-pass**: 1차 정상 solve → `reproj_px>τ`(예 20px) 프레임만 P0a로 재solve. tail만 교정, good 불변.
- **P0b (부분 freeze, 신규 코드 필요)**: presence-gate(off-frame 키포인트 판정, `--edge-gate`가 이미 crop kp→frame 매핑 보유)로 **off-frame 관절의 θ만 head-prior에 freeze**, 관측 관절 θ+R+t는 계속 최적화. §1.2의 "proximal 앵커 + distal cap"을 가장 충실히 구현. 현 솔버는 all-or-nothing `freeze_theta`뿐 — **per-joint 마스크 requires_grad가 미구현.**
- **refuted와의 구분(중요)**: `edge-gate drop`(off-frame kp 드롭)·`min-reproj MS`·`RC-실루엣 손목 MS`·`prior-adaptive fill`은 전부 **θ를 free로 둔 채** kp만 조작하거나 손목을 *복원*하려 함 → dump상 "proximal 앵커만/distal cap만"에 해당 → net-zero. P0는 **θ의 flip 자체를 막는** 것(둘을 동시). **미시도 영역.**
- **예상 회수**: +0.03~0.045 (distal이 mean-fill ~120–150mm에 머문다는 보수 가정).

### P1 — good 프레임 proximal 관절 정확도 (별도 학습형 관절-네트워크) [학습, 2차 레버]
진단 연결: H2. 목표: good 프레임 J2–J4 오차 ~3–5° → ~2°(RoboPEPP 수준).
- **P1a**: **키포인트 검출기와 분리된 별도 관절-각 네트워크**(fine-tune DINO 사본 또는 소형 ViT/CNN, 입력=crop). **반증 경계 명확화**: 3회 반증된 것은 *키포인트를 먹이는 공유 백본*의 적응(sub-pixel 파괴). **별도 네트워크는 키포인트 백본을 건드리지 않으므로 반증 대상 아님** — RoboPEPP의 승리 메커니즘(적응 백본에서 관절 회귀)을 키포인트 정밀도 희생 없이 도입. 미시도.
- **P1b**: 현 AngleHead를 **IEF iterative(4-step residual, RoboPEPP JointNet 이식)** 로 — frozen feature 위 헤드 레벨. `TRAIN/model_angle.py`. PARE(진행중)와 병렬 후보.
- **P1c**: good 프레임 2D 키포인트가 정확(~0.86 PCK)하므로 **2D→θ 조건부 solve**(RoboKeyGen식 or 잘 조건화된 bundle) — 단 단일이미지 depth 모호(J5 관측천장)는 잔존. proximal에는 유효.
- 게이트: val 관절 MAE를 RoboPEPP Tab.3(J1–J6 4.9/2.3/2.7/2.2/4.9/5.4 DR)과 매칭 비교. **base J0 미퇴화**(제로섬 재발 방지 — 07-21 MoE 교훈).

### P2 — 측정으로 H1/H2 배분 확정 [분석, 선행 권장]
- good vs tail **관절 MAE 분리** + **per-joint good-frame MAE**를 RoboPEPP Tab.3와 직접 대조. good proximal이 이미 ~2°면 P1 폐기·전량 P0; ~4–5°면 P1 유효.
- P0 적용 후 dump 재분해로 tail이 실제 graded(<100mm)로 가는지, distal이 몇 mm에 멈추는지 실측(회수량 확정).

---

## 5. 열린 질문

1. **conf-filter 6DOF-only PnP가 base를 앵커할 수 있나?** proximal 키포인트는 base에 밀집(depth 약) → PnP 조건수 우려. 우리 **rot-head R_init**(원거리 basin 고정에 이미 +0.117)이 이를 상쇄하는지가 P0 성패의 관건.
2. **good 프레임 proximal 오차의 실제 값**(P2). H2 EV의 전제.
3. **PnP를 아예 버려야 하나?** RoboTAG의 무-PnP 직접 3D 회귀가 off-frame에 더 강하다는 자기주장 — 우리 frozen-kp 강점과 상충. P0(입력 격리)가 PnP를 유지하며 같은 이득을 주는지 vs 3D 브랜치가 필요한지.
4. **distal mean-fill의 실제 하한**: off-frame 손목을 head-prior에 freeze했을 때 hand 키포인트가 실제 몇 mm인가(§1.2는 80/100/120/150mm 시나리오만). ≤120mm면 +0.045, ≤150mm면 +0.033. P0b 실측 필요.
5. **P0 + P1 가산성**: tail 교정(+0.03~0.045)과 good proximal(+0.02~0.03)이 독립 축이면 합산으로 RoboPEPP 초과 가능 — 검증 대상.

---

## 부록 — 재현 커맨드/파일

- 분해 계산: `Eval/rc_dumps_oas/dr_pred.npz` 로드 → per-keypoint L2, `AUC = mean(clip(1−10·ADD, 0, 1))`, tail=ADD>0.1m. (본 문서 §1 표 전부 이 dump에서 산출, overall 0.7041 = test.py 일치 검증됨.)
- 인용 코드: RoboPEPP `test.py:252-292`, `models/model.py:15-51,235`, `datasets/image_proc.py:541,569`, `datasets/dream_ssl.py:246-286` · robopose `robopose/evaluation/meters/dream_meters.py:33-71` · DREAM `dream/analysis.py:246-355,858-994` · 우리 `Eval/solve_pose_kinematic.py:218-360`.
- 경쟁 수치 출처: RoboPEPP 2411.17662 Tab.2–3, RoboTAG 2511.07717 Tab.1–2, HoRoPose 2402.05655 Tab.1, DREAM 1911.09231. ⚠️ RoboTAG Tab.1의 RoboPEPP 베이스라인 행은 열 오정렬 의심(Panda DR 84.1만 일치) — RoboPEPP 수치는 자체 Tab.2 신뢰.

---

## 6. P0 사후 (07-22 실측) — 진짜 격차는 **good-frame regressed 각도 정확도**, P1 = IEF

**P0 결과 (coordinator 실행)**:
- naive freeze-head-θ = **0.533** (good 프레임의 재투영 joint-opt를 죽여 대폭 악화 — head θ가 noisy해서 그대로 얼리면 손해).
- 조건부 flip-trigger(reproj 게이트) = **겨우 +0.009** (oracle 상한 +0.027). → basin-flip 진단은 옳으나 **실전 레버는 작다.**
- **결정적 측정**: good 프레임(893장) base **0.7884** vs oracle-angle(GTθ) **0.8991** → good에서만 각도로 **+0.11 헤드룸**, 0.899 > RoboPEPP 0.83. 아키텍처 재배치(mlp_patch 0.780 / MoE 0.773)는 이 0.788 천장을 못 깸.

**재조준(§2 랭킹 갱신)**: H2(good-frame 각도)가 **주(主) 레버로 승격**, H1(basin-flip)은 부차(+0.009). 우리 head θ가 noisy → 솔버가 2D를 과적합해 0.788에서 막힘. **P0와 P1은 시너지**: IEF로 θ가 정확해지면 (i) good 프레임 각도가 오르고(H2 직접), (ii) freeze-θ-solve-6DOF(P0a)가 손해→이득으로 전환(basin-flip도 동시 해소). 즉 **P1(정확한 feed-forward 각도 회귀기)이 P0를 잠금해제한다.**

**왜 flat head는 천장인가**: `mlp`/`mlp_patch`/`transformer`/`MoE`는 전부 *동일 frozen feature → 각도*를 **1-shot**으로 매핑. RoboPEPP 3.2°의 절반은 masked-pretrain **feature**(우리가 못 씀, 반증), 나머지 절반은 **IEF 회귀기 구조**(운동학 상태를 반복 피드백 = flat head엔 없는 귀납편향, frozen 호환). P1은 이 후자를 이식.
> ⚠️ 정직한 리스크: 천장이 **head**가 아니라 frozen **feature**의 정보한계면 IEF도 0.788에서 막힌다. 그 경우 폴백 = **P1b 별도 학습형 관절-net**(feature 자체를 바꿈; §4 P1a). **∴ IEF(싸다)로 head 가설을 먼저 판정 → 막히면 별도-net으로.**

---

## 7. RoboPEPP JointNet = Iterative Error Feedback (정확 스펙, `models/model.py:15-51`)

Carreira'16 IEF / HMR(Kanazawa'18) 계열. 인용 그대로:

| 항목 | RoboPEPP 구현 | 근거 |
|---|---|---|
| feature `xf` | `img_feat.mean(dim=1)` — predictor 토큰 **전역 평균**, **모든 스텝 동일**(재추출·re-crop 없음) | `model.py:196` |
| 스텝 수 | `n_iter = 4` | `model.py:23` |
| 상태 init | `init_pose = zeros(npose)` = **정규화 공간의 0 = 관절 평균**(gt를 `(gt-mean)/std`로 정규화하므로 0→평균 denorm) | `model.py:31`, `test.py:214` |
| 스텝 입력 | `xc = cat([xf, pred_pose], 1)` — context에 **현재 θ 상태를 concat**(FiLM 아님, 단순 concat) | `model.py:42` |
| 회귀기 | `fc(feat+npose→1024)→drop.3→fc(1024→1024)→drop.3→decpose(1024→npose)` | `model.py:24-27` |
| 갱신 | `pred_pose = decpose(xc) + pred_pose` — **residual(=error feedback)**, absolute 아님 | `model.py:47` |
| init 트릭 | `xavier_uniform_(decpose.weight, gain=0.01)` — 초기 residual을 작게(안정) | `model.py:29` |
| loss | **최종 iterate만** `L1(pred_joints·std+mean, gt)` (deep-sup 아님) | `train.py:228`, `test.py:201` |
| 관절/포즈 분리 | θ는 IEF **회귀만**; 카메라 6DOF는 conf-filter kp로 **BPnP 별도** | §3, `test.py:252-258` |

**최소변경 이식 핵심**: (a) context feature 1회 계산 후 스텝마다 재사용, (b) 상태(θ or sin/cos) concat, (c) residual 갱신, (d) decpose 작은 init. 우리 frozen DINOv3 + crop 토큰 위 fused feature를 `xf`로 쓰면 그대로 성립.

---

## 8. 우리 IEF-head 구체 설계 (`TRAIN/model_angle.py` + `train_angle.py`, head-type `ief`)

### 8.1 클래스 (model_angle.py에 추가)

```python
# RoboPEPP Panda joint_mean (test.py:113) — IEF 상태 init(=평균 config)
PANDA_ANGLE_MEAN = torch.tensor([-0.0522, 0.2677, 0.0060, -2.0052, 0.0149, 1.9856])

class AngleHeadIEF(nn.Module):
    """RoboPEPP JointNet(model.py:15-51)의 IEF 이식. flat head는 fused-feature→각도를 1-shot 매핑해
    noisy(P0: good-frame oracle +0.11) → 솔버가 2D 과적합. IEF는 평균 config에서 시작해 n_iter 스텝
    residual 정제, 매 스텝 '현재 각도 상태'를 조건으로 넣어 운동학 체인 결합을 반복 해소(flat엔 없는
    구조 편향). frozen fused feature는 1회 계산(재-crop 없음), 각도 상태만 피드백."""
    def __init__(self, feat_dim=768, hidden=1024, n_ang=NUM_ANG, dropout=0.3, kp_in=None,
                 n_iter=3, deep_sup=True):
        super().__init__()
        self.n_ang, self.n_iter, self.deep_sup = n_ang, n_iter, deep_sup
        kp_in = kp_in or feat_dim
        # --- context encoder: AngleHead와 동일 fusion, 1회 계산 ---
        self.geo_mlp = nn.Sequential(nn.Linear(14+42,256), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(256,256), nn.GELU())
        self.conf_proj   = nn.Sequential(nn.Linear(NUM_KP,64), nn.GELU())
        self.global_proj = nn.Sequential(nn.Linear(feat_dim,256), nn.GELU())
        self.kp_proj     = nn.Sequential(nn.Linear(kp_in,128), nn.GELU())
        ctx = 256+64+256+NUM_KP*128
        # --- IEF 회귀기: [ctx, sin/cos 상태(2*n_ang)] -> residual sin/cos ---
        self.fc1 = nn.Linear(ctx + 2*n_ang, hidden); self.fc2 = nn.Linear(hidden, hidden)
        self.dec = nn.Linear(hidden, 2*n_ang)
        self.d1, self.d2 = nn.Dropout(dropout), nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.dec.weight, gain=0.01); nn.init.zeros_(self.dec.bias)  # 작은 초기 residual
        m = PANDA_ANGLE_MEAN[:n_ang]
        self.register_buffer('init_sc', torch.stack([torch.sin(m), torch.cos(m)], -1).reshape(-1))  # (2*n_ang,)

    def forward(self, geo, conf, gfeat, kpfeat):
        ctx = torch.cat([self.geo_mlp(geo), self.conf_proj(conf), self.global_proj(gfeat),
                         self.kp_proj(kpfeat).flatten(1)], dim=1)              # (B,ctx) 1회
        B = ctx.shape[0]
        sc = self.init_sc.to(ctx.dtype).unsqueeze(0).expand(B,-1).clone()      # 평균 config init
        iters = []
        for _ in range(self.n_iter):
            xc = self.d2(F.gelu(self.fc2(self.d1(F.gelu(self.fc1(torch.cat([ctx, sc], 1)))))))
            sc = sc + self.dec(xc)                                             # residual (IEF)
            scn = F.normalize(sc.view(B, self.n_ang, 2), dim=-1)              # 단위원 복귀
            iters.append(scn); sc = scn.view(B, -1)                           # 정규화 상태 피드백
        ang = torch.atan2(iters[-1][...,0], iters[-1][...,1])
        return (ang, iters[-1], torch.stack(iters, 1)) if self.deep_sup else (ang, iters[-1])
```

### 8.2 AnglePredictor 배선 (model_angle.py)
- `__init__` head-type 분기에 추가:
  ```python
  elif head_type == 'ief':
      self.angle_head = AngleHeadIEF(feat_dim=feat_dim,
                                     n_iter=int(os.environ.get('IEF_ITERS','3')))
  ```
- `forward`의 head 호출부(`else: ang, sc = self.angle_head(geo,conf,gfeat,kpfeat)` 앞)에 분기:
  ```python
  elif self.head_type == 'ief':
      r = self.angle_head(geo, conf, gfeat, kpfeat)
      if len(r) == 3: ang, sc, out_iters = r  # deep-sup
      else:           ang, sc = r; out_iters = None
  ```
  그리고 out dict에 `if out_iters is not None: out['sin_cos_iters'] = out_iters` 추가.
- `choices=[...]`에 `'ief'` 추가(train_angle.py `--head-type`, line 259).

### 8.3 학습 loss 분기 (train_angle.py, line ~161 `else: sc_loss=...` 앞에 삽입)
```python
elif args.head_type == 'ief' and 'sin_cos_iters' in out_d:
    it = out_d['sin_cos_iters']                         # (B,n_iter,6,2)
    w  = torch.linspace(0.5, 1.0, it.shape[1], device=it.device)   # 후반 iterate 가중
    per = F.smooth_l1_loss(it[has], gt_sc[has].unsqueeze(1).expand(-1, it.shape[1], -1, -1),
                           reduction='none').mean(dim=(2,3))        # (Nhas,n_iter) deep-sup
    sc_loss = (per * w).sum(1).mean() / w.sum()
    loss = sc_loss
```
기존 `fk_weight`(최종 `out_d['joint_angles']` = 마지막 iterate) 그대로 유효 — 추가 변경 불필요.

### 8.4 학습 명령 (즉시 실행 가능, DINOv3 배포 recipe와 동일)
```bash
cd /home/najo/NAS/DIP/3_pose_models/DINObotPose3/TRAIN
# free-memory 최대 GPU를 UUID로 (util 아님)
U=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1)
IEF_ITERS=3 CUDA_VISIBLE_DEVICES=$U /home/najo/.conda/envs/dino/bin/python train_angle.py \
  --detector-ckpt outputs_heatmap/crop_20260605_010622/best_heatmap.pth \
  --train-dir ../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr \
  --val-dir   ../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --head-type ief --image-size 512 --batch-size 32 --epochs 60 \
  --lr 1e-3 --min-lr 1e-6 --weight-decay 1e-4 --fk-weight 10.0 \
  --kp-jitter 2.0 --crop-to-robot --crop-margin 1.5 --num-workers 8 \
  --output-dir ./outputs_angle/ief_dr --use-wandb --wandb-run-name ief_dr
```
- **예상 시간**: IEF 루프(3×소형 MLP)는 frozen 백본 no_grad forward 대비 무시가능 → 기존 mlp angle(50ep)과 사실상 동일 wall-clock, 60ep ≈ +20%. A6000 1장.
- **하이퍼**: `IEF_ITERS=3`(RoboPEPP 4 — 3부터, val MAE로 4 스윕), `--kp-jitter 2.0`(2D 노이즈 강건화, 배포 recipe엔 없지만 P0가 "노이즈 과적합" 지목 → 소량 권장). deep-sup는 기본 on.

### 8.5 검증 게이트 (기존 하네스 재사용)
1. **val angle MAE**를 배포 mlp(9.09°)·RoboPEPP Tab.3(DR 평균 3.8°, J2–J4 1.7–2.5°)와 대조. **base J0 미퇴화**(제로섬 재발 방지 — 07-21 MoE 교훈) 필수.
2. good-frame ADD: `Eval/eval_synth_head.sh --crop-head-type ief` → base 0.7884에서 상승하는지(목표 oracle 0.899로의 접근). **fail%(>100mm) do-no-harm.**
3. 통과 시 **P0a 재실행**(freeze-θ=IEF-θ + 6DOF BPnP) — IEF θ가 정확하면 이번엔 이득 예상(P0×P1 시너지 검증).

---

## 9. 보조 레버 우선순위 (2D-노이즈 과적합 정규화)

| 레버 | 판정 | 근거 |
|---|---|---|
| **IEF 회귀기 구조** (P1) | ✅ **1순위** | flat 천장 0.788의 head 가설을 직접 판정, frozen 호환, 싸다 |
| kp-jitter train-aug | ✅ 병행(무료) | 이미 `--kp-jitter` 존재; P0가 지목한 "노이즈 과적합" 완화 |
| **reproj-consistency 항** | ❌ **반증(재시도 금지)** | [2026-07-04_robotag_reproj_consistency.md] azure ADD −0.014, angle MAE도 악화. head-레벨 이식으론 RoboTAG azure 우위 재현 안 됨 |
| **VPoser/NRDF 학습 config prior** | ⚠️ **저EV(권장 안 함)** | DREAM synth는 config가 ~균등·독립 → prior 무정보. "모집단 통계 prior" 이미 반증(−0.09@20%, [next_directions §3]). 실로봇 궤적에서만 유효 |
| 별도 학습형 관절-net (P1b) | 🔜 **폴백** | IEF가 frozen feature 천장에 막히면 feature 자체를 바꾸는 유일 수단. 키포인트 백본 무접촉이라 반증 대상 아님 |

**결론**: 정규화/prior 계열은 대부분 반증·저EV. 실질 레버는 **회귀기 아키텍처(IEF)** 하나로 좁혀지며, 그것이 막히면 **별도 학습형 관절-net**이 유일 폴백. reproj·config-prior는 재시도하지 말 것.

---

## 10. 탐색·비평 로그 (07-22, IEF 4종 학습 중) — 로컬 SOTA 코드 정밀 리뷰

> 로컬 clone: RoboPEPP · HoRoPose(`Holistic-Robot-Pose-Estimation`) · CtRNet-X. 세 SOTA의 head/loss/solver를 파일:라인으로 대조. **핵심 발견: 세 방법의 각도 head가 전부 동일(pooled-feature IEF) → head는 그들의 레버가 아니다. 우위는 co-trained feature + camera-frame FK/reproj loss에 있고, 둘 다 우리 frozen 세팅에선 부분적으로만 이식된다.**

### 10.1 SOTA 각도회귀 해부 — head는 3사 동일, 정확도 원천은 loss/feature
| | 우리(계획 IEF) | RoboPEPP | HoRoPose |
|---|---|---|---|
| head 구조 | pooled-feature IEF | **동일** IEF (`model.py:15-51`) | **동일** IEF (`full_net.py:318-331`) |
| n_iter / init | 3 / 평균 | 4 / 평균(=norm 0) | 4 / 평균(`init_pose_from_mean`) |
| 표현 | sin/cos | normalized angle | **raw radian** |
| feature | frozen DINOv3 pooled | **co-trained** predictor pooled(`:196`) | **co-trained** avgpool(`:294,310`) |
| soft-argmax(integral)? | — | — | **키포인트에만**, 각도엔 안 씀(`reg_joint_map=False` 전 config; `integral.py` `HeatmapIntegralJoint` 미사용) |
| **각도 정확도 driver** | (미정) | masked-pretrain + co-train | **미분FK 3D loss ×10 + FK-reproj 2D ×10** vs angle-MSE ×1 (`function.py:256-268`) |

**판정 A1/A2(각도정확도 원천)**: integral-soft-argmax **아님**(키포인트용), aug **아님**(2차), 아키텍처 **아님**(head는 pooled MLP로 우리와 동형). 진짜 = **미분FK/재투영 multi-task loss(×10)** + **co-trained feature**. **IEF(iterative)는 2차 기여** — 세 SOTA가 같은 head를 쓴다는 사실 자체가 "head는 차별점 아님"의 증거. **∴ IEF는 옳은 선택이지만 "mlp보다 크게 나은 레버"라는 기대는 제한적.**

### 10.2 우리가 이미 하는 것 vs 아직 안 하는 것 (HoRoPose 레시피 대조)
- ✅ 이미: `--fk-weight 10`(robot-frame FK MSE, `train_angle.py:169`) — HoRoPose `loss_error3d`의 robot-frame 판.
- ❌ 아직/반증: HoRoPose `loss_error2d`(**camera-frame 재투영** ×10, **co-trained pose**로). 우리 head-level `--reproj-weight`(GT-pose Kabsch 재투영)는 **이미 반증**([2026-07-04], azure −0.014).
- **판정**: HoRoPose 각도 레버의 transferable 부분(robot-frame FK)은 **이미 적용 중**. IEF가 더하는 건 head의 iteration+feedback뿐 → 기대 상향폭 modest. **높은-가치 미이식분(camera-frame reproj)은 co-trained pose 전제라 우리 frozen head엔 반증됨.**

### 10.3 판정 A1(해석적 IK / HybrIK twist-swing): 로봇엔 실효 없음
- HybrIK: swing(bone 방향)=3D 키포인트 **위치에서 해석적**, twist(roll)만 회귀. 로봇 대응 = swing 관절은 키포인트로 결정, roll만 appearance.
- **FK 민감도 실측**(dump `dr_pred.npz`, good frames, `panda_forward_kinematics`): **proximal J1–J4 = 3.4–5.4 mm/deg(고레버·관측가능)**, **wrist J5/J6 = 0.28–0.56 mm/deg(저레버·ADD-benign)**. → ADD를 지배하는 건 **proximal swing**(이미 2D 관측·솔버가 품), roll은 ADD 거의 무영향(J5 write-off 정당).
- **결정적 반론**: 해석 IK는 **3D 키포인트 위치**가 있어야 하는데 우리는 **2D만** → 3D는 FK(θ)로부터 = 순환. 우리 재투영 솔버가 이미 "2D→θ"의 gradient 해(解). **HybrIK식 해석 IK가 추가로 주는 것 없음.**
- 단 HybrIK **adaptive** 통찰 1개는 유효: naive FK는 체인 오차 누적(=우리 basin-flip), adaptive는 관측된 child로 링크를 재고정 → 오차 **국소화**. 이게 §1.2 "proximal 앵커"의 원리이며 **P0b(per-joint freeze)가 이미 이 방향**.

### 10.4 판정 B(CtRNet-X): 결합솔버 basin-flip 대안 = 구조적으로 맞으나 unknown-angle 직접이식 불가
- CtRNet BPnP는 **6-DOF rigid만** 품(`CtRNet.py:80-82,208-214`), θ는 안 품 → *"a bad wrist keypoint can only corrupt the 6-DOF camera pose"* = **basin-flip 구조적 회피**. 우리 P0(regress-θ + 6DOF-only)와 동일 원리 독립 확인. BPnP=declarative IFT(Chen) + EPnP init.
- **단 CtRNet은 관절각을 proprioception에서 읽음(known-angle), regress 안 함** — decoupling이 더 강한 가정. unknown-angle 회귀는 안 풂. (CtRNet-X는 VLM으로 가시 링크 키포인트 선택 추가.)
- **이식가치 recipe 2개**: (i) DARK가 **Hessian near-singular 또는 이미지 경계 2px 이내** 키포인트를 마스크 + `conf_thresh 0.08` + **최소 5점** — 우리 edge-gate/conf-gate의 검증된 형태(P0b 튜닝 참고). (ii) **RANSAC/outlier reject**은 우리 Adam 솔버에 없음 — 단 tail은 AUC 0 기여라 저순위.

### 10.5 적대적 비평 — IEF가 mlp-control 못 넘길 시나리오·tell (조기판정 ep~20–30)
- **시나리오(확률 높음)**: 천장이 head가 아니라 **frozen feature의 정보한계**면 IEF≈mlp. §10.1이 이를 시사(SOTA도 같은 head인데 **co-trained feature**로 이김).
- **tell**:
  1. IEF val MAE ≈ mlp-control(±0.3° 이내) → **feature-bound. IEF 폐기 → P1b(별도 학습형 관절-net)**.
  2. `i3≈i4≈i5`(iteration 단조성 없음) → iteration 무정보 = feature-bound.
  3. train MAE↓·val MAE flat → 용량 아닌 feature가 병목.
  - **성공 tell**: IEF MAE가 mlp보다 유의미↓(예 9→7°) **AND** `i5>i4>i3` → head-bound, eval ADD 진행.
- **필수**: base **J0 미퇴화** 확인(07-21 MoE 제로섬 재발 방지).

### 10.6 good-frame 진단 airtight? — 대체로 yes, 정정 1개
- oracle-angle = **GT θ + SOLVED(R,t)** (`selfbbox_eval.py:366` freeze_theta; **포즈는 GT 아님, 솔버가 품**) → **0.899는 "각도가 완벽하면"의 도달가능 상한, GT-포즈 아티팩트 아님. airtight.**
- **정정**: 0.788→0.899엔 두 성분 혼재 — (a) 정확한 3D, (b) freeze로 **13DOF→6DOF 솔브 조건수 개선**. (b)는 각도가 정확할 때만 발현(P0a naive=0.533이 증거) → **P0×P1 시너지** 재확인.
- **정량화**(FK 민감도): 0.788→0.899(≈good ADD 21→~10mm, Δ11mm) = **proximal ~3°/joint 개선**(4–5mm/deg). RoboPEPP 3.8° 도달 시 대부분 회수. mlp-control은 동일 프로토콜 → **매칭 공정**.
- **반례 주의(HoRoPose)**: HoRoPose는 ~9° 각도로도 높은 ADD → 각도 외 **pose/depth 레버** 존재 가능성(RootNet `k_value` depth prior). 단 **우리 oracle 0.899 > HoRoPose 0.82** → 우리에겐 각도가 더 큰 헤드룸. pose/depth는 2차(P2).

### 10.7 클론 필요 repo 판정 → **충분 (추가 불필요)**
- 로컬 RoboPEPP + HoRoPose + CtRNet-X로 세 SOTA head/loss/solver 전부 정밀 확인 완료.
- **HybrIK**: 해석 IK가 로봇에 실효 없음(§10.3) → **클론 불필요**.
- **RoboKeyGen**: 키포인트-조건부 diffusion angle solver(appearance 미사용 → **frozen-feature 천장 우회**)는 개념적으로 유일한 미탐색 각도지만 depth(SPDH) 입력 필요 + DREAM 미평가. **IEF/별도-net이 막힌 뒤에만 재검토** → 지금 클론 불필요.
- **폴백 순위(IEF 실패 시)**: P1b(별도 학습형 관절-net, feature 자체 학습) > RoboKeyGen식 kp-conditioned solver.

### 10.8 다음 배정 권고
1. IEF **ep~25 조기 tell**(§10.5) → **feature-bound면 즉시 P1b 착수**(별도 학습형 관절-net = SOTA가 이기는 "co-trained feature"의 frozen-호환 유일 경로, 키포인트 백본 무접촉).
2. 병렬 저비용: HoRoPose식 **camera-frame FK-reproj를 예측-pose(rot-head `with_translation`)와 함께** — head-level GT-pose reproj(반증)와 달리 **예측-pose** 재투영이라 재시도 가치 있음. 단 azure 반증 이력 유의, do-no-harm 게이트.

---

## 11. P1b 설계 — 각도 branch에 co-trained feature 주입 (frozen 키포인트 백본 불변)

> **동기(§10 결론)**: 세 SOTA head가 우리와 동형(pooled-IEF)인데도 이기는 이유 = **co-trained feature**. frozen-DINOv3-pooled 각도 천장(0.788)은 head가 아니라 **feature 정보한계**일 공산이 큼(IEF tell이 확정). P1b = 그 유일 frozen-호환 탈출로.
>
> **반증 "공유 백본 적응"과의 구조적 차이 (핵심)**: 반증은 *키포인트 토큰을 만드는 바로 그 백본*을 적응 → 토큰이 이동 → 키포인트 head가 sub-pixel 정밀도 상실(솔버 붕괴). P1b는 **물리적으로 분리된 trainable 네트워크**가 각도 feature를 만들고, 그 gradient는 **DINOv3 토큰에 절대 도달하지 않음.** 키포인트는 **변경되지 않은 frozen DINOv3**를 계속 읽음 → 정밀도 상실 실패모드가 **구조적으로 불가능**. 각도 feature만 task-적응되는 것은 반증이 못 했던 **원하던 동작**.

### 11.1 아키텍처 선택 — **(b) 분리 trainable ResNet50 추천**

| 옵션 | 효과 | 리스크 | frozen-kp 무결성 | 판정 |
|---|---|---|---|---|
| **(a) frozen DINOv3 + 각도전용 LoRA/adapter** | 경량, DINOv3 특징 활용 | **여전히 DINOv3 특징 경유** — 천장이 DINOv3 귀납편향/pooling이면 저용량 LoRA로 못 탈출. 키포인트 보호하려면 어차피 별도 forward 필요(비용 절감 안 됨). "co-trained feature 필요" 가설의 **가장 약한 검정** | 별도 forward 시 보장 | 저EV |
| **(b) 분리 trainable ResNet50 (ImageNet-init, @256)** | **HoRoPose가 정확히 이 recipe로 목표 각도 달성**(`full_net.py:75` `get_resnet("resnet50")`, `image_size 256`). frozen 특징에 **독립** → 천장에 못 갇힘. 최청정 decoupling | 큰 학습(전 백본), DINOv3 특징 미활용 | **자명**(별도 네트워크) | ✅ **1순위** |
| **(c) frozen DINOv3 마지막 N블록 trainable 복제** | DINOv3 init(강) + co-train + 키포인트 무결(키포인트=frozen 복사본) | 무거움(ViT-B N블록 복제+활성), DINOv3 frozen 초기층 표현 상속, 2-경로 forward 구현 부담 | 보장(경로 분리) | 폴백 2순위 |

**추천 = (b).** 근거: (i) **frozen-feature 천장에 원리적으로 독립한 유일 옵션**((a)/(c)는 DINOv3 경유), (ii) HoRoPose가 **동일 recipe로 실증**(ImageNet-init ResNet50 + FK loss), (iii) 키포인트 무결성 자명(분리 네트워크), (iv) DR train **104,973장** + FK×10로 co-train 충분. 코디네이터 우려("from-scratch가 DINOv3 이길까")에 답: **from-scratch 아님(ImageNet-init)**, 그리고 요구 기준은 "DINOv3를 일반 특징으로 이기기"가 아니라 "**frozen-DINOv3-pooled를 각도회귀에서** 이기기(0.788 돌파)" — HoRoPose가 넘는 바. (b)가 실패하면 (c)로 DINOv3 prior+co-train 결합.

### 11.2 loss — FK 3D(기존) + 예측-pose camera-frame reproj(신규)
- **코어(P1b-FKonly)**: sin/cos SmoothL1 + **FK 3D robot-frame MSE ×10**(기존 `--fk-weight 10`, `train_angle.py:169`). 이미 HoRoPose `loss_error3d`의 robot-frame 판.
- **신규(P1b-reproj)**: **예측 pose로** camera-frame 재투영 ×10. **반증 버전과의 차이**: 반증(2026-07-04)은 **GT-pose Kabsch**로 재투영 → 고정 GT 기준에 각도만 묶어 frozen head에선 좋은 해와 싸움(azure −0.014). 신규는 **같은 ResNet50이 rot-head(R, `with_translation` t)도 구동** → `reproj = project(FK(θ)|R̂,t̂) vs GT2D`로 **각도·pose를 상호 reproj-일관되게 co-train**(HoRoPose 메커니즘). pose가 학습되므로 고정기준과 안 싸움.
- **azure do-no-harm 게이트**: azure=근거리 real(우리 depth/RC off, reproj 과거 −0.014). → **두 arm 학습**(FKonly / FK+reproj), 4 real split 평가. **reproj arm은 azure held-out ADD ≥ 기준 AND 타 카메라 개선일 때만 채택**; 아니면 FKonly P1b 배포. reproj는 ablation, 기본 아님.

### 11.3 코드 이식 (최소변경, `--angle-backbone` 신규 플래그)
- **`train_angle.py`**: `--angle-backbone {dino_frozen(기본), resnet50}` 추가(line ~259 인근). optimizer(line 101)를
  ```python
  params = list(model.angle_head.parameters())
  if args.angle_backbone == 'resnet50':
      params += list(model.angle_feat.parameters())   # trainable 백본 포함
  opt = optim.AdamW([{'params': model.angle_head.parameters(), 'lr': args.lr},
                     {'params': getattr(model,'angle_feat',nn.Module()).parameters(), 'lr': args.lr*0.2}],
                    weight_decay=args.weight_decay)   # 백본 LR = head×0.2
  ```
- **`model_angle.py` `AnglePredictor.__init__`**: DINOv3 detector(backbone+keypoint_head)는 **그대로 frozen 유지**(kp2d/conf 전용). 추가:
  ```python
  self.angle_backbone = angle_backbone
  if angle_backbone == 'resnet50':
      import torchvision as tv
      m = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
      self.angle_feat = nn.Sequential(*list(m.children())[:-2])   # conv trunk -> (B,2048,h,w)
      afd = 2048
      self.angle_head = AngleHeadIEF(feat_dim=afd, kp_in=afd, n_iter=int(os.environ.get('IEF_ITERS','4')))
      if with_rotation: self.rot_head = RotationHead(feat_dim=afd, predict_t=with_translation)
  ```
- **`forward`**: kp2d/conf는 기존대로 **frozen DINOv3 no_grad**로. 각도 특징만 교체:
  ```python
  if self.angle_backbone == 'resnet50':
      x256 = F.interpolate(image, size=256, mode='bilinear', align_corners=False)
      fmap = self.angle_feat(x256)                    # (B,2048,8,8) trainable, grad on
      gfeat = fmap.mean(dim=(2,3))                     # GAP -> (B,2048)
      kpfeat = sample_kp_features_map(fmap, kp2d, self.heatmap_size)   # grid_sample @ kp (신규 helper, 기존 sample_kp_features의 conv-map 판)
      geo = keypoints_to_geo(kp2d, camera_K); conf = conf
      ang, sc, sc_it = self.angle_head(geo, conf, gfeat, kpfeat)
  ```
  `freeze_detector()`는 불변(DINOv3+kp head만 freeze; `angle_feat`은 trainable 유지).
- **속도 최적화(선택)**: kp2d/conf/geo는 frozen detector 출력이라 **학습 중 불변** → train set 1회 dump 후 캐시 로드 시 DINOv3 forward 생략, ResNet만 학습(대폭 가속).

### 11.4 학습 명령 (즉시 실행 가능)
```bash
cd /home/najo/NAS/DIP/3_pose_models/DINObotPose3/TRAIN
U=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1)
IEF_ITERS=4 CUDA_VISIBLE_DEVICES=$U /home/najo/.conda/envs/dino/bin/python train_angle.py \
  --detector-ckpt outputs_heatmap/crop_20260605_010622/best_heatmap.pth \
  --train-dir ../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr \
  --val-dir   ../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --angle-backbone resnet50 --head-type ief --image-size 512 --batch-size 64 --epochs 80 \
  --lr 3e-4 --min-lr 1e-6 --weight-decay 1e-4 --fk-weight 10.0 \
  --kp-jitter 2.0 --crop-to-robot --crop-margin 1.5 --num-workers 8 \
  --output-dir ./outputs_angle/p1b_resnet50 --use-wandb --wandb-run-name p1b_resnet50
```
- **예상시간**: 병목 = frozen DINOv3-512 forward(kp2d, no_grad) + ResNet50-256 fwd/bwd. 105k/bs64 ≈ 1640 it/ep. A6000 1장 ~6–10 min/ep → 80ep ≈ **8–13h**(캐시 최적화 시 절반). IEF 4종과 다른 GPU 병렬.
- **reproj arm**(게이트 통과 시): 위에 `--reproj-weight 10 --with-rotation --with-translation` 추가한 2번째 런.
- **스모크(투입 전 5분)**: `--epochs 1 --batch-size 8` 1회 — (i) 손실 하강, (ii) `angle_feat.requires_grad=True`·DINOv3 `requires_grad=False` 어써트, (iii) val MAE 산출 확인.

### 11.5 판정 — 성공/실패 지표와 시점
- **PASS(→ 배포 후보)**: val angle MAE가 frozen-DINOv3 mlp **9.09°를 명확히 하회**(목표 HoRoPose/RoboPEPP 4–6°) **AND** `eval_synth_head.sh --crop-...`로 good-frame base **0.788 → 상승**(oracle 0.899 접근) **AND base J0 미퇴화**. 조기판정 **ep~30**(MAE plateau), 최종 ep80.
- **FAIL(→ 가설 기각)**: val MAE ≈ frozen mlp(9°대). 이는 천장이 **feature가 아니라** 데이터/라벨/2D-degeneracy임을 의미 → **RoboKeyGen식 kp-조건부 solver**(appearance 무관, 2D→θ diffusion)로 escalate 또는 천장 수용.
- **결정적 대비**: IEF(frozen)와 P1b(resnet)를 **동일 val로 병렬 비교** — IEF≈mlp AND P1b≫mlp면 **"feature가 병목" 확정**(P1b 배포). IEF≫mlp면 head도 유효(둘 결합: ResNet+IEF).

---

## 12. 멀티로봇 SOTA-근접 계획 — KUKA iiwa7 / Baxter (07-22, GPU 불필요 오프라인 분석)

> 새 공식 목표: **각 DREAM 로봇에서 개별 모델이라도 SOTA 근접.** Panda는 IEF/P1b 진행 중. 여기선 KUKA/Baxter를 목표에 올리기 위한 **경쟁 목표수치 + 병목 재진단(Panda 렌즈) + Panda 승자별 확장 시나리오**. 근거: `2026-07-10_multirobot_dream_detectors.md`, 로컬 로그(`Eval/mr_logs/`,`synth_logs/`), RoboPEPP configs(`kuka.yaml`/`baxter.yaml`).

### 12.1 현재 성적 + 경쟁 목표 (synth ADD-AUC@100mm, ×100)

**우리(실측, `Eval/synth_logs/`, direct-pose, 5000f)**: KUKA **35.7**(DR)/**31.9**(Photo), Baxter **25.2**(DR).
> **Baxter는 Photo set이 존재하지 않음**(DREAM 데이터에 `baxter_synth_test_dr`만; RoboTAG §4.1 확인) → Baxter는 DR만 평가.

**⚠️ 프로토콜 2종 — 교차비교 금지(핵심 caveat)**: 문헌에 호환 안 되는 두 eval이 공존. **Protocol A**(RoboPose/HoRoPose/RoboTAG 공유, RoboTAG Tab.1) vs **Protocol B**(RoboPEPP 자체 Tab.2, KUKA에서 A보다 ~7–10점 높음). 같은 방법·로봇도 값이 크게 다름(RoboPEPP Baxter-DR 자체보고 **75.3**(B) vs RoboTAG 재평가 **34.4**(A)). **우리 auto-bbox는 Protocol A camp에 대응**(Panda 비교와 동일) → **A를 1차 목표로.**

**Protocol A (predicted-angle, RoboTAG Tab.1 직독 + RoboPose 원값):**
| 방법 | KUKA-DR | KUKA-Photo | Baxter-DR | 각도 |
|---|---|---|---|---|
| DREAM (RoboPose 재평가) | 73.3 | 72.1 | 75.5 | **known** |
| RoboPose | **80.2** | 73.2 | 32.7 | predicted |
| HoRoPose | 75.1 | 73.9 | **58.8** | predicted |
| RoboTAG | 75.0 | **76.6** | **58.8** | predicted |
| RoboPEPP (RoboTAG 재평가) | 76.2 | 76.1 | 34.4 | predicted |
| **Ours (direct-pose)** | **35.7** | **31.9** | **25.2** | predicted |

(참고 Protocol B, RoboPEPP 자체: KUKA-DR 83.0/Photo 80.5/Baxter-DR 75.3 — **우리와 비교 부적합**. DREAM 원논문은 KUKA/Baxter 수치 **없음**(§III.D "did not perform quantitative analysis"); 그 73.3/75.5는 **RoboPose 재평가**값 — 출처 표기 주의.)

**격차·"근접" 재정의(수치 확정 후 — 로봇별 상황이 크게 다름)**:
- **KUKA-DR**: 우리 35.7 vs 프론티어 **75–80**(RoboPose 80.2). **격차 ~0.40–0.44, 큼.** KUKA는 경쟁자 전원 0.75–0.80로 **본질적으로 어렵지 않음** → 우리만 낮음. 1차 마일스톤 KUKA-DR **0.55+**, SOTA-근접 **0.72+**.
- **Baxter-DR**: 우리 25.2 vs **RoboPose 32.7 / RoboPEPP(재평가) 34.4에 근접**(−0.07~0.09), 프론티어는 **HoRoPose/RoboTAG 58.8**. **Baxter-DR은 predicted-angle에 본질적으로 어려움**(RoboPose·RoboPEPP도 0.33대) → **wrist 관측성이 공통 한계**(Panda off-frame과 동형). 프론티어 58.8은 wrist가 아니라 **pose/depth 우위**. 1차 마일스톤 Baxter-DR **0.40+**(RoboPose 상회), SOTA-근접 **0.55+**.

### 12.2 병목 재진단 (Panda 렌즈: 각도 vs rot-head vs off-frame) — 로그 실측

**공통**: KUKA/Baxter는 배포 모드가 `--direct-pose`(head 각도 + rot-head R,t **직접**, 2D 솔버 **미사용**) → **Panda의 off-frame/basin-flip 축은 무관**(재투영 솔버를 안 씀; 검출기 link-혼동 tail은 direct-pose가 우회). 따라서 ADD = **FK(head 각도) 형상 + rot-head (R,t)** 두 성분만.

**KUKA — 병목 = rot-head (각도 아님)**:
- per-joint MAE 실측(`mr_logs/kuka_directpose.log`): J0–J5 = **5.5/4.5/7.1/4.4/7.3/9.0°** (전부 양호, Panda good-frame급).
- **GT각도 == head각도 ADD 동일**(0.34, doc §최종 ADD) → **각도 병목 아님**(oracle-angle 무이득).
- 병목 = **rot-head R 7.4° + t 56mm**(t=depth). 세부: R은 feature-개선 여지, **t(depth)는 monocular 본질** — Panda는 RC로 풀었으나 **KUKA RC는 정확 mesh 부재로 차단**(bullet3 iiwa 변종 20mm 오차, RoboPose rclone 사망).

**Baxter — 🔴 병목은 wrist가 아니라 pose (§13.4에서 정정)**:
- per-joint MAE 실측(`mr_logs/baxter_directpose.log`): J0–J3 = 6.6/4.5/10.6/7.1°, **J4/J5(wrist) = 25.3/21.0°** — MAE만 보면 wrist가 최악.
- **그러나 FK 레버암 실측(§13.4): wrist 25°/21° 오차의 키포인트 변위는 각 5.7/6.0mm, 합 ~8mm** (체인 말단이라 레버암 거의 없음). **wrist를 완벽히 고쳐도 ADD-AUC +0.005** = 무의미.
- **실제 격차 = pose(t 60mm + R 5.7° → 39mm)**. pose만 고치면 모델 AUC 0.28→0.72. ⇒ **"wrist 관측성 천장"은 angle-MAE 서사였을 뿐 ADD 병목이 아님**(frozen mlp_patch 반증도 사실상 무해한 대상에 쓴 것).

**핵심 답 (Panda 승리 fix가 KUKA/Baxter에 통하나?)** — 로봇마다 다름:
| 로봇 | 병목 | P1b(co-trained 각도 feature) 적용성 | 확신 |
|---|---|---|---|
| Panda | good-frame proximal **각도** | **직격** | 높음 |
| Baxter | **wrist 각도**(관측성) | wrist-roll을 **appearance로 읽기** — frozen mlp_patch(반증)의 **co-trained 강화판**. Meca 선례(appearance head로 wrist 돌파)로 재시도 가치. 단 관측성이 근본이면 feature 용량만으론 미해결 | **중** |
| KUKA | **rot-head R,t**(각도 아님) | 각도head P1b는 **무효**. 단 **P1b의 trainable ResNet이 rot-head도 구동** → KUKA rot-head(현 frozen-DINO-pooled)를 co-trained feature로 개선(R↑ 기대; t/depth는 RC 필요) | R=중 / t=낮음 |

**통합 통찰(중요, 경쟁 수치로 강화)**: 세 로봇의 현 천장이 전부 **frozen-feature head**이고 **P1b ResNet이 angle+rot 모두 구동** → 단일 백본으로 3로봇 잠재 상향(확신 Panda>Baxter-wrist>KUKA-R≫KUKA-t). **그러나 경쟁 수치가 진짜 SOTA 레버를 드러냄 — 각도가 아니라 pose/depth다**:
- **Baxter**: 우리 25.2 ≈ RoboPose 32.7·RoboPEPP 34.4 → **wrist 관측성은 경쟁자 전원의 공통 한계**(Panda off-frame과 동형, "명백한 병목이 실은 공통 한계"). 프론티어 HoRoPose/RoboTAG **58.8은 wrist가 아니라 pose/depth**(HoRoPose **RootNet depth head**)로 달성. → **Baxter SOTA 레버 = depth, wrist 아님.**
- **KUKA**: 경쟁자 전원 75–80(우리 35.7). 우리 병목(rot-head **t 56mm=depth**)이 곧 경쟁자가 RC/RootNet으로 푸는 그것. → **KUKA SOTA 레버 = depth.**
- **∴ KUKA·Baxter 공통 SOTA 레버 = pose/depth(rot-head t / RootNet식 depth head / RC)**. **P1b(각도 feature)는 2차** — Baxter wrist·KUKA R에 소폭, 격차 0.4의 대부분은 depth라 **P1b만으론 SOTA-근접 불가.** 이는 Panda의 메타교훈(명백한 병목=공통 한계, 진짜 격차는 다른 곳)의 멀티로봇 재현.

> ⚠️ **오프라인 한계**: KUKA/Baxter add_eval은 per-frame dump가 없어(`np.savez` 미구현) Panda식 good/tail **AUC 분해 불가**. 집계(per-joint MAE, oracle-R rot 분리)로 병목은 확정됨. **정밀 good/tail 분해가 필요하면**: `kuka/baxter_add_eval.py`에 Panda식 `--dump`(fid/theta/kp_cam/gt3d/reproj) 추가 후 5000f 재평가(짧은 GPU, GPU 여유 시).

### 12.3 확장 계획 — Panda 결정실험 승자별 시나리오

**시나리오 (a): P1b(ResNet 각도백본) 승** →
1. Panda 배포.
2. **Baxter 각도head P1b 재학습**(`--angle-backbone resnet50 --fk-robot baxter --angle-joint-names left_s0,left_s1,left_e0,left_e1,left_w0,left_w1,left_w2`) — **wrist-roll appearance 재시도**(frozen mlp_patch 반증의 강화판). 각도 병목 직접 → **우선순위 1**. GPU잡 ~8–13h.
3. **KUKA rot-head를 P1b화**(rot-head가 ResNet feature 소비 → R 개선). t는 RC 대기. **우선순위 2**. ~8–13h.
4. (선택) 같은 ResNet이 angle+rot 동시 구동하는 **통합 P1b 백본**(HoRoPose식 멀티head) — 3로봇 공통.

**시나리오 (b): IEF(frozen head) 승 = frozen feature로 충분** →
- Baxter도 **IEF 각도head 재학습**(저비용, `--head-type ief --fk-robot baxter`). KUKA rot-head는 **IEF-style iterative rot** 시도.
- 단 IEF 승은 "frozen feature 천장이 head였다"는 뜻 → Baxter wrist(관측성)엔 무력할 공산(feature 아닌 기하 한계). KUKA rot-R엔 소폭 기대.

**시나리오 (c): 둘 다 mlp-control 못 넘음(frozen feature ceiling 확정)** →
- 각도는 근본 한계 수용. **KUKA = RC용 정확 iiwa7 mesh 외부 확보**(Panda RC +0.10 레버 재현), **Baxter = 관측성 벽 수용**(temporal/멀티뷰 외 방법 없음).

**로봇별 우선순위·GPU·시간(Panda 실험 종료 후)** — *경쟁 수치가 depth를 1순위로 밀어올림*:
| 순위 | 작업 | 근거 | GPU잡 | 예상 |
|---|---|---|---|---|
> ⚠️ **아래 표는 §13.7(KUKA RC 해제)로 재작성됨.** 최신 순서는 §13.7 참조.

| **P1** | **Baxter rot-head 학습 완주**(Ep11→40) | rot-head **미수렴** 발견(§13.5). KUKA 선례 t 82→56mm·AUC **0.22→0.34**. 신규 코드 0 | 1 GPU | 4–8h |
| **P3** | **Baxter RC 수리**(Panda식 앵커링+do-no-harm 게이팅 이식) | mesh 0.00mm·SAM IoU 0.82 **자산 보유**, 실패는 방법(발산). test-time, 학습 불필요 | — | 저비용 |
| **P4** | **rot-head R 개선**(P1b feature 또는 IEF-rot) | §13.4: t와 R 동등 기여. **depth 단독 0.19→0.38, 둘 다여야 0.72** | 1–2 GPU | 8–13h |
| ~~폐기~~ | ~~Baxter 각도head P1b(wrist appearance)~~ | **§13.4: wrist 완전수정도 +0.005** → EV 없음 | — | — |
| ~~불필요~~ | ~~Panda RootNet~~ | **Panda는 RC가 최대 레버(+0.043) — 유지** | — | — |

### 13.7 Q4 — KUKA RC 해제 후 우선순위 재작성 (RC vs RootNet)

**핵심 판단: KUKA는 RC 우선, RootNet은 조건부 강등(폐기 아님).**

| 축 | **RC(실루엣 depth/scale 보정)** | **RootNet depth head** |
|---|---|---|
| 검증 상태 | ✅ **우리 스택에서 검증된 최대 레버**(Panda +0.043) | ❌ 우리 스택 미검증(HoRoPose 논문 근거뿐) |
| 비용 | **test-time, 학습 0** | 학습 4–8h/로봇 |
| 겨냥 대상 | **t(깊이/스케일)** = KUKA 병목 정확히 일치 | t의 **z 성분만** |
| 차단 요인 | ~~메쉬 부재~~ → **해제됨**(§13.6) | z/xy 분해 **미측정**(§13.5)에 게이트됨 |
| 리스크 | 발산(Baxter 선례) — **체크리스트로 방어 가능** | 학습 후 무효일 수 있음 |

**⇒ RC를 먼저 하는 게 명백**: 비용 0, 검증됨, 병목과 정확히 맞물림, 차단 해제됨. RootNet은 **RC가 실패할 때의 대안 또는 RC의 init 개선재**로 보류.

**단 3가지 유보(추측 명시)**:
1. **RC의 capture range 미확인** — Panda RC는 **이미 좋은 init**에서 출발한다(rot-head가 basin을 잡아줌). KUKA init은 **R 7.4° 오차**로 Panda보다 나쁘다. 실루엣 정합이 이 오차에서 수렴할지는 **미검증(추측)**. → RC 1차 실행은 **capture-range 진단을 겸해야** 한다(개선/무해/발산 3분류).
2. **RC가 t를 얼마나 줄이는지 우리 KUKA에서 미측정** — Panda 실적(+0.043)을 KUKA에 그대로 옮기는 것은 **추측**. 민감도(§13.4)상 t 56→20mm면 모델 AUC 0.19→0.38.
3. 🔴 **RC만으론 SOTA-근접 불가(실측 기반 산술)** — RC는 **depth/scale 보정기라 R을 고치지 않는다**. §13.4: t를 완벽히 고쳐도 **R 7.4°가 남으면 0.38 천장**. 목표 0.72는 **R≤2.5° 동시 필요**. ⇒ **rot-head R 개선은 depth 경로와 무관하게 필수**이며, RC 해제로 **오히려 R이 단독 최대 병목으로 승격**한다.

**재작성된 KUKA 순서**:
| 순위 | 작업 | 근거 | 비용 |
|---|---|---|---|
| **K1** | **iiwa7 RC 배선 + 1차 실행**(§13.6 체크리스트 준수) | 검증된 레버, 비용 0, 차단 해제. capture-range 진단 겸용 | test-time |
| **K2** | **rot-head R 개선**(P1b co-trained feature / IEF-rot) | **RC로도 못 고치는 축**. 0.38→0.72의 필요조건 | 8–13h |
| K3 | t-err **z/xy 분해 측정** | RootNet 필요성 판정 게이트(RC 결과와 함께 해석) | ~5분 |
| K4 | RootNet depth head | **K1 실패(발산/capture range 초과) 시에만**, 또는 RC의 init 개선용 | 4–8h |

**Baxter 순서(변경 없음)**: B1 rot-head **완주**(미수렴, 신규코드 0) → B2 RC 수리(체크리스트) → B3 rot-R 개선. RootNet은 z/xy 분해 이후 판단. wrist P1b는 폐기 유지(+0.005).

**전체 전략 한 줄**: depth 경로가 **RootNet(학습) → RC(검증된 test-time)** 로 바뀌었고, 그 결과 **rot-head R이 KUKA/Baxter 공통의 단독 최대 병목으로 승격**했다.

### 13.8 Q3 — RoboPEPP Table 3 관절각 오차 vs 우리 (검증 완료)

**출처**: RoboPEPP(2411.17662) **Table 3** — *"Mean absolute error between the predicted and actual joint angles (in degrees) for the Panda and Kuka synthetic test sets."* HTML의 `rowspan` 구조를 직접 파싱해 열정렬 확인, **행/열 평균 산술검증 양방향 통과**(열 오정렬 시 산술이 깨지므로 정렬 확정).

**⚠️ 비교 조건 3가지**:
1. **synthetic 전용** — real split 관절각 수치는 논문에 **없음**.
2. **J1–J6만** — *"we predict the angles of all joints except the last one, assigning a random angle to it"* → **J7은 예측 안 하고 랜덤 배정**. 우리도 J7=0 고정에 J0–J5(=J1–J6) MAE를 보므로 **apples-to-apples 성립**.
3. **`HPE (Known BBox)` = HoRoPose** (논문이 HoRoPose를 "HPE"로 지칭; "HoRoPose" 문자열 미등장). **GT bbox 사용** = 우리보다 쉬운 프로토콜. DREAM은 known-angle이라 Table 3에 행 없음.

| 로봇/셋 | 방법 | J1 | J2 | J3 | J4 | J5 | J6 | **Avg** |
|---|---|---|---|---|---|---|---|---|
| **Panda DR** | RoboPose | 6.1 | 2.7 | 3.6 | 2.5 | 6.3 | 8.1 | 4.9 |
| | HPE(=HoRoPose, GT-bbox) | 6.2 | 2.2 | 3.9 | 1.9 | 5.9 | 6.6 | 4.4 |
| | **RoboPEPP** | 4.9 | 2.3 | 2.7 | 2.2 | 4.9 | 5.4 | **3.8** |
| | **Ours** (배포 mlp) | — | — | — | — | — | — | **~9.1** (IEF 학습중 11.5→하강) |
| **Panda Photo** | **RoboPEPP** | 4.4 | 1.8 | 2.2 | 1.8 | 4.4 | 4.8 | **3.2** |
| **KUKA DR** | RoboPose | 4.4 | 2.8 | 5.4 | 3.4 | 12.5 | 8.5 | 6.2 |
| | HPE(GT-bbox) | 4.6 | 3.6 | 4.9 | 2.8 | 5.2 | 6.1 | 4.5 |
| | **RoboPEPP** | 3.7 | 3.5 | 5.1 | 3.5 | 4.1 | 6.2 | **4.3** |
| | **Ours** (`mr_logs/kuka_directpose.log`) | 5.45 | 4.48 | 7.11 | 4.42 | 7.26 | 8.98 | **6.28** |
| **KUKA Photo** | **RoboPEPP** | 3.8 | 2.8 | 4.6 | 3.1 | 3.8 | 5.4 | **3.9** |
| **Baxter** | — | — | — | — | — | — | — | **논문에 관절각 수치 없음**(Table 2 ADD만) |
| | **Ours** | 6.58 | 4.44 | 10.61 | 7.08 | 25.37 | 21.20 | 12.55 |

**해석**:
- **KUKA**: 우리 **6.28°** ≈ RoboPose 6.2°, RoboPEPP 4.3°·HoRoPose 4.5°에 **약 2° 열세**. **그러나 §12.2 실측(GT각도=head각도 → ADD 동일)상 KUKA 각도는 ADD 병목이 아니다** → 이 2°는 **낮은 우선순위**. (KUKA 계획은 §13.7대로 RC+rot-R.)
- **Panda**: 우리 ~9.1° vs RoboPEPP 3.8° — **약 5°의 최대 격차**. P1b/IEF 실험이 겨냥하는 바로 그 축. RoboPEPP의 J2–J4가 **2.2–2.7°**로 특히 낮은 점이 §10.1의 "proximal 우위" 진단과 일치.
- **Baxter**: 우리 wrist J5/J6가 **25.4/21.2°**로 극단적이나 §13.4대로 **ADD 기여 ~8mm**라 무해. 경쟁 수치 자체가 없어 비교 불가.
- **주의**: RoboPEPP J5/J6가 상대적으로 큰 것(4.9/5.4 Panda DR)은 손목이 **모두에게 어려운 공통 축**임을 재확인 — §1의 "off-frame 손목=공통 한계" 및 §13.4 레버암 논리와 정합.

---

## 14. 통일 아키텍처 재작성 — 제약: **"하나의 알고리즘, 로봇별 가중치"**

> **제약**: 파이프라인/알고리즘은 세 로봇 동일. *가중치(모델)*·*FK 값*만 로봇별 허용.
> **현재 부채(위반)**: `--direct-pose`가 **수동 로봇별 플래그**(KUKA/Baxter만) = 명백한 알고리즘 분기. 부수적으로 Panda RC의 **azure-off 하드코딩**(카메라명 분기)도 경미한 위반.

### 14.1 검증 — Panda basin-flip과 KUKA/Baxter link-혼동은 같은 실패모드인가? → **YES(구조적 동일)**

**공통 병리 = "confident하지만 correspondence가 틀린 2D"가 13-DOF 재투영 solve를 오염**(실측):
| 로봇 | 오염원 | 실측 |
|---|---|---|
| Panda | 화면밖 키포인트를 **confident하게 환각**(§1) | tail reproj **90px**(정상 1.5px), 전 팔 이동(base 62.7mm) |
| KUKA | **link-identity 혼동 11.2%**, 그중 **90%가 다른 키포인트 GT로 스냅** | median 2D는 **2.1px**(우수) |
| Baxter | 동일 혼동 6.2%, wrist w1/w2 집중 + off-frame 잦음(in/tot 683·670/800) | median **1.5px** |

**우리 솔버의 구조적 공백(실측)**: `conf_gate`는 **신뢰도 축**으로만 거른다(`solve_pose_kinematic.py:229,283`). **신뢰도 ≠ 정확도**이므로 세 로봇 병리를 **원리적으로 못 잡는다**(§9의 "confidence축 ≠ presence축" 결론과 동일). IRLS(Geman-McClure, `huber_px=8.0`, L296-315)는 있으나 **이미 오염된 PnP init에서 출발하는 국소 재가중**일 뿐, consensus·재시작이 없다.
⇒ **동일 실패모드 확정.** 로봇별 우회가 아니라 **하나의 correspondence-robust solve**가 정공법. (코디네이터 관찰 지지)

> ⚠️ **단, 솔버 포기의 근거는 감사 불가(중요)**: KUKA 솔버 발산 기록("**oracle 2D+R에서도** 발산, J2 7.5°→25°, 원인=link-혼동")은 ① **Ep~15 미수렴 head**의 예비측정이고 ② **로그가 남아있지 않으며**(`Eval/` 내 KUKA 로그 3개는 전부 direct-pose) ③ **서술이 자기모순**이다 — *oracle 2D면 link-혼동이 존재할 수 없는데* 원인을 link-혼동이라 적었다. ⇒ **이 근거로 솔버 경로를 영구 배제해선 안 된다. 수렴 head로 재측정 필요(eval만, 저비용).**

### 14.2 통일 파이프라인 정의
```
검출 → 각도head → [① 대응 신뢰도 선별] → [② 강건 consensus 6-DOF solve]
     → [③ 조건부 θ 정제] → [④ 조건부 RC]
```
**모든 분기는 런타임 측정량(inlier 수·합의도·IoU)으로만 결정. 로봇 이름으로 분기 금지.**

| 단계 | 내용 | 세 로봇 공통성 |
|---|---|---|
| ① **대응 선별**(신규, 핵심) | conf + **경계 margin**(CtRNet 2px) + **kinematic-consistency 잔차**(로버스트 적합 후 잔차 큰 대응 = 혼동/환각 검출) | Panda 환각·KUKA/Baxter 혼동을 **한 규칙**으로 포착 |
| ② **강건 consensus solve** | θ는 head값 고정, **6-DOF만** hypothesize-and-verify(RANSAC/EPnP subset + 최소점수) | 조건수 양호·오염 내성. P0 decoupled와 동일 원리(CtRNet `CtRNet.py:80-82`, RoboPEPP `test.py:252-258` 구조) |
| ③ **조건부 θ 정제** | inlier 충분할 때만 재투영으로 θ 정제 | Panda 13-DOF 발산과 KUKA 각도정제 발산을 동시 방지 |
| ④ **조건부 RC** | min-iou 게이트 + 재투영 앵커(§13.6 체크리스트) | 세 로봇 메쉬 확보 완료 |

### 14.3 direct-pose를 게이트로 흡수 가능한가? → **가능·정당, 단 조건부**
"2D 증거가 부족하면 정제를 건너뛴다"는 **데이터 의존 규칙**이고, 전 로봇 **동일 코드**로 구현되면 제약을 만족한다. 정당화 조건: **게이트 입력이 측정량(inlier count/합의도/reproj)이어야 하며 robot id가 아니어야 한다.** 현재 `--direct-pose` 수동 플래그는 위반이므로 **측정 기반 게이트로 대체**해야 한다.
> ⚠️ **단 "솔버를 고칠 필요 없다"는 뜻은 아니다**: ①(대응 선별) 없이 게이트만 넣으면 KUKA/Baxter는 게이트가 **상시 발동**해 영구히 direct-pose로 남는다 = **실질적 알고리즘 분기**. **①+게이트 둘 다 필요.**

### 14.4 기대효과 (🔶 추측 — 근거와 검증법 명시)
KUKA 2D는 **median 2.1px로 이미 우수**하고 문제는 **catastrophic 11.2%뿐**이다. 강건 선별로 그 outlier를 제거하면 남은 ~89%(2px급)에 대한 PnP pose는 **rot-head 회귀(R 7.4°/t 56mm)를 크게 상회할 개연**이 크다. §13.4 민감도상 **t≈20mm·R≈2.5°면 모델 AUC 0.72** = SOTA-근접 목표. Baxter는 median 1.5px·6.2%로 더 유리.
⇒ **이는 추측이나 저비용 검증 가능**: 수렴 head + 강건 게이팅으로 `kuka/baxter_add_eval`을 **solver 모드**로 재실행(학습 불필요, eval만).
**전략적 함의**: 사실이면 통일 솔버가 **RC·RootNet·rot-R보다 큰 KUKA/Baxter 최대 레버**다 — 약한 회귀 pose를 정확한 2D 기반 solve로 대체하는 것이기 때문.

### 14.5 RootNet 판정 → **폐기**
KUKA/Baxter에만 붙이면 **알고리즘 분기 = 제약 위반**. 전 로봇 공통 삽입은 Panda RC와 **기능 중복**이며 우리 스택 미검증. 무엇보다 **RootNet의 존재이유였던 "KUKA mesh 부재"가 §13.6으로 소멸**했다. ⇒ **폐기.** (RC가 세 로봇 모두에서 실패할 때만 전 로봇 공통으로 부활 검토.)

### 14.6 RC 통일
세 로봇 메쉬 확보(Panda ✓ / KUKA ✓ 신규 / Baxter ✓) → **동일 RC + 동일 앵커·게이트**(§13.6 체크리스트) 전 로봇 적용. Panda의 **azure-off 하드코딩은 카메라명 분기라 경미한 위반** → **거리/init-IoU 기반 측정 게이트로 대체 권고**(`--min-iou`는 이미 측정 기반이므로 확장이 자연스럽다).

### 14.7 rot-head R — 전 로봇 공통
동일 구조·로봇별 가중치 → **제약 준수**. 단 **14.4가 성공하면 pose가 solve에서 나오므로 rot-head 역할이 `R_init`(basin pin)으로 축소** → 우선순위 강등 가능. **14.4 결과 확인 후 확정.**

### 14.8 재작성 우선순위 (전 항목 제약 준수)
| 순위 | 작업 | 제약 준수 | 비용 |
|---|---|---|---|
| **U1** | **KUKA/Baxter solver-모드 재측정**(수렴 head, 현 게이팅 그대로) | 측정만 | **eval only** |
| **U2** | **①대응 선별 + ②강건 consensus solve** 구현(공통 코드) — Panda P0와 **통합**(별개 처방 금지) | ✅ 단일 코드경로 | 중 |
| **U3** | **direct-pose를 측정 기반 게이트로 대체**(플래그 제거) | ✅ 부채 청산 | 소 |
| **U4** | **RC 통일**(앵커·게이트) 전 로봇 + azure-off를 측정 게이트로 | ✅ | 중 |
| **U5** | rot-head R 개선 (U1/U2 결과 따라 강등 가능) | ✅ | 8–13h |
| ~~폐기~~ | ~~RootNet~~ (§14.5) · ~~Baxter wrist P1b~~ (§13.4 +0.005) | — | — |

**논문 관점**: "**하나의 알고리즘, 로봇별 가중치**"는 "로봇마다 다른 처방"보다 훨씬 강한 주장이며, §14.1의 **공통 실패모드 진단(신뢰도축≠정확도축)**이 그 주장을 뒷받침하는 실측 근거가 된다.

---

## 15. U2 전제 진단 — KUKA R 병목의 oracle 상한 (GPU-free CPU 시뮬, 실측)

> **배경**: intrinsics 버그(focal 320× 오류) 수정 후 KUKA 솔버 **0.686**(t **16.8mm** ✅ 목표달성, R **6.13°** ❌ 미달). R이 유일 잔여 레버.
> **방법**: `solve_batch`는 (kp2d, conf, K, θ_init, R_init)의 순수 함수 → **GPU 없이 CPU에서** 실제 iiwa7 FK·실제 K·실제 GT로 시뮬. 200 프레임.
> **검증(선행)**: FK→GT3D Kabsch 잔차 **0.0043mm**, GT3D를 K=(320,320,320,240)로 투영 시 데이터셋 `projected_location`과 **0.001px** 일치 → 파이프라인·K 확정.

### 15.1 🔴 핵심 결과 — **outlier 제거는 R을 못 고친다** (U2 전제 반증)

| 실험 (θ free, θ_init 6.28°오차, R_init 7.28°오차) | R | t | ADD med | AUC |
|---|---|---|---|---|
| (a) **GT 2D 완벽** | **3.57°** | 6.1mm | 3.6mm | 0.911 |
| (b) +2.1px 노이즈(우리 median) | 4.70° | 17.3mm | 12.9mm | 0.807 |
| (c) +11.2% link-혼동 outlier | 5.16° | 19.8mm | 15.8mm | 0.753 |
| (d) (c) + **oracle inlier 제거** | **5.00°** | 19.0mm | 15.5mm | 0.786 |

- **(c)→(d) outlier를 완벽히 제거해도 R은 5.16→5.00°(−0.16°)뿐.** ⇒ **"link-혼동 outlier가 R을 오염시킨다"는 유력 가설은 반증.**
- **(a) 2D가 완벽해도 R=3.57°** — R은 **2D 품질/outlier가 아니라 solve 구조**에 막혀 있다.

### 15.2 R 한계의 진짜 원인 — **θ↔R 결합(gauge)**

| 진단 (clean GT 2D) | R | AUC |
|---|---|---|
| (e) **θ를 GT에 FREEZE** | **0.00°** | 0.982 |
| (f) θ_init=GT 이지만 θ **free** | 2.70° | 0.966 |
| (g) R_init=GT, θ noisy+free | 2.49° | 0.916 |
| (h) R_init=GT **+** θ=GT free | 0.00° | 0.987 |
| (i) iters 4배(1000) | 3.14° | 0.938 |

**θ를 풀어놓는 것 자체가 R을 망친다**: θ가 GT에서 출발해도 free면 표류해 R 2.70°. θ를 고정하면 R=0.00°(완벽). ⇒ **원인은 재투영이 (θ, R)을 맞바꿀 수 있는 결합/게이지 자유도**이지, 2D도 capture range도 iteration도 아니다.

### 15.3 그럼 decoupled solve(θ freeze)를 쓰면? → **현재 θ 정확도로는 오히려 손해**

| θ freeze 위치 (2.1px 2D) | R | AUC |
|---|---|---|
| θerr **0°** | 1.12° | 0.879 |
| θerr 1° | 2.01° | 0.862 |
| θerr 2° | 3.30° | 0.849 |
| θerr 3° | 4.80° | 0.810 |
| θerr 4° | 6.89° | 0.773 |
| **θerr 6.28°(우리 KUKA)** | **10.56°** | 0.681 |
| θerr 8° | 12.41° | 0.634 |

**freeze 시 R ≈ 1.5 × θ오차**로 선형 전이. 우리 θ=6.28°에서 freeze하면 **R 9.4–10.6°로 free(4.7°)보다 2배 악화**. ⇒ **P0/decoupled solve는 θ가 ≲2°일 때만 유효.** 우리 6.28°·RoboPEPP 4.3°로는 **아직 불가**. (free-θ가 θ오차를 R로 일부 흡수해 주는 것이 현재로선 이득.)

### 15.4 배포 모드(θ free)에서의 실제 민감도 — **R은 θ에 거의 둔감**

| θ_init 오차 (θ free, 2.1px+outlier) | R | AUC |
|---|---|---|
| 0° | 4.31° | 0.784 |
| 2° | 4.33° | 0.786 |
| 4° | 5.09° | 0.762 |
| **6.28°(현재)** | 5.49° | 0.757 |
| 8° | 5.82° | 0.742 |

**θ를 완벽하게 만들어도 free-θ solve의 R은 4.31°가 바닥.** ⇒ 각도head 개선(P1b/IEF)은 KUKA R에 **2차 효과**(6.28→2°면 R 5.49→4.33, AUC +0.03).

**outlier 제거 순이득(동일 θ에서 with/without 비교)**: θ 6.28° 기준 **AUC 0.757 → 0.805 (+0.048)**, R은 5.49→5.06°.
⇒ **U2는 "R 수정"이 아니라 "ADD/t 개선"으로 +0.05 가치가 있다. 목적은 바뀌었지만 투자 가치는 유지.**

### 15.5 ⚠️ 이전 민감도 모델(§13.4) 정정
§13.4의 상수-ADD 근사는 "**R≤2.5° 필수**"라 했으나, 실제 FK·K·GT 기반 시뮬은 **R≈5°에서도 AUC 0.805**를 낸다. **"R≤2.5° 필요" 주장은 폐기**(내 근사 모델의 과도한 비관 — 당시 caveat로 표기했던 그대로). **AUC 0.80대는 현재 R 수준에서도 도달 가능.**

### 15.6 결론 및 U2 재정의 (실측 기반)
1. **U2를 R 수정 수단으로 정당화할 수 없다**(oracle 상한 −0.16°). **그러나 AUC +0.05 근거로는 유효** → **축소된 형태로 진행 권고**: 강건 대응 선별(경계·consistency 잔차)만, "R이 좋아진다"는 기대는 삭제.
2. **decoupled/freeze는 보류** — θ≲2° 달성 전에는 유해(§15.3).
3. **θ↔R 결합이 R의 근본 한계** → 이를 깨려면 재투영 밖의 독립 R 증거가 필요: **rot-head R_init를 prior로 유지(현행)** 또는 **RC의 실루엣 방향 정보**(실루엣은 θ와 무관하게 R을 제약) — **RC가 R을 고칠 유일한 후보**라는 점이 §15.2에서 처음 드러남.
4. **🔶 추측(미측정, GPU 필요)**: RC를 direct-pose가 아니라 **진짜-K 솔버(0.686) 위에** 얹으면, RC는 t/scale뿐 아니라 **실루엣 방향으로 R도 제약**하므로 §15.2의 θ↔R 게이지를 **외부 증거로 깨는** 역할이 기대된다. **이것이 KUKA를 0.75–0.80으로 밀 가장 유력한 카드**이며, fallback 논의(꼬리 게이팅이 direct-pose로 되돌아가 이득이 갇히는 문제)도 **RC-fallback으로 자연 해결**된다. 우선 실행: `solver(0.686) + RC`, 게이트는 §13.6 체크리스트(min-iou, 재투영 앵커, DOF 최소화).

---

## 16. 붕괴 모드 처방 탐색 — "confident하지만 대응이 틀린 2D"

> 목표: 네 경우(Panda synth/azure, KUKA, Baxter)에서 확정된 **공통 실패모드**를 없앨 방법론 탐색. GPU0 점유 중이므로 **오프라인 CPU 시뮬 + 문헌**으로 수행.

### 16.1 🔴 오프라인 실측 — B계열(잔차/consensus 기반 거부)은 **실패**

KUKA 200프레임 CPU 시뮬(§15 하네스), 실제 FK·K·GT, 11.2% link-혼동 outlier 주입, θ_init 6.28°오차:

**(i) kinematic-consistency 잔차를 outlier 검출기로** (RANSAC-PnP 합의 후 per-kp 재투영 잔차):
| 지표 | 값 |
|---|---|
| 검출 **ROC-AUC** | **0.8585** (분리력은 있음) |
| recall @15px | 0.887 (대부분 잡음) |
| **precision @15px** | **0.406** ← 문제 |
| 실제 flagged 비율 | 23.6% (실제 outlier는 11.2%) |

**(ii) 그 검출기로 실제 거부했을 때 end-to-end**:
| 설정 | AUC | R |
|---|---|---|
| (c) outlier 있음, 거부 없음 | **0.753** | 5.16° |
| (d) **oracle** inlier 제거(상한) | **0.786** | 5.00° |
| (q) 잔차 거부 thr=15px | **0.734** | 5.42° |
| (q) 잔차 거부 thr=25px | **0.742** | 5.21° |

⇒ **실용 검출기는 baseline보다 나쁘다(0.753 → 0.734~0.742).** 키포인트가 7개뿐이라 **precision 0.4로 23%를 버리면 정상 키포인트 손실이 outlier 제거 이득을 초과**한다.
**근본 원인(실측)**: 모델 3D가 `FK(θ_head)`이고 θ가 **6.28° 틀려서**, 재투영 잔차가 **"대응 오류"와 "모델 오류"를 구분하지 못한다.** ⇒ **B3(잔차 기반)·B4(consensus PnP)는 KUKA/Baxter에 대해 반증. 재제안 금지.**

### 16.2 결정적 구분 — **실패 모드는 공통이나 "검출 가능성"은 다르다**

| 부류 | 오염원 | 검출 신호 | 정밀도 | oracle 상한 |
|---|---|---|---|---|
| **① 화면밖(Panda synth/azure)** | 프레임 밖 kp | **기하학적으로 결정적**(좌표가 경계 밖인지) | **높음**(결정론적) | azure **+0.041**(실측), synth +0.033~0.063 |
| **② link-혼동(KUKA/Baxter)** | 다른 kp의 **그럴듯한 실제 위치**로 스냅 | 잔차뿐 → 모델오차와 혼동 | **낮음(0.4)** | KUKA **+0.033**(실측), 실현 시 **음수** |

⇒ **"하나의 규칙으로 네 곳을 동시에" 는 성립하지 않는다.** 공통인 것은 *증상*이지 *처방*이 아니다. ①은 test-time으로 해결 가능, ②는 test-time 신호가 원리적으로 부족하다.
> 이는 §14의 통일 아키텍처와 모순되지 않는다 — **동일 코드·동일 규칙(경계+presence 게이트)** 을 세 로봇에 적용하되, 그 규칙이 ②에서는 발화하지 않을 뿐이다(알고리즘 분기 아님).

### 16.3 루트 코즈(학습 시) 확인 — 실측 + 문헌

- **RoboPEPP**: off-frame kp를 **heatmap loss에서 마스크**한다. `valid_indices_mask = (0<x<crop)&(0<y<crop)`(`datasets/dream.py:258-260`) → focal/L1 keypoint loss에 곱함(`train.py:229,237`). **off-frame을 아예 지도하지 않으므로** 응답이 약해지고, test-time conf 필터(`test.py:252`)가 걸러낸다.
- **우리**: `dataset.py:611`이 off-frame kp에 대해 `continue` → **전(全)0 타깃 히트맵**을 준다. `train_heatmap.py:236-237`의 손실엔 **valid 마스크가 없다**(`criterion(preds, gt_hms).mean([2,3])`). 즉 **우리도 "없음"을 지도하고는 있다**(RoboPEPP와 목적은 같고 구현만 다름).
- ⚠️ **따라서 "검출기가 화면밖을 confident하게 환각한다"는 진단은 재검증이 필요하다** — 전0 타깃으로 학습됐다면 히트맵 max(=conf)가 **이미 낮을 수 있다.** 그렇다면 **presence 정보가 이미 존재**하고 우리가 안 쓰고 있을 뿐이다(`conf_gate=0.05`는 너무 낮고, 그 sweep은 azure 발산 꼬리가 아니라 전체 평균에서 flat이었다).

**⇒ 가장 싸고 가장 결정적인 다음 측정(소표본, GPU0 여유 시 수분)**: **truly-off-frame kp vs in-frame kp의 히트맵 max(conf) 분포**. 분리되면 → **재학습 0, 임계 캘리브레이션만으로 ①을 해결**. 분리 안 되면 → A계열(aux presence head) 필요.

### 16.4 아이디어 순위 (a=시점, b=기대이득, c=배포 0.804 위험, d=통일제약)

| 순위 | 아이디어 | a | b 기대이득 | c 0.804 위험 | d 제약 | 판정 |
|---|---|---|---|---|---|---|
| **1** | **presence 임계 캘리브레이션**(off-frame conf 분포 측정 후 게이트 재설정) | test | ①의 상당부분(azure ≤+0.041) | **없음**(추론 파라미터) | ✅ 전 로봇 동일 규칙 | **즉시** |
| **2** | **경계-margin 게이트 + RC/재solve fallback**(구현중) | test | ① 일부 | 낮음(do-no-harm 게이트) | ✅ | 진행 |
| **3** | **ProbPose식 presence/visibility aux head** | train(aux만) | ① 잔여 | **낮음** — 기존 출력 불변, 백본·kp head **동결**, 새 분기만 추가 | ✅ | 1·2 실패 시 |
| **4** | **중복/상호배타(assignment) 검출** — 혼동의 **90%가 다른 kp GT로 스냅**하므로 "두 kp가 같은 위치" 는 **고정밀 신호** | test | ② 유일 후보 | 없음 | ✅ | **미검증**(내 시뮬은 outlier를 스냅으로 합성해 **순환**) → 실데이터 확인 필요 |
| — | ~~kinematic-consistency 잔차 거부~~ | — | **음수(실측)** | — | — | **§16.1 반증** |
| — | ~~RANSAC/consensus PnP~~ | — | **이득 없음(실측)** | — | — | **§16.1 반증** |
| — | ~~RLE 불확실성 회귀~~ | train(검출기 전체) | 불명 | **높음**(검출기 재학습) | ✅ | 위험 대비 EV 낮음, 보류 |
| — | ~~RoboPEPP식 loss 마스킹 이식~~ | train(검출기 재학습) | ①(우린 이미 전0 타깃) | **높음** | ✅ | **중복** — 우리가 이미 동등 처리 중 |

### 16.5 REFUTED 대조 (재제안 아님을 명시)
`SUMMARY.md` 및 세션 반증 목록과 교차확인: **backbone 적응 전 계열 / mlp_patch / MCL / conf-gate 튜닝 / union-bbox / depth·t_z prior / Baxter 실루엣 RC / 모집단 prior / prior-adaptive fill / min-reproj MS**는 재제안하지 않았다.
- **구분 표기(부분 이식이었던 것)**: ① **edge-gate**는 "**단독**"이 반증됐다(§1.2: proximal 앵커만 = +0.006) — 위 2순위는 **fallback과 결합**한 형태라 별개. ② **RoboTAG ℒalign**은 head-레벨 GT-pose 재투영만 이식해 반증됐다(§9 표) — 원논문의 co-train 형태는 미시도. ③ **가시성 헤드**는 과거 "cov-PnP가 같은 신호를 커버"라며 **보류**됐으나(sota_survey §3.2), **그 전제가 무효**임이 확정됐다(cov-PnP·conf는 *신뢰도 축*이라 confident-wrong을 원리적으로 못 잡음) → **격상 재검토가 정당.**

### 16.6 정직한 상한 총계 — **이 축의 총 가치**

| 경우 | 상한 | 근거 | 실현 가능성 |
|---|---|---|---|
| Panda **azure** | **+0.041** | 실측(코디네이터) | **높음**(off-frame=결정론적 신호) |
| Panda **synth** | +0.033~0.063 | §1.2 counterfactual | **중** — proximal 앵커+distal cap **동시** 필요 |
| **KUKA** | +0.033 | §16.1 실측 oracle | **낮음** — 실용 검출기는 음수 |
| **Baxter** | ~+0.02 (추정) | outlier 6.2%(KUKA 11.2%의 절반) | **낮음**(동일 이유) |

**결론**: 이 축의 **실현 가능한 가치는 사실상 Panda(①)에 집중**되며 **약 +0.04(azure)**, synth는 조건부. **KUKA/Baxter(②)의 기여는 상한 자체가 작고(+0.02~0.03) 실용적으로 음수**다.
⇒ **"한 번에 네 곳을 고치는 자리"라는 기대는 실측으로 지지되지 않는다.** KUKA/Baxter의 남은 격차는 이 축이 아니라 **§15.6의 RC(θ↔R 게이지 파괴)** 에 있다. 자원 배분을 그쪽으로 유지할 것을 권고한다.

---

## 17. P1b 대안 탐색 + "DINOv3가 합성에 약한가?" 해리 진단

> 확정 사실: frozen DINOv3 위 각도 head는 **아키텍처 무관 ~9.1° 천장**(IEF 9.11 ≈ mlp 9.16), P1b(학습형 ResNet50 feature) **7.47°** 돌파. 두 질문이 얽혀 있다 — **(Q-task)** 병목이 각도-과제 특화 feature인가, **(Q-domain)** frozen DINOv3가 **합성 도메인**에 약한가(사용자 가설). P1b ResNet은 합성으로 학습되므로 7.47° 승리가 둘 중 무엇인지 **혼입(confound)** 돼 있다.

### 17.1 🔴 사용자 가설("DINOv3가 합성에 약함") — **detection 축에서 반증** (실측)

frozen DINOv3의 2D 국소화 품질을 **domain별로 직접 측정**(good-frame 재투영 px = 순수 2D 검출 품질, oracle-angle dumps):
| 도메인 | good-frame reproj (px) | good-frac |
|---|---|---|
| **synth** DR(pred) | **1.46** | 0.89 |
| **synth** Photo(pred) | **1.21** | 0.91 |
| real azure | 1.71 | 0.98 |
| real kinect | 1.39 | 1.00 |
| real realsense | 1.48 | 1.00 |
| real orb | 1.89 | 0.98 |

**frozen DINOv3 2D는 합성에서 real과 동급이거나 오히려 더 정밀**(synth 1.2–1.5 vs real 1.4–1.9). ⇒ **"DINOv3가 합성에 약하다"는 feature/검출 수준에서 반증.** (synth의 good-frac이 낮은 것은 검출 품질이 아니라 **off-frame 손목**의 기하 문제 §1.)

**정합성 확인(코디네이터 요청)**:
- oracle-angle synth **0.899** & synth 2D median **1.5px** → "합성 2D가 충분히 좋다"와 **정합**(위 표가 재확인).
- **KUKA 합성 부진 = intrinsics 버그**(§15, focal 320×) → DINOv3 무죄, "합성 약함" 가설에 **반대 증거**.
- realsense angle J0 **44°**(oracle-2D→14°) → **real에서도 각도가 나쁘다**(2D/depth-limited). 즉 **real이 각도에서 더 낫지도 않다** — real 배포가 0.804인 것은 각도가 좋아서가 아니라 **각도가 binding constraint가 아니기 때문**(real 병목 = depth/translation, `SUMMARY.md` 06-08). synth는 depth가 쉬워 **각도가 노출된 binding constraint**가 될 뿐.

**⇒ 결론(실측 기반)**: 사용자 가설은 **틀리다**. 문제는 "DINOv3가 합성에 약함"이 아니라 **"각도 회귀가, 도메인 무관하게, frozen 범용 feature가 쉽게 노출하지 않는 정보를 요구함"**(Q-task). real에서 안 보였던 이유는 depth가 먼저 막혀서다. ⇒ **논문 서사는 "합성 도메인 적응"이 아니라 "각도의 과제-특화 feature 필요(검출은 frozen으로 충분)"인 dissociation이 정확.**

### 17.2 layer-wise 선형 probe 결과 (실측, 학습 0) — **idea 4(층 선택) 반증**

frozen DINOv3 13개 hidden layer 각각의 **pooled(GAP) feature → ridge → 각도 MAE**(Panda synth, 600 train/250 val, GPU3 추론만):
| layer | 0 | 2 | 4 | 6 | 8 | 10 | 12(last) |
|---|---|---|---|---|---|---|---|
| MAE° | **34.96** | 36.4 | 37.2 | 38.2 | 39.6 | 39.2 | 39.3 |

- **전 층이 35–40°로 균일하게 나쁨.** 각도 정보를 유의하게 더 담은 **중간층 sweet-spot 없음**(layer0가 근소 최저이나 이는 초기층의 raw 위치정보 잔존일 뿐, 35°는 여전히 무의미).
- ⇒ **idea 4(frozen 층 선택/조합, 학습 0)는 반증.** 어느 frozen 층의 pooled feature도 각도를 선형-노출하지 않는다.

**해석(중요·정직)**: 이 probe는 **pooled+선형**이라 배포 head(비선형 MLP + **kp-sampled feature + 2D bearings/geo**)보다 훨씬 약하다. 배포가 9.1°에 도달하는 힘은 **pooled appearance가 아니라 2D 기하(bearings)** 에서 온다(pooled 단독은 35°). ⇒ **결론 두 가지**: (i) frozen appearance는 pooled-선형으로 각도를 안 담는다 → **"올바른 층만 고르면 된다"는 우아한 길은 없다.** (ii) **그러나 정보 부재 증명은 아니다** — 각도 정보는 토큰의 **공간 배치**(kp-conditioning이 쓰는 것)나 **비선형**에 있을 수 있어, LoRA(공간·비선형 적응)의 성패는 이 probe로 판정 불가 → **실제 LoRA 학습 실험이 유일한 결정자.**

### 17.3 후보 순위 (기대이득·비용·우아함·논문 서사)

| 후보 | a 시점 | b 기대(≤7.5° 근접?) | c 0.804 위험 | 우아함 | 논문 서사 |
|---|---|---|---|---|---|
| **① LoRA/어댑터**(각도전용, frozen DINOv3 위) | train(어댑터만) | **미측정 — 핵심 실험**. P1b가 상한(7.47°) | **없음**(키포인트 경로 무손상, 원본 출력 불변) | **최고**(파라미터 미미) | **최강**: "frozen 범용 백본 + 경량 각도 적응 = 검출·각도 dissociation" |
| **② 마지막 N블록 학습 복제**(각도전용) | train(N블록) | P1b≤ 기대(DINOv3 init 유리) | 없음(복제본, 원본 frozen) | 중 | 중: "부분 적응" |
| **③ 2D 강주입/kp-conditioned head** | train(head) | **불확실** — 병목이 feature면 2D 강화로 못 넘음(단 §17.2가 "정보 있음"이면 유효) | 없음 | 상 | 중 |
| ~~④ frozen feature 층 선택/조합~~ | train 0 | **반증(§17.2)** — 전 층 35–40° 균일 | 없음 | — | **폐기** |
| P1b(ResNet50 통째) | train(백본) | **7.47°(실측)** | 없음(별도망) | **낮음**(무겁·"co-train 재탕" 비판) | 약: 리뷰어가 "그냥 co-train" |

### 17.4 논문 프레이밍 판정
- **P1b가 유일 해면**: 기여는 "co-train 재탕"이 아니라 **통제된 dissociation 실험**으로 서술해야 값이 있다 — *"동일 백본에서 head만 바꾸면(IEF/mlp/transformer/MoE) 9.1°에 고정, feature를 과제-학습하면 7.47°. 그리고 §17.1로 이는 **합성 도메인 적응이 아니라 각도-과제 특화**임을 통제." (검출은 frozen으로 SOTA인데 각도만 과제-feature 필요)*. 이 **이중 해리(head-불변 & domain-불변)** 자체가 기여.
- **①/④가 P1b에 근접하면 훨씬 강한 서사**: "**키포인트는 frozen 범용 백본으로 충분, 각도는 극소 파라미터의 과제-어댑터로 충분**" — 별도 백본 불필요, 파라미터 미미, 키포인트 sub-pixel 정밀도 무손상. 우아함·재현성·기여 모두 우위. **∴ ①(LoRA)을 P1b와 병행 학습해 7.47° 회수율을 측정하는 것이 최우선 실험.**

### 17.5 실제 돌려볼 실험 (GPU 해제 후, 우선순위)
1. **① LoRA 어댑터**(rank 8/16, 마지막 4블록 attn·mlp, 각도전용 경로) vs **P1b**를 **동일 synth·동일 epoch**로 학습 → val 각도 MAE. **게이트: LoRA가 ≤7.8°면 P1b 대체(우아).** 비용: LoRA ~수백만 파라미터, P1b와 동급 wall-clock, 키포인트 경로는 frozen 원본 그대로(별도 forward).
2. **④ layer-probe**(진행 중, 학습 0) — ①의 사전 예측: 정보가 중간층에 있으면 LoRA 성공 가능성↑.
3. (보조) **③ kp-conditioned**: ①/④가 "정보 있음"을 보이면, 2D 좌표를 각도 head 회귀 입력에 직접 추가(백본 무학습)로 저비용 확인.

---

## 18. LoRA 각도 어댑터 (`--angle-backbone dino_lora`) — 구현 설계 (§17 idea① 결정 실험)

> 목적: P1b(ResNet50 통째, 7.47°)를 **frozen DINOv3 + 극소 파라미터 LoRA**로 대체 가능한지. 성공 시 논문 기여가 "별도 백본"에서 "**키포인트는 frozen 범용 백본으로 충분, 각도는 극소 어댑터로 충분**"으로 격상.
> ⚠️ **§17.2 명시**: layer-probe(pooled-linear 35–40° 균일)는 **정보 부재 증명이 아니다**(각도정보가 공간배치·비선형에 있을 수 있음). **LoRA 성패는 probe로 예측 불가 — 실제 학습만이 유일 결정자.**

### 18.1 환경 실측
- **peft 미설치**(`import peft` 실패), transformers **5.2.0**. → **peft 설치 대신 수동 LoRA 권장**(dep 충돌 위험 회피 + "키포인트 경로 무손상"을 코드로 확실히 통제).
- DINOv3 ViT-B: **12블록**, HF 이름 `model.layer.{i}`. 블록 내 Linear: `attention.{q,k,v,o}_proj`(768↔768), `mlp.{up_proj(768→3072), down_proj(3072→768)}`.

### 18.2 핵심 제약 — 키포인트 경로 무손상 (2-forward 설계)
**LoRA를 공유 블록에 in-place로 넣으면 키포인트 forward도 오염**(= 반증된 백본 적응). 따라서:
- 키포인트 검출: **원본 frozen forward**(LoRA off) — kp2d/conf/heatmap은 배포와 **비트 동일**.
- 각도 feature: **2번째 forward**(LoRA on) — 같은 base 가중치(frozen) + 학습 low-rank delta.
- base 가중치는 **공유**(복제 안 함) → 메모리 추가분 = LoRA 파라미터뿐. 키포인트 forward는 `no_grad`, 각도 forward는 LoRA 파라미터에만 grad.

### 18.3 수동 LoRA 골격 (`model_angle.py` 추가)
```python
class LoRALinear(nn.Module):
    """base(frozen) + scaling * B(A x). enabled=False면 base와 완전 동일(키포인트 경로 보장)."""
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base                       # frozen 원본 (requires_grad=False)
        for p in self.base.parameters(): p.requires_grad_(False)
        self.r, self.scaling = r, alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)   # B=0 -> 초기엔 base와 동일(무손상 시작)
        self.drop = nn.Dropout(dropout); self.enabled = True
    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            out = out + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scaling
        return out

def inject_lora(dino_model, last_n=4, r=8, alpha=16, targets=('q_proj','v_proj')):
    """마지막 last_n 블록의 target Linear를 LoRALinear로 교체. 반환: LoRA 파라미터 리스트."""
    blocks = dino_model.layer                    # ModuleList (12)
    lora_params = []
    for blk in blocks[-last_n:]:
        for name in targets:
            parent, attr = (blk.attention, name) if 'proj' in name and name[0] in 'qkvo' else (blk.mlp, name)
            lin = getattr(parent, attr)
            if isinstance(lin, nn.Linear):
                wrapped = LoRALinear(lin, r=r, alpha=alpha); setattr(parent, attr, wrapped)
                lora_params += [wrapped.A, wrapped.B]
    return lora_params

def set_lora(dino_model, on: bool):
    for m in dino_model.modules():
        if isinstance(m, LoRALinear): m.enabled = on
```

### 18.4 `AnglePredictor` 배선 (최소 변경)
- `__init__`: `angle_backbone=='dino_lora'`일 때
  ```python
  self.lora_params = inject_lora(self.backbone.model, last_n=int(os.environ.get('LORA_LAST_N','4')),
                                 r=int(os.environ.get('LORA_R','8')), alpha=int(os.environ.get('LORA_ALPHA','16')),
                                 targets=tuple(os.environ.get('LORA_TARGETS','q_proj,v_proj').split(',')))
  self.angle_head = AngleHeadIEF(feat_dim=feat_dim, kp_in=feat_dim, n_iter=int(os.environ.get('IEF_ITERS','4')))
  ```
- `forward`:
  ```python
  set_lora(self.backbone.model, False)
  with torch.no_grad():
      tokens_kp = self.backbone(image)          # 키포인트용 (원본 frozen, 배포 동일)
      heatmaps = self.keypoint_head(tokens_kp); kp2d = decode(...); conf = ...
  if self.angle_backbone == 'dino_lora':
      set_lora(self.backbone.model, True)
      tokens_ang = self.backbone(image)         # 각도용 (LoRA on, grad 흐름)
      gfeat = tokens_ang.mean(1); kpfeat = sample_kp_features(tokens_ang, kp2d, self.heatmap_size)
  ```
  (kp2d/conf/geo는 항상 **키포인트 forward**에서 나옴 = 배포 불변.)
- `freeze_detector()`: 불변(base·keypoint_head frozen; LoRA A/B는 trainable 유지).
- optimizer(`train_angle.py`): `params = list(angle_head.parameters()) + model.lora_params`.

### 18.5 공정 비교 설계 (유일 변수 = feature 경로)
동일 고정: **synth 데이터**(panda_synth_train_dr) · **epoch 60** · **head=IEF(n_iter=4)** · **LR 스케줄**(cosine 1e-3→1e-6) · **fk-weight 10** · **kp-jitter 2.0** · **batch 32**.
| arm | feature 경로 | 학습 파라미터 |
|---|---|---|
| **mlp-control** | frozen DINOv3 last-layer(순정) | head만 |
| **P1b** | 학습 ResNet50(ImageNet) | head + ResNet ~24M |
| **LoRA-r8** | frozen DINOv3 + LoRA(last4, q/v, r8) | head + LoRA **≈ 4블록×2타깃×2행렬×(768×8+8×768) ≈ 0.20M** |
| **LoRA-r16** | 〃 r16 | head + LoRA ≈ 0.39M |
> LoRA 파라미터 0.2–0.4M = ResNet50의 **~1–2%** = "극소 파라미터" 주장 근거.

### 18.6 판정 게이트
1. **1차(MAE)**: LoRA val 각도 MAE **≤ 7.8°**면 P1b(7.47°)의 대체 후보로 격상. (mlp-control 9.1° 대비 명확한 돌파여야 함.)
2. **2차(진짜 판정, ADD 전이)**: **good-frame ADD-AUC**가 0.788에서 상승하는지(`eval_synth_head.sh --crop-...`). **MAE↓가 ADD로 전이돼야 유효** — J1 게이지처럼 ADD-무의미 각도만 개선되면 기각(§13.4·§15의 MAE↔ADD 해리 교훈). base J0 **미퇴화** 필수.
3. **효율**: rank 8 vs 16의 파라미터 대비 이득. r8이 r16의 ~90% 회수면 r8 채택.
4. **최종 서사 게이트**: LoRA-r8이 **P1b의 ≥80% 이득(9.1→7.8° 이상)** + ADD 전이 + 키포인트 무손상이면 → **P1b 폐기, LoRA를 배포·논문 주장으로.**

### 18.7 학습명령
```bash
cd /home/najo/NAS/DIP/3_pose_models/DINObotPose3/TRAIN
U=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1)
IEF_ITERS=4 LORA_LAST_N=4 LORA_R=8 LORA_ALPHA=16 LORA_TARGETS=q_proj,v_proj \
CUDA_VISIBLE_DEVICES=$U /home/najo/.conda/envs/dino/bin/python train_angle.py \
  --detector-ckpt outputs_heatmap/crop_20260605_010622/best_heatmap.pth \
  --train-dir ../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr \
  --val-dir   ../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --angle-backbone dino_lora --head-type ief --image-size 512 --batch-size 32 --epochs 60 \
  --lr 1e-3 --min-lr 1e-6 --weight-decay 1e-4 --fk-weight 10.0 --kp-jitter 2.0 \
  --crop-to-robot --crop-margin 1.5 --num-workers 8 \
  --output-dir ./outputs_angle/lora_r8 --use-wandb --wandb-run-name lora_r8
# r16 arm: LORA_R=16 LORA_ALPHA=32 ... --output-dir .../lora_r16
```
- **예상시간**: 백본 forward **2회**(키포인트 no_grad + 각도), backward는 LoRA(0.2M)+head만 → **P1b보다 가벼움**(ResNet bwd 없음). 60ep **~8–12h**/arm. P1b/mlp-control과 **다른 GPU 병렬**.
- **스모크(5분, `--epochs 1 --batch-size 8`)**: (i) `set_lora(False)`일 때 kp2d가 배포와 **동일**(‖Δ‖<1e-4) 어써트 = 키포인트 무손상 증명, (ii) LoRA A/B만 `requires_grad=True`·base/keypoint_head `False` 어써트, (iii) 각도 loss 하강, (iv) LoRA 파라미터 수 출력(~0.2M 확인).

### 18.8 초기 target/rank 선택 근거 (문헌, 추측 표기)
- **q/v_proj 우선**(LoRA 원논문 Hu'21 §7.1: q,v가 k,o보다 효율적) 🔶. mlp까지 확장(`LORA_TARGETS=q_proj,v_proj,up_proj,down_proj`)은 r8-qv가 미달일 때 2차.
- **last 4 블록**: §17.2에서 pooled 각도정보가 특정 층에 없었으므로 **후반 블록 폭넓게** 적응이 안전(단 이는 추측 — last_n=6/8 스윕 여지).
- **초기 B=0**: 학습 시작 시 base와 완전 동일 → do-no-harm 시작, 안정.

**전략 요약**: KUKA/Baxter의 SOTA-근접은 **P1b(각도)가 아니라 depth head가 1순위 레버.** RootNet식 depth(학습형, mesh 불필요)를 rot-head에 붙이는 것이 양 로봇 공통 최대 EV. P1b는 Panda 승리 시 KUKA-R의 2차 보강. (**§13.4 민감도로 Baxter-wrist 항목은 폐기** — 아래 참조.)

---

## 13. depth head 설계 (RootNet 이식) — KUKA/Baxter 최대 격차의 핵심 레버

### 13.1 HoRoPose RootNet 정밀 스펙 (로컬 코드 직독)

**핵심 구조: depth를 "기하 prior × 학습 보정"으로 분해.** 네트워크는 metric depth를 직접 회귀하지 않고 **무차원 보정계수 γ만** 예측한다.

| 요소 | 구현 | 파일:라인 |
|---|---|---|
| **k_value**(기하 prior) | `k = sqrt(fx·fy·real_bbox_w·real_bbox_h / area)`, `real_bbox=[1000,1000]`(mm), `area = max(\|bbox_w\|,\|bbox_h\|)²`(px²) | `lib/core/function.py:88-97` |
| feature | rootnet_backbone(HRNet32/ResNet) → **GAP** (+선택 `depth_fc_*` residual MLP) | `full_net.py:252-268` |
| **γ 회귀** | `gamma = self.depth_layer(img_feat)` (1×1 conv → 스칼라) | `full_net.py:274-275` |
| **z 합성** | `pred_depth = gamma · k_value / 1000` (→ meters) | `full_net.py:282-283` |
| **x,y** | `pred_trans = uvz2xyz_singlepoint(pred_root_uv, pred_depth, K)` — root 키포인트 **uv를 z로 역투영** (x,y는 회귀 안 함) | `full_net.py:305` |
| loss | root depth **L1 ×10**(`loss_depth`), root uv ×1, trans ×1 | `function.py:216-252`, `configs/panda/full.yaml` |

**왜 강한가**: k_value는 "1m×1m 물체가 이 bbox를 채우는 깊이" = **겉보기 크기→깊이의 기하 법칙**. 네트워크는 O(1) 보정만 배우면 되어 raw metric depth 회귀보다 훨씬 쉽고 스케일이 데이터에서 온다. **mesh 불필요** → KUKA RC 차단(에셋 부재)을 우회하는 유일한 depth 경로.

### 13.2 우리 rot-head와의 접목

**현 상태(문제)**: `RotationHead`(`TRAIN/model_angle.py:199-233`)는 `t_out(h) + t_base`로 t를 **직접 회귀**, `t_base=[0,0,1.1]` — **깊이 prior가 "고정 1.1m 상수"**. 겉보기 크기 단서가 전혀 없음. 학습은 `t_loss = SmoothL1(o['trans'], tg)`(`train_rotation.py:132`). **이것이 KUKA t-err 56mm의 직접 원인.**

**접목안 (a) — z만 대체 [추천]**: rot-head에 `gamma_out(hidden→1)` 추가. `z = γ·k_value/1000`, `t = [t_out[0], t_out[1], z]`.
- 근거: (i) 최소변경, (ii) **direct-pose 성질 보존**(x,y는 계속 appearance 회귀 → KUKA link-혼동 재유입 없음), (iii) monocular에서 **z가 지배적 오차** → 최대 이득, (iv) γ 학습이 쉬움.

**접목안 (b) — full RootNet [ablation]**: `z=γ·k_value` + `x,y = uvz2xyz(root_uv, z, K)`.
- t 전체를 고침(정확한 키포인트 uv 활용)이나 **안정적 reference 키포인트 필요**. 로봇별 선택 근거(검출 catastrophic률): **KUKA는 base가 혼동 심함(link_1 15.1%) → `iiwa7_link_6`(7.5%) 권장**, **Baxter는 어깨가 거의 완벽(`left_s0` 1.6%)** → s0 권장. (a) 통과 후 추가 이득 확인용.

> ⚠️ **구현 필수 주의**: k_value는 **원본 이미지의 robot bbox 면적 + 원본 K**로 계산해야 한다. 우리는 `--crop-to-robot`로 잘라 512로 **리사이즈**하므로 crop 안에서는 겉보기 크기가 정규화되어 **깊이 단서가 소멸**한다. batch의 원본 keypoints/`camera_K`/`original_size`를 써야 함(`scale_K` 적용 전).

### 13.3 코드 이식 골격
```python
# model_angle.py — 신규 helper
def compute_k_value(kp2d_orig, K_orig, real_size_mm=1000.0):
    """RootNet 기하 depth prior. kp2d_orig/K_orig는 반드시 '원본 프레임'(crop-resize 전)."""
    x, y = kp2d_orig[..., 0], kp2d_orig[..., 1]
    bw = x.amax(1) - x.amin(1); bh = y.amax(1) - y.amin(1)
    area = torch.clamp(torch.maximum(bw, bh) ** 2, min=1.0)          # px^2
    fx, fy = K_orig[:, 0, 0], K_orig[:, 1, 1]
    return torch.sqrt(fx * fy * real_size_mm * real_size_mm / area)  # (B,) mm

# RotationHead.__init__ 에 추가
self.use_rootnet_depth = use_rootnet_depth
self.gamma_out = nn.Linear(hidden, 1) if use_rootnet_depth else None

# RotationHead.forward(..., k_value=None) 말미
if self.predict_t:
    t = self.t_out(h) + self.t_base.to(h.device)
    if self.use_rootnet_depth and k_value is not None:
        z = (F.softplus(self.gamma_out(h)).squeeze(-1) * k_value) / 1000.0   # softplus: z>0 보장
        t = torch.cat([t[:, :2], z.unsqueeze(-1)], dim=-1)                    # (a) z만 대체
    return d6, t
```
- `AnglePredictor.forward`: 원본 kp2d/K로 `k_value` 계산 → `rot_head(..., k_value=k_value)` 전달.
- `train_rotation.py`: `--depth-head` 플래그(→`use_rootnet_depth=True`) + 선택 `--depth-weight 10`로 z 직접 감독 추가:
  ```python
  if args.depth_weight > 0:
      loss = loss + args.depth_weight * F.l1_loss(o['trans'][keep][:, 2], tg[keep][:, 2])
  ```
  (HoRoPose가 depth에 ×10을 주는 것과 대응; 기존 `t_loss`만으론 z 그래디언트가 약함.)

**학습명령 (KUKA / Baxter, Panda GPU 해제 후)**:
```bash
cd /home/najo/NAS/DIP/3_pose_models/DINObotPose3/TRAIN
U=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1)
CUDA_VISIBLE_DEVICES=$U /home/najo/.conda/envs/dino/bin/python train_rotation.py \
  --detector-ckpt outputs_heatmap/kuka_dream_detector_20260709_183119/best_heatmap.pth \
  --train-dir /home/najo/NAS/DIP/datasets/synthetic/kuka_synth_train_dr \
  --val-dir   /home/najo/NAS/DIP/datasets/synthetic/kuka_synth_test_dr \
  --fk-robot kuka --angle-joint-names iiwa7_joint_1,...,iiwa7_joint_7 \
  --depth-head --depth-weight 10 --t-weight 1.0 \
  --epochs 40 --batch-size 32 --lr 1e-3 --crop-to-robot --crop-margin 1.5 \
  --output-dir ./outputs_rotation/kuka_depthhead --use-wandb
# Baxter: --fk-robot baxter, baxter 검출기/데이터, --angle-joint-names left_s0,...,left_w2
```
- **예상시간**: 기존 rot-head 학습과 동급(γ는 스칼라 1개 추가) — 40ep **~4–8h**/로봇, 2 GPU 병렬.
- **스모크(5분)**: `--epochs 1 --batch-size 8` — (i) `k_value` 분포가 로봇 실제 깊이 대역(~1–2m)과 일치하는지 출력, (ii) γ가 ~O(1)인지, (iii) val `t-err mm`이 기존 56mm 근방에서 시작해 하강하는지.

### 13.4 판정 — z-err ↔ ADD-AUC 민감도 (오프라인 FK 실측)

**전제**: KUKA는 **GT-pose 주입 시 AUC 0.9999**(doc §구조정합) → **격차 전체가 rot-head(R,t)**. FK로 오차→ADD 변환(iiwa7 평균 키포인트 반경 **0.456m**, baxter **0.394m**; 400 랜덤 config).

**KUKA** (R-오차 → `r̄·sin θ`, t-오차 → 그대로, 독립 합성):
| 시나리오 | t | R | ADD | 모델 AUC |
|---|---|---|---|---|
| 현재 | 56mm | 7.4° | 81mm | 0.19 |
| **depth만 고침** | 20mm | 7.4° | 62mm | 0.38 |
| R만 고침 | 56mm | 2.5° | 59mm | 0.41 |
| **둘 다** | 20mm | 2.5° | 28mm | **0.72** |
| 공격적 | 15mm | 2.0° | 22mm | 0.78 |

→ **결정적: t(56mm)와 R(7.4°→59mm)이 거의 동등 기여. depth head 단독으론 SOTA-근접 불가**(목표 ~0.72는 **t≤20mm AND R≤2.5° 동시** 필요). **depth head는 필수지만 rot-head R 개선과 반드시 페어링**(→ P1b co-trained feature를 rot-head에, 또는 IEF-style iterative rot).
> 모델 caveat: 프레임별 ADD 이질성을 무시한 상수근사라 절대값은 보수적(현재 모델 0.19 vs 실측 0.357). **상대 델타만 신뢰.**

**Baxter — 예상 밖 결과(진단 정정)**: wrist 각도 오차(J4 25.3°, J5 21.0°)의 키포인트 변위는 **각 5.7mm/6.0mm, 합계 ~8mm**에 불과. 체인 말단이라 **레버암이 거의 없음**.
| 시나리오 | ADD | 모델 AUC |
|---|---|---|
| 현재 (t60, R5.7°, wrist 8mm) | 72mm | 0.28 *(실측 0.252 — 모델 잘 맞음)* |
| **wrist만 고침** | 72mm | **0.28 (+0.005)** |
| **pose(t20,R2.5°)만 고침, wrist 방치** | 28mm | **0.72** |
| 둘 다 | 26mm | 0.74 |

→ 🔴 **"Baxter 병목 = wrist 관측성"은 ADD 기준으로 오진.** wrist를 완벽히 고쳐도 **+0.005**(무의미). **Baxter 격차는 사실상 전부 pose(t,R)**. ⇒ **§12.3의 Baxter 각도head P1b(wrist appearance)는 폐기.** (angle MAE 지표가 ADD와 해리되는 Panda 교훈의 재확인 — 각도 MAE 개선이 ADD로 이어진다는 가정을 로봇마다 **FK 레버암으로 검증**해야 함.)
> ⚠️ **단 "pose가 지배적" ≠ "depth head가 유효"**. t-err는 **3D 노름**이지 z성분이 아니다(§13.5) — RootNet-(a)는 z만 고치므로, 오차가 측면(x,y) 위주면 무효. **분해 측정 전엔 미확정.**

**판정 기준(학습 후)**: val `t-err` **56→≤25mm**(KUKA)·**60→≤25mm**(Baxter)면 depth head PASS. 그 다음 rot-R을 ≤2.5°로 낮추는 2단계가 SOTA-근접(0.72+)의 필요조건.

### 13.5 충돌 해소 — "Baxter RC 실패 = depth는 병목 아님"인가? (사용자 지적)

**충돌**: 07-13 기록은 Baxter 실루엣 RC(깊이 보정)가 포즈를 악화시켰고 그 진단이 "**병목은 깊이가 아니라 wrist 각도**"였다. §13.4는 정반대(“wrist 무관, pose 지배”)를 주장한다.

**실측 재확인 (`Eval/ablation_logs/baxter_rc/baxter_rc.log`)**:
```
BEFORE (direct-pose): ADD-AUC 0.2621 | mean 87.3mm | median 74.0mm
AFTER  (RC silhouette): ADD-AUC 0.0089 | mean 219.8mm | median 210.9mm
```
전 구간에서 균일하게 악화(85mm→215mm, 500/500 프레임). **깊이가 이미 정확했다면 보정기는 '중립'이어야지 전 프레임 3배 악화가 나오지 않는다. AUC 0.26→0.009는 최적화 발산(divergence)의 지문** — 즉 **(a) 방법 문제**(실루엣-깊이 모호성 하에서 앵커·게이팅 없는 자유 최적화가 표류). 07-13 문서 자신도 "*Panda식 앵커링+do-no-harm 게이팅 없이는 해로움*"이라 적고 있다.
**⇒ 판정: (a) 방법 실패. 이 실험은 "Baxter의 depth가 정확하다"의 근거가 아니다.** 다만 이것이 "depth가 레버다"를 **증명하지도 않는다**(아래 미측정 항목).

**🔴 미측정 결정 변수 (GPU 태우기 전 반드시 확인)**: `train_rotation.py:152`는 `terrs = (trans - tg).norm(dim=-1)*1000` — **3D 노름**이다(변수명 `tz_med`는 오해 유발, z성분 아님). **어떤 코드도 z vs xy 분해를 로깅하지 않는다.** 따라서 "t-err 56/60mm ≈ 깊이 오차"는 **가정이지 실측이 아니다.** RootNet-(a)는 z만 대체하므로 이 분해가 유효성을 결정한다.

**🟢 새로 발견 — Baxter rot-head는 미수렴**: `outputs_rotation/baxter_rot_20260713_074833/train.log` 최종이 **Ep11**(doc "Ep11 중단"), t-err 59.7mm. 반면 KUKA는 **Ep29**까지 수렴(t-err 82→56mm, ADD-AUC **0.22→0.34**). **Baxter는 단순히 학습을 끝내는 것만으로 KUKA와 같은 +0.1 규모 이득이 기대되는 미소진 구간이 남아 있다** — 새 아키텍처 이전에 이것부터.

**RootNet의 정당성 (문서상 명확화)**: RootNet은 **"RC보다 낫다"가 아니다.** 우리는 이미 렌더러 기반 scale→depth 보정(RC)을 보유하며 **Panda 최대 레버(+0.043)**다 — **Panda는 RC 유지.** RootNet의 유일한 정당성은 **로봇별 RC 가용성**:
| 로봇 | mesh/RC 가용 | depth 경로 |
|---|---|---|
| **Panda** | ✅ 작동, 최대 레버 +0.043 | **RC 유지**(RootNet 불필요) |
| **KUKA** | ❌ **정확 iiwa7 mesh 부재 → RC 원천 차단**(bullet3 변종 20mm 불일치, RoboPose 배포 사망) | **RootNet이 유일한 depth 경로** |
| **Baxter** | ✅ mesh DREAM과 **0.00mm 일치**, SAM IoU 0.82 — **차단 아님**, 실패는 방법(앵커/게이팅 부재) | **RC 수리가 1순위 후보**(자산 보유 + Panda 레시피 검증됨), RootNet은 2순위 |

**⇒ 우선순위 정정**:
- **P0(게이트, ~5분 GPU)**: 기존 rot-head 체크포인트로 **t-err의 z/xy 분해 측정**(val-only, KUKA+Baxter). `train_rotation.py` val 루프에 2줄 추가. **z가 지배적일 때만 depth head에 GPU 투입.**
- **P1(Baxter, 저위험)**: **rot-head 학습 완주**(Ep11→40). 새 코드 0, KUKA 선례상 최대 이득 구간.
- **P2(KUKA)**: RootNet depth head — RC 원천 차단이라 대안 없음. 단 P0 통과 조건.
- **P3(Baxter)**: RC에 **Panda식 앵커링+do-no-harm 게이팅** 이식(test-time, 학습 불필요) → 실패 시에만 RootNet.
- **추측 표기**: "HoRoPose/RoboTAG의 Baxter 58.8이 depth 덕"은 **내 추론이며 미검증**(§12.1). 근거는 정황뿐 — 명시적 depth 기제 보유(HoRoPose RootNet / RoboTAG DepthAnything)가 58.8, 미보유(RoboPose 32.7·RoboPEPP 34.4)가 낮다는 **패턴 상관**. 게다가 두 방법이 **정확히 동일한 58.8**인 점은 baseline 행 전재 가능성도 있어 신뢰도를 낮춘다. **인과로 취급하지 말 것.**

### 13.6 KUKA RC 차단 해제 (07-22) + Baxter RC 실패 원인 확정

**전제 정정(실측, 사용자/코디네이터 확인)**: `RoboPEPP/urdfs/iiwa_description/`의 iiwa7 URDF joint origin이 우리 DREAM 피팅과 **최대 0.04mm 일치**(joints1–5 = 0.15/0.19/0.21/0.19/0.21 정확; joint6 `hypot(0.0607,0.19)=0.19946` vs 우리 0.1995; joint7 `hypot(0.081,0.0607)=0.10122` vs 우리 0.1012), `meshes/iiwa7/visual/link_0~7.stl` 전부 존재. ⇒ **"KUKA는 정확 메쉬 부재로 RC 원천 차단"은 오판**(bullet3 변종만 보고 내린 판정). **KUKA RC 차단 해제.**

> 🔴 **배선 시 필수 주의(실측)**: 우리 `model_v4.iiwa7_forward_kinematics`는 DREAM 링크 **위치**를 0.003mm로 재현하지만 **중간 링크 프레임의 방향(orientation)은 URDF와 다르다** — θ=0에서 URDF link_2 `[0,0,0.34]` vs 우리 `[-0.082,0.0512,0.3136]`이고, **관절 부호 128조합 전탐색으로도 일치 없음**(최소 236mm). 이는 문서가 경고한 "중간 프레임 방향은 gauge" 그대로다. **RC 메쉬 배치는 반드시 URDF FK 프레임으로 해야 하며, 우리 피팅 FK를 쓰면 실루엣이 조용히 틀어진다.**

**Q2 — Baxter RC 실패는 SAM 품질인가, 최적화 발산인가? → 발산(방법) 확정.**

| | SAM-vs-init-render IoU |
|---|---|
| **Panda**(정상 RC 런, 로그 실측) | mean **0.83–0.88**, median ~0.85, frac≥0.5 = **0.97–1.00** |
| **Baxter** | **0.82** |

**마스크 품질은 사실상 동일** → SAM은 원인이 아니다. 원인은 **목적함수·안전장치의 부재**(코드 대조):

| 안전장치 | Panda `rc_refine_from_dump.py` | Baxter `baxter_rc_eval.py` |
|---|---|---|
| **do-no-harm IoU 게이트** | ✅ `--min-iou 0.35`, IoU 미달 시 **refine 자체를 skip**(L52, L152) | ❌ **없음** — 전 프레임 무조건 refine |
| **재투영 앵커** | ✅ "structure + **reproj anchor**"(L65) — 검출 2D가 해를 고정 | ❌ **없음** — 실루엣 항 단독 |
| 가설 채택 마진 | ✅ `--ms-margin 0.01`(도전자가 이겨야 채택) | ❌ 없음 |
| 최적화 DOF | 깊이/스케일 보정 중심 | ❌ **t + 전 관절각 동시**(13+ DOF) |

Baxter 손실은 `loss = 1 - soft_iou(r, mask_t) + 0.5*(da**2).mean()`(L107) — **실루엣 IoU + 약한 각도 prior뿐**. 실루엣은 깊이/스케일 방향으로 **평평한 매니폴드**를 가지므로, 앵커 없이 13+ DOF를 자유 최적화하면 표류가 필연. 500/500 프레임 균일 악화(85→215mm, AUC 0.262→0.009)가 그 지문. ⇒ **확정: 발산(방법). 마스크 아님. depth가 병목이 아니라는 근거도 아님.**

**⇒ KUKA RC 배선 체크리스트(배선 에이전트 전달용)**:
1. **메쉬 배치는 URDF FK 프레임으로**(위 경고). 우리 피팅 FK는 위치 전용.
2. **`--min-iou` do-no-harm 게이트 필수** — IoU 미달 프레임은 refine 건너뛰고 원본 유지.
3. **재투영 앵커 필수** — 검출 2D 키포인트로 해를 고정(Baxter가 빠뜨린 바로 그 항). 이것 없이는 실루엣 단독 = 발산.
4. **DOF 최소화** — **t(깊이/스케일) 우선**, 관절각 동시 최적화 금지(Baxter 실패 모드). 각도는 앵커/prior로 묶을 것.
5. **채택 게이트** — refine 후 IoU/재투영이 개선될 때만 채택, 아니면 원본 롤백(do-no-harm).
6. **근거리 off 규칙 확인** — Panda는 근거리(azure) RC off. KUKA 거리 분포에 맞춰 조건부 on/off.

*(Baxter 메쉬: RoboPEPP에도 있으나 우리 `baxter_common`이 이미 DREAM과 0.00mm 일치 → 새 정보 없음. 확인 완료, 조치 불필요.)*

---

## 14. 데이터 면책: oracle-angle 0.899의 원인은 예측 2D이지 데이터 오류가 아님 (07-22)

**질문.** oracle-angle(GT 관절각을 주입한 뒤 R, t를 솔버로 푸는 구성)이 ADD-AUC@100mm = 0.899에 그치고 1.0에 도달하지 못한다. 그렇다면 DREAM 합성 데이터(`panda_synth_test_dr`) 자체가 부정확한 것은 아닌가. 즉 GT 관절각, GT 3D 키포인트, GT 2D 투영, 그리고 우리 FK 사이에 정합 오차가 존재해 상한을 0.899로 눌러버리는가.

**결론: 아니다. 데이터와 FK는 마이크로미터 이하 정밀도로 면책된다.** 오늘(2026-07-22) `panda_synth_test_dr` 앞 300프레임에서 두 독립 검증을 수행했으며, 남은 격차를 다음 3단계로 분해한다.

| 구성 | ADD-AUC@100mm | ADD 평균 | 의미 |
|---|---|---|---|
| GT 각도 + GT 2D 키포인트 + PnP | **0.9998** | 0.011mm | 데이터/FK 자기정합 상한, 사실상 완벽 |
| GT 각도 + 예측 2D (oracle-angle) | 0.899 | 약 10mm | 2D 검출 비용 = **−0.10** |
| 예측 각도 + 예측 2D (배포) | 약 0.788 (good-frame CLEAN) | | 각도 예측 비용 = **−0.11** |

세 행을 위에서 아래로 읽으면, 모든 항이 GT일 때 파이프라인은 사실상 완벽(0.9998)하고, 2D만 예측으로 바꾸면 0.10을 잃으며, 각도까지 예측으로 바꾸면 다시 0.11을 잃는다. 데이터가 상한을 누르는 항은 어디에도 없다.

### 14.1 검증 방법과 재현 정보

두 검증 모두 대상은 `panda_synth_test_dr`의 앞 300프레임이고, 사용한 데이터 필드는 다음과 같다.

- **GT 관절각**: `sim_state.joints`
- **GT 3D 키포인트(카메라 프레임)**: `objects[0].keypoints[].location` (단위 cm, m로 환산)
- **GT 2D 키포인트**: `projected_location`
- **카메라 내부 파라미터**: `meta.K`
- **FK**: `model_v4.panda_forward_kinematics`, 키포인트 인덱스 `kp_indices = [0, 2, 3, 4, 6, 7, 9]` (각각 link0, link2, link3, link4, link6, link7, hand의 7개 키포인트)

**검증 1: 순수 3D-3D 정합 (Kabsch, 검출기 없음, PnP 없음).** `panda_forward_kinematics(sim_state.joints의 GT 관절각)`로 로봇 프레임 7개 키포인트(link0, 2, 3, 4, 6, 7, hand)를 생성하고, 이를 GT 카메라 프레임 3D 키포인트(`objects[0].keypoints[].location`, cm를 m로 환산)에 강체 정렬(Kabsch/SVD)했다. 잔차 RMS는 평균 **0.005mm**, 최대 **0.013mm**, 키포인트별 **0.003mm에서 0.008mm**였다. 즉 우리 FK는 DREAM 자체 기구학을 5마이크로미터 수준으로 재현하며, 데이터의 GT 관절각, GT 3D 키포인트, GT 2D 투영은 상호 자기정합적이다.

**검증 2: GT 각도 + GT 2D PnP.** FK(GT 각도)로 생성한 3D 키포인트와 GT `projected_location` 2D를 `meta.K` 내부 파라미터와 함께 `cv2.solvePnP`(ITERATIVE)에 입력해 포즈를 풀고, 카메라 프레임으로 변환한 뒤 GT 3D `location`과 비교했다. ADD 평균 **0.011mm**, ADD-AUC **0.9998**이었다. 즉 모든 입력이 ground-truth일 때 파이프라인은 완벽하며, oracle-angle을 0.899로 누르는 유일한 원인은 예측 2D 검출기 오차이다.

### 14.2 전략 함의: 두 레버는 각 약 0.10, 상보적이며 데이터는 레버가 아니다

남은 합성 격차에는 크기가 거의 동등한(각 약 0.10) 두 레버가 있다.

1. **2D 키포인트 검출 (−0.10)**: 완벽한 각도 oracle조차 이 항 때문에 0.899에서 천장에 부딪힌다.
2. **각도 예측 (−0.11)**: P1b 및 별도 학습형 각도-헤드 라인이 공략하는 항이다.

**데이터는 레버가 아니다. 면책되었다.** 완벽한 각도 헤드를 얻더라도 검출기 때문에 0.899에 상한이 걸리므로, 두 레버는 서로 대체재가 아니라 상보재이다. 즉 각도 예측을 아무리 개선해도 2D 검출 개선 없이는 0.899를 넘을 수 없고, 그 역도 성립한다. 따라서 논문 진단/분석 섹션은 합성 격차를 "데이터 결함"이 아니라 "2D 검출 + 각도 예측 두 축의 동시 공략 대상"으로 기술해야 한다.

---

## 19. 왜 DREAM Baxter는 우리에게 유리한가 (공정성 감사 + 메커니즘 규명, 07-22)

> **질문.** intrinsics/solver 수정 이후 Baxter synth(`baxter_synth_test_dr`, LEFT arm 7키포인트, 전체 5982프레임)에서 우리 ADD-AUC@100mm = **0.7125(=71.3)**. 이는 Protocol A 프론티어 RoboTAG/HoRoPose 58.8, RoboPEPP 34.4, RoboPose 32.7을 큰 격차로 앞서 우리를 1위로 만든다. 이는 KUKA(우리 69.0 < RoboPEPP 76.2)나 Panda(우리 74.2 < RoboPEPP 83.0)와 **정반대** 패턴이다. 우선 이 우위가 측정 아티팩트가 아닌 실재임을 검증하고, 그다음 메커니즘을 규명한다.
> 근거: `Eval/baxter_add_eval.py`(solver 모드, true-K), `TRAIN/dataset.py:682-715`(crop_to_robot), `RoboPEPP/test.py:272-292`, `RoboPEPP/datasets/{image_proc.py:541,dream_ssl.py:236-250}`, `Holistic-Robot-Pose-Estimation/lib/models/depth_net.py:92-125`, `robopose/robopose/models/articulated.py:21-246`, CPU 기하 분석 5982프레임 서브샘플.

### 19.1 공정성 감사 (게이팅) — 판정: **우위는 실재, 단 bbox는 GT-크롭이며 자동-bbox 재측정이 논문 주장과의 정합을 위해 필요**

네 축(bbox, 메트릭, test set, 프로토콜)을 순서대로 감사한다.

**(1) bbox source (핵심 우려).** `baxter_add_eval.py`는 `PoseEstimationDataset(crop_to_robot=True, crop_margin=1.5)`를 쓴다. `dataset.py:682-715`를 정독하면 이 크롭은 **GT projected keypoint**(`keypoints_data['projections']` = JSON `projected_location`)의 in-frame 점들로 bbox 중심·변 길이를 잡는다. 즉 **GT-키포인트 bbox(오라클 위치·스케일)이지 검출기 자동 bbox가 아니다.** 이것이 유일한 오라클 요소다(2D 키포인트와 각도는 전부 예측값). 경쟁자 bbox는: **HoRoPose = GT-box**(논문 본문·`sota_survey.md`), **RoboTAG·RoboPEPP = 자동-bbox**(RoboPEPP는 `dream_ssl.py`가 `*_annotated/`의 precomputed `bounding_boxes`를 로드, Baxter는 `get_extended_bbox`가 패딩 없이 raw 반환 후 [0,640]x[0,480] clip). 따라서 프론티어 **HoRoPose 58.8과는 bbox 축에서 apples-to-apples**(둘 다 오라클 box)이고, RoboTAG/RoboPEPP 대비만 우리가 GT-크롭 이점을 가진다.

**GT-bbox 대 자동-bbox 이점의 실측 상한**: 같은 배포 파이프라인의 Panda real 4스플릿에서 `ablation_logs/{gt_bbox,full}_*_base.log` 직접 대조:

| 스플릿 | GT-bbox | 자동(full)-bbox | Δ(GT−auto) |
|---|---|---|---|
| kinect | 0.7635 | 0.7672 | **−0.004** (자동이 오히려 우세) |
| orb | 0.7440 | 0.7382 | +0.006 |
| realsense | 0.7480 | 0.7452 | +0.003 |
| azure | 0.8079 | 0.7945 | +0.013 |
| 평균 | 0.7659 | 0.7613 | **+0.005** |

GT-bbox의 평균 이점은 **+0.005**로 미미하다. 단 Baxter는 원거리(1.31m)이고 프레임의 약 30%에서 팔이 부분적으로 화면 밖으로 나가(in-frame 6.41/7) 자동-bbox 1차 검출이 base 가림 등으로 mis-center될 여지가 Panda real보다 크므로, Baxter의 GT-크롭 이점은 +0.005보다 클 수 있다(보수적으로 +0.02~0.05로 추정, 미측정). **그럼에도 0.7125 − 0.05 = 0.66 > 0.588**이므로 프론티어 대비 우위는 이 축을 최대로 할인해도 살아남는다.

**(2) 메트릭 동일성.** 우리 `add_auc`(`refine_eval.py:26-33`)는 `AUC = trapz(mean(ADD ≤ t), dx=1e-5)/0.1`이고, 프레임별 ADD는 valid 키포인트(=Baxter 7개 전부, off-frame 포함) L2의 평균이다. `RoboPEPP/test.py:272,281-292`는 `dist = norm(pred−gt).mean(-1)` 후 동일 trapz 공식을 쓴다. **두 코드는 수식·off-frame 포함 여부까지 동일**(off-frame 키포인트도 3D location이 있어 valid로 집계되는 harsh 방식). Panda dump에서 우리 AUC가 test.py 산출과 일치함은 §1에서 이미 검증됨.

**(3) test set 동일성.** `datasets/synthetic/baxter_synth_test_dr` = **5982 json 프레임**(실측), 이것이 전체 test set이고 우리 71.3은 그 위에서 산출된다(baxter_add_eval의 `--max-frames`를 전체로 두는 실행 기준). 경쟁자도 full test set 보고이므로 프레임 집합 불일치 없음.

**(4) 프로토콜 A/B 혼동.** 우리 메트릭은 all-keypoints harsh ADD = **Protocol A**이며, 인용된 경쟁 수치(32.7/34.4/58.8)도 RoboTAG/RoboPEPP 평가(Protocol A)다. 과거 문서 §12.1이 "RoboPEPP Baxter 자체보고 75.3(B)"로 적은 것은 **오인**임을 `sota_survey.md:25`가 명시 정정한다: **75.3은 RoboPEPP의 Panda real 컬럼값**이지 Baxter 값이 아니다(paper Table row 165: `... 75.3(Panda) ... 34.4(Baxter)`). 즉 Baxter에서 우리를 넘는 경쟁 수치는 어느 프로토콜에도 존재하지 않는다.

**공정성 판정.** 메트릭·test set·프로토콜은 완전 동일하고, 유일한 오라클 요소인 GT-크롭 bbox의 이점은 실측 상한 +0.005(Panda real), Baxter 보수 추정 +0.05 이내다. 프론티어 HoRoPose 58.8과는 bbox까지 동급(둘 다 오라클 box). **∴ +0.125 우위 중 최소 +0.07~0.12가 살아남는다. 우위는 실재한다.** 단 두 가지 정직한 단서: (a) 논문 본문(`PAPER_OVERLEAF.tex:137`)은 우리 프로토콜을 "bounding box fully automatically"로 서술하나 KUKA/Baxter 수치는 실제로 GT-크롭 solver eval에서 나왔다. 이 불일치를 해소하려면 `selfbbox_eval.py`(자동-bbox)로 Baxter를 재측정해 본문 주장과 수치를 일치시키는 것이 옳다(본 조사에서 재실행하지 않음, 수치 조작 금지). (b) Baxter의 GT-크롭 이점 정밀값은 미측정이므로 위 +0.02~0.05는 추정이다.

### 19.2 H1 (기하 조건수) — 키포인트 spread 측정: **Baxter는 절대 baseline 최대이나 원거리라 subtense는 최소**

`keypoints[].location`(cm를 m로 환산, 카메라 프레임)에서 각 로봇의 tracked 7키포인트 3D 확산을 측정(GPU 불필요, 로봇당 ~1000프레임 서브샘플):

| 지표 | Panda | KUKA | Baxter |
|---|---|---|---|
| 3D bbox 대각선 (m) | 0.875 | 0.791 | **0.937** |
| 최대 pairwise 거리 = 팔 baseline (m) | 0.756 | 0.699 | **0.892** |
| 평균 pairwise 거리 (m) | 0.438 | 0.387 | 0.442 |
| 카메라 거리 (centroid, m) | 0.949 | 0.766 | **1.312** |
| 평균 depth z (m) | 0.919 | 0.704 | **1.203** |
| **subtense = baseline/거리** (PnP depth 조건수) | 0.859 | **1.042** | 0.724 |
| in-frame kp / 7 | 6.70 | 6.07 | 6.41 |
| **우리 ADD-AUC (solver, true-K)** | **74.2** | **69.0** | **71.3** |

**판독(H1은 부분 반증).** Baxter는 물리적으로 가장 커서 **절대 3D baseline이 최대**(0.892m)이나 동시에 **가장 멀다**(1.31m). 그 결과 PnP depth 조건수를 지배하는 **angular subtense(baseline/거리)는 셋 중 최소(0.724)**이고 KUKA가 최대(1.042)다. 게다가 ADD는 metric(mm) 오차라 원거리일수록 동일 픽셀오차의 metric 비용이 커진다(오차 대략 Z/f 비례). **∴ "Baxter는 spread가 넓어 PnP가 잘 조건화되어 우리가 1위"라는 단순 H1은 성립하지 않는다.** 우리 Baxter(71.3)는 우리 Panda(74.2)보다 오히려 낮다. 살아남는 기하적 사실은 약한 형태다: 우리 경로는 **metric-정확 FK 모델 + true K로 R,t를 PnP로 푼다**. 넓고 강체인 팔 모델은 원거리에서도 회귀 없이 포즈를 well-posed하게 만들어 71.3을 낸다. 그러나 이는 "우리에게 특별히 쉬움"이 아니라 "우리 기하 경로가 스케일 불변으로 어디서나 ~0.7"임을 뜻한다. **1위의 진짜 원인은 H1(우리가 급등)이 아니라 H2(경쟁자가 붕괴)다.**

### 19.3 H2 (경쟁자의 학습형 depth가 Baxter에서 붕괴) — 핵심 표

로봇을 Panda-DR에서 Baxter-DR로 옮길 때의 ADD-AUC 낙폭(paper Table 값):

| 방법 | Panda-DR | Baxter-DR | **낙폭** | Baxter depth/pose 메커니즘 |
|---|---|---|---|---|
| RoboPose | 82.9 | 32.7 | **−50.2** | render-and-compare 반복(`articulated.py` renderer+iterative). 원거리·부분 off-frame·좌팔단독은 init/수렴 불량 |
| RoboPEPP | 83.0 | 34.4 | **−48.6** | 관절 IEF 회귀 + confident 키포인트로 BPnP 6DOF(`test.py:252-258`). off-frame 30%로 usable 키포인트 축소 시 BPnP ill-posed |
| RoboTAG | 82.5 | 58.8 | **−23.7** | 3D 브랜치 직접 회귀(DepthAnything init) + topological align. 최선의 경쟁자이나 여전히 24점 붕괴 |
| HoRoPose* | 41.4 | 9.8 | −31.6 | RootNet depth = γ·k_value, k_value ∝ sqrt(f²·A_real/A_img)(`depth_net.py:125`). off-frame로 bbox area가 실제 로봇 크기와 불일치 → apparent-size depth prior가 편향 |
| **Ours** | **74.2** | **71.3** | **−2.9** | 키포인트 → **PnP(metric FK + true K)**로 depth를 **기하적으로** 해결. apparent-size 회귀·render-compare 없음 |

**메커니즘 요지.** 경쟁자는 전부 depth/pose를 **학습**한다: RoboPose는 render-and-compare, RoboPEPP는 부분 가시 키포인트의 BPnP, HoRoPose는 apparent-size 기반 RootNet 회귀. 이 세 메커니즘은 compact·근거리·전신가시인 Panda/KUKA에 최적화되어 있어, **크고(0.94m) 멀고(1.31m) 좌팔만 평가되며 자주 부분 off-frame인 Baxter는 그들에게 out-of-distribution·depth-ambiguous**하다. 특히 HoRoPose의 k_value depth prior는 팔이 화면 밖으로 나가면 bbox area가 전체 로봇 크기를 대표하지 못해 체계적으로 편향된다. 우리는 이 학습형 depth를 아예 우회한다: 넓은 강체 FK 모델을 true K로 PnP하여 원근 foreshortening에서 depth를 **직접** 얻고, off-frame 키포인트는 solver의 conf-gate로 배제한다. 그 결과 우리 낙폭은 −2.9(사실상 평탄)인데 경쟁자는 −24에서 −50으로 붕괴한다. **+0.125 격차는 우리가 급등해서가 아니라 그들이 무너져서 벌어진다.**

### 19.4 H3 (각도는 Baxter 병목이 아님) — 확인

Baxter 관절각 MAE(`mr_logs/baxter_directpose.log`, §13.8): J0–J3 6.6/4.5/10.6/7.1°, **손목 J4/J5 = 25.4/21.2°**, 평균 **12.55°**. 이는 Panda(~9.1°)·KUKA(6.28°)보다 **나쁘다.** 그럼에도 Baxter ADD(71.3)는 그 둘에 필적한다. 이 방향성 자체가 각도 정확도가 아니라 **pose-solve 기하가 ADD 드라이버**임을 가리킨다. §13.4의 FK 레버암 실측이 이를 뒷받침한다: 손목 25°/21° 오차의 키포인트 변위는 각 5.7/6.0mm(체인 말단이라 레버암 거의 없음), 손목을 완벽 교정해도 ADD-AUC는 +0.005에 불과하다. **∴ 우위는 더 나은 각도가 아니라 H1의 약한 형태(스케일 불변 metric-FK PnP) + H2(경쟁자 붕괴)의 결합이다.** 손목 관측성 한계(J4/J5 25°/21°)는 우리·경쟁자 공통이며 ADD 무해하다.

### 19.5 평이한 한 문단 답

Baxter가 우리에게 유리한 이유는 "Baxter가 우리에게 쉬워서"가 아니라 "**Baxter가 경쟁자에게 어려워서**"다. Baxter는 셋 중 가장 크고(팔 baseline 0.89m) 가장 멀며(1.31m) 좌팔만 평가되고 프레임의 30%에서 팔이 화면 밖으로 나간다. 경쟁자들은 depth를 학습으로 추정한다: RoboPose는 전신 메시를 렌더링해 이미지와 반복 정합하고, RoboPEPP는 신뢰 키포인트로 BPnP를 풀며, HoRoPose는 겉보기 크기(bbox 면적)로부터 깊이를 회귀한다. 이 방식들은 작고 가깝고 전신이 보이는 Panda/KUKA에 맞춰져 있어, 크고 멀고 부분적으로만 보이는 Baxter에서 일제히 무너진다(Panda 대비 24에서 50점 하락). 우리는 깊이를 학습하지 않는다. 대신 마이크로미터급으로 정확한 FK 모델과 참(true) 카메라 내부파라미터를 이용해 2D 키포인트로부터 카메라 포즈를 PnP로 **기하적으로** 푼다. 이 경로는 로봇 크기·거리에 불변이라 Panda·KUKA·Baxter 어디서나 약 0.7을 안정적으로 낸다(우리 낙폭은 −2.9로 평탄). 요컨대 우리 절대 성능은 Baxter에서 특별하지 않지만(우리 Panda 74.2보다 오히려 낮음), 경쟁자의 학습형 깊이 추정이 이 구성에서 붕괴하기 때문에 상대 순위가 뒤집혀 우리가 1위가 된다. 공정성 감사 결과 이 우위는 실재하며(메트릭·test set·프로토콜 동일, GT-크롭 bbox 이점은 최대 +0.05로 격차를 못 지움), 유일한 정직한 미결은 논문의 "자동 bbox" 서술과 실제 GT-크롭 eval 사이의 불일치를 `selfbbox_eval.py` 재측정으로 정리하는 일이다.

---

## 20. 원위 키포인트 복원가능성 게이트 — multi-hypothesis 히트맵-모드 디코더 판정 (07-22)

> **질문.** §1의 catastrophic distal 2D tail(good-frame ADD 0.788, argmax err>10px 6.5%)을 고치는 **discrete top-M 히트맵-모드 열거 + kinematic FK-재투영 모드선택** 디코더는, **정답 링크 위치가 2차 히트맵 모드로 살아남아야만** 성립한다. 현 dump는 argmax kp2d만 있고 히트맵이 없어 이 복원가능성은 **한 번도 측정된 적 없다.** 이것이 디코더 계열 전체를 여는/닫는 게이트.
> 측정: `Eval/_debate_tmp/recover_gate.py`(배포 crop 파이프라인 그대로 — stage1 solve→bbox, crop, crop-detector 2차패스 → 히트맵당 top-5 NMS 모드를 sub-pixel DARK 정제해 orig-640 px로 덤프) + `gt_response_probe.py`(정답 위치 히트맵 응답) + `analyze_gate.py`. panda_synth_test_dr 1000 good-frame. **측정 전용, 학습·커밋 없음.**

### 20.1 검증(sanity) — 좌표계·파이프라인 정합 확인
- **top-1 NMS 모드 = 배포 dark-decode argmax 정확 일치**: `|peak0 − dark_argmax| = 0.000px`. 좌표사슬(crop-IS→full-IS→orig-640) 무결.
- **argmax catastrophic tail 재현**: orig-640에서 **>10px = 6.48%**, median **1.46px**(found & on-frame 6678 kp) — 진단 주장(~6.5% / 1.46px)과 정확 일치.
- **base argmax ADD-AUC = 0.703** (배포 0.704 재현).

### 20.2 복원가능성 (정답 근방 2차 모드 | argmax가 catastrophic >10px, n=433 kp)
| secondary 범위 | ≤3px | ≤5px | ≤8px |
|---|---|---|---|
| **top-2/top-3** (모드 idx 1–2) | 3.7% | **7.2%** | 15.2% |
| top-2..top-5 (상한) | 4.2% | 9.2% | 19.6% |
| distal(l3/6/7/hand) top-3 | 4.0% | **7.4%** | 16.5% |

per-distal ≤5px: **link3 13.1 · link6 8.2 · link7 7.1 · hand 2.5%**. → **복원가능성 ≈ 7~9%, 게이트 40%에 크게 미달.**

### 20.3 특성화 — "confidently-wrong, 정답 모드 부재"
- catastrophic kp: 평균 유의모드 수 **n_sig=2.76**, peak2/peak1=**0.493**, **unimodal(nsig=1) 34.9% / bimodal 65.1%** (clean은 n_sig=1.01, 0.021, 99.1% unimodal). → tail은 **35%가 단일-강한-오답모드**, **65%는 2차 강모드가 있으나 그 모드가 정답 링크가 아님**(복원 7~9%).
- **결정적(정답 위치 히트맵 응답 / peak-max)**: catastrophic **median 0.074**(56.8%가 <0.1, >0.3은 14.7%뿐) vs clean **median 0.894**(99.2%가 >0.3). → **정답 위치에 히트맵 질량이 사실상 없다.** 즉 정답 모드는 "약한 2차 피크로 존재하나 열거하면 잡히는" 게 아니라 **아예 부재**다. 열거 디코더가 선택할 대상이 없음.
- 메커니즘: argmax가 **다른 kp의 GT 5px 내로 스냅 17.1%**(8px 25.2%) — §14.1/§16.2의 link-혼동 시그니처 존재(단 KUKA 90%보다 약함; 나머지는 self-similar 텍스처/배경으로 확산).

### 20.4 보너스 — dense 히트맵 정제기는 tail을 못 살린다
`solve_pose_kinematic.solve_batch_heatmap`(argmax 앵커 단일시작 Adam, heatmap 응답 최대화): base **0.703 → 0.463 (−0.24)**. 복원은커녕 대폭 악화 — good 프레임까지 강한 distractor 모드로 끌려감. 디베이트 예측(단일시작·argmax 앵커라 tail 복원 불가) 확정.

### 20.5 게이트 판정 — **디코더 계열 DEAD**
복원가능성 **7~9% ≪ 40%**. catastrophic distal tail은 **confidently-unimodal-wrong(35%) 또는 multimodal-but-all-wrong(65%)** 이며, GT-위치 응답 median 0.074로 **정답 모드가 부재**하다. top-M 모드 열거 + kinematic 선택 디코더는 없는 모드를 못 고른다. → **§16.4 idea④(assignment 검출)·모드-열거 디코더 계열 전부 사망 확정**(이전엔 "미검증"이었음). **남은 유일한 synth 2D 레버 = crop-aspect 수정(`cropasp_a43`, 학습중).**
- **가정법 이득(만약 지었다면)**: distal catastrophic 프레임 18.8%(188/1000) × 복원 9% × 부분 ADD 헤드룸 ⇒ **<+0.005**. §1.2 perfect-distal 상한(+0.03~0.05)의 <10%만 포착 → 착수가치 없음. (2D tail은 §16.4대로 **화면밖(azure +0.041)** 축에서만 test-time으로 회수 가능하고, on-frame link-혼동은 원리적으로 test-time 신호 부족이라는 §16.2 결론을 Panda synth에서 정량 재확인.)

---

## 21. 격차 인과 재판정 + 백본 질문 종결 (2026-07-22, 재검토 에이전트)

> 오케스트레이터 질의: "왜 Panda **synthetic**에서 프론티어(RoboPEPP 0.830 / RoboTAG 0.825 / RoboPose 0.829)에 −0.06 뒤지는가 — 인과적으로 규명하고 닫을 수 있는지 판정하라. 주(主) 용의자 = 경쟁자의 **co-trained 백본** vs 우리 **frozen DINOv3**." 본 절은 (1) −0.06의 인과 분해, (2) 백본 co-finetune 반증의 confound/synth-coverage 상태, (3) 판정(closeable vs concede)을 확정한다. **신규 학습·커밋 없음** — dump 재계산 + CPU 측정만(`Eval/rc_dumps_gf/{mlpctrl,p1b_ep17}_gf.npz`, `ablation_logs/oracle_angle_synth/`).

### 21.1 −0.06 격차는 사실상 100% "각도 예측"이다 — 2D도 off-frame도 backbone-broad도 아니다

full-set 실측(`ablation_logs/oracle_angle_synth/results.tsv`):

| 구성 | DR base | DR +RC | Photo base | Photo +RC |
|---|---|---|---|---|
| pred (배포) | 0.704 | **0.769** | 0.738 | **0.799** |
| oracle-angle (GTθ + pred-2D) | 0.861 | **0.886** | 0.869 | **0.897** |
| GTθ + GT-2D + PnP (§14) | 0.9998 | — | — | — |
| RoboPEPP (직접비교) | — | **0.830** | — | **0.841** |

**인과 판독(결정적).** GT 각도만 주입하면 우리 2D+솔버는 **이미 RoboPEPP를 넘는다**(DR +RC 0.886 > 0.830 = **+0.056**, Photo 0.897 > 0.841 = +0.056). ⇒ 우리 **2D 검출·bbox·depth·가림은 −0.06 격차의 원천이 아니다.** off-frame tail은 AUC 기여 0(§1.1), 2D distal tail은 복원 불가 floor(§20)이나 **그 floor를 포함한 oracle-angle이 이미 0.886 > 0.830**이므로 2D floor조차 RoboPEPP 격차엔 **0% 기여**. **−0.06 전량이 각도 예측 축에 있다**(배포 0.769 → oracle-angle 0.886까지 +0.117 헤드룸 중 RoboPEPP를 넘는 데 필요한 건 +0.061뿐).

⇒ 오케스트레이터의 주 용의자 **"co-trained 백본의 broad accuracy 우위"는 반증된다**: (i) detection 축 직접 반증(§17.1 — synth 2D reproj 1.2–1.5px ≥ real 1.4–1.9px, frozen DINOv3가 합성에서 더 정밀), (ii) oracle-angle 우위(0.886 > 0.830)로 재확인. 우리가 뒤지는 것은 backbone-broad가 아니라 **각도 회귀 단일 축**이다. **0.06 분해: 각도 ≈100%, 2D-broad ≈0%, off-frame tail 0%.**

### 21.2 🔴 각도 축의 유일 frozen-호환 레버(P1b co-trained feature)를 dump로 실측 — angle-MAE는 이기나 ADD는 무효 (신규 반증)

P1b(별도 학습 ResNet50 각도 feature, keypoint 경로는 frozen DINOv3 불변 — **반증된 "공유 백본 적응"과 구조적으로 다름**, §11)의 최신 체크포인트 dump를 mlp-control(frozen DINOv3 768d)과 **동일 1000프레임**에서 A/B 재계산:

| 지표 | mlpctrl (DINOv3, ~9.1° angle) | **P1b ep17 (ResNet50, 7.47° angle)** | Δ |
|---|---|---|---|
| angle head_theta 차이 (mlp 대비) | — | **6.70° mean** (예측이 실제로 크게 다름) | — |
| solved θ 차이 (솔버 통과 후) | — | **3.59° mean** (솔버가 절반 washout) | — |
| good-frame(859) ADD-AUC | **0.7707** | 0.7672 | **−0.004** |
| good-frame median ADD | 14.8 mm | 14.7 mm | ~0 |
| good-frame tail>100mm | 4.5% | 4.8% | +0.3%p |
| ALL-1000 ADD-AUC | 0.7128 | 0.7113 | −0.002 |
| p1b가 프레임별 ADD 우세 비율 | — | **47.8%** (동전던지기) | — |

**메커니즘(왜 무효인가).** 우리 솔버는 θ(7)+R(3)+t(3)=13DOF를 재투영으로 **공동 최적화**한다(`solve_pose_kinematic.py:274`). 따라서 feed-forward 각도가 6.70° 개선돼도 **솔버가 그 절반을 washout**(6.70°→3.59°)하고, 남은 차이는 **noisy 2D에 과적합된 최종 θ**에 흡수돼 **median ADD가 비트 동일(14.7≈14.8mm)**. 즉 **good-frame ADD의 binding constraint는 각도 head/feature가 아니라 "솔버의 2D-과적합"**이다 — §6의 "head θ가 noisy → 솔버가 2D 과적합 → 0.788 천장" 진단이 여기서 실증되며, 나아가 **각도 feature를 고쳐도 이 천장은 안 열린다**는 것이 증명됐다(P1b는 §17의 낙관적 "7.47° 돌파" 서술이 예측한 ADD 이득을 내지 못함). 이전 dump `p1b_gf.npz`(더 이른 ep)도 0.7672로 동일 → **에폭이 지나며 angle-MAE는 좋아지는데 ADD는 0.767 고정** = 각도-feature가 ADD의 binding이 아님을 재확인(angle-MAE↔ADD 해리, §2 실패표의 재현).

**보강 사실.** from-scratch 백본(random-init ViT full-train, `bbabl_random-unfrozen`)은 detection real-val AUC **0.455 ≪ frozen DINOv3 0.80** — 키포인트 경로에서 **DINOv3 사전학습이 결정적 load-bearing**이며 "unfreeze/from-scratch가 더 낫다"는 방향은 detection에서도 성립 안 함(SUMMARY REFUTED와 일관).

### 21.3 co-finetune 반증 상태 — confound인가? synth를 커버했나?

| 실험 | 시점 | 대상 도메인 | 지표 | frozen/unfrozen | 판정 | crop-aspect confound? |
|---|---|---|---|---|---|---|
| SSL masked-feature (6-block) | 2026-06-08 | **REAL 적응** | realsense **ADD** | 공유 백본 unfreeze | 0.531 < 0.567 | 무관(real 문제, synth 미측정) |
| pseudo-keypoint co-finetune | 2026-06-08 | **REAL 적응** | realsense **ADD** | 공유 백본 unfreeze | 0.497→0.434 단조↓ | 무관(real 문제, synth 미측정) |
| **P1b co-trained angle feature** | **2026-07-22** | **SYNTH-DR** | **synth good-frame ADD** | keypoint frozen / angle-net trainable | **angle 7.47° PASS / ADD 0.767 FAIL** | **아니오(within-dump A/B, mlp-control도 동일 크롭)** |

**정밀 답변.**
- **(a) 2026-06-08 반증은 synth를 커버하지 않았다** — 그것은 *합성-사전학습 백본을 REAL에 적응*시키는 실험이고 지표는 realsense ADD였다. 근본원인("솔버가 요구하는 sub-pixel 키포인트 정밀도를 coarse robustness와 맞바꿔 real ADD 손해")은 **real 주장에 한해 여전히 유효**하며 crop-aspect와 무관(정밀도 손실은 crop 종횡비가 아니라 백본 이동이 원인). 이 반증은 **synth-DR 질문엔 애초에 적용 대상이 아니다.**
- **(b) 그러나 synth-DR co-train 질문은 이미(이번 주) 직접 측정됐다** — P1b가 그것이다. 그리고 P1b는 **가장 clean한 버전**(keypoint 경로 무손상 → 반증된 정밀도-파괴 실패모드가 구조적으로 불가능)임에도 **angle-MAE는 이기고 ADD는 무효**. 이 A/B는 mlp-control이 **같은 (혹시 종횡비-왜곡된) 크롭**을 쓰므로 **crop-aspect에 confound되지 않는다** — 왜곡은 양쪽에 동일하게 작용하고, angle→ADD 해리는 §21.2의 솔버 washout(기계적 측정)로 설명되지 확인 오차가 아니다.
- **(c) 결론**: "co-finetune 반증이 confound/미커버라 synth 재실험이 정당"이라는 조건은 **성립하지 않는다.** 공유 백본 unfreeze는 refuted 정밀도-파괴 경로(배포 real 0.804 do-no-harm 위반이 **구성적으로 확정**)이고, clean 버전(P1b)은 이번 주 ADD 게이트에서 이미 탈락했다. **신규 co-finetune-on-synth 실험은 정당화되지 않는다.**

### 21.4 판정 — Panda-synth 격차는 **우리 아키텍처에서 닫히지 않는다**. CONCEDE + KUKA 재배정.

**왜 닫히지 않는가(인과).** 우리는 뒤지는 것이 backbone도 2D도 아니라 **각도 축**인데, 그 각도 축을 개선하는 모든 frozen-호환 레버가 소진·무효다:
1. head 아키텍처(mlp/mlp_patch/transformer/MoE/PARE/IEF) — 전부 ~9.1°/0.788 천장(§2, §6).
2. co-trained 각도 feature(P1b, 가장 강한 frozen-호환 이식) — angle 7.47°이나 **ADD 무효**(§21.2, 솔버 washout).
3. decoupled solve(freeze-θ) — 반증(naive 0.533 / flip-trigger +0.009), **near-oracle 각도**가 있어야 이득인데 우리가 도달 불가(P1b 7.47°로도 freeze는 손해 확정).

**근원은 backbone이 아니라 우리 detect-then-**공동solve** 아키텍처다.** 솔버가 θ를 2D 재투영과 공동 최적화 → feed-forward 각도가 washout되고 각도 정밀도가 노이즈 2D에 갇힘. RoboPEPP는 θ를 **회귀만**(JointNet IEF)하고 6DOF만 PnP → 각도가 2D-과적합에 오염 안 됨(§3, §7). **이 공동-solve는 우리가 real(depth가 binding, metric-FK PnP가 빛남 → 0.804 SOTA)과 Baxter(경쟁자 학습형 depth 붕괴 → 0.713 1위)에서 이기는 바로 그 메커니즘**이므로, synth를 위해 제거하면 우리 실제 차별점을 잃는다. **frozen 백본은 synth 격차의 원인이 아니라(2D는 이미 동급·우세, oracle-angle은 이미 RoboPEPP 초과) real 배포를 위한 의도된 설계**다.

**정직한 기대치.** 남은 유일한 live 레버 = **cropasp_a43 크롭-종횡비 수정**(이미 학습 중, 신규 GPU 불요). 이는 **2D 레버**이지 각도 레버가 아니다 → oracle-angle 천장(0.861/0.886)과 배포 base를 소폭 올릴 수 있으나 **RoboPEPP 격차(각도)는 못 닫는다**(GT-angle이면 이미 초과하므로). 예상: 배포 synth-DR가 0.769 → **0.77–0.79**(RoboPEPP 0.830 미달 유지). ⇒ **synth는 concede가 정직**하다: 우리는 real(0.804)+Baxter(0.713) SOTA를 보유하고, synth-DR은 경쟁자의 학습분포 홈그라운드이며, §17.4의 **이중 해리 서사**("키포인트는 frozen 범용 백본으로 충분 / 각도는 과제-feature가 필요하나 우리 공동-solve가 그 이득을 흡수")가 **negative를 논문 기여로 전환**한다.

**재배정 = KUKA(§13·§15).** KUKA는 헤드룸이 더 크고(−0.07~0.11) **아키텍처-호환 레버가 명확**하다: 각도 축이 아니라 **rot-head R + depth(RootNet식) 페어링 = RC/gauge 축**(§15.6: θ↔R gauge가 진짜 병목, §13.4: t·R 동등 기여, depth 단독 0.19→0.38이나 R까지여야 0.72). GPU EV가 각도 재시도보다 높다. do-no-harm: real 0.804·Baxter 0.713 불변(별개 로봇 가중치·별개 축).

**단 하나의 저비용 확인(선택, 오케스트레이터 판단)**: P1b head_theta로 **freeze-θ-solve-6DOF(P0a)** 를 good-frame에서 1회 eval(GPU1/GPU3, 솔버만). → **§21.5에서 실행 완료(오케스트레이터 지시).**

### 21.5 🔴 freeze-θ + solve-6DOF 결정 게이트 실행 (2026-07-22, GPU1 추론 전용) — RoboPEPP식 분리경로는 우리에게 **사망**

오케스트레이터 지시로 "RoboPEPP식 freeze-θ + 6DOF solve가 우리에게 生이 있나"를 세 각도-MAE 점에서 실측(`Eval/freeze_gate.sh`, 배포 crop 파이프라인 = rc_dumps_gf와 동일, 1000f panda_synth_test_dr, cov-PnP+DARK, rot-head R_init 사용):

| θ 고정원 (angle-MAE) | freeze-6DOF **CLEAN** (good, n=859) | freeze-6DOF FULL | 참고: joint-solve CLEAN |
|---|---|---|---|
| **oracle GT (0°)** | **0.8871** | 0.8611 | 0.899 (joint oracle) — 앵커 정합 |
| **P1b (7.47°)** | **0.5841** | 0.5633 | 0.7672 |
| **mlpctrl (9.1°)** | **0.6152** | 0.5859 | 0.7706 |
| — 배포 목표 | — | — | **0.7884** |
| — RoboPEPP(full) | — | **0.830** | — |

**판독 (결정적, 인과).**
1. **freeze@7.47°(0.584) ≪ joint-solve 0.788** — 우리 최선의 각도(P1b)로 freeze해도 good-frame ADD가 **−0.20 대폭 퇴화**. "P1이 P0를 잠금해제한다"(§6 가설)는 **반증**.
2. **freeze@7.47°(0.584) < freeze@9.1°(0.615)** — 평균 MAE가 더 좋은데 freeze-ADD는 더 나쁨 → **angle-MAE↔ADD 비단조**(freeze-ADD는 평균이 아니라 proximal 관절 오차구조에 좌우; 관측불가 손목이 평균 MAE를 지배). **P1b의 7.47° "돌파"가 ADD로 전이되지 않는다는 §21.2 결론이 freeze 모드에서도 재확인.**
3. **절벽: 0°→0.887, 7.47°→0.584** = −0.303/7.47° ≈ **−0.041/도**. 외삽(oracle→P1b 구간, 타깃 근방):
   - joint-solve 0.788에 **도달**하는 데 필요한 각도 = 0.887−0.788=0.099 → **≈2.4° MAE**(도달 불가 — 최선 frozen-호환이 7.47°).
   - RoboPEPP급 3.8°라도 freeze ≈ 0.887−0.156 = **~0.73 < joint 0.788 < RoboPEPP 0.830.** 즉 **RoboPEPP 수준의 각도를 얻어도 freeze-6DOF는 우리 현 joint-solve보다 나쁘다.**
4. **oracle freeze CLEAN 0.887 ≈ joint oracle 0.899** — GT 각도에서 freeze와 joint가 등가 → 커브 유효, 퇴화는 순수하게 각도오차의 freeze 하 전파임을 확인.

**메커니즘 (완결된 인과).** good 프레임은 2D가 정확(reproj ~1.4px)하므로 **joint-solve가 head의 coarse 각도(9.1°/7.47°)를 2D-정합 해로 정제**한다 — 이 정제가 **+0.15~0.19의 가치**(freeze 0.58–0.62 → joint 0.77). freeze는 이 정제를 버리고 coarse 각도를 FK로 전파 → 어떤 rigid R,t로도 보정 불가. ⇒ **(i)** feed-forward **각도 head는 good-frame ADD에 거의 무관**(솔버가 2D에서 각도를 재도출; §21.2의 washout = 솔버가 mlp·p1b를 같은 2D-정합 각도로 정제해서 ADD 동일이 근본 원인). **(ii)** binding constraint는 각도 feature가 아니라 **솔버의 2D→각도 추론, 2D 노이즈가 상한**(oracle-angle 0.899가 그 잔여 2D-노이즈 비용을 드러냄).

**판정 (오케스트레이터 CLOSE 기준 정확 충족).** freeze@7.47° ≪ 0.788 ✓ AND 커브가 요구하는 각도(joint 매칭 ~2.4°, RoboPEPP 초과엔 <1.4°)가 **도달 불가** ✓, 나아가 **RoboPEPP급 3.8°조차 freeze는 joint보다 나쁨** ✓ ⇒ **synth 각도 축 공식 종결(CLOSED).** joint 모드는 better 각도를 washout하고, freeze 모드는 어떤 도달가능 각도에서도 joint보다 나쁘다 — **두 solve 모드 모두 각도-head 레버를 거부.** 각도-head/feature/solve-mode로는 synth 격차가 닫히지 않음이 실측 확정. **concede airtight, KUKA 재배정 확정.** (재현: `Eval/freeze_gate.sh <gpu>`, 로그 `Eval/ablation_logs/freeze_gate/`, dump `Eval/rc_dumps_gf/freeze_{mlp,p1b,oracle}.npz`.)

### 21.6 🔴 freeze 커브 진짜 형태 실측 — **CONVEX 아님(CONCAVE)**, 각도 목표는 도달 불가 (2026-07-22, GPU1+GPU3, 8점 2-seed)

§21.5 판정이 2점 선형 외삽에 의존한다는 반박(사용자)에 답해, **GT 각도 + 보정 가우시안 노이즈**(per-batch 재정규화로 실현 wrapped-abs MAE = 타깃)로 freeze-6DOF ADD-AUC를 8개 각도-MAE에서 실측(`Eval/freeze_curve.sh`, selfbbox_eval `--oracle-angle-noise-mae` 신규 플래그, 배포 파이프라인 동일, seed0 완주 + seed1 저-MAE 4점 → 평균 Δ<0.007로 안정):

| angle-MAE (실현) | freeze **CLEAN**(good) | freeze FULL | median mm |
|---|---|---|---|
| **0.0°** | **0.887** (=oracle 앵커 ✓) | 0.861 (=ceiling.tsv 0.861 ✓) | 7.1 |
| 1.5° | 0.808 | 0.787 | 13.5 |
| 2.4° | 0.737 | 0.719 | 19.4 |
| 3.0° | 0.689 | 0.674 | 22.9 |
| **3.8°** (RoboPEPP급) | **0.631** | 0.615 | 27.7 |
| 5.0° | 0.548 | 0.534 | 35.8 |
| 6.0° | 0.494 | 0.480 | 42.0 |
| 7.47° | 0.422 | 0.409 | 51.8 |

**커브 형태 = CONCAVE(concede 쪽).** 선형(0°→7.47°) 예측 3.8°=0.650 vs **실측 0.631**(선형보다 낮음) → **CONVEX 반증**. iid-노이즈 교차점: **joint-solve 0.788 = MAE 1.78°**, **RoboPEPP 0.830 = MAE 1.12°**. 즉 순수 iid 모델로도 우리 현 pipeline을 넘으려면 ~1.8°, RoboPEPP를 넘으려면 ~1.1°가 필요.

**정직한 보정 — 실제 head는 iid보다 낫다(그러나 여전히 부족).** iid 노이즈는 전 관절 균등이라 비관적이다. 실측 per-joint MAE(good, GT `sim_state.joints` 대조):
- **mlp**: J1–6 = 8.1/4.7/7.5/5.2/12.4/10.8 → MEAN 8.11, PROX(J1-4) **6.36**, WRIST(J5-6) 11.59 → freeze **0.615** (= iid-**3.9°** 등가).
- **p1b**: J1–6 = 6.6/4.5/7.3/5.8/9.7/8.7 → MEAN 7.07, PROX **6.03**, WRIST 9.17 → freeze **0.584** (= iid-**4.3°** 등가).
- 실제 head는 오차를 **ADD-benign 손목(J5/J6)에 집중** → 같은 mean에서 iid보다 유리(등가 iid ≈ mean의 0.5×). **그러나 mean도 proximal도 freeze-ADD를 예측 못 함**: p1b가 mean·prox 둘 다 mlp보다 낮은데 freeze-ADD는 **더 나쁨**(0.584<0.615). ⇒ **angle-MAE↔freeze-ADD 관계는 비단조·구조의존**(§21.5-2 재확인). MAE 타깃으로 ADD를 겨냥하는 것 자체가 ill-posed.
- **RoboPEPP head를 우리 freeze에 넣으면?** RoboPEPP Tab.3 DR: mean 3.73/prox 3.03/wrist 5.15. iid@mean=**0.636**, iid@prox=**0.687**, 관대한 등가(0.5×mean≈1.9°)=**~0.77**. ⇒ **RoboPEPP급 각도라도 우리 freeze-6DOF는 [0.64, 0.77]** — **현 joint-solve 0.788을 겨우 매치(초과 못 함), 0.830엔 한참 못 미침.**

**결론(사용자 반박에 대한 결정적 답).**
1. **커브는 CONVEX가 아니라 CONCAVE** — freeze@3.8°=0.631(iid)~0.77(관대). **0.83 도달 불가.** (오케스트레이터의 "0.73 ⇒ concede" 임계보다도 낮음.)
2. **freeze-6DOF가 joint-solve 0.788을 넘는 교차점 = 실제 head 기준 mean ~3–3.5°(prox ~3°) = RoboPEPP급**, 그마저도 **매치 수준**(0.77–0.78)이고 **0.830은 sub-RoboPEPP(<1.5° iid) 필요 = 도달 불가.** oracle(0°)조차 FULL 0.861로 0.830을 겨우 넘음.
3. **"아키텍처가 막나 vs 덜 시도했나"**: 우리 최선의 **분리 각도망 P1b(23.5M ResNet50, ImageNet-init, FK loss = HoRoPose 레시피)** 가 mean **7.07°/prox 6.03°** 에서 멈춤 — RoboPEPP 3.73°/3.03°의 **약 2배**. 분리망은 **키포인트 백본 무손상 조건에서 ~7°가 천장**으로 보이며(§17), 3° 도달은 RoboPEPP식 **공유 백본 co-train(masked-pretrain+end-to-end)** 을 요구 = **real-2D 파괴 REFUTED 경로**(real 0.804·Baxter 0.713 몰수). 설령 3° head를 얻어도 (2)에 의해 freeze는 0.77–0.78(매치)에 그침.
4. ⇒ **두 게이트 모두 실패**: (a) freeze 커브가 concave라 도달가능 각도에서 0.83 불가, (b) 도달가능 분리-head(7°)는 필요 각도(3°)의 2배. **synth 각도 축 CLOSED 재확인, concede airtight.** RoboPEPP의 0.830은 freeze-decouple 자체가 아니라 **자기 co-designed pipeline(confident-kp BPnP + masked-pretrain feature)** 에서 나오며, 우리는 그 pipeline을 refuted 경로 없이 이식 불가. (재현: `Eval/freeze_curve.sh <gpu> [seed]`, 로그 `Eval/ablation_logs/freeze_curve/`, dump `rc_dumps_gf/fc_mae*_s*.npz`. selfbbox_eval `--oracle-angle-noise-mae`/`--oracle-noise-seed` 신규, default 0=무영향.)
