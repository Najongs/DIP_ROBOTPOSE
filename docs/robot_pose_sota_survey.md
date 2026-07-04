# DREAM 벤치마크 SOTA 서베이 + 가림-강건 자세추론 아이디어 (2026-07-03)

> 목적: DINObotPose3의 SOTA 초월 로드맵에 필요한 (1) DREAM 벤치마크 경쟁 방법 정리, (2) 휴먼 포즈/월드모델 문헌에서 **가림·불가시 상황에서의 대략적 자세 추론** 아이디어 발굴.
> 모든 아이디어는 `3_pose_models/DINObotPose3/SUMMARY.md`의 REFUTED 목록과 교차검증됨.
> 제약: **이미지 백본 적응 금지** (SSL/co-finetune 3회 반증 — 솔버가 요구하는 sub-pixel 키포인트 정밀도 파괴. V-JEPA 2.1 논문이 같은 결론을 독립적으로 확인: masked-latent 특징은 "국소 공간 구조가 파편적").

---

## 1. DREAM 벤치마크 SOTA 계보 (A1 검증 완료, 2026-07)

### ⚠️ 프로토콜 3축 — 이게 맞지 않으면 수치 비교 무효
1. **관절각 known vs predicted**: DREAM/CtRNet/SGTAPose/GISR/MonoSE(3)=known(쉬움), RoboPose(unknown)/HoRoPose/RoboPEPP/RoboTAG/우리=predicted(어려움)
2. **bbox GT vs 자동**: panda-orb는 카메라 배치가 다양해 자동 검출이 붕괴 — **RoboPEPP orb 77.5는 GT-bbox, 자동 bbox면 ~34**. **우리 bbox-from-solved는 완전 자동** → 우리 비교가 더 엄격한(불리한) 조건.
3. **학습 데이터**: 순수 sim-to-real vs real self-train vs real 라벨 학습

| 방법 | 연도 | 계열 | azure | kinect360 | realsense | orb | mean | 관절각 | 비고 |
|---|---|---|---|---|---|---|---|---|---|
| DREAM (R101-H) | ICRA'20 | keypoint+PnP | 68.9 | 24.4 | 76.1 | 61.9 | 57.8 | known | 가림 무대응 |
| RoboPose | CVPR'21 | render&compare | 70.4 | 77.6 | 74.3 | 70.4 | 73.2 | predicted | init bbox 의존 (auto orb 32.7) |
| SGTAPose | CVPR'23 | temporal | 67.8 | 2.1 | 87.6 | 72.3 | 57.5 | known | kinect 붕괴 |
| CtRNet | CVPR'23 | self-sup kp+seg | 89.9 | 79.5 | 90.8 | 85.3 | 86.4 | **known** | real self-train — 비교 불가(쉬운 축 2개) |
| CtRNet-X | '24 | +VLM 가시링크 선택 | — | — | — | — | ~86.2 | **known** | out-of-view 처리 최고 수준 |
| HoRoPose | ECCV'24 | RootNet식 depth | 82.2 | 76.0 | 75.2 | 75.2 | 77.2 | predicted | root-depth로 orb 강건 |
| GISR | RA-L'24 | 실루엣 정제 | 80.6 | 73.9 | 79.3 | — | — | predicted | orb는 seg 파인튜닝 |
| **RoboPEPP** | CVPR'25 | masking-pretrain+PnP | **75.3** | **78.5** | **80.5** | **77.5** | **77.9** | predicted | **GT-bbox 헤드라인** (auto orb≈34); 가림 최강(40% 가림 AUC 35.1) |
| RoboTAG | '25-11 | topological align graph | 83.1 | 75.7 | 78.3 | 58.8 | **74.0** | predicted | **auto-bbox(우리와 동일)**; orb 58.8=auto 붕괴. 검증완료 arXiv:2511.07717 v2 |
| ~~PoseDiff~~ | '25-09 | diffusion E2E | ~~96.4~~ | ~~94.8~~ | ~~96.6~~ | ~~96.5~~ | ~~96.1~~ | — | 🔴 **2025-10-30 저자 철회** — 철회 사유 원문: "The experimental setup and metrics lacks rigor, affecting the fairness of the comparisons". real 80/20 in-domain 학습이었음. **수치 전면 무시** |
| **Ours** | '26-07-04 | kp+solver+RC+**DARK** | **79.2** | **81.3** | **82.1** | **77.1** | **79.9** | predicted | **완전 자동 bbox**; +DARK decode+cov-PnP(무료); rs/kinect/orb anti-leak held-out |

**결론 (2026-07-04 갱신): mean 0.799로 RoboPEPP 0.780 초월** (azure/kinect/realsense 3승, orb −0.004 근접). DARK decode(무료)가 orb 격차 −0.010→−0.004. PoseDiff는 **저자가 직접 철회**(2025-10-30, "실험 설정·지표의 엄밀성 부족으로 비교 공정성 훼손" — arXiv:2509.24591 v2 확인) — 수치 무효. CtRNet 계열 0.86은 known-angle+real self-train.

**RoboTAG(2025-11, 검증완료) 직접 비교 — 동일 auto-bbox 프로토콜**: RoboTAG 4-split mean **74.0** vs 우리 **79.9**. 우리가 kinect(+5.6)/realsense(+3.8)/**orb(+18.3)** 승, azure(−3.9)만 패. orb 압승은 우리 bbox-from-solved가 RoboTAG/RoboPEPP를 침몰시키는 auto-detection 붕괴(orb 58.8/34.4)를 해결하기 때문. **RoboTAG가 인용한 RoboPEPP auto-bbox mean=74.0** → 동일 프로토콜에서 우리는 RoboPEPP +5.9(GT-bbox 비교 +1.9보다 큼). RoboTAG의 유일한 우위 azure 83.1: closed-loop 2D-3D 일관성 + depth regulator λ(학습 시 `ℒalign=α₁‖p³−p²‖²+α₂‖κ₃−κ₂‖²+α₃‖κ₃−κ_fk‖²`) — 우리 약점(근거리 azure, RC off)의 학습 시 처방. → 로드맵 후보(depth-consistency 학습 항).

### 가림 대응 메커니즘 비교 (핵심만)
- **RoboPEPP**: 관절 영역 마스킹 + embedding-predictive 사전학습 + conf 필터 PnP → 40% 가림 AUC 35.1 (RoboPose 14.5). 단 **from-scratch co-train 인코더** 전제 — frozen-DINOv3 이식은 우리 반증과 동일 계열이라 금지.
- **CtRNet-X**: CLIP+LoRA VLM으로 **가시 링크 판별 → 보이는 링크 키포인트만 선택** — out-of-view/절단 처리의 문헌 최고 수준. 우리 conf-gate의 상위 호환 개념.
- **HoRoPose**: root-relative + 직접 depth 회귀 → bbox-init 취약성 회피 (orb 강건성의 비결).
- **RoboPose/GISR**: 반복 정제(mesh/실루엣) — 가림-다봉성에서 wrong-basin 탈출 불가(우리 2026-06-06 분석과 일치).
- **우리 현재**: conf-gate 0.05 (가림 키포인트 96% 캐치) + self-train(가림 빈 최대 이득 Q1 +0.107) + nvdr render-compare(진행 중). 극한 가림 Q1 bin 0.628이 잔여 한계.

---

## 2. 사용자 질문에 대한 구조적 답: "가려져도 대략적으로 추론하려면?"

문헌 종합 결과, 가림 강건성은 세 층위로 분해되고 각 층위마다 우리 파이프라인의 접목 지점이 다름:

| 층위 | 질문 | 휴먼 포즈 해법 | 우리 접목 지점 |
|---|---|---|---|
| ① 증거 가중 | "보이는 것 중 뭘 믿을까" | visibility head, 캘리브레이션 확률(ProbPose), 공분산 가중 PnP | `solve_pose_kinematic.py` (conf-gate→연속 가중 업그레이드) |
| ② 상태 완성 | "안 보이는 관절을 어떻게 채울까" | masked joint modeling(MotionBERT), skeleton completion, diffusion prior(DPoser/GFPose) | solver의 `prior_w` 항 (현재 0.0!) + FK 매니폴드 |
| ③ 시간 문맥 | "직전 프레임 정보를 쓸까" | HuMoR 전이 prior, STRIDE 테스트타임 적응 | DREAM real은 연속 시퀀스 — 단 single-frame 프로토콜과 공정성 구분 필요 |

**월드모델 판정 (A3)**: 기하·자기가림·가시성은 우리 분석적 월드모델(mesh+FK+렌더러)이 이미 정확·캘리브레이션됨 — 학습형이 이길 수 없는 영역. 학습형 월드모델이 이기는 곳은 (a) 시간 문맥을 통한 가림 통과 예측, (b) 구성(configuration) prior, (c) 외부 가림체에 대한 amodal 마스크 — 즉 **"prior/완성 모듈"로만 유효, 기하 엔진으로는 무효**. V-JEPA류 백본 활용은 기각.

---

## 3. 아이디어 카탈로그 (REFUTED 교차검증 포함)

각 항목: ①메커니즘 ②접목 지점 ③반증 교차검증 ④1-day probe 설계

### 3.1 공분산 가중 로버스트 PnP/재투영 (A2 랭킹 1위) — 위험 최소
- ① 키포인트를 버리는(gate) 대신 예측 불확실도(공분산)로 연속 가중, Mahalanobis 거리 + robust kernel로 재투영 최소화 (CEPPnP/EPnPU 계열).
- ② `solve_pose_kinematic.py::solve_batch` — 이미 conf 가중 + IRLS 존재. 업그레이드: 등방 conf → **이방성 공분산**(히트맵 2차 모멘트에서 공짜로 추출) + 캘리브레이션.
- ③ 반증 교차: conf-gate는 이미 배포(+0.018 azure). **연속·이방성 가중은 미시도** — gate의 자연 확장이라 저위험.
- ④ probe: 히트맵에서 공분산 추출 → solve_batch 가중 교체 → `decompose_occlusion.py`로 가림 빈별 AUC 비교. 재학습 없음, 반나절.

### 3.2 가시성 헤드 + amodal 키포인트 (A2 랭킹 2위)
- ① 히트맵 헤드에 per-joint visibility 분기 추가 — 가려진 관절도 문맥으로 위치는 찍되(amodal) 가시성 플래그로 다운스트림 가중. **로봇은 자기-가림 GT를 렌더로 공짜 생성 가능**.
- ② `model.py::ViTKeypointHead`에 1개 분기 추가, `dataset.py`에 렌더 기반 visibility 라벨.
- ③ 반증 교차: "occlusion-aware detector retraining NOT warranted"(2026-06-03)와 구분 필요 — 그 결론은 *가림 증강 재학습*이 불필요하다는 것(conf가 이미 96% 캐치). visibility 헤드는 **새 출력**이며 confident-wrong 4-9%를 잡는 것이 목적. 단 기대 EV는 그 4-9% 잔여분이므로 **중간 이하** — 3.1보다 후순위.
- ④ probe: 우선 3.1의 공분산이 confident-wrong을 얼마나 잡는지 본 후 결정.

### 3.3 관절각 masked-state prior (A3 랭킹 1위, "월드모델"의 실용형)
- ① MTM(ICLR'23)/MotionBERT식: 작은 트랜스포머를 관절각 시퀀스/구성에 masked 사전학습 — **FK로 데이터 무한 생성, 라벨 불필요, 백본 무접촉**. 테스트타임에 (a) 솔버 출력 완성/디노이즈, 또는 (b) DPoser식 one-step-denoiser 항으로 solver 정규화.
- ② `solve_batch`의 `prior_w` 항(현재 0.0) — DPoser 공식: `w_t·||θ − sg[θ̂_denoised]||²`, 스텝당 네트워크 1회 평가로 저렴.
- ③ 반증 교차: **MCL 반증과 구분** — MCL은 멀티가설+selector(selector가 병목). 이건 단일 가설 정규화/완성이라 selector 불필요. **anchor/mean-fallback 반증과도 구분** — 그건 오염된 init에 앵커링; 이건 학습된 구성 분포로의 pull. diffusion head 반증(MLP에 패배)과도 구분 — 그건 각도 *예측기*; 이건 *prior*.
- ④ probe: DREAM synth 관절각 분포로 작은 denoiser 학습(1-2h) → prior_w 스윕 → 가림 빈별 AUC. 1일.
- ⚠️ 한계 정직하게: DREAM synth는 무작위 구성이라 "그럴듯한 구성 분포"의 정보량이 적을 수 있음(joint-limit 균등에 가까우면 prior가 무정보). **실로봇 궤적 분포**(우리 통제 데이터)에서 진가 — DREAM 벤치마크용으로는 EV 불확실.

### 3.4 render-compare의 가림-로버스트 실루엣 (Phase 4와 직결, 진행 중 트랙의 보강)
- ① SMPLify 계열의 part-visibility 가중: 실루엣/IoU 손실에서 **가려진 영역을 제외**(로버스트 마스킹). 우리는 per-link 가시성을 렌더로 정확히 계산 가능.
- ② `rc_refine_from_dump.py`의 soft-IoU 항 — SAM 마스크와 렌더의 불일치 영역(외부 가림체 후보)을 손실에서 다운웨이트.
- ③ 반증 교차: 없음(신규). 현재 do-no-harm 게이트/uv-shift 원복은 프레임 단위 — 이것은 픽셀 단위 세분화.
- ④ probe: SAM-렌더 불일치 마스크로 IoU 가중 → orb/realsense 200프레임 A/B. 반나절.

### 3.5 amodal 로봇 마스크 (A3 랭킹 2위, 조건부)
- ① SAMEO/ViTA-Seg로 외부 가림체 너머의 로봇 마스크 완성 → render-compare 타깃 개선.
- ③ 반증 교차: 자기-가림은 렌더러가 이미 해석적 처리 — **DREAM에는 외부 가림체가 거의 없음**(off-frame이 주 가림, 2026-06-03 분석). → **DREAM용 EV 낮음, 커스텀/인서션 데이터용으로 보류**.

### 3.6 시간 prior (별도 트랙 — 벤치마크 프로토콜 주의)
- ① HuMoR식 전이 prior 또는 단순 속도 제한(관절은 순간이동 불가) — 로봇은 정확한 동역학 한계를 앎.
- ③ 반증 교차: 없음(미시도). 단 기존 SOTA는 전부 single-frame — temporal 사용 시 별도 표기 필수(공정성).
- ④ probe: held-out 시퀀스에서 인접 프레임 융합 → 가림 프레임 AUC. DREAM 벤치마크 수치로는 주장하지 않고 "video 세팅" 별도 행으로.

### 기각 (재확인)
- V-JEPA/월드모델 백본 (3회 반증 + V-JEPA 2.1이 독립 확인), RoboPEPP식 사전학습의 frozen-백본 이식, world-model inversion (경쟁력 있는 선행 없음), MCL 재시도, GFPose식 멀티가설 샘플링(로봇 운동학은 준결정적).

---

## 4. 실행 랭킹 → 실험 결과 (2026-07-03 가림 벤치 판정 완료)

| 아이디어 | 판정 (RoboPEPP 프로토콜 벤치 @20% ablation) |
|---|---|
| 3.1 공분산 가중 PnP | ✅ **채택** — 전 구간 do-no-harm, +0.011@20% (`selfbbox_eval --cov-pnp`) |
| 3.4 가림-로버스트 실루엣 | ❌ **반증(현 설계)** — render∧¬SAM 다운웨이트가 depth 편향(−0.019). 구제는 명시적 가림체 세그멘테이션 필요 |
| 3.3 masked-state prior | ❌ **반증** — 모집단 평균 prior는 0.005에서도 −0.09 (구조적: 정답과 싸움). 학습형은 사전체크(관절 독립)로 생략 |
| 3.2 가시성 헤드 | 보류 — cov-pnp가 같은 신호를 커버, 잔여 EV 낮음 |
| 3.6 시간 prior | 미착수 (벤치마크 외 트랙) |

**핵심 발견**: 기존 스택(conf-gate 솔버 + rot-adapt + 정밀렌더 RC)이 이미 RoboPEPP급 가림 강건성 보유 —
가림 곡선 0.775/0.726/**0.626**/**0.525**/0.328 vs RoboPEPP 0.795/0.730/0.600/0.470/0.351 (**20-30% 승**,
열화 기울기 동일). RC는 가림 하에서도 유효(+0.06@10-20%). 상세: EXPERIMENTS.md 2026-07-03 occlusion track.

---

## 5. 출처

### A2 (휴먼 포즈 가림)
- ProbPose (CVPR'25) arxiv.org/abs/2412.02254 · PARE (ICCV'21) arxiv.org/abs/2104.08527 · Divide-and-Fuse (ECCV'24) arxiv.org/html/2407.09694
- DPoser arxiv.org/abs/2312.05541 · GFPose (CVPR'23) arxiv.org/abs/2212.08641 · VPoser/SMPLify-X arxiv.org/abs/1904.05866 · Pose-NDF arxiv.org/abs/2207.13807 · NRDF (CVPR'24) arxiv.org/abs/2403.03122 · ZeDO arxiv.org/abs/2307.03833
- MotionBERT (ICCV'23) arxiv.org/abs/2210.06551 · PoseBERT (TPAMI'22) arxiv.org/abs/2208.10211 · H3WB arxiv.org/pdf/2211.15692
- HuMoR (ICCV'21) arxiv.org/abs/2105.04668 · STRIDE (WACV'25) arxiv.org/abs/2312.16221 · 가림 벤치마크 arxiv.org/html/2504.10350v2
- 합성 가림 증강 github.com/isarandi/synthetic-occlusion · Limb Joint Aug arxiv.org/html/2410.09885v1

### A3 (월드모델)
- V-JEPA 2 arxiv.org/pdf/2506.09985 · V-JEPA 2.1 (dense) arxiv.org/html/2603.14482v1 — 국소 기하 부정확 인정
- MTM (ICLR'23) arxiv.org/abs/2305.02968 · Dual-Masked AE arxiv.org/pdf/2207.07381 · 가림 모션 prior arxiv.org/pdf/2207.05375
- SAMEO arxiv.org/html/2503.06261v1 · ViTA-Seg arxiv.org/pdf/2512.09510 · Amodal3R (옥스퍼드)
- RoboPEPP arxiv.org/abs/2411.17662 · PoseDiff arxiv.org/abs/2509.24591 (**검증 필요**) · DreamGen arxiv.org/abs/2505.12705

### A1 (로봇 포즈 SOTA) — 완료 시 기입

---

## 6. 2라운드 아이디어 발굴 (2026-07-04, post-SOTA)

SOTA 달성 후 남은 격차 4종에 대한 신규 문헌 탐색. done/refuted 목록으로 필터링, head/decoder 또는 test-time 레벨만.

**핵심 통찰**: 계획했던 "photometric RGB RC"(로드맵 ③)는 틀린 버전 — 최신 흐름은 **DINO feature-metric RC**로, 렌더-실사 비교를 픽셀이 아닌 **frozen ViT 특징 공간**에서 수행 → albedo/조명 도메인 갭을 특징이 흡수. 우리는 frozen DINOv3 + nvdiffrast를 이미 보유하므로 **두 조각을 붙이면 끝, 학습 불필요**.

| 순위 | 아이디어 | 공격 격차 | 비용 | 근거 |
|---|---|---|---|---|
| 1 | **DINO feature-metric RC** (MCLoc, AlignPose) | azure/근거리(d), 부분 (a)(c) | 낮음(재사용) | 렌더/실사를 DINO 특징 dense-map L2로 정합. MCLoc: 렌더 도메인 차이는 절대값만 바꾸고 상대차 보존→basin 안정. AlignPose: DINOv2가 dense-SIFT 대비 +10.8 AR, 투명/무텍스처에서 최대 이득 |
| 2 | **Featuremetric sub-patch 키포인트 refine** (FoundPose) | orb far/small (a) −0.010 | 낮음(test-time) | 중간 DINO 레이어가 위치정보 최강(대칭/저텍스처에서 last layer보다↑). coarse patch(16px) 한계를 특징 재투영으로 sub-patch 보정 |
| 3 | dense 2D-3D correspondence head (SurfEmb) | 40% 가림(c), synth갭(b) | 중-고(새 head) | per-pixel 다수결 → 40% 가림에도 붕괴 안 함. **순수 synth 학습으로 BOP SOTA**(우리 synth 갭 직격). 단 새 head+solver 빌드 |
| 4 | 이방성 불확실도 head (CNF/RLE, CFRE) | (a)(c) 가중 | 낮음(head, 학습 중과 결합) | cov-PnP에 먹일 **캘리브레이션된 이방성 공분산**. 추론 비용 0(추론시 회귀망만) |
| 5 | DARK / 구조적 heatmap decode | (a), bimodal→(c) | ~0(DARK) | Taylor 재국소화 sub-pixel 보정, 저해상도 열화 작음. soft-argmax bimodal 퇴화(가림 시 발생) 제거 |

**Idea 1+2는 같은 프리미티브**(DINOv3 feature-metric alignment)를 RC(dense)와 키포인트(sparse sub-patch) 두 단계에 적용 → 특징 비교 모듈 1개 구현으로 둘 다. **단일 최고 EV 엔지니어링**.

**하지 말 것(신규 부정 증거)**: training-free VLM 포즈(GPT-5.1/Gemini-3 등 관절각 프롬프트) — 평균오차 0.42-0.66 rad, 스케일링으로도 안 닫힘(arXiv:2512.06017). Gaussian-splat photometric RC(6DOPE-GS) — photometric이라 azure 도메인갭 재수입, Idea 1에 지배됨.

출처: MCLoc arxiv.org/abs/2404.10438 · AlignPose arxiv.org/abs/2512.20538 · FoundPose arxiv.org/abs/2311.18809 · SurfEmb arxiv.org/abs/2111.13489 · CFRE arxiv.org/abs/2505.02287 · no-soft-argmax arxiv.org/abs/2508.14929 · DARK arxiv.org/abs/1910.06278
