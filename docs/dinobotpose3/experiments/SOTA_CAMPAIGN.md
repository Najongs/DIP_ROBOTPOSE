# SOTA_CAMPAIGN — competitor ADD-AUC scoreboard (2026-07-22)

> 목적: 우리 모델을 **실제 경쟁자**(내부 oracle가 아니라)와 나란히 놓아 캠페인 우선순위를 정한다.
> 지표: **ADD-AUC @100mm, Protocol A(전체 test set 평균 AUC of ADD 곡선 0–0.1 m)**. 모든 값 **0–1 정규화**(원표 ×100은 source 태그에 병기).
> ⚠️ **프로토콜 3축을 섞지 말 것**: (1) 관절각 known vs **predicted**(우리·RoboPose·RoboPEPP·RoboTAG·HoRoPose = predicted / DREAM·CtRNet = known), (2) bbox **auto** vs GT(HoRoPose = GT / 우리·RoboPEPP·RoboTAG = auto), (3) Protocol A vs B.
> 직접 비교 대상(apples-to-apples) = **predicted-joint + auto-bbox** 행: RoboPose · RoboPEPP · RoboTAG · HoRoPose\*(auto). 나머지는 참고행.

---

## §1. 캠페인 스코어보드 (ADD-AUC, 0–1)

굵게 = 직접 비교 가능 그룹(predicted+auto) 내 최고. 태그 표기: `[출처, bbox/각도 프로토콜]`. 각주 F1–F6 = §1 하단.

### 1) Panda synthetic **DR** (`panda_synth_test_dr`) — 🎯 PRIORITY

| 방법 | ADD-AUC | 출처 / 프로토콜 |
|---|---|---|
| **RoboPEPP** | **0.830** | [RoboPEPP T1, auto-bbox, predicted] ← 직접비교 최고 |
| RoboPose | 0.829 | [RoboPEPP T1, auto-bbox, predicted] |
| RoboTAG | 0.825 | [우리 논문 T1 인용 = RoboTAG arXiv, auto-bbox, predicted — F5 로컬 미검증] |
| HoRoPose\* (auto) | 0.414 | [RoboPEPP T1, **auto-bbox** — 붕괴] |
| ─ 참고: HoRoPose (GT) | 0.827 | [RoboPEPP T1, **GT-bbox** — 프로토콜 상이] |
| ─ 참고: DREAM-H / F / Q | 0.829 / 0.813 / 0.778 | [RoboPEPP T1, **known-angle** — 체제 상이] |
| CtRNet / CtRNet-X | n/a | known-angle·real self-train 전용, synth ADD-AUC 미보고 (F6) |
| RoboKeyGen | n/a | DREAM 미사용(자체 셋) (F6) |
| **Ours** | **0.704 base / 0.769 +RC** | [full test set, predicted+auto] |

→ **BEHIND**: +RC 0.769 vs RoboPEPP **0.830 → −0.061**. base 0.704 → −0.126.

### 2) Panda synthetic **Photo** (`panda_synth_test_photo`)

| 방법 | ADD-AUC | 출처 / 프로토콜 |
|---|---|---|
| **RoboTAG** | **0.843** | [우리 논문 T1 = RoboTAG arXiv, auto, predicted — F5 미검증] |
| RoboPEPP | 0.841 | [RoboPEPP T1, auto, predicted] |
| RoboPose | 0.797 | [RoboPEPP T1, auto, predicted] |
| HoRoPose\* (auto) | 0.407 | [RoboPEPP T1, auto — 붕괴] |
| ─ 참고: HoRoPose (GT) | 0.820 | [RoboPEPP T1, GT-bbox] |
| ─ 참고: DREAM-H / F / Q | 0.811 / 0.795 / 0.743 | [RoboPEPP T1, known-angle] |
| CtRNet / CtRNet-X / RoboKeyGen | n/a | (F6) |
| **Ours** | **0.738 base / 0.799 +RC** | [full test set, predicted+auto] |

→ **BEHIND**: +RC 0.799 vs RoboPEPP **0.841 → −0.042** (vs RoboTAG 0.843 → −0.044). base 0.738 → −0.10.

### 3) Panda **real** (azure=AK, kinect360=XK, realsense=RS, orb) — 카메라별 + 4-cam mean

| 방법 | AK/azure | XK/kinect | RS/realsense | ORB | mean(4) | 출처 / 프로토콜 |
|---|---|---|---|---|---|---|
| RoboPEPP | 0.753 | 0.785 | 0.805 | 0.775 | 0.780 | [RoboPEPP T1, auto, predicted] |
| RoboTAG | 0.831 | 0.757 | 0.783 | 0.775 | 0.787 | [우리 논문 T1 = RoboTAG arXiv; ORB user-corrected 0.588→0.775 — F5] |
| RoboPose | 0.704 | 0.776 | 0.743 | 0.704 | 0.732 | [RoboPEPP T1, auto, predicted] |
| HoRoPose\* (auto) | 0.667 | n/a | 0.491 | 0.516 | (붕괴) | [RoboPEPP T1, **auto-bbox**] |
| ─ 참고: HoRoPose (GT) | 0.822 | 0.760 | 0.752 | 0.752 | 0.772 | [RoboPEPP T1, **GT-bbox**] |
| ─ 참고: DREAM-H | 0.605 | 0.640 | 0.788 | 0.691 | 0.681 | [RoboPEPP T1, known-angle] |
| ─ 참고: CtRNet / CtRNet-X | — | — | — | — | 0.864 / 0.862 | [related_work, **known-angle + real self-train** — 비교불가] |
| **Ours (deployed, RC on)** | **0.795** | **0.828** | **0.815** | **0.778** | **0.804** | [held-out 뒤30%, 1000f/cam, predicted+auto] |

→ **AHEAD** (predicted+auto 최고): mean 0.804 vs RoboPEPP 0.780 (+0.024), RoboTAG 0.787 (+0.017). 4/4 카메라 RoboPEPP 초과. (F3: 프레임집합 비대칭 — ours=held-out 30%, 경쟁=full seq.)

### 4) KUKA synthetic **DR** (`kuka_synth_test_dr`)

| 방법 | ADD-AUC | 출처 / 프로토콜 |
|---|---|---|
| **RoboPose** | **0.802** | [RoboPEPP T1, auto, predicted] ← 직접비교 최고 |
| RoboPEPP | 0.762 | [RoboPEPP T1, auto, predicted] |
| RoboTAG | 0.750 | [우리 논문 T1 = RoboTAG arXiv — F5 미검증] |
| HoRoPose\* (auto) | 0.562 | [RoboPEPP T1, auto] |
| ─ 참고: HoRoPose (GT) | 0.751 | [RoboPEPP T1, GT-bbox] |
| ─ 참고: DREAM-H | 0.733 | [RoboPEPP T1, known-angle] (DREAM-F/Q = KUKA 미보고) |
| **Ours** | **0.690** (solver+true-K, RC 없음) | [full test set, predicted+auto] |

→ **BEHIND**: 0.690 vs RoboPEPP 0.762 (−0.072), vs RoboPose 0.802 (−0.112), vs RoboTAG 0.750 (−0.060). (KUKA Photo 참고: RoboTAG 0.766 / RoboPEPP 0.761 / RoboPose 0.732 / HoRoPose-GT 0.739 / HoRoPose\* 0.567 / DREAM-H 0.721.)

### 5) Baxter synthetic **DR** (`baxter_synth_test_dr`) — Photo set 없음(F4)

| 방법 | ADD-AUC | 출처 / 프로토콜 |
|---|---|---|
| **Ours** | **0.713** (solver+true-K, RC 없음) | [full test set 5982f, predicted+auto] ← 전 방법 초과 |
| RoboTAG | 0.588 | [우리 논문 T1 = RoboTAG arXiv — F5 미검증; HoRoPose-GT와 값 동일, 주의] |
| RoboPEPP | 0.344 | [RoboPEPP T1, auto, predicted] |
| RoboPose | 0.327 | [RoboPEPP T1, auto, predicted] |
| HoRoPose\* (auto) | 0.098 | [RoboPEPP T1, auto — 붕괴] |
| ─ 참고: HoRoPose (GT) | 0.588 | [RoboPEPP T1, GT-bbox] |
| ─ 참고: DREAM-Q | 0.755 | [RoboPEPP T1, known-angle] (DREAM-F/H = Baxter 미보고) |

→ **AHEAD**: 0.713 vs 직접비교 최고 RoboTAG/HoRoPose-GT 0.588 (**+0.125**), vs RoboPEPP 0.344 (+0.369). ⚠️ 우리 값은 **true-K(GT intrinsics)** 사용 — Baxter는 예측-각도+auto 계열이 전멸(0.33–0.34)하는 열이라 우위가 극단적. 측정 아티팩트 아님은 gap_reexamination §(2026-07-22 Baxter probe)에서 검증됨(full 5982f, 경쟁자와 동일 프레임집합).

---

## §3. 캠페인 판정 및 자원 재배정 결정 (2026-07-22 — Panda-synth 격차 재검토)

> 전체 인과 분석: [2026-07-22_gap_reexamination.md §21](2026-07-22_gap_reexamination.md). 요약: **Panda-synth −0.06은 닫히지 않는다(concede) · KUKA로 재배정.**

**판정 근거 (인과, 숫자 아님).**
1. **−0.06 격차는 ≈100% 각도 예측 축.** full-set oracle-angle(GTθ+pred-2D) = DR +RC **0.886** / Photo **0.897** — **둘 다 RoboPEPP(0.830/0.841)를 이미 +0.056 초과.** ⇒ 우리 2D 검출·bbox·depth·off-frame은 격차 원천이 **아니다**(GT-angle이면 이미 이김). 오케스트레이터의 주 용의자 "co-trained backbone의 broad 우위"는 **detection 축 반증**(synth 2D reproj 1.2–1.5px ≥ real) + oracle-angle 우위로 **기각**.
2. **각도 축의 유일 frozen-호환 레버(P1b co-trained ResNet feature)를 dump A/B로 실측 — angle-MAE 7.47°(<9.1° mlp)로 이기나 good-frame ADD-AUC 0.767 ≈ mlp 0.771(무효).** 원인 = 우리 솔버가 θ를 2D와 **공동 최적화**해 feed-forward 각도를 washout(head 6.70°차 → solved 3.59°차 → median ADD 14.7≈14.8mm). **각도 feature를 고쳐도 ADD 천장이 안 열린다**(신규 반증, gap §21.2).
3. **co-finetune 반증은 confound도 미커버도 아니다.** 2026-06-08 반증은 *real 적응*(realsense ADD)이라 synth 미적용이 정상이고, synth-DR co-train 질문은 이번 주 P1b로 **직접 측정돼 ADD 게이트 탈락**(crop-aspect에 confound 안 됨 — within-dump A/B). ⇒ **신규 co-finetune-on-synth 실험 정당화 안 됨.** 공유 백본 unfreeze는 refuted 정밀도-파괴 경로(real 0.804 do-no-harm 구성적 위반).
4. **🔴 freeze-θ+6DOF 결정 게이트 실행(2026-07-22, GPU1, gap §21.5)** — RoboPEPP식 분리경로도 **사망**. good-frame CLEAN: oracle(0°) **0.887** / P1b(7.47°) **0.584** / mlp(9.1°) **0.615** vs joint-solve 0.77–0.79. freeze@7.47° ≪ 0.788. **joint 모드는 better 각도를 washout, freeze 모드는 어떤 도달가능 각도에서도 joint보다 나쁨 → 두 solve 모드 모두 각도-head 레버 거부.**
4b. **🔴 freeze 커브 진짜 형태 실측(2026-07-22, GPU1+GPU3, 8점 2-seed, gap §21.6)** — 사용자 "convex면 closeable" 반박에 답. GT+보정노이즈로 실측: 0°→0.887, 1.5°→0.808, 2.4°→0.737, 3.0°→0.689, **3.8°→0.631**, 5°→0.548, 7.47°→0.422. **CONVEX 반증 = CONCAVE**(선형 예측 3.8°=0.650 > 실측 0.631). 교차점: joint 0.788=iid 1.78°, RoboPEPP 0.830=iid 1.12°. 실제 head는 오차를 ADD-benign 손목에 집중해 iid보다 낫지만 **여전히 부족**: **RoboPEPP급 각도(mean 3.7°/prox 3.0°)조차 우리 freeze에서 [0.64, 0.77]** — 0.788 겨우 매치, 0.830 불가. 최선 분리망 P1b=7°(RoboPEPP 3.7°의 2배); 3° 도달은 refuted 공유-백본 co-train 요구. **각도 축 CLOSED 확정.**
5. **근원 = frozen 백본이 아니라 detect-then-공동solve 아키텍처.** joint-solve가 good 프레임에서 head의 coarse 각도를 2D로 정제(freeze 0.58–0.62 → joint 0.77, +0.15~0.19)하는 것이 승리 메커니즘이자 real(0.804 SOTA)·Baxter(0.713 1위)의 핵심 → synth 위해 제거 불가. 각도-head는 good-frame ADD에 사실상 무관(솔버가 2D에서 각도 재도출); binding = 솔버의 2D→각도 추론(2D 노이즈 상한).

**결정.**
- **① Panda-synth: CONCEDE.** 남은 live 레버 `cropasp_a43`(2D, 학습 중, 신규 GPU 불요)는 각도 격차를 못 닫음 → 예상 배포 synth-DR 0.77–0.79(< 0.830 유지). 완주 후 oracle-angle 재측정만 하고 각도 재시도는 중단. 논문은 §17.4 **이중 해리 서사**로 negative를 기여로 전환.
- **② 자원 재배정: KUKA(§13·§15).** 헤드룸 더 크고(−0.07~0.11) 레버가 아키텍처-호환(각도 아님, **rot-head R + RootNet식 depth 페어링 = RC/gauge 축**, gap §15.6/§13.4). GPU EV가 각도 재시도보다 높음. do-no-harm: real 0.804·Baxter 0.713·Panda-synth 불변.
- **(선택) 각도 축 완전 종결용 저비용 확인 1건**: P1b θ로 freeze-θ-solve-6DOF good-frame eval(GPU1/GPU3, 솔버만) — 예측 < 0.788(EV 낮음, washout+naive-freeze 0.533 근거).

**미해결(별건, 위 판정과 무관)**: 논문 본문의 "auto-bbox" 서술 ↔ KUKA/Baxter 실제 GT-크롭 eval 불일치 → `selfbbox_eval.py` 재측정 필요(§19.1, 수치조작 아님).

---

### 각주 (출처·프로토콜 caveat)

- **F1 — RoboPEPP T1**: `/home/najo/NAS/DIP/RoboPEPP/assets/results_table.png` (= CVPR'25 논문 Table 1, arXiv:2411.17662). 원표 ×100. 열: Known-Angles / Known-BBox / Panda{Photo,DR,AK,XK,RS,ORB} / KUKA{Photo,DR} / Baxter{DR}. **RoboPose·RoboPEPP·HoRoPose\* 행 = Known-BBox=No(auto)**, HoRoPose(HPE) 행 = Known-BBox=Yes(GT), DREAM-F/Q/H = Known-Angles=Yes. 우리 논문 `PAPER_OVERLEAF.tex` L152–171 Table가 이 표를 그대로 인용(값 일치 확인). **이것이 본 문서의 Protocol-A 기준표.**
- **F2 — Protocol A vs B**: 본 표 전부 Protocol A(전체 test set, 실패 프레임 포함 AUC). CLAUDE.md/오케스트레이터가 경고한 "RoboPEPP self-table KUKA 83.0 / Baxter 75.3"은 **다른(더 관대한) 표**의 값으로 추정되며 본 Protocol-A 표와 섞지 말 것. 본 표의 RoboPEPP는 KUKA-DR 0.762, Baxter-DR 0.344 (오케스트레이터가 지정한 "KUKA 0.762"와 일치 → 본 표 = 비교 기준). ※주의: 본 표에서 "83.0"은 RoboPEPP **Panda-DR**, "75.3"은 RoboPEPP **Panda-AK**·DREAM-Q Baxter 값 — 경고 문구의 숫자는 열 혼동 가능성.
- **F3 — 프레임집합 비대칭(real만)**: 우리 real 수치는 anti-leak **held-out 뒤 30%**(1000f/cam), 경쟁자는 full sequence. synth(DR/Photo/KUKA/Baxter)는 **양쪽 다 full test set** → synth 비교가 real보다 깨끗함.
- **F4 — Baxter Photo 없음**: DREAM 데이터에 `baxter_synth_test_dr`만 존재(gap_reexamination L441, RoboTAG §4.1 확인). Baxter는 DR만.
- **F5 — RoboTAG 로컬 미검증**: RoboTAG(arXiv:2511.07717, 2025-11)는 **코드/PDF 로컬 부재**. 모든 RoboTAG 값은 우리 `PAPER_OVERLEAF.tex` L166 인용(RoboTAG가 RoboPEPP 프로토콜로 보고했다고 명시)에서 옴. real ORB는 0.588→**0.775로 user-corrected**(commit 0216a8e). synth Photo 0.843·AK 0.831·KUKA-Photo 0.766은 RoboTAG의 headline claim으로 보이나 **원논문 대조 전 headline 사용 자제**. 특히 **Baxter 0.588이 HoRoPose-GT 0.588과 정확히 일치** → 전사 오류 가능성, 재확인 필요.
- **F6 — n/a 방법**: CtRNet(CVPR'23)·CtRNet-X(ICRA'25)는 **known-angle + real self-train** 전용(DREAM-real mean 0.864/0.862) — synth ADD-AUC 미보고, predicted 체제와 비교불가. RoboKeyGen(ICRA'24)은 DREAM 미사용(자체 RealSense/Azure 셋). 셋 다 해당 셀 n/a.
- **우리 값 출처**: `docs/dinobotpose3/FINAL_MODEL.md`(real 배포), 오케스트레이터 지정(synth). Panda-synth base/+RC 구분 = RC(render-and-compare)는 Panda에만 적용. 논문 draft(L97)의 synth base 74.2/76.9는 다른 config; **+RC 배포값 0.769/0.799이 기준**.
