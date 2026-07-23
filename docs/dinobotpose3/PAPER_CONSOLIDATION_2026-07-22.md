# PAPER_CONSOLIDATION — 캠페인 진단의 논문-준비 통합 (2026-07-22)

> **목적.** 이번 캠페인에서 확정한 모든 결론을 하나의 정합적 문서로 통합하고, 각 결론이 기존 논문(`PAPER_OVERLEAF.tex`)의 어느 절에 들어가는지 매핑한다. 진행 중 두 수치가 도착하면 곧바로 반영할 수 있도록 `[PENDING]` 자리표시자를 명시한다.
> **범위.** 읽기+쓰기 전용 통합. 학습·평가·커밋 없음. 논문 재작성 아님(매핑 문서).
> **출처.** `experiments/2026-07-22_gap_reexamination.md`(§14 데이터 면책, §15 게이지, §16, §17, §19 Baxter, §20 원위 tail, §21.1–§21.5), `experiments/2026-07-22_kuka_swap_gate.md`, `experiments/SOTA_CAMPAIGN.md`, `Eval/failure_viz/panda_synth_{1,2,3}.png`, `FINAL_MODEL.md`.
> **문체.** 격식 문어체. 실행 문장에는 화살표·대시를 쓰지 않는다. 아래 §0 요약은 초록-유사이므로 경쟁자명·수치를 일반화한다.

---

## 0. 통합 서사 요약 (초록-유사, 일반화)

본 연구는 단안 로봇 포즈를 키포인트 검출 이후 계량 순기구학과 참(true) 내부 파라미터, 동결 범용 백본을 결합해 해석적으로 복원한다. 깊이를 학습으로 회귀하지 않고 기하로 푸는 이 경로는 스케일에 불변이며, 실측 배포와 크고 원거리이며 자주 부분 가려지는 로봇에서 학습형 깊이 방식들이 붕괴할 때 안정적으로 우위를 확보한다. 동일한 해석적 설계가 학습분포에 밀착한 합성 도메인에서는 공동학습형 방법에 소폭 뒤진다. 본 통합의 분석 기여는 그 격차를 데이터 면책, 2D 축 면책, 각도 축 국소화, 두 solve 모드 모두의 각도-레버 거부라는 측정 사슬로 엄밀히 분해하고, 격차의 원인이 동결 백본이 아니라 실측·다로봇 우위를 만드는 바로 그 공동-solve 구조임을 정량으로 규명한 데 있다. 다로봇 확장은 단일 알고리즘에 로봇별 가중치와 순기구학만 바꾸며, 남은 격차의 레버가 로봇마다 다른 축(한쪽은 각도 관측성 천장, 다른 쪽은 2D 대응 혼동)임을 실측으로 구분한다. 합성 결과는 승리가 아니라 기전이 규명된 양보(concede-with-mechanism)로 서술한다.

---

## 1. 최종 결과 스코어보드 (ADD-AUC ×100, Protocol A)

> 직접 비교군 = predicted-joint + auto-bbox. 값은 `SOTA_CAMPAIGN.md §1` 기준. 우리 수치의 출처·프로토콜 단서는 §4 체크리스트 참조. 프론티어 = 직접 비교군 내 최고.

| 로봇 · 도메인 | Ours | 직접비교 프론티어 | 판정 | 격차 축 (기전) |
|---|---|---|---|---|
| Panda real (4-cam mean) | **80.4** | RoboPEPP 78.0 · RoboTAG 78.7 | **AHEAD (SOTA)** | 계량-FK PnP 기하 depth |
| — AK / XK / RS / ORB | 79.5 / 82.8 / 81.5 / 77.8 | RoboPEPP 75.3/78.5/80.5/77.5 | 4/4 초과 | RS·ORB는 노이즈 내 동률 |
| Baxter synth-DR | **71.3** | RoboTAG 58.8 · (HoRoPose-GT 58.8) | **AHEAD** | 경쟁 학습형 depth 붕괴(H2) |
| Panda synth-DR | 70.4 base / 76.9 +RC | RoboPEPP 83.0 · RoboPose 82.9 · RoboTAG 82.5 | **BEHIND −6.1** | **각도 예측 축**(§21) |
| Panda synth-Photo | 73.8 base / 79.9 +RC | RoboTAG 84.3 · RoboPEPP 84.1 | **BEHIND −4.2** | 각도 예측 축 |
| KUKA synth-DR | 69.0 | RoboPose 80.2 · RoboPEPP 76.2 · RoboTAG 75.0 | **BEHIND −7.2** | **2D link-swap**(검출기 한정) |
| KUKA synth-Photo | 69.8 | RoboTAG 76.6 · RoboPEPP 76.1 | **BEHIND −6.3** | 동일(2D link-swap) |

**핵심 대조(우위의 기전).** Panda-DR 기준 로봇을 Baxter-DR로 옮길 때 우리 낙폭은 −2.9(74.2→71.3, solver 모드)인 반면 경쟁자는 −23.7(RoboTAG) ~ −50.2(RoboPose)로 붕괴한다. 우위는 우리가 급등해서가 아니라 학습형 깊이가 크고(팔 baseline 0.89m) 멀고(1.31m) 부분 off-frame인 구성에서 무너지기 때문이다(§19.3).

### 1.1 `[PENDING]` — 도착 대기 중인 두 수치

| # | 수치 | 대체 대상 | 현재 조기신호 | 측정 방법 |
|---|---|---|---|---|
| **P1** | **KUKA 재학습-검출기 ADD-AUC** (solver+true-K, 전체 5,997f) | 스코어보드 KUKA-DR 69.0 및 논문 Table 1 KUKA 열 | 2D val AUC 배포 0.735 → ep1 0.775 (PCK@10 80.2→85.5%, 파국-kp율 22.95→14.5%). ADD-AUC는 미도착 | RANSAC seed 고정(`cv2.setRNGSeed`) 또는 다중-seed 평균 (solver가 ±0.03 노이즈) |
| **P2** | **cropasp_a43 (crop 종횡비 수정) A/B** — Panda synth-DR base **및** real 4-cam mean | 스코어보드 Panda synth base 70.4/73.8; do-no-harm 검증은 real mean 80.4 | 학습 중. 예상 synth-DR base 0.769 → **0.77–0.79**(각도 격차는 못 닫음, §21.4) | 완주 후 oracle-angle 재측정 + real 4-cam do-no-harm A/B |

> P2는 **2D 레버**이지 각도 레버가 아니다. GT-각도를 넣으면 이미 프론티어를 넘으므로(§2.2b) synth-DR의 RoboPEPP 격차(각도)는 P2로 닫히지 않는다. 예상 배포 synth-DR은 프론티어 미달 유지이며, 이는 concede 서사와 정합한다.

---

## 2. 논문-준비 분석 서사 (절 매핑 포함)

### 2.1 강점 — 학습 없는 스케일-불변 기하 depth (Method · Experiments)

제안 파이프라인은 키포인트에서 계량 순기구학과 참 내부 파라미터로 카메라 포즈를 PnP로 복원하며, 깊이를 학습형 회귀 없이 원근 foreshortening에서 직접 얻는다. 이 경로의 계량 정확성은 독립 검증되었다. 우리 순기구학은 DREAM 자체 기구학의 링크 원점을 Kabsch 잔차 평균 0.005mm(최대 0.013mm)로 재현하고, 모든 입력이 ground truth일 때(GT 각도 + GT 2D + PnP) ADD 평균 0.011mm, ADD-AUC 0.9998을 낸다(§14). 즉 기하 경로 자체에는 사실상 오차가 없다.

경험적 우위는 두 곳에서 나타난다. 첫째, Panda 실측 4카메라 평균 ADD-AUC 80.4로 동일 프로토콜(predicted-joint + 자동 bbox)의 최고 성능을 달성하며, 평균 마진 2.4는 실행-간 노이즈 추정치 약 1.0의 두 배를 넘고 네 카메라 모두에서 프론티어를 상회한다. 둘째, Baxter 합성에서 71.3으로 프론티어를 12.5점 앞선다. 이 두 번째 우위의 기전은 우리가 특별히 잘해서가 아니라 경쟁자가 붕괴하기 때문이다. Baxter는 세 로봇 중 가장 크고 멀며 좌팔만 평가되고 프레임의 약 30%에서 팔이 화면 밖으로 나간다. 경쟁자의 학습형 깊이(반복 렌더-비교, 부분 가시 키포인트 BPnP, 겉보기 크기 기반 RootNet 회귀)는 작고 가깝고 전신 가시인 로봇에 맞춰져 이 구성에서 out-of-distribution이 된다. 우리 경로는 넓은 강체 FK 모델을 참 K로 PnP하여 크기와 거리에 불변이므로 어디서나 약 0.7을 안정적으로 낸다.

동일 파이프라인이 세 로봇에서 로봇별 재설계 없이 end-to-end로 동작하며, 알고리즘은 하나이고 로봇마다 바뀌는 것은 가중치와 순기구학 값뿐이다. 이 "하나의 알고리즘, 로봇별 가중치"는 "로봇마다 다른 처방"보다 강한 주장이며, 세 로봇의 공통 실패 증상(신뢰도 축으로 거를 수 없는 대응 오류)이 그 주장을 뒷받침한다(§14.1).

> **매핑.** Method §Frozen Foundation Front-End·§Solver(계량 FK, true-K PnP), Experiments §Runtime and Generalization(다로봇), Introduction 기여 목록(기하 depth). Baxter 기전은 §2.3에서 상술하며 새 분석 문단으로 승격 권고.

### 2.2 한계/분석 — 합성 −0.06의 인과 분해 (신설 Analysis 절 · Limitations)

이 분해가 논문의 **신규 분석 기여**다. 합성 도메인 랜덤화에서 제안 방법은 프론티어에 약 0.06 뒤진다(배포 synth-DR +RC 0.769 대 프론티어 0.830, Photo 0.799 대 0.841). 이 격차를 다음 다섯 사슬로 규명한다.

**(a) 데이터 면책.** 데이터·FK가 상한을 누르는가라는 질문은 부정된다. FK와 GT의 3D-3D 정합 잔차는 0.005mm이고, 모든 입력이 GT일 때 ADD-AUC 0.9998이다. 데이터는 레버가 아니며 면책되었다(§14). 남은 격차는 예측 검출기와 예측 각도의 비용으로만 구성된다.

**(b) 격차는 사실상 100% 각도 예측 축이다(2D도 off-frame도 backbone-broad도 아니다).** GT 각도만 주입하고 2D는 예측으로 두면(oracle-angle) ADD-AUC는 DR +RC 0.886, Photo 0.897로, 둘 다 프론티어(0.830 / 0.841)를 이미 0.056 초과한다. 즉 GT 각도가 있으면 우리 2D와 솔버는 이미 프론티어를 이긴다. 따라서 2D 검출·bbox·깊이·off-frame은 격차의 원천이 아니다. 이는 검출 축에서 직접 재확인된다. 동결 백본의 good-frame 재투영 오차는 합성에서 1.2–1.5px로 실측(1.4–1.9px)과 동급이거나 더 정밀하므로, "동결 백본이 합성에 약하다"는 가설은 반증된다(§17.1). 남은 격차 안에는 크기가 거의 동등한 두 레버(2D 검출 약 −0.10, 각도 예측 약 −0.11)가 있으며, 완벽한 각도조차 2D 비용 때문에 0.899에서 천장에 부딪히므로 두 레버는 상보재이다(§14.2). 프론티어를 넘는 데 필요한 것은 각도 헤드룸 +0.117 중 +0.061뿐이다.

**(c) 각도 헤드는 사실상 무관하다(솔버가 각도를 2D에서 재도출).** 각도 축을 개선하는 유일한 동결-호환 레버는 별도 학습형 각도 feature(P1b, 키포인트 경로는 동결 유지이므로 반증된 "공유 백본 적응"과 구조적으로 다름)다. P1b는 각도 MAE 7.47°로 mlp 9.1°를 이기지만, good-frame ADD-AUC는 0.767로 mlp 0.771과 동일하고(Δ −0.004) median ADD도 14.7과 14.8mm로 비트 동일하다. 원인은 우리 솔버가 θ(7)·R(3)·t(3)를 재투영으로 공동 최적화하는 데 있다. feed-forward 각도가 6.70° 개선돼도 솔버가 절반을 washout(6.70°에서 3.59°로)하고 나머지는 noisy 2D에 흡수된다. 따라서 good-frame ADD의 binding constraint는 각도 feature가 아니라 솔버의 2D 재적합 천장이다(§21.2).

**(d) 분리(freeze-θ) 경로도 어떤 도달가능 각도에서도 열리지 않는다.** 각도를 고정하고 6DOF만 푸는 RoboPEPP식 경로를 세 각도-MAE 점에서 실측했다. good-frame CLEAN에서 oracle 각도(0°)는 0.887, P1b(7.47°)는 0.584, mlp(9.1°)는 0.615로, 모두 joint-solve 0.77–0.79에 못 미친다. 커브 기울기는 −0.041/도이며, joint-solve 0.788에 도달하려면 약 2.4° MAE가 필요한데 이는 도달 불가하고(최선 동결-호환이 7.47°), RoboPEPP급 3.8°를 얻어도 freeze는 약 0.73으로 여전히 joint보다 나쁘다(§21.5). 결론적으로 joint 모드는 더 나은 각도를 washout하고 freeze 모드는 어떤 도달가능 각도에서도 joint보다 나쁘므로, 두 solve 모드 모두 각도-헤드 레버를 거부한다. 합성 각도 축은 공식적으로 종결(CLOSED)된다.

**(e) 정량화된 설계 긴장: 동결(real)과 공동학습(synth)의 맞바꿈.** 격차를 닫으려면 공유 백본을 공동학습해야 하나(프론티어의 경로), 이는 솔버가 요구하는 서브픽셀 2D 정밀도를 내주어 실측 SOTA를 무너뜨린다. 이 맞바꿈은 정량 측정되었다. 공유 백본을 실측에 적응시킨 두 clean 변형은 realsense ADD를 0.567에서 0.531(SSL), 0.497에서 0.434(pseudo-keypoint)로 단조 악화시켰고, 백본을 무작위 초기화해 전학습하면 검출 real-val AUC가 0.455로 동결 DINOv3의 0.80에 크게 못 미친다(§21.2–§21.3). 근원은 동결 백본이 아니라 우리 detect-then-공동solve 구조다. 솔버가 head의 coarse 각도를 2D-정합 해로 정제하는 것(freeze 0.58–0.62에서 joint 0.77로, +0.15–0.19)이 승리 메커니즘이자 실측(0.804)·Baxter(0.713) 우위의 핵심이므로, 합성을 위해 제거하면 우리 차별점을 잃는다.

**(f) 잔여 실패 taxonomy(직접 이미지 검사 + 게이트).** 실패는 프레임 전반에 퍼지지 않고 소수 파국 프레임에 집중된다. Panda 합성 이미지 직접 검사(`Eval/failure_viz/`) 결과, 완전 가림은 약 0.5%로 예측 불가능하고 부분 가림은 21%다. 파국 꼬리는 검출 실패 약 61%와 솔버 발산 약 34%(2D는 맞으나 3D가 km 규모로 폭발하는 모드)로 나뉘며, 전형적 격차는 검출 절반과 clean 프레임의 각도 잔차 절반으로 갈린다. 원위 2D 꼬리는 argmax 오차 10px 초과가 6.5%인데, 정답 위치의 히트맵 응답이 median 0.074로 사실상 정답 모드가 부재하며(35%는 단일-강한-오답, 65%는 2차 강모드가 있으나 정답 링크가 아님), 복원가능성은 7–9%로 유의미한 이득(약 40%)에 크게 미달한다. 따라서 top-M 모드 열거 디코더 계열은 없는 모드를 선택할 수 없어 사망이다(§20). 이 원위 꼬리가 논문이 말하는 "신뢰도로 거를 수 없는 confident-wrong" 잔차의 실체다.

> **매핑.** 신설 Analysis 하위절 "Analysis of the Synthetic Gap"(또는 Experiments 말미 확장). 데이터 면책 사실은 Method(이미 "0.003 mm"로 인용)와 이 분석절을 함께 지지. (e)는 Introduction·Method §Frozen·Conclusion의 "적응이 서브픽셀 정밀도를 파괴" 주장을 합성 측 정량으로 보강. Refuted 표에 두 행 추가 권고(§3). taxonomy는 Conclusion 잔여-실패 문단을 정정·상세화(§3).

### 2.3 다로봇 마무리 — 로봇마다 다른 레버 (Experiments · Analysis)

다로봇 격차의 레버가 로봇마다 다른 축이라는 점이 분석의 두 번째 축이다.

**Baxter(우위, 기전 규명).** 우위는 우리가 급등해서가 아니라 경쟁자가 붕괴해서 발생한다. angular subtense(baseline/거리)는 셋 중 최소(0.724)여서 단순 기하 조건수 가설은 성립하지 않으며, 우리 Baxter(71.3)는 우리 Panda(74.2)보다 오히려 낮다. 살아남는 사실은 우리 기하 경로가 스케일에 불변이라 어디서나 약 0.7을 낸다는 것이고, 경쟁자의 학습형 깊이가 이 구성에서 −24에서 −50으로 붕괴하기 때문에 순위가 뒤집힌다(§19).

**KUKA(열위, 진행 중, 다른 레버).** KUKA 격차는 Panda의 각도 천장과 전혀 다른 축인 2D 대응 혼동이다. 원통형 링크의 시각적 동일성 때문에 검출기가 확산(diffuse)·저신뢰 히트맵을 내며, 전체 유효 키포인트의 22.95%가 파국이고 그중 85.2%가 정답 모드가 argmax에 부재한 참 swap이다. 결정적으로 이 파국 키포인트는 저신뢰(peak conf 0.04 대 good 0.65)여서 검출기가 자신의 실패를 안다(캘리브레이션됨). 즉 신호 자체가 약한 검출기-한정 문제이며, 솔버의 min_kp=6 바닥이 이 쓰레기 키포인트를 강제 사용해 파국을 유발한다(§kuka_swap_gate). 완벽 2D(oracle-2D)의 천장은 0.834로 프론티어 0.762를 크게 상회하므로 헤드룸은 충분하나, 그 완벽 2D에서도 일부 프레임이 mean 563m로 발산하는 2차 잔차(솔버 안정성, 결정론적 t_z 클램프 필요)가 남는다. 처방은 새 정보를 주는 검출기 재학습이며(추론-decode 레버 windowed/cov-PnP/conf-adaptive는 전부 net flat-or-harmful로 실측 반증), 조기신호는 2D val AUC 0.735에서 ep1 0.775다. 배포 ADD-AUC는 `[PENDING P1]`.

> **매핑.** Experiments §Runtime and Generalization(다로봇 문단). Baxter 기전은 새 분석 문단으로 승격. KUKA는 §3의 정정 항목(현행 "link-identity confusion / confident wrong" 서술을 검출기-한정 확산 + 솔버 발산으로 정정)과 결부.

---

## 3. 매핑 표 — 발견 → 논문 절 → 정정 사항

| # | 발견 (근거) | 들어갈 절 (`PAPER_OVERLEAF.tex`) | 성격 | 정정하는 기존 주장 |
|---|---|---|---|---|
| M1 | 계량 FK 0.005mm·GT-all 0.9998 (§14) | Method §Solver / 신설 Analysis | 보강 | (없음; 기존 "0.003 mm" 인용을 데이터 면책으로 확장) |
| M2 | synth 격차 ≈100% 각도 축, oracle-angle 0.886/0.897 > 프론티어 (§21.1) | 신설 Analysis "Synthetic Gap" | **신규 기여** | 현행 본문은 합성 열위를 "학습분포 불일치"로만 서술(L179). 인과(각도 축)로 대체 |
| M3 | 동결 DINOv3 2D는 합성에서 real과 동급·우세 1.2–1.5px (§17.1) | 신설 Analysis / Method §Frozen | 보강 | "합성 도메인 적응 필요" 프레이밍 금지(검출은 frozen으로 충분) |
| M4 | P1b: 각도 7.47° 승리, ADD 0.767 무효 (§21.2) | Refuted 표(신규 행) + Analysis | **신규 negative** | (신규) Refuted 표에 "co-trained angle feature: PCK-MAE 개선, ADD 평탄" 추가 |
| M5 | freeze-θ 6DOF: 도달가능 각도에서 joint보다 열위, 각도 축 CLOSED (§21.5) | Refuted 표(신규 행) + Analysis | **신규 negative** | (신규) "decoupled/freeze solve: worse than joint at any reachable angle" 추가 |
| M6 | frozen(real)↔co-trained(synth) 맞바꿈 정량화 (§21.2–§21.3) | Introduction·Method §Frozen·Conclusion | 보강 | 기존 "적응이 서브픽셀 정밀도 파괴" 주장에 합성 측 정량·설계 긴장 추가 |
| M7 | Baxter 우위 = 경쟁 학습형 depth 붕괴(H2), subtense 최소(H1 반증) (§19) | Experiments §Generalization(승격 문단) | 보강 | "spread가 넓어 우리에게 쉬움"류 단순 서술 금지(H1 반증) |
| M8 | KUKA 꼬리 = 검출기-한정 확산 저신뢰 + 솔버 발산, 재학습 레버 (§kuka_swap_gate) | Experiments §Generalization | **정정** | L245–246·L312 "link-identity confusion, confident wrong, not detectable from confidence" — KUKA에는 부정확(키포인트는 저신뢰=검출가능). 아래 상세 |
| M9 | Panda 원위 꼬리 = confidently-wrong(정답 모드 부재, GT-resp 0.074) (§20) | Conclusion 잔여-실패 문단 | 정정/분리 | 현행은 Panda·KUKA를 한 모드로 뭉침. 두 로봇의 기전이 다름을 분리 |
| M10 | 실패 taxonomy: 완전가림 0.5%/부분 21%, 파국 꼬리 검출 61%+솔버발산 34% (`failure_viz`) | 신설 Analysis / Conclusion | 보강 | (신규) 정성 서술을 정량 taxonomy로 |
| M11 | KUKA 재학습 ADD-AUC `[PENDING P1]`; 2D val 0.735→0.775 | Table 1 KUKA 열 + §Generalization | **수치 대기** | Table 1 KUKA 69.0/69.8은 재학습 완료 시 갱신 후보 |
| M12 | cropasp_a43 A/B `[PENDING P2]` (2D 레버, 각도 격차 미해결) | Table 1 Panda synth 열 | **수치 대기** | 아래 M13과 함께 처리 |

### 3.1 명시적 정정 필요 항목 (제출 전 반드시 반영)

- **C1 — Table 1 Ours Panda synth-DR/Photo = 74.2/76.9는 stale.** `SOTA_CAMPAIGN.md` 각주가 명시하듯 이 값은 배포와 다른 config다. 배포 기준은 base 70.4/73.8 또는 +RC 76.9/79.9다. Table이 어느 config를 보고하는지 확정하고 두 계열 중 하나로 통일할 것(74.2/76.9는 어느 쪽과도 일치하지 않음). Panda 실측에만 RC를 적용한다는 현행 caption과의 정합도 함께 점검.
- **C2 — KUKA 잔여-실패 서술(L245–246, L312) 정정.** 현행 "link-identity confusion으로 confident wrong pose를 낳으며 신뢰도만으로는 검출 불가"는 KUKA에 부정확하다. KUKA 파국 키포인트는 저신뢰 확산 히트맵(peak 0.04)이라 키포인트 수준에서는 신뢰도로 검출 가능하며, 문제는 솔버 min_kp 바닥이 이를 강제 사용하는 것과 완벽 2D에서도 남는 솔버 발산이다. Panda 원위 꼬리(정답 모드 부재, GT-응답 0.074)만이 진정한 confident-wrong이다. 두 로봇의 기전을 분리 서술할 것.
- **C3 — "fully automatic bounding boxes"(L137) 대 실제 GT-크롭 불일치.** KUKA·Baxter 수치는 `crop_to_robot=True`(GT projected keypoint bbox) solver eval에서 나왔다(§19.1). 본문은 완전 자동 bbox를 주장한다. 해소책은 `selfbbox_eval.py`(자동-bbox)로 KUKA·Baxter를 재측정해 본문과 수치를 일치시키는 것이며(본 통합에서 미실행, 수치 조작 금지), 그 전까지는 caption에 KUKA/Baxter가 GT-크롭 solver eval임을 명시.

---

## 4. 정직성·무결성 체크리스트 (제출 전 재확인)

| 항목 | 상태 | 필요 조치 |
|---|---|---|
| **Baxter/KUKA true-K(GT intrinsics) 사용** | 사실 | 합성에서 K는 데이터 제공값이라 관례적이나, "geometric depth = true K 의존"을 명시. real은 캘리브 K(공정). caption에 true-K 표기 |
| **KUKA/Baxter GT-크롭 bbox vs 본문 "자동 bbox"** | 불일치(C3) | `selfbbox_eval.py` 재측정 또는 caption caveat. GT-크롭 이점 상한은 Panda real 실측 +0.005, Baxter 보수 추정 +0.02–0.05(미측정) |
| **RoboTAG 수치 로컬 미검증(F5)** | 미검증 | 코드·PDF 로컬 부재. Baxter 0.588이 HoRoPose-GT 0.588과 정확 일치 = 전사 오류 의심. ORB는 0.588→0.775 user-corrected. 원논문 대조 전 headline(Photo 84.3 등) 사용 자제 |
| **real 수치 프레임집합 비대칭(F3)** | 사실 | 우리 real = held-out 뒤 30%(1000f/cam), 경쟁 = full seq. synth는 양쪽 full test set(더 깨끗). caption 명시 유지 |
| **실행-간 노이즈 ~1.0** | 사실 | real mean 마진 2.4 > 2×노이즈. RS·ORB 우위는 노이즈 내 동률로 정직 서술. self-train 기여 1.1도 노이즈급이라 보수 해석 |
| **CtRNet/CtRNet-X는 known-angle + real self-train** | 비교불가 | predicted 체제와 직접 비교 금지. related work에만, 별 체제로 명시 |
| **Panda synth base vs +RC vs stale 74.2/76.9** | 불일치(C1) | Table config 통일 |
| **oracle-angle 행(Table 1 italic, real 84.1)** | 사실 | GT-angle 주입 참조행. synth oracle-angle(0.886/0.897)은 분석절 근거이며 Table 본행과 구분 |
| **solver 발산(km-scale) 잔차** | 미해결(별건) | KUKA oracle-2D도 mean 563m 발산. do-no-harm 서술 시 "solver robust-init은 후속" 명시 |
| **P2(cropasp) do-no-harm** | 대기 | real 4-cam 0.804 불변 확인 후에만 배포 반영. 각도 격차 미해결임을 명시 |
| **HoRoPose\* 붕괴(auto-bbox)** | 사실 | HoRoPose(GT-box)와 HoRoPose\*(auto)를 혼동 금지. Table의 별표·caption 유지 |
| **"백본 최초/backbone-agnostic" 과장 금지** | 규칙 | frozen 기여는 승격하되 "동결이 배포-무관을 뜻하지 않음" 유지. SAM=v1 표기 |

---

## 부록 A. 우리 수치 출처·재현 포인터

- **real 배포(0.804)**: `docs/dinobotpose3/FINAL_MODEL.md`(held-out 1000f/cam, RC on, azure RC off). 배포 config의 유일 authority.
- **Panda synth base/+RC, oracle-angle(0.886/0.897)**: `Eval/ablation_logs/oracle_angle_synth/results.tsv`, `Eval/rc_dumps_gf/{mlpctrl,p1b_ep17}_gf.npz`.
- **freeze-θ 게이트**: `Eval/freeze_gate.sh`, 로그 `Eval/ablation_logs/freeze_gate/`, dump `Eval/rc_dumps_gf/freeze_{mlp,p1b,oracle}.npz`.
- **데이터 면책(Kabsch 0.005mm, GT-all 0.9998)**: `2026-07-22_gap_reexamination.md §14`(panda_synth_test_dr 앞 300f 재현).
- **KUKA swap 게이트·재학습**: `Eval/_debate_tmp/kuka_gate/`(게이트 `kuka_swap_gate.py`), 재학습 로그 `TRAIN/outputs_heatmap/kuka_detector_retrain_20260722_214107/`.
- **Baxter 공정성 감사·기전**: `2026-07-22_gap_reexamination.md §19`(`Eval/baxter_add_eval.py` solver 모드, true-K).
- **원위 tail 복원가능성**: `2026-07-22_gap_reexamination.md §20`(`Eval/_debate_tmp/recover_gate.py`).
- **실패 taxonomy 이미지**: `Eval/failure_viz/panda_synth_{1_catastrophic,2_typical,3_fully_occluded}.png`.
- **프론티어 수치 기준표**: `SOTA_CAMPAIGN.md §1`(= RoboPEPP Table 1, arXiv:2411.17662; 우리 논문 Table 1과 대조 완료).

## 부록 B. 반증되어 재제안 금지 (교차확인용)

공유 백본 적응(SSL·co-finetune 전 계열), mlp_patch/MCL/MoE 각도 재분배, conf-gate 튜닝, union-bbox, depth·t_z prior, Baxter 실루엣 RC(앵커 없는 13-DOF 발산), 모집단 prior, min-reproj multi-start, kinematic-consistency 잔차 거부(precision 0.4), RANSAC/consensus PnP(이득 없음), top-M 모드 열거 디코더(정답 모드 부재), freeze-θ decoupled solve(도달가능 각도에서 열위). 상세는 `SUMMARY.md` REFUTED 목록 및 §16.5.
