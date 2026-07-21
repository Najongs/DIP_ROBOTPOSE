# 2026-07-21 — PAPER_OVERLEAF.tex 논문 조립 세션 (요약)

> 측정 없음 — `docs/dinobotpose3/PAPER_OVERLEAF.tex`를 오버리프 제출 직전 상태로 조립한 편집 세션의 결정 기록. 이전 맥락: [2026-07-20_paper_positioning.md](2026-07-20_paper_positioning.md)(2축 포지셔닝), [2026-07-20_critic_debate.md](2026-07-20_critic_debate.md)(토론·C8).

## 현재 tex 상태 (커밋 기준 최신)

**구성**: Title → Abstract → 1 Introduction(4문단+기여 3항목) → 2 Related Work(subsection 3: Markerless / Render-and-Compare / Test-Time Optimization) → 3 Method(subsection 5: Problem+Overview / Frozen Front-End / Solver+AutoBBox / RC / Occlusion+Training) → 4 Experiments(4.1 Dataset and Implementation Details / 4.2 Results → subsubsection 4: Comparison / Ablation / Runtime+타로봇 / Refuted+Backbone) → 5 Conclusion(3문단) → Appendix(mm 표).

**표 체계 (5+부록 1)**:
- `tab:main`(table*): 로봇별 그룹(Panda: Synthetic DR·Photo + Real AK·XK·RS·ORB / KUKA: DR·Photo / Baxter: DR), 2단 Real/Synthetic 헤더, AUC×100만. **Known Angles/Box 열 제거(07-21)** — midrule 선으로만 2그룹 구분: 위=known-joint(DREAM-F/Q/H + known-angle 상한 이탤릭), 아래=predicted(RoboPose·HoRoPose*·RoboPEPP·RoboTAG·Ours). **HoRoPose GT-box 행도 삭제**(본문에서 82.7→41.4 붕괴로만 언급). 사용자가 경쟁 수치 보충(RoboTAG 전열 등) → 볼드 재배정: Photo 84.3·AK 83.1·KUKA Photo 76.6·Baxter 58.8 = RoboTAG (RoboTAG ORB는 58.8→77.5로 사용자 정정 — Baxter 값 오기입이었음. 실측 4캠 평균 78.7 < Ours 80.4, 본문에 한 문장 반영). **ALL 열 제거**(전 로봇 평균은 direct-pose 조건 불일치로 오도 — Ours 62.8 < RoboPEPP 74.0이 나와 삭제 결정; 실측 4캠 평균 80.4는 절제 표 Full 행에 있고 캡션이 연결).
- `tab:occlusion`(0~40%), `tab:ablation`(leave-one-out + zero-real-adaptation 행), `tab:refuted`, `tab:backbone`, 부록 `tab:mean_mm`(카메라별 평균 ADD mm — 본문에서 제외해 보관).

**수식 (2개만)**: 솔버 백색화 재투영 잔차(w_i 신뢰도 가중 포함), RC 목적함수(soft-IoU + 재투영 앵커). 학습 손실(각도 SmoothL1+FK, 회전 Frobenius+병진)은 산문. DARK 오프셋은 인라인.

**스타일 규칙(확정)**: 무인칭(we/our 금지, 표 "(Ours)"·"to the best of the authors' knowledge"만 예외) / AUC 전량 ×100 / 2D·3D 축약 / 모델명 DINObotPose / KUKA(iiwa7 생략) / 한국어 완역을 각 EN 문단 뒤 % 주석으로 / 초록에 경쟁 논문명·수치 금지 / §상호참조·그림 참조 미기입(마지막에 일괄) / 캡션=표 읽는 법·본문=해석으로 역할 분담 / Experiments 전 결과 수치는 기여의 80.4 하나만.

## 이 세션의 주요 결정 이력 (시간순)

1. 초록 재작성(가이드 5질문 구조) → 경쟁 논문명·수치 제거·격식 문어체로 재수정
2. test-time optimization 용어 근거 확인(HuMoR·iNeRF 웹 검증) → related_work.md §5 + bib
3. 서론 4문단 구조 + 기여 5→3 압축(RC 시스템 / frozen 발견 / 가림), 3로봇은 한 문장 격하
4. 구 인트로에서 선별 이식: HRC·마커 동기, DREAM 계보, 배포 비용 논거 (zero-shot·"RoboPEPP=타깃 SSL" 주장은 배제)
5. references.bib 구축: 사용자 보유분 + TODO 채움(현재 tex 인용 33키 전량 매칭; sgdr는 bib에서 제거되어 tex에서도 제외)
6. Method 전면 작성(코드 검증 기반: PixelShuffle·AdaptiveNorm·soft-argmax 유지, 상위2블록 언프리즈·반복정제·zero-shot 서술 배제) → 수식 코드 대조(w_i 추가, RC·회전 손실 수식화) → 이후 학습 손실 수식은 산문 전환
7. Experiments 작성 → RoboPEPP식 4.1/4.2 재편 → paragraph 리드 → subsubsection 4개
8. 무인칭 전환(~25곳) + 결론 5→3문단(HRC·파운데이션 데이터 생성 전망 포함)
9. ×100 통일(스크립트 치환, 보존: mm·비율·임계값)
10. RW 보강(4문단) → 재압축(3문단, foundation-models 문단은 ¶1·¶2·Experiments로 분산)
11. 표 진화: 실측 대형표(AUC+mm) → 합성 표 추가 → 실측+합성 통합 → mm 부록 이동 → 로봇별 그룹 → ALL 실험(전 로봇 평균은 62.8 문제로 폐기) → Synthetic/Real 2단 헤더
12. 캡션-본문 중복 제거(가림 프로토콜=캡션, known-joint 메커니즘=캡션, 해석=본문)

## 합성·타로봇 열세의 원인 정리 (사용자 질문 답변, 재확인)

학습 부족 아님: ① Panda 합성 열세(74.2/76.9 vs RoboPEPP 83.0/84.1) = 실측 지향 설계의 트레이드(합성은 경쟁자 학습분포 홈그라운드; 동일 조건 HoRoPose* 41.4는 33점 차 승) ② KUKA/Baxter(35.7/31.9/25.2) = 솔버 정제(링크 혼동으로 발산)와 RC(메쉬 부재/레버 불일치) 둘 다 빠진 direct-pose 반쪽 구성 + 로봇 고유 병목(KUKA rot-head R·t, Baxter 손목 관측성 천장 — GT 키포인트로도 불개선).

## 2차 검증·토론 라운드 (07-21 저녁, 8-에이전트 워크플로우)

**원논문 수치 검증 (4 verifier, arXiv HTML 원문 대조)**:
- RoboPEPP(2411.17662) Table 2 대조 — 54셀 중 50 일치. **DREAM-F·DREAM-Q의 Panda DR/Photo가 원문과 스왑**되어 있었음(원문은 Photo-먼저 순서: F는 DR=81.3/Photo=79.5, Q는 DR=77.8/Photo=74.3) → 표 수정. DREAM-H·RoboPose·HoRoPose\*·RoboPEPP 행은 전부 정확. HoRoPose\* XK 결측은 원문 각주(가중치 미공개)와 일치.
- 가림 표: RoboPEPP는 **Fig. 6 플롯으로만** 보고, 본문에 명시된 수치는 40\% 지점(35.1/28.2/14.5)뿐 — 정확히 일치. 0~30\%는 플롯 판독값 → 캡션에 출처 명시 추가.
- HoRoPose(2402.05655): 9개 값 전부 원문 Table 1과 일치. GT-bbox 전제(A_bbox 사용, RoboPEPP known-box 라벨) 확인 → 하단 그룹 제외 타당.
- RoboTAG(2511.07717): 9셀 전부 정확. 관절각 예측 확인(known은 \* 표시, Ours 무표시), 박스는 명시 없음(RoboPEPP 프로토콜 준수 서술에서 추론) → §4.1 문구를 "RoboPEPP 평가 프로토콜을 따라 보고"로 완화. **RoboTAG도 실측 테스트 이미지에 자기지도 2단계 적응**(transductive) → §4.1에 공정성 문맥으로 추가.
- RoboTAG Table 3 = 누적 빌드업 표(Baseline → +컴포넌트, 괄호 증분, 최종행 볼드). 우리 빌드업 로그는 RealSense 단일이라 전면 이식 불가 → leave-one-out 유지 + 3블록 재구성(배포 앵커 / leave-one-out |Δ|정렬 / 적응·프로토콜 변형)으로 tab:ablation 교체.

**비평↔보완 토론 (critic 10건 → defender → verdict)**: FIX 8 / DEFER 2.
- C1 (P0, FIX): §4.2 프로토콜 문단의 "HoRoPose ORB 9.8 붕괴 / RoboPose 32.7"은 **Baxter 열 오전사** — 실제 ORB는 51.6/70.4. tex + PAPER_DRAFT(§4.5, 표1 노트) + SUMMARY까지 정정. "붕괴" 프레이밍도 "24점 하락, RoboPose는 유지"로 완화.
- C2 (FIX): tab:main 캡션에 held-out 30% 프레임셋 차이 한 문장, 기여 1번에 프로토콜 정의 명시, Comparison에 zero-adapt 79.3 한 문장.
- C4 (FIX): RW RoboTAG 서술을 위상 정렬 그래프·종단간으로 정확화 (concurrent 라벨은 부적절해 미기입).
- C5 (FIX): 이탤릭 행 "upper bound" → **known-angle reference** (Azure 78.8 < 배포 79.5라 상한 명칭 모순; 노이즈 범위 내 서술로 교체).
- C6 (FIX): 가림 표 캡션에 측정 스플릿(Panda photorealistic synthetic, occ-aug 헤드+RC) 명시.
- C7 (FIX): tab:backbone 캡션에 Pose (mean)=실측 4캠 평균·unfrozen 명시 (로그 확인: dino 0.742, siglip 0.752).
- C8 (FIX): 배포 하이퍼파라미터 명시 — conf-gate 0.05, RC Adam lr 5e-4·250회·λ_uv=100·min-IoU 0.35·렌더 448/448/512. **revert 가드(max-uv-shift)는 코드 기본 0.0=비활성으로 배포에서 미사용** → 본문의 "픽셀 한계 초과 시 되돌림" 주장 삭제.
- C10 (FIX): 결론의 "파운데이션 모델 학습 데이터 생성 도구" 단정 → "경로 시사, 향후 과제"로 완화.
- C3 (**DEFER, 미해결**): "카메라별 최적 구성 선택"의 선택 기준 — RC on/off는 근거리 깊이신호 규칙(문제없음)이나, **헤드/구성 선택이 held-out(뒤 30%) 평가 점수 argmax였음**(EXPERIMENTS 574/578/593/846). 선택지: ① 70% 적응 스플릿 점수로 재선택(분석 재실행) ② 현행대로 정직 공개 + zero-adapt 79.3에 기대기. **사용자 결정 필요.**
- C9 (DEFER→표 재설계로 해소): tab:ablation 3블록 교체 적용됨.
- 부수 정정: RC Kinect 기여 6.2→6.3 (82.8−76.5 산술).

## 남은 작업

- [ ] **C3 결정**: 카메라별 구성 선택이 평가 프레임 argmax였던 문제 — 70% 스플릿 재선택 vs 정직 공개 (위 2차 라운드 참조)
- [ ] 그림 삽입(fig_pipeline은 PPT 재제작 예정) + 표·수식 \ref 일괄 연결
- [ ] bib: zhou2019continuity·kabsch1976solution 추가됨(연결 완료). sgdr 재추가 여부 사용자 결정
- [ ] RoboPEPP 동일 1000-프레임 재현(사용자 PEPP env 대기) → 페어드 부트스트랩
- [ ] P1: 프레임 CI(--dump-adds), 3-시드 std, zero_adapt angle 변형
- [ ] 오버리프 프리앰블: multirow·booktabs·resizebox·**amsmath**(RC 수식 split 사용) 패키지 필요

## 3차 다듬기 (07-21 밤, 사용자 요청)

- RC 수식이 단 폭을 넘어 `\begin{split}`로 2줄 분할 (amsmath 필요).
- 중복 제거: "hardest setting" 3→2회(§4.1 삭제), frozen 서술 §3.2 압축+§3.5 첫 문장 삭제(§3.2와 중복), 결론의 "learned depth regression...not the only answers"(초록 중복)·"파운데이션 데이터" 문장 잔여 정리.
- 경쟁 수치 본문 반복 축소: 합성 문단에서 RoboPEPP 83.0/84.1 제거(표에 있음), ORB 문단에서 51.6/70.4/77.8 나열 제거("약 24점 하락"만 유지, RoboPEPP 인용 각주 문장은 캡션과 중복이라 삭제), zero-adapt 79.3 vs 78.0 재비교 삭제(Comparison에 이미 있음), 백본 문단 80.0/72.0 제거(표에 있음).
- 캡션 축약: tab:main ~200→~120단어, tab:occlusion·tab:ablation도 읽는 법 핵심만.
- 부록 mm 표: DREAM-F/Q/H 행 삭제(수십만 mm 무의미 값) + 캡션에 생략 사유 1문장.
