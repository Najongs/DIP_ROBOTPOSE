# 2026-07-20 — 비평가↔보완자 에이전트 토론 (논문 사전 리뷰)

> 목적: 제출 전 적대적 리뷰 시뮬레이션. 비평가(탑티어 리뷰어 페르소나)와 보완자(저자 측) 에이전트가 2라운드 공방, 양측 모두 ablation 로그를 직접 검증. **최대 성과 = C8 발견**(아래).

## 라운드 요약

- **R1 비평**: C1~C7 (치명 2·중대 3·보통 2). 초기 판정 **Reject**.
- **R1 보완**: RC=테스트타임 최적화(RoboPose 계보) 논변, "self-train 제거해도 0.794" 반박, DREAM 실측=Panda 전용은 리그 표준 등.
- **R2 비평**: 로그 재검증 — RC 논변 **수용·철회**. 그러나 0.794 반박이 **로그와 불일치**함을 적발(→C8). 판정 **Reject→Borderline(major revision)**.
- **R2 보완**: 로그 재확인 후 **오독 인정·철회**. C8 임계 실험 스펙 + 정직한 예상(0.78~0.79) + 이행 체크리스트.

## 🔴 C8 — 토론이 발견한 실제 구멍 (임계 실험)

표 6의 `−occ-aug/self-train` 행(0.794)은 **angle 헤드만 합성으로 되돌리고, rs/kinect/orb의 카메라별 자가학습 rot 헤드는 유지**한다(`ablation_run.sh`의 `no_occaug` 케이스가 `HEAD=$CLEANHEAD`만 교체, `ROT`는 lightstack 유지 — 로그 대조로 양측 모두 확인). 따라서:

- "자가학습 완전 제거해도 0.794로 이긴다"는 방어는 **불성립** (기존에 아무도 인지 못 했던 절제 설계 결함)
- 표 6의 self-train 기여 −0.010은 **과소평가** (angle 성분만)
- **"완전 합성 헤드(angle+rot) + RC" = zero-real-adaptation 수치가 미측정** — 이 한 수치가 "적응 없이도 SOTA" 주장의 진위, 나아가 accept 등급을 가른다

**실험 (실행됨, `zero_adapt` 케이스)**: `ablation_run.sh`에 `zero_adapt` 추가 — 합성 angle `angle_occaug_light_20260704_015400` + 합성 rot `rot_crop_occaug_20260704_002102`(= azure 배포 헤드)를 rs/kinect/orb에 적용 + 배포 RC(448/448/512). azure는 배포 구성 자체가 zero-adaptation이라 0.7945 재사용. 드라이버 `c8_zero_adapt_driver.sh`(A6000 순차, ~2 GPU-h).

**정직한 사전 예상 (보완자)**: FINAL_MODEL 주석(합성 light head 직행 시 −0.01~0.05/카메라)과 rot-adapt 이득(+0.007~0.011/카메라) 양쪽 추정 모두 **mean 0.78~0.79** — RoboPEPP 0.780에 걸침. 결과별 서사:
- CI가 0.780 배제하며 초과 → "적응 없이도 SOTA" 성립
- 동률/미달 → "적응 허용 체제 0.804; 학습-free 코어 ~0.78은 RoboPEPP와 동급, per-camera 자가학습이 결정적 마진" 으로 정직 귀속

## 판정표

| # | 비판 | 최종 판정 | 처리 |
|---|---|---|---|
| C1 | 자가학습·RC = 테스트-도메인 튜닝, fair 비교 불성립 | CONDITIONAL | RC 논변은 비평가 철회. C8 실험이 판가름. cross-sequence self-train은 P1 |
| C2 | SOTA가 Panda-실측 한정, 합성·타로봇 열세 | CONDITIONAL→반영 | "실측=Panda뿐"은 리그 표준 인정. ✅ "generalizes" 삭제·SOTA 스코프 명시 |
| C3 | 노이즈(0.010) > rs/orb 마진, 통계 검정 부재 | CONDITIONAL | ✅ "4/4 상회"→"mean SOTA(노이즈 2배)+kinect/azure 명확+rs/orb 동률" 재프레이밍. 부트스트랩 CI는 P0 잔여 |
| C4 | 경쟁 수치 인용·프레임 집합 상이·HPE\* 출처 | CONDITIONAL | ✅ §4.1 비교 주의문·§4.5 HPE\* 출처 명시. RoboPEPP 동일-프레임 재현은 **보류(사용자가 PEPP 환경 설정 후)** |
| C5 | 신규성 얇음(재구성+음성결과) | CONDITIONAL | 한 문장 기여 확정(학습형 깊이 해법을 학습-free 기하로 대체한 시스템 + cov-PnP·bbox-from-solved). 기여 재배열은 **C8 결과 확인 후** |
| C6 | 3-block ADD 0.0=아티팩트, SigLIP2 모순, frozen 정의 모호 | RESOLVED→반영 | ✅ 표 11 아티팩트 처리·"3종 반증"→"2종+아티팩트" 전면 정정·frozen 용어 정의 추가. "백본-무관≠배포-무관" 논리 유지 |
| C7 | 0.6fps vs 원격조작·서보잉 동기 | RESOLVED→반영 | ✅ 서론 동기를 캘리브레이션(저빈도·정확도-우선) 중심으로 교체 |
| C8 | zero-real-adaptation 절제 부재 (신규 발견) | **실험 실행 중** | ✅ 표 6 행 라벨 정정(‡ angle만 제거 명시). 결과 나오면 표 6에 행 추가 |

## 잔여 작업

- [ ] **C8 결과 반영**: `ablation_logs/results.tsv`의 `zero_adapt` 3행 + azure 0.7945 → mean 산출 → 표 6 행 추가 + 서사 결정
- [ ] **프레임-부트스트랩 CI**: rc_dumps의 프레임별 ADD 재표집 → 배포·zero_adapt·per-cam 95% CI (P0, 수 분)
- [ ] **RoboPEPP 동일 1000-프레임 재현**: 사용자가 환경(`PEPP` env?) 설정 후 → 페어드 부트스트랩 (보류)
- [ ] P1: zero_adapt angle 변형(plain `angle_crop_20260605` — 합성 헤드 선택 민감도), cross-sequence self-train, 3-시드 std
- [ ] C8 결과에 따라 기여 목록·초록 서사 최종 조정 (C5와 연동)

## 이 세션에서 논문에 반영된 P0 수정 (전부 한/영 병기)

§1 동기(캘리브레이션 중심)·2축 문단 재프레이밍 / 초록·기여1 "DREAM-Panda" 스코프 / 기여2·§3.2·§4.9·결론 "3종 반증"→"깨끗한 2종(+아티팩트 1)" / §3.2 frozen 용어 정의 신설 / §4.1 비교 주의문([cited]·프레임 집합 상이) / §4.2·§4.6·표1 "4/4 상회"→노이즈-인지 재프레이밍 / §4.5 HPE\* 출처 / 표 4 "일반화"→"적용 가능성" / 표 6 ‡행 라벨 정정 / 표 11 3-block ⚠️ 아티팩트 처리
