# DINObotPose3 — 진행 요약 + 다음 시도 로드맵 (2026-07-04)

> 2026-07-03 세션의 진행사항 정리와, 축적된 진단 증거에 근거해 **자세 정확도를 더 올리기 위해 시도할 가치가 있는 방향들**의 우선순위 제안.
> 상세 실험 기록: `3_pose_models/DINObotPose3/EXPERIMENTS.md` · 확정 결론: `SUMMARY.md` · 문헌 서베이: `robot_pose_sota_survey.md`

---

## 1. 진행 요약 (2026-07-03 세션, sota-dream 브랜치)

| 마일스톤 | 결과 |
|---|---|
| 저장소 3원 동기화 + NAS 마이그레이션 | GPU서버↔GitHub↔모노레포 동기화, 체크포인트 7.4GB 로컬 미러, 베이스라인 재현 ±0.005 검증 |
| **DREAM real 4-split SOTA** | **mean 0.796 vs RoboPEPP 0.780** (rs 0.818 / kinect 0.811 / azure 0.788 / orb 0.765) — 완전 자동 bbox(더 엄격한 프로토콜) |
| 결정적 레버 | nvdiffrast 정밀 mesh 실루엣 + SAM 마스크(init-render 일관성 선택) render-and-compare — 해상도 448에서 포화, depth/scale 보정기라 원거리 카메라만 적용 |
| 문헌 서베이 | RoboPEPP 0.780이 동일 프로토콜 프론티어 확정 (PoseDiff는 저자 철회 확인), 가림 아이디어 카탈로그 |
| **가림 강건성 벤치** | RoboPEPP Fig.6 프로토콜 재현 — **20-30% 가림에서 승** (0.626/0.525 vs 0.600/0.470), 열화 기울기 동일 |
| 가림 레버 ablation | cov-PnP 채택(+0.011@20%) · 로버스트 실루엣 반증(depth 편향) · 모집단 prior 반증(정답과 싸움) |
| 학습 수행 여부 | **이번 세션 신규 학습 없음** — 전부 테스트타임/솔버 레벨. 계획했던 디노이저 학습은 사전 체크(관절 독립)로 증거 기반 취소 |

남은 격차: orb −0.010 (real 유일 열세), synth 0% −0.020, 40% 극한 가림 −0.023.

---

## 2. 다음 시도 로드맵 (내 분석 — 진단 증거 기반 우선순위)

### ① 멀티스타트 RC + SAM-IoU basin 선택 — 최고 EV, 학습 불필요
- **근거**: 남은 실패의 공통 근원은 "wrong basin" — orb 실패 프레임(rot corr +0.58)과 40% 가림(pose 0.315로 붕괴 후 RC가 못 살림)은 모두 init이 틀린 분지에 있어 conservative refine이 못 빠져나오는 경우. 6월 MCL이 반증된 이유는 **selector가 학습 모델**이라 가설을 구분 못 해서였는데, 지금은 **SAM 마스크 + 정밀 렌더 IoU라는 외부 증거**가 생겼음 — 가설 선택기가 공짜로 확보된 상태.
- **설계**: RC 시작점을 K개(현재 R_init + base-yaw ±30°/±60° 섭동, 또는 rot-head의 상위 K 후보)로 병렬 refine → 최종 SAM-IoU 최고 가설 채택. 배치 병렬이라 비용은 K배가 아니라 ~K/배치.
- **검증**: 가림 벤치 40% + orb held-out에서 A/B. 게이트: 40% ≥ 0.351(RoboPEPP), orb +0.01, 클린 do-no-harm.
- **위험**: IoU가 basin을 구분 못 하는 대칭 포즈 — do-no-harm 게이트(현 가설 IoU가 최고면 유지)로 방어.

### ② 가림-증강 head 재학습 — 유일한 고EV 학습 항목
- **근거**: 40% 구간의 pose 붕괴(0.315)는 detector conf가 낮아지며 angle/rot head 입력이 OOD가 되는 것. RoboPEPP의 가림 강건성 원천이 학습 시 관절 마스킹인데, 우리는 head가 가림 입력을 본 적이 없음. **backbone 동결 + head만** kp_drop/occluder-페이스트 증강 fine-tune → 백본 반증 우회.
- **주의(반증 경계)**: 6월 "occlusion-aware detector retraining 불필요" 판정은 **detector** 얘기 (conf가 이미 96% 캐치). 이건 **angle/rot head**의 가림-입력 강건화로 별개.
- **비용**: head당 2-4h (GPU 1장). 검증: 가림 벤치 30-40% + 클린 do-no-harm.

### ③ RGB/텍스처 render-and-compare — 다음 "큰" 레버
- **근거**: 현재 RC는 실루엣(면적=depth 신호)만 사용. 링크 경계·색·음영 등 내부 텍스처는 **회전/관절각 신호**인데 버리고 있음. nvdiffrast는 텍스처+조명 렌더 가능 — RoboPose(render-and-compare 원조)가 RGB를 쓰는 이유. azure처럼 depth가 이미 정확한 카메라에서 RC가 해로웠던 것도 실루엣-전용의 한계 — RGB 항이면 azure에도 이득 가능.
- **설계**: visual mesh에 단색/추정 albedo + 간단 셰이딩 → 렌더-실사 photometric 항(robust, 마스크 내부만) + 실루엣 항 병행. 정합 문제(조명 도메인 갭)가 리스크라 **먼저 oracle 프로브**(GT 포즈 렌더 vs 실사 유사도가 포즈 섭동에 민감한지)로 신호 존재부터 검증.
- **비용**: 프로브 1일, 본 구현 2-3일.

### ④ crop 해상도 증대 (512→768) — ⚠️ 공짜 아님 (07-04 프로브 결과)
- **근거**: orb 실패의 detector 2D 성분(fail/ok 3.1×)은 far/작은 로봇에서 픽셀 부족. crop 파이프라인은 이미 bbox를 아니까 **crop만 고해상도**로 올리면 백본 재학습 없이(ViT는 가변 해상도 보간) 2D 정밀도가 오를 수 있음. A6000 여유 충분.
- **07-04 프로브 결과**: frozen 512-스택에 768 입력 → orb 0.718→0.582 회귀. conf 캘리브레이션/특징 샘플링이 512 분포 특화 — **detector+head 재학습 캐스케이드 전제**로 격상 (T1/T2 결과 후 투자 판단).

### ⑤ RC per-joint anchor 세분화 — 미세 개선
- 현재 reproj anchor가 모든 관절을 dump 포즈에 균일하게 묶음. 가림 관절만 anchor를 풀면(conf 가중) 가려진 부분의 "대략적 추론"이 실루엣 증거를 더 활용. 반나절.

### ⑥ 시간(temporal) prior — 벤치마크 외 별도 트랙
- DREAM real은 연속 시퀀스. 관절 속도 제한만으로도 가림 프레임 브리징 가능. 단 기존 SOTA와 프로토콜이 달라져 **"video 세팅" 별도 표기 필수** — 논문에선 부가 실험 포지션.

### 엔지니어링 (수치 아님)
- 논문용 full-split 재잠금 (현재 rs/kinect/orb는 held-out 800), orb 576+ 렌더 확인, `sota-dream` → main merge + GitHub/GPU 서버 역동기화.

---

## 3. 하지 말 것 (반증 맵 — 누적)

| 방향 | 반증 근거 |
|---|---|
| 백본 적응 전 계열 (SSL/co-finetune/V-JEPA) | 3회 반증 + V-JEPA 2.1 논문 독립 확인 — sub-pixel 정밀도 파괴 |
| 모집단 통계 prior (평균 끌기, 학습 디노이저 포함) | 관절 독립·광분산 → 정답과 싸움 (−0.09@20%) |
| render∧¬SAM 블랭킷 다운웨이트 | depth 편향 (−0.019). 구제는 명시적 가림체 세그멘테이션 전제 |
| 학습 selector 기반 멀티가설 (MCL) | selector가 병목 — 단 ①의 외부증거(IoU) 선택은 별개 |
| scalar depth head / depth-lift / t prior | sim2real 불가 (6월) |
| 근거리 카메라(azure)에 실루엣 RC | depth 이미 정확 → 마스크 노이즈만 주입 (−0.047) |
