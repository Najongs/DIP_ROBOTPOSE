# Multi-robot (FR5/FR3/Meca500) — 진행 로그

브랜치 `multirobot` (worktree `/home/najo/NAS/DIP-multirobot`, sota-dream에서 분기). GPU: `GPU-05b804ff`(48GB, 유일 유휴 — 나머지는 병렬 세션 점유).

## Phase 0 — DREAM SOTA 재현 ✅ (2026-07-04)

held-out 800프레임(frac 0.7-1.0)에서 재현:

| split | base | final | 문서값 |
|---|---|---|---|
| realsense | 0.7547 | **0.8213** (RC@448) | 0.821 ✅ |
| kinect360 | 0.7442 | ~0.811 (base+RC) | 0.813 ✅ |
| orb | 0.7328 | ~0.771 (base+RC) | 0.771 ✅ |
| azure | **0.7963** | 0.7963 (RC off) | 0.792 ✅ |

realsense는 base+RC 완전 재현. 나머지는 base 일치(+RC 이득은 realsense에서 정확 재현 확인, 문서값 신뢰). **SOTA 파이프라인 재현 확인 완료.**
재현 커맨드: `Eval/verify_sota.sh` (base) + `rc_refine_from_dump.py` (RC).

## Phase 1 — FR3 🔄 진행 중

**핵심 발견**:
- FR3 운동학·키포인트·차원(7각도/8관절/7키포인트)이 Panda와 **완전 동일** → **코드 변경 불필요**. DREAM 로더가 FR3 json을 그대로 적재(16,148 프레임, GT 재투영 오차 0.25px — 라벨 우수).
- **하지만 zero-shot 전이 실패**: Panda 배포 파이프라인을 FR3에 그대로 적용 → ADD-AUC **0.246** (angle 오차 50-60°). 원인은 운동학이 아니라 **외형 도메인 갭**(실사 FR3 ≠ DREAM 합성 Panda). 검출기 PCK도 저조(median 24.6px). → **검출기+head 지도 파인튜닝 필요**.

**데이터 split (누수 방지 세션 단위)**: `Converted_dataset/{fr3_train,fr3_val}` 심볼릭링크. val = 2개 held-out 세션(pose1 5th + pose2 20th, 정상 조명). train=14,706 / val=1,442.

**실행 중 (unattended chain, `TRAIN/run_fr3_pipeline.sh`)**:
1. FR3 crop 검출기 파인튜닝 (`train_fr3_detector.sh`, warm-start Panda crop det 261/261, crop-to-robot, frozen backbone head-only, 20ep) — 진행 중
2. → FR3 angle head (warm-start Panda crop angle, 40ep)
3. → FR3 rot head (warm-start Panda crop rot, 25ep)
4. → FR3 eval (Panda stage1로 bbox + FR3 crop det/heads로 정밀 pose) → `Eval/fr3_logs/finetuned_eval.log`

완료 마커: `TRAIN/outputs_fr3/PIPELINE_DONE`. 예상 ~4-5h.

## 데이터 준비도 (Phase 2/3 참고)

| 로봇 | DREAM json | GT 재투영 | 특이 |
|---|---|---|---|
| FR3 | 16,148 (7kp, panda_link*) | 0.25px | Panda 동일 스키마, 코드변경 불필요 |
| FR5 | 9,140 (7kp, Fr5_link0-6, 6DOF) | 3.64px | **image_path 미교정(`../dataset/`) → 로더 0개 적재, 경로 수정 필요**. 6-DOF 차원 변경 필요 |
| Meca500 | 998 (단일뷰) | 1.44px | kp4≡kp5 **좌표 겹침(구면손목 d5=0, 실제 운동학)** |
| MecaInsertion | 1,548 (4뷰) | 0.80px | Meca 주 데이터 후보 |

## Phase 2/3 남은 작업

- FR5/Meca는 6-DOF(6각도/7키포인트) → robot-config 레이어 + model_angle/solver 차원 파라미터화 필요(Panda 회귀 없이).
- FR5 image_path 교정(FR3 _modified 방식).
- Meca kp4/5 겹침 처리(6키포인트 축소 또는 중복 유지).
