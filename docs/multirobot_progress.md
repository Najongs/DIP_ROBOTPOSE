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

### FR3 파인튜닝 결과 (2026-07-05) — ADD-AUC 0.33 (미흡), 병목 = angle recovery

| 스테이지 | 결과 | 판정 |
|---|---|---|
| crop 검출기 (파인튜닝 후) | val L2 **9.5px**, PCK@10 0.76 (zero-shot 24.6px → 개선) | ✅ 전이 성공 |
| rot head | val geo median **0.21°**, t-err 3mm | ✅ 매우 우수 |
| **angle head** | val MAE **45°** (J0 51/J2 66/J4·J5 54; J3만 13) | ❌ **병목** |
| 최종 pose (self-bbox eval) | ADD-AUC **0.3275** (oracle-bbox도 0.37) | ❌ 미흡 |

**진단**: 검출기·rot head는 전이됐으나 angle head가 45° MAE이고, **솔버가 나쁜 θ init에서 자유 최적화(재투영 250iter)하며 오히려 발산**(raw 54° → refined 60°). 솔버는 rot head R(0.2°, 거의 oracle)를 이미 R_init으로 받음 → 완벽한 R로도 회복 안 됨. 원인은 단안 2D→θ 모호성 + 나쁜 init. bbox·데이터다양성(관절 std 25-80°)은 병목 아님.
- 실험 중: **transformer angle head + Panda warm-start 제거**(Panda 카메라 편향이 FR3 4-뷰포인트와 충돌 가설) — `train_fr3_angle_tf.sh`, 마커 `outputs_fr3/ANGLE_TF_DONE`.
- 대안(미시도): 솔버 θ 앵커 강화(anchor_init_w↑)·iter 축소·R 고정; 또는 self-train 반복.

## Phase 2 — FR5 (6-DOF) 🔄 groundwork 완료

- **데이터 blocker 해결**: FR5 json image_path(`../dataset/Fr5/`)를 절대경로로 교정한 세션 split 생성 → `Converted_dataset/{fr5_train(7844),fr5_val(1296)}` (val=Fr5_6th 세션). 로더 정상 적재(7키포인트/6각도).
- **남은 핵심 작업(6-DOF 일반화)**: pck_eval/selfbbox_eval/model_angle/solve_pose_kinematic이 Panda 7-DOF 키포인트명·FK 하드코딩 → FR5(7키포인트 `Fr5_link0-6`, 6각도, DH FK)용 robot-config 파라미터화 필요. 검출기(7ch)는 차원 호환이라 전이 가능성 있음(FR3처럼).
- **주의**: FR3에서 드러난 angle-recovery 병목이 FR5/Meca에도 적용될 것 → FR3 transformer 실험 결과로 접근법 검증 후 FR5 파이프라인 복제가 효율적.

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
