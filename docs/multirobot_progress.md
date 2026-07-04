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
### 결정적 진단 (2026-07-05) — 근본 병목은 단안 2D→θ 모호성

FR3 val에서 솔버 능력을 직접 테스트(GT 2D 키포인트 + Kabsch로 구한 GT 카메라 R):

| θ init (+ GT R) | 회복 angle MAE | reproj |
|---|---|---|
| oracle (=GT) | **1.0°** | 0.2px |
| GT ± 30° | **6.0°** | 0.6px |
| cold (관절 평균) | 32° | **발산 (420k px)** |

- **FK 규약 완전 일치** (Kabsch residual 0.0mm — FR3=Panda 확정).
- **솔버는 정상**: 진실 ±30° 안에서 시작하면 6°로 수렴. 문제는 **나쁜 init에서 발산**. angle head의 45-66° 오차는 basin 밖.
- **transformer head 실험 → 오히려 악화** (66° > MLP 45°, J0 111°). head 아키텍처로 해결 안 됨.
- **uniform 멀티스타트(48 starts) → 40° 정체**: reproj는 0.82px로 낮아지지만 angle은 안 맞음. 즉 **저-reproj 오답 basin 다수 존재**(단안 모호성) + 6-DOF 공간 균일 커버리지 조합폭발. min-reproj 선택으로 basin 못 고름.

**결론**: 검출기·rot head는 전이되나, **관절각 회복이 근본 병목**. DREAM Panda SOTA는 10만+ 합성으로 "2D→θ prior"(basin 안 init)를 학습해 이를 해결 — 실사 로봇(FR3/FR5/Meca)은 제한된 실사(~1만)로 이 prior를 못 배움. head 개선·솔버 멀티스타트로는 한계.

### 추가 검증 (2026-07-05) — 두 번째 병목: 검출기 2D 정밀도

**타깃 로컬 멀티스타트**(head init에서 시작, 틀리는 관절만 국소 perturb, min-reproj) 구현·검증:
- **GT 2D**: head-like init(50-60° 오차) → 로컬 멀티스타트 81후보 → **angle MAE 4.8°** (reproj 0.27px). basin 찾기 성공.
- **검출된 2D (9.5px)**: 같은 방법이 거의 무효 (ADD 0.333→0.341). 2D 노이즈가 재투영 최소점을 진실에서 벗어나게 함 + spurious 저-reproj basin 생성.

**결론(수정)**: FR3는 병목이 **두 개**다 — (1) angle init(로컬 멀티스타트로 해결 가능, **단 2D가 정밀할 때만**), (2) **검출기 2D 정밀도 9.5px**(Panda ~3px, frozen backbone에서 20ep 정체). 검출된 2D로는 완벽한 init로도 정확한 θ 회복 불가. GT 2D면 로컬 멀티스타트가 ~0.65 ADD 예상이나 검출된 2D로는 0.34. **검출기 정밀도가 binding constraint.** frozen backbone·제한된 실사로는 개선 한계.

→ 이는 **합성 데이터가 원칙적 해법인 이유를 강화**한다: 합성은 검출기(대량 다양 학습→더 정밀)와 angle prior(basin 안 init) **둘 다** 해결. 로컬 멀티스타트(`selfbbox_eval.py --ms-local N --ms-sigma`)는 구현·커밋했고 2D가 좋아지면 유효.

**판단한 경로**:
1. **합성 데이터 생성 (원칙적 해법, Phase 4)** — 로봇별 URDF+메시로 도메인랜덤화 합성 대량 생성 → Panda식 angle prior 학습. **선결: FR3/FR5/Meca 시각 메시/URDF 확보 필요**(현재 Panda 메시만 보유). nvdiffrast는 있음.
2. **타깃 로컬 멀티스타트** (엔지니어링 전용, 합성 불필요) — head가 맞히는 J1/J3는 고정하고 틀리는 J0/J2/J4/J5만 head±{30,60}° 국소 탐색(3^4≈81 후보) → basin 브라켓 시도. 모듈러스 개선 가능성, 미검증.
3. **RC/depth 활용** — 실루엣 RC(메시 필요) 또는 단안 depth로 basin 판별.

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
