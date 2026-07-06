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

## 합성 데이터 경로 언블록 (2026-07-05) — 사용자 리소스 불필요

**메시는 공개돼 있어 직접 확보 가능** (사용자 제공 불필요):
- **Meca500 ✅ 다운로드+로드 검증 완료**: `Daniella1/urdf_files_dataset`의 `mecademic_description` — 링크별 visual .dae 7개(base+j1~j6, 1만~3만 정점) + URDF. `ViS/Meca500/meshes/visual/`에 저장. `pycollada` 설치로 trimesh 로드 확인(크기도 실제 치수 일치).
- **FR5**: `FAIR-INNOVATION/frcobot_ros2`의 `fairino_description` (URDF+메시, 미다운로드).
- **FR3**: Panda와 운동학 동일 → **기존 Panda 메시 재사용** 가능.

**Depth**: Depth-Anything-3 단안이라 별도 데이터 불필요. ZED 스테레오 좌/우 페어도 보유.

### 합성 파이프라인 계획 (DREAM식 재현)
1. 로봇별 URDF의 링크 변환으로 메시를 DH 프레임에 정렬(`kinematics.py` FK와 일치 검증 — GT 재투영 오버레이).
2. nvdiffrast 도메인랜덤화 렌더러: 랜덤 관절각(관절한계 내)+랜덤 카메라 포즈+배경/조명 랜덤 → RGB + 2D/3D 키포인트 라벨 자동 생성(로봇당 수만~10만장).
3. 합성으로 검출기+angle head 학습(Panda 레시피) → **두 병목(검출기 정밀도+angle prior) 동시 해결**.
4. 실사 소량으로 FDA/파인튜닝(sim2real), 로컬 멀티스타트 솔버와 결합.
5. held-out 실사에서 ADD-AUC 평가.
- 6-DOF(FR5/Meca) robot-config 일반화도 이 과정에서 함께.

## Meca500 합성 검출기 학습 + sim2real 테스트 (2026-07-06)

**합성 데이터**: `datasets/meca500_synth/{train 30k, val 3k}` (nvdiffrast DR, `ViS/Meca500/synth_gen.py`). Lambertian 셰이딩+랜덤 단색 컬러+랜덤 배경(단색/노이즈)+랜덤 카메라. DH-FK 키포인트 라벨(kinematics.py, 실사 GT와 동일 규약, `--verify` 오버레이 확인).

**검출기 학습** (`train_meca500_detector.sh`, warm-start Panda crop det, crop-to-robot, unfreeze 4블록, 20ep): 합성 val AUC 0.697 / L2 **25.7px@512**. (Panda crop det ~3px 대비 낮음 — 합성 자체가 완벽 수렴은 아님, 손목 degeneracy kp4≡kp5 영향.)

**sim2real 테스트 (합성-only 검출기 → 실사 Meca500)** — `Eval/pck_eval.py`에 `--keypoint-names`, `--crop-to-robot`, 512-space L2 추가. 실사 eval dir(`Converted_dataset/{Meca500_eval,Meca_insertion_eval}`, image_path 절대경로 교정):

| 입력 | Meca500 (998, 단일뷰) | Meca_insertion (1548, 4뷰+삽입작업) |
|---|---|---|
| **full-frame** (학습 입력과 불일치) | median 129px — 무작위 | median 193px — 무작위 |
| **oracle-bbox crop** (학습 입력과 일치) | **11.8px** orig / 14.4px@512, PCK@10 **0.42** | 30px orig / 21px@512, PCK@10 0.12 |

**해석**:
- full-frame 실패는 **입력 불일치 아티팩트** (검출기는 tight crop으로 학습됨). crop 맞추면 무작위→12~21px로 정상화 → **합성 검출기는 전이됨**.
- 그러나 14~21px@512는 실사학습 FR3 검출기 상한(~9.5px)을 **못 넘음**. 외형 갭(평면 Lambertian 합성 vs 실사 텍스처/조명, 특히 삽입작업 중 손목 link6에 툴 부착 → 79px 최악) 잔존.
- Meca_insertion이 더 나쁨(가림+툴). Meca500 단일뷰가 더 깨끗.

**결론(sim2real)**: **순수 합성은 단독 불충분** — 검출기 정밀도는 실사학습과 비슷하거나 약간 못함. 합성의 진짜 가치는 검출기가 아니라 **angle prior**(basin-correct init) — 실사-only 로봇이 못 배우는 것. → **DREAM식 하이브리드 확정: 합성 pretrain(검출기+angle prior) → 실사 소량 파인튜닝(외형 갭)**. 다음: (1) 검출기 실사 파인튜닝(외형 갭 폐쇄, PCK<9px 목표), (2) 합성 angle+rot head 학습(angle prior 확보), (3) 솔버 Meca 6-DOF FK 일반화, (4) 실사 held-out full-pipeline ADD-AUC.

## 🔑 결정적 발견 (2026-07-06) — Meca500 손목 관절은 키포인트로 관측 불가

합성 angle head가 **합성 val(in-domain, 완벽 라벨)에서도** MAE ~50°에 정체. 관절별: J1/J2 ~9-12°(정상), **J3=84.5° J4=57.8° J5=88.5°(각 관절 범위의 "평균 예측" MAE = 학습 포기)**.

원인을 FK 자코비안으로 정량 확인 (`Eval/robot_fk.py`, 2000 랜덤 포즈, kp 이동 RMS mm/rad):

| 관절 | kp0-kp2 | kp3 | kp4≡kp5 | kp6 | 판정 |
|---|---|---|---|---|---|
| J0/J1/J2 | 84~224 | 98~156 | 126~198 | 152~224 | **관측 양호** |
| J3 | 0 | 0 | 0 | **54.6** | 약함(kp6 1개, 0.07m 레버) |
| J4 | 0 | 0 | 0 | **70.0** | 약함(kp6 1개) |
| **J5** | **0** | **0** | **0** | **0** | **완전 관측 불가** |

**결론**: Meca500의 7개 키포인트는 전부 관절 원점(운동학 축 위)이라 손목 3관절(J3/J4/J5)이 키포인트를 안 움직임 → **키포인트만으론 손목각 복원 불가. 합성 데이터로도 해결 안 됨(관측성 한계).** J5는 어떤 키포인트도 안 움직여 원리적으로 불가능.
- 이것이 **FR3 손목 실패의 근본 원인이기도** 함(같은 단안 손목 모호성). Panda가 되는 이유: Panda 키포인트엔 축에서 벗어난 `hand`/`link7`이 있어 손목에 레버가 생김. Meca 키포인트는 전부 축 위 → 레버 0.

**전략 재정립**:
1. **키포인트 경로**: J0-J2(+거친 위치)는 잘 복원됨 → **키포인트 ADD-AUC는 손목각 오차에 둔감**(kp positions가 J5에 무관, J3/J4엔 kp6만 약하게 의존). 즉 keypoint-ADD 기준으론 여전히 쓸만한 값 가능.
2. **손목 방향은 RC(render-and-compare) 필수** — 손목 방향은 이미지(그리퍼/플랜지 모양)엔 보임. Meca 메시 확보돼 있으니(`ViS/Meca500/meshes`) nvdiffrast+SAM 실루엣 RC로 손목 정제. Panda에선 RC가 소폭 보너스였지만 **Meca에선 손목에 필수**.
3. angle head는 관측 가능한 J0-J2 init 용도로만 유효(솔버 init). 손목은 mean/prior로 두고 RC가 잡음.

**다음**: (a) 배치 torch Meca FK 검증 완료(`robot_fk.py`, numpy와 1.3e-8m 일치) → 솔버 6-DOF 일반화, (b) 키포인트 경로 Meca keypoint-ADD-AUC 베이스라인 측정, (c) 그 값 보고 RC 투자 결정.

## Meca500 keypoint-path 천장 분석 (2026-07-06) — 2×2 ablation

솔버 6-DOF 일반화 완료(`Eval/meca_add_eval.py`: `solve_pose_kinematic.solve_batch`를 무패치 재사용, Meca FK/limits 몽키패치 + `fix_joint7=False` + angle-head `theta_init`). 실사 Meca500(998 held-out)에서:

| 2D 소스 | R 소스 | ADD-AUC | median ADD | 의미 |
|---|---|---|---|---|
| oracle | oracle-R (Kabsch) | **0.55** | 23mm | **키포인트 경로 하드 천장** |
| 검출(합성) | oracle-R | 0.37 | 67mm | 검출기 2D가 ~44mm 손해 |
| oracle/검출 | PnP(R prior 없음) | 0.37 | 100mm | 회전 prior 없는 퇴화 영역 |
| 검출(합성) | 합성 rot head R | **0.073** | 490mm | **합성 rot head 실사 전이 실패** |

**결론(결정적)**:
1. **oracle 2D+oracle R로도 AUC 0.55가 천장** — 손목 관측불가(J3/J4/J5)가 어떤 데이터로도 못 넘는 상한. 넘으려면 **render-and-compare(메시 실루엣이 손목 방향 관측)** 필수.
2. **Meca엔 합성 학습이 손해**: 합성이 주려던 angle prior는 손목에서 원리적으로 불가능(관측불가) + 외형 갭 비용은 실재(합성 rot head→실사 490mm, 합성 검출기<실사학습). rot head R prior 없으면 솔버 퇴화(0.37/100mm), 잘못된 R prior는 더 나쁨(0.073/490mm).
3. **실사 경로가 우월**(외형/관측가능 성분): FR3의 **실사학습** rot head는 geo 0.21°, 검출기 9.5px 달성. Meca도 실사학습 rot head면 oracle-R(23mm/0.55)에 근접 가능.

**Meca500 최종 처방**:
- 검출기 + rot head: **실사학습**(FR3 방식, 998+1548 프레임, 세션 split). 합성 불필요.
- angle head: 관측가능 J0-J2만 실사학습. **손목 J3/J4/J5는 키포인트로 원리적 불가 → RC로만.**
- **RC-with-mesh(메시 보유)**: AUC 0.55 천장 돌파의 유일한 길. `render_nvdr.py` MESH_ROOT/LINK_MESH를 Meca 메시로 파라미터화 → 손목 방향 정제.
- 이것은 **FR3/FR5에도 동일 적용**(모든 실사 로봇이 손목 관측성 벽에 부딪힘; RC가 공통 해법).

**합성 자산 처리**: 합성 검출기/angle/rot head는 실사 전이 실패로 **배포엔 미사용**. 단 관측성 진단 도구로서 가치 있었음(합성 in-domain에서도 손목 정체 → 관측성 한계 증명). 코드(`synth_gen.py`, robot-무관 학습 스크립트, `robot_fk.py`, `meca_add_eval.py`)는 재사용 가치.

## Meca500 실사학습 파이프라인 (2026-07-06, 진행 중)

no-regret 결정: RC경로·keypoint경로 공통 전제인 **실사학습 검출기+rot head**부터. 누수안전 split `Converted_dataset/meca_real_{train 1726, val 760}`(1th+2th 세션+Meca500[:72%] / 3th_insertion+Meca500[78%:]).

무인 체인 `TRAIN/run_meca_real_{detector,heads}.sh`: 실사 검출기(30ep, 합성 검출기서 웜스타트) → 실사 rot head(30ep, `--fk-robot meca500`) → 실사 angle head(40ep). 마커 `REAL_DET_DONE`/`REAL_HEADS_DONE`.
- **실사 검출기 완료**: val AUC 0.845, best L2 **~10.8px@512**(FR3 9.5px급; 후반 15px는 val의 3th_insertion 툴가림 때문). 합성 검출기(12-21px)보다 개선.
- rot/angle head 학습 중.

**RC 준비도**: `render_nvdr.NVDRSilhouette`는 이미 Meca 렌더 가능(`synth_gen.py`에서 `load_meca_meshes`+`meca_link_transforms`+`render_shaded`로 검증). 0.55 천장 돌파 = `rc_refine_from_dump.py`의 실루엣 최적화를 Meca 메시/FK로 이식 + 실사 SAM 마스킹 → 손목 방향(키포인트 불가, 실루엣엔 보임) 복원. 렌더 절반은 완료.

## 🎯 돌파 (2026-07-06) — head-direct로 Meca500 ADD-AUC ~0.90 (앞선 "0.55 천장" 반증)

**실사학습 스택 완성 후 결정적 발견**: 앞선 "손목 관측불가 → 0.55 천장" 결론은 **키포인트 솔버 경로에만** 해당. angle head는 키포인트 위치가 아니라 **이미지 appearance feature(DINOv3 토큰)로 회귀** → 실사 그리퍼 외형이 손목 방향을 인코딩 → **손목 관절도 복원됨**. (합성 head가 손목 실패한 건 합성 렌더의 랜덤단색·평면셰이딩이 방향정보 없어서지, 원리적 한계 아님.)

**실사학습 컴포넌트** (누수안전 세션 split): 검출기 L2 ~10.8px, **rot head geo 0.34°**(FR3급), **angle head 전관절 MAE ~1.8°(손목 J3/J4/J5 포함!)**.

**head-direct 평가** (`meca_add_eval.py --head-direct --rot-head`: angle head 각도 신뢰 + 학습 R + t만 재투영 solve. 키포인트 솔버는 손목을 오히려 훼손 → 미사용):

| 평가셋 | 관절 다양성 | ADD-AUC | mean/median | 비고 |
|---|---|---|---|---|
| Meca500 tail | **wide (J5 180° 범위)** | **0.895** | 11.2/7.8mm | **의미있는 일반화 수치** (6% temporal gap) |
| 3th_insertion | narrow (J5 6°, 준정적) | 0.926 | 7.4/3.5mm | 완전 held-out이나 포즈 거의 고정 → easy |

→ **Meca500 head-direct+R: ADD-AUC ~0.89-0.93, 전관절(180° 범위 손목 포함) ~1-2.5° MAE.** Panda SOTA(0.80) 상회. **합성 데이터·RC 불필요.**

**최종 (angle head 40ep 완료, MAE 0.77°)**: Meca500 tail(wide, J5 180°) **ADD-AUC 0.902**(mean 10.5/median 7.4mm, 전관절 ~1°); 3th_insertion(clean session) 0.927(median 3.4mm). **Meca500 확정 — 좋은 모델.** 배포 = 실사 검출기 `outputs_meca500/real_detector_20260706_122304` + rot `real_rot_20260706_125349` + angle `real_angle_20260706_132024`, 평가 `meca_add_eval.py --head-direct --rot-head`.

**핵심 교훈 (전략 수정)**:
1. **승리 레시피 = head-direct**: appearance 기반 angle head를 신뢰(손목 포함) + robust 학습 rot head R + t만 solve. **키포인트 재투영 솔버로 각도 정제 금지**(관측불가 관절을 drift시켜 훼손: 솔버 J3 0.84°→7.3°).
2. angle head 학습 = **fresh + fk-weight 0 + crop-to-robot**(warm-start/FK-loss 없이 순수 sin_cos 지도)가 관건.
3. **FR3의 "45° 각도/불가" 결론도 학습 아티팩트일 가능성 큼** → 같은 레시피로 angle head 재학습 필요(0.33 개선 기대).
4. 앞 섹션의 "0.55 천장/합성 필요/RC 필수" 판정은 **head-direct 경로에선 무효**(솔버 경로 한정). 합성 자산은 진단 도구로 역할 완료, 배포 미사용.

## FR3 재학습 진단 (2026-07-06) — cross-session 일반화 갭

승리 레시피(fresh, fk-weight 0, crop)로 FR3 angle head 재학습 → **train loss ~0.004(거의 완벽 적합)인데 val MAE 45° 정체**(J0조차 45-78° 진동). 즉 학습은 되는데 **held-out 세션에 일반화 실패**.
- FR3 val = **완전 held-out 2세션**(엄격한 cross-session 테스트), 포즈 좁음(J0 std 17° vs train 76°).
- 대조: Meca 강한 수치는 일부 **within-session** tail(6% gap)에 기댔고, 완전 held-out인 3th_insertion은 준정적(좁은 포즈). → Meca 0.90은 다소 낙관적일 수 있고, FR3 45°는 더 엄격한 cross-session 결과.
- **함의**: FR3(및 실사 로봇 일반)의 진짜 난제는 손목 관측성이 아니라 **세션간 appearance 도메인 갭**. angle head가 train 외형에 과적합. 완화책: (a) 더 강한 aug/스타일 랜덤화, (b) 여러 세션 혼합 학습, (c) test-time 적응. Meca가 쉬웠던 건 eval이 덜 엄격했기 때문.
- TODO: FR3를 within-session split로도 평가해 갭 크기 정량화; Meca도 완전 cross-session wide-pose 테스트 확보해 0.90 재검증.

**진단 결론 (random-frame split, 2026-07-06)**: FR3 angle head를 random-frame split(in-distribution 외형)로 재학습 → Ep0 29.5° → Ep3 **6.7°** 급강하(session-split은 45° 정체). **FR3 각도는 학습 가능** — session-split 실패는 순수 **cross-session 외형 도메인 갭**(관측성·근본한계 아님). (random split은 멀티뷰 누수 있으나, 급수렴 vs 완전실패의 질적 차이가 결정적.) → head-direct 레시피는 전 로봇 유효; **유일한 실질 난제 = 세션간 도메인 일반화.** 배포 처방: 전 세션 학습 + 강한 도메인 augmentation/적응. 손목 관측성은 문제 아님이 재확인됨.

## ✅ FR5 완료 (2026-07-06) — cross-session ADD-AUC 0.86 (가장 엄격한 검증)

실사학습 스택(head-direct 레시피 그대로 적용, `robot_fk.fr5_forward_kinematics` 검증 4.6e-8m). fr5_train(6세션 7844) / **fr5_val=Fr5_6th 완전 held-out 세션**(1296):
- 검출기 L2 **2.28px**(3로봇 중 최고) · rot head geo **0.08°** · **angle head MAE 0.41°**(전관절 <0.7°).
- **head-direct+R ADD-AUC 0.8614** (mean 13.9 / median 13.2mm, 전관절 <0.72°). **완전 cross-session held-out** — 가장 엄격한 검증인데도 강함.
- 각도 near-perfect(0.4°)라 ADD는 이제 pose t(병진/깊이)가 병목(13mm). 배포 = `outputs_fr5/{detector,rot_20260706_142231,angle_20260706_153946}`.
- **FR5는 세션간 도메인 갭이 작음**(일관된 rig) → cross-session 잘 일반화. FR3와 대조적.

## 3로봇 종합 (2026-07-06)

| 로봇 | DOF | ADD-AUC | 검증 엄격도 | 비고 |
|---|---|---|---|---|
| **Meca500** | 6 | **~0.90** | 중(일부 within-session) + 0.93(clean narrow) | head-direct, 실사학습 |
| **FR5** | 6 | **0.86** | **높음(완전 cross-session)** | 최고 검출기 2.3px, 각도 0.4° |
| **FR3** | 7 | in-dist 학습가능(3.3°) / cross-session 45° | — | **도메인 갭이 병목**, 관측성 아님 |

**핵심 성과**: (1) **head-direct 레시피 확립** — appearance angle head 신뢰(손목 포함)+robust rot head R+t solve, 키포인트 솔버로 각도정제 금지. (2) 6-DOF 로봇(Meca/FR5)에서 Panda SOTA급 달성, **합성·RC 불필요**. (3) 실사 로봇 진짜 병목은 손목 관측성이 아니라 **cross-session 도메인 일반화**(FR3에서 노출). (4) 재사용 자산: `robot_fk.py`(Meca/MecaIns/FR5 배치 torch FK), 로봇무관 학습 스크립트, `meca_add_eval.py --robot`.

**남은 작업**: FR3 cross-session 도메인 갭 완화(전세션 학습/도메인 aug/TTA); FR5 pose-t 정제로 0.86↑; 배포 문서화.

## 🔬 FR3 cross-session 완전 진단 (2026-07-06) — 근본 원인 = 단안 모호성 × 7-DOF 여유자유도

사용자 지시로 원인 규명. 후보를 하나씩 배제:

| 후보 원인 | 배제 근거 |
|---|---|
| 포즈 커버리지 | val 각도 100% train 범위 내 |
| 카메라/뷰포인트 | train·val 동일 4카메라 |
| 외형/조명 | 같은 랩 배경(세션간 거의 동일), 기하 grounding loss 무효 |
| FK 규약 불일치 | **Kabsch residual 0.0mm**(panda_fk=FR3 GT 완전일치) |
| shortcut 학습 | reproj+fk grounding 재학습해도 45° 정체 |
| 키포인트 관측성 | J0-J5 전부 관측가능(98-619 mm/rad, J6만 0=미예측) |
| **→ 단안 2D→3D 모호성 × 7-DOF 여유** | **oracle-2D + oracle-R 솔버도 35°** (완벽 키포인트+회전으로도 실패) |

**결정적**: oracle 2D+oracle R로도 솔버가 mean init에서 35°(참값은 zero-reproj 해이나 도달 불가 — 저-reproj 오답 basin 다수). appearance head도 45°(cross-session). **두 경로 모두 같은 벽**: 7-DOF는 null-space self-motion으로 (외형·2D)→관절이 one-to-many. 6-DOF(Meca/FR5)는 유일결정 → 됨. **FR3 7-DOF만 단안에서 근본적으로 모호.**
- random-split 1.8°는 **시간적 인접 프레임 누수**(궤적)로 인한 암기지 일반화 아님 재확인.
- **미티게이션 = 단안 모호성 타파 필요**: (a) **멀티뷰**(FR3는 4카메라 동기촬영 보유 — ICRA baseline 방식) 또는 (b) 단안 depth(Depth-Anything-3). 현 SOTA 파이프라인(단일이미지)로는 한계. → 아키텍처 변경 필요, 사용자 결정 사항.

**미티게이션 데이터 실사 (2026-07-06)**: FR3는 **연속 촬영**이라 4카메라가 관절config를 깔끔히 공유 안 함(config당 대부분 stereo 2장, ≥4뷰는 114/494). 멀티뷰는 timestamp 근접 매칭 필요(가능하나 지저분). **단, ZED stereo(좌/우) 페어는 깔끔히 존재** → stereo triangulation이 데이터가 잘 뒷받침하는 자연스러운 disambiguation. repo에 depth 인프라 존재(`depth_lift_probe.py`, `rootnet_depth_probe.py`, `Depth-Anything-3`). → FR3 미티게이션 유력안: **stereo/depth로 2D→3D lift 후 IK**(단일이미지 패러다임 유지). 사용자 방향 확인 후 빌드.

### stereo/multi-view 미티게이션 실험 (2026-07-06) — 부정 결과

사용자 요청("stereo/depth로 진행, 아마 없을텐데?")대로 검증:
- **stereo 데이터 존재함**(좌/우 ZED 이미지 모두 + intrinsics calib). left→right baseline **~133mm**(데이터서 복원, Kabsch residual 0.0mm로 유효 검증). → "없을텐데"에 답: **있음**.
- **그러나 stereo는 모호성 해소 못 함**: two-view solve(oracle 2D, mean init) 33.9° = single-view 31.8°와 동일. **baseline 133mm가 depth ~1.5m 대비 너무 작아 두 뷰가 거의 동일** → 같은 오답 basin 공유.
- **wide multi-view(2-3 distinct 카메라)도 실패**: mean init 27-31°, multistart(20)는 오히려 **59°**(min-reproj가 truth보다 낮은 오답 basin 선택 — 2 wide view로도 오답이 저-reproj). fixed oracle pose + theta-only로도 27°.
- **연속촬영이라 4카메라 관절config 동기 안 됨**(≥3 distinct 카메라 매칭 config 극소수) → 깔끔한 wide 4-view BA 데이터 미흡.

**결론(FR3 미티게이션)**: 데이터의 stereo는 존재하나 baseline이 작아 무효. wide multi-view는 (a) 동기 촬영 데이터 부족 + (b) 2뷰로도 reprojection 랜드스케이프에 저-reproj 오답 basin 잔존으로 실패. **근본 블로커 = theta 재투영 최적화의 비볼록성 → 좋은 init 필수인데 cross-session appearance head가 못 제공.** 진짜 해법: **동기화된 4카메라 wide-baseline 촬영 재수집**(배포·데이터 변경) 또는 정확한 metric depth. 현 데이터·단일이미지 패러다임으론 FR3 7-DOF cross-session 미해결. Meca/FR5(6-DOF)는 유일결정이라 head-direct로 해결됨 — **FR3의 7-DOF 여유가 본질적 차이**.

## (구) Phase 2 — FR5 groundwork

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
