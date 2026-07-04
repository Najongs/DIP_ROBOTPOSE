# 평가 (Evaluation)

> 하네스: `3_pose_models/DINObotPose3/Eval/`. 지표 = ADD-AUC@100mm (카메라 프레임, 높을수록 좋음).

## 프로토콜 3축 (비교 시 필수 확인)

1. **관절각 known vs predicted** — 우리는 predicted(어려움). DREAM/CtRNet은 known.
2. **bbox GT vs 자동** — 우리는 **완전 자동**(bbox-from-solved). RoboPEPP 헤드라인은 GT-bbox(auto면 orb 붕괴).
3. **학습 데이터** — 우리 sim-to-real + self-train. PoseDiff처럼 real 라벨 학습은 비교 불가.

세부·SOTA 표: [../references/sota_survey.md](../references/sota_survey.md).

## 지표 계산

- ADD = 예측 카메라 프레임 키포인트 vs GT 카메라 프레임 키포인트 L2 평균, 프레임별.
- AUC = ADD 임계 0→100mm 곡선 아래 면적. `compute_add_auc` (inference_4tier_eval.py:158, delta 1e-5).
- 부가: PCK@{2.5,5,10}px (pck_eval.py, RoboPEPP/DREAM와 640×480 직접 비교), 관절각 MAE(deg).

## 주요 하네스

| 스크립트 | 용도 |
|---|---|
| `selfbbox_eval.py` | **배포 파이프라인 평가** (self-bbox crop + 솔버). `--cov-pnp --dark-decode --bbox-from-solved --bbox-guard`, `--frac-range`(anti-leak held-out), `--dump-npz`(RC용 포즈 덤프), `--occlude-ratio`(가림 벤치) |
| `rc_refine_from_dump.py` | 덤프 포즈에 nvdiffrast+SAM render-and-compare refine. `--render-h`, `--feat-w`(feature-metric, 종료), `--multi-start`(반증) |
| `ab_eval.sh` | 6-split ADD-AUC A/B (realsense/azure/kinect/orb/synth) |
| `occlusion_bench.sh` | RoboPEPP Fig.6 가림 프로토콜 (RoI {0,10,20,30,40}% occluder) |
| `refine_eval.py` | MLP→운동학 refine 평가 (비정렬 dataset — 재현 비교엔 selfbbox 권장) |

## 배포 성적 (2026-07-04, held-out 800/cam; azure full-1000 RC-off)

| 카메라 | 우리 | 구성 |
|---|---|---|
| realsense | **0.821** | crop+rot-adapt +cov-PnP +DARK +RC@448 |
| kinect360 | **0.813** | crop+rot-adapt +cov-PnP +DARK +RC@448 |
| azure | **0.792** | crop base +cov-PnP +DARK (RC **off** — 근거리) |
| orb | **0.771** | crop+rot-adapt +cov-PnP +DARK +RC@512 |
| **mean** | **0.799** | vs RoboPEPP 0.780 (+0.019) |

## 재현 게이트

- **결정적 샘플링**: EvalDataset(정렬+stride). refine_eval의 비정렬 PoseEstimationDataset은 기계 간 다른 subset.
- **K 스케일링 정합**: GT 2D+FK(GT각도)→PnP가 GT keypoints_3d를 수 mm 이내 복원해야 함(A0 게이트).
- 카메라별 RC on/off: depth/scale 보정기라 원거리만 이득 (azure off).

## 진단 프로브 (다수)

`decompose_occlusion.py`(conf-bin AUC), `realsense_failure_diag.py`(ADD 성분 분해), `occlusion_probe_inframe.py`(가림 페이스트), `depth_ceiling_probe.py`(GT-depth 천장), `rgb_rc_probe.py`/`feat_rc_probe.py`(RC oracle 프로브). 진단 결과는 [../00_overview.md](../00_overview.md) 반증 맵 참조.

## 관련
- 실험별 결과: [../experiments/](../experiments/README.md)
- 로드맵: [../references/next_directions.md](../references/next_directions.md)
