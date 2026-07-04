# 데이터 구조 (Data)

> DREAM 벤치마크 (NVIDIA, Franka Panda). 데이터는 `datasets/ICRA_multiview/Converted_dataset/`.

## 위치 & 심링크

- 원본: `datasets/ICRA_multiview/Converted_dataset/`
- 프로젝트 내 접근: gitignored 심링크 `3_pose_models/DINObotPose3/Dataset/` (원격 레이아웃 미러)
  - `Dataset/DREAM_real → datasets/ICRA_multiview/DREAM_real` (최상위 필수 — image_path가 `..`를 문자열 해석)
  - `Dataset/DREAM_syn → .../DREAM_syn`
  - `Dataset/Converted_dataset/DREAM_real → .../Converted_dataset/DREAM_to_DREAM`
  - `Dataset/Converted_dataset/DREAM_to_DREAM_syn → .../DREAM_to_DREAM_syn`

## 스플릿

| 스플릿 | 경로 | 용도 |
|---|---|---|
| 합성 학습 | `DREAM_to_DREAM_syn/panda_synth_train_dr` | 도메인 랜덤화 학습 (라벨) |
| 합성 검증 | `DREAM_to_DREAM_syn/panda_synth_test_dr` | 클린 val (angle MAE) |
| 합성 photo | `DREAM_to_DREAM_syn/panda_synth_test_photo` | 가림 벤치 베이스(RoboPEPP 프로토콜) |
| **real** | `DREAM_to_DREAM/{panda-3cam_azure, panda-3cam_kinect360, panda-3cam_realsense, panda-orb}` | 4개 SOTA 평가 스플릿 |

real 카메라 특성: azure(근거리 ~0.87m), kinect360, realsense(원거리 ~1.33m, 최난), orb(궤도 카메라, auto-bbox 붕괴 지점).

## 프레임 포맷 (Converted_dataset JSON)

각 프레임 = `NNNNNN.json` + `NNNNNN.rgb.jpg`. JSON 스키마:
```
meta.image_path         → 상대 이미지 경로 (../dataset/... 문자열 해석 주의)
meta.K                  → 3×3 카메라 내부 파라미터 (원본 해상도)
objects[0].keypoints[i]:
  .name                 → panda_link0 / link2 / ... / hand
  .location             → 3D 키포인트, 카메라 프레임 (m). GT extrinsic 별도 없음
  .projected_location   → 2D 픽셀 (원본 해상도)
sim_state.joints[:7]    → GT 관절각 (rad). J7은 nonzero지만 모델은 J7=0 고정
```

## 키포인트 (7개)

`link0, link2, link3, link4, link6, link7, hand` → Panda FK all_transforms 인덱스 `[0,2,3,4,6,7,9]`.
- link0/link2는 로봇 base Z축 위(xy≈0) → **J0 회전에 불변** (J0 base-yaw 모호성의 근원, PnP가 흡수).

## Dataset 로더 반환 필드

`Eval/inference_4tier_eval.py::EvalDataset` (정렬+stride, 결정적) / `TRAIN/dataset.py::PoseEstimationDataset` (비정렬, 학습용):
```
image (3,S,S) 정규화 · gt_2d/keypoints (7,2) @IS · gt_3d/keypoints_3d (7,3) 카메라 프레임
found/valid_mask (7) · camera_K (3,3) 원본해상도 · original_size (W,H) · gt_angles/angles (7) · name
```
- **K 스케일링 필수**: 원본 해상도 K → 512 해상도 (`refine_eval.py::scale_K`). 모든 PnP/투영에서.
- `EvalDataset`(정렬+stride)은 기계 간 결정적, `PoseEstimationDataset`(listdir 순서)은 비결정적 → 재현 비교 시 EvalDataset 사용.

## 주의

- 대용량, NAS 유일본, git 미추적 — 삭제 금지. 지도: [../../datasets.md](../../datasets.md).
- 체크포인트도 gitignored, 로컬 `TRAIN/outputs_*` 미러 (GPU 서버에서 rsync).
