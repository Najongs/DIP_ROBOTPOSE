# 5_apps — 응용 파이프라인

## collision_risk_pipeline (활성)

목적: 정적 이미지에서 로봇-사람 충돌 위험 점수. 상세는 프로젝트 자체 [README](../5_apps/collision_risk_pipeline/README.md) 참조.

4단계: SegFormer 로봇 마스크 → YOLO 사람 마스크 → 두 마스크 최소 픽셀거리 → 정적 위험점수. 기본은 의도적으로 2D·고속, depth는 optional.

### 실행 (`/home/najo/NAS/DIP`에서)

```bash
python 5_apps/collision_risk_pipeline/static_pipeline.py \
  --image /home/najo/NAS/DIP/4_perception/Fr5_robot_SegFormer/fr5_2.jpeg \
  --robot-checkpoint /home/najo/NAS/DIP/4_perception/Fr5_robot_SegFormer/best_segformer_robot_arm.pth \
  --human-model yolov8n-seg.pt \
  --out-dir 5_apps/collision_risk_pipeline/outputs
```

출력: `overlay.png`(마스크+최근접 거리선), `robot_mask.png`, `human_mask.png`, `result.json`(distance, overlap, risk score, time)

### 구성

- `mask_distance.py` — 마스크 정리, 최소 픽셀거리, `risk_from_distance_px()` (≤danger_px→1.0, ≥caution_px→0.0, 선형보간), 오버레이
- `static_pipeline.py` — SegFormer/YOLO 예측기 + CLI (`--danger-px 20 --caution-px 80`)
- `depth_geometry.py` — depth map+카메라 행렬이 있을 때 metric 3D 거리 (`mask_to_pointcloud`, `minimum_pointcloud_distance`)

### 의존성

- `4_perception/Fr5_robot_SegFormer/best_segformer_robot_arm.pth` (로봇 마스크)
- 루트 `yolov8n-seg.pt` (사람 마스크 — ultralytics가 없으면 자동 다운로드)

### 확장 방향 (README 제안)

- 시계열: `distance_t`, `delta_distance`, `closing_speed`, `time_to_contact`
- metric depth(ZED)가 있으면 `depth_geometry.py` 사용 (`stride=4~8` 권장), monocular depth는 캘리브 전엔 상대 증거로만
- 참고: DINObotPose3의 `Collision/` 데모는 같은 문제의 다른 접근(FK capsule + Mask R-CNN, 2D/3D) — [3_pose_models.md](3_pose_models.md) 참조
