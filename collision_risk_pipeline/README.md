# Robot-Human Collision Risk Pipeline

This folder is a cleaned-up starting point for the static-image version of the idea:

1. Predict the robot area with the trained SegFormer checkpoint.
2. Predict the human area with a YOLO segmentation model.
3. Compute the minimum distance between the two binary mask areas.
4. Convert that distance to a simple static risk score.

The current default is intentionally 2D and fast. Depth is kept optional because monocular depth models can add latency and may not provide metric distance without calibration or scaling.

## Files

- `mask_distance.py`: pure mask cleanup, minimum pixel distance, static risk score, overlay drawing.
- `static_pipeline.py`: SegFormer robot predictor, YOLO person predictor, command-line runner.
- `depth_geometry.py`: optional utilities for metric 3D distance if a depth map and camera matrix are available.

## Static Image Run

From `/home/najo/NAS/DIP`:

```bash
python collision_risk_pipeline/static_pipeline.py \
  --image /home/najo/NAS/DIP/Fr5_robot_SegFormer/fr5_2.jpeg \
  --robot-checkpoint /home/najo/NAS/DIP/Fr5_robot_SegFormer/best_segformer_robot_arm.pth \
  --human-model yolov8n-seg.pt \
  --out-dir collision_risk_pipeline/outputs
```

Outputs:

- `overlay.png`: original image with robot mask, human mask, and nearest-distance line.
- `robot_mask.png`: predicted robot binary mask.
- `human_mask.png`: combined person binary mask.
- `result.json`: distance, overlap, area, risk score, and inference time.

## Current Risk Rule

`risk_from_distance_px()` maps pixel distance to `[0, 1]`.

- `distance <= danger_px`: risk `1.0`
- `distance >= caution_px`: risk `0.0`
- between them: linearly interpolated

This is only a static-image heuristic. For time-ordered data, keep the same per-frame area distance, then add temporal features:

- `distance_t`
- `delta_distance = distance_t - distance_t_minus_1`
- `closing_speed = max(0, -delta_distance / delta_time)`
- `time_to_contact = distance_t / closing_speed`

## Depth Direction

Use depth only if you can get a depth map with acceptable latency and stable scale.

Recommended order:

1. Start with 2D mask distance and validate false positives/false negatives.
2. If ZED depth or another metric depth source is available, use `depth_geometry.py`.
3. If only monocular depth is available, treat it as relative risk evidence, not metric collision distance, unless calibrated.

With metric depth:

```python
from depth_geometry import mask_to_pointcloud, minimum_pointcloud_distance

robot_points = mask_to_pointcloud(robot_mask, depth_m, camera_matrix, stride=4)
human_points = mask_to_pointcloud(human_mask, depth_m, camera_matrix, stride=4)
distance_3d = minimum_pointcloud_distance(robot_points, human_points)
```

The `stride` parameter is important for speed. Dense mask point clouds can be slow, so start with `stride=4` or `stride=8`.

