from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PointCloudDistanceResult:
    distance_m: float
    robot_point_3d: Optional[np.ndarray]
    human_point_3d: Optional[np.ndarray]
    robot_points: int
    human_points: int


def mask_to_pointcloud(
    mask: np.ndarray,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    stride: int = 4,
    min_depth_m: float = 0.05,
    max_depth_m: float = 10.0,
) -> np.ndarray:
    """Back-project masked depth pixels into camera-coordinate 3D points."""
    if mask.shape[:2] != depth_m.shape[:2]:
        raise ValueError(f"mask/depth shape mismatch: {mask.shape[:2]} vs {depth_m.shape[:2]}")

    valid = (mask > 0) & np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)

    if stride > 1:
        xs = xs[::stride]
        ys = ys[::stride]

    z = depth_m[ys, xs].astype(np.float32)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def minimum_pointcloud_distance(
    robot_points: np.ndarray,
    human_points: np.ndarray,
    chunk_size: int = 2048,
) -> PointCloudDistanceResult:
    """Compute nearest 3D distance between two sampled point clouds."""
    if len(robot_points) == 0 or len(human_points) == 0:
        return PointCloudDistanceResult(
            distance_m=float("inf"),
            robot_point_3d=None,
            human_point_3d=None,
            robot_points=int(len(robot_points)),
            human_points=int(len(human_points)),
        )

    best_dist_sq = float("inf")
    best_robot = None
    best_human = None
    human = human_points.astype(np.float32)

    for start in range(0, len(robot_points), chunk_size):
        robot = robot_points[start : start + chunk_size].astype(np.float32)
        diff = robot[:, None, :] - human[None, :, :]
        dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
        flat_idx = int(np.argmin(dist_sq))
        local_best = float(dist_sq.reshape(-1)[flat_idx])

        if local_best < best_dist_sq:
            row, col = np.unravel_index(flat_idx, dist_sq.shape)
            best_dist_sq = local_best
            best_robot = robot_points[start + row].copy()
            best_human = human_points[col].copy()

    return PointCloudDistanceResult(
        distance_m=float(np.sqrt(best_dist_sq)),
        robot_point_3d=best_robot,
        human_point_3d=best_human,
        robot_points=int(len(robot_points)),
        human_points=int(len(human_points)),
    )

