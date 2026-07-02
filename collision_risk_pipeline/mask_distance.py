from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


@dataclass(frozen=True)
class MaskDistanceResult:
    distance_px: float
    robot_point: Optional[Point]
    human_point: Optional[Point]
    overlap_pixels: int
    robot_area: int
    human_area: int

    @property
    def has_overlap(self) -> bool:
        return self.overlap_pixels > 0


def as_bool_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert common binary/probability mask formats to a 2D bool mask."""
    if mask.ndim == 3:
        mask = mask.squeeze()
    if mask.dtype == np.bool_:
        return mask
    return mask > threshold


def clean_mask(mask: np.ndarray, min_area: int = 64, kernel_size: int = 3) -> np.ndarray:
    """Remove tiny regions and smooth a binary mask without changing its shape."""
    bool_mask = as_bool_mask(mask)
    mask_u8 = bool_mask.astype(np.uint8)

    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1

    return cleaned.astype(bool)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    bool_mask = as_bool_mask(mask)
    mask_u8 = bool_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return bool_mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def mask_boundary_points(mask: np.ndarray) -> np.ndarray:
    """Return boundary points as Nx2 int array in (x, y) order."""
    bool_mask = as_bool_mask(mask)
    if not np.any(bool_mask):
        return np.empty((0, 2), dtype=np.int32)

    mask_u8 = bool_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        ys, xs = np.nonzero(bool_mask)
        return np.stack([xs, ys], axis=1).astype(np.int32)

    points = np.concatenate([c.reshape(-1, 2) for c in contours], axis=0)
    return np.unique(points.astype(np.int32), axis=0)


def _chunked_nearest_pair(
    a_points: np.ndarray,
    b_points: np.ndarray,
    chunk_size: int = 2048,
) -> tuple[float, Point, Point]:
    best_dist_sq = float("inf")
    best_a: Point = (0, 0)
    best_b: Point = (0, 0)

    b_float = b_points.astype(np.float32)
    for start in range(0, len(a_points), chunk_size):
        a_chunk = a_points[start : start + chunk_size].astype(np.float32)
        diff = a_chunk[:, None, :] - b_float[None, :, :]
        dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
        flat_idx = int(np.argmin(dist_sq))
        local_best = float(dist_sq.reshape(-1)[flat_idx])

        if local_best < best_dist_sq:
            row, col = np.unravel_index(flat_idx, dist_sq.shape)
            best_dist_sq = local_best
            best_a = tuple(map(int, a_points[start + row]))
            best_b = tuple(map(int, b_points[col]))

    return float(np.sqrt(best_dist_sq)), best_a, best_b


def minimum_mask_distance(
    robot_mask: np.ndarray,
    human_mask: np.ndarray,
    clean: bool = True,
    min_area: int = 64,
) -> MaskDistanceResult:
    """Compute the minimum 2D pixel distance between robot and human mask areas."""
    robot = clean_mask(robot_mask, min_area=min_area) if clean else as_bool_mask(robot_mask)
    human = clean_mask(human_mask, min_area=min_area) if clean else as_bool_mask(human_mask)

    robot_area = int(robot.sum())
    human_area = int(human.sum())
    if robot_area == 0 or human_area == 0:
        return MaskDistanceResult(
            distance_px=float("inf"),
            robot_point=None,
            human_point=None,
            overlap_pixels=0,
            robot_area=robot_area,
            human_area=human_area,
        )

    overlap = robot & human
    overlap_pixels = int(overlap.sum())
    if overlap_pixels > 0:
        ys, xs = np.nonzero(overlap)
        idx = len(xs) // 2
        point = (int(xs[idx]), int(ys[idx]))
        return MaskDistanceResult(
            distance_px=0.0,
            robot_point=point,
            human_point=point,
            overlap_pixels=overlap_pixels,
            robot_area=robot_area,
            human_area=human_area,
        )

    robot_boundary = mask_boundary_points(robot)
    human_boundary = mask_boundary_points(human)
    distance_px, robot_point, human_point = _chunked_nearest_pair(robot_boundary, human_boundary)

    return MaskDistanceResult(
        distance_px=distance_px,
        robot_point=robot_point,
        human_point=human_point,
        overlap_pixels=0,
        robot_area=robot_area,
        human_area=human_area,
    )


def risk_from_distance_px(
    distance_px: float,
    danger_px: float = 20.0,
    caution_px: float = 80.0,
) -> float:
    """Map pixel distance to a simple static risk score in [0, 1]."""
    if not np.isfinite(distance_px):
        return 0.0
    if distance_px <= danger_px:
        return 1.0
    if distance_px >= caution_px:
        return 0.0
    return float((caution_px - distance_px) / (caution_px - danger_px))


def draw_distance_overlay(
    image_rgb: np.ndarray,
    robot_mask: np.ndarray,
    human_mask: np.ndarray,
    result: MaskDistanceResult,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = image_rgb.copy()
    robot = as_bool_mask(robot_mask)
    human = as_bool_mask(human_mask)

    color_layer = np.zeros_like(overlay)
    color_layer[robot] = (0, 180, 60)
    color_layer[human] = (240, 50, 50)

    blended = cv2.addWeighted(overlay, 1.0, color_layer, alpha, 0)
    overlay[robot | human] = blended[robot | human]

    if result.robot_point is not None and result.human_point is not None:
        cv2.line(overlay, result.robot_point, result.human_point, (255, 230, 0), 2)
        cv2.circle(overlay, result.robot_point, 5, (0, 255, 80), -1)
        cv2.circle(overlay, result.human_point, 5, (255, 80, 80), -1)

    label = f"distance={result.distance_px:.1f}px"
    if result.has_overlap:
        label = f"overlap={result.overlap_pixels}px"
    cv2.putText(overlay, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.putText(overlay, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 1)
    return overlay

