#!/usr/bin/env python3
"""Overlay the body-only FR5 v6 mesh on one converted FR5 GT sample."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from render_preview import JOINTS, MESH_DIR, rotation_rpy, transform


ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLE = (
    ROOT.parents[3]
    / "datasets/ICRA_multiview/Converted_dataset/fr5_val"
    / "zed_30779426_left_1748249284.558.json"
)


def load_posed_meshes(q):
    result = []
    world = np.eye(4)
    base = trimesh.load(MESH_DIR / "base_link.DAE", force="mesh", process=False)
    result.append((base, world.copy()))
    for theta, (name, xyz, rpy) in zip(q, JOINTS):
        world = world @ transform(xyz, rpy, theta)
        mesh = trimesh.load(MESH_DIR / f"{name}.DAE", force="mesh", process=False)
        result.append((mesh, world.copy()))
    return result


def dh_matrix(alpha, a, d, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fr5_fk(q):
    dh = [
        (np.pi / 2, 0.0, 0.152),
        (0.0, -0.425, 0.0),
        (0.0, -0.395, 0.0),
        (np.pi / 2, 0.0, 0.102),
        (-np.pi / 2, 0.0, 0.102),
        (0.0, 0.0, 0.100),
    ]
    pose = np.eye(4)
    points = [pose[:3, 3].copy()]
    for theta, (alpha, a, d) in zip(q, dh):
        pose = pose @ dh_matrix(alpha, a, d, theta)
        points.append(pose[:3, 3].copy())
    return np.asarray(points)


def kabsch(source, target):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out", type=Path, default=ROOT / "preview/fr5_data_overlay.png")
    args = parser.parse_args()

    data = json.loads(args.sample.read_text())
    meta = data["meta"]
    image = cv2.imread(meta["image_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(meta["image_path"])
    k = np.asarray(meta["K"], dtype=np.float64)
    q = np.asarray([joint["position"] for joint in data["sim_state"]["joints"][:6]])
    gt3d = np.asarray(
        [keypoint["location"] for keypoint in data["objects"][0]["keypoints"]],
        dtype=np.float64,
    )
    fk3d = fr5_fk(q)
    r_cam_base, t_cam_base = kabsch(fk3d, gt3d)
    fit_rms_mm = np.sqrt(np.mean(np.sum((fk3d @ r_cam_base.T + t_cam_base - gt3d) ** 2, axis=1))) * 1000

    layer = np.zeros_like(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    triangles = []
    light = np.array([-0.35, -0.35, -0.87])
    light /= np.linalg.norm(light)
    colors = [
        np.array((45, 49, 55)), np.array((220, 226, 232)),
        np.array((205, 214, 222)), np.array((190, 201, 211)),
        np.array((220, 226, 232)), np.array((190, 201, 211)),
        np.array((20, 130, 240)),
    ]

    for index, (mesh, pose_base_link) in enumerate(load_posed_meshes(q)):
        vertices_base = trimesh.transform_points(mesh.vertices, pose_base_link)
        vertices_cam = vertices_base @ r_cam_base.T + t_cam_base
        valid_z = vertices_cam[:, 2] > 1e-4
        uvw = vertices_cam @ k.T
        uv = uvw[:, :2] / uvw[:, 2:3]
        tri3 = vertices_cam[mesh.faces]
        normals = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        shades = 0.55 + 0.45 * np.abs(normals @ light)
        face_valid = valid_z[mesh.faces].all(axis=1)
        for face, z, shade in zip(
            mesh.faces[face_valid],
            vertices_cam[mesh.faces[face_valid], 2].mean(axis=1),
            shades[face_valid],
        ):
            polygon = np.rint(uv[face]).astype(np.int32)
            color = tuple(int(x) for x in np.clip(colors[index] * shade, 0, 255))
            triangles.append((float(z), polygon, color))

    for _, polygon, color in sorted(triangles, key=lambda item: item[0], reverse=True):
        cv2.fillConvexPoly(layer, polygon, color, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_AA)

    alpha = 0.48
    output = image.copy()
    covered = mask > 0
    output[covered] = (
        image[covered].astype(np.float32) * (1.0 - alpha)
        + layer[covered].astype(np.float32) * alpha
    ).astype(np.uint8)

    # Converted GT keypoints are drawn in green for a quick pose-coordinate check.
    for kp in data["objects"][0]["keypoints"]:
        u, v = np.rint(kp["projected_location"]).astype(int)
        cv2.circle(output, (u, v), 6, (40, 230, 70), -1, cv2.LINE_AA)
        cv2.circle(output, (u, v), 9, (15, 70, 20), 2, cv2.LINE_AA)

    cv2.putText(
        output, f"FR5v6 body mesh + GT keypoints (green), FK fit RMS {fit_rms_mm:.3f} mm",
        (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3, cv2.LINE_AA,
    )
    cv2.putText(
        output, f"FR5v6 body mesh + GT keypoints (green), FK fit RMS {fit_rms_mm:.3f} mm",
        (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 1, cv2.LINE_AA,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), output):
        raise RuntimeError(f"Failed to write {args.out}")
    print(args.out)


if __name__ == "__main__":
    main()
