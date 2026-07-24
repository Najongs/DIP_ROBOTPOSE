#!/usr/bin/env python3
"""Render three lightweight FR5 body-only previews with matplotlib."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parent
MESH_DIR = ROOT / "meshes" / "fr5v6" / "visual"
OUT_DIR = ROOT / "preview"

# A non-singular inspection pose, in radians.
Q = np.deg2rad([25.0, -45.0, 60.0, -35.0, 45.0, 20.0])
JOINTS = [
    ("j1_Link", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("j2_Link", (0.0, 0.0, 0.152), (np.pi / 2, 0.0, 0.0)),
    ("j3_Link", (-0.425, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("j4_Link", (-0.39501, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("j5_Link", (0.0, 0.0, 0.1021), (np.pi / 2, 0.0, 0.0)),
    ("j6_Link", (0.0, 0.0, 0.102), (-np.pi / 2, 0.0, 0.0)),
]
COLORS = ["#343941", "#e9edf2", "#dce2e8", "#cbd3dc", "#e9edf2", "#cbd3dc", "#f28e2b"]


def rotation_rpy(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0), theta=0.0):
    t = np.eye(4)
    t[:3, :3] = rotation_rpy(rpy) @ rotation_rpy((0.0, 0.0, theta))
    t[:3, 3] = xyz
    return t


def posed_meshes():
    result = []
    world = np.eye(4)
    base = trimesh.load(MESH_DIR / "base_link.DAE", force="mesh", process=False)
    result.append((base, world.copy()))
    for theta, (name, xyz, rpy) in zip(Q, JOINTS):
        world = world @ transform(xyz, rpy, theta)
        mesh = trimesh.load(MESH_DIR / f"{name}.DAE", force="mesh", process=False)
        result.append((mesh, world.copy()))
    return result


def render(meshes, filename, elev, azim):
    size = 1400
    image = np.full((size, size, 3), (248, 246, 244), dtype=np.uint8)
    all_vertices = []
    posed = []
    for mesh, pose in meshes:
        vertices = trimesh.transform_points(mesh.vertices, pose)
        posed.append((vertices, mesh.faces))
        all_vertices.append(vertices)
    points = np.concatenate(all_vertices, axis=0)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    er, ar = np.deg2rad([elev, azim])
    camera_dir = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])
    forward = -camera_dir
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    centered = points - center
    projected_all = np.column_stack((centered @ right, centered @ up))
    span = np.ptp(projected_all, axis=0).max()
    scale = size * 0.78 / span
    light = np.array([0.35, -0.45, 0.82])
    light /= np.linalg.norm(light)

    triangles = []
    for index, (vertices, faces) in enumerate(posed):
        local = vertices - center
        xy = np.column_stack((local @ right, local @ up))
        xy = xy * scale + size / 2
        xy[:, 1] = size - xy[:, 1]
        depth = local @ forward
        tri3 = vertices[faces]
        normals = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(norm, 1e-12)
        shade = 0.55 + 0.45 * np.abs(normals @ light)
        base = np.array([int(COLORS[index][i:i + 2], 16) for i in (1, 3, 5)][::-1])
        for face, z, value in zip(faces, depth[faces].mean(axis=1), shade):
            color = tuple(int(x) for x in np.clip(base * value, 0, 255))
            triangles.append((float(z), xy[face].astype(np.int32), color))

    for _, polygon, color in sorted(triangles, key=lambda item: item[0], reverse=True):
        cv2.fillConvexPoly(image, polygon, color, lineType=cv2.LINE_AA)
    cv2.imwrite(str(OUT_DIR / filename), image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=("front", "side", "perspective"))
    args = parser.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    meshes = posed_meshes()
    views = {
        "front": ("fr5_front.png", 18, -90),
        "side": ("fr5_side.png", 18, 0),
        "perspective": ("fr5_perspective.png", 28, -45),
    }
    selected = [args.view] if args.view else list(views)
    for name in selected:
        render(meshes, *views[name])
        print(OUT_DIR / views[name][0])


if __name__ == "__main__":
    main()
