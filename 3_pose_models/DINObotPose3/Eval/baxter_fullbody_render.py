"""Canonical Baxter full-body differentiable silhouette renderer.

The 17 frames are ordered like the DREAM keypoints:
  torso_t0,
  left_s0..left_w2, left_hand,
  right_s0..right_w2, right_hand.

Unlike the position-fitted FK in model_v4.py, these transforms come from the official
Baxter URDF and therefore have meaningful link orientations suitable for attaching meshes.
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import torch

HERE = os.path.dirname(__file__)
BAXTER_DESC = os.path.abspath(os.path.join(
    HERE, "../../../RoboPEPP/urdfs/Baxter/baxter_description"
))
EE_DESC = os.path.abspath(os.path.join(
    HERE, "../../../_assets_src/baxter_common/rethink_ee_description"
))

KP17 = [
    "torso_t0",
    "left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2", "left_hand",
    "right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1", "right_w2", "right_hand",
]
ANG12 = [
    "left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1",
    "right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1",
]

# parent torso -> arm mount
_LEFT_MOUNT = ((0.0, 0.0, 0.7854), (0.024645, 0.219645, 0.118588))
_RIGHT_MOUNT = ((0.0, 0.0, -0.7854), (0.024645, -0.219645, 0.118588))

# Joint origin (rpy, xyz), copied from baxter.urdf. Axis is +z for every revolute joint.
_LEFT_JOINTS = [
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-math.pi / 2, 0.0, 0.0), (0.069, 0.0, 0.27035)),
    ((math.pi / 2, 0.0, math.pi / 2), (0.102, 0.0, 0.0)),
    ((-math.pi / 2, -math.pi / 2, 0.0), (0.069, 0.0, 0.26242)),
    ((math.pi / 2, 0.0, math.pi / 2), (0.10359, 0.0, 0.0)),
    ((-math.pi / 2, -math.pi / 2, 0.0), (0.01, 0.0, 0.2707)),
    ((math.pi / 2, 0.0, math.pi / 2), (0.115975, 0.0, 0.0)),
]
_RIGHT_JOINTS = [
    ((0.0, 0.0, 0.0), (0.055695, 0.0, 0.011038)),
    *_LEFT_JOINTS[1:],
]
_HAND = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.11355))

# Fixed placement of the canonical URDF arm gauges into the DREAM 17kp common gauge.
# Fitted once over 2,000 random poses against model_v4.baxter_forward_kinematics.
# Residual: left 0.0039mm RMS, right 0.0052mm RMS. The shared rotation preserves
# canonical mesh orientations; the two translations encode DREAM's arm-base convention.
_DREAM_R = (
    (-0.89817865, -0.02101528, -0.43912808),
    (0.43820545, -0.12315938, -0.89039752),
    (-0.03537078, -0.99216436, 0.11982814),
)
_DREAM_T_LEFT = (0.33508837, 0.06407628, 0.56002011)
_DREAM_T_RIGHT = (0.37613464, 0.06149459, 0.59916325)

_ARM_MESHES = [
    ("upper_shoulder/S0.DAE", 0),
    ("lower_shoulder/S1.DAE", 1),
    ("upper_elbow/E0.DAE", 2),
    ("lower_elbow/E1.DAE", 3),
    ("upper_forearm/W0.DAE", 4),
    ("lower_forearm/W1.DAE", 5),
    ("wrist/W2.DAE", 6),
]


def _fixed_transform(rpy, xyz, device, dtype):
    rx, ry, rz = rpy
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    out = torch.eye(4, device=device, dtype=dtype)
    out[:3, :3] = torch.tensor([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ], device=device, dtype=dtype)
    out[:3, 3] = torch.tensor(xyz, device=device, dtype=dtype)
    return out


def _rz(theta):
    batch = theta.shape[0]
    out = torch.eye(4, device=theta.device, dtype=theta.dtype).unsqueeze(0).repeat(batch, 1, 1)
    c, s = torch.cos(theta), torch.sin(theta)
    out[:, 0, 0] = c
    out[:, 0, 1] = -s
    out[:, 1, 0] = s
    out[:, 1, 1] = c
    return out


def _arm_frames(theta7, mount, joints):
    batch = theta7.shape[0]
    device, dtype = theta7.device, theta7.dtype
    current = _fixed_transform(*mount, device, dtype).unsqueeze(0).repeat(batch, 1, 1)
    frames = []
    for index, joint in enumerate(joints):
        current = current @ _fixed_transform(*joint, device, dtype).unsqueeze(0) @ _rz(theta7[:, index])
        frames.append(current)
    current = current @ _fixed_transform(*_HAND, device, dtype).unsqueeze(0)
    frames.append(current)
    return torch.stack(frames, dim=1)


def baxter_fullbody_all_link_transforms(theta12):
    """theta12 -> (B,17,4,4), with both unobservable w2 angles fixed to zero."""
    batch = theta12.shape[0]
    zero = torch.zeros(batch, 1, device=theta12.device, dtype=theta12.dtype)
    left = _arm_frames(torch.cat((theta12[:, :6], zero), dim=1), _LEFT_MOUNT, _LEFT_JOINTS)
    right = _arm_frames(torch.cat((theta12[:, 6:12], zero), dim=1), _RIGHT_MOUNT, _RIGHT_JOINTS)
    align_left = torch.eye(4, device=theta12.device, dtype=theta12.dtype)
    align_left[:3, :3] = torch.tensor(_DREAM_R, device=theta12.device, dtype=theta12.dtype)
    align_left[:3, 3] = torch.tensor(_DREAM_T_LEFT, device=theta12.device, dtype=theta12.dtype)
    align_right = torch.eye(4, device=theta12.device, dtype=theta12.dtype)
    align_right[:3, :3] = torch.tensor(_DREAM_R, device=theta12.device, dtype=theta12.dtype)
    align_right[:3, 3] = torch.tensor(_DREAM_T_RIGHT, device=theta12.device, dtype=theta12.dtype)
    left = align_left.view(1, 1, 4, 4) @ left
    right = align_right.view(1, 1, 4, 4) @ right
    torso = align_right.view(1, 1, 4, 4).repeat(batch, 1, 1, 1)
    return torch.cat((torso, left, right), dim=1)


def baxter_fullbody_urdf_fk(theta12):
    return baxter_fullbody_all_link_transforms(theta12)[..., :3, 3]


def _load_mesh(path):
    import trimesh
    mesh = trimesh.load(path, force="mesh", process=False)
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int64)


def baxter_fullbody_load_meshes(device, include_grippers=True):
    verts, faces, slices = [], [], []
    offset = 0

    def add(path, frame_index, local_rpy=(0.0, 0.0, 0.0), local_xyz=(0.0, 0.0, 0.0)):
        nonlocal offset
        vertices, triangles = _load_mesh(path)
        local = _fixed_transform(local_rpy, local_xyz, torch.device("cpu"), torch.float64).numpy()
        vertices = (vertices @ local[:3, :3].T + local[:3, 3]).astype(np.float32)
        verts.append(vertices)
        faces.append(triangles + offset)
        slices.append((frame_index, offset, offset + len(vertices)))
        offset += len(vertices)

    mesh_root = os.path.join(BAXTER_DESC, "meshes")
    add(os.path.join(mesh_root, "torso/base_link.DAE"), 0)
    for side_offset in (1, 9):
        for relative, arm_index in _ARM_MESHES:
            add(os.path.join(mesh_root, relative), side_offset + arm_index)

    if include_grippers:
        gripper = os.path.join(
            EE_DESC, "meshes/electric_gripper/electric_gripper_w_fingers.DAE"
        )
        # hand -> gripper_base is +25mm z; official visual frame is Rx(-90) Ry(180).
        add(gripper, 8, (-math.pi / 2, math.pi, 0.0), (0.0, 0.0, 0.025))
        add(gripper, 16, (-math.pi / 2, math.pi, 0.0), (0.0, 0.0, 0.025))

    return {
        "verts": torch.from_numpy(np.concatenate(verts)).to(device),
        "faces": torch.from_numpy(np.concatenate(faces).astype(np.int32)).to(device),
        "slices": slices,
    }


def make_baxter_fullbody_renderer(device, include_grippers=True):
    from render_nvdr import NVDRSilhouette
    import nvdiffrast.torch as dr

    renderer = NVDRSilhouette.__new__(NVDRSilhouette)
    renderer.dr = dr
    renderer.ctx = dr.RasterizeCudaContext(device=device)
    renderer.mesh = baxter_fullbody_load_meshes(device, include_grippers)
    renderer.near, renderer.far = 0.05, 20.0
    renderer.kind = "baxter_fullbody"
    return renderer


def _kabsch(source, target):
    source_center, target_center = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    correction = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, correction]) @ u.T
    return (source - source_center) @ rotation.T + target_center


def self_test(data_dir, max_frames=200):
    errors = []
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))[:max_frames]
    for path in files:
        data = json.load(open(path))
        joints = {j["name"].split("/")[-1]: j.get("position", 0.0) for j in data["sim_state"]["joints"]}
        keypoints = {k["name"]: k for k in data["objects"][0]["keypoints"]}
        try:
            theta = torch.tensor([[joints[name] for name in ANG12]], dtype=torch.float64)
            gt = np.asarray([keypoints[name]["location"] for name in KP17], np.float64) / 100.0
        except KeyError:
            continue
        fk = baxter_fullbody_urdf_fk(theta)[0].numpy()
        errors.append(np.linalg.norm(_kabsch(fk, gt) - gt, axis=1))
    if not errors:
        raise RuntimeError("No compatible Baxter samples found")
    errors = np.asarray(errors) * 1000.0
    print(f"canonical URDF FK vs DREAM 17kp ({len(errors)} frames, per-frame Kabsch)")
    print(f"RMS {np.sqrt(np.mean(errors ** 2)):.3f} mm | mean {errors.mean():.3f} mm | max {errors.max():.3f} mm")
    print("per-keypoint mean mm: " + ", ".join(
        f"{name}={errors[:, index].mean():.2f}" for index, name in enumerate(KP17)
    ))
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/najo/NAS/DIP/datasets/synthetic/baxter_synth_test_dr")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--render-smoke", action="store_true")
    args = parser.parse_args()
    self_test(args.data, args.max_frames)
    if args.render_smoke:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for nvdiffrast smoke test")
        renderer = make_baxter_fullbody_renderer("cuda")
        theta = torch.zeros(1, 12, device="cuda")
        vertices = renderer.robot_verts(theta, baxter_fullbody_all_link_transforms)
        print(f"render mesh: {vertices.shape[1]} vertices, {renderer.mesh['faces'].shape[0]} faces")
