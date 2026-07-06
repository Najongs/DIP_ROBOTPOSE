"""Batched, differentiable torch forward-kinematics for the 6-DOF robots, producing the SAME
DH keypoints as kinematics.py (the convention the GT labels + detector use). Mirrors
model_v4.panda_forward_kinematics (which returns Panda/FR3 robot-frame keypoints) so the
kinematic solver can be robot-parameterized without touching its PnP/refine core.

Meca500: standard DH (get_dh_matrix), 6 joints -> 7 keypoints [base, J1..J6]. No base correction.
MecaInsertion: identical DH but with a base_correction rotation (Rz(90)*Rx(180)).

Verified against kinematics.py numpy FK to <1e-5 m (see __main__).
"""
import math
import torch

# (alpha_deg, a, d, theta_offset_deg) per joint — copied from kinematics.Meca500Kinematics
_MECA500_DH = [
    (-90.0, 0.0,   0.135, 0.0),
    (0.0,   0.135, 0.0,   -90.0),
    (-90.0, 0.038, 0.0,   0.0),
    (90.0,  0.0,   0.120, 0.0),
    (-90.0, 0.0,   0.0,   0.0),
    (0.0,   0.0,   0.070, 0.0),
]


# (alpha_deg, a, d, theta_offset_deg) per joint — copied from kinematics.Fr5Kinematics
_FR5_DH = [
    (90.0,  0.0,    0.152, 0.0),
    (0.0,   -0.425, 0.0,   0.0),
    (0.0,   -0.395, 0.0,   0.0),
    (90.0,  0.0,    0.102, 0.0),
    (-90.0, 0.0,    0.102, 0.0),
    (0.0,   0.0,    0.100, 0.0),
]


def _dh_matrix(alpha_deg, a, d, theta):  # theta: (B,) rad tensor -> (B,4,4)
    B = theta.shape[0]; dev = theta.device; dt = theta.dtype
    ar = math.radians(alpha_deg); ca, sa = math.cos(ar), math.sin(ar)
    ct, st = torch.cos(theta), torch.sin(theta)
    T = torch.zeros(B, 4, 4, device=dev, dtype=dt)
    T[:, 0, 0] = ct;       T[:, 0, 1] = -st * ca;  T[:, 0, 2] = st * sa;   T[:, 0, 3] = a * ct
    T[:, 1, 0] = st;       T[:, 1, 1] = ct * ca;   T[:, 1, 2] = -ct * sa;  T[:, 1, 3] = a * st
    T[:, 2, 1] = sa;       T[:, 2, 2] = ca;        T[:, 2, 3] = d
    T[:, 3, 3] = 1.0
    return T


def _meca_fk(theta, dh, base_R=None):  # theta (B,6) rad -> (B,7,3)
    B = theta.shape[0]; dev = theta.device; dt = theta.dtype
    T = torch.eye(4, device=dev, dtype=dt).unsqueeze(0).repeat(B, 1, 1)
    if base_R is not None:
        T[:, :3, :3] = torch.as_tensor(base_R, device=dev, dtype=dt).unsqueeze(0)
    pts = [T[:, :3, 3].clone()]  # base keypoint (0,0,0) [+ base_R has no translation]
    for i, (alpha, a, d, toff) in enumerate(dh):
        th = theta[:, i] + math.radians(toff)
        T = T @ _dh_matrix(alpha, a, d, th)
        pts.append(T[:, :3, 3].clone())
    return torch.stack(pts, dim=1)  # (B,7,3)


def meca500_forward_kinematics(theta):
    """theta: (B,6) radians -> (B,7,3) base-frame keypoints (matches Meca500Kinematics)."""
    return _meca_fk(theta, _MECA500_DH, base_R=None)


# base_correction = Rz(90) * Rx(180), computed once as a constant matrix
def _meca_insertion_base_R():
    # Rx(180): diag(1,-1,-1); Rz(90): [[0,-1,0],[1,0,0],[0,0,1]]
    Rx = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    Rz = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    return [[sum(Rz[i][k] * Rx[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def meca_insertion_forward_kinematics(theta):
    return _meca_fk(theta, _MECA500_DH, base_R=_meca_insertion_base_R())


def fr5_forward_kinematics(theta):
    """theta: (B,6) radians -> (B,7,3) base-frame keypoints (matches Fr5Kinematics, identity base).
    view_rotations are multiview-training only; the single-image solver estimates camera R itself."""
    return _meca_fk(theta, _FR5_DH, base_R=None)


if __name__ == '__main__':
    import numpy as np, sys
    sys.path.insert(0, '/home/najo/NAS/DIP/4_perception/DINOv3_fine_tunning')
    from kinematics import Meca500Kinematics, MecaInsertionKinematics, Fr5Kinematics
    torch.manual_seed(0)
    cases = [('Meca500', Meca500Kinematics(), meca500_forward_kinematics, None),
             ('MecaInsertion', MecaInsertionKinematics(), meca_insertion_forward_kinematics, None),
             ('Fr5', Fr5Kinematics(), fr5_forward_kinematics, 'none')]  # 'none' -> identity base
    for name, kin, fk, view in cases:
        th = (torch.rand(8, 6) * 2 - 1) * 2.0
        got = fk(th.double()).numpy()
        exp = np.stack([(kin.forward_kinematics(th[b].numpy(), view=view) if view else
                         kin.forward_kinematics(th[b].numpy())) for b in range(8)])
        err = np.abs(got - exp).max()
        print(f'{name:14s} max abs err vs numpy FK = {err:.2e} m  ({"OK" if err < 1e-5 else "MISMATCH"})')
