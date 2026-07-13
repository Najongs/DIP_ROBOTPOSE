"""Baxter left-arm nvdiffrast silhouette rendering — baxter_common meshes (== DREAM baxter model,
verified 0.00mm) posed by the URDF FK. Reuses render_nvdr.NVDRSilhouette's projection/rasterize
by swapping in the baxter mesh buffer + baxter per-link transforms.

Link mesh i (S0..W2) is placed at FK frame i = cumul after joint i (visual origin = identity in
the baxter URDF). 7 links, 7 keypoints (left_s0..w2)."""
import os, math, numpy as np, torch

VIS = os.path.join(os.path.dirname(__file__), '../ViS/Baxter')
_MESH = ['S0', 'S1', 'E0', 'E1', 'W0', 'W1', 'W2']
# URDF left-arm joint origins (rpy, xyz), axis z
_BAXTER_JOINTS = [
    ((0, 0, 0),                         (0.055695, 0, 0.011038)),
    ((-math.pi/2, 0, 0),                (0.069, 0, 0.27035)),
    ((math.pi/2, 0, math.pi/2),         (0.102, 0, 0)),
    ((-math.pi/2, -math.pi/2, 0),       (0.069, 0, 0.26242)),
    ((math.pi/2, 0, math.pi/2),         (0.10359, 0, 0)),
    ((-math.pi/2, -math.pi/2, 0),       (0.01, 0, 0.2707)),
    ((math.pi/2, 0, math.pi/2),         (0.115975, 0, 0)),
]


def _T(rpy, xyz, device, dtype):
    rx, ry, rz = rpy
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    R = torch.tensor([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                      [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                      [-sy,   cy*sx,          cy*cx]], device=device, dtype=dtype)
    T = torch.eye(4, device=device, dtype=dtype); T[:3, :3] = R; T[:3, 3] = torch.tensor(xyz, device=device, dtype=dtype)
    return T


def baxter_all_link_transforms(theta):
    """theta: (B,7) rad -> (B,7,4,4) per-link world(base-frame) transforms (cumul after each joint)."""
    B = theta.shape[0]; device, dtype = theta.device, theta.dtype
    fixed = [_T(rp, xy, device, dtype) for rp, xy in _BAXTER_JOINTS]
    cumul = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    frames = []
    for i in range(7):
        Rz = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        c, s = torch.cos(theta[:, i]), torch.sin(theta[:, i])
        Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s; Rz[:, 1, 0] = s; Rz[:, 1, 1] = c
        cumul = cumul @ fixed[i].unsqueeze(0) @ Rz
        frames.append(cumul.clone())
    return torch.stack(frames, dim=1)   # (B,7,4,4)


def baxter_load_link_meshes(device, cap=4000):
    """Load 7 DAE arm-link meshes into one concatenated buffer. slices: (frame_idx, lo, hi)."""
    import trimesh
    verts, faces, slices, off = [], [], [], 0
    for i, name in enumerate(_MESH):
        m = trimesh.load(os.path.join(VIS, 'meshes', name + '.DAE'), force='mesh', process=False)
        v = np.asarray(m.vertices, dtype=np.float32); f = np.asarray(m.faces, dtype=np.int64)
        verts.append(v); faces.append(f + off)
        slices.append((i, off, off + len(v))); off += len(v)
    return {
        'verts': torch.from_numpy(np.concatenate(verts)).to(device),
        'faces': torch.from_numpy(np.concatenate(faces).astype(np.int32)).to(device),
        'slices': slices,
    }


def make_baxter_renderer(device):
    """NVDRSilhouette with the baxter mesh buffer swapped in. Use .robot_verts(theta,
    baxter_all_link_transforms) then rdr(pts,R,t,K,H,img_size) for the soft silhouette."""
    from render_nvdr import NVDRSilhouette
    rdr = NVDRSilhouette.__new__(NVDRSilhouette)
    import nvdiffrast.torch as dr
    rdr.dr = dr
    rdr.ctx = dr.RasterizeCudaContext(device=device)
    rdr.mesh = baxter_load_link_meshes(device)
    rdr.near, rdr.far = 0.05, 20.0
    rdr.kind = 'baxter'
    return rdr
