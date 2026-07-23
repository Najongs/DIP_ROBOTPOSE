"""KUKA iiwa7 nvdiffrast silhouette rendering — RoboPEPP `urdfs/iiwa_description` meshes
(read-only external clone) posed by the URDF FK. Mirrors baxter_render.py: reuses
render_nvdr.NVDRSilhouette's projection/rasterize by swapping in the iiwa7 mesh buffer +
iiwa7 per-link transforms.

WHY A SEPARATE FK HERE (gauge note — read before touching):
`model_v4.iiwa7_forward_kinematics` is a data-FIT chain. Its link *origins* reproduce the DREAM
kuka keypoints to 0.003 mm, but its intermediate frame *orientations* are gauge (the fit residual
is Kabsch-aligned and position-only), so they are NOT URDF-canonical — model_v4 itself flags this
("revisit before mesh render-and-compare"). Meshes must hang off canonical frames, so this module
runs the URDF chain verbatim.

Verified (Eval/iiwa7_render.py --self-test):
  * URDF joint origins fed with the raw DREAM `sim_state` joint angles — no sign flips, no zero
    offsets — reproduce the DREAM kuka link_1..7 locations to 0.0029 mm RMS / 0.016 mm max on
    kuka_synth_test_dr (300 frames). The one free parameter the fit wanted (joint-1 zero, 92.7 deg)
    is a pure base yaw that Kabsch absorbs, hence set to 0 here.
  * URDF-FK vs model_v4 fitted-FK are per-frame congruent to 27 um, so the two are
    interchangeable downstream of a Kabsch/PnP solve.
So: keypoints AND meshes both come from THIS chain inside the RC path -> self-consistent by
construction. The keypoint-path evaluators (kuka_add_eval.py) are untouched and keep using model_v4.

Link mesh i is placed at frame i (frame 0 = link_0 base, frame i = cumul after joint i), plus the
URDF <visual><origin> offset baked into the vertices. 8 meshes, 7 keypoints (link_1..7).
"""
import os, math, numpy as np, torch

# read-only: external clone, has its own .git — never write here
IIWA_DESC = '/home/najo/NAS/DIP/RoboPEPP/urdfs/iiwa_description'

# iiwa7.urdf joint origins (rpy, xyz), axis z, joint_1..7 — verbatim from the URDF
_IIWA7_URDF_JOINTS = [
    ((0.0,             0.0,             0.0),             (0.0, 0.0,     0.15)),
    (( 1.57079632679,  0.0,             3.14159265359),   (0.0, 0.0,     0.19)),
    (( 1.57079632679,  0.0,             3.14159265359),   (0.0, 0.21,    0.0)),
    (( 1.57079632679,  0.0,             0.0),             (0.0, 0.0,     0.19)),
    ((-1.57079632679,  3.14159265359,   0.0),             (0.0, 0.21,    0.0)),
    (( 1.57079632679,  0.0,             0.0),             (0.0, 0.06070, 0.19)),
    ((-1.57079632679,  3.14159265359,   0.0),             (0.0, 0.081,   0.06070)),
]
# iiwa7.urdf <visual><origin> per link_0..7 (all rpy 0; z-shims only)
_IIWA7_VISUAL_XYZ = [(0., 0., 0.), (0., 0., 0.0075), (0., 0., 0.), (0., 0., -0.026),
                     (0., 0., 0.), (0., 0., -0.026), (0., 0., 0.), (0., 0., -0.0005)]
NUM_FRAMES = 8   # link_0 base + link_1..7


def _T(rpy, xyz, device, dtype):
    rx, ry, rz = rpy
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry),
                              math.sin(ry), math.cos(rz), math.sin(rz))
    R = torch.tensor([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                      [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                      [-sy,   cy*sx,          cy*cx]], device=device, dtype=dtype)
    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = torch.tensor(xyz, device=device, dtype=dtype)
    return T


def iiwa7_all_link_transforms(theta):
    """theta: (B,7) rad -> (B,8,4,4) per-link base-frame transforms.
    frame 0 = link_0 (base, identity); frame i = cumul after joint i (link_i)."""
    B = theta.shape[0]
    device, dtype = theta.device, theta.dtype
    fixed = [_T(rp, xy, device, dtype) for rp, xy in _IIWA7_URDF_JOINTS]
    cumul = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    frames = [cumul.clone()]                                    # link_0
    for i in range(7):
        Rz = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        c, s = torch.cos(theta[:, i]), torch.sin(theta[:, i])
        Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s; Rz[:, 1, 0] = s; Rz[:, 1, 1] = c
        cumul = cumul @ fixed[i].unsqueeze(0) @ Rz
        frames.append(cumul.clone())
    return torch.stack(frames, dim=1)                           # (B,8,4,4)


def iiwa7_urdf_forward_kinematics(theta):
    """theta: (B,7) rad -> (B,7,3) link_1..7 origins in the URDF base frame. Same keypoint set as
    model_v4.iiwa7_forward_kinematics (DREAM `iiwa7_link_1..7`), canonical gauge."""
    return iiwa7_all_link_transforms(theta)[:, 1:, :3, 3]


def iiwa7_load_link_meshes(device, kind='visual'):
    """Load link_0..7 STLs into one concatenated buffer with the URDF visual origin baked in.
    slices: (frame_idx, lo, hi)."""
    import trimesh
    verts, faces, slices, off = [], [], [], 0
    for i in range(NUM_FRAMES):
        p = os.path.join(IIWA_DESC, 'meshes', 'iiwa7', kind, f'link_{i}.stl')
        m = trimesh.load(p, force='mesh', process=False)
        v = np.asarray(m.vertices, dtype=np.float32) + np.asarray(_IIWA7_VISUAL_XYZ[i], dtype=np.float32)
        f = np.asarray(m.faces, dtype=np.int64)
        verts.append(v); faces.append(f + off)
        slices.append((i, off, off + len(v))); off += len(v)
    return {
        'verts': torch.from_numpy(np.concatenate(verts)).to(device),
        'faces': torch.from_numpy(np.concatenate(faces).astype(np.int32)).to(device),
        'slices': slices,
    }


def make_iiwa7_renderer(device, kind='visual'):
    """NVDRSilhouette with the iiwa7 mesh buffer swapped in. Use
    .robot_verts(theta, iiwa7_all_link_transforms) then rdr(pts, R, t, K, H, img_size)."""
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from render_nvdr import NVDRSilhouette
    import nvdiffrast.torch as dr
    rdr = NVDRSilhouette.__new__(NVDRSilhouette)
    rdr.dr = dr
    rdr.ctx = dr.RasterizeCudaContext(device=device)
    rdr.mesh = iiwa7_load_link_meshes(device, kind)
    rdr.near, rdr.far = 0.05, 20.0
    rdr.kind = kind
    return rdr


if __name__ == '__main__':
    # --self-test: re-derive the gauge claim in the docstring from the DREAM data itself.
    import argparse, glob, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/home/najo/NAS/DIP/datasets/synthetic/kuka_synth_test_dr')
    ap.add_argument('--n', type=int, default=300)
    args = ap.parse_args()

    TH, GT = [], []
    for f in sorted(glob.glob(os.path.join(args.data, '*.json')))[:args.n]:
        d = json.load(open(f))
        kps = {k['name']: k for k in d['objects'][0]['keypoints']}
        js = {j['name'].split('/')[-1]: j.get('position', 0) for j in d.get('sim_state', {}).get('joints', [])}
        try:
            TH.append([js[f'iiwa7_joint_{i}'] for i in range(1, 8)])
            GT.append(np.array([kps[f'iiwa7_link_{i}']['location'] for i in range(1, 8)]) / 100.0)
        except KeyError:
            continue
    TH = torch.tensor(np.array(TH), dtype=torch.float64); GT = np.array(GT)
    fk = iiwa7_urdf_forward_kinematics(TH).numpy()

    def kab(A, B):
        ca, cb = A.mean(0), B.mean(0)
        U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        return (A - ca) @ (Vt.T @ np.diag([1, 1, d]) @ U.T).T + cb

    e = np.array([np.linalg.norm(kab(fk[k], GT[k]) - GT[k], axis=-1) for k in range(len(fk))])
    print(f"URDF-FK vs DREAM kuka GT ({len(fk)} frames, Kabsch-aligned):")
    print(f"  RMS {np.sqrt((e**2).mean())*1000:.4f} mm   max {e.max()*1000:.4f} mm")
    for i in range(7):
        print(f"  link_{i+1}: mean {e[:, i].mean()*1000:.4f} mm  max {e[:, i].max()*1000:.4f} mm")
    assert e.max() * 1000 < 0.5, "URDF FK does not reproduce DREAM kuka keypoints"

    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
    from model_v4 import iiwa7_forward_kinematics as FIT
    ff = FIT(TH).numpy()
    c = np.array([np.linalg.norm(kab(fk[k], ff[k]) - ff[k], axis=-1).max() for k in range(len(fk))])
    print(f"URDF-FK vs model_v4 fitted-FK, per-frame congruence: max {c.max()*1e6:.1f} um")

    m = iiwa7_load_link_meshes('cpu')
    print(f"mesh buffer: {m['verts'].shape[0]} verts / {m['faces'].shape[0]} faces, "
          f"bounds {np.round(m['verts'].numpy().min(0), 3)}..{np.round(m['verts'].numpy().max(0), 3)} m")
