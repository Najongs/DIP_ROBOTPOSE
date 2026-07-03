"""
High-fidelity differentiable Panda silhouette via nvdiffrast (CUDA rasterizer, no OpenGL).

Replaces the bilinear-splat `render_mesh` in silhouette_mesh_probe.py. The splat renderer was
self-consistent-only: matching it against a TRUE robot mask (SAM) diverged because the splatted
(bulky/holey) silhouette never matches real robot pixels (IoU capped 0.24-0.36). Rasterizing the
actual visual meshes with faces closes that fidelity gap so a true segmenter mask becomes a usable
render-and-compare target.

Gradients: dr.antialias on a constant-white attribute gives silhouette-edge gradients w.r.t. vertex
positions -> flows back to (theta, R, t) through the FK/link transforms. Area ~ 1/z^2 stays the
depth cue, same as the validated splat probe (+0.108 oracle ceiling on realsense).

Interface mirrors render_mesh(pts, R, t, K, H, img_size) so probe/eval code can swap renderers.
"""
import os
import numpy as np
import torch

_MESH_KIND_DEFAULT = os.environ.get('MESH_KIND', 'visual')  # visual = matches real robot pixels (for SAM targets)
MESH_ROOT = os.path.join(os.path.dirname(__file__), '../ViS/Panda/meshes')
# link mesh -> index into all_link_transforms frames (base, j1..j7, j8, hand) = 0..9
LINK_MESH = [('link0', 0), ('link1', 1), ('link2', 2), ('link3', 3), ('link4', 4),
             ('link5', 5), ('link6', 6), ('link7', 7), ('hand', 9)]


def _mesh_path(kind, name):
    """visual meshes live in per-link subfolders (link0/link0.obj); collision are flat (link0.obj)."""
    sub = os.path.join(MESH_ROOT, kind, name, name + '.obj')
    return sub if os.path.exists(sub) else os.path.join(MESH_ROOT, kind, name + '.obj')


def load_link_meshes(device, kind=_MESH_KIND_DEFAULT, finger_open=0.02):
    """Load every link mesh with FACES (trimesh handles multi-object OBJ), concatenated into one
    buffer with per-link vertex slices so the whole robot rasterizes in a single call.
    Both gripper fingers are baked into the HAND frame at a fixed half-opening `finger_open`
    (URDF: finger joints prismatic ±y at xyz 0,0,0.0584; right finger mesh z-rotated pi) —
    SAM masks include the fingers, so the render must too or IoU pays a systematic tax.
    Returns dict(verts=(V,3) float32 tensor, faces=(F,3) int32 tensor, slices=[(fidx, lo, hi)])."""
    import trimesh
    verts, faces, slices = [], [], []
    off = 0

    def _add(v, f, fidx):
        nonlocal off
        verts.append(v.astype(np.float32)); faces.append(f + off)
        slices.append((fidx, off, off + len(v)))
        off += len(v)

    for name, fidx in LINK_MESH:
        m = trimesh.load(_mesh_path(kind, name), force='mesh', process=False)
        _add(np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int64), fidx)

    fm = trimesh.load(_mesh_path(kind, 'finger'), force='mesh', process=False)
    fv = np.asarray(fm.vertices, dtype=np.float32); ff = np.asarray(fm.faces, dtype=np.int64)
    HAND_FIDX = 9
    _add(fv + np.array([0.0, finger_open, 0.0584], dtype=np.float32), ff, HAND_FIDX)          # left
    rz = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)   # Rz(pi)
    _add(fv @ rz.T + np.array([0.0, -finger_open, 0.0584], dtype=np.float32), ff, HAND_FIDX)  # right

    return {
        'verts': torch.from_numpy(np.concatenate(verts)).to(device),
        'faces': torch.from_numpy(np.concatenate(faces).astype(np.int32)).to(device),
        'slices': slices,
    }


def transform_robot_verts(theta, mesh, all_link_transforms_fn):
    """FK-pose the concatenated mesh verts: (B, V, 3) in robot base frame."""
    frames = all_link_transforms_fn(theta)                              # (B,10,4,4)
    B = theta.shape[0]
    V = mesh['verts'].shape[0]
    out = mesh['verts'].new_zeros(B, V, 3)
    vh = torch.cat([mesh['verts'], torch.ones(V, 1, device=mesh['verts'].device)], 1)  # (V,4)
    for fidx, lo, hi in mesh['slices']:
        T = frames[:, fidx]                                             # (B,4,4)
        out[:, lo:hi] = torch.einsum('bij,nj->bni', T, vh[lo:hi])[..., :3]
    return out


class NVDRSilhouette:
    """Batched differentiable silhouette renderer. K is at `img_size` scale; output (B, H, H)."""

    def __init__(self, device, kind=_MESH_KIND_DEFAULT, near=0.05, far=20.0):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.ctx = dr.RasterizeCudaContext(device=device)
        self.mesh = load_link_meshes(device, kind)
        self.near, self.far = near, far
        self.kind = kind

    def robot_verts(self, theta, all_link_transforms_fn):
        return transform_robot_verts(theta, self.mesh, all_link_transforms_fn)

    def render_depth(self, pts_robot, R, t, K, H, img_size):
        """Camera-space depth map (B,H,H); 0 outside the robot. Depth DISCONTINUITIES encode
        link boundaries + self-occlusion contours — the internal structure the silhouette
        throws away (probe signal for photometric/feature render-and-compare)."""
        dr = self.dr
        B = pts_robot.shape[0]
        cam = torch.einsum('bij,bpj->bpi', R, pts_robot) + t.unsqueeze(1)
        x, y, z = cam[..., 0], cam[..., 1], cam[..., 2]
        W = float(img_size)
        fx, fy = K[:, 0, 0:1], K[:, 1, 1:2]
        cx, cy = K[:, 0, 2:3], K[:, 1, 2:3]
        n, f = self.near, self.far
        xc = x * (2.0 * fx / W) + z * (2.0 * cx / W - 1.0)
        yc = y * (2.0 * fy / W) + z * (2.0 * cy / W - 1.0)
        zc = z * (f + n) / (f - n) - (2.0 * f * n) / (f - n)
        pos = torch.stack([xc, yc, zc, z], dim=-1).contiguous()
        rast, _ = dr.rasterize(self.ctx, pos, self.mesh['faces'], resolution=[H, H])
        depth, _ = dr.interpolate(z.unsqueeze(-1).contiguous(), rast, self.mesh['faces'])
        return (depth.squeeze(-1) * (rast[..., 3] > 0).float())          # (B,H,H)

    def __call__(self, pts_robot, R, t, K, H, img_size):
        """pts_robot: (B,V,3) FK-posed verts (from .robot_verts). Returns soft mask (B,H,H)."""
        dr = self.dr
        B = pts_robot.shape[0]
        cam = torch.einsum('bij,bpj->bpi', R, pts_robot) + t.unsqueeze(1)   # (B,V,3), z forward
        x, y, z = cam[..., 0], cam[..., 1], cam[..., 2]
        W = float(img_size)
        fx, fy = K[:, 0, 0:1], K[:, 1, 1:2]
        cx, cy = K[:, 0, 2:3], K[:, 1, 2:3]
        n, f = self.near, self.far
        # clip coords with w = z (pinhole): NDC x = (fx*X/Z + cx) * 2/W - 1, y likewise (y-down kept;
        # mask is orientation-consistent with the K projection used across the repo).
        xc = x * (2.0 * fx / W) + z * (2.0 * cx / W - 1.0)
        yc = y * (2.0 * fy / W) + z * (2.0 * cy / W - 1.0)
        zc = z * (f + n) / (f - n) - (2.0 * f * n) / (f - n)
        pos = torch.stack([xc, yc, zc, z], dim=-1).contiguous()             # (B,V,4)
        rast, _ = dr.rasterize(self.ctx, pos, self.mesh['faces'], resolution=[H, H])
        colour = torch.ones(B, pts_robot.shape[1], 1, device=pts_robot.device)
        mask, _ = dr.interpolate(colour, rast, self.mesh['faces'])
        mask = dr.antialias(mask, rast, pos, self.mesh['faces'])            # edge gradients
        return mask.squeeze(-1).clamp(0, 1)                                 # (B,H,H)


if __name__ == '__main__':
    # Smoke: render GT-pose silhouettes for a few realsense frames and report IoU vs the splat
    # renderer + save overlays for visual inspection.
    import argparse, sys
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
    from silhouette_mesh_probe import (all_link_transforms, robot_pointcloud, render_mesh,
                                       load_obj_verts, mesh_path, kabsch_batch, KPN)
    import silhouette_mesh_probe as SMP
    from inference_4tier_eval import EvalDataset
    from refine_eval import scale_K
    from model_v4 import panda_forward_kinematics
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--render-h', type=int, default=224)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--kind', default='visual')
    ap.add_argument('--out', default='render_nvdr_smoke')
    args = ap.parse_args()

    device = torch.device('cuda')
    S, H = args.image_size, args.render_h
    ds = EvalDataset(args.val_dir, KPN, image_size=(S, S))
    st = max(1, len(ds.json_files) // args.n); ds.json_files = ds.json_files[::st][:args.n]
    loader = DataLoader(ds, batch_size=args.n, shuffle=False, num_workers=2)
    batch = next(iter(loader))
    img = batch['image'].to(device)
    K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
    gt3d = batch['gt_3d'].to(device); ga = batch['gt_angles'].to(device).clone(); ga[:, 6] = 0.0
    fkg = panda_forward_kinematics(ga); Rg, tg = kabsch_batch(fkg, gt3d)

    rdr = NVDRSilhouette(device, kind=args.kind)
    pts = rdr.robot_verts(ga, all_link_transforms)
    with torch.no_grad():
        m_nvdr = rdr(pts, Rg, tg, K, H, S)
        # splat baseline (collision, as in the probe)
        mesh_verts = [((nm, fi), torch.from_numpy(load_obj_verts(mesh_path(nm))).to(device))
                      for nm, fi in SMP.LINK_MESH]
        m_splat = render_mesh(robot_pointcloud(ga, mesh_verts), Rg, tg, K, H, S)
    a = (m_nvdr > 0.5).float(); b = (m_splat > 0.5).float()
    iou = ((a * b).sum((-1, -2)) / ((a + b) > 0).float().sum((-1, -2)).clamp(min=1))
    print(f"nvdr({args.kind}) vs splat(collision) IoU: mean {iou.mean():.3f}  per-frame {[f'{v:.2f}' for v in iou.cpu().tolist()]}")

    # overlays: real image + nvdr mask boundary
    import torch.nn.functional as Fn
    import cv2
    os.makedirs(args.out, exist_ok=True)
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    rgb = ((img * STD + MEAN).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    up = Fn.interpolate(m_nvdr.unsqueeze(1), size=(S, S), mode='bilinear', align_corners=False).squeeze(1)
    for i in range(img.shape[0]):
        o = rgb[i].copy()
        mm = (up[i] > 0.5).cpu().numpy().astype(np.uint8)
        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cnts, -1, (0, 255, 0), 2)
        o[..., 1] = np.maximum(o[..., 1], (mm * 60).astype(np.uint8))
        cv2.imwrite(os.path.join(args.out, f'overlay_{i:02d}.png'), o[..., ::-1])
    print(f"overlays -> {args.out}/  (green contour must hug the real robot)")
