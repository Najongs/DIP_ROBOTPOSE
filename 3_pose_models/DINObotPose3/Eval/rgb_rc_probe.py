"""
ORACLE probe for RGB/structure render-and-compare (roadmap ③): does the robot's INTERNAL
structure — link boundaries and self-occlusion contours, which the silhouette throws away —
carry a pose signal measurable against the real image?

Signal: masked normalized correlation between |∇(rendered depth)| and |∇(gray image)| (edge
agreement, lighting/albedo-free by construction). Test: is the score at the GT pose higher than
at perturbed poses (yaw/depth/joint wiggles)? If GT wins consistently and the margin grows with
perturbation size, a structure-comparison term can refine poses where the silhouette is blind
(near cameras like azure, rotations at fixed silhouette area).
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
from model_v4 import panda_forward_kinematics
from silhouette_mesh_probe import kabsch_batch, all_link_transforms, KPN
from inference_4tier_eval import EvalDataset
from refine_eval import scale_K
from render_nvdr import NVDRSilhouette

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def grad_mag(x):
    """|∇x| via Sobel, x (B,1,H,W) -> (B,1,H,W)."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 4
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(x, kx, padding=1); gy = F.conv2d(x, ky, padding=1)
    return (gx ** 2 + gy ** 2).sqrt()


def edge_score(depth, gray, H):
    """Masked normalized correlation of edge maps. depth (B,H,H) (0 outside robot), gray (B,1,S,S)."""
    B = depth.shape[0]
    g = F.interpolate(gray, size=(H, H), mode='bilinear', align_corners=False)
    eg = grad_mag(g)                                        # image edges
    ed = grad_mag(depth.unsqueeze(1))                       # render structure edges
    m = (depth > 0).unsqueeze(1).float()
    m = -F.max_pool2d(-m, 5, 1, 2)                          # erode: kill the outer silhouette contour
    ed = ed * m; eg = eg * m
    def _n(t):
        mu = t.sum((-1, -2), keepdim=True) / m.sum((-1, -2), keepdim=True).clamp(min=1)
        return (t - mu) * m
    ed = _n(ed); eg = _n(eg)
    num = (ed * eg).sum((-1, -2))
    den = (ed.pow(2).sum((-1, -2)).sqrt() * eg.pow(2).sum((-1, -2)).sqrt()).clamp(min=1e-6)
    return (num / den).squeeze(1)                           # (B,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--render-h', type=int, default=448)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()
    device = torch.device('cuda'); S = args.image_size; H = args.render_h

    ds = EvalDataset(args.val_dir, KPN, image_size=(S, S))
    st = max(1, len(ds.json_files) // args.n); ds.json_files = ds.json_files[::st][:args.n]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    rdr = NVDRSilhouette(device, kind='visual')

    # perturbation grid: (name, fn(theta, R, t) -> (theta', R', t'))
    def yaw(deg):
        d = np.deg2rad(deg); c, s = np.cos(d), np.sin(d)
        Rz = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32, device=device)
        return lambda th, R, t: (th, R @ Rz, t)
    def depth(frac):
        return lambda th, R, t: (th, R, t * (1 + frac))
    def joint(j, deg):
        def f(th, R, t):
            th2 = th.clone(); th2[:, j] += np.deg2rad(deg); return th2, R, t
        return f
    PERTS = [('yaw+5', yaw(5)), ('yaw-5', yaw(-5)), ('yaw+15', yaw(15)), ('yaw-15', yaw(-15)),
             ('depth+5%', depth(0.05)), ('depth-5%', depth(-0.05)),
             ('J1+10', joint(1, 10)), ('J3+10', joint(3, 10)), ('J4+15', joint(4, 15))]

    wins = {n: 0 for n, _ in PERTS}; margins = {n: [] for n, _ in PERTS}; tot = 0
    for batch in tqdm(loader, desc='rgb-rc-probe'):
        img = torch.stack([torch.as_tensor(x) for x in batch['image']]) if isinstance(batch['image'], list) else batch['image']
        img = img.to(device)
        K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        gt3d = batch['gt_3d'].to(device); ga = batch['gt_angles'].to(device).clone(); ga[:, 6] = 0.0
        fkg = panda_forward_kinematics(ga); Rg, tg = kabsch_batch(fkg, gt3d)
        gray = ((img * STD.to(device) + MEAN.to(device)).clamp(0, 1)).mean(1, keepdim=True)
        with torch.no_grad():
            d0 = rdr.render_depth(rdr.robot_verts(ga, all_link_transforms), Rg, tg, K, H, S)
            s0 = edge_score(d0, gray, H)                     # GT score
            tot += img.shape[0]
            for name, fn in PERTS:
                th2, R2, t2 = fn(ga, Rg, tg)
                d2 = rdr.render_depth(rdr.robot_verts(th2, all_link_transforms), R2, t2, K, H, S)
                s2 = edge_score(d2, gray, H)
                wins[name] += int((s0 > s2).sum())
                margins[name].extend((s0 - s2).cpu().tolist())

    cam = os.path.basename(args.val_dir)
    print(f"\n=== RGB/STRUCTURE RC ORACLE PROBE  {cam}  (n={tot}, edge-NCC, silhouette contour eroded) ===")
    print(f"  {'perturbation':>10} | GT-wins | mean margin")
    for name, _ in PERTS:
        print(f"  {name:>10} |  {wins[name]/tot:5.0%} | {np.mean(margins[name]):+.4f}")
    print("  (GT-wins >~75% with positive margin => internal-structure signal exists => build the term)")


if __name__ == '__main__':
    main()
