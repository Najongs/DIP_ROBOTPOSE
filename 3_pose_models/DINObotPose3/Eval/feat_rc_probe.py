"""
ORACLE probe for DINO feature-metric render-and-compare (survey round-2, Idea 1).
Go/no-go before investing in the heavy differentiable build.

Hypothesis (MCLoc/AlignPose): comparing a rendered pose to the real image in FROZEN-ViT feature
space absorbs the albedo/lighting domain gap that sinks plain photometric RC — so it should
discriminate the GT pose from perturbations even on near cameras (azure) where silhouette RC hurts.

Test: render a Lambertian normal-shaded image at GT vs perturbed poses -> DINOv3 patch features ->
masked per-patch cosine similarity to the REAL crop's DINOv3 features (robot patches only).
Report GT-wins fraction + margin per perturbation, alongside the edge-NCC baseline (rgb_rc_probe)
so we know if features beat the cheap structure signal enough to justify the build.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
from model_v4 import panda_forward_kinematics, DINOv3Backbone
from silhouette_mesh_probe import kabsch_batch, all_link_transforms, KPN
from inference_4tier_eval import EvalDataset
from refine_eval import scale_K
from render_nvdr import NVDRSilhouette
from rgb_rc_probe import edge_score

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def dino_feats(backbone, img3, size=512):
    """img3 (B,3,h,w) in [0,1] -> L2-normalized patch features (B,C,gh,gw)."""
    x = F.interpolate(img3, size=(size, size), mode='bilinear', align_corners=False)
    x = (x - MEAN.to(x.device)) / STD.to(x.device)
    tok = backbone(x)                                   # (B, Np, C)
    B, Np, C = tok.shape
    g = int(round(Np ** 0.5))
    f = tok.transpose(1, 2).reshape(B, C, g, g)
    return F.normalize(f, dim=1)


def feat_score(fr, fr_real, robot_mask_grid):
    """masked mean per-patch cosine similarity. fr,fr_real (B,C,g,g); mask (B,g,g)."""
    cos = (fr * fr_real).sum(1)                         # (B,g,g)
    m = robot_mask_grid
    return (cos * m).sum((-1, -2)) / m.sum((-1, -2)).clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--render-h', type=int, default=512)
    ap.add_argument('--n', type=int, default=80)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    args = ap.parse_args()
    device = torch.device('cuda'); S = args.image_size; H = args.render_h

    ds = EvalDataset(args.val_dir, KPN, image_size=(S, S))
    st = max(1, len(ds.json_files) // args.n); ds.json_files = ds.json_files[::st][:args.n]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    rdr = NVDRSilhouette(device, kind='visual')
    backbone = DINOv3Backbone(args.model_name, unfreeze_blocks=0).to(device).eval()

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
    PERTS = [('yaw+5', yaw(5)), ('yaw-5', yaw(-5)), ('yaw+15', yaw(15)),
             ('depth+5%', depth(0.05)), ('depth-5%', depth(-0.05)),
             ('J1+10', joint(1, 10)), ('J3+10', joint(3, 10)), ('J4+15', joint(4, 15))]

    fw = {n: 0 for n, _ in PERTS}; fm = {n: [] for n, _ in PERTS}          # feature
    ew = {n: 0 for n, _ in PERTS}; tot = 0                                  # edge baseline
    for batch in tqdm(loader, desc='feat-rc-probe'):
        img = batch['image'].to(device)
        K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        gt3d = batch['gt_3d'].to(device); ga = batch['gt_angles'].to(device).clone(); ga[:, 6] = 0.0
        fkg = panda_forward_kinematics(ga); Rg, tg = kabsch_batch(fkg, gt3d)
        rgb = (img * STD.to(device) + MEAN.to(device)).clamp(0, 1)
        gray = rgb.mean(1, keepdim=True)
        with torch.no_grad():
            f_real = dino_feats(backbone, rgb, S)
            g = f_real.shape[-1]
            def render_and_feat(th, R, t):
                sh = rdr.render_shaded(rdr.robot_verts(th, all_link_transforms), R, t, K, H, S)
                sh3 = sh.unsqueeze(1).repeat(1, 3, 1, 1)
                fr = dino_feats(backbone, sh3, S)
                mask = (F.interpolate(sh.unsqueeze(1), size=(g, g), mode='bilinear', align_corners=False).squeeze(1) > 0.05).float()
                return fr, mask, sh
            fr0, m0, sh0 = render_and_feat(ga, Rg, tg)
            fs0 = feat_score(fr0, f_real, m0)
            d0 = rdr.render_depth(rdr.robot_verts(ga, all_link_transforms), Rg, tg, K, H, S)
            es0 = edge_score(d0, gray, H)
            tot += img.shape[0]
            for name, fn in PERTS:
                th2, R2, t2 = fn(ga, Rg, tg)
                fr2, m2, _ = render_and_feat(th2, R2, t2)
                fs2 = feat_score(fr2, f_real, m2)
                fw[name] += int((fs0 > fs2).sum()); fm[name].extend((fs0 - fs2).cpu().tolist())
                d2 = rdr.render_depth(rdr.robot_verts(th2, all_link_transforms), R2, t2, K, H, S)
                es2 = edge_score(d2, gray, H)
                ew[name] += int((es0 > es2).sum())

    cam = os.path.basename(args.val_dir)
    print(f"\n=== DINO FEATURE-METRIC RC ORACLE PROBE  {cam}  (n={tot}) ===")
    print(f"  {'perturbation':>10} | feat GT-wins | feat margin | edge GT-wins")
    for name, _ in PERTS:
        print(f"  {name:>10} |     {fw[name]/tot:5.0%}   | {np.mean(fm[name]):+.4f}    |    {ew[name]/tot:5.0%}")
    print("  (feature GT-wins > edge => features carry MORE pose signal => build differentiable feature-metric RC)")


if __name__ == '__main__':
    main()
