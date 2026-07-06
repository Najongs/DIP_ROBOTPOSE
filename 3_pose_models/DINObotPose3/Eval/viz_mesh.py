"""
Qualitative MESH-SILHOUETTE overlay: run the full pipeline, recover the camera pose (R,t) from the
solved keypoints (Kabsch), render the EXACT Panda mesh (nvdiffrast, Lambertian-shaded) at the predicted
(theta,R,t), and alpha-blend it onto the real image. Unlike the skeleton overlay this shows the whole
posed robot body aligned (or not) with the photo — the most direct "does the estimate match reality" check.

Modes:
  default : N frames (--indices), clean or --occlude-ratio.
  --ladder "IDX:0,0.1,0.2,0.3,0.4" : one frame at escalating occlusion.
Tint = orange mesh overlay; faint GREEN GT skeleton drawn for reference.
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
os.environ.setdefault('MESH_KIND', 'visual')
from model_angle import AnglePredictor
from dataset import PoseEstimationDataset
from solve_pose_kinematic import solve_batch
from refine_eval import scale_K, wrapped_abs_deg
from occl_util import paste_occluders_batch_
from model_v4 import panda_forward_kinematics
from silhouette_mesh_probe import kabsch_batch, all_link_transforms
from render_nvdr import NVDRSilhouette

CHAIN = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
NAMES = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
_M = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_S = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
TINT = np.array([1.0, 0.55, 0.15])   # orange mesh


def project(kp_cam, K):
    z = kp_cam[:, 2:3].clamp(min=1e-4)
    return (K @ (kp_cam / z).T).T[:, :2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--indices', default='50,800,1600,2400,3200,4000')
    ap.add_argument('--occlude-ratio', type=float, default=0.0)
    ap.add_argument('--ladder', default=None)
    ap.add_argument('--conf-gate', type=float, default=0.05)
    ap.add_argument('--alpha', type=float, default=0.55, help='mesh overlay opacity')
    ap.add_argument('--kind', default='visual')
    ap.add_argument('--gt-skel', action='store_true', help='also draw faint GT skeleton')
    ap.add_argument('--cell', type=int, default=384)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'viz_outputs/viz_mesh.png'))
    args = ap.parse_args()
    CW, CH = args.cell, args.cell * 3 // 4

    device = torch.device('cuda'); assert torch.cuda.is_available()
    S = args.image_size
    m = AnglePredictor(args.model_name, S, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict()
                       and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))
    rdr = NVDRSilhouette(device, kind=args.kind)

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=NAMES, image_size=(S, S),
                               heatmap_size=(S, S), augment=False, include_angles=True, sigma=2.5)
    if args.ladder:
        li, lr = args.ladder.split(':')
        jobs = [(int(li), float(r), f"{li}_{r}") for r in lr.split(',')]
    else:
        jobs = [(ix, args.occlude_ratio, str(ix)) for ix in
                (int(x) for x in args.indices.split(',') if int(x) < len(ds))]

    cells, titles = [], []
    for ix, ratio, seed in jobs:
        b = ds[ix]
        img = b['image'].unsqueeze(0).to(device)
        if ratio > 0:
            kp_np = b['keypoints'].numpy()[None]
            vm = b['valid_mask'].numpy()[None] if 'valid_mask' in b else np.ones((1, 7), bool)
            paste_occluders_batch_(img, kp_np, vm, ratio, [seed])

        gt = b['angles'][:6].unsqueeze(0).to(device)
        gt3d = b['keypoints_3d'].unsqueeze(0).to(device)
        gt2d = b['keypoints'].numpy()
        K = scale_K(b['camera_K'].unsqueeze(0), b['original_size'].unsqueeze(0), S).to(device)
        with torch.no_grad():
            out = m(img, K)
        init = out['joint_angles']; kp2d = out['keypoints_2d']; conf = out['confidence']
        # solve_batch runs internal autograd (loss.backward) -> must NOT be under no_grad
        refined, kp_cam, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=200, lr=2e-2,
                                         img_size=S, device=device, prior_w=0.0,
                                         theta_init=init, conf_gate=args.conf_gate)
        with torch.no_grad():
            # recover camera pose (R,t) from FK <-> solved camera keypoints
            fk = panda_forward_kinematics(refined)
            R, t = kabsch_batch(fk, kp_cam)
            verts = rdr.robot_verts(refined, all_link_transforms)
            shade = rdr.render_shaded(verts, R, t, K, S, S)[0].cpu().numpy()   # (S,S) [0,1]
            mask = (rdr(verts, R, t, K, S, S)[0].cpu().numpy())                # soft (S,S)

        valid = (gt3d[0].abs().sum(-1) > 0)
        add = float((kp_cam[0][valid] - gt3d[0][valid]).norm(dim=-1).mean() * 1000)
        ang_mae = float(wrapped_abs_deg(refined[:, :6], gt).mean())

        base = (img[0].cpu() * _S + _M).clamp(0, 1).permute(1, 2, 0).numpy()   # occluded/real bg (S,S,3)
        a = args.alpha * np.clip(mask, 0, 1)[..., None]
        col = shade[..., None] * TINT[None, None, :]                            # shaded orange mesh
        over = base * (1 - a) + col * a
        im = Image.fromarray((np.clip(over, 0, 1) * 255).astype(np.uint8)).resize((CW, CH))
        if args.gt_skel:
            dr = ImageDraw.Draw(im); sx, sy = CW / S, CH / S
            def P(p): return (float(p[0]) * sx, float(p[1]) * sy)
            for u, v in CHAIN:
                dr.line([P(gt2d[u]), P(gt2d[v])], fill=(0, 230, 0), width=2)
        cells.append(im)
        tag = f"occ {int(ratio*100)}%  " if (ratio > 0 or args.ladder) else ""
        titles.append(f"#{ix}  {tag}ang {ang_mae:.1f}deg  ADD {add:.0f}mm")
        print(titles[-1])

    cols = min(args.cols, len(cells)); rows = (len(cells) + cols - 1) // cols
    cw, ch = CW, CH + 20
    grid = Image.new('RGB', (cw * cols, ch * rows), (15, 15, 15))
    dr = ImageDraw.Draw(grid)
    for k, (im, t) in enumerate(zip(cells, titles)):
        x, y = (k % cols) * cw, (k // cols) * ch
        grid.paste(im, (x, y + 20)); dr.text((x + 4, y + 5), t, fill=(255, 255, 255))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    grid.save(args.out)
    print("saved", args.out, "| ORANGE = predicted mesh silhouette (nvdiffrast) over real image")


if __name__ == '__main__':
    main()
