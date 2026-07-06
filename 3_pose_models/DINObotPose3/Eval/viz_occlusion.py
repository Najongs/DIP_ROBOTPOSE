"""
Qualitative OCCLUSION check: paste occluders on the input (RoboPEPP protocol), run the FULL pipeline
on the OCCLUDED image, and overlay the estimate — so you can SEE the robot partly hidden yet the pose
still inferred ("가려져도 대략 추론").

Per frame, background = the occluded image the model actually saw. Overlays:
  GREEN = GT 2D skeleton (true pose)
  RED   = predicted FK reprojection from the refined (theta,R,t)  -> model's inferred pose UNDER occlusion
  CYAN  = keypoints whose detection was gated out by conf (behind an occluder) yet still placed by kinematics
Title: occlusion ratio, angle MAE (deg), ADD (mm).
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from dataset import PoseEstimationDataset
from solve_pose_kinematic import solve_batch
from refine_eval import scale_K, wrapped_abs_deg
from occl_util import paste_occluders_batch_

CHAIN = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
NAMES = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
_M = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_S = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def project(kp_cam, K):
    z = kp_cam[:, 2:3].clamp(min=1e-4)
    return (K @ (kp_cam / z).T).T[:, :2]


def draw(bg_pil, gt2d, pred2d, det_gated, S, scale_to):
    im = bg_pil.convert('RGB').resize(scale_to)
    sx, sy = scale_to[0] / S, scale_to[1] / S
    dr = ImageDraw.Draw(im)
    def P(p): return (float(p[0]) * sx, float(p[1]) * sy)
    for a, b in CHAIN:
        dr.line([P(gt2d[a]), P(gt2d[b])], fill=(0, 220, 0), width=3)
        dr.line([P(pred2d[a]), P(pred2d[b])], fill=(240, 40, 40), width=3)
    for i in range(7):
        gx, gy = P(gt2d[i]); px, py = P(pred2d[i])
        dr.ellipse([gx-4, gy-4, gx+4, gy+4], fill=(0, 220, 0))
        col = (0, 230, 230) if det_gated[i] else (240, 40, 40)   # cyan = inferred behind occluder
        dr.ellipse([px-4, py-4, px+4, py+4], fill=col)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--indices', default='50,800,1600,2400,3200,4000')
    ap.add_argument('--occlude-ratio', type=float, default=0.3)
    ap.add_argument('--ladder', default=None,
                    help='ONE index; render it at escalating occlusion ratios (e.g. "1600:0,0.1,0.2,0.3,0.4")')
    ap.add_argument('--conf-gate', type=float, default=0.05)
    ap.add_argument('--cell', type=int, default=384)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'viz_outputs/viz_occlusion.png'))
    args = ap.parse_args()
    CW, CH = args.cell, args.cell * 3 // 4

    device = torch.device('cuda'); assert torch.cuda.is_available()
    S = args.image_size
    m = AnglePredictor(args.model_name, S, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict()
                       and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=NAMES, image_size=(S, S),
                               heatmap_size=(S, S), augment=False, include_angles=True, sigma=2.5)
    # ladder mode: one frame, escalating occlusion. jobs = list of (index, ratio, fid_seed)
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
        # paste occluders on the INPUT (deterministic per fid), RoboPEPP protocol
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
        refined, kp_cam, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=200, lr=2e-2,
                                         img_size=S, device=device, prior_w=0.0,
                                         theta_init=init, conf_gate=args.conf_gate)
        pred2d = project(kp_cam[0], K[0]).cpu().numpy()
        gated = (conf[0].cpu().numpy() < args.conf_gate)   # keypoints kinematics had to infer
        ang_mae = float(wrapped_abs_deg(refined[:, :6], gt).mean())
        valid = (gt3d[0].abs().sum(-1) > 0)
        add = float((kp_cam[0][valid] - gt3d[0][valid]).norm(dim=-1).mean() * 1000)
        occ_bg = (img[0].cpu() * _S + _M).clamp(0, 1).permute(1, 2, 0).numpy()
        src = Image.fromarray((occ_bg * 255).astype(np.uint8))
        cells.append(draw(src, gt2d, pred2d, gated, S, (CW, CH)))
        ngated = int(gated.sum())
        titles.append(f"#{ix}  occ {int(ratio*100)}%  ang {ang_mae:.1f}deg  ADD {add:.0f}mm  gated {ngated}/7")
        print(titles[-1])

    cols = min(args.cols, len(cells)); rows = (len(cells) + cols - 1) // cols
    cw, ch = CW, CH + 20
    grid = Image.new('RGB', (cw * cols, ch * rows), (15, 15, 15))
    dr = ImageDraw.Draw(grid)
    for k, (im, t) in enumerate(zip(cells, titles)):
        x, y = (k % cols) * cw, (k // cols) * ch
        grid.paste(im, (x, y + 20))
        dr.text((x + 4, y + 5), t, fill=(255, 255, 255))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    grid.save(args.out)
    print("saved", args.out, "| GREEN=GT  RED=pred  CYAN=inferred-behind-occluder")


if __name__ == '__main__':
    main()
