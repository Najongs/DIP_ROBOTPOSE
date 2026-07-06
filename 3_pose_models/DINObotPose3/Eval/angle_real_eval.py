"""Angle-head MAE on REAL data (sim2real angle-prior test).

Loads a detector (backbone+keypoint_head) + a trained angle_head, runs on a real
DREAM-format val dir with the SAME crop-to-robot input used in training, and reports
per-joint angle MAE vs the JSON GT joint angles. This is the decisive test of whether a
SYNTHETIC-trained angle head supplies a basin-correct prior on real robots — the FR3
bottleneck was 45 deg MAE from a real-trained head.

Two modes:
  (default)   angles from the model's own detected 2D keypoints  -> realistic end-to-end
  --oracle-2d not applicable here (angle head consumes features, not 2D) ; use --gt-crop to
              isolate detector localization by cropping on GT keypoints (already the default
              via --crop-to-robot which uses GT-keypoint bbox).
"""
import argparse, os, sys, math
import numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from dataset import PoseEstimationDataset
from refine_eval import scale_K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--angle-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--keypoint-names', default='link0,link1,link2,link3,link4,link5,link6')
    ap.add_argument('--num-angles', type=int, default=6)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=2000)
    ap.add_argument('--crop-to-robot', action='store_true', default=True)
    ap.add_argument('--no-crop', dest='crop_to_robot', action='store_false')
    ap.add_argument('--crop-margin', type=float, default=1.5)
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available(); S = args.image_size
    KPN = args.keypoint_names.split(','); NA = args.num_angles
    m = AnglePredictor(args.model_name, S, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.angle_head, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=KPN, image_size=(S, S), heatmap_size=(S, S),
                               augment=False, include_angles=True, sigma=2.5,
                               crop_to_robot=args.crop_to_robot, crop_margin=args.crop_margin)
    if args.max_frames and args.max_frames < len(ds): ds.samples = ds.samples[:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    errs = []
    for batch in tqdm(loader, desc='angle'):
        imgs = batch['image'].to(device)
        gt = batch['angles'].to(device)                        # (B, NA) radians
        K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        with torch.no_grad():
            pred = m(imgs, K)['joint_angles']                  # (B, >=NA)
        d = pred[:, :NA] - gt[:, :NA]
        d = torch.atan2(torch.sin(d), torch.cos(d)).abs() * 180 / math.pi
        errs.append(d.cpu())
    errs = torch.cat(errs, 0)                                   # (M, NA)
    per = errs.mean(0)
    print(f"\nAngle MAE on REAL {os.path.basename(args.val_dir)}  (n={len(errs)} frames, crop={args.crop_to_robot})")
    print(f"   overall MAE(J0-{NA-1}) = {per.mean().item():.2f} deg")
    print("   per-joint: " + ", ".join(f"J{j}={per[j].item():.1f}" for j in range(NA)))
    print(f"   (FR3 real-trained head was 45 deg -> lower is better; <~15 deg validates synth angle prior)")


if __name__ == '__main__':
    main()
