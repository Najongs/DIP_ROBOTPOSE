"""
Val-only diagnostic for a trained rotation head: decompose the translation error into its
DEPTH (|dz|) and LATERAL (dxy) components.

WHY THIS EXISTS: `train_rotation.py` reports `t-err = |t_pred - t_gt|` — a 3D NORM, despite the
misleading `tz_med` variable name it used to be stored in. Every downstream claim of the form
"KUKA t-err 56 mm, i.e. the depth is wrong" was therefore an ASSUMPTION, never a measurement.
It matters because the RootNet depth head (train_rotation.py --depth-head) replaces the z
component ONLY: if the error is actually lateral, that fix is structurally incapable of helping.
This script measures the split on an existing checkpoint before any GPU is spent on training.

Usage (GPU by UUID — integer CUDA indices are scrambled on this machine):
  CUDA_VISIBLE_DEVICES=GPU-<uuid> python eval_rot_head.py \
    --rot-head outputs_rotation/kuka_rot_20260712_060214/best_rot_head.pth \
    --detector-ckpt outputs_heatmap/kuka_dream_detector_20260709_183119/best_heatmap.pth \
    --val-dir ../../../datasets/synthetic/kuka_synth_test_dr --fk-robot kuka \
    --keypoint-names iiwa7_link_1,...  --angle-joint-names iiwa7_joint_1,...
"""
import argparse
import torch
from torch.utils.data import DataLoader

import train_rotation as TR
from model_angle import AnglePredictor
from model_v4 import iiwa7_forward_kinematics, baxter_left_forward_kinematics, panda_forward_kinematics
from dataset import PoseEstimationDataset


def main(args):
    device = torch.device('cuda'); assert torch.cuda.is_available()
    TR._FK = {'kuka': iiwa7_forward_kinematics, 'iiwa7': iiwa7_forward_kinematics,
              'baxter': baxter_left_forward_kinematics, 'baxter_left': baxter_left_forward_kinematics,
              'panda': panda_forward_kinematics}[args.fk_robot]

    ds = PoseEstimationDataset(
        data_dir=args.val_dir, keypoint_names=args.keypoint_names.split(','),
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.image_size, args.image_size), augment=False,
        include_angles=True, sigma=2.5,
        crop_to_robot=args.crop_to_robot, crop_margin=args.crop_margin,
        angle_joint_names=args.angle_joint_names.split(',') if args.angle_joint_names else None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = AnglePredictor(args.model_name, args.image_size, fix_joint7_zero=True, head_type='mlp',
                           with_rotation=True, with_translation=True).to(device)
    det = {k.replace('module.', ''): v for k, v in torch.load(args.detector_ckpt, map_location=device).items()}
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in det.items() if k in msd and v.shape == msd[k].shape}, strict=False)
    model.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))
    model.eval()

    ge, dz, dxy, tn, gzs = [], [], [], [], []
    n = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            kp3d = batch['keypoints_3d'].to(device)
            K = TR.scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)
            valid = (kp3d.abs().sum(-1) > 1e-6) & (kp3d[..., 2] > 0)
            keep = valid.sum(1) >= 4
            if keep.sum() == 0:
                continue
            Rg, tg = TR.gt_pose(batch['angles'].to(device), kp3d, valid)
            o = model(imgs, K)
            dt = (o['trans'][keep] - tg[keep]) * 1000.0                      # mm
            ge.append(TR.geodesic_deg(o['rot_matrix'][keep], Rg[keep]).cpu())
            tn.append(dt.norm(dim=-1).cpu())
            dz.append(dt[:, 2].abs().cpu())
            dxy.append(dt[:, :2].norm(dim=-1).cpu())
            gzs.append((tg[keep][:, 2] * 1000.0).cpu())
            n += int(keep.sum())
            if args.max_frames and n >= args.max_frames:
                break

    ge, tn, dz, dxy, gz = (torch.cat(x) for x in (ge, tn, dz, dxy, gzs))
    m = lambda t: (t.median().item(), t.mean().item())
    print(f"\n===== {args.tag or args.fk_robot} | {args.val_dir} | n={ge.numel()} =====")
    print(f"  geodesic R err : median {m(ge)[0]:.2f}  mean {m(ge)[1]:.2f} deg")
    print(f"  t-err (3D norm): median {m(tn)[0]:.1f}  mean {m(tn)[1]:.1f} mm   <- the legacy number")
    print(f"  |dz|  (DEPTH)  : median {m(dz)[0]:.1f}  mean {m(dz)[1]:.1f} mm")
    print(f"  dxy   (LATERAL): median {m(dxy)[0]:.1f}  mean {m(dxy)[1]:.1f} mm")
    print(f"  ratio median|dz|/median dxy = {m(dz)[0]/max(m(dxy)[0],1e-9):.2f}")
    print(f"  GT root depth  : median {m(gz)[0]:.0f} mm   (|dz| = {100*m(dz)[0]/m(gz)[0]:.1f}% of depth)")
    print(f"  variance share : dz {100*(dz**2).mean()/((dz**2).mean()+(dxy**2).mean()):.1f}%  "
          f"xy {100*(dxy**2).mean()/((dz**2).mean()+(dxy**2).mean()):.1f}%  (of mean squared t-err)")
    print(f"  VERDICT: depth-dominated (dz >= 2x dxy)? -> "
          f"{'YES -> RootNet depth head justified' if m(dz)[0] >= 2*m(dxy)[0] else 'NO'}\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rot-head', required=True); p.add_argument('--detector-ckpt', required=True)
    p.add_argument('--val-dir', required=True); p.add_argument('--fk-robot', default='kuka')
    p.add_argument('--keypoint-names', required=True); p.add_argument('--angle-joint-names', default=None)
    p.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    p.add_argument('--image-size', type=int, default=512); p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--crop-to-robot', action='store_true'); p.add_argument('--crop-margin', type=float, default=1.5)
    p.add_argument('--num-workers', type=int, default=8); p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--tag', default=None)
    main(p.parse_args())
