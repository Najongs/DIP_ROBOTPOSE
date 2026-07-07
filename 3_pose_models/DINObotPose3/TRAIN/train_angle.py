"""
Train the learned angle predictor (Stage 1.5) on top of a frozen Stage-1 detector.

Only AngleHead trains. Input = the model's OWN predicted 2D keypoints (robust to detector
noise). Loss = sin/cos SmoothL1 to GT angles + FK robot-frame consistency. Eval = per-joint
angle MAE (deg) on the synthetic val set. Output checkpoint -> init for the kinematic refiner.
"""
import argparse, math, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_angle import AnglePredictor
from model_v4 import panda_forward_kinematics
import sys as _sys, os as _os
_sys.path.append(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../Eval'))
from silhouette_mesh_probe import kabsch_batch
from dataset import PoseEstimationDataset

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


def scale_K(camera_K, original_size, hm):
    """camera_K (B,3,3) original-res -> heatmap-res, using original_size (B,2)=(W,H)."""
    K = camera_K.clone().float()
    for b in range(K.shape[0]):
        ow, oh = float(original_size[b][0]), float(original_size[b][1])
        sx, sy = hm / ow, hm / oh
        K[b, 0, 0] *= sx; K[b, 1, 1] *= sy
        K[b, 0, 2] *= sx; K[b, 1, 2] *= sy
    return K


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | available={torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "Refusing to train on CPU (check GPU UUID selection)."

    kp_names = (args.keypoint_names.split(',') if getattr(args, 'keypoint_names', None)
                else ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand'])
    train_ds = PoseEstimationDataset(
        data_dir=args.train_dir, keypoint_names=kp_names,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.image_size, args.image_size),
        augment=True, aug_level='strong', include_angles=True, sigma=2.5,
        crop_to_robot=args.crop_to_robot, crop_margin=args.crop_margin)
    val_ds = PoseEstimationDataset(
        data_dir=args.val_dir, keypoint_names=kp_names,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.image_size, args.image_size),
        augment=False, include_angles=True, sigma=2.5,
        crop_to_robot=args.crop_to_robot, crop_margin=args.crop_margin)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = AnglePredictor(args.model_name, args.image_size, fix_joint7_zero=True,
                           head_type=args.head_type).to(device)
    print(f"==> Angle head type: {args.head_type}")

    # Load the Stage-1 detector weights into backbone + keypoint_head.
    ckpt = torch.load(args.detector_ckpt, map_location=device)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    msd = model.state_dict()
    loaded = {k: v for k, v in ckpt.items() if k in msd and v.shape == msd[k].shape}
    model.load_state_dict(loaded, strict=False)
    n_det = sum(1 for k in loaded if k.startswith('backbone.') or k.startswith('keypoint_head.'))
    print(f"==> Loaded {len(loaded)} tensors from detector ({n_det} into backbone+keypoint_head)")
    model.freeze_detector()
    if args.init_head:
        model.angle_head.load_state_dict(torch.load(args.init_head, map_location=device))
        print(f'[warm-start] angle_head <- {args.init_head}')

    opt = optim.AdamW(model.angle_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)

    if args.use_wandb and _HAS_WANDB:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    best_mae = 1e9
    for epoch in range(args.epochs):
        model.angle_head.train()
        run_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep{epoch} [train]")
        for batch in pbar:
            imgs = batch['image'].to(device)
            gt = batch['angles'].to(device).clone()
            if gt.shape[1] > 6: gt[:, 6] = 0.0            # Panda: fix joint7=0 (Meca has only 6 angles)
            has = batch['has_angles'].to(device).bool() if 'has_angles' in batch else torch.ones(len(imgs), dtype=torch.bool, device=device)
            K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)

            if args.occlude_aug > 0:
                import sys as _s, os as _o
                _s.path.append(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '../Eval'))
                from occl_util import paste_random_occluders_
                paste_random_occluders_(imgs, batch['keypoints'].numpy(), batch['valid_mask'].numpy(), args.occlude_aug)
            out_d = model(imgs, K, kp_jitter=args.kp_jitter, kp_drop=args.kp_drop)
            sc = out_d['sin_cos']                       # (B,6,2)
            gt6 = gt[:, :6]
            gt_sc = torch.stack([torch.sin(gt6), torch.cos(gt6)], dim=-1)
            sc_loss = F.smooth_l1_loss(sc[has], gt_sc[has])
            loss = sc_loss
            # FK robot-frame consistency (Panda/FR3 only; needs 7-angle panda FK). Skipped for
            # robots without a wired-up FK by passing --fk-weight 0 --reproj-weight 0.
            if args.fk_weight > 0 or args.reproj_weight > 0:
                fk_pred = panda_forward_kinematics(out_d['joint_angles'])
                fk_gt = panda_forward_kinematics(gt)
            if args.fk_weight > 0:
                fk_loss = F.mse_loss(fk_pred[has], fk_gt[has])
                loss = loss + args.fk_weight * fk_loss
            # RoboTAG-style cross-dimensional (2D<->3D) consistency: project FK(pred_angles) through
            # the GT camera pose (Kabsch of FK(gt) onto camera-frame GT keypoints) and match GT 2D.
            # Adds the camera-frame reprojection signal the robot-frame fk_loss lacks — sharpens
            # angles where small errors move 2D (near cameras / azure, our RoboTAG-relative weakness).
            if args.reproj_weight > 0 and 'keypoints_3d' in batch:
                kp3d = batch['keypoints_3d'].to(device)              # (B,7,3) camera frame
                kp2d = batch['keypoints'].to(device).float()        # (B,7,2) @ IS
                vm = batch['valid_mask'].to(device).float()         # (B,7)
                with torch.no_grad():
                    Rg, tg = kabsch_batch(fk_gt.detach(), kp3d)      # GT camera pose
                cam = torch.einsum('bij,bpj->bpi', Rg, fk_pred) + tg.unsqueeze(1)
                z = cam[..., 2].clamp(min=1e-3)
                u = cam[..., 0] / z * K[:, 0, 0:1] + K[:, 0, 2:3]
                v = cam[..., 1] / z * K[:, 1, 1:2] + K[:, 1, 2:3]
                proj = torch.stack([u, v], -1)
                valid_ok = ((kp3d.abs().sum(-1) > 1e-6) & (kp3d[..., 2] > 0)).float() * vm
                rp = (F.smooth_l1_loss(proj / args.image_size, kp2d / args.image_size,
                                       reduction='none').sum(-1) * valid_ok)[has]
                loss = loss + args.reproj_weight * rp.sum() / valid_ok[has].sum().clamp(min=1)

            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{opt.param_groups[0]['lr']:.1e}"})
        sched.step()

        # ---- validation: per-joint angle MAE (deg) ----
        model.angle_head.eval()
        errs = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                gt = batch['angles'].to(device).clone()
                if gt.shape[1] > 6: gt[:, 6] = 0.0
                has = batch['has_angles'].bool() if 'has_angles' in batch else torch.ones(len(imgs), dtype=torch.bool)
                K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)
                pred = model(imgs, K)['joint_angles']
                d = pred[:, :6] - gt[:, :6]
                d = torch.atan2(torch.sin(d), torch.cos(d)).abs() * 180 / math.pi
                errs.append(d[has.to(device)].cpu())
        errs = torch.cat(errs, dim=0)            # (M,6)
        per_joint = errs.mean(0)
        mae = per_joint.mean().item()
        print(f"Ep{epoch} | val angle MAE(J0-5)={mae:.2f} deg | per-joint=" +
              ",".join(f"{v:.1f}" for v in per_joint))

        if args.use_wandb and _HAS_WANDB:
            log = {'epoch': epoch, 'train_loss': run_loss / len(train_loader),
                   'val_angle_mae': mae, 'lr': opt.param_groups[0]['lr']}
            for j in range(6):
                log[f'val_mae_J{j}'] = per_joint[j].item()
            wandb.log(log)

        torch.save(model.angle_head.state_dict(), out / 'last_angle_head.pth')
        if mae < best_mae:
            best_mae = mae
            torch.save(model.angle_head.state_dict(), out / 'best_angle_head.pth')
            print(f"  -> new best {best_mae:.2f} deg")
    print(f"Done. Best val angle MAE = {best_mae:.2f} deg")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector-ckpt', required=True, help='Stage-1 detector checkpoint (backbone+keypoint_head)')
    p.add_argument('--train-dir', required=True)
    p.add_argument('--val-dir', required=True)
    p.add_argument('--keypoint-names', default=None,
                   help='comma-separated (substring-matched). Meca500: link0,link1,link2,link3,link4,link5,link6')
    p.add_argument('--output-dir', default='./outputs_angle')
    p.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--fk-weight', type=float, default=10.0)
    p.add_argument('--head-type', type=str, default='mlp', choices=['mlp', 'transformer', 'mlp_patch'])
    p.add_argument('--crop-to-robot', action='store_true',
                   help='crop image to robot bbox (train+test), RoboPEPP-style; must match detector ckpt')
    p.add_argument('--crop-margin', type=float, default=1.5)
    p.add_argument('--occlude-aug', type=float, default=0.0,
                   help='train-time occlusion augmentation: with prob 0.5 paste black occluders covering U(0.05,THIS) of the robot RoI (frozen detector -> head learns to handle degraded conf/keypoints)')
    p.add_argument('--kp-drop', type=float, default=0.0,
                   help='keypoint-level occlusion aug: randomly displace+deconfidence keypoints (model_angle.forward kp_drop)')
    p.add_argument('--reproj-weight', type=float, default=0.0,
                   help='RoboTAG-style camera-frame reprojection consistency (project FK(pred) via GT pose, match GT 2D) — adds the 2D<->3D alignment the robot-frame fk_loss lacks')
    p.add_argument('--init-head', default=None, help='warm-start angle_head from this state dict')
    p.add_argument('--kp-jitter', type=float, default=0.0,
                   help='train-time Gaussian px noise on detected 2D before geo/sampling (J0 noise-robustness)')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--use-wandb', action='store_true')
    p.add_argument('--wandb-project', default='dinov3-angle-predictor')
    p.add_argument('--wandb-run-name', default=None)
    main(p.parse_args())
