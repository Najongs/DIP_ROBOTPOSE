"""Train the whole-body Baxter angle head (12 joints) on a frozen 17-keypoint detector.

Mirrors train_angle.py but for the DREAM baxter WHOLE-BODY task: 17 keypoints, 12 observable
joints (left {s0,s1,e0,e1,w0,w1} + right {s0,s1,e0,e1,w0,w1}; both w2 fixed 0 — the hands sit on
the w2 roll axis, so w2 moves no keypoint). Loss = sin/cos SmoothL1 to GT + robot-frame FK
consistency. Val = per-joint angle MAE (deg). Only the angle head trains.
"""
import argparse, math
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_angle import AnglePredictor
from model_v4 import baxter_forward_kinematics
from dataset import PoseEstimationDataset

try:
    import wandb; _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

KP17 = ['torso_t0', 'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2',
        'left_hand', 'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1',
        'right_w2', 'right_hand']
ANG12 = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1',
         'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1']
NA = 12


def scale_K(camera_K, original_size, hm):
    K = camera_K.clone().float()
    for b in range(K.shape[0]):
        ow, oh = float(original_size[b][0]), float(original_size[b][1])
        K[b, 0, 0] *= hm / ow; K[b, 1, 1] *= hm / oh
        K[b, 0, 2] *= hm / ow; K[b, 1, 2] *= hm / oh
    return K


def main(args):
    device = torch.device('cuda'); assert torch.cuda.is_available(), "need GPU (check UUID)"
    mk = lambda d, aug: PoseEstimationDataset(
        data_dir=d, keypoint_names=KP17, image_size=(args.image_size, args.image_size),
        heatmap_size=(args.image_size, args.image_size), augment=aug, aug_level='strong',
        include_angles=True, sigma=2.5, crop_to_robot=args.crop_to_robot, crop_margin=args.crop_margin,
        angle_joint_names=ANG12)
    train_ds, val_ds = mk(args.train_dir, True), mk(args.val_dir, False)
    if args.max_train and args.max_train < len(train_ds):
        train_ds.samples = train_ds.samples[::max(1, len(train_ds.samples) // args.max_train)][:args.max_train]
        print(f"==> subsampled train to {len(train_ds)} frames")
    if args.max_val and args.max_val < len(val_ds):
        val_ds.samples = val_ds.samples[::max(1, len(val_ds.samples) // args.max_val)][:args.max_val]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = AnglePredictor(args.model_name, args.image_size, fix_joint7_zero=False,
                           head_type='mlp', num_kp=17, num_ang=NA).to(device)
    ckpt = torch.load(args.detector_ckpt, map_location=device)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    msd = model.state_dict()
    loaded = {k: v for k, v in ckpt.items() if k in msd and v.shape == msd[k].shape}
    model.load_state_dict(loaded, strict=False)
    n_det = sum(1 for k in loaded if k.startswith('backbone.') or k.startswith('keypoint_head.'))
    print(f"==> Loaded {len(loaded)} tensors ({n_det} into backbone+keypoint_head)")
    model.freeze_detector()
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert not any(p.requires_grad for p in model.keypoint_head.parameters())

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
            gt = batch['angles'].to(device)                # (B,12)
            has = batch['has_angles'].to(device).bool()
            K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)
            out_d = model(imgs, K, kp_jitter=args.kp_jitter)
            sc = out_d['sin_cos']                            # (B,12,2)
            gt_sc = torch.stack([torch.sin(gt), torch.cos(gt)], dim=-1)
            sc_loss = F.smooth_l1_loss(sc[has], gt_sc[has])
            loss = sc_loss
            if args.fk_weight > 0:
                fk_pred = baxter_forward_kinematics(out_d['joint_angles'])   # (B,17,3)
                fk_gt = baxter_forward_kinematics(gt)
                loss = loss + args.fk_weight * F.mse_loss(fk_pred[has], fk_gt[has])
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{opt.param_groups[0]['lr']:.1e}"})
        sched.step()

        model.angle_head.eval()
        errs = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                gt = batch['angles'].to(device)
                K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)
                pred = model(imgs, K)['joint_angles']        # (B,12)
                d = torch.atan2(torch.sin(pred - gt), torch.cos(pred - gt)).abs() * 180 / math.pi
                errs.append(d.cpu())
        errs = torch.cat(errs, dim=0)
        per_joint = errs.mean(0); mae = per_joint.mean().item()
        print(f"Ep{epoch} | val MAE(12)={mae:.2f} deg | per-joint=" +
              ",".join(f"{v:.1f}" for v in per_joint))
        if args.use_wandb and _HAS_WANDB:
            log = {'epoch': epoch, 'train_loss': run_loss / len(train_loader),
                   'val_angle_mae': mae, 'lr': opt.param_groups[0]['lr']}
            for j, n in enumerate(ANG12):
                log[f'val_mae/{n}'] = per_joint[j].item()
            wandb.log(log)
        torch.save(model.angle_head.state_dict(), out / 'last_angle_head.pth')
        if mae < best_mae:
            best_mae = mae
            torch.save(model.angle_head.state_dict(), out / 'best_angle_head.pth')
            print(f"  -> new best {best_mae:.2f} deg")
    print(f"Done. Best val angle MAE = {best_mae:.2f} deg")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector-ckpt', required=True)
    p.add_argument('--train-dir', required=True); p.add_argument('--val-dir', required=True)
    p.add_argument('--output-dir', default='./outputs_angle')
    p.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    p.add_argument('--image-size', type=int, default=512); p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=60); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min-lr', type=float, default=1e-6); p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--fk-weight', type=float, default=10.0)
    p.add_argument('--kp-jitter', type=float, default=0.0)
    p.add_argument('--crop-to-robot', action='store_true'); p.add_argument('--crop-margin', type=float, default=1.5)
    p.add_argument('--max-train', type=int, default=0, help='subsample training frames (0=all 105k)')
    p.add_argument('--max-val', type=int, default=0, help='subsample val frames (0=all)')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--use-wandb', action='store_true')
    p.add_argument('--wandb-project', default='dinov3-baxter-fullbody-angle'); p.add_argument('--wandb-run-name', default=None)
    main(p.parse_args())
