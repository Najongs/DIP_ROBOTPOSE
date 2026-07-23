"""Whole-body Baxter (17-keypoint) ADD-AUC — apples-to-apples with RoboPEPP / the DREAM benchmark.

Predicts the 12 observable joints (left+right {s0,s1,e0,e1,w0,w1}; both w2 fixed 0 — the hands
sit on the w2 roll axis) + robot->camera R,t, runs baxter_forward_kinematics (12 -> 17 keypoints),
and scores ADD over ALL 17 keypoints (all-keypoints convention, NOT the in-frame valid_mask).

The kinematic solver in solve_pose_kinematic.py is hardcoded to a 7-joint arm (fix_joint7); it is
NOT edited here. Instead this script runs a self-contained refine that is FK-agnostic:
  --mode solver      : refine theta(12)+R+t by conf-weighted Huber reprojection (TRUE K), rot-head
                       R+t init, do-no-harm guard (revert if reproj worsens). [default / headline]
  --mode head-direct : trust head angles; R = rot-head, refine only t by reprojection.
  --mode direct      : trust head angles + rot-head R AND t directly (no optimization).
Diagnostics: --oracle-angle (GT angles), --oracle-2d (GT 2D), --gt-pose (Kabsch FK(gt)->gt3d ceiling).
"""
import argparse, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(__file__)
TRAIN = os.path.abspath(os.path.join(HERE, '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(HERE)
from model_angle import AnglePredictor, rot6d_to_matrix
from model_v4 import baxter_forward_kinematics, _BAXTER_FB_JOINT_LIMITS
from dataset import PoseEstimationDataset
from refine_eval import scale_K, add_auc, geometric_K, wrapped_abs_deg

FK = baxter_forward_kinematics
KP17 = ['torso_t0', 'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2',
        'left_hand', 'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1',
        'right_w2', 'right_hand']
ANG12 = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1',
         'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1']


def matrix_to_rot6d(R):
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)     # first two columns


def kabsch_batch(A, B):
    ca = A.mean(1, keepdim=True); cb = B.mean(1, keepdim=True)
    H = (A - ca).transpose(1, 2) @ (B - cb)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.eye(3, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(A.shape[0], 1, 1)
    D[:, 2, 2] = d
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    t = cb.squeeze(1) - torch.einsum('bij,bj->bi', R, ca.squeeze(1))
    return R, t


def project(fk, R, t, K):
    cam = torch.einsum('bij,bnj->bni', R, fk) + t.unsqueeze(1)
    z = cam[..., 2].clamp(min=1e-3)
    u = cam[..., 0] / z * K[:, 0, 0:1] + K[:, 0, 2:3]
    v = cam[..., 1] / z * K[:, 1, 1:2] + K[:, 1, 2:3]
    return torch.stack([u, v], -1), cam


def reproj_px(fk, R, t, K, kp2d, w):
    uv, _ = project(fk, R, t, K)
    return ((uv - kp2d).norm(dim=-1) * w).sum(1) / w.sum(1).clamp(min=1e-6)   # (B,)


def refine_t(fk, R, t0, K, kp2d, w, iters=100, lr=5e-3):
    t = t0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([t], lr=lr)
    cam0 = torch.einsum('bij,bnj->bni', R, fk)
    for _ in range(iters):
        cam = cam0 + t.unsqueeze(1); z = cam[..., 2].clamp(min=1e-3)
        u = cam[..., 0] / z * K[:, 0, 0:1] + K[:, 0, 2:3]
        v = cam[..., 1] / z * K[:, 1, 1:2] + K[:, 1, 2:3]
        loss = (((torch.stack([u, v], -1) - kp2d).norm(dim=-1)) * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return t.detach()


def refine_full(theta0, R0, t0, K, kp2d, w, lo, hi, iters=200, lr=2e-2, anchor_w=1e-2):
    """Refine theta(12)+R+t by conf-weighted Huber reprojection (true K). Anchor theta to init
    (under-observable distal joints keep the learned estimate). Returns theta,R,t and reproj."""
    B = theta0.shape[0]
    p = theta0.clone().detach().requires_grad_(True)
    d6 = matrix_to_rot6d(R0).clone().detach().requires_grad_(True)
    t = t0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([p, d6, t], lr=lr)
    theta_a = theta0.detach().clone()
    for it in range(iters):
        theta = torch.max(torch.min(p, hi), lo)
        fk = FK(theta)
        R = rot6d_to_matrix(d6)
        uv, _ = project(fk, R, t, K)
        err = (uv - kp2d) / 512.0
        loss_per = torch.nn.functional.huber_loss(err, torch.zeros_like(err), delta=0.01,
                                                  reduction='none').sum(-1)   # (B,N)
        if it > 30:
            resid = (uv - kp2d).norm(dim=-1).detach()
            robust = 64.0 / (64.0 + resid ** 2)
            ww = w * robust
        else:
            ww = w
        ww = ww / ww.sum(1, keepdim=True).clamp(min=1e-6)
        loss = (ww * loss_per).sum(1).mean() + anchor_w * ((theta - theta_a) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    theta = torch.max(torch.min(p, hi), lo).detach()
    R = rot6d_to_matrix(d6).detach(); t = t.detach()
    return theta, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--angle-head', required=True)
    ap.add_argument('--rot-head', required=True)
    ap.add_argument('--val-dir', default='/home/najo/NAS/DIP/datasets/synthetic/baxter_synth_test_dr')
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=0, help='0 = all frames')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--crop-margin', type=float, default=1.5)
    ap.add_argument('--conf-gate', type=float, default=0.05)
    ap.add_argument('--mode', default='solver', choices=['solver', 'head-direct', 'direct', 'all'])
    ap.add_argument('--oracle-angle', action='store_true')
    ap.add_argument('--oracle-2d', action='store_true')
    ap.add_argument('--gt-pose', action='store_true', help='Kabsch FK(gt)->gt3d ceiling (no detector)')
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available(); IS = args.image_size
    lo = torch.tensor([l for l, _ in _BAXTER_FB_JOINT_LIMITS], device=device)
    hi = torch.tensor([h for _, h in _BAXTER_FB_JOINT_LIMITS], device=device)

    m = AnglePredictor(args.model_name, IS, fix_joint7_zero=False, head_type='mlp',
                       num_kp=17, num_ang=12, with_rotation=True, with_translation=True).to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.angle_head, map_location=device))
    m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=KP17, image_size=(IS, IS), heatmap_size=(IS, IS),
                               augment=False, include_angles=True, sigma=2.5,
                               crop_to_robot=True, crop_margin=args.crop_margin, angle_joint_names=ANG12)
    if args.max_frames and args.max_frames < len(ds):
        ds.samples = ds.samples[::max(1, len(ds.samples) // args.max_frames)][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print(f"frames: {len(ds)} | mode: {args.mode} | val: {args.val_dir}")

    modes = ['solver', 'head-direct', 'direct'] if args.mode == 'all' else [args.mode]
    adds = {mo: [] for mo in modes}
    if args.gt_pose:
        adds['gt-pose'] = []
    raw_err = torch.zeros(12); ref_err = torch.zeros(12); n = 0

    for batch in tqdm(loader, desc='fb-add'):
        img = batch['image'].to(device)
        gt = batch['angles'].to(device)                        # (B,12)
        gt3d = batch['keypoints_3d'].to(device)                # (B,17,3) camera frame, m
        Kmod = scale_K(batch['camera_K'], batch['original_size'], IS).to(device)
        Ktrue = geometric_K(args.val_dir, batch['camera_K'], batch['original_size'], IS).to(device)
        with torch.no_grad():
            o = m(img, Kmod)
        init_ang = o['joint_angles'] if not args.oracle_angle else gt   # (B,12)
        if args.oracle_2d:
            kp2d = batch['keypoints'].to(device).float()
            conf = batch['valid_mask'].to(device).float().clamp(min=1e-3)
        else:
            kp2d = o['keypoints_2d']; conf = o['confidence']
        w = conf.clamp(min=1e-3)
        if args.conf_gate > 0:
            w = w * (conf >= args.conf_gate).float()
        fk_h = FK(init_ang.double()).float()
        R_h = o['rot_matrix'].float(); t_h = o['trans'].float()

        # all-keypoints ADD over all 17
        B = img.shape[0]
        valid17 = (gt3d.abs().sum(-1) > 1e-6)                   # (B,17) all True in DREAM baxter
        def add_from(kp_cam):
            per = (kp_cam - gt3d).norm(dim=-1)                  # (B,17)
            return [(per[b][valid17[b]].mean().item()) for b in range(B)]

        if 'gt-pose' in adds:
            fk_gt = FK(gt.double()).float()
            Rg, tg = kabsch_batch(fk_gt, gt3d)
            adds['gt-pose'] += add_from(torch.einsum('bij,bnj->bni', Rg, fk_gt) + tg.unsqueeze(1))

        for mo in modes:
            if mo == 'direct':
                kp_cam = torch.einsum('bij,bnj->bni', R_h, fk_h) + t_h.unsqueeze(1)
                refined = init_ang
            elif mo == 'head-direct':
                t_ref = refine_t(fk_h, R_h, t_h, Ktrue, kp2d, w, iters=max(100, args.iters // 2))
                kp_cam = torch.einsum('bij,bnj->bni', R_h, fk_h) + t_ref.unsqueeze(1)
                refined = init_ang
            else:  # solver
                with torch.enable_grad():
                    theta, R, t = refine_full(init_ang.float(), R_h, t_h, Ktrue, kp2d, w, lo, hi,
                                              iters=args.iters)
                # do-no-harm guard: revert to rot-head init where refine worsened reprojection
                r_init = reproj_px(fk_h, R_h, t_h, Ktrue, kp2d, w)
                fk_ref = FK(theta)
                r_ref = reproj_px(fk_ref, R, t, Ktrue, kp2d, w)
                worse = (r_ref > r_init).view(-1, 1, 1)
                kp_ref = torch.einsum('bij,bnj->bni', R, fk_ref) + t.unsqueeze(1)
                kp_init = torch.einsum('bij,bnj->bni', R_h, fk_h) + t_h.unsqueeze(1)
                kp_cam = torch.where(worse, kp_init, kp_ref)
                refined = torch.where(worse.view(-1, 1), init_ang, theta)
            adds[mo] += add_from(kp_cam)
            if mo == modes[0]:
                raw_err += wrapped_abs_deg(init_ang, gt).sum(0).cpu()
                ref_err += wrapped_abs_deg(refined, gt).sum(0).cpu()
        n += B

    tag = ('ORACLE-2D ' if args.oracle_2d else '') + ('ORACLE-ANGLE ' if args.oracle_angle else '')
    raw = (raw_err / n).numpy(); ref = (ref_err / n).numpy()
    print(f"\n{'='*68}\n  Baxter WHOLE-BODY 17-kp ADD  [{tag}]  ({n} frames, all-keypoints)\n{'='*68}")
    print(f"  angle MAE (12 joints, deg):  raw head mean = {raw.mean():.2f}")
    print("   " + "  ".join(f"{ANG12[j].replace('_','')}:{raw[j]:.1f}" for j in range(12)))
    print('-' * 68)
    for mo in list(adds.keys()):
        a = np.array(adds[mo])
        print(f"  [{mo:11s}] ADD-AUC@100mm = {add_auc(a):.4f} | mean {a.mean()*1000:6.1f}mm | "
              f"median {np.median(a)*1000:6.1f}mm")
    print('=' * 68)


if __name__ == '__main__':
    main()
