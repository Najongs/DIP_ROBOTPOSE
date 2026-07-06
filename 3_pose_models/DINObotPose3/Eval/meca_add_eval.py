"""Meca500 keypoint-path ADD-AUC baseline (6-DOF).

Reuses the SOTA kinematic solver (solve_pose_kinematic.solve_batch) WITHOUT editing it, by
monkeypatching the module's FK + joint limits to the (verified) batched Meca500 torch FK. The
solver skips its Panda-hardcoded theta expand/mean whenever theta_init is supplied, so passing the
angle-head init (B,6) is enough. fix_joint7=False (Meca has 6 real joints).

Input crop = GT-keypoint bbox (crop_to_robot) to isolate the pose-lift from bbox localization —
this is the ceiling of the keypoint path given the current detector.

  --oracle-2d : feed GT 2D keypoints to the solver instead of detected (upper bound: how good is
                the KEYPOINT PARAMETERIZATION itself, independent of detector precision). The
                wrist joints stay unobservable either way (see robot_fk.py Jacobian).
"""
import argparse, os, sys, math, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(__file__)
TRAIN = os.path.abspath(os.path.join(HERE, '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(HERE)
from model_angle import AnglePredictor
from dataset import PoseEstimationDataset
from refine_eval import scale_K, add_auc, wrapped_abs_deg
import solve_pose_kinematic as spk
from robot_fk import meca500_forward_kinematics, fr5_forward_kinematics

_ROBOT_FK = {'meca500': meca500_forward_kinematics, 'fr5': fr5_forward_kinematics}
FK = meca500_forward_kinematics   # set per --robot in main()

MECA_LIMITS = [(-3.05, 3.05), (-1.22, 1.57), (-2.36, 1.22), (-2.97, 2.97), (-2.01, 2.01), (-3.14, 3.14)]


def kabsch_batch(A, B):
    """Rigid R,t mapping A(B,N,3) onto B(B,N,3). Returns R(B,3,3), t(B,3)."""
    ca = A.mean(1, keepdim=True); cb = B.mean(1, keepdim=True)
    H = (A - ca).transpose(1, 2) @ (B - cb)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.eye(3, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(A.shape[0], 1, 1)
    D[:, 2, 2] = d
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    t = cb.squeeze(1) - torch.einsum('bij,bj->bi', R, ca.squeeze(1))
    return R, t


def _patch_solver_for_meca():
    spk.panda_forward_kinematics = meca500_forward_kinematics
    def _meca_limits(device, dtype):
        lo = torch.tensor([l for l, _ in MECA_LIMITS], device=device, dtype=dtype)
        hi = torch.tensor([h for _, h in MECA_LIMITS], device=device, dtype=dtype)
        return lo, hi
    spk.make_limits = _meca_limits
    spk.PANDA_JOINT_MEAN = torch.tensor([(l + h) / 2 for l, h in MECA_LIMITS], dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--angle-head', required=True)
    ap.add_argument('--rot-head', default=None, help='trained rot head -> R_init/t_init for the solver (the SOTA basin-pin)')
    ap.add_argument('--robot', default='meca500', choices=['meca500', 'fr5'])
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--keypoint-names', default='link0,link1,link2,link3,link4,link5,link6')
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=1500)
    ap.add_argument('--iters', type=int, default=250)
    ap.add_argument('--crop-margin', type=float, default=1.5)
    ap.add_argument('--oracle-2d', action='store_true')
    ap.add_argument('--oracle-R', action='store_true', help='pin R_init/t_init via Kabsch(FK(gt_angles), gt3d) — isolates the missing rot-head')
    ap.add_argument('--head-direct', action='store_true',
                    help='TRUST the angle head angles (incl. wrist, which the keypoint solver corrupts); '
                         'get (R,t) via PnP of FK(head-angles) vs detected 2D — no joint refinement')
    ap.add_argument('--cov-pnp', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available(); IS = args.image_size
    KPN = args.keypoint_names.split(',')
    global FK
    FK = _ROBOT_FK[args.robot]
    _patch_solver_for_meca()
    if args.robot != 'meca500':                 # keep solver-path FK consistent if used
        spk.panda_forward_kinematics = FK

    m = AnglePredictor(args.model_name, IS, head_type='mlp',
                       with_rotation=bool(args.rot_head), with_translation=bool(args.rot_head)).to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.angle_head, map_location=device))
    if args.rot_head:
        m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=KPN, image_size=(IS, IS), heatmap_size=(IS, IS),
                               augment=False, include_angles=True, sigma=2.5,
                               crop_to_robot=True, crop_margin=args.crop_margin)
    if args.max_frames and args.max_frames < len(ds): ds.samples = ds.samples[::max(1, len(ds.samples)//args.max_frames)][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    raw_err = torch.zeros(6); ref_err = torch.zeros(6); n = 0; adds = []
    for batch in tqdm(loader, desc='meca-add'):
        img = batch['image'].to(device)
        gt = batch['angles'].to(device)[:, :6]
        gt3d = batch['keypoints_3d'].to(device)                 # (B,7,3) camera frame, meters
        K = scale_K(batch['camera_K'], batch['original_size'], IS).to(device)
        with torch.no_grad():
            o = m(img, K)
        init_ang = o['joint_angles'][:, :6]
        if args.oracle_2d:
            kp2d = batch['keypoints'].to(device).float()        # (B,7,2) GT in IS space
            conf = batch['valid_mask'].to(device).float().clamp(min=1e-3)
        else:
            kp2d = o['keypoints_2d']; conf = o['confidence']
        cov_inv = spk.heatmap_cov_inv(o['heatmaps_2d'], kp2d) if (args.cov_pnp and not args.oracle_2d) else None
        if args.head_direct:
            # trust head angles (incl. wrist, which the solver corrupts). Global pose (R,t):
            #  - with --rot-head: robust learned R (no PnP outliers) + t solved by reprojection
            #  - else: PnP of FK(head-angles) vs detected 2D
            fk_h = FK(init_ang.double()).float()   # (B,7,3) robot frame
            if args.rot_head:
                R_h = o['rot_matrix'].float()
                # solve t (3 params) minimizing reprojection, R fixed — kills PnP's outlier tail
                t_h = o['trans'].float().clone().detach().requires_grad_(True)
                opt_t = torch.optim.Adam([t_h], lr=5e-3)
                cam0 = torch.einsum('bij,bnj->bni', R_h, fk_h)
                w = conf.clamp(min=1e-3)
                for _ in range(80):
                    cam = cam0 + t_h.unsqueeze(1); z = cam[..., 2].clamp(min=1e-3)
                    u = cam[..., 0] / z * K[:, 0, 0:1] + K[:, 0, 2:3]
                    v = cam[..., 1] / z * K[:, 1, 1:2] + K[:, 1, 2:3]
                    proj = torch.stack([u, v], -1)
                    loss = (((proj - kp2d).norm(dim=-1)) * w).sum() / w.sum()
                    opt_t.zero_grad(); loss.backward(); opt_t.step()
                t_h = t_h.detach()
            else:
                Rn, tn, _ = spk.pnp_init(kp2d.detach().cpu().numpy(), fk_h.detach().cpu().numpy(),
                                         K.detach().cpu().numpy(), conf.detach().cpu().numpy(),
                                         min_kp=6, pnp_drop=1)
                R_h = torch.from_numpy(Rn).float().to(device); t_h = torch.from_numpy(tn).float().to(device)
            refined = init_ang
            kp_cam = torch.einsum('bij,bnj->bni', R_h, fk_h) + t_h.unsqueeze(1)
        else:
            R_init = t_init = None
            if args.oracle_R:
                fk_gt = FK(gt.double()).float()   # (B,7,3) robot frame
                R_init, t_init = kabsch_batch(fk_gt, gt3d)                # base->camera from GT
            elif args.rot_head:
                R_init = o.get('rot_matrix'); t_init = o.get('trans')     # learned basin-pin
            with torch.enable_grad():
                refined, kp_cam, reproj = spk.solve_batch(kp2d, conf, K, fix_joint7=False, iters=args.iters,
                                                          lr=2e-2, img_size=IS, device=device, prior_w=0.0,
                                                          theta_init=init_ang, cov_inv=cov_inv,
                                                          R_init=R_init, t_init=t_init)
        raw_err += wrapped_abs_deg(init_ang, gt).sum(0).cpu()
        ref_err += wrapped_abs_deg(refined[:, :6], gt).sum(0).cpu()
        valid = (gt3d.abs().sum(-1) > 0)
        per_j = (kp_cam - gt3d).norm(dim=-1)
        for b in range(img.shape[0]):
            if valid[b].any(): adds.append(float(per_j[b][valid[b]].mean().item()))
        n += img.shape[0]

    raw = (raw_err / n).numpy(); ref = (ref_err / n).numpy(); adds = np.array(adds)
    tag = 'ORACLE-2D' if args.oracle_2d else 'detected-2D'
    print(f"\n{'='*58}\n  {args.robot.upper()} keypoint-path ADD  [{tag}]  ({n} frames)  {os.path.basename(args.val_dir)}\n{'='*58}")
    print(f"  {'joint':<6}{'raw head':>10}{'refined':>10}{'delta':>9}   (deg MAE; J3-5 unobservable)")
    for j in range(6):
        tagj = '  <- unobs' if j >= 3 else ''
        print(f"  J{j:<5}{raw[j]:>10.2f}{ref[j]:>10.2f}{ref[j]-raw[j]:>+9.2f}{tagj}")
    print('-'*58)
    print(f"  [Pose] ADD-AUC@100mm: {add_auc(adds):.4f} | mean ADD {adds.mean()*1000:.1f}mm | "
          f"median {np.median(adds)*1000:.1f}mm ({len(adds)} frames)")
    print('='*58)


if __name__ == '__main__':
    main()
