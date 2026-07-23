"""U1 — KUKA/Baxter: direct-pose vs SOLVER on IDENTICAL frames, converged heads.

Single forward pass per batch, then several pose back-ends evaluated on the same
model outputs so every mode sees exactly the same detections/angles/frames.

Modes
  direct        : head angles + rot-head R,t used directly (current deployed path)
  solver        : spk.solve_batch, rot-head R,t as init, conf_gate (default 0.05),
                  min_kp=6 (repo default -> at most 1 of 7 kp can be hard-gated)
  solver_g4     : same but min_kp=4 -> conf-gate may drop up to 3 of 7 keypoints
  solver_oracle_out : ORACLE diagnostic. conf:=0 on keypoints whose DETECTED 2D is
                  further than --out-px from GT 2D (i.e. the link-confusion cases),
                  min_kp=4. Upper bound of "perfect outlier rejection".
  solver_oracle2d   : ORACLE diagnostic. GT 2D keypoints fed to the solver.

Reports per mode: ADD-AUC@100mm, mean/median ADD, fail rate (ADD>100mm),
median translation error (mm) and median geodesic rotation error (deg) of the
recovered camera->robot pose, so it is directly comparable to the rot-head's
own validation numbers.
"""
import argparse, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(HERE, '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(HERE)
from model_angle import AnglePredictor
from model_v4 import (iiwa7_forward_kinematics, _IIWA7_JOINT_LIMITS,
                      baxter_left_forward_kinematics, _BAXTER_LEFT_JOINT_LIMITS)
from dataset import PoseEstimationDataset
from refine_eval import scale_K, add_auc
import solve_pose_kinematic as spk

ROBOTS = {
    'kuka': dict(
        FK=iiwa7_forward_kinematics, LIMS=_IIWA7_JOINT_LIMITS,
        KP=[f'iiwa7_link_{i}' for i in range(1, 8)],
        JOINTS=[f'iiwa7_joint_{i}' for i in range(1, 8)],
    ),
    'baxter': dict(
        FK=baxter_left_forward_kinematics, LIMS=_BAXTER_LEFT_JOINT_LIMITS,
        KP=['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2'],
        JOINTS=['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2'],
    ),
}


def geometric_K(val_dir, camera_K, original_size, S):
    """TRUE metric intrinsics for the SOLVER, at `S` scale. Reconstruction logic taken from
    Eval/iiwa7_rc_eval.py:geometric_K (verified <0.09 px reprojection over 60 frames), but
    GENERALIZED to pass non-identity K through untouched.

    Whether frame JSONs carry `meta.K` is TREE-DEPENDENT, which is why this bug stayed hidden:
      - datasets/synthetic/{kuka,baxter}_*  : NO meta.K -> PoseEstimationDataset falls back to
        eye(3), so the solver silently received fx=1 instead of ~555 (320x off) and PnP
        collapsed depth.
      - Converted_dataset/DREAM_to_DREAM{,_syn} (Panda): meta.K IS present and real
        (synth 320, rs/kinect/orb 615.5, azure 399.7) -> must NEVER be overwritten.
    Hence: reconstruct ONLY when the input is the identity fallback, else scale and return
    the dataset's own K.

    The eye(3) fallback is LOAD-BEARING for the MODEL (every checkpoint learned its bearing
    features from fx=fy=1), so the discipline is: dataset K -> model, true K -> solver.
    The crop block shifts the principal point in place (camera_K[0,2] -= bx0), so on top of
    eye(3) the returned K[0,2], K[1,2] ARE exactly (-bx0, -by0); combined with
    _camera_settings.json this rebuilds the true crop intrinsics.

    ASSUMES one camera per directory: `_camera_settings.json` is read once and applied to the
    whole batch. Unsafe for a directory mixing several cameras/intrinsics.
    """
    import json
    is_identity = (abs(float(camera_K[0, 0, 0]) - 1.0) < 1e-6 and
                   abs(float(camera_K[0, 1, 1]) - 1.0) < 1e-6)
    if not is_identity:
        return scale_K(camera_K, original_size, S)      # real meta.K (e.g. Panda) — pass through
    p = os.path.join(val_dir, '_camera_settings.json')
    if not os.path.exists(p):
        raise FileNotFoundError(
            f'dataset camera_K is the eye(3) fallback but {p} is missing — cannot recover '
            f'true intrinsics for the solver')
    it = json.load(open(p))['camera_settings'][0]['intrinsic_settings']
    B = camera_K.shape[0]
    Kt = torch.zeros(B, 3, 3)
    Kt[:, 0, 0] = it['fx']; Kt[:, 1, 1] = it['fy']; Kt[:, 2, 2] = 1.0
    Kt[:, 0, 2] = it['cx'] + camera_K[:, 0, 2]      # cx0 - bx0
    Kt[:, 1, 2] = it['cy'] + camera_K[:, 1, 2]      # cy0 - by0
    return scale_K(Kt, original_size, S)


def patch_solver(cfg):
    spk.panda_forward_kinematics = cfg['FK']
    lims = cfg['LIMS']

    def _limits(device, dtype):
        lo = torch.tensor([l for l, _ in lims], device=device, dtype=dtype)
        hi = torch.tensor([h for _, h in lims], device=device, dtype=dtype)
        return lo, hi
    spk.make_limits = _limits
    spk.PANDA_JOINT_MEAN = torch.tensor([(l + h) / 2 for l, h in lims], dtype=torch.float32)


def kabsch_w(A, B, w):
    """Weighted rigid R,t mapping A(B,N,3) -> B(B,N,3). w:(B,N) nonneg."""
    w = w.clamp(min=0) + 1e-8
    wn = (w / w.sum(1, keepdim=True)).unsqueeze(-1)          # (B,N,1)
    ca = (A * wn).sum(1, keepdim=True); cb = (B * wn).sum(1, keepdim=True)
    H = ((A - ca) * wn).transpose(1, 2) @ (B - cb)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.eye(3, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(A.shape[0], 1, 1)
    D[:, 2, 2] = d
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    t = cb.squeeze(1) - torch.einsum('bij,bj->bi', R, ca.squeeze(1))
    return R, t


def geo_deg(Ra, Rb):
    """Geodesic angle (deg) between rotation batches."""
    tr = torch.einsum('bii->b', Ra.transpose(1, 2) @ Rb)
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', required=True, choices=list(ROBOTS))
    ap.add_argument('--detector', required=True)
    ap.add_argument('--angle-head', required=True)
    ap.add_argument('--rot-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=6000)
    ap.add_argument('--iters', type=int, default=250)
    ap.add_argument('--crop-margin', type=float, default=1.5)
    ap.add_argument('--conf-gate', type=float, default=0.05)
    ap.add_argument('--out-px', type=float, default=20.0,
                    help='oracle outlier threshold: |pred2d - gt2d| above this -> conf 0')
    ap.add_argument('--modes', default='direct,solver,solver_g4,solver_oracle_out,solver_oracle2d')
    ap.add_argument('--dump', default=None, help='npz path for per-frame arrays')
    args = ap.parse_args()

    cfg = ROBOTS[args.robot]
    FK = cfg['FK']
    device = torch.device('cuda'); assert torch.cuda.is_available(); IS = args.image_size
    patch_solver(cfg)
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]

    m = AnglePredictor(args.model_name, IS, fix_joint7_zero=True, head_type='mlp',
                       with_rotation=True, with_translation=True).to(device).eval()
    sd = torch.load(args.detector, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.angle_head, map_location=device))
    m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=cfg['KP'], image_size=(IS, IS),
                               heatmap_size=(IS, IS), augment=False, include_angles=True, sigma=2.5,
                               crop_to_robot=True, crop_margin=args.crop_margin,
                               angle_joint_names=cfg['JOINTS'])
    if args.max_frames and args.max_frames < len(ds):
        ds.samples = ds.samples[::max(1, len(ds.samples) // args.max_frames)][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    acc = {k: dict(add=[], terr=[], rerr=[], reproj=[]) for k in modes}
    reproj_px_all = []          # detected-2D reprojection error vs GT 2D (px), per keypoint
    outlier_frac = []           # fraction of kp beyond out-px, per frame
    n = 0
    t0 = time.time()

    for batch in tqdm(loader, desc=f'u1-{args.robot}'):
        img = batch['image'].to(device)
        gt_ang = batch['angles'].to(device)
        gt3d = batch['keypoints_3d'].to(device)                 # (B,N,3) camera frame, m
        gt2d = batch['keypoints'].to(device).float()            # (B,N,2) px in 512 crop
        vmask = batch['valid_mask'].to(device).float()          # (B,N)
        # DISCIPLINE: dataset K -> model (checkpoints learned bearings from the eye(3) fallback);
        # true metric K -> solver (identity K puts a 320x-wrong focal into PnP, collapsing depth).
        K = scale_K(batch['camera_K'], batch['original_size'], IS).to(device)
        K_true = geometric_K(args.val_dir, batch['camera_K'], batch['original_size'], IS).to(device)
        with torch.no_grad():
            o = m(img, K)
        init_ang = o['joint_angles']
        kp2d = o['keypoints_2d']; conf = o['confidence']
        R_h = o['rot_matrix'].float(); t_h = o['trans'].float()
        B = img.shape[0]

        # GT camera pose from the GT keypoints themselves (gauge-consistent with ADD)
        fk_gt = FK(gt_ang.double()).float()
        R_gt, t_gt = kabsch_w(fk_gt, gt3d, vmask)

        # 2D quality bookkeeping
        d2 = (kp2d - gt2d).norm(dim=-1)                          # (B,N)
        bad = (d2 > args.out_px) & (vmask > 0)
        for b in range(B):
            vv = vmask[b] > 0
            if vv.any():
                reproj_px_all.extend(d2[b][vv].tolist())
                outlier_frac.append(float(bad[b].float().sum().item() / vv.float().sum().item()))

        for mode in modes:
            if mode == 'direct':
                fk_h = FK(init_ang.double()).float()
                R_p, t_p = R_h, t_h
                kp_cam = torch.einsum('bij,bnj->bni', R_p, fk_h) + t_p.unsqueeze(1)
            else:
                use2d, usec, min_kp = kp2d, conf, 6
                t_init, freeze = t_h, False
                if mode == 'solver_g4':
                    min_kp = 4
                elif mode == 'solver_oracle_out':
                    usec = conf.clone(); usec[bad] = 0.0; min_kp = 4
                elif mode == 'solver_oracle2d':
                    use2d = gt2d; usec = vmask.clamp(min=1e-3)
                elif mode == 'solver_tpnp':
                    t_init = None                       # Panda deployed config: t from PnP
                elif mode == 'solver_freeze':
                    freeze = True                       # pose-only refine, angles held at head
                elif mode == 'solver_freeze_tpnp':
                    freeze = True; t_init = None
                elif mode == 'solver_freeze_oracle_out':
                    freeze = True; usec = conf.clone(); usec[bad] = 0.0; min_kp = 4
                with torch.enable_grad():
                    _, kp_cam, _, R_p, t_p = spk.solve_batch(
                        use2d, usec, K_true, fix_joint7=True, iters=args.iters, lr=2e-2,
                        img_size=IS, device=device, prior_w=0.0, theta_init=init_ang,
                        cov_inv=None, conf_gate=args.conf_gate, min_kp=min_kp,
                        R_init=R_h, t_init=t_init, freeze_theta=freeze, return_pose=True)

            per_j = (kp_cam - gt3d).norm(dim=-1)
            te = (t_p - t_gt).norm(dim=-1) * 1000.0
            re = geo_deg(R_p.float(), R_gt)
            # final reprojection of the recovered pose against the DETECTED 2D, in true-K pixels
            # (this is the only signal a deployable guard could use — no GT involved).
            z = kp_cam[..., 2].clamp(min=1e-3)
            uu = kp_cam[..., 0] / z * K_true[:, 0, 0:1] + K_true[:, 0, 2:3]
            vv_ = kp_cam[..., 1] / z * K_true[:, 1, 1:2] + K_true[:, 1, 2:3]
            rp = (torch.stack([uu, vv_], -1) - kp2d).norm(dim=-1).mean(dim=1)
            valid = vmask > 0
            for b in range(B):
                if valid[b].any():
                    acc[mode]['add'].append(float(per_j[b][valid[b]].mean().item()))
                    acc[mode]['terr'].append(float(te[b].item()))
                    acc[mode]['rerr'].append(float(re[b].item()))
                    acc[mode]['reproj'].append(float(rp[b].item()))
        n += B

    dt = time.time() - t0
    px = np.array(reproj_px_all); of = np.array(outlier_frac)
    print(f"\n{'='*94}")
    print(f"  U1  {args.robot.upper()}  |  {n} frames  |  {args.val_dir}  |  {dt/60:.1f} min")
    print(f"  detected-2D vs GT-2D: median {np.median(px):.2f} px, mean {px.mean():.2f} px, "
          f">{args.out_px:.0f}px outliers {100*(px > args.out_px).mean():.1f}% of kp, "
          f"{100*(of > 0).mean():.1f}% of frames affected")
    print(f"{'='*94}")
    print(f"  {'mode':<20}{'ADD-AUC':>9}{'meanADD':>10}{'medADD':>9}{'fail>100':>10}"
          f"{'med t-err':>11}{'med R-err':>11}{'mean t':>9}{'mean R':>8}")
    print(f"  {'':<20}{'':>9}{'(mm)':>10}{'(mm)':>9}{'(%)':>10}{'(mm)':>11}{'(deg)':>11}{'(mm)':>9}{'(deg)':>8}")
    print('-'*94)
    for mode in modes:
        a = np.array(acc[mode]['add']); te = np.array(acc[mode]['terr']); re = np.array(acc[mode]['rerr'])
        print(f"  {mode:<20}{add_auc(a):>9.4f}{a.mean()*1000:>10.1f}{np.median(a)*1000:>9.1f}"
              f"{100*(a > 0.1).mean():>10.1f}{np.median(te):>11.1f}{np.median(re):>11.2f}"
              f"{te.mean():>9.1f}{re.mean():>8.2f}")
    print('='*94)

    # ---- divergence-tail analysis: is the remaining tail DETECTABLE without GT? ----
    print("\n  Divergence tail (mean >> median ?) and whether reprojection could gate it:")
    print(f"  {'mode':<20}{'p50':>8}{'p90':>9}{'p95':>9}{'p99':>9}{'max':>10}"
          f"{'reproj|good':>13}{'reproj|bad':>12}{'AUC if':>9}")
    print(f"  {'':<20}{'(mm)':>8}{'(mm)':>9}{'(mm)':>9}{'(mm)':>9}{'(mm)':>10}"
          f"{'(px)':>13}{'(px)':>12}{'gated':>9}")
    print('-'*94)
    base_add = np.array(acc['direct']['add']) if 'direct' in acc else None
    for mode in modes:
        a = np.array(acc[mode]['add']); rp = np.array(acc[mode]['reproj'])
        good, badm = a <= 0.1, a > 0.1
        rg = np.median(rp[good]) if good.any() else float('nan')
        rb = np.median(rp[badm]) if badm.any() else float('nan')
        # deployable guard: fall back to direct-pose wherever solver reprojection exceeds thr
        if base_add is not None and len(base_add) == len(a):
            thr = np.percentile(rp, 90)
            hyb = np.where(rp > thr, base_add, a)
            gated = add_auc(hyb)
        else:
            gated = float('nan')
        print(f"  {mode:<20}{np.percentile(a,50)*1000:>8.1f}{np.percentile(a,90)*1000:>9.1f}"
              f"{np.percentile(a,95)*1000:>9.1f}{np.percentile(a,99)*1000:>9.1f}{a.max()*1000:>10.1f}"
              f"{rg:>13.2f}{rb:>12.2f}{gated:>9.4f}")
    print('='*94)
    if args.dump:
        np.savez(args.dump, **{f'{mo}_{k}': np.array(v) for mo in modes for k, v in acc[mo].items()})
        print(f"  per-frame arrays -> {args.dump}")


if __name__ == '__main__':
    main()
