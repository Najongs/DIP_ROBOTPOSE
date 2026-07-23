"""KUKA iiwa7 render-and-compare on top of the TRUE-K kinematic solver.

Meshes come from the RoboPEPP iiwa_description URDF via iiwa7_render.py (read-only clone).

  --mode gtcheck : CORRECTNESS GATE. Render at the GT pose, IoU vs the SAM mask. Measured 0.858
                   mean / 0.869 median (Baxter 0.82, Panda comparable) => mesh + FK frame
                   convention are correct. A low number here is a gauge bug, not RC tuning.
  --mode rc      : refinement + before/after ADD.

BASELINE HISTORY — READ BEFORE QUOTING ANY NUMBER FROM THIS FILE.
The kuka/baxter synth trees carry no `meta.K`, so PoseEstimationDataset falls back to eye(3).
That fallback is LOAD-BEARING for the MODEL but catastrophic for any GEOMETRIC solve: it put a
320x-wrong focal into PnP/refine and collapsed solved depth. That, not "link confusion", is why
the solver used to look worse than the rot-head and why `direct_pose` was ever the baseline.
Fixing intrinsics alone (refine_eval.geometric_K) moved KUKA 0.390 -> 0.686 ADD-AUC
(median ADD 57.0 -> 13.1 mm, t-err 54.1 -> 16.8 mm). So:
  * --init solver (DEFAULT) starts RC from the true-K solver — the current baseline.
  * --init direct reproduces the OLD rot-head-only pose, for A/B only. Any RC gain measured
    against `direct` is inflated by the intrinsics bug and MUST NOT be quoted as an RC result.
The earlier 50-frame sweep in this file's history (A/B/C/D configs, +0.29 "gain" over a 0.2804
direct-pose baseline) is superseded for exactly that reason: it was measured on the broken init.

SAFETY (a previous Baxter RC attempt without these collapsed 0.2621 -> 0.0089, 500/500 frames
~3x worse — the fingerprint of free-optimization drift on the silhouette's flat depth manifold):
  1. meshes posed by the URDF FK chain (iiwa7_render), NOT model_v4's position-only data-fit
     chain whose intermediate frame orientations are gauge. R,t come from Kabsch of URDF-FK onto
     the init's camera-frame keypoints, so the gauge cancels exactly (~4 um).
  2. do-no-harm INIT gate: SAM candidate chosen by IoU vs the INIT-pose render (render
     consistency, not SAM's own score); frames below --min-iou keep the init pose untouched.
  3. REPROJECTION ANCHOR (--repro-w). Silhouette-only is forbidden: the manifold is flat along
     depth, so an unanchored optimum drifts.
  4. MINIMAL DOF: translation only by default; --refine-rot / --refine-angles are opt-in.
  5. ADOPTION MARGIN: refined pose kept only if its hard IoU beats the init render's by
     --adopt-margin and the 2D moved less than --max-uv-shift px. Else the init is restored.
  6. NEAR-FIELD / SELECTIVE: --rc-min-z skips close frames; --rc-only-failures spends RC only on
     frames the solver already flags as failed via reprojection (GT-free: ~2.3 px on success vs
     ~138 px on failure), which matters now that the fallback is strong.

Additive only: does not touch the Panda (rc_refine_from_dump.py) or Baxter (baxter_rc_eval.py)
paths, and does not modify kuka_add_eval.py.

--------------------------------------------------------------------------------------------
RESULT ON THE TRUE-K BASELINE (2026-07-22, kuka_synth_test_dr, 500 strided held-out frames).
GT-pose render vs SAM IoU 0.858 => wiring correct. Baseline = true-K solver: ADD-AUC 0.6834,
mean 90.4 mm, median 13.5 mm.

  RC config (t+R+angles, 120 it)                    ADD-AUC     delta
  ungated, anchor w=1                                0.6920     +0.0085
  ungated, anchor w=5                                0.6941     +0.0106
  w=5 + full guard set (min-iou/margin/uv<=12)       0.6970     +0.0136
  w=1 + uv-shift<=12 only                            0.6982     +0.0148
  w=1 + selective on solver failures (reproj>80px)   0.7013     +0.0179   <- best measured
  ORACLE (per-frame best of base vs RC)              0.7451     +0.0617

CONCLUSION: RC is largely ABSORBED by the intrinsics fix. On the broken direct-pose init it
looked like a +0.29 lever; on the correct baseline it is worth +0.010 to +0.018. It is NOT the
Panda-scale lever here (+0.043) and it does NOT reach the 0.75 target on its own.

WHY — RC does not fix ROTATION, which is the remaining lever:
  R-err  8.28 -> 8.13 deg  (-0.16 deg; improved on 339/500 frames)
  t-err  65.6 -> 66.2 mm   (no improvement)
  stratified: good frames (reproj<5px) R 4.20->4.07, failed frames (>50px) R 15.02->14.74.
The iiwa7 is a chain of near-cylindrical links, so its silhouette is nearly invariant to
rotation about each link axis — the silhouette simply does not observe R. Expect the same for
any RC variant here; the R lever must come from the pose/rot head or the solver init, not RC.

Two further findings that change how the guards should be read:
  * The IoU adoption margin is a POOR selector on this baseline (margin 0.02 scores -0.0005):
    rendered-IoU improvement does not predict ADD improvement. Keep it as a safety floor, not
    as a gain mechanism.
  * The anchor weight stopped mattering (w=1 vs w=5 differ by 0.002, and by 0.0001 once gated).
    On the old broken init it was the dominant knob. Do not carry that tuning intuition over.
  * The 0.745 oracle says RC carries ~+0.062 of extractable signal (RC helps 267/500 frames,
    hurts 166); the gain is DIFFUSE, not tail-concentrated (top 50 frames = 60% of upside), and
    reproj>80px wastes 44% of its RC budget on frames dead-either-way (both base & RC >100mm).

SELECTOR STUDY (2026-07-22, offline over the per-frame dumps, honest 5-fold nested CV, 10 seeds):
  Which GT-free signal predicts "RC helps this frame"?  End-to-end AUC when used as the adoption
  gate (baseline 0.6834, RC-everywhere 0.6941, oracle 0.7450):
    iou_fin  (absolute final render-vs-SAM IoU)  0.7156  (52% oracle recovery)  <- DEPLOYED
    + sam_ratio (2-feature logreg)               0.7170  (54%; +0.0014, within noise)
    iou_init                                     0.7077
    reproj>thr                                   0.6953  (29%; the previous selector)
    uv_shift, conf, size, iou_slope, z           <= 0.699
    multivariate logreg (6-14 feat)              0.708-0.723, UNSTABLE across feature sets
  Validity of each as an ADOPTION criterion (corr with per-frame ADD gain):
    iou_fin ABSOLUTE   +0.244     <- valid
    Δreproj improvement +0.220    <- WEAK (azure was 0.463; reprojection-improvement is NOT a
                                     reliable adoption signal, as warned)
    iou MARGIN (fin-init) +0.041  <- disqualified
  DEPLOY: run RC everywhere (t+R+angles, w=5), adopt per-frame iff final render IoU >= 0.83
  (plateau 0.80-0.85). --adopt-iou-abs 0.83 (default) implements exactly this.
  Multivariate models do NOT robustly beat the single iou_fin threshold at n=500 — the binding
  constraint is selector signal, and the silhouette's rotation-blindness caps how much any
  selector can recover (R is unfixable by RC).

OUT-OF-SAMPLE CONFIRMATION (--frame-offset 5, 300 frames DISJOINT from the offset-0 tuning set,
threshold T=0.83 fixed from tuning): solver 0.6784 -> deployed selective-RC 0.7031, +0.0247,
185/300 adopted, worst regression 28.6 mm (no divergence). The in-sample gain was +0.034; the
honest cross-subset gain is ~+0.025 to +0.034 (this subset has a heavier tail, baseline mean
105 vs 90 mm). Full-set (5997) estimate: solver ~0.686 + deployed RC ~ 0.71-0.72.

DEPLOY COMMAND (once approved), sequential on GPU0 or GPU3:
  CUDA_VISIBLE_DEVICES=GPU-<uuid> python iiwa7_rc_eval.py --mode rc \\
    --detector <best_heatmap.pth> --angle-head <best_angle_head.pth> \\
    --rot-head <best_rot_head.pth> --val-dir datasets/synthetic/kuka_synth_test_dr \\
    --max-frames 5997 --init solver --refine-rot --refine-angles --repro-w 5 \\
    --adopt-iou-abs 0.83 --min-iou 0.0 --max-uv-shift 0 --rc-min-z 0
  (~40 min/300 frames when the GPU is shared; budget accordingly or run in shards by --frame-offset.)
"""
import argparse, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(HERE, '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(HERE)

from model_angle import AnglePredictor
from dataset import PoseEstimationDataset
from refine_eval import scale_K, add_auc, geometric_K
from solve_pose_kinematic import rot6d_to_matrix, matrix_to_rot6d
import solve_pose_kinematic as spk
from kuka_add_eval import kabsch_batch, _patch_solver_for_iiwa7
from model_v4 import iiwa7_forward_kinematics as fitted_fk   # baseline gauge (keypoint path only)
from iiwa7_render import (make_iiwa7_renderer, iiwa7_all_link_transforms,
                          iiwa7_urdf_forward_kinematics)

KP_NAMES = [f'iiwa7_link_{i}' for i in range(1, 8)]
ANGLE_JOINTS = [f'iiwa7_joint_{i}' for i in range(1, 8)]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def project(kp3d_cam, K):
    z = kp3d_cam[..., 2].clamp(min=1e-3)
    u = kp3d_cam[..., 0] / z * K[:, 0, 0:1] + K[:, 0, 2:3]
    v = kp3d_cam[..., 1] / z * K[:, 1, 1:2] + K[:, 1, 2:3]
    return torch.stack([u, v], -1)


def soft_iou(a, b):
    inter = (a * b).sum((-1, -2))
    union = (a + b - a * b).sum((-1, -2)).clamp(min=1e-6)
    return inter / union


def hard_iou(a, b):
    inter = (a * b).sum((-1, -2))
    union = ((a + b) > 0).float().sum((-1, -2)).clamp(min=1)
    return inter / union


def sam_masks(sam_pred, u8, uv, valid, ref_mask, S, H):
    """SAM candidates prompted by `uv`, resized to HxH, ranked by IoU against `ref_mask`
    (render-consistency selection, guard 2). Returns (best_mask, best_iou) per batch item."""
    import torch.nn.functional as F
    out_m = torch.zeros_like(ref_mask)
    out_i = torch.zeros(len(u8), device=ref_mask.device)
    for b in range(len(u8)):
        p = uv[b][valid[b] > 0].detach().cpu().numpy()
        if len(p) < 2:
            continue
        x0, y0 = p[:, 0].min(), p[:, 1].min(); x1, y1 = p[:, 0].max(), p[:, 1].max()
        mx = 0.15 * max(x1 - x0, y1 - y0)
        box = np.array([max(0, x0 - mx), max(0, y0 - mx), min(S, x1 + mx), min(S, y1 + mx)])
        sam_pred.set_image(u8[b])
        mm, _, _ = sam_pred.predict(point_coords=p.astype(np.float32),
                                    point_labels=np.ones(len(p)), box=box, multimask_output=True)
        cands = torch.from_numpy(mm.astype('float32')).to(ref_mask.device)
        cands = (F.interpolate(cands.unsqueeze(1), size=(H, H), mode='bilinear',
                               align_corners=False).squeeze(1) > 0.5).float()
        iou3 = hard_iou(cands, ref_mask[b].unsqueeze(0))
        j = int(iou3.argmax())
        out_m[b] = cands[j]; out_i[b] = iou3[j]
    return out_m, out_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['gtcheck', 'rc'], default='rc')
    ap.add_argument('--detector', required=True)
    ap.add_argument('--angle-head', required=True)
    ap.add_argument('--rot-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--sam-checkpoint', default=os.path.join(HERE, '../weights_sam/sam_vit_b_01ec64.pth'))
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--render-h', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-frames', type=int, default=50)
    ap.add_argument('--crop-margin', type=float, default=1.5)
    ap.add_argument('--mesh-kind', default='visual')
    ap.add_argument('--init', choices=['solver', 'direct'], default='solver',
                    help="RC starting point. 'solver' = kinematic solver on the TRUE metric K "
                         "(current baseline). 'direct' = old rot-head-only pose, for A/B only.")
    ap.add_argument('--solver-iters', type=int, default=250)
    ap.add_argument('--conf-gate', type=float, default=0.05)
    ap.add_argument('--cov-pnp', action='store_true')
    ap.add_argument('--rc-only-failures', type=float, default=0.0,
                    help='if >0: run RC ONLY on frames whose solver reprojection exceeds this many '
                         'px. Solver success/failure separates ~60x in reprojection (2.3 px vs '
                         '138 px) with NO ground truth, so RC can be spent only where the (now '
                         'strong) solver fallback actually failed. 0 = RC every frame.')
    # --- optimisation ---
    ap.add_argument('--rc-iters', type=int, default=120)
    ap.add_argument('--rc-lr', type=float, default=3e-3, help='lr for translation')
    ap.add_argument('--rot-lr', type=float, default=5e-4)
    ap.add_argument('--ang-lr', type=float, default=2e-3)
    ap.add_argument('--repro-w', type=float, default=100.0, help='guard 3: reprojection anchor weight')
    ap.add_argument('--ang-prior-w', type=float, default=0.5)
    ap.add_argument('--refine-rot', action='store_true', help='guard 4: OFF by default')
    ap.add_argument('--refine-angles', action='store_true', help='guard 4: OFF by default')
    # --- guards ---
    ap.add_argument('--min-iou', type=float, default=0.35, help='guard 2: init-render vs SAM gate')
    ap.add_argument('--adopt-margin', type=float, default=0.005,
                    help='LEGACY guard 5: IoU GAIN (fin-init) required to keep RC. Superseded by '
                         '--adopt-iou-abs (the IoU margin does not predict ADD improvement, '
                         'corr +0.04). Used only when --adopt-iou-abs <= 0.')
    ap.add_argument('--adopt-iou-abs', type=float, default=0.83,
                    help='DEPLOYABLE adoption selector: keep RC only when the ABSOLUTE final '
                         'render-vs-SAM IoU >= this. Validated best GT-free selector (see code). '
                         '0 = fall back to the legacy --adopt-margin gate.')
    ap.add_argument('--max-uv-shift', type=float, default=12.0,
                    help='guard 5: px; larger 2D drift -> revert. <=0 disables. Redundant with '
                         '--adopt-margin (which gates on the objective directly); at repro-w 10 it '
                         'fires on frames that were in fact improving, so widen it if you weaken the anchor.')
    ap.add_argument('--rc-min-z', type=float, default=0.0,
                    help='guard 6: skip RC when init depth < this (m). 0 = off. Measured on '
                         'kuka_synth_test_dr: a 0.6 m cut-off skipped 8/50 frames and LOWERED the '
                         'net gain, so it is off by default here and kept only as a knob.')
    ap.add_argument('--frame-offset', type=int, default=0, help='stride phase; use !=0 for a subset disjoint from the tuning set')
    ap.add_argument('--viz', default=None)
    ap.add_argument('--dump-npz', default=None,
                    help='write per-frame RAW (ungated) RC outcome + every gate signal, so any '
                         'gate/threshold combination can be evaluated offline from ONE run '
                         'instead of one 20-min GPU run per setting.')
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available()
    IS, H = args.image_size, args.render_h
    # MUST precede any solve_batch call: swaps the solver's FK/limits/mean from Panda to iiwa7.
    _patch_solver_for_iiwa7()

    m = AnglePredictor(args.model_name, IS, fix_joint7_zero=True, head_type='mlp',
                       with_rotation=True, with_translation=True).to(device).eval()
    sd = torch.load(args.detector, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.angle_head, map_location=device))
    m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))

    rdr = make_iiwa7_renderer(device, args.mesh_kind)
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry['vit_b'](checkpoint=args.sam_checkpoint).to(device).eval()
    sam_pred = SamPredictor(sam)

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=KP_NAMES, image_size=(IS, IS),
                               heatmap_size=(IS, IS), augment=False, include_angles=True, sigma=2.5,
                               crop_to_robot=True, crop_margin=args.crop_margin,
                               angle_joint_names=ANGLE_JOINTS)
    if args.max_frames and args.max_frames < len(ds):
        st = max(1, len(ds.samples) // args.max_frames)
        # --frame-offset shifts the stride phase to get a subset DISJOINT from the offset-0 set
        # used for selector tuning (honest out-of-sample confirmation of the adoption threshold).
        ds.samples = ds.samples[args.frame_offset % st::st][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    MEAN = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    STD = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)

    gt_ious, init_ious, final_ious = [], [], []
    adds_base, adds_rc = [], []
    n_skip_iou = n_skip_z = n_skip_ok = n_revert_margin = n_revert_uv = n_adopt = 0
    reprojs = []
    D = {k: [] for k in ('reproj', 'iou_init', 'iou_fin', 'add_base', 'add_rc_raw', 'uv_shift', 'z',
                       'rerr_base', 'rerr_rc', 'terr_base', 'terr_rc',
                       'reproj_rc', 'conf_mean', 'conf_min', 'size_px', 'sam_ratio',
                       'iou_t0', 'iou_t30', 'iou_t60', 'n_conf')}
    viz_done = 0
    if args.viz:
        os.makedirs(args.viz, exist_ok=True)

    for batch in tqdm(loader, desc=f'iiwa7-{args.mode}'):
        img = batch['image'].to(device)
        B = img.shape[0]
        gt3d = batch['keypoints_3d'].to(device)
        gt_ang = batch['angles'].to(device).float()
        # K_model: the eye(3)-derived K every checkpoint was trained on (feed to the network).
        # K:       true metric intrinsics (render + project). See geometric_K.
        K_model = scale_K(batch['camera_K'], batch['original_size'], IS).to(device)
        K = geometric_K(args.val_dir, batch['camera_K'], batch['original_size'], IS).to(device)
        valid = (gt3d.abs().sum(-1) > 0).float()
        u8 = ((img * STD + MEAN).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

        if args.mode == 'gtcheck':
            # GT angles + GT keypoints -> exact camera pose via Kabsch on the URDF chain.
            fk_g = iiwa7_urdf_forward_kinematics(gt_ang.double()).float()
            Rg, tg = kabsch_batch(fk_g, gt3d)
            with torch.no_grad():
                mask_g = (rdr(rdr.robot_verts(gt_ang, iiwa7_all_link_transforms),
                              Rg, tg, K, H, IS) > 0.5).float()
            uv_g = project(torch.einsum('bij,bnj->bni', Rg, fk_g) + tg.unsqueeze(1), K)
            sm, si = sam_masks(sam_pred, u8, uv_g, valid, mask_g, IS, H)
            gt_ious += si.cpu().tolist()
            if args.viz and viz_done < 12:
                import cv2, torch.nn.functional as F
                up = F.interpolate(mask_g.unsqueeze(1), size=(IS, IS), mode='nearest').squeeze(1)
                ups = F.interpolate(sm.unsqueeze(1), size=(IS, IS), mode='nearest').squeeze(1)
                for b in range(B):
                    if viz_done >= 12:
                        break
                    o = u8[b].copy()
                    mm = (up[b] > 0.5).cpu().numpy().astype(np.uint8)
                    ss = (ups[b] > 0.5).cpu().numpy().astype(np.uint8)
                    c1, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    c2, _ = cv2.findContours(ss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(o, c2, -1, (255, 0, 0), 1)     # SAM = blue
                    cv2.drawContours(o, c1, -1, (0, 255, 0), 2)     # GT render = green
                    cv2.imwrite(os.path.join(args.viz, f'gt_{viz_done:02d}_iou{si[b]:.2f}.png'), o[..., ::-1])
                    viz_done += 1
            continue

        # ---------------- RC ----------------
        with torch.no_grad():
            o = m(img, K_model)
        head_ang = o['joint_angles'].float()                # (B,7), joint_7 == 0

        # ---- RC starting point -------------------------------------------------------------
        # `solver` (default) = the kinematic solver on the TRUE metric K. Feeding the eye(3)
        # dataset K here put a 320x-wrong focal into PnP/refine and collapsed solved depth; THAT
        # bug — not "link confusion" — is what made direct-pose look like the better baseline.
        # `direct` reproduces the old rot-head-only baseline, for A/B only.
        reproj = torch.zeros(B, device=device)
        if args.init == 'solver':
            kp2d = o['keypoints_2d']; conf = o['confidence']
            cov_inv = spk.heatmap_cov_inv(o['heatmaps_2d'], kp2d) if args.cov_pnp else None
            with torch.enable_grad():
                th0, kpc0, rp = spk.solve_batch(
                    kp2d, conf, K, fix_joint7=True, iters=args.solver_iters, lr=2e-2,
                    img_size=IS, device=device, prior_w=0.0, theta_init=head_ang,
                    cov_inv=cov_inv, conf_gate=args.conf_gate,
                    R_init=o['rot_matrix'], t_init=o['trans'])
            th0 = th0.detach().float(); kpc0 = kpc0.detach().float()
            reproj = torch.as_tensor(rp, device=device).detach().float().view(-1)
        else:
            th0 = head_ang
            kpc0 = (torch.einsum('bij,bnj->bni', o['rot_matrix'].float(),
                                 fitted_fk(th0.double()).float())
                    + o['trans'].float().unsqueeze(1))
        # Both branches give keypoints on the model_v4 (fitted) chain; Kabsch onto the URDF chain
        # re-expresses them in the mesh gauge, exact to ~4 um (see iiwa7_render).
        fk0 = iiwa7_urdf_forward_kinematics(th0.double()).float()
        R0, t0 = kabsch_batch(fk0, kpc0)
        uv_anchor = project(kpc0, K)

        with torch.no_grad():
            init_mask = (rdr(rdr.robot_verts(th0, iiwa7_all_link_transforms),
                             R0, t0, K, H, IS) > 0.5).float()
        tgt, si = sam_masks(sam_pred, u8, uv_anchor, valid, init_mask, IS, H)
        init_ious += si.cpu().tolist()
        reprojs += reproj.cpu().tolist()

        use = (si >= args.min_iou).float()                                    # guard 2
        n_skip_iou += int((use == 0).sum())
        near = (t0[:, 2] < args.rc_min_z)                                     # guard 6
        n_skip_z += int((near & (use > 0)).sum())
        use = use * (~near).float()
        if args.rc_only_failures > 0:                                         # selective RC
            ok = reproj <= args.rc_only_failures
            n_skip_ok += int((ok & (use > 0)).sum())
            use = use * (~ok).float()

        # guard 4: minimal DOF — translation only unless explicitly widened
        tt = t0.clone().detach().requires_grad_(True)
        groups = [{'params': [tt], 'lr': args.rc_lr}]
        d6 = matrix_to_rot6d(R0).clone().detach()
        if args.refine_rot:
            d6 = d6.requires_grad_(True); groups.append({'params': [d6], 'lr': args.rot_lr})
        da = torch.zeros_like(th0[:, :6])
        if args.refine_angles:
            da = da.requires_grad_(True); groups.append({'params': [da], 'lr': args.ang_lr})
        opt = torch.optim.Adam(groups)
        zc = torch.zeros(B, 1, device=device)
        usem = use.view(-1, 1, 1)

        traj = {}
        for _it in range(args.rc_iters):
            opt.zero_grad()
            th = torch.cat([th0[:, :6] + da, zc], 1)
            R = rot6d_to_matrix(d6) if args.refine_rot else R0
            verts = rdr.robot_verts(th, iiwa7_all_link_transforms)
            mask = rdr(verts, R, tt, K, H, IS)
            siou = soft_iou(mask, tgt)                       # (B,) per-frame
            if _it in (0, 30, 60):
                traj[_it] = siou.detach().clone()
            l_iou = (use * (1 - siou)).sum() / use.sum().clamp(min=1)
            fk = iiwa7_urdf_forward_kinematics(th)
            cam = torch.einsum('bij,bpj->bpi', R, fk) + tt.unsqueeze(1)
            uv = project(cam, K)
            # guard 3: reprojection anchor — pins u,v so t may only slide along the depth ray
            l_uv = (((uv - uv_anchor) / IS) * valid.unsqueeze(-1)).pow(2).mean()
            loss = l_iou + args.repro_w * l_uv
            if args.refine_angles:
                loss = loss + args.ang_prior_w * (da ** 2).mean()
            loss.backward(); opt.step()

        with torch.no_grad():
            th = torch.cat([th0[:, :6] + da, zc], 1)
            R = rot6d_to_matrix(d6) if args.refine_rot else R0
            fk = iiwa7_urdf_forward_kinematics(th)
            kp_rc = torch.einsum('bij,bpj->bpi', R, fk) + tt.unsqueeze(1)
            uv_rc = project(kp_rc, K)
            fin_mask = (rdr(rdr.robot_verts(th, iiwa7_all_link_transforms), R, tt, K, H, IS) > 0.5).float()
            fin_iou = hard_iou(fin_mask, tgt)
        final_ious += fin_iou.cpu().tolist()
        # extra GT-FREE selector features (all computable at deploy time)
        with torch.no_grad():
            kp2d_det = o['keypoints_2d'].float(); cf = o['confidence'].float()
            w_ = cf.clamp(min=1e-3)
            rp_rc = (((uv_rc - kp2d_det).norm(dim=-1)) * w_).sum(1) / w_.sum(1)
            span = uv_anchor.amax(1) - uv_anchor.amin(1)         # (B,2) apparent robot size
            size_px = span.amax(-1)
            sam_ratio = tgt.sum((-1, -2)) / init_mask.sum((-1, -2)).clamp(min=1)
            n_conf = (cf > 0.3).float().sum(1)

        # GT camera pose on the SAME (URDF) chain -> R/t error before vs after RC. The open
        # question is whether the silhouette fixes ROTATION (the solver's remaining lever,
        # ~6.1 deg) or only depth; ADD alone cannot separate those.
        with torch.no_grad():
            Rg, tg = kabsch_batch(iiwa7_urdf_forward_kinematics(gt_ang.double()).float(), gt3d)
            def _rerr(Ra):
                c = ((Ra.transpose(1, 2) @ Rg).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2
                return torch.rad2deg(torch.acos(c.clamp(-1, 1)))
            rb, rr = _rerr(R0), _rerr(R)
            tb = (t0 - tg).norm(dim=-1); tr = (tt.detach() - tg).norm(dim=-1)

        for b in range(B):
            vb = valid[b].bool()
            if vb.sum() < 5:
                continue
            base = float((kpc0[b][vb] - gt3d[b][vb]).norm(dim=-1).mean())
            adds_base.append(base)
            shift = float((uv_rc[b][vb] - uv_anchor[b][vb]).norm(dim=-1).mean())
            D['reproj'].append(float(reproj[b])); D['iou_init'].append(float(si[b]))
            D['iou_fin'].append(float(fin_iou[b])); D['add_base'].append(base)
            D['add_rc_raw'].append(float((kp_rc[b][vb] - gt3d[b][vb]).norm(dim=-1).mean()))
            D['uv_shift'].append(shift); D['z'].append(float(t0[b, 2]))
            D['rerr_base'].append(float(rb[b])); D['rerr_rc'].append(float(rr[b]))
            D['terr_base'].append(float(tb[b])); D['terr_rc'].append(float(tr[b]))
            D['reproj_rc'].append(float(rp_rc[b])); D['conf_mean'].append(float(cf[b].mean()))
            D['conf_min'].append(float(cf[b].min())); D['size_px'].append(float(size_px[b]))
            D['sam_ratio'].append(float(sam_ratio[b])); D['n_conf'].append(float(n_conf[b]))
            for _t in (0, 30, 60):
                D[f'iou_t{_t}'].append(float(traj[_t][b]) if _t in traj else float('nan'))
            keep = use[b] > 0
            if keep and args.adopt_iou_abs > 0:
                # VALIDATED adoption selector (2026-07-22): keep RC only when the ABSOLUTE final
                # render-vs-SAM IoU clears a fixed bar. On 500 held-out frames this recovers 52%
                # of the oracle (0.6834 -> 0.7176), vs 29% for reproj>80px and ~0% for the IoU
                # MARGIN. corr(iou_fin, ΔADD)=+0.244 while corr(Δreproj, ΔADD)=+0.220 and
                # corr(iou_margin, ΔADD)=+0.04 — only the absolute final IoU is a valid criterion.
                if fin_iou[b] < args.adopt_iou_abs:
                    keep = False; n_revert_margin += 1
            elif keep and fin_iou[b] < si[b] + args.adopt_margin:              # legacy margin gate
                keep = False; n_revert_margin += 1
            if keep and args.max_uv_shift > 0 and shift > args.max_uv_shift:   # guard 5
                keep = False; n_revert_uv += 1
            if keep:
                adds_rc.append(float((kp_rc[b][vb] - gt3d[b][vb]).norm(dim=-1).mean())); n_adopt += 1
            else:
                adds_rc.append(base)

    W = 64
    if args.mode == 'gtcheck':
        g = np.array(gt_ious)
        print(f"\n{'='*W}\n  iiwa7 GT-POSE RENDER CHECK  ({len(g)} frames)  mesh={args.mesh_kind}\n{'='*W}")
        print(f"  GT-pose render vs SAM IoU: mean {g.mean():.4f}  median {np.median(g):.4f}  "
              f"min {g.min():.3f}  max {g.max():.3f}")
        print(f"  frac >= 0.5: {(g >= 0.5).mean():.2f}   frac >= 0.7: {(g >= 0.7).mean():.2f}")
        print(f"  [gate] high IoU => mesh + FK frame convention are correct.")
        print('=' * W)
        return

    a0, a1 = np.array(adds_base), np.array(adds_rc)
    ii, fi = np.array(init_ious), np.array(final_ious)
    print(f"\n{'='*W}\n  KUKA iiwa7 RENDER-AND-COMPARE  ({len(a0)} frames)  init={args.init}\n{'='*W}")
    print(f"  DOF: t{'+R' if args.refine_rot else ''}{'+angles' if args.refine_angles else ''}"
          f"   iters {args.rc_iters}  repro-w {args.repro_w}")
    print(f"  SAM-vs-init-render IoU: mean {ii.mean():.3f}  median {np.median(ii):.3f}  "
          f"frac>=0.5 {(ii >= 0.5).mean():.2f}")
    print(f"  final render IoU:       mean {fi.mean():.3f}")
    if reprojs:
        rp = np.array(reprojs)
        print(f"  solver reprojection: median {np.median(rp):.2f}px  p90 {np.percentile(rp,90):.2f}px  "
              f"frac>10px {(rp>10).mean():.2f}  (GT-free failure detector)")
    print(f"  gates: skip(min-iou) {n_skip_iou}  skip(near-z) {n_skip_z}  skip(reproj-ok) {n_skip_ok}  "
          f"revert(margin) {n_revert_margin}  revert(uv) {n_revert_uv}  ADOPTED {n_adopt}")
    print('-' * W)
    lbl = 'true-K solver' if args.init == 'solver' else 'direct-pose (OLD/buggy-K)'
    print(f"  BEFORE ({lbl}): ADD-AUC {add_auc(a0):.4f} | mean {a0.mean()*1000:.1f}mm | "
          f"median {np.median(a0)*1000:.1f}mm")
    print(f"  AFTER  (RC)         : ADD-AUC {add_auc(a1):.4f} | mean {a1.mean()*1000:.1f}mm | "
          f"median {np.median(a1)*1000:.1f}mm")
    print(f"  delta AUC {add_auc(a1)-add_auc(a0):+.4f} | mean {(a1.mean()-a0.mean())*1000:+.1f}mm")
    if n_adopt:
        adopted = a1 != a0
        print(f"  on ADOPTED frames only ({adopted.sum()}): "
              f"{a0[adopted].mean()*1000:.1f}mm -> {a1[adopted].mean()*1000:.1f}mm")
        d = (a1 - a0)[adopted] * 1000
        # Baxter's failure fingerprint was UNIFORM worsening (500/500 frames ~3x worse). Report the
        # better/worse split explicitly so that signature cannot hide behind an aggregate mean.
        print(f"  adopted split: improved {int((d < 0).sum())} (median {np.median(d[d < 0]) if (d<0).any() else 0:+.1f}mm) | "
              f"worsened {int((d > 0).sum())} (median {np.median(d[d > 0]) if (d>0).any() else 0:+.1f}mm)")
        if (d > 0).any():
            print(f"  worst regressions (mm): {np.round(np.sort(d)[-5:], 1).tolist()}")
    if args.dump_npz:
        np.savez(args.dump_npz, **{k: np.array(v) for k, v in D.items()})
        print(f"  per-frame dump -> {args.dump_npz}  ({len(D['reproj'])} rows)")
    print('=' * W)


if __name__ == '__main__':
    main()
