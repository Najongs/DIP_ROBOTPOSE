"""
Phase-4 deployable render-and-compare: refine DUMPED deployed-pipeline poses (selfbbox_eval
--dump-npz) against SAM true-robot masks using the EXACT-mesh nvdiffrast silhouette.

Per frame: init (theta, R, t) from the dump (R,t via Kabsch of FK(theta) onto the solved camera-frame
keypoints); SAM mask prompted by the dump pose's projected keypoints, with the mask candidate SELECTED
by IoU against the INIT-pose render (render-consistency selection, not SAM's own score); conservative
refine (Adam lr 5e-4, soft-IoU + reprojection anchor to the dump pose's 2D); DO-NO-HARM gate: frames
whose best SAM-vs-init-render IoU < --min-iou keep the baseline pose untouched.

Reports ADD-AUC@100mm before/after on identical frames. Target (plan gate): >= +0.05 on realsense
over the deployed 0.7525 held-out baseline.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
from model_v4 import panda_forward_kinematics
from solve_pose_kinematic import rot6d_to_matrix, matrix_to_rot6d
from silhouette_mesh_probe import kabsch_batch, all_link_transforms, soft_iou, add_auc, KPN
from inference_4tier_eval import EvalDataset
from refine_eval import scale_K
from render_nvdr import NVDRSilhouette

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def project(kp3d_cam, K):
    z = kp3d_cam[..., 2].clamp(min=1e-3)
    u = kp3d_cam[..., 0] / z * K[:, 0, 0:1] + K[:, 0, 2:3]
    v = kp3d_cam[..., 1] / z * K[:, 1, 1:2] + K[:, 1, 2:3]
    return torch.stack([u, v], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True, help='npz from selfbbox_eval --dump-npz (deployed poses)')
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--sam-checkpoint', default=None)
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--render-h', type=int, default=224)
    ap.add_argument('--kind', default='visual')
    ap.add_argument('--rc-iters', type=int, default=150)
    ap.add_argument('--rc-lr', type=float, default=5e-4)
    ap.add_argument('--repro-w', type=float, default=100.0)
    ap.add_argument('--min-iou', type=float, default=0.35, help='skip refine when SAM-vs-init-render IoU below this (do-no-harm)')
    ap.add_argument('--max-uv-shift', type=float, default=0.0,
                    help='if >0: REVERT frames whose refined pose moved the 2D reprojection more than this many px from the dump anchor. The intended depth correction is 2D-invariant (monocular ambiguity direction); a large 2D shift means the mask dragged the pose sideways.')
    ap.add_argument('--max-frames', type=int, default=0, help='0 = all dumped frames')
    ap.add_argument('--occlude-ratio', type=float, default=0.0,
                    help='paint the SAME deterministic occluders as selfbbox_eval --occlude-ratio (seeded per frame+ratio) so SAM sees the occluded image the pose stage saw')
    ap.add_argument('--struct-w', type=float, default=0.0,
                    help='weight of the internal-STRUCTURE term: edge-NCC between rendered-depth gradients and image gradients (lighting/albedo-free; probe-validated GT-discriminative even on azure where the silhouette term hurts)')
    ap.add_argument('--feat-w', type=float, default=0.0,
                    help='weight of the DINO FEATURE-METRIC term: (1 - masked patch-cosine) between the Lambertian-shaded render and the real crop in frozen-DINOv3 feature space (absorbs albedo/lighting gap; probe-validated to beat edge-NCC on azure). Backprops through DINOv3 forward.')
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--feat-size', type=int, default=384, help='DINOv3 input resolution for the feature term (smaller = faster inner loop)')
    ap.add_argument('--no-sil', action='store_true',
                    help='drop the SAM/silhouette IoU term entirely (pure model-based refinement: structure + reproj anchor; no mask needed)')
    ap.add_argument('--multi-start', action='store_true',
                    help='multi-start RC over base-Z rotation perturbations of the init pose; final hypothesis chosen by SAM-IoU (external-evidence basin selection)')
    ap.add_argument('--ms-deltas', default='30,60', help='perturbation magnitudes in degrees (each used as +/-)')
    ap.add_argument('--ms-margin', type=float, default=0.01, help='challenger must beat hypothesis-0 IoU by this margin')
    ap.add_argument('--ms-wrist-n', type=int, default=0, help='number of wrist-angle (J4/J5/J6) perturbation hypotheses for silhouette-selected basin escape (0=off)')
    ap.add_argument('--ms-wrist-sigma', type=float, default=40.0, help='std (deg) of Gaussian wrist-angle perturbations')
    ap.add_argument('--occl-robust-w', type=float, default=-1.0,
                    help='if >=0: occlusion-robust pixel-weighted IoU — downweight to this value the pixels where the INIT render says robot but SAM says background (candidate external occluder covering the robot), so the occluded part is inferred from FK + the visible remainder instead of being penalized. -1 = off (plain soft-IoU).')
    ap.add_argument('--viz', default=None)
    args = ap.parse_args()
    device = torch.device('cuda'); S = args.image_size; H = args.render_h

    d = np.load(args.dump, allow_pickle=True)
    fids = [str(x) for x in d['fid']]
    theta_d = torch.from_numpy(d['theta']).float()
    kpcam_d = torch.from_numpy(d['kp_cam']).float()
    gt3d_d = torch.from_numpy(d['gt3d']).float()
    found_d = torch.from_numpy(d['found']).float()
    if args.max_frames and args.max_frames < len(fids):
        fids = fids[:args.max_frames]

    ds = EvalDataset(args.val_dir, KPN, image_size=(S, S))
    by_fid = {os.path.basename(str(p)).replace('.json', ''): i for i, p in enumerate(ds.json_files)}
    order = [by_fid[f] for f in fids if f in by_fid]
    assert len(order) == len(fids), f"dump/frames mismatch: {len(order)} vs {len(fids)}"

    sam_pred = None
    if not args.no_sil:
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry['vit_b'](checkpoint=args.sam_checkpoint).to(device); sam.eval()
        sam_pred = SamPredictor(sam)
    rdr = NVDRSilhouette(device, kind=args.kind)
    if args.struct_w > 0:
        from rgb_rc_probe import edge_score
    backbone = None
    if args.feat_w > 0:
        from model_v4 import DINOv3Backbone
        from feat_rc_probe import dino_feats, feat_score
        backbone = DINOv3Backbone(args.model_name, unfreeze_blocks=0).to(device).eval()
    MEAN = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    STD = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)

    adds_base, adds_rc, skipped, sam_ious = [], [], 0, []
    ms_switched = 0
    B = args.batch_size
    for lo in tqdm(range(0, len(fids), B), desc='rc-refine'):
        idxs = list(range(lo, min(lo + B, len(fids))))
        items = [ds[order[i]] for i in idxs]
        img = torch.stack([torch.as_tensor(it['image']) for it in items]).to(device)
        if args.occlude_ratio > 0:
            from occl_util import paste_occluders_
            for b, it in enumerate(items):
                sc = S / np.asarray(it['original_size'], dtype=np.float32)      # (2,) w,h scale
                paste_occluders_(img[b], np.asarray(it['gt_2d']) * sc[None, :],
                                 np.asarray(it['found']), args.occlude_ratio, fids[idxs[b]])
        K = scale_K(torch.stack([torch.as_tensor(it['camera_K']).float() for it in items]),
                    torch.stack([torch.as_tensor(it['original_size']) for it in items]), S).to(device)
        th0 = theta_d[idxs].to(device); kpc0 = kpcam_d[idxs].to(device)
        gt3d = gt3d_d[idxs].to(device); found = found_d[idxs].to(device)

        fk0 = panda_forward_kinematics(th0)
        R0, t0 = kabsch_batch(fk0, kpc0)
        uv_anchor = project(kpc0, K)                       # dump pose's 2D (solver-consistent anchor)
        wconf = found.clone()

        with torch.no_grad():
            init_mask = (rdr(rdr.robot_verts(th0, all_link_transforms), R0, t0, K, H, S) > 0.5).float()
        u8 = ((img * STD + MEAN).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        gray = ((img * STD + MEAN).clamp(0, 1)).mean(1, keepdim=True) if args.struct_w > 0 else None
        tgt = torch.zeros_like(init_mask); use = torch.zeros(len(idxs), device=device)
        if sam_pred is None:
            use += 1.0                                     # struct/anchor-only: refine every frame
        for b in range(len(idxs) if sam_pred is not None else 0):
            sam_pred.set_image(u8[b])
            p = uv_anchor[b][found[b] > 0].detach().cpu().numpy()
            if len(p) < 2:
                continue
            x0, y0 = p[:, 0].min(), p[:, 1].min(); x1, y1 = p[:, 0].max(), p[:, 1].max()
            mx = 0.15 * max(x1 - x0, y1 - y0)
            box = np.array([max(0, x0 - mx), max(0, y0 - mx), min(S, x1 + mx), min(S, y1 + mx)])
            mm, _, _ = sam_pred.predict(point_coords=p, point_labels=np.ones(len(p)), box=box, multimask_output=True)
            cands = torch.from_numpy(mm.astype('float32')).to(device)          # (3,S,S)
            cands = (F.interpolate(cands.unsqueeze(1), size=(H, H), mode='bilinear', align_corners=False).squeeze(1) > 0.5).float()
            inter = (cands * init_mask[b]).sum((-1, -2))
            union = ((cands + init_mask[b]) > 0).float().sum((-1, -2)).clamp(min=1)
            iou3 = inter / union
            j = int(iou3.argmax())
            sam_ious.append(float(iou3[j]))
            if iou3[j] >= args.min_iou:                                        # do-no-harm gate
                tgt[b] = cands[j]; use[b] = 1.0

        # occlusion-robust pixel weights: "init-render robot BUT SAM background" = candidate
        # external occluder ON the robot -> don't penalize the render there; the hidden part is
        # then constrained by FK + the visible remainder ("infer roughly through the occluder").
        pix_w = None
        if args.occl_robust_w >= 0:
            occluder_cand = (init_mask > 0.5) & (tgt < 0.5)               # (B,H,H)
            pix_w = torch.where(occluder_cand, torch.full_like(init_mask, args.occl_robust_w),
                                torch.ones_like(init_mask))

        def w_soft_iou(a, b):
            if pix_w is None:
                return soft_iou(a, b)
            inter = (pix_w * a * b).sum((-1, -2))
            union = (pix_w * (a + b - a * b)).sum((-1, -2))
            return inter / (union + 1e-6)

        f_real = dino_feats(backbone, (img * STD + MEAN).clamp(0, 1), args.feat_size) if backbone is not None else None

        def refine_from(R_init, theta_init=None, t_init=None):
            """One conservative RC refine from a given camera-rotation (and optional theta/t) init.
            Returns per-frame (kp_rc, uv_rc, final hard IoU vs tgt)."""
            d6 = matrix_to_rot6d(R_init).clone().detach().requires_grad_(True)
            tt = (t0 if t_init is None else t_init).clone().detach().requires_grad_(True)
            th_src = th0 if theta_init is None else theta_init
            pth = th_src[:, :6].clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([d6, tt, pth], lr=args.rc_lr)
            zc = torch.zeros(pth.shape[0], 1, device=device)
            for _ in range(args.rc_iters):
                opt.zero_grad()
                th = torch.cat([pth, zc], 1); R = rot6d_to_matrix(d6)
                verts = rdr.robot_verts(th, all_link_transforms)
                if sam_pred is not None:
                    mask = rdr(verts, R, tt, K, H, S)
                    l_iou = (use * (1 - w_soft_iou(mask, tgt))).sum() / use.sum().clamp(min=1)
                else:
                    l_iou = torch.zeros((), device=device)
                if args.struct_w > 0:
                    dmap = rdr.render_depth(verts, R, tt, K, H, S)
                    l_struct = (1 - edge_score(dmap, gray, H)).mean()
                else:
                    l_struct = torch.zeros((), device=device)
                if backbone is not None:
                    sh = rdr.render_shaded(verts, R, tt, K, H, S)
                    fr = dino_feats(backbone, sh.unsqueeze(1).repeat(1, 3, 1, 1), args.feat_size)
                    gmask = (F.interpolate(sh.unsqueeze(1), size=fr.shape[-2:], mode='bilinear',
                                           align_corners=False).squeeze(1) > 0.05).float()
                    l_feat = (1 - feat_score(fr, f_real, gmask)).mean()
                else:
                    l_feat = torch.zeros((), device=device)
                fk = panda_forward_kinematics(th)
                cam = torch.einsum('bij,bpj->bpi', R, fk) + tt.unsqueeze(1)
                uv = project(cam, K)
                l_uv = (((uv - uv_anchor) / S) * wconf.unsqueeze(-1)).pow(2).mean()
                (l_iou + args.struct_w * l_struct + args.feat_w * l_feat + args.repro_w * l_uv).backward()
                opt.step()
            with torch.no_grad():
                th = torch.cat([pth, zc], 1); R = rot6d_to_matrix(d6)
                fk = panda_forward_kinematics(th)
                kp = torch.einsum('bij,bpj->bpi', R, fk) + tt.unsqueeze(1)
                uv = project(kp, K)
                m = (rdr(rdr.robot_verts(th, all_link_transforms), R, tt, K, H, S) > 0.5).float()
                inter = (m * tgt).sum((-1, -2)); union = ((m + tgt) > 0).float().sum((-1, -2)).clamp(min=1)
            return kp, uv, inter / union

        # multi-start over the base-Z gauge circle: 2D keypoints CANNOT tell these hypotheses
        # apart (that IS the foreshortening/base-yaw ambiguity), but the exact-mesh silhouette
        # can — selection by SAM-IoU is an EXTERNAL-evidence selector (unlike the refuted
        # learned-selector MCL). Hypothesis 0 = original init; challengers must beat it by a
        # margin (do-no-harm).
        deltas = [0.0]
        if args.multi_start:
            for dgn in [float(x) for x in args.ms_deltas.split(',') if x.strip()]:
                deltas += [np.deg2rad(dgn), -np.deg2rad(dgn)]
        kp_rc, uv_rc, best_iou = refine_from(R0)
        for dlt in deltas[1:]:
            c, s = float(np.cos(dlt)), float(np.sin(dlt))
            Rz = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], device=device)
            kp_k, uv_k, iou_k = refine_from(R0 @ Rz)
            better = (iou_k > best_iou + args.ms_margin) & (use > 0)
            if better.any():
                kp_rc = torch.where(better.view(-1, 1, 1), kp_k, kp_rc)
                uv_rc = torch.where(better.view(-1, 1, 1), uv_k, uv_rc)
                best_iou = torch.where(better, iou_k, best_iou)
                ms_switched += int(better.sum())
        # wrist-angle multi-start: the synthetic failure mode is a J4/J5/J6 (wrist) flip that
        # reprojects correctly (reproj/min-reproj can't catch it) but has a WRONG silhouette. Seed
        # hypotheses with perturbed wrist angles, re-solve R,t by Kabsch onto the solved
        # (2D-consistent) keypoints, RC-refine each, and select by SAM-IoU (external-evidence).
        if args.ms_wrist_n > 0:
            g = torch.Generator(device=device).manual_seed(1234 + lo)
            sig = float(np.deg2rad(args.ms_wrist_sigma))
            for _h in range(args.ms_wrist_n):
                th_h = th0.clone()
                th_h[:, 3:6] = th_h[:, 3:6] + torch.randn(th_h.shape[0], 3, generator=g, device=device) * sig
                fk_h = panda_forward_kinematics(th_h)
                R_h, t_h = kabsch_batch(fk_h, kpc0)
                kp_k, uv_k, iou_k = refine_from(R_h, theta_init=th_h, t_init=t_h)
                better = (iou_k > best_iou + args.ms_margin) & (use > 0)
                if better.any():
                    kp_rc = torch.where(better.view(-1, 1, 1), kp_k, kp_rc)
                    uv_rc = torch.where(better.view(-1, 1, 1), uv_k, uv_rc)
                    best_iou = torch.where(better, iou_k, best_iou)
                    ms_switched += int(better.sum())
        for b in range(len(idxs)):
            fb = found[b].bool()
            if fb.sum() < 5:
                continue
            base = float((kpc0[b][fb] - gt3d[b][fb]).norm(dim=-1).mean())
            adds_base.append(base)
            shift = float((uv_rc[b][fb] - uv_anchor[b][fb]).norm(dim=-1).mean())
            reverted = args.max_uv_shift > 0 and shift > args.max_uv_shift
            if use[b] > 0 and not reverted:
                adds_rc.append(float((kp_rc[b][fb] - gt3d[b][fb]).norm(dim=-1).mean()))
            else:
                adds_rc.append(base); skipped += 1

    print(f"\n=== DEPLOYABLE NVDR+SAM RENDER-COMPARE  {os.path.basename(args.val_dir)}  (n={len(adds_base)}, skipped {skipped}) ===")
    if sam_ious:
        si = np.array(sam_ious)
        print(f"  SAM-vs-init-render IoU: mean {si.mean():.3f}  median {np.median(si):.3f}  frac>=0.5: {(si>=0.5).mean():.2f}")
    else:
        print(f"  [no-sil] pure model-based refinement (struct_w={args.struct_w})")
    print(f"  baseline (deployed dump) ADD-AUC@100mm {add_auc(adds_base):.4f}  mean {1000*np.mean(adds_base):.1f}mm")
    print(f"  + nvdr/SAM render-compare ADD-AUC@100mm {add_auc(adds_rc):.4f}  mean {1000*np.mean(adds_rc):.1f}mm")
    print(f"  Δ: {add_auc(adds_rc)-add_auc(adds_base):+.4f}")
    if args.multi_start:
        print(f"  [multi-start] deltas ±{args.ms_deltas}°, switched hypothesis on {ms_switched} frames")


if __name__ == '__main__':
    main()
