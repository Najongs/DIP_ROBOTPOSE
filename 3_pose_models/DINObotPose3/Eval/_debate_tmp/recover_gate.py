"""
DISTAL-KEYPOINT RECOVERABILITY GATE (measure-only; no train, no commit).

Replicates the DEPLOYED synth crop pipeline (selfbbox_eval --bbox-from-solved --bbox-guard
--cov-pnp --dark-decode) EXACTLY by reusing its helper functions, then for every keypoint dumps
the top-K NMS heatmap modes (sub-pixel DARK-refined, in original-640 px) so we can measure whether
the CORRECT-link location survives as a secondary heatmap mode when the argmax (top-1) is
catastrophically wrong (>10px). Also runs the dense heatmap refiner (solve_batch_heatmap) as a
bonus, and reproduces the base argmax ADD-AUC (~0.704) as a sanity anchor.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import roi_align
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.abspath(os.path.join(HERE, '..'))
TRAIN = os.path.abspath(os.path.join(EVAL, '../TRAIN'))
sys.path.append(EVAL); sys.path.append(TRAIN)

from selfbbox_eval import (build_predictor, detected_bbox, project_points,  # noqa
                           crop_K, clamp_boxes)
from solve_pose_kinematic import solve_batch, solve_batch_heatmap, heatmap_cov_inv  # noqa
from decode_util import dark_decode, _gaussian_blur  # noqa
from refine_eval import add_auc, scale_K  # noqa
from dataset import PoseEstimationDataset  # noqa

IS = 512
W0, H0 = 640, 480          # panda synth DR original frame
SX, SY = W0 / IS, H0 / IS   # IS512 -> orig640 (anisotropic)

# deployed checkpoints (oracle_angle_synth.sh / eval_synth_head.sh)
DET   = f'{TRAIN}/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth'
S1ANG = f'{TRAIN}/outputs_angle/angle_20260603_013948/best_angle_head.pth'
S1ROT = f'{TRAIN}/outputs_rotation/rot_20260604_162336/best_rot_head.pth'
CROPDET = os.environ.get('CROPDET', f'{TRAIN}/outputs_heatmap/crop_20260605_010622/best_heatmap.pth')
CROPANG = f'{TRAIN}/outputs_angle/angle_crop_20260605_174740/best_angle_head.pth'  # deployed mlp (base 0.704)
ROT   = f'{TRAIN}/outputs_rotation/rot_crop_20260606_022535/best_rot_head.pth'
MODEL = 'facebook/dinov3-vitb16-pretrain-lvd1689m'


def extract_peaks(heatmaps, topk=5, nms_ksize=17, sigma=2.5):
    """Top-K NMS local maxima of each heatmap, sub-pixel DARK-refined. Blur matches dark_decode so
    the top-1 peak reproduces the deployed dark-decode kp2d exactly.
    Returns coords (B,N,topk,2) in crop-IS px, vals (B,N,topk) blurred-heatmap peak values."""
    hm = heatmaps.detach().clamp(min=0)
    hmb = _gaussian_blur(hm, ksize=11, sigma=sigma)
    B, N, H, W = hmb.shape
    pooled = F.max_pool2d(hmb, kernel_size=nms_ksize, stride=1, padding=nms_ksize // 2)
    ispeak = hmb >= pooled
    masked = torch.where(ispeak, hmb, torch.full_like(hmb, -1.0))
    vals, idx = masked.reshape(B, N, -1).topk(topk, dim=-1)     # (B,N,topk)
    px = (idx % W).long(); py = (idx // W).long()
    logh = hmb.clamp(min=1e-10).log()
    bi = torch.arange(B, device=hm.device)[:, None, None]
    ni = torch.arange(N, device=hm.device)[None, :, None]

    def at(dy, dx):
        yy = (py + dy).clamp(0, H - 1); xx = (px + dx).clamp(0, W - 1)
        return logh[bi, ni, yy, xx]                              # (B,N,topk)

    c = at(0, 0)
    Dx = (at(0, 1) - at(0, -1)) / 2;  Dy = (at(1, 0) - at(-1, 0)) / 2
    Dxx = at(0, 1) - 2 * c + at(0, -1); Dyy = at(1, 0) - 2 * c + at(-1, 0)
    Dxy = (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)) / 4
    det = Dxx * Dyy - Dxy * Dxy; ok = det.abs() > 1e-6
    inv00 = torch.where(ok, Dyy / det, torch.zeros_like(det))
    inv01 = torch.where(ok, -Dxy / det, torch.zeros_like(det))
    inv11 = torch.where(ok, Dxx / det, torch.zeros_like(det))
    ox = (-(inv00 * Dx + inv01 * Dy)).clamp(-1, 1).nan_to_num()
    oy = (-(inv01 * Dx + inv11 * Dy)).clamp(-1, 1).nan_to_num()
    coords = torch.stack([px.float() + ox, py.float() + oy], dim=-1)  # (B,N,topk,2)
    return coords, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-dir', default=f'{TRAIN}/../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr')
    ap.add_argument('--max-frames', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--nms-ksize', type=int, default=17)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--conf-gate', type=float, default=0.05)
    ap.add_argument('--margin', type=float, default=1.5)
    ap.add_argument('--bbox-conf', type=float, default=0.1)
    ap.add_argument('--run-hm-solve', action='store_true', help='bonus: dense heatmap refiner ADD')
    ap.add_argument('--out', default=f'{HERE}/gate_peaks.npz')
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available()
    det1 = build_predictor(MODEL, IS, DET, device, angle_ckpt=S1ANG, rot_ckpt=S1ROT)
    cropm = build_predictor(MODEL, IS, CROPDET, device, angle_ckpt=CROPANG, rot_ckpt=ROT,
                            head_type='mlp')
    print(f"val: {args.val_dir}\ncrop det: {CROPDET}\ncrop ang: {CROPANG}", flush=True)

    ds = PoseEstimationDataset(args.val_dir,
                               keypoint_names=['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand'],
                               image_size=(IS, IS), heatmap_size=(IS, IS),
                               augment=False, include_angles=True, sigma=2.5, crop_to_robot=False)
    if args.max_frames and args.max_frames < len(ds):
        stride = max(1, len(ds.samples) // args.max_frames)
        ds.samples = ds.samples[::stride][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    D = {k: [] for k in ['fid', 'peaks_orig', 'peak_vals', 'conf', 'gt2d_orig', 'gtoff', 'found',
                         'gt3d', 'base_add', 'hm_add', 'argmax_orig']}
    for batch in tqdm(loader, desc="recover-gate"):
        img = batch['image'].to(device)
        gt3d = batch['keypoints_3d'].to(device)
        gt2d_is = batch['keypoints'].to(device).float()                 # (B,7,2) IS512 full-frame
        K = scale_K(batch['camera_K'], batch['original_size'], IS).to(device)
        bidx = torch.arange(img.shape[0], device=device).view(-1, 1).float()
        with torch.no_grad():
            # PASS 1: full-frame detect + solve -> project 7 FK kp -> bbox (bbox_from_solved + guard)
            o1 = det1(img, K); R1 = o1['rot_matrix']
            with torch.enable_grad():
                _, kp_cam1, reproj1 = solve_batch(o1['keypoints_2d'], o1['confidence'], K,
                                                  fix_joint7=True, iters=args.iters, lr=2e-2, img_size=IS,
                                                  device=device, prior_w=0.0, theta_init=o1['joint_angles'],
                                                  conf_gate=args.conf_gate, R_init=R1)
            kp_cam1 = kp_cam1.detach()
            uv = project_points(kp_cam1, K)
            boxes = detected_bbox(uv, torch.ones(uv.shape[:2], device=device), IS, args.margin, 0.0)
            det_boxes = detected_bbox(o1['keypoints_2d'], o1['confidence'], IS, args.margin, args.bbox_conf)
            off = ((uv < -IS) | (uv > 2 * IS)).any(dim=2).any(dim=1)
            span = (uv.amax(dim=1) - uv.amin(dim=1)).amax(dim=1); huge = span > 3.0 * IS
            bad = off | huge | torch.isnan(uv).any(dim=2).any(dim=1)
            boxes = torch.where(bad.unsqueeze(1), det_boxes, boxes)
            boxes = clamp_boxes(boxes, IS)
            # crop + final detect
            rois = torch.cat([bidx, boxes], dim=1)
            crop_img = roi_align(img, rois, output_size=(IS, IS), spatial_scale=1.0, aligned=True)
            Kc = crop_K(K, boxes, float(IS))
            o2 = cropm(crop_img, Kc)
            hm = o2['heatmaps_2d']                                       # (B,7,512,512) crop-IS
            conf = o2['confidence']
            kp2d = dark_decode(hm, sigma=2.5)                            # deployed top-1 (crop-IS)
            coords, vals = extract_peaks(hm, topk=args.topk, nms_ksize=args.nms_ksize, sigma=2.5)

        # map crop-IS -> full-frame IS -> orig-640
        side = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0).view(-1, 1)     # (B,1)
        def to_orig(c):  # c (...,2) crop-IS -> orig640
            full = boxes[:, None, :2] + c.reshape(c.shape[0], -1, 2) * (side / IS).unsqueeze(-1)
            full = full.reshape(c.shape)
            o = full.clone(); o[..., 0] *= SX; o[..., 1] *= SY
            return o
        peaks_orig = to_orig(coords)                                     # (B,7,topk,2)
        argmax_orig = to_orig(kp2d)                                      # (B,7,2)
        gt2d_orig = gt2d_is.clone(); gt2d_orig[..., 0] *= SX; gt2d_orig[..., 1] *= SY
        gtoff = ((gt2d_is[..., 0] < 0) | (gt2d_is[..., 0] >= IS) |
                 (gt2d_is[..., 1] < 0) | (gt2d_is[..., 1] >= IS))
        valid = (gt3d.abs().sum(-1) > 0)

        # base argmax solve (reproduce 0.704) + bonus dense heatmap refiner
        init_ang = o2['joint_angles']
        cov_inv = heatmap_cov_inv(hm, kp2d)
        with torch.enable_grad():
            _, kp_cam, _ = solve_batch(kp2d, conf, Kc, fix_joint7=True, iters=args.iters, lr=2e-2,
                                       img_size=IS, device=device, prior_w=0.0, theta_init=init_ang,
                                       conf_gate=args.conf_gate, R_init=o2['rot_matrix'], cov_inv=cov_inv)
        base_add = (kp_cam - gt3d).norm(dim=-1)                          # (B,7)
        if args.run_hm_solve:
            with torch.enable_grad():
                _, kp_cam_hm, _ = solve_batch_heatmap(hm, Kc, fix_joint7=True, iters=args.iters,
                                                      theta_init=init_ang, device=device)
            hm_add = (kp_cam_hm - gt3d).norm(dim=-1)
        else:
            hm_add = torch.zeros_like(base_add)

        for b in range(img.shape[0]):
            D['fid'].append(batch['name'][b]); D['peaks_orig'].append(peaks_orig[b].cpu().numpy())
            D['peak_vals'].append(vals[b].cpu().numpy()); D['conf'].append(conf[b].cpu().numpy())
            D['gt2d_orig'].append(gt2d_orig[b].cpu().numpy()); D['gtoff'].append(gtoff[b].cpu().numpy())
            D['found'].append(valid[b].cpu().numpy()); D['gt3d'].append(gt3d[b].cpu().numpy())
            D['base_add'].append(base_add[b].cpu().numpy()); D['hm_add'].append(hm_add[b].cpu().numpy())
            D['argmax_orig'].append(argmax_orig[b].cpu().numpy())

    np.savez(args.out, **{k: np.array(v) for k, v in D.items()})
    # in-script sanity: reproduce base ADD-AUC and the argmax >10px tail
    ba = np.array(D['base_add']); v = np.array(D['found'])
    frame_add = np.array([ba[i][v[i]].mean() for i in range(len(ba)) if v[i].any()])
    print(f"\n[sanity] base argmax ADD-AUC = {add_auc(frame_add):.4f}  (expect ~0.704)", flush=True)
    if args.run_hm_solve:
        ha = np.array(D['hm_add'])
        fh = np.array([ha[i][v[i]].mean() for i in range(len(ha)) if v[i].any()])
        print(f"[bonus]  dense heatmap-refiner ADD-AUC = {add_auc(fh):.4f}", flush=True)
    am = np.array(D['argmax_orig']); g2 = np.array(D['gt2d_orig']); go = np.array(D['gtoff'])
    err = np.linalg.norm(am - g2, axis=2); m = v & (~go)
    print(f"[sanity] argmax >10px(orig) tail = {(err[m] > 10).mean()*100:.2f}%  median={np.median(err[m]):.2f}px  (expect ~6.5% / 1.46px)", flush=True)
    print(f"[dump] -> {args.out}  ({len(D['fid'])} frames)", flush=True)


if __name__ == '__main__':
    main()
