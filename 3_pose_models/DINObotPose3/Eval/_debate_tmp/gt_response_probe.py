"""Decisive probe: blurred-heatmap response AT the GT 2D location (normalized by the peak max),
per keypoint. If catastrophic keypoints (argmax >10px) have near-zero response at the correct
location, the correct heatmap mode does not exist -> the multi-hypothesis decoder is dead.
Reuses the exact deployed crop pipeline. No dense solve (cheap)."""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import roi_align
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__)); EVAL = os.path.abspath(os.path.join(HERE, '..'))
TRAIN = os.path.abspath(os.path.join(EVAL, '../TRAIN'))
sys.path.append(EVAL); sys.path.append(TRAIN)
from selfbbox_eval import build_predictor, detected_bbox, project_points, crop_K, clamp_boxes
from solve_pose_kinematic import solve_batch
from decode_util import dark_decode, _gaussian_blur
from refine_eval import scale_K
from dataset import PoseEstimationDataset
from recover_gate import (IS, W0, H0, SX, SY, DET, S1ANG, S1ROT, CROPDET, CROPANG, ROT, MODEL)

device = torch.device('cuda')
det1 = build_predictor(MODEL, IS, DET, device, angle_ckpt=S1ANG, rot_ckpt=S1ROT)
cropm = build_predictor(MODEL, IS, CROPDET, device, angle_ckpt=CROPANG, rot_ckpt=ROT, head_type='mlp')
VAL = f'{TRAIN}/../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr'
ds = PoseEstimationDataset(VAL, keypoint_names=['link0','link2','link3','link4','link6','link7','hand'],
                           image_size=(IS, IS), heatmap_size=(IS, IS), augment=False,
                           include_angles=True, sigma=2.5, crop_to_robot=False)
stride = max(1, len(ds.samples) // 1000); ds.samples = ds.samples[::stride][:1000]
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

D = {k: [] for k in ['resp_ratio', 'gt_in_crop', 'err1', 'found', 'gtoff', 'conf']}
for batch in tqdm(loader, desc="gt-response"):
    img = batch['image'].to(device); gt3d = batch['keypoints_3d'].to(device)
    gt2d_is = batch['keypoints'].to(device).float()
    K = scale_K(batch['camera_K'], batch['original_size'], IS).to(device)
    bidx = torch.arange(img.shape[0], device=device).view(-1, 1).float()
    with torch.no_grad():
        o1 = det1(img, K); R1 = o1['rot_matrix']
        with torch.enable_grad():
            _, kp_cam1, _ = solve_batch(o1['keypoints_2d'], o1['confidence'], K, fix_joint7=True,
                                        iters=200, lr=2e-2, img_size=IS, device=device, prior_w=0.0,
                                        theta_init=o1['joint_angles'], conf_gate=0.05, R_init=R1)
        uv = project_points(kp_cam1.detach(), K)
        boxes = detected_bbox(uv, torch.ones(uv.shape[:2], device=device), IS, 1.5, 0.0)
        det_boxes = detected_bbox(o1['keypoints_2d'], o1['confidence'], IS, 1.5, 0.1)
        off = ((uv < -IS) | (uv > 2 * IS)).any(dim=2).any(dim=1)
        span = (uv.amax(dim=1) - uv.amin(dim=1)).amax(dim=1); huge = span > 3.0 * IS
        bad = off | huge | torch.isnan(uv).any(dim=2).any(dim=1)
        boxes = clamp_boxes(torch.where(bad.unsqueeze(1), det_boxes, boxes), IS)
        rois = torch.cat([bidx, boxes], dim=1)
        crop_img = roi_align(img, rois, output_size=(IS, IS), spatial_scale=1.0, aligned=True)
        Kc = crop_K(K, boxes, float(IS)); o2 = cropm(crop_img, Kc)
        hm = o2['heatmaps_2d']; conf = o2['confidence']
        kp2d = dark_decode(hm, sigma=2.5)                    # crop-IS argmax
        hmb = _gaussian_blur(hm.clamp(min=0), ksize=11, sigma=2.5)
        peak_max = hmb.flatten(2).max(dim=2)[0]              # (B,7) blurred peak
    side = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0).view(-1, 1)
    # GT full-frame IS -> crop-IS
    gt_crop = (gt2d_is - boxes[:, None, :2]) * (IS / side).unsqueeze(-1)   # (B,7,2)
    in_crop = ((gt_crop[..., 0] >= 0) & (gt_crop[..., 0] < IS) &
               (gt_crop[..., 1] >= 0) & (gt_crop[..., 1] < IS))
    grid = gt_crop.clone()
    grid[..., 0] = grid[..., 0] / (IS - 1) * 2 - 1; grid[..., 1] = grid[..., 1] / (IS - 1) * 2 - 1
    samp = F.grid_sample(hmb, grid.unsqueeze(1), mode='bilinear', align_corners=True)  # (B,7,1,7)
    resp = samp.squeeze(2).diagonal(dim1=1, dim2=2)         # (B,7) value at each kp's own GT loc
    ratio = (resp / (peak_max + 1e-9))
    # argmax error (orig-640) to define catastrophic
    am_full = boxes[:, None, :2] + kp2d * (side / IS).unsqueeze(-1)
    am_o = am_full.clone(); am_o[..., 0] *= SX; am_o[..., 1] *= SY
    gt_o = gt2d_is.clone(); gt_o[..., 0] *= SX; gt_o[..., 1] *= SY
    err1 = (am_o - gt_o).norm(dim=-1)
    gtoff = ((gt2d_is[..., 0] < 0) | (gt2d_is[..., 0] >= IS) | (gt2d_is[..., 1] < 0) | (gt2d_is[..., 1] >= IS))
    valid = (gt3d.abs().sum(-1) > 0)
    for b in range(img.shape[0]):
        D['resp_ratio'].append(ratio[b].cpu().numpy()); D['gt_in_crop'].append(in_crop[b].cpu().numpy())
        D['err1'].append(err1[b].cpu().numpy()); D['found'].append(valid[b].cpu().numpy())
        D['gtoff'].append(gtoff[b].cpu().numpy()); D['conf'].append(conf[b].cpu().numpy())

D = {k: np.array(v) for k, v in D.items()}
np.savez(f'{HERE}/gt_response.npz', **D)
rr = D['resp_ratio']; err = D['err1']; m = D['found'] & (~D['gtoff']) & D['gt_in_crop']
cat = m & (err > 10); clean = m & (err <= 3)
print(f"\n[GT-location blurred-heatmap response / peak-max]  (n_cat={int(cat.sum())}, n_clean={int(clean.sum())})")
for lab, mm in [('catastrophic(>10px)', cat), ('clean(<=3px)', clean)]:
    v = rr[mm]
    print(f"  {lab:22s} median={np.median(v):.3f}  mean={v.mean():.3f}  "
          f"%>0.5={np.mean(v>0.5)*100:.1f}  %>0.3={np.mean(v>0.3)*100:.1f}  %<0.1={np.mean(v<0.1)*100:.1f}")
print("  -> if catastrophic median response << clean, the correct mode is ABSENT (decoder dead).")
