"""FR3 multi-hypothesis temporal tracking + wrist anchoring — push toward the tracking ceiling.

Single-seed tracking (fr3_track_eval) is limited by (1) a wrong frame-0 seed derailing the whole
stream and (2) weak-observability wrist drift. This fixes both:

  Multi-hypothesis: carry K seeds (head + K-1 random) through the WHOLE stream in parallel (batched
    B=K solve), accumulate per-hypothesis reprojection, and at the end keep the trajectory with the
    lowest cumulative reprojection. The correct branch stays low-reproj; wrong branches accrue error.
  Wrist anchoring: anchor_init_w > 0 anchors theta to the previous frame (temporal smoothness). The
    strongly-observed joints still follow the motion via the reprojection gradient; the weakly-observed
    wrist (J4/J5) leans on the anchor instead of drifting.

Seeding needs NO GT and NO recapture. Reports cross-session (fr3_val).
"""
import argparse, os, sys, re, math, collections, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

HERE = os.path.dirname(__file__); TRAIN = os.path.abspath(os.path.join(HERE, '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(HERE)
from model_angle import AnglePredictor
from dataset import PoseEstimationDataset
from refine_eval import scale_K, add_auc, wrapped_abs_deg
import solve_pose_kinematic as spk

KPN = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']


def stream_key_ts(sample):
    p = sample['image_path']; b = os.path.basename(p)
    m = re.match(r'zed_(\d+)_(left|right)_(\d+\.\d+)', b)
    cam, lr, ts = (m.group(1), m.group(2), float(m.group(3))) if m else ('?', '?', 0.0)
    parts = p.split('/'); sess = [x for x in parts if x.startswith('Panda_dataset')]
    view = [x for x in parts if x.startswith('view')]
    return (sess[0] if sess else '?', view[0] if view else '?', cam, lr), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--rot-head', required=True); ap.add_argument('--angle-head', required=True)
    ap.add_argument('--val-dir', default='/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset/fr3_val')
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--iters', type=int, default=120)
    ap.add_argument('--nhyp', type=int, default=12, help='number of tracked hypotheses (head + random seeds)')
    ap.add_argument('--anchor', type=float, default=0.0, help='temporal anchor weight to previous frame (wrist drift control)')
    ap.add_argument('--max-streams', type=int, default=16); ap.add_argument('--max-frames-per-stream', type=int, default=100000)
    args = ap.parse_args()
    device = torch.device('cuda'); IS = args.image_size; K = args.nhyp

    m = AnglePredictor(args.model_name, IS, head_type='mlp', with_rotation=True, with_translation=True).to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))
    m.angle_head.load_state_dict(torch.load(args.angle_head, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=KPN, image_size=(IS, IS), heatmap_size=(IS, IS),
                               augment=False, include_angles=True, sigma=2.5, crop_to_robot=True, crop_margin=1.5)
    streams = collections.defaultdict(list)
    for i, s in enumerate(ds.samples):
        k, ts = stream_key_ts(s); streams[k].append((ts, i))
    streams = {k: sorted(v) for k, v in streams.items()}
    keys = list(streams.keys())[:args.max_streams]
    lo, hi = spk.make_limits(device, torch.float32)
    g = torch.Generator(device=device).manual_seed(0)

    raw_err = torch.zeros(6); trk_err = torch.zeros(6); n = 0; adds = []
    for key in tqdm(keys, desc='mh-track'):
        idxs = [i for _, i in streams[key]][:args.max_frames_per_stream]
        # PASS 1: batch detector/heads over the stream
        loader = DataLoader(Subset(ds, idxs), batch_size=32, shuffle=False, num_workers=6)
        KP, CF, RI, HA, GT, G3, KK = [], [], [], [], [], [], []
        for b in loader:
            img = b['image'].to(device); Kb = scale_K(b['camera_K'], b['original_size'], IS).to(device)
            with torch.no_grad():
                o = m(img, Kb)
            KP.append(o['keypoints_2d']); CF.append(o['confidence']); RI.append(o['rot_matrix'])
            HA.append(o['joint_angles']); GT.append(b['angles'].to(device)); G3.append(b['keypoints_3d'].to(device)); KK.append(Kb)
        KP = torch.cat(KP); CF = torch.cat(CF); RI = torch.cat(RI); HA = torch.cat(HA); GT = torch.cat(GT); G3 = torch.cat(G3); KK = torch.cat(KK)
        F = len(idxs)
        # PASS 2: K hypotheses tracked in parallel (batched B=K)
        # frame-0 seeds: hyp0 = head; hyp1.. = random within limits
        seeds = HA[0:1].repeat(K, 1).clone()
        rnd = (torch.rand(K - 1, 7, generator=g, device=device) * 2 - 1) * math.pi; rnd[:, 6] = 0.0
        seeds[1:, :6] = rnd[:, :6]
        theta = seeds                                  # (K,7)
        cum_reproj = torch.zeros(K, device=device)
        store_th = torch.zeros(F, K, 7, device=device)
        store_kc = torch.zeros(F, K, 7, 3, device=device)
        for fi in range(F):
            kp2d = KP[fi:fi+1].repeat(K, 1, 1); conf = CF[fi:fi+1].repeat(K, 1)
            Kf = KK[fi:fi+1].repeat(K, 1, 1); R_init = RI[fi:fi+1].repeat(K, 1, 1)
            with torch.enable_grad():
                refined, kp_cam, reproj = spk.solve_batch(kp2d, conf, Kf, fix_joint7=True, iters=args.iters, lr=2e-2,
                                                          img_size=IS, device=device, prior_w=0.0,
                                                          theta_init=theta, R_init=R_init, anchor_init_w=args.anchor)
            theta = refined.detach()
            cum_reproj = cum_reproj + reproj.detach()
            store_th[fi] = theta; store_kc[fi] = kp_cam.detach()
        best = int(cum_reproj.argmin())
        for fi in range(F):
            gt = GT[fi:fi+1]; gt3d = G3[fi:fi+1]
            raw_err += wrapped_abs_deg(HA[fi:fi+1, :6], gt[:, :6]).sum(0).cpu()
            trk_err += wrapped_abs_deg(store_th[fi, best:best+1, :6], gt[:, :6]).sum(0).cpu()
            valid = (gt3d.abs().sum(-1) > 0); pj = (store_kc[fi, best:best+1] - gt3d).norm(dim=-1)
            if valid[0].any(): adds.append(float(pj[0][valid[0]].mean()))
            n += 1
    raw = (raw_err / n).numpy(); trk = (trk_err / n).numpy(); adds = np.array(adds)
    print(f"\n{'='*60}\n  FR3 MULTI-HYP TRACKING  K={K} anchor={args.anchor}  ({n} frames, {len(keys)} streams)\n{'='*60}")
    print(f"  {'joint':<6}{'head':>9}{'tracked':>9}")
    for j in range(6): print(f"  J{j:<5}{raw[j]:>9.1f}{trk[j]:>9.1f}")
    print(f"  angle MAE: head {raw.mean():.1f}deg -> tracked {trk.mean():.1f}deg")
    print(f"  [Pose] ADD-AUC@100mm {add_auc(adds):.4f} | median {np.median(adds)*1000:.1f}mm")


if __name__ == '__main__':
    main()
