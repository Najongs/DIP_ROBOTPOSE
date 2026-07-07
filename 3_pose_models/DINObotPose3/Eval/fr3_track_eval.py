"""FR3 temporal tracking — resolve 7-DOF monocular ambiguity via trajectory continuity.

A single FR3 frame is ambiguous (many low-reproj joint configs). But the capture is a smooth
continuous trajectory (median 2.4deg/frame joint change), so initializing each frame's solve from
the PREVIOUS frame's solution keeps it in the correct basin. The only hard part is seeding frame 0.

Per camera-stream (session,view,cam,lr) sorted by timestamp:
  theta[0] = seed (--seed gt|head|multistart)
  theta[i] = solve_batch(detected_2d[i], R_init=rothead[i], theta_init=theta[i-1])
Reports angle MAE + ADD-AUC vs GT on held-out sessions (fr3_val = cross-session).
"""
import argparse, os, sys, re, math, collections, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from tqdm import tqdm

HERE = os.path.dirname(__file__); TRAIN = os.path.abspath(os.path.join(HERE, '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(HERE)
from model_angle import AnglePredictor
from dataset import PoseEstimationDataset
from refine_eval import scale_K, add_auc, wrapped_abs_deg
import solve_pose_kinematic as spk
from model_v4 import panda_forward_kinematics

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
    ap.add_argument('--detector', required=True); ap.add_argument('--rot-head', required=True)
    ap.add_argument('--angle-head', required=True)
    ap.add_argument('--val-dir', default='/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset/fr3_val')
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--seed', default='gt', choices=['gt', 'head', 'multistart'])
    ap.add_argument('--nstart', type=int, default=24)
    ap.add_argument('--max-streams', type=int, default=40)
    ap.add_argument('--max-frames-per-stream', type=int, default=100000)
    args = ap.parse_args()
    device = torch.device('cuda'); IS = args.image_size

    m = AnglePredictor(args.model_name, IS, head_type='mlp', with_rotation=True, with_translation=True).to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))
    m.angle_head.load_state_dict(torch.load(args.angle_head, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=KPN, image_size=(IS, IS), heatmap_size=(IS, IS),
                               augment=False, include_angles=True, sigma=2.5, crop_to_robot=True, crop_margin=1.5)
    # group sample indices into streams, sorted by timestamp
    streams = collections.defaultdict(list)
    for i, s in enumerate(ds.samples):
        k, ts = stream_key_ts(s); streams[k].append((ts, i))
    streams = {k: sorted(v) for k, v in streams.items()}
    keys = list(streams.keys())[:args.max_streams]

    from torch.utils.data import DataLoader, Subset
    lo, hi = spk.make_limits(device, torch.float32)
    raw_err = torch.zeros(6); trk_err = torch.zeros(6); n = 0; adds = []
    for k in tqdm(keys, desc='track'):
        idxs = [i for _, i in streams[k]][:args.max_frames_per_stream]
        # PASS 1: batch the detector/rot/head forward over the whole stream (independent)
        loader = DataLoader(Subset(ds, idxs), batch_size=32, shuffle=False, num_workers=6)
        KP, CF, RI, HA, GT, G3, KK = [], [], [], [], [], [], []
        for b in loader:
            img = b['image'].to(device)
            K = scale_K(b['camera_K'], b['original_size'], IS).to(device)
            with torch.no_grad():
                o = m(img, K)
            KP.append(o['keypoints_2d']); CF.append(o['confidence']); RI.append(o['rot_matrix'])
            HA.append(o['joint_angles']); GT.append(b['angles'].to(device))
            G3.append(b['keypoints_3d'].to(device)); KK.append(K)
        KP = torch.cat(KP); CF = torch.cat(CF); RI = torch.cat(RI); HA = torch.cat(HA)
        GT = torch.cat(GT); G3 = torch.cat(G3); KK = torch.cat(KK)
        # PASS 2: sequential solve, init-from-prev (fast — no backbone)
        prev_theta = None
        for fi in range(len(idxs)):
            kp2d = KP[fi:fi+1]; conf = CF[fi:fi+1]; R_init = RI[fi:fi+1]; K = KK[fi:fi+1]
            gt = GT[fi:fi+1]; gt3d = G3[fi:fi+1]
            if fi == 0:
                theta0 = (gt.clone() if args.seed == 'gt' else HA[fi:fi+1].clone())
                if args.seed == 'gt': theta0[:, 6] = 0.0
            else:
                theta0 = prev_theta
            with torch.enable_grad():
                if fi == 0 and args.seed == 'multistart':
                    best = None; bestr = 1e18
                    g = torch.Generator(device=device).manual_seed(0)
                    for s in range(args.nstart):
                        ti = (torch.rand(1, 7, generator=g, device=device) * 2 - 1) * math.pi; ti[:, 6] = 0.0
                        th, kc, rp = spk.solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters, lr=2e-2,
                                                     img_size=IS, device=device, prior_w=0.0, theta_init=ti, R_init=R_init)
                        if float(rp) < bestr: bestr = float(rp); best = (th, kc)
                    refined, kp_cam = best
                else:
                    refined, kp_cam, _ = spk.solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters, lr=2e-2,
                                                         img_size=IS, device=device, prior_w=0.0,
                                                         theta_init=theta0, R_init=R_init)
            prev_theta = refined.detach()
            raw_err += wrapped_abs_deg(HA[fi:fi+1, :6], gt[:, :6]).sum(0).cpu()
            trk_err += wrapped_abs_deg(refined[:, :6], gt[:, :6]).sum(0).cpu()
            valid = (gt3d.abs().sum(-1) > 0); pj = (kp_cam - gt3d).norm(dim=-1)
            if valid[0].any(): adds.append(float(pj[0][valid[0]].mean()))
            n += 1
    raw = (raw_err / n).numpy(); trk = (trk_err / n).numpy(); adds = np.array(adds)
    print(f"\n{'='*58}\n  FR3 TEMPORAL TRACKING  seed={args.seed}  ({n} frames, {len(keys)} streams)\n{'='*58}")
    print(f"  {'joint':<6}{'head':>9}{'tracked':>9}")
    for j in range(6): print(f"  J{j:<5}{raw[j]:>9.1f}{trk[j]:>9.1f}")
    print(f"  angle MAE: head {raw.mean():.1f}deg -> tracked {trk.mean():.1f}deg")
    print(f"  [Pose] ADD-AUC@100mm {add_auc(adds):.4f} | median {np.median(adds)*1000:.1f}mm")


if __name__ == '__main__':
    main()
