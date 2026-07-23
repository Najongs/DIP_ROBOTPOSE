"""KUKA inference-lever A/B: reuses the deployed detector+angle head and the SOTA solver, but
exposes the non-refuted levers the baseline kuka_add_eval leaves OFF:
  --cov-pnp            Mahalanobis downweight of diffuse heatmaps (deployed on Panda, off for KUKA)
  --conf-gate G        hard-reject peak<G   (default 0.05)
  --min-kp M           PnP/refine floor (baseline 6 -> forces keeping garbage on 3+bad-kp frames)
  --anchor-init-w W    anchor angles to the ANGLE-HEAD prediction (kinematic prior) for gated joints
  --decode-window D    windowed soft-argmax (env DECODE_WINDOW) — distractor-robust decode
Reports ADD-AUC@100, mean/median ADD, and the TAIL FRACTION (>100mm) — the mission's gate metric.
Same algorithm/code path for all robots (only weights+FK differ)."""
import argparse, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch, cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
cv2.setRNGSeed(0); np.random.seed(0); torch.manual_seed(0)  # deterministic RANSAC/PnP for clean A/B

HERE = os.path.dirname(__file__)
TRAIN = os.path.abspath(os.path.join(HERE, '../../../TRAIN'))
EVAL = os.path.abspath(os.path.join(HERE, '../..'))
sys.path.append(TRAIN); sys.path.append(EVAL)
from model_angle import AnglePredictor
from model_v4 import iiwa7_forward_kinematics, _IIWA7_JOINT_LIMITS
from dataset import PoseEstimationDataset
from refine_eval import scale_K, add_auc, wrapped_abs_deg, geometric_K
import solve_pose_kinematic as spk

FK = iiwa7_forward_kinematics
KP_NAMES = [f'iiwa7_link_{i}' for i in range(1, 8)]
ANGLE_JOINTS = [f'iiwa7_joint_{i}' for i in range(1, 8)]


def _patch():
    spk.panda_forward_kinematics = iiwa7_forward_kinematics
    lims = _IIWA7_JOINT_LIMITS
    def _lim(device, dtype):
        lo = torch.tensor([l for l, _ in lims], device=device, dtype=dtype)
        hi = torch.tensor([h for _, h in lims], device=device, dtype=dtype)
        return lo, hi
    spk.make_limits = _lim
    spk.PANDA_JOINT_MEAN = torch.tensor([(l + h) / 2 for l, h in lims], dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--angle-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=6000); ap.add_argument('--iters', type=int, default=250)
    ap.add_argument('--crop-margin', type=float, default=1.5)
    ap.add_argument('--cov-pnp', action='store_true')
    ap.add_argument('--conf-gate', type=float, default=0.05)
    ap.add_argument('--min-kp', type=int, default=6)
    ap.add_argument('--anchor-init-w', type=float, default=0.0)
    ap.add_argument('--pnp-drop', type=int, default=3)
    ap.add_argument('--tag', default='ab')
    args = ap.parse_args()
    device = torch.device('cuda'); IS = args.image_size; _patch()

    m = AnglePredictor(args.model_name, IS, fix_joint7_zero=True, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.angle_head, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=KP_NAMES, image_size=(IS, IS), heatmap_size=(IS, IS),
                               augment=False, include_angles=True, sigma=2.5, crop_to_robot=True,
                               crop_margin=args.crop_margin, angle_joint_names=ANGLE_JOINTS)
    if args.max_frames and args.max_frames < len(ds):
        ds.samples = ds.samples[::max(1, len(ds.samples) // args.max_frames)][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    adds = []; n = 0
    for batch in tqdm(loader, desc=args.tag):
        img = batch['image'].to(device); gt3d = batch['keypoints_3d'].to(device)
        K = scale_K(batch['camera_K'], batch['original_size'], IS).to(device)
        K_true = geometric_K(args.val_dir, batch['camera_K'], batch['original_size'], IS).to(device)
        with torch.no_grad():
            o = m(img, K)
        init_ang = o['joint_angles']; kp2d = o['keypoints_2d']; conf = o['confidence']
        cov_inv = spk.heatmap_cov_inv(o['heatmaps_2d'], kp2d) if args.cov_pnp else None
        with torch.enable_grad():
            refined, kp_cam, reproj = spk.solve_batch(
                kp2d, conf, K_true, fix_joint7=True, iters=args.iters, lr=2e-2, img_size=IS, device=device,
                prior_w=0.0, theta_init=init_ang, cov_inv=cov_inv, conf_gate=args.conf_gate,
                min_kp=args.min_kp, pnp_drop=args.pnp_drop, anchor_init_w=args.anchor_init_w)
        valid = (gt3d.abs().sum(-1) > 0); per_j = (kp_cam - gt3d).norm(dim=-1)
        for b in range(img.shape[0]):
            if valid[b].any(): adds.append(float(per_j[b][valid[b]].mean().item()))
        n += img.shape[0]
    adds = np.array(adds)
    tail = 100 * (adds > 0.1).mean()
    print(f"\n[{args.tag}] covpnp={args.cov_pnp} gate={args.conf_gate} min_kp={args.min_kp} "
          f"anchor={args.anchor_init_w} window={os.environ.get('DECODE_WINDOW','0')}")
    print(f"  ADD-AUC@100: {add_auc(adds):.4f} | mean {adds.mean()*1000:.1f}mm | median {np.median(adds)*1000:.1f}mm "
          f"| TAIL>100mm {tail:.2f}% | {len(adds)} frames")


if __name__ == '__main__':
    main()
