"""FR3 keypoint-solver ceiling probe (cross-session mitigation test).

FR3's appearance angle head does NOT generalize cross-session (45 deg), but the detector DOES
(9.5px) and all 6 predicted joints ARE observable from keypoints (unlike Meca's wrist). So the
keypoint SOLVER should recover FR3 angles cross-session — IF the reprojection basin is reachable.

This probes the ceiling on held-out sessions (fr3_val): feed the solver oracle/detected 2D and
oracle/rot-head R, from several theta inits, and report recovered angle MAE vs GT. If oracle-2D+R
recovers angles to a few degrees, the keypoint path is the FR3 mitigation (init from keypoints,
not appearance). Panda FK is solve_pose_kinematic's native default (no monkeypatch).
"""
import argparse, os, sys, math, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(__file__); TRAIN = os.path.abspath(os.path.join(HERE, '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(HERE)
from model_angle import AnglePredictor
from dataset import PoseEstimationDataset
from refine_eval import scale_K, add_auc, wrapped_abs_deg
import solve_pose_kinematic as spk
from model_v4 import panda_forward_kinematics


def kabsch_batch(A, B):
    ca = A.mean(1, keepdim=True); cb = B.mean(1, keepdim=True)
    H = (A - ca).transpose(1, 2) @ (B - cb)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.eye(3, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(A.shape[0], 1, 1); D[:, 2, 2] = d
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    t = cb.squeeze(1) - torch.einsum('bij,bj->bi', R, ca.squeeze(1))
    return R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--angle-head', required=True)
    ap.add_argument('--rot-head', default=None)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=800); ap.add_argument('--iters', type=int, default=250)
    ap.add_argument('--oracle-2d', action='store_true'); ap.add_argument('--oracle-R', action='store_true')
    ap.add_argument('--init', default='head', choices=['head', 'mean', 'cold'])
    args = ap.parse_args()
    device = torch.device('cuda'); IS = args.image_size
    KPN = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']

    m = AnglePredictor(args.model_name, IS, head_type='mlp',
                       with_rotation=bool(args.rot_head), with_translation=bool(args.rot_head)).to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.angle_head, map_location=device))
    if args.rot_head: m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=KPN, image_size=(IS, IS), heatmap_size=(IS, IS),
                               augment=False, include_angles=True, sigma=2.5, crop_to_robot=True, crop_margin=1.5)
    if args.max_frames and args.max_frames < len(ds): ds.samples = ds.samples[::max(1, len(ds.samples)//args.max_frames)][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    raw_err = torch.zeros(6); ref_err = torch.zeros(6); n = 0; adds = []
    for batch in tqdm(loader, desc='fr3-solve'):
        img = batch['image'].to(device); gt = batch['angles'].to(device)      # (B,7)
        gt3d = batch['keypoints_3d'].to(device); K = scale_K(batch['camera_K'], batch['original_size'], IS).to(device)
        with torch.no_grad(): o = m(img, K)
        head_ang = o['joint_angles']                                          # (B,7)
        if args.init == 'head': theta_init = head_ang
        elif args.init == 'mean': theta_init = spk.PANDA_JOINT_MEAN.to(device).unsqueeze(0).expand(img.shape[0], 7).clone()
        else: theta_init = None
        kp2d = batch['keypoints'].to(device).float() if args.oracle_2d else o['keypoints_2d']
        conf = batch['valid_mask'].to(device).float().clamp(min=1e-3) if args.oracle_2d else o['confidence']
        R_init = t_init = None
        if args.oracle_R:
            ga = gt.clone(); ga[:, 6] = 0.0
            R_init, t_init = kabsch_batch(panda_forward_kinematics(ga), gt3d)
        elif args.rot_head:
            R_init = o.get('rot_matrix'); t_init = o.get('trans')
        with torch.enable_grad():
            refined, kp_cam, reproj = spk.solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters,
                                                      lr=2e-2, img_size=IS, device=device, prior_w=0.0,
                                                      theta_init=theta_init, R_init=R_init, t_init=t_init)
        raw_err += wrapped_abs_deg(head_ang[:, :6], gt[:, :6]).sum(0).cpu()
        ref_err += wrapped_abs_deg(refined[:, :6], gt[:, :6]).sum(0).cpu()
        valid = (gt3d.abs().sum(-1) > 0); per_j = (kp_cam - gt3d).norm(dim=-1)
        for b in range(img.shape[0]):
            if valid[b].any(): adds.append(float(per_j[b][valid[b]].mean().item()))
        n += img.shape[0]
    raw = (raw_err / n).numpy(); ref = (ref_err / n).numpy(); adds = np.array(adds)
    print(f"\n{'='*60}\n  FR3 SOLVER  2d={'oracle' if args.oracle_2d else 'detected'} "
          f"R={'oracle' if args.oracle_R else ('rothead' if args.rot_head else 'pnp')} init={args.init}  ({n} frames)\n{'='*60}")
    print(f"  {'joint':<6}{'head':>9}{'solved':>9}")
    for j in range(6): print(f"  J{j:<5}{raw[j]:>9.1f}{ref[j]:>9.1f}")
    print(f"  angle MAE: head {raw.mean():.1f}deg -> solved {ref.mean():.1f}deg")
    print(f"  [Pose] ADD-AUC@100mm {add_auc(adds):.4f} | median {np.median(adds)*1000:.1f}mm ({len(adds)} frames)")


if __name__ == '__main__':
    main()
