"""FR3 two-view (stereo) pose solve — resolves the 7-DOF monocular ambiguity.

Single-view solve lands in wrong low-reproj basins (oracle-2D still 35deg). A stereo pair (ZED
left+right, baseline ~133mm, recovered from data) with KNOWN relative pose constrains the solution
uniquely: optimize (theta, base_pose) to reproject correctly in BOTH views at once. This probes
whether the 2nd view removes the ambiguity — oracle-2D + mean init should now recover angles.

Stereo extrinsic T_LR (left-cam -> right-cam) is recovered per pair via Kabsch(gt3d_L, gt3d_R)
(in deployment: a fixed per-camera average). Pairs matched by (session, camera, joint-config).
"""
import argparse, os, sys, glob, json, math, collections, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from tqdm import tqdm

HERE = os.path.dirname(__file__); TRAIN = os.path.abspath(os.path.join(HERE, '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(HERE)
from model_v4 import panda_forward_kinematics, _PANDA_JOINT_LIMITS
from solve_pose_kinematic import rot6d_to_matrix, matrix_to_rot6d, theta_to_p, p_to_theta, PANDA_JOINT_MEAN
from refine_eval import wrapped_abs_deg, add_auc

KPN = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']


def kabsch(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0); H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T; t = Q.mean(0) - R @ P.mean(0)
    return R, t


def load(jf):
    d = json.load(open(jf)); o = d['objects'][0]; p = d['meta']['image_path']
    kp3 = {k['name']: np.array(k['location']) for k in o['keypoints']}
    kp2 = {k['name']: np.array(k['projected_location']) for k in o['keypoints']}
    g3 = np.array([kp3[[n for n in kp3 if k in n][0]] for k in KPN], np.float64)
    g2 = np.array([kp2[[n for n in kp2 if k in n][0]] for k in KPN], np.float64)
    ang = tuple(np.round([x['position'] for x in d['sim_state']['joints'][:7]], 3))
    parts = p.split('/'); cam = [x for x in parts if x.startswith('zed_')][0].split('_')[1]
    lr = 'left' if '_left_' in p else 'right'
    sess = [x for x in parts if x.startswith('Panda_dataset')][0]
    return dict(cam=cam, lr=lr, sess=sess, ang=ang, g3=g3, g2=g2, K=np.array(d['meta']['K'], np.float64))


def project(cam_pts, K):  # (N,3)->(N,2)
    z = cam_pts[:, 2].clamp(min=1e-3)
    u = cam_pts[:, 0] / z * K[0, 0] + K[0, 2]; v = cam_pts[:, 1] / z * K[1, 1] + K[1, 2]
    return torch.stack([u, v], -1)


def solve_two_view(g2L, KL, g2R, KR, R_LR, t_LR, theta0, R0, t0, lo, hi, iters=300, lr=2e-2, two_view=True):
    dev = 'cpu'
    fkfn = panda_forward_kinematics
    p = theta_to_p(theta0, lo, hi).clone().detach().requires_grad_(True)
    d6 = matrix_to_rot6d(R0.unsqueeze(0)).squeeze(0).clone().detach().requires_grad_(True)
    t = t0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([p, d6, t], lr=lr)
    for _ in range(iters):
        th = p_to_theta(p, lo, hi); th7 = torch.cat([th, torch.zeros(1)])
        fk = fkfn(th7.unsqueeze(0))[0]                    # (7,3) robot frame
        R = rot6d_to_matrix(d6.unsqueeze(0))[0]
        camL = fk @ R.T + t                               # left cam frame
        loss = ((project(camL, KL) - g2L) ** 2).sum(-1).mean()
        if two_view:
            camR = camL @ R_LR.T + t_LR
            loss = loss + ((project(camR, KR) - g2R) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        th = p_to_theta(p, lo, hi)
    return th.detach()


FIX_POSE = False


def _solve_once(rels, theta0, R0, t0, lo, hi, iters=150, lr=3e-2):
    p = theta_to_p(theta0, lo, hi).clone().detach().requires_grad_(True)
    d6 = matrix_to_rot6d(R0.unsqueeze(0)).squeeze(0).clone().detach()
    t = t0.clone().detach()
    params = [p]
    if not FIX_POSE:
        d6 = d6.requires_grad_(True); t = t.requires_grad_(True); params += [d6, t]
    opt = torch.optim.Adam(params, lr=lr)
    loss = torch.tensor(0.0)
    for _ in range(iters):
        th = p_to_theta(p, lo, hi); th7 = torch.cat([th, torch.zeros(1)])
        fk = panda_forward_kinematics(th7.unsqueeze(0))[0]
        R = rot6d_to_matrix(d6.unsqueeze(0))[0]; cam0 = fk @ R.T + t
        loss = sum(((project(cam0 @ R0i.T + t0i, Ki) - g2i) ** 2).sum(-1).mean() for R0i, t0i, g2i, Ki in rels)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return p_to_theta(p, lo, hi).detach(), float(loss)


def solve_nview(rels, theta0, R0, t0, lo, hi, nstart=1):
    """Multi-start: nstart random theta inits, keep the min-reprojection solution."""
    best_th, best_l = None, 1e18
    for s in range(nstart):
        ti = theta0 if s == 0 else torch.empty(6).uniform_(-1, 1) * (hi - lo) / 2 + (hi + lo) / 2
        th, l = _solve_once(rels, ti, R0, t0, lo, hi)
        if l < best_l: best_l, best_th = l, th
    return best_th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-dir', default='/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset/fr3_val')
    ap.add_argument('--max-pairs', type=int, default=300)
    ap.add_argument('--ndist', type=int, default=3, help='min distinct cameras per config set')
    ap.add_argument('--single', action='store_true', help='ablation: single-view only (should stay ambiguous)')
    ap.add_argument('--init', default='mean', choices=['mean', 'oracle'])
    ap.add_argument('--nstart', type=int, default=1, help='multi-start random inits (min-reproj wins)')
    ap.add_argument('--fix-pose', action='store_true', help='fix base R,t at oracle; solve theta only')
    args = ap.parse_args()
    global FIX_POSE; FIX_POSE = args.fix_pose
    lo = torch.tensor([l for l, _ in _PANDA_JOINT_LIMITS[:6]]); hi = torch.tensor([h for _, h in _PANDA_JOINT_LIMITS[:6]])

    # group by (session, joint-config); collect ONE image per distinct camera (widest baseline)
    groups = collections.defaultdict(dict)
    for jf in glob.glob(args.val_dir + '/*.json'):
        r = load(jf)
        g = groups[(r['sess'], r['ang'])]
        if r['cam'] not in g:                       # first (left) image per distinct camera
            g[r['cam']] = r
    ndist = args.ndist
    sets = [(k, list(v.values())) for k, v in groups.items() if len(v) >= ndist]
    sets = sets[:args.max_pairs]
    print(f"config sets with >={ndist} distinct cameras: {len(sets)}  mode={'SINGLE' if args.single else f'{ndist}-view-WIDE'}  init={args.init}")

    errs = []
    for (s, ang), views in tqdm(sets, desc='solve'):
        views = views[:ndist]
        gtang = torch.tensor(ang[:6], dtype=torch.float32)
        v0 = views[0]
        # relative pose view0->view_i from GT 3d (recoverable calibration)
        rels = []
        for vi in views:
            Ri, ti = kabsch(v0['g3'], vi['g3'])
            rels.append((torch.tensor(Ri, dtype=torch.float32), torch.tensor(ti, dtype=torch.float32),
                         torch.tensor(vi['g2'], dtype=torch.float32), torch.tensor(vi['K'], dtype=torch.float32)))
        ga = np.concatenate([np.array(ang[:6]), [0.0]])
        fk_gt = panda_forward_kinematics(torch.tensor(ga).unsqueeze(0))[0].numpy()
        Rb_np, tb_np = kabsch(fk_gt, v0['g3'])
        R0 = torch.tensor(Rb_np, dtype=torch.float32); t0 = torch.tensor(tb_np, dtype=torch.float32)
        theta0 = gtang.clone() if args.init == 'oracle' else PANDA_JOINT_MEAN[:6].clone()
        nv = 1 if args.single else len(rels)
        th = solve_nview(rels[:nv], theta0, R0, t0, lo, hi, nstart=args.nstart)
        errs.append(wrapped_abs_deg(th.unsqueeze(0), gtang.unsqueeze(0))[0])
    errs = torch.stack(errs); per = errs.mean(0)
    print(f"  recovered angle MAE(J0-5) = {per.mean():.1f} deg   per-joint=" + ",".join(f"{x:.1f}" for x in per))
    print(f"  (single-view was ~32deg; wide multi-view should drop if wide baseline breaks the ambiguity)")


if __name__ == '__main__':
    main()
