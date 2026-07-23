"""FR3 GT-exoneration / data-consistency check (pure geometry, no model, no solver).

For each frame in a DREAM-converted FR3 dir:
  angles  = sim_state.joints[0:7].position          (panda_joint1..7, encoder GT)
  FK_base = panda_forward_kinematics(angles)         (7,3) robot-base frame [link0,2,3,4,6,7,hand]
  GT3D    = objects[0].keypoints[i].location         (7,3) CAMERA frame  (annotation)
  GT2D    = objects[0].keypoints[i].projected_location (7,2) pixels      (annotation)
  K,dist  = meta.K, meta.dist_coeffs

Checks:
  A  per-frame Kabsch(FK_base -> GT3D) residual  [mm]   (3D self-consistency: does FK(angles) rigidly match GT3D?)
  B  project(GT3D, K, dist) vs GT2D              [px]   (intrinsic consistency)
  C  project(Kabsch(FK)_cam, K, dist) vs GT2D    [px]   (full chain angles->2D)
  D  within static (session,cam) group: std of GT base(link0) camera-frame position [mm]
     and std of recovered camera pose (t mm / R deg)      (ArUco / extrinsic stability = GT noise floor)
"""
import os, sys, glob, json, math, re, argparse
import numpy as np
import torch

TRAIN = '/home/najo/NAS/DIP/3_pose_models/DINObotPose3/TRAIN'
sys.path.append(TRAIN)
from model_v4 import panda_forward_kinematics
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

KP_ORDER = ['panda_link0','panda_link2','panda_link3','panda_link4','panda_link6','panda_link7','panda_hand']


def kabsch(A, B):
    """rigid R,t mapping A(N,3)->B(N,3). returns R(3,3), t(3), per-point residual(N)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    res = np.linalg.norm((A @ R.T + t) - B, axis=1)
    return R, t, res


def project(pts_cam, K, dist):
    if HAVE_CV2:
        p, _ = cv2.projectPoints(pts_cam.reshape(-1,1,3).astype(np.float64),
                                 np.zeros(3), np.zeros(3), K.astype(np.float64),
                                 np.asarray(dist, np.float64))
        return p.reshape(-1, 2)
    z = np.clip(pts_cam[:,2], 1e-6, None)
    u = pts_cam[:,0]/z*K[0,0] + K[0,2]; v = pts_cam[:,1]/z*K[1,1] + K[1,2]
    return np.stack([u, v], 1)


def session_key(meta, fname):
    ip = meta.get('image_path','')
    m = re.search(r'(Panda_dataset[^/]*)', ip)
    sess = m.group(1) if m else '?'
    cam = re.search(r'zed_(\d+)_(left|right)', fname)
    camid = f"{cam.group(1)}_{cam.group(2)}" if cam else '?'
    view = meta.get('view','?')
    return f"{sess}|{view}|{camid}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--n', type=int, default=400)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, '*.json')))
    if args.n and args.n < len(files):
        files = files[::max(1, len(files)//args.n)][:args.n]

    A_res = []          # per-keypoint 3D Kabsch residual (mm)
    A_frame_med = []    # per-frame median residual (mm)
    B_res = []          # 3D->2D reprojection (px)
    C_res = []          # angles->2D reprojection (px)
    groups = {}         # session -> list of (link0_cam(3), R, t)
    depths = []
    nkp_bad = 0; nframes = 0
    for f in files:
        d = json.load(open(f))
        obj = d['objects'][0]
        kps = {k['name']: k for k in obj['keypoints']}
        if not all(n in kps for n in KP_ORDER):
            nkp_bad += 1; continue
        gt3d = np.array([kps[n]['location'] for n in KP_ORDER], float)          # (7,3) cam
        gt2d = np.array([kps[n]['projected_location'] for n in KP_ORDER], float) # (7,2)
        joints = d['sim_state']['joints']
        ang = np.array([joints[i]['position'] for i in range(7)], float)         # panda_joint1..7
        fk = panda_forward_kinematics(torch.tensor(ang, dtype=torch.float64).unsqueeze(0))[0].numpy()  # (7,3) base
        K = np.array(d['meta']['K'], float)
        dist = d['meta'].get('dist_coeffs', [0,0,0,0,0])

        # A: 3D Kabsch
        R, t, res = kabsch(fk, gt3d)
        A_res.extend((res*1000).tolist()); A_frame_med.append(float(np.median(res)*1000))
        # B: 3D->2D
        p2 = project(gt3d, K, dist)
        B_res.extend(np.linalg.norm(p2 - gt2d, axis=1).tolist())
        # C: angles->cam->2D
        fk_cam = fk @ R.T + t
        p2c = project(fk_cam, K, dist)
        C_res.extend(np.linalg.norm(p2c - gt2d, axis=1).tolist())
        depths.append(float(gt3d[:,2].mean()))
        g = session_key(d['meta'], os.path.basename(f))
        groups.setdefault(g, []).append((gt3d[0], R, t))
        nframes += 1

    def stats(x, u):
        x = np.array(x)
        return f"median {np.median(x):.2f} | mean {np.mean(x):.2f} | p90 {np.percentile(x,90):.2f} | p99 {np.percentile(x,99):.2f} | max {x.max():.2f} {u}"

    print(f"\n================ FR3 GT CONSISTENCY  ({nframes} frames, {os.path.basename(args.dir)}) ================")
    print(f"  cv2 distortion: {HAVE_CV2}  |  skipped(missing kp): {nkp_bad}  |  mean scene depth {np.mean(depths):.2f} m")
    print(f"  [A] 3D Kabsch residual  FK(angles) vs GT-3D  : {stats(A_res,'mm')}")
    print(f"      per-frame median                          : {stats(A_frame_med,'mm')}")
    print(f"  [B] reproj GT-3D -> 2D  vs GT-2D annotation   : {stats(B_res,'px')}")
    print(f"  [C] reproj FK(angles)+extr -> 2D vs GT-2D     : {stats(C_res,'px')}")

    # D: extrinsic stability per static (session,cam) group
    print(f"  [D] static-group extrinsic stability ({len(groups)} groups; base & camera should be CONSTANT per group):")
    base_stds = []; t_stds = []; rot_stds = []; sizes = []
    for g, items in groups.items():
        if len(items) < 5: continue
        base = np.array([it[0] for it in items])       # link0 cam pos per frame
        ts = np.array([it[2] for it in items])         # T_cam<-base translation per frame
        Rs = [it[1] for it in items]
        base_std = np.linalg.norm(base.std(0))*1000
        t_std = np.linalg.norm(ts.std(0))*1000
        Rmean = np.mean(Rs, 0)
        # nearest orthonormal
        U,_,Vt = np.linalg.svd(Rmean); Rm = U@Vt
        rot_dev = [math.degrees(math.acos(np.clip((np.trace(Rm.T@R)-1)/2,-1,1))) for R in Rs]
        base_stds.append(base_std); t_stds.append(t_std); rot_stds.append(np.mean(rot_dev)); sizes.append(len(items))
    base_stds=np.array(base_stds); t_stds=np.array(t_stds); rot_stds=np.array(rot_stds)
    print(f"      base(link0) cam-pos std per group [mm] : median {np.median(base_stds):.1f} | mean {base_stds.mean():.1f} | p90 {np.percentile(base_stds,90):.1f} | max {base_stds.max():.1f}")
    print(f"      cam translation std per group     [mm] : median {np.median(t_stds):.1f} | mean {t_stds.mean():.1f} | p90 {np.percentile(t_stds,90):.1f} | max {t_stds.max():.1f}")
    print(f"      cam rotation dev per group       [deg] : median {np.median(rot_stds):.2f} | mean {rot_stds.mean():.2f} | p90 {np.percentile(rot_stds,90):.2f} | max {rot_stds.max():.2f}")
    print(f"      group sizes: median {int(np.median(sizes))} frames, {len(sizes)} groups>=5")
    print("="*84)


if __name__ == '__main__':
    main()
