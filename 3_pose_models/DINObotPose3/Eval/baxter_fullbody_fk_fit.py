"""Fit Baxter WHOLE-BODY (17-keypoint) fixed joint transforms to DREAM baxter data.

Extends baxter_fk_fit.py (left arm, 7 kp) to the full DREAM baxter keypoint set:
  torso_t0, left_{s0,s1,e0,e1,w0,w1,w2}, left_hand, right_{s0,s1,e0,e1,w0,w1,w2}, right_hand
= torso (1) + left arm (7 joints + hand) + right arm (7 joints + hand) = 17.

Method (positions only; PnP/solver needs correct 3D POSITIONS, frame orientations are gauge):
  1. Fit each ARM standalone as a serial Rz chain with a trailing fixed HAND transform
     (8 keypoints per arm), 40-start scipy LM + per-frame Kabsch — exactly like the left arm.
     Adding the hand keypoint makes w2 OBSERVABLE (hand sits off the w2 roll axis).
  2. Assemble both chains + torso into ONE robot frame: freeze the left chain as the common
     gauge, recover the per-frame camera->robot transform from the left points, map the right +
     torso GT into that gauge, and Kabsch the right chain into place (deterministic).
  3. Joint LM polish over all params with per-frame Kabsch over all 17 points.

Validation: FK(gt angles) must reproduce all 17 GT `location`s to sub-mm on held-out + test.
"""
import json, glob, math, argparse, numpy as np
from scipy.optimize import least_squares

TRAIN_DIR = '/home/najo/NAS/DIP/datasets/synthetic/baxter_synth_train_dr'
TEST_DIR = '/home/najo/NAS/DIP/datasets/synthetic/baxter_synth_test_dr'

LKP = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2', 'left_hand']
RKP = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2', 'right_hand']
LJ = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']   # 7 joints
RJ = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']
ALL17 = ['torso_t0'] + LKP + RKP     # final output order

# ---- SE(3) helpers (identical math to baxter_fk_fit.py) ----

def make_T(xyz, rpy):
    rx, ry, rz = rpy
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    R = np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                  [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                  [-sy, cy*sx, cy*cx]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = xyz; return T

def Rz(t):
    c, s = math.cos(t), math.sin(t); T = np.eye(4); T[0, 0] = c; T[0, 1] = -s; T[1, 0] = s; T[1, 1] = c; return T

def kabsch_align(A, B):
    """Return A rigidly aligned onto B (both (N,3))."""
    ca, cb = A.mean(0), B.mean(0); H = (A-ca).T @ (B-cb); U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (A-ca) @ R.T + cb

def kabsch_Rt(A, B):
    """Rigid R,t with R@A+t ~= B (A,B (N,3))."""
    ca, cb = A.mean(0), B.mean(0); H = (A-ca).T @ (B-cb); U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cb - R @ ca; return R, t

# ---- data ----

def load(files, joint_names, kp_names):
    """-> TH (F, nJoint), GT (F, nKp, 3) meters."""
    TH, GT = [], []
    for f in files:
        d = json.load(open(f))
        kps = {k['name']: k for k in d['objects'][0]['keypoints']}
        js = {j['name'].split('/')[-1]: j.get('position', 0) for j in d.get('sim_state', {}).get('joints', [])}
        try:
            TH.append([js[n] for n in joint_names])
            GT.append(np.array([kps[n]['location'] for n in kp_names]) / 100.0)
        except KeyError:
            continue
    return np.array(TH), np.array(GT)

# ---- single-arm serial chain (7 Rz joints + trailing fixed hand => 8 keypoints) ----

def arm_fk_batch(params, TH):
    """params: 8*6 (7 joint transforms + 1 hand transform). TH: (F,7). -> (F,8,3)."""
    joints = [(params[i*6:i*6+3], params[i*6+3:i*6+6]) for i in range(8)]
    out = []
    for th in TH:
        cumul = np.eye(4); pts = []
        for i in range(7):
            rpy, xyz = joints[i]
            cumul = cumul @ make_T(xyz, rpy) @ Rz(th[i]); pts.append(cumul[:3, 3])
        rpy, xyz = joints[7]                       # hand: fixed transform, no joint
        pts.append((cumul @ make_T(xyz, rpy))[:3, 3])
        out.append(pts)
    return np.array(out)

def arm_residuals(params, TH, GT):
    FK = arm_fk_batch(params, TH)
    return np.concatenate([(kabsch_align(FK[k], GT[k]) - GT[k]).ravel() for k in range(len(TH))])

def fit_arm(TH, GT, THv, GTv, seed_left=None, n_start=40, tag=''):
    """40-start LM on one arm (8 kp). Optionally seed start 0 with known left params."""
    OFF = [0.279, 0.102, 0.271, 0.104, 0.271, 0.116, 0.10, 0.11]   # per-joint offset magnitudes
    rng = np.random.default_rng(0)
    best, best_rms = None, 1e9
    for trial in range(n_start):
        if trial == 0 and seed_left is not None:
            p0 = seed_left
        else:
            XYZ0 = [(0, 0, OFF[0])]
            for i in range(1, 8):
                ax = rng.integers(0, 3); v = [0, 0, 0]; v[ax] = OFF[i]; XYZ0.append(tuple(v))
            RPY0 = [(0, 0, 0)] + [tuple(rng.uniform(-math.pi, math.pi, 3)) for _ in range(7)]
            p0 = np.array([v for i in range(8) for v in (list(RPY0[i]) + list(XYZ0[i]))])
        try:
            s = least_squares(arm_residuals, p0, args=(TH, GT), method='lm', max_nfev=8000)
        except Exception:
            continue
        rms = np.sqrt((arm_residuals(s.x, THv, GTv)**2).mean()) * 1000
        if rms < best_rms:
            best_rms, best = rms, s
            print(f"  [{tag}] trial {trial:2d}: held-out RMS = {rms:.4f}mm (new best)")
        if best_rms < 0.02:
            break
    return best.x, best_rms

# ---- full-body FK (given left params, right params, B_R rigid, torso point) ----

def fullbody_fk(pL, pR, BR, torso, TH):
    """TH: (F,14) = [left7, right7]. -> (F,17,3) in order ALL17."""
    F = len(TH)
    fkL = arm_fk_batch(pL, TH[:, :7])          # (F,8,3) left gauge
    fkR = arm_fk_batch(pR, TH[:, 7:])          # (F,8,3) right gauge
    RR, tR = BR[:3, :3], BR[:3, 3]
    fkR = fkR @ RR.T + tR                        # place right chain into left gauge
    out = np.zeros((F, 17, 3))
    out[:, 0] = torso
    out[:, 1:9] = fkL
    out[:, 9:17] = fkR
    return out

def full_residuals(x, TH, GT):
    pL = x[:48]; pR = x[48:96]; br = x[96:102]; torso = x[102:105]
    BR = make_T(br[3:], br[:3])
    FK = fullbody_fk(pL, pR, BR, torso, TH)
    return np.concatenate([(kabsch_align(FK[k], GT[k]) - GT[k]).ravel() for k in range(len(TH))])

def per_kp_err(FK, GT):
    """mean/max per-keypoint error (mm) after per-frame Kabsch over all 17."""
    errs = np.zeros((len(GT), 17))
    for k in range(len(GT)):
        A = kabsch_align(FK[k], GT[k])
        errs[k] = np.linalg.norm(A - GT[k], axis=1)
    return errs * 1000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ntrain', type=int, default=250)
    ap.add_argument('--nheld', type=int, default=150)
    ap.add_argument('--ntest', type=int, default=400)
    ap.add_argument('--nstart', type=int, default=40)
    ap.add_argument('--polish', action='store_true', help='final joint LM over all 105 params')
    args = ap.parse_args()

    trainfs = sorted(glob.glob(TRAIN_DIR + '/*.json'))
    testfs = sorted(glob.glob(TEST_DIR + '/*.json'))
    ntot = args.ntrain + args.nheld
    TH14, GT17 = load(trainfs[:ntot], LJ + RJ, ALL17)
    THt, GTt = load(testfs[:args.ntest], LJ + RJ, ALL17)
    print(f"loaded train+held {len(TH14)}  test {len(THt)}")
    tr = slice(0, args.ntrain); hv = slice(args.ntrain, ntot)

    # left/right GT slices (8 kp each) and torso
    GT_L = GT17[:, 1:9]; GT_R = GT17[:, 9:17]; GT_T = GT17[:, 0]
    THL = TH14[:, :7]; THR = TH14[:, 7:]

    # seed left from the known 7-joint fit (extend with a hand-offset guess)
    seedL = None
    try:
        import sys, os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../TRAIN'))
        from model_v4 import _BAXTER_LEFT_JOINTS
        s = []
        for j in _BAXTER_LEFT_JOINTS:
            s += list(j['rpy']) + list(j['xyz'])
        s += [0, 0, 0, 0.0, 0.02, 0.08]           # hand transform init guess
        seedL = np.array(s)
        print("seeded left start 0 from _BAXTER_LEFT_JOINTS")
    except Exception as e:
        print("no left seed:", e)

    print("\n=== Fit LEFT arm (8 kp) ===")
    pL, rmsL = fit_arm(THL[tr], GT_L[tr], THL[hv], GT_L[hv], seed_left=seedL, n_start=args.nstart, tag='L')
    print(f"LEFT held-out RMS = {rmsL:.4f}mm")
    print("\n=== Fit RIGHT arm (8 kp) ===")
    pR, rmsR = fit_arm(THR[tr], GT_R[tr], THR[hv], GT_R[hv], seed_left=None, n_start=args.nstart, tag='R')
    print(f"RIGHT held-out RMS = {rmsR:.4f}mm")

    # ---- assemble: left gauge is the common frame; place right + torso ----
    fkL = arm_fk_batch(pL, THL[tr])                # (F,8,3) left gauge
    fkR = arm_fk_batch(pR, THR[tr])                # (F,8,3) right gauge
    # per-frame camera<-robot(left gauge): R@fkL+t ~= GT_L  => map GT into left gauge
    QR = []; QT = []; stackR = []
    for k in range(fkL.shape[0]):
        R, t = kabsch_Rt(fkL[k], GT_L[tr][k])      # left gauge -> camera
        QR.append((GT_R[tr][k] - t) @ R)           # camera -> left gauge : R^T (x - t) = (x-t)@R
        QT.append((GT_T[tr][k] - t) @ R)
        stackR.append(fkR[k])
    QR = np.concatenate(QR); stackR = np.concatenate(stackR)
    RR, tR = kabsch_Rt(stackR, QR)                 # right gauge -> left gauge
    BR = np.eye(4); BR[:3, :3] = RR; BR[:3, 3] = tR
    torso = np.mean(QT, axis=0)
    print(f"\nassembled: B_R placement residual = "
          f"{np.sqrt(((stackR @ RR.T + tR - QR)**2).sum(1)).mean()*1000:.4f}mm  "
          f"torso spread = {np.std(QT, axis=0)*1000} mm")

    x = np.concatenate([pL, pR, [0, 0, 0], tR, torso])   # BR rpy init 0 -> replaced below
    # store BR as rpy/xyz: recover rpy from RR
    from scipy.spatial.transform import Rotation as Rsc
    br_rpy = Rsc.from_matrix(RR).as_euler('xyz')
    x[96:99] = br_rpy; x[99:102] = tR; x[102:105] = torso

    def report(x, TH, GT, label):
        pL_, pR_, br_, torso_ = x[:48], x[48:96], x[96:102], x[102:105]
        BR_ = make_T(br_[3:], br_[:3])
        FK = fullbody_fk(pL_, pR_, BR_, torso_, TH)
        e = per_kp_err(FK, GT)
        print(f"\n--- {label} ({len(GT)} frames) ---  overall RMS {np.sqrt((e**2).mean()):.4f}mm")
        for i, n in enumerate(ALL17):
            print(f"  {n:10s}: mean={e[:,i].mean():7.4f}mm  max={e[:,i].max():8.3f}mm")
        return e

    report(x, TH14[hv], GT17[hv], 'ASSEMBLED held-out')

    if args.polish:
        print("\n=== joint LM polish (105 params) ===")
        s = least_squares(full_residuals, x, args=(TH14[tr], GT17[tr]), method='lm', max_nfev=4000)
        x = s.x
        report(x, TH14[hv], GT17[hv], 'POLISHED held-out')

    e_test = report(x, THt, GTt, 'TEST')

    # w2 observability check: perturb left_w2 / right_w2, does the hand move?
    THp = THt.copy(); THp[:, 6] += 0.5; THp[:, 13] += 0.5
    pL_, pR_, br_, torso_ = x[:48], x[48:96], x[96:102], x[102:105]
    BR_ = make_T(br_[3:], br_[:3])
    fk0 = fullbody_fk(pL_, pR_, BR_, torso_, THt)
    fk1 = fullbody_fk(pL_, pR_, BR_, torso_, THp)
    dlh = np.linalg.norm(fk1[:, 8] - fk0[:, 8], axis=1).mean() * 1000    # left_hand
    drh = np.linalg.norm(fk1[:, 16] - fk0[:, 16], axis=1).mean() * 1000  # right_hand
    print(f"\n[w2 observability] +0.5rad w2 moves left_hand {dlh:.1f}mm, right_hand {drh:.1f}mm "
          f"(>0 => w2 observable via hand)")

    # emit transforms
    print("\n\n# ==== paste into model_v4.py ====")
    def emit(name, p, n):
        print(f"{name} = [")
        for i in range(n):
            rpy = p[i*6:i*6+3]; xyz = p[i*6+3:i*6+6]
            print(f"    {{'xyz': ({xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}), "
                  f"'rpy': ({rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f})}},")
        print("]")
    emit("_BAXTER_FB_LEFT", x[:48], 8)
    emit("_BAXTER_FB_RIGHT", x[48:96], 8)
    br_ = x[96:102]; torso_ = x[102:105]
    print(f"_BAXTER_FB_BR = {{'xyz': ({br_[3]:.6f}, {br_[4]:.6f}, {br_[5]:.6f}), "
          f"'rpy': ({br_[0]:.6f}, {br_[1]:.6f}, {br_[2]:.6f})}}")
    print(f"_BAXTER_FB_TORSO = ({torso_[0]:.6f}, {torso_[1]:.6f}, {torso_[2]:.6f})")


if __name__ == '__main__':
    main()
