"""Fit Baxter left-arm fixed joint transforms to DREAM baxter data (positions only).
Same method as iiwa7_fk_fit.py: solver needs correct link 3D POSITIONS for PnP; intermediate
frame orientations are unobservable/irrelevant. Optimize rpy+xyz per joint (Kabsch residual).
Keypoints = joints = left_{s0,s1,e0,e1,w0,w1,w2}. Last joint (w2) does not move its own keypoint
-> unobservable from keypoints (head predicts s0..w1, fixes w2=0), same as KUKA joint_7."""
import json, glob, math, numpy as np
from scipy.optimize import least_squares

FS = sorted(glob.glob('/home/najo/NAS/DIP/datasets/synthetic/baxter_synth_train_dr/*.json'))
KP = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']

def load(nframes, files=None):
    files = files or FS[:nframes]
    TH, GT = [], []
    for f in files:
        d = json.load(open(f))
        kps = {k['name']: k for k in d['objects'][0]['keypoints']}
        js = {j['name'].split('/')[-1]: j.get('position', 0) for j in d.get('sim_state', {}).get('joints', [])}
        try:
            TH.append([js[n] for n in KP])                          # joint names == keypoint names
            GT.append(np.array([kps[n]['location'] for n in KP]) / 100.0)  # cm -> m
        except KeyError:
            continue
    return np.array(TH), np.array(GT)

def make_T(xyz, rpy):
    rx, ry, rz = rpy
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    R = np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx], [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = xyz; return T

def Rz(t):
    c, s = math.cos(t), math.sin(t); T = np.eye(4); T[0, 0] = c; T[0, 1] = -s; T[1, 0] = s; T[1, 1] = c; return T

def fk_batch(params, TH):
    joints = [(params[i*6:i*6+3], params[i*6+3:i*6+6]) for i in range(7)]
    out = []
    for th in TH:
        cumul = np.eye(4); pts = []
        for i in range(7):
            rpy, xyz = joints[i]; cumul = cumul @ make_T(xyz, rpy) @ Rz(th[i]); pts.append(cumul[:3, 3])
        out.append(pts)
    return np.array(out)

def kabsch_align(A, B):
    ca, cb = A.mean(0), B.mean(0); H = (A-ca).T @ (B-cb); U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (A-ca) @ R.T + cb

def residuals(params, TH, GT):
    FK = fk_batch(params, TH)
    return np.concatenate([(kabsch_align(FK[k], GT[k]) - GT[k]).ravel() for k in range(len(TH))])

# Multi-start: serial-chain FK is prone to rpy local minima (Baxter geometry unknown a priori).
# Try many random rpy inits (xyz init from measured offset magnitudes), keep best on held-out.
OFF = [0.279, 0.102, 0.271, 0.104, 0.271, 0.116, 0.10]
TH, GT = load(300)
THv, GTv = load(500); THv, GTv = THv[300:], GTv[300:]   # held-out 200
rng = np.random.default_rng(0)
best = None; best_rms = 1e9
for trial in range(40):
    XYZ0 = [(0, 0, OFF[0])]
    for i in range(1, 7):
        ax = rng.integers(0, 3); v = [0, 0, 0]; v[ax] = OFF[i]; XYZ0.append(tuple(v))
    RPY0 = [(0, 0, 0)] + [tuple(rng.uniform(-math.pi, math.pi, 3)) for _ in range(6)]
    p0 = np.array([v for i in range(7) for v in (list(RPY0[i]) + list(XYZ0[i]))])
    try:
        s = least_squares(residuals, p0, args=(TH, GT), method='lm', max_nfev=6000)
    except Exception:
        continue
    rms = np.sqrt((residuals(s.x, THv, GTv)**2).mean()) * 1000
    if rms < best_rms:
        best_rms = rms; best = s
        print(f"  trial {trial:2d}: held-out RMS = {rms:.3f}mm  (new best)")
    if best_rms < 0.05:
        break
sol = best
print(f"fit on {len(TH)} frames (40-start). best held-out RMS = {best_rms:.3f}mm")
print(f"train RMS = {np.sqrt((residuals(sol.x, TH, GT)**2).mean())*1000:.3f}mm")
THt, GTt = load(300, files=sorted(glob.glob('/home/najo/NAS/DIP/datasets/synthetic/baxter_synth_test_dr/*.json'))[:300])
print(f"TEST set({len(THt)}) RMS = {np.sqrt((residuals(sol.x, THt, GTt)**2).mean())*1000:.3f}mm")
FKt = fk_batch(sol.x, THt)
for i in range(7):
    e = np.array([np.linalg.norm(kabsch_align(FKt[k], GTt[k])[i] - GTt[k][i]) for k in range(len(THt))])
    print(f"  {KP[i]:9s}: mean={e.mean()*1000:6.2f}mm max={e.max()*1000:6.2f}mm")
print("\n_BAXTER_LEFT_JOINTS = [")
for i in range(7):
    rpy = sol.x[i*6:i*6+3]; xyz = sol.x[i*6+3:i*6+6]
    print(f"    {{'xyz': ({xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}), 'rpy': ({rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f})}},")
print("]")
