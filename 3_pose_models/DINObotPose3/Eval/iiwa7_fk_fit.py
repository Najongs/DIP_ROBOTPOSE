"""Fit iiwa7 fixed joint transforms to DREAM kuka data (positions only).
Solver needs correct link 3D POSITIONS for PnP; intermediate frame orientations are
unobservable/irrelevant. Optimize rpy+xyz per joint to minimize Kabsch residual."""
import json, glob, math, numpy as np
from scipy.optimize import least_squares

FS = sorted(glob.glob('/home/najo/NAS/DIP/datasets/synthetic/kuka_synth_train_dr/*.json'))

def load(nframes):
    TH, GT = [], []
    for f in FS[:nframes]:
        d=json.load(open(f))
        kps={k['name']:k for k in d['objects'][0]['keypoints']}
        js={j['name'].split('/')[-1]:j.get('position',0) for j in d.get('sim_state',{}).get('joints',[])}
        try:
            TH.append([js[f'iiwa7_joint_{i}'] for i in range(1,8)])
            GT.append(np.array([kps[f'iiwa7_link_{i}']['location'] for i in range(1,8)])/100.0)
        except KeyError: continue
    return np.array(TH), np.array(GT)

def make_T(xyz, rpy):
    rx,ry,rz=rpy
    cx,sx,cy,sy,cz,sz=math.cos(rx),math.sin(rx),math.cos(ry),math.sin(ry),math.cos(rz),math.sin(rz)
    R=np.array([[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx],[sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx],[-sy,cy*sx,cy*cx]])
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=xyz; return T

def Rz(t):
    c,s=math.cos(t),math.sin(t); T=np.eye(4); T[0,0]=c;T[0,1]=-s;T[1,0]=s;T[1,1]=c; return T

def fk_batch(params, TH):   # params: 7*(rpy3+xyz3)=42 ; TH (F,7) -> (F,7,3)
    joints=[(params[i*6:i*6+3], params[i*6+3:i*6+6]) for i in range(7)]
    out=[]
    for th in TH:
        cumul=np.eye(4); pts=[]
        for i in range(7):
            rpy,xyz=joints[i]
            cumul=cumul@make_T(xyz,rpy)@Rz(th[i]); pts.append(cumul[:3,3])
        out.append(pts)
    return np.array(out)

def kabsch_align(A,B):
    ca,cb=A.mean(0),B.mean(0); H=(A-ca).T@(B-cb); U,S,Vt=np.linalg.svd(H)
    d=np.sign(np.linalg.det(Vt.T@U.T)); R=Vt.T@np.diag([1,1,d])@U.T
    return (A-ca)@R.T+cb

def residuals(params, TH, GT):
    FK=fk_batch(params, TH); r=[]
    for k in range(len(TH)):
        r.append((kabsch_align(FK[k],GT[k])-GT[k]).ravel())
    return np.concatenate(r)

# init: measured offsets + iiwa7 std rpy
OFF=[0.15,0.19,0.21,0.19,0.21,0.1995,0.1012]
RPY0=[(0,0,0),(math.pi/2,0,math.pi),(math.pi/2,0,math.pi),(math.pi/2,0,0),(-math.pi/2,math.pi,0),(math.pi/2,0,0),(-math.pi/2,math.pi,0)]
XYZ0=[(0,0,OFF[0]),(0,0,OFF[1]),(0,OFF[2],0),(0,0,OFF[3]),(0,OFF[4],0),(0,0,OFF[5]),(0,OFF[6],0)]
p0=np.array([v for i in range(7) for v in (list(RPY0[i])+list(XYZ0[i]))])

# Fix J1 to physical base (0,0,0.15)/rpy0 so base frame matches URDF meshes; fit J2..J7.
FIXED_J1 = np.array([0,0,0, 0,0,0.15])
def residuals_fixed(pv, TH, GT):
    return residuals(np.concatenate([FIXED_J1, pv]), TH, GT)

TH,GT=load(120)
print(f"fit on {len(TH)} frames (J1 fixed to base, {len(p0)-6} free params)")
r0=residuals(p0,TH,GT); print(f"init RMS = {np.sqrt((r0**2).mean())*1000:.2f}mm")
sol_=least_squares(residuals_fixed, p0[6:], args=(TH,GT), method='lm', max_nfev=6000)
sol=type('S',(),{'x':np.concatenate([FIXED_J1, sol_.x])})()
rf=residuals(sol.x,TH,GT); print(f"fitted RMS = {np.sqrt((rf**2).mean())*1000:.3f}mm")

# validate on held-out train frames AND the separate test_dr set
THv,GTv=load(400); THv,GTv=THv[300:],GTv[300:]
rv=residuals(sol.x,THv,GTv); print(f"held-out train(100) RMS = {np.sqrt((rv**2).mean())*1000:.3f}mm")
# test set
FS_bak=FS[:]
globals()['FS']=sorted(glob.glob('/home/najo/NAS/DIP/datasets/synthetic/kuka_synth_test_dr/*.json'))
THt,GTt=load(300)
rt=residuals(sol.x,THt,GTt); print(f"TEST set({len(THt)}) RMS = {np.sqrt((rt**2).mean())*1000:.3f}mm")
globals()['FS']=FS_bak
FKv=fk_batch(sol.x,THv)
for i in range(7):
    e=np.array([np.linalg.norm(kabsch_align(FKv[k],GTv[k])[i]-GTv[k][i]) for k in range(len(THv))])
    print(f"  link_{i+1}: mean={e.mean()*1000:6.2f}mm max={e.max()*1000:6.2f}mm")

print("\nfitted params (rpy_rad + xyz_m per joint):")
for i in range(7):
    rpy=sol.x[i*6:i*6+3]; xyz=sol.x[i*6+3:i*6+6]
    print(f"  J{i+1}: rpy=({rpy[0]:+.4f},{rpy[1]:+.4f},{rpy[2]:+.4f})  xyz=({xyz[0]:+.4f},{xyz[1]:+.4f},{xyz[2]:+.4f})")
