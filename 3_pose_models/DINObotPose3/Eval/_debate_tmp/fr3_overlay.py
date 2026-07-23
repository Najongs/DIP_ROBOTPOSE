"""Overlay GT-2D keypoints (and FK-reprojected) on real FR3 images to validate GT vs reality."""
import os, sys, glob, json, re
import numpy as np, torch, cv2
sys.path.append('/home/najo/NAS/DIP/3_pose_models/DINObotPose3/TRAIN')
from model_v4 import panda_forward_kinematics
KP=['panda_link0','panda_link2','panda_link3','panda_link4','panda_link6','panda_link7','panda_hand']
CD='/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset/fr3_val'
files=sorted(glob.glob(CD+'/*.json'))
# pick frames from distinct (session,view,cam) groups
seen={}; picks=[]
for f in files:
    d=json.load(open(f)); ip=d['meta']['image_path']
    m=re.search(r'(Panda_dataset[^/]*)',ip)
    key=(m.group(1),d['meta']['view'],d['meta']['cam'])
    if key not in seen:
        seen[key]=1; picks.append(f)
    if len(picks)>=4: break
out=[]
for i,f in enumerate(picks):
    d=json.load(open(f)); o=d['objects'][0]
    kp={k['name']:k for k in o['keypoints']}
    gt2d=np.array([kp[n]['projected_location'] for n in KP],float)
    gt3d=np.array([kp[n]['location'] for n in KP],float)
    ang=np.array([d['sim_state']['joints'][j]['position'] for j in range(7)],float)
    fk=panda_forward_kinematics(torch.tensor(ang,dtype=torch.float64).unsqueeze(0))[0].numpy()
    # recover extrinsic via kabsch, reproject fk
    ca,cb=fk.mean(0),gt3d.mean(0); H=(fk-ca).T@(gt3d-cb)
    U,S,Vt=np.linalg.svd(H); Dd=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]); R=Vt.T@Dd@U.T; t=cb-R@ca
    fkc=fk@R.T+t
    K=np.array(d['meta']['K'],float); dist=np.array(d['meta'].get('dist_coeffs',[0,0,0,0,0]),float)
    fk2d,_=cv2.projectPoints(fkc.reshape(-1,1,3),np.zeros(3),np.zeros(3),K,dist); fk2d=fk2d.reshape(-1,2)
    ipath=os.path.normpath(os.path.join(CD,d['meta']['image_path']))
    img=cv2.imread(ipath)
    if img is None:
        print('MISS image',ipath); continue
    for j,(x,y) in enumerate(gt2d):
        cv2.circle(img,(int(x),int(y)),9,(0,255,0),2)                 # GT 2D = green
        cv2.putText(img,KP[j].replace('panda_',''),(int(x)+8,int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    for (x,y) in fk2d:
        cv2.circle(img,(int(x),int(y)),4,(0,0,255),-1)                # FK reproj = red dot
    # crop around robot for visibility
    xs=gt2d[:,0]; ys=gt2d[:,1]
    x0=max(0,int(xs.min()-120)); x1=min(img.shape[1],int(xs.max()+120))
    y0=max(0,int(ys.min()-120)); y1=min(img.shape[0],int(ys.max()+120))
    crop=img[y0:y1,x0:x1]
    op=f'/home/najo/NAS/DIP/3_pose_models/DINObotPose3/Eval/_debate_tmp/overlay_{i}.png'
    cv2.imwrite(op,crop)
    reproj_err=np.linalg.norm(fk2d-gt2d,axis=1).mean()
    print(f'{op}  {os.path.basename(ipath)}  depth={gt3d[:,2].mean():.2f}m  FK-vs-GT2D={reproj_err:.3f}px  size={crop.shape}')
    out.append(op)
print('DONE',len(out))
