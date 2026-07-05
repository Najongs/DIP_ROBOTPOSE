"""Meca500 mesh<->kinematics alignment check.
Build FK from the URDF chain, place per-link meshes, project onto a real image at
its GT joint angles + camera pose (Kabsch from GT 3D keypoints). Saves an overlay.
"""
import json, glob, os, numpy as np, cv2, trimesh, math

HERE = os.path.dirname(__file__)
MESH = os.path.join(HERE, 'meshes/visual')
# URDF chain (meca500r3.urdf): (xyz, axis) per joint, all rpy=0, mesh origin=identity
JOINTS = [((0.012498,0,0.091),'z'), ((0,0,0.044),'y'), ((0,0,0.135),'y'),
          ((0,0,0.038),'x'), ((0.12,0,0),'y'), ((0.07,0,0),'x')]
MESHES = ['meca_500_r3_base.dae','meca_500_r3_j1.dae','meca_500_r3_j2.dae',
          'meca_500_r3_j3.dae','meca_500_r3_j4.dae','meca_500_r3_j5.dae','meca_500_r3_j6.dae']

def rot(axis, th):
    c,s=math.cos(th),math.sin(th); R=np.eye(4)
    if axis=='z': R[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
    elif axis=='y': R[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
    else: R[:3,:3]=[[1,0,0],[0,c,-s],[0,s,c]]
    return R
def trans(xyz):
    T=np.eye(4); T[:3,3]=xyz; return T

def link_transforms(theta):
    """Return 7 link world transforms (base + 6). theta: (6,) rad."""
    Ts=[np.eye(4)]  # base
    T=np.eye(4)
    for i,((xyz,ax)) in enumerate(JOINTS):
        T = T @ trans(xyz) @ rot(ax, theta[i])
        Ts.append(T.copy())
    return Ts  # len 7

def main():
    base='/home/najo/NAS/DIP/datasets/ICRA_multiview/Converted_dataset/Meca500_to_DREAM'
    f=sorted(glob.glob(f'{base}/*.json'))[0]
    d=json.load(open(f)); o=d['objects'][0]; m=d['meta']
    img_path=os.path.normpath(os.path.join(base, m['image_path'].replace('../dataset/','../../',1)))
    theta=np.array([j['position'] for j in d['sim_state']['joints']],dtype=np.float64)  # (6,)
    gt3d=np.array([k['location'] for k in o['keypoints']],dtype=np.float64)  # (7,3) cam frame
    gt2d=np.array([k['projected_location'] for k in o['keypoints']],dtype=np.float64)
    K=np.array(m['K'],dtype=np.float64)

    Ts=link_transforms(theta)
    # camera pose from the DH FK that GENERATED the GT (kinematics.py), NOT from URDF keypoints.
    # DH and URDF share the same base frame; URDF meshes then render correctly in that pose.
    import sys as _s; _s.path.insert(0,'/home/najo/NAS/DIP/4_perception/DINOv3_fine_tunning')
    from kinematics import Meca500Kinematics
    kp_dh=Meca500Kinematics().forward_kinematics(theta)   # (7,3) DH keypoints, base frame
    Pc=kp_dh-kp_dh.mean(0); Qc=gt3d-gt3d.mean(0)
    Hm=Pc.T@Qc; U,S,Vt=np.linalg.svd(Hm); dsign=np.sign(np.linalg.det(Vt.T@U.T))
    R=Vt.T@np.diag([1,1,dsign])@U.T
    t=gt3d.mean(0)-R@kp_dh.mean(0)
    resid=np.linalg.norm((R@kp_dh.T).T+t-gt3d,axis=1)
    print(f'DH-FK Kabsch residual per-kp (mm): {np.round(resid*1000,1).tolist()}  mean={resid.mean()*1000:.1f}  (low => camera pose exact)')
    kp_urdf=np.array([T[:3,3] for T in Ts])

    img=cv2.imread(img_path)
    H_img,W_img=img.shape[:2]
    print(f'image {W_img}x{H_img}  {os.path.basename(img_path)}')
    # project mesh vertices
    colors=[(0,255,0),(0,200,255),(255,0,255),(255,200,0),(0,255,255),(200,0,255),(255,255,0)]
    for i,mf in enumerate(MESHES):
        mesh=trimesh.load(os.path.join(MESH,mf),force='mesh')
        v=np.asarray(mesh.vertices)[::15]  # subsample
        vh=(Ts[i][:3,:3]@v.T).T+Ts[i][:3,3]        # base frame
        vc=(R@vh.T).T+t                             # camera frame
        uv=(K@vc.T).T; uv=uv[:,:2]/uv[:,2:3]
        for p in uv:
            x,y=int(p[0]),int(p[1])
            if 0<=x<W_img and 0<=y<H_img: cv2.circle(img,(x,y),1,colors[i],-1)
    # GT 2D keypoints (white), projected DH-FK keypoints (red) — should coincide
    kp_cam=(R@kp_dh.T).T+t; kpp=(K@kp_cam.T).T; kpp=kpp[:,:2]/kpp[:,2:3]
    for p in gt2d: cv2.circle(img,(int(p[0]),int(p[1])),6,(255,255,255),2)
    for p in kpp:  cv2.circle(img,(int(p[0]),int(p[1])),4,(0,0,255),-1)
    out=os.path.join(HERE,'align_check.png'); cv2.imwrite(out,img)
    print('saved',out)
    print(f'proj DH-FK vs GT-2D kp err (px): mean={np.linalg.norm(kpp-gt2d,axis=1).mean():.1f}')

if __name__=='__main__': main()
