"""Baxter render-and-compare DEPTH correction: does optimizing the rendered silhouette to match
the SAM mask improve pose (ADD) over the rot-head's direct t?

Pipeline per frame: heads -> (angles, R, t) [direct-pose]; SAM mask of the arm (prompted by
FK-projected keypoints); render baxter silhouette; optimize t (Adam) to max soft-IoU(render, mask);
ADD(kp_cam, gt3d) before vs after. rot-head R,t are full-camera-frame (crop-independent), so we
render/segment on the full padded frame with the dataset K.
"""
import argparse, os, sys, json, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from PIL import Image

HERE = os.path.dirname(__file__); TRAIN = os.path.abspath(os.path.join(HERE, '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(HERE)
from model_angle import AnglePredictor
from model_v4 import baxter_left_forward_kinematics as FK
from dataset import PoseEstimationDataset
from refine_eval import scale_K, add_auc
from baxter_render import make_baxter_renderer, baxter_all_link_transforms
from kuka_add_eval import kabsch_batch
from segment_anything import sam_model_registry, SamPredictor

KP = ['left_s0','left_s1','left_e0','left_e1','left_w0','left_w1','left_w2']


def soft_iou(r, m):
    inter = (r * m).sum(); union = (r + m - r * m).sum().clamp(min=1e-6)
    return inter / union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--angle-head', required=True)
    ap.add_argument('--rot-head', required=True); ap.add_argument('--val-dir', required=True)
    ap.add_argument('--sam-checkpoint', default='../weights_sam/sam_vit_b_01ec64.pth')
    ap.add_argument('--max-frames', type=int, default=150); ap.add_argument('--rc-iters', type=int, default=60)
    ap.add_argument('--image-size', type=int, default=512)
    args = ap.parse_args()
    dev = 'cuda'; IS = args.image_size

    m = AnglePredictor('facebook/dinov3-vitb16-pretrain-lvd1689m', IS, fix_joint7_zero=True, head_type='mlp',
                       with_rotation=True, with_translation=True).to(dev).eval()
    sd = torch.load(args.detector, map_location=dev); sd = {k.replace('module.',''):v for k,v in sd.items()}
    m.load_state_dict({k:v for k,v in sd.items() if k in m.state_dict() and v.shape==m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.angle_head, map_location=dev))
    m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=dev))

    rdr = make_baxter_renderer(dev)
    sam = sam_model_registry['vit_b'](checkpoint=os.path.join(HERE, args.sam_checkpoint)).to(dev).eval()
    sam_pred = SamPredictor(sam)

    cs = json.load(open(os.path.join(args.val_dir, '_camera_settings.json')))['camera_settings'][0]['intrinsic_settings']
    W0, H0 = cs['resolution']['width'], cs['resolution']['height']; PAD = (W0 - H0)//2; S = W0
    Kfull = torch.tensor([[cs['fx'],0,cs['cx']],[0,cs['fy'],cs['cy']+PAD],[0,0,1]], dtype=torch.float32, device=dev)

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=KP, image_size=(IS,IS), heatmap_size=(IS,IS),
                               augment=False, include_angles=True, sigma=2.5, crop_to_robot=True,
                               crop_margin=1.5, angle_joint_names=KP)
    idx = list(range(0, len(ds), max(1, len(ds)//args.max_frames)))[:args.max_frames]

    add_before, add_after = [], []
    for n, i in enumerate(idx):
        b = ds[i]
        img = b['image'].unsqueeze(0).to(dev)
        Kcrop = scale_K(b['camera_K'].unsqueeze(0), [b['original_size']], IS).to(dev)
        gt3d = b['keypoints_3d'].unsqueeze(0).to(dev)
        with torch.no_grad():
            o = m(img, Kcrop)
        ang = o['joint_angles']; R = o['rot_matrix'].float(); t0 = o['trans'].float()
        fk = FK(ang.double()).float()                                     # (1,7,3)

        # full-frame RGB for SAM (from annotation path)
        rgb_path = b['annotation_path'].replace('.json', '.rgb.jpg')
        raw = np.array(Image.open(rgb_path).convert('RGB'))
        imgp = np.zeros((S, S, 3), np.uint8); imgp[PAD:PAD+H0] = raw

        # FK-projected keypoints (full frame) as SAM prompts
        def proj(tt):
            cam = torch.einsum('bij,bnj->bni', R, fk) + tt.unsqueeze(1)
            z = cam[...,2].clamp(min=1e-3)
            u = cam[...,0]/z*Kfull[0,0] + Kfull[0,2]; v = cam[...,1]/z*Kfull[1,1] + Kfull[1,2]
            return torch.stack([u,v],-1)
        kp2d = proj(t0)[0].detach().cpu().numpy()
        inb = (kp2d[:,0]>0)&(kp2d[:,0]<S)&(kp2d[:,1]>0)&(kp2d[:,1]<S)
        if inb.sum() < 3:
            continue
        sam_pred.set_image(imgp)
        masks, scores, _ = sam_pred.predict(point_coords=kp2d[inb].astype(np.float32),
                                            point_labels=np.ones(int(inb.sum())), multimask_output=True)
        mask_t = torch.tensor(masks[int(np.argmax(scores))].astype(np.float32), device=dev)   # (S,S)

        # ADD before
        kp_cam0 = torch.einsum('bij,bnj->bni', R, fk) + t0.unsqueeze(1)
        valid = (gt3d.abs().sum(-1) > 0)
        add_before.append(float((kp_cam0-gt3d).norm(dim=-1)[valid].mean()))

        # RC: articulated — optimize t AND joint angles to max soft-IoU (t-only can't fix the
        # wrong silhouette SHAPE from Baxter's wrist-angle error). angles kept near the head prior.
        t = t0.clone().detach().requires_grad_(True)
        da = torch.zeros_like(ang, requires_grad=True)          # angle delta (rad)
        opt = torch.optim.Adam([{'params':[t],'lr':3e-3},{'params':[da],'lr':1e-2}])
        for _ in range(args.rc_iters):
            a = ang.detach() + da
            pts = rdr.robot_verts(a, baxter_all_link_transforms)
            r = rdr(pts, R, t, Kfull.unsqueeze(0), S, S)[0]
            loss = 1 - soft_iou(r, mask_t) + 0.5 * (da**2).mean()   # prior: stay near head angles
            opt.zero_grad(); loss.backward(); opt.step()
        t = t.detach(); a = (ang.detach() + da.detach())
        fk1 = FK(a.double()).float()
        kp_cam1 = torch.einsum('bij,bnj->bni', R, fk1) + t.unsqueeze(1)
        add_after.append(float((kp_cam1-gt3d).norm(dim=-1)[valid].mean()))
        if (n+1) % 25 == 0:
            print(f'  {n+1}/{len(idx)}  ADD before {np.mean(add_before)*1000:.1f}mm  after {np.mean(add_after)*1000:.1f}mm')

    a0, a1 = np.array(add_before), np.array(add_after)
    print(f"\n{'='*56}\n  Baxter RC depth-correction ({len(a0)} frames)\n{'='*56}")
    print(f"  BEFORE (direct-pose): ADD-AUC {add_auc(a0):.4f} | mean {a0.mean()*1000:.1f}mm | median {np.median(a0)*1000:.1f}mm")
    print(f"  AFTER  (RC silhouette): ADD-AUC {add_auc(a1):.4f} | mean {a1.mean()*1000:.1f}mm | median {np.median(a1)*1000:.1f}mm")
    print(f"  delta AUC {add_auc(a1)-add_auc(a0):+.4f} | mean {(a1.mean()-a0.mean())*1000:+.1f}mm")
    print('='*56)


if __name__ == '__main__':
    main()
