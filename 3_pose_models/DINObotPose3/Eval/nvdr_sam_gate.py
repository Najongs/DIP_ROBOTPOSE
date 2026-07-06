"""
Phase-2/3 gate: does the EXACT-mesh nvdiffrast render close the shape gap to SAM's true robot mask?

Per frame: detect keypoints (deployment prompts) -> SAM box+point mask; render the GT-pose visual-mesh
silhouette with nvdiffrast; report IoU(SAM, render). The splat renderer capped this at ~0.36 (visual)
and render-compare diverged against SAM. Gate to proceed to render-and-compare with SAM targets: >= ~0.7.

Saves side-by-side viz (real | SAM mask | render mask) for the worst/best frames.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
from model_angle import AnglePredictor
from model_v4 import panda_forward_kinematics
from inference_4tier_eval import EvalDataset
from refine_eval import scale_K
from silhouette_mesh_probe import kabsch_batch, KPN, all_link_transforms
from render_nvdr import NVDRSilhouette

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--sam-checkpoint', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-frames', type=int, default=100)
    ap.add_argument('--render-h', type=int, default=512, help='IoU is measured at this res')
    ap.add_argument('--kind', default='visual')
    ap.add_argument('--out', default='../ViS/nvdr_sam_gate')
    args = ap.parse_args()
    device = torch.device('cuda'); S = args.image_size; H = args.render_h

    m = AnglePredictor(args.model_name, S, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))

    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry['vit_b'](checkpoint=args.sam_checkpoint).to(device); sam.eval()
    sam_pred = SamPredictor(sam)

    rdr = NVDRSilhouette(device, kind=args.kind)
    MEAN = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    STD = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    ds = EvalDataset(args.val_dir, KPN, image_size=(S, S))
    if args.max_frames and args.max_frames < len(ds.json_files):
        st = max(1, len(ds.json_files) // args.max_frames); ds.json_files = ds.json_files[::st][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    ious, keep = [], []
    for batch in tqdm(loader, desc='nvdr-sam-gate'):
        img = batch['image'].to(device)
        K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        gt3d = batch['gt_3d'].to(device); ga = batch['gt_angles'].to(device).clone(); ga[:, 6] = 0.0
        with torch.no_grad():
            out = m(img, K)
            kp2d = out['keypoints_2d']; conf = out['confidence']
            fkg = panda_forward_kinematics(ga); Rg, tg = kabsch_batch(fkg, gt3d)
            pts = rdr.robot_verts(ga, all_link_transforms)
            rmask = (rdr(pts, Rg, tg, K, H, S) > 0.5).float()               # (B,H,H)
        u8 = ((img * STD + MEAN).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        for b in range(img.shape[0]):
            sam_pred.set_image(u8[b])
            pts2 = kp2d[b].detach().cpu().numpy(); cf = conf[b].detach().cpu().numpy()
            sel = cf > 0.3
            if sel.sum() < 2:
                sel = cf >= np.sort(cf)[-2]
            p = pts2[sel]
            x0, y0 = p[:, 0].min(), p[:, 1].min(); x1, y1 = p[:, 0].max(), p[:, 1].max()
            mx = 0.15 * max(x1 - x0, y1 - y0)
            box = np.array([max(0, x0 - mx), max(0, y0 - mx), min(S, x1 + mx), min(S, y1 + mx)])
            mm, sc, _ = sam_pred.predict(point_coords=p, point_labels=np.ones(len(p)), box=box, multimask_output=True)
            smask = torch.from_numpy(mm[int(np.argmax(sc))].astype('float32')).to(device)
            smask = (F.interpolate(smask[None, None], size=(H, H), mode='bilinear', align_corners=False)[0, 0] > 0.5).float()
            inter = (smask * rmask[b]).sum(); union = ((smask + rmask[b]) > 0).float().sum().clamp(min=1)
            ious.append(float(inter / union))
            keep.append((u8[b], smask.cpu().numpy(), rmask[b].cpu().numpy()))

    ious = np.array(ious)
    print(f"\n=== NVDR({args.kind}) GT-pose render vs SAM mask  ({os.path.basename(args.val_dir)}, n={len(ious)}) ===")
    print(f"  IoU mean {ious.mean():.3f}  median {np.median(ious):.3f}  frac>=0.7: {(ious>=0.7).mean():.2f}  frac<0.4: {(ious<0.4).mean():.2f}")

    import cv2
    os.makedirs(args.out, exist_ok=True)
    order = np.argsort(ious)
    for tag, idxs in [('worst', order[:4]), ('best', order[-4:])]:
        for i in idxs:
            rgb, sm, rm = keep[i]
            sm5 = cv2.resize(sm, (S, S), interpolation=cv2.INTER_NEAREST)
            rm5 = cv2.resize(rm, (S, S), interpolation=cv2.INTER_NEAREST)
            o = rgb.copy()
            o[..., 2] = np.maximum(o[..., 2], (sm5 * 120).astype(np.uint8))   # SAM = red tint
            o[..., 1] = np.maximum(o[..., 1], (rm5 * 120).astype(np.uint8))   # render = green tint
            cv2.imwrite(os.path.join(args.out, f'{tag}_iou{ious[i]:.2f}_{i:03d}.png'), o[..., ::-1])
    print(f"viz -> {args.out}/ (red=SAM, green=render, yellow=agreement)")


if __name__ == '__main__':
    main()
