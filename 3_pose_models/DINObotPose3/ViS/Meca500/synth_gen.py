"""Meca500 domain-randomized synthetic data generator (nvdiffrast).
Renders the URDF meshes at random joint configs + random camera poses over random
backgrounds; labels DH-FK keypoints (kinematics.py convention -> matches real GT).
--verify saves a few frames with keypoint overlay to confirm labels align before scaling.
"""
import os, sys, json, math, argparse, numpy as np, cv2, torch
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '../../Eval'))
sys.path.insert(0, '/home/najo/NAS/DIP/4_perception/DINOv3_fine_tunning')
from render_nvdr import NVDRSilhouette
from kinematics import Meca500Kinematics

MESH_DIR = os.path.join(HERE, 'meshes/visual')
MESHES = ['meca_500_r3_base.dae'] + [f'meca_500_r3_j{i}.dae' for i in range(1, 7)]
JOINTS = [((0.012498, 0, 0.091), 'z'), ((0, 0, 0.044), 'y'), ((0, 0, 0.135), 'y'),
          ((0, 0, 0.038), 'x'), ((0.12, 0, 0), 'y'), ((0.07, 0, 0), 'x')]
LIMITS = np.array([[-3.05, 3.05], [-1.22, 1.57], [-2.36, 1.22], [-2.97, 2.97], [-2.01, 2.01], [-3.14, 3.14]])
KIN = Meca500Kinematics()


def load_meca_meshes(device):
    import trimesh
    verts, faces, slices, off = [], [], [], 0
    for fidx, mf in enumerate(MESHES):
        m = trimesh.load(os.path.join(MESH_DIR, mf), force='mesh', process=False)
        v = np.asarray(m.vertices, np.float32); f = np.asarray(m.faces, np.int64)
        verts.append(v); faces.append(f + off); slices.append((fidx, off, off + len(v))); off += len(v)
    return {'verts': torch.from_numpy(np.concatenate(verts)).to(device),
            'faces': torch.from_numpy(np.concatenate(faces).astype(np.int32)).to(device),
            'slices': slices}


def _rot(ax, th):  # th (B,) -> (B,4,4)
    B = th.shape[0]; dev = th.device; c, s = torch.cos(th), torch.sin(th)
    R = torch.eye(4, device=dev).unsqueeze(0).repeat(B, 1, 1)
    if ax == 'z': R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c
    elif ax == 'y': R[:, 0, 0] = c; R[:, 0, 2] = s; R[:, 2, 0] = -s; R[:, 2, 2] = c
    else: R[:, 1, 1] = c; R[:, 1, 2] = -s; R[:, 2, 1] = s; R[:, 2, 2] = c
    return R


def meca_link_transforms(theta):  # (B,6) -> (B,7,4,4), base frame (matches URDF + kinematics.py base)
    B = theta.shape[0]; dev = theta.device
    T = torch.eye(4, device=dev).unsqueeze(0).repeat(B, 1, 1)
    Ts = [T.clone()]
    for i, (xyz, ax) in enumerate(JOINTS):
        Tr = torch.eye(4, device=dev).unsqueeze(0).repeat(B, 1, 1); Tr[:, :3, 3] = torch.tensor(xyz, device=dev)
        T = T @ Tr @ _rot(ax, theta[:, i]); Ts.append(T.clone())
    return torch.stack(Ts, 1)


def sample_cam(B, device):
    """look-at camera pose base->cam (R,t). Robot ~0.3m tall centered near (0,0,0.15)."""
    c = torch.tensor([0, 0, 0.15], dtype=torch.float32, device=device).expand(B, 3)
    d = torch.empty(B, device=device).uniform_(0.4, 1.1)
    az = torch.empty(B, device=device).uniform_(0, 2 * math.pi)
    el = torch.empty(B, device=device).uniform_(math.radians(-15), math.radians(70))
    dir = torch.stack([torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az), torch.sin(el)], -1)
    cam_pos = c + d.unsqueeze(1) * dir
    fwd = (c - cam_pos); fwd = fwd / fwd.norm(dim=1, keepdim=True)
    up0 = torch.tensor([0, 0, 1.0], device=device).expand(B, 3)
    right = torch.cross(fwd, up0, dim=1); right = right / right.norm(dim=1, keepdim=True)
    up = torch.cross(right, fwd, dim=1)
    R = torch.stack([right, -up, fwd], dim=1)  # world->cam (y-down image convention)
    t = -torch.einsum('bij,bj->bi', R, cam_pos)
    return R, t


def make_K(B, device, S):
    fx = torch.empty(B, device=device).uniform_(650, 820)
    K = torch.zeros(B, 3, 3, device=device)
    K[:, 0, 0] = fx; K[:, 1, 1] = fx; K[:, 0, 2] = S / 2; K[:, 1, 2] = S / 2; K[:, 2, 2] = 1
    return K


def rand_bg(B, S, device):
    kind = torch.randint(0, 3, (B,))
    bg = torch.rand(B, 3, 1, 1, device=device).expand(B, 3, S, S).clone()          # solid color
    noise = torch.rand(B, 3, S, S, device=device)
    m = (kind == 1).float().to(device).view(B, 1, 1, 1)
    bg = bg * (1 - m) + noise * m
    return bg


def project(kp3d_base, R, t, K):  # (B,7,3) base -> (B,7,2) px
    cam = torch.einsum('bij,bnj->bni', R, kp3d_base) + t.unsqueeze(1)
    uv = torch.einsum('bij,bnj->bni', K, cam); return uv[..., :2] / uv[..., 2:3], cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--out', default=os.path.join(HERE, 'synth'))
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()
    dev = torch.device('cuda'); S = args.size
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rdr = NVDRSilhouette(dev, kind='visual'); rdr.mesh = load_meca_meshes(dev)
    os.makedirs(args.out, exist_ok=True); os.makedirs(os.path.join(args.out, 'images'), exist_ok=True)
    done = 0
    while done < args.n:
        B = min(args.batch, args.n - done)
        theta = torch.from_numpy(np.random.uniform(LIMITS[:, 0], LIMITS[:, 1], (B, 6)).astype(np.float32)).to(dev)
        R, t = sample_cam(B, dev); K = make_K(B, dev, S)
        pts = rdr.robot_verts(theta, meca_link_transforms)
        shaded = rdr.render_shaded(pts, R, t, K, S, S)                 # (B,S,S) gray in [0,1]
        # colorize robot + composite over bg
        col = torch.rand(B, 3, 1, 1, device=dev).clamp(0.3, 1.0)
        robot = shaded.unsqueeze(1) * col                             # (B,3,S,S)
        mask = (shaded > 1e-4).float().unsqueeze(1)
        img = robot * mask + rand_bg(B, S, dev) * (1 - mask)
        img = (img.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()  # (B,S,S,3) RGB
        # labels: DH keypoints (kinematics.py) projected
        kp3d = torch.from_numpy(np.stack([KIN.forward_kinematics(theta[b].cpu().numpy()) for b in range(B)])).float().to(dev)
        kp2d, kpcam = project(kp3d, R, t, K)
        for b in range(B):
            idx = done + b
            im = img[b][..., ::-1].copy()  # to BGR for cv2
            if args.verify:
                for p in kp2d[b].cpu().numpy():
                    cv2.circle(im, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(args.out, 'images', f'{idx:06d}.jpg'), im)
            if not args.verify:
                Kb = K[b].cpu().numpy(); rec = {
                    'objects': [{'class': 'Meca500', 'visibility': 1,
                        'keypoints': [{'name': f'Meca500_link{j}', 'location': kpcam[b, j].cpu().tolist(),
                                       'projected_location': kp2d[b, j].cpu().tolist()} for j in range(7)]}],
                    'sim_state': {'joints': [{'position': float(theta[b, j])} for j in range(6)]},
                    'meta': {'image_path': f'images/{idx:06d}.jpg', 'K': Kb.tolist()}}
                json.dump(rec, open(os.path.join(args.out, f'{idx:06d}.json'), 'w'))
        done += B
    print(f'wrote {done} synthetic frames -> {args.out}' + ('  (verify overlays)' if args.verify else ''))


if __name__ == '__main__':
    main()
