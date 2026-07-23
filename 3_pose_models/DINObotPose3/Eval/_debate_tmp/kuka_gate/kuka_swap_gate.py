"""KUKA recoverability / swap gate (forward-only, deployed KUKA detector).

For every VALID keypoint on the kuka_synth_test_dr set, dumps per-keypoint heatmap diagnostics
in the SAME 512 crop frame the deployed detector sees:
  - hard-argmax peak (the heatmap MODE the detector commits to)
  - global soft-argmax (what the deployed solver actually consumes; DECODE_WINDOW=0)
  - windowed soft-argmax (win=15) — the repo's distractor-robust decode
  - top-M modes via iterative NMS suppression (radius r), with values
  - gt_response = (max heatmap value in +/-3px window around GT) / (global peak max)  [Panda gate metric]
  - swap target: nearest OTHER-keypoint GT to the hard-argmax peak

VERDICT logic (printed):
  Among CATASTROPHIC keypoints (deployed decode err > CATA px):
    * how many are TRUE hard-peak swaps (hard-argmax on wrong link) vs soft-argmax-pull
      (hard peak correct, soft pulled) -> the latter is fixed for free by windowed decode
    * of the true swaps: recoverable@5px in top-2 / top-3 / top-5 modes? gt_response distribution?
  recoverable -> kinematic/skeleton-consistent decode is VIABLE. absent -> detector-limited.
"""
import argparse, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(__file__)
TRAIN = os.path.abspath(os.path.join(HERE, '../../../TRAIN'))
EVAL = os.path.abspath(os.path.join(HERE, '../..'))
sys.path.append(TRAIN); sys.path.append(EVAL)
from model_angle import AnglePredictor
from model_v4 import soft_argmax_2d, windowed_soft_argmax_2d
from dataset import PoseEstimationDataset

KP_NAMES = [f'iiwa7_link_{i}' for i in range(1, 8)]
ANGLE_JOINTS = [f'iiwa7_joint_{i}' for i in range(1, 8)]


def topM_modes(hm, M=5, r=8):
    """hm (G,H,W) raw heatmap -> top-M local maxima via iterative argmax + window suppression.
    Returns coords (G,M,2) [x,y] and vals (G,M)."""
    G, H, W = hm.shape
    dev = hm.device
    work = hm.clone()
    ys = torch.arange(H, device=dev); xs = torch.arange(W, device=dev)
    coords = torch.zeros(G, M, 2, device=dev)
    vals = torch.zeros(G, M, device=dev)
    for m in range(M):
        flat = work.reshape(G, -1)
        v, idx = flat.max(1)
        y = (idx // W).float(); x = (idx % W).float()
        coords[:, m, 0] = x; coords[:, m, 1] = y
        vals[:, m] = v
        if m < M - 1:
            ymask = (ys[None, :] - y[:, None]).abs() <= r    # (G,H)
            xmask = (xs[None, :] - x[:, None]).abs() <= r    # (G,W)
            mask = ymask[:, :, None] & xmask[:, None, :]      # (G,H,W)
            work = work.masked_fill(mask, -1e9)
    return coords, vals


def gt_window_max(hm, gt, r=3):
    """hm (G,H,W), gt (G,2)[x,y] -> max heatmap value in +/-r window around GT (G,)."""
    G, H, W = hm.shape
    dev = hm.device
    ys = torch.arange(H, device=dev); xs = torch.arange(W, device=dev)
    gx = gt[:, 0].clamp(0, W - 1); gy = gt[:, 1].clamp(0, H - 1)
    ymask = (ys[None, :] - gy[:, None]).abs() <= r
    xmask = (xs[None, :] - gx[:, None]).abs() <= r
    mask = ymask[:, :, None] & xmask[:, None, :]
    masked = hm.masked_fill(~mask, -1e9)
    return masked.reshape(G, -1).max(1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--crop-margin', type=float, default=1.5)
    ap.add_argument('--cata', type=float, default=10.0, help='catastrophic decode-err threshold px')
    ap.add_argument('--rec', type=float, default=5.0, help='recoverable radius px')
    ap.add_argument('--swap-thr', type=float, default=15.0, help='swap match radius px')
    ap.add_argument('--nms-r', type=int, default=8)
    ap.add_argument('--out', default='kuka_gate_dump.npz')
    args = ap.parse_args()

    device = torch.device('cuda'); IS = args.image_size
    m = AnglePredictor(args.model_name, IS, fix_joint7_zero=True, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict()
                       and v.shape == m.state_dict()[k].shape}, strict=False)

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=KP_NAMES, image_size=(IS, IS),
                               heatmap_size=(IS, IS), augment=False, include_angles=True, sigma=2.5,
                               crop_to_robot=True, crop_margin=args.crop_margin,
                               angle_joint_names=ANGLE_JOINTS)
    if args.max_frames and args.max_frames < len(ds):
        ds.samples = ds.samples[::max(1, len(ds.samples) // args.max_frames)][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    rec = {k: [] for k in ['frame', 'kp', 'hp_err', 'sa_err', 'wsa_err', 'gt_resp', 'peak',
                           'oracle_err', 'rec_top2', 'rec_top3', 'rec_top5', 'nmode_close',
                           'swap_to', 'swap_dist', 'hp_x', 'hp_y', 'gt_x', 'gt_y']}
    # also keep top-M coords/vals for the decode experiment
    modes_x = []; modes_y = []; modes_v = []; midx = []
    fi = 0
    for batch in tqdm(loader, desc='kuka-gate'):
        img = batch['image'].to(device)
        gt2d = batch['keypoints'].to(device).float()          # (B,7,2) @512
        valid = batch['valid_mask'].to(device)                # (B,7)
        with torch.no_grad():
            tokens = m.backbone(img)
            hm = m.keypoint_head(tokens)                       # (B,7,512,512)
        B, N, H, W = hm.shape
        sa = soft_argmax_2d(hm)                                # (B,7,2) deployed decode
        wsa = windowed_soft_argmax_2d(hm, win=15)              # distractor-robust
        peak = hm.flatten(2).max(2)[0]                         # (B,7)
        G = B * N
        hmg = hm.reshape(G, H, W)
        coords, vals = topM_modes(hmg, M=5, r=args.nms_r)      # (G,5,2),(G,5)
        coords = coords.reshape(B, N, 5, 2); vals = vals.reshape(B, N, 5)
        hp = coords[:, :, 0, :]                                # hard-argmax = mode 0 (B,7,2)
        gtresp = gt_window_max(hmg, gt2d.reshape(G, 2), r=3).reshape(B, N) / peak.clamp(min=1e-6)

        for b in range(B):
            for j in range(N):
                if not bool(valid[b, j]):
                    continue
                g = gt2d[b, j]
                hp_err = float((hp[b, j] - g).norm())
                sa_err = float((sa[b, j] - g).norm())
                wsa_err = float((wsa[b, j] - g).norm())
                # top-M mode distances to GT
                md = (coords[b, j] - g[None]).norm(dim=-1)     # (5,)
                oracle_err = float(md.min())
                rec_top2 = bool((md[:2] <= args.rec).any())
                rec_top3 = bool((md[:3] <= args.rec).any())
                rec_top5 = bool((md[:5] <= args.rec).any())
                nmode_close = int((md <= args.rec).sum())
                # swap: nearest OTHER valid keypoint GT to the hard-argmax peak
                sd_best = 1e9; sto = -1
                for k in range(N):
                    if k == j or not bool(valid[b, k]):
                        continue
                    d = float((hp[b, j] - gt2d[b, k]).norm())
                    if d < sd_best:
                        sd_best = d; sto = k
                rec['frame'].append(fi + b); rec['kp'].append(j)
                rec['hp_err'].append(hp_err); rec['sa_err'].append(sa_err); rec['wsa_err'].append(wsa_err)
                rec['gt_resp'].append(float(gtresp[b, j])); rec['peak'].append(float(peak[b, j]))
                rec['oracle_err'].append(oracle_err)
                rec['rec_top2'].append(rec_top2); rec['rec_top3'].append(rec_top3); rec['rec_top5'].append(rec_top5)
                rec['nmode_close'].append(nmode_close)
                rec['swap_to'].append(sto); rec['swap_dist'].append(sd_best)
                rec['hp_x'].append(float(hp[b, j, 0])); rec['hp_y'].append(float(hp[b, j, 1]))
                rec['gt_x'].append(float(g[0])); rec['gt_y'].append(float(g[1]))
        # dump modes for decode expt (per frame, all 7 kp, 5 modes)
        modes_x.append(coords[..., 0].cpu().numpy()); modes_y.append(coords[..., 1].cpu().numpy())
        modes_v.append(vals.cpu().numpy()); midx.append(np.arange(fi, fi + B))
        fi += B

    R = {k: np.array(v) for k, v in rec.items()}
    np.savez(os.path.join(HERE, args.out), **R)
    print(f"\nsaved per-keypoint dump -> {args.out}  ({len(R['frame'])} valid keypoints, {fi} frames)")

    # ------------------- VERDICT -------------------
    CATA, REC = args.cata, args.rec
    hp_err, sa_err, wsa_err = R['hp_err'], R['sa_err'], R['wsa_err']
    print("\n" + "=" * 70)
    print(f"  KUKA RECOVERABILITY / SWAP GATE  ({len(hp_err)} valid kp, cata>{CATA}px, rec<={REC}px)")
    print("=" * 70)
    print(f"  decoded-2D error (px)   :  mean/median/p90")
    for nm, e in [('hard-argmax', hp_err), ('soft-argmax(deployed)', sa_err), ('windowed-sa(win15)', wsa_err)]:
        print(f"    {nm:<24} {e.mean():7.2f} {np.median(e):7.2f} {np.percentile(e,90):7.2f}")
    print(f"    PCK@{CATA:.0f}px  hard {100*(hp_err<=CATA).mean():5.1f}%  "
          f"soft {100*(sa_err<=CATA).mean():5.1f}%  windowed {100*(wsa_err<=CATA).mean():5.1f}%")

    cata = sa_err > CATA                                       # catastrophic by DEPLOYED decode
    ncat = int(cata.sum())
    print(f"\n  CATASTROPHIC (deployed soft-argmax err > {CATA}px): {ncat} kp = {100*cata.mean():.2f}% of valid kp")
    if ncat == 0:
        return
    # decompose: true hard-peak swap vs soft-argmax pull
    hard_wrong = cata & (hp_err > CATA)
    soft_pull = cata & (hp_err <= CATA)                        # hard peak correct, soft pulled off
    win_fix = cata & (wsa_err <= CATA)                         # windowed decode already fixes
    print(f"    of catastrophic:")
    print(f"      hard-peak ALSO wrong (true swap)   : {int(hard_wrong.sum()):5d} ({100*hard_wrong.sum()/ncat:5.1f}%)")
    print(f"      hard-peak OK, soft-argmax pulled   : {int(soft_pull.sum()):5d} ({100*soft_pull.sum()/ncat:5.1f}%)")
    print(f"      windowed decode ALREADY fixes (<={CATA}px): {int(win_fix.sum()):5d} ({100*win_fix.sum()/ncat:5.1f}%)  [FREE]")

    # recoverability among TRUE hard-peak swaps (the residual after windowed decode)
    tw = hard_wrong
    ntw = int(tw.sum())
    print(f"\n  TRUE hard-peak swaps: {ntw} kp  (these survive windowed decode)")
    if ntw > 0:
        print(f"    recoverable@{REC}px in top-2 modes : {100*R['rec_top2'][tw].mean():5.1f}%")
        print(f"    recoverable@{REC}px in top-3 modes : {100*R['rec_top3'][tw].mean():5.1f}%")
        print(f"    recoverable@{REC}px in top-5 modes : {100*R['rec_top5'][tw].mean():5.1f}%")
        print(f"    oracle-mode-select err (px)        : mean {R['oracle_err'][tw].mean():6.2f}  median {np.median(R['oracle_err'][tw]):6.2f}")
        gr = R['gt_resp'][tw]
        print(f"    gt_response (GT-window max / peak)  : mean {gr.mean():5.3f}  median {np.median(gr):5.3f}  "
              f">0.3: {100*(gr>0.3).mean():4.1f}%  >0.5: {100*(gr>0.5).mean():4.1f}%")

    # swap structure (on true hard-peak swaps within swap-thr of another kp's GT)
    print(f"\n  SWAP STRUCTURE (hard-argmax within {args.swap_thr}px of ANOTHER kp's GT):")
    swapped = tw & (R['swap_dist'] <= args.swap_thr)
    print(f"    of {ntw} true swaps, {int(swapped.sum())} ({100*swapped.sum()/max(ntw,1):.1f}%) land on another link's GT")
    kpn = R['kp'][swapped]; sto = R['swap_to'][swapped]
    mat = np.zeros((7, 7), int)
    for a, b2 in zip(kpn, sto):
        mat[a, b2] += 1
    print("      rows = true link (0..6), cols = link it snapped ONTO:")
    print("        " + " ".join(f"L{c}" for c in range(7)))
    for a in range(7):
        print(f"      L{a} " + " ".join(f"{mat[a,c]:3d}" for c in range(7)) + f"   (n={mat[a].sum()})")
    # per-keypoint catastrophic rate
    print(f"\n  per-link catastrophic rate (deployed decode):")
    for j in range(7):
        mk = R['kp'] == j
        if mk.sum():
            print(f"    L{j}: {100*(sa_err[mk]>CATA).mean():5.1f}%  (n={int(mk.sum())})")
    print("=" * 70)


if __name__ == '__main__':
    main()
