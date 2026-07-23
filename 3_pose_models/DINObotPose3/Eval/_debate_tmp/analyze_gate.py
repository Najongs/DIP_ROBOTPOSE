"""Analyze the distal-keypoint recoverability gate from gate_peaks.npz."""
import numpy as np, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
p = sys.argv[1] if len(sys.argv) > 1 else f'{HERE}/gate_peaks.npz'
d = np.load(p, allow_pickle=True)
TRACK = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
DISTAL = {'link3', 'link6', 'link7', 'hand'}
distal_idx = [i for i, n in enumerate(TRACK) if n in DISTAL]
prox_idx = [i for i, n in enumerate(TRACK) if n not in DISTAL]

peaks = d['peaks_orig']       # (F,7,K,2) orig-640 px
pvals = d['peak_vals']        # (F,7,K)
gt2d = d['gt2d_orig']         # (F,7,2)
gtoff = d['gtoff']            # (F,7) GT 2D off-frame
found = d['found']            # (F,7) GT 3D exists
argmax = d['argmax_orig']     # (F,7,2) deployed dark-decode
F_, N, K, _ = peaks.shape
print(f"frames={F_} kp={N} topk={K}")

# sanity: peak[...,0] == deployed argmax?
d01 = np.linalg.norm(peaks[:, :, 0, :] - argmax, axis=2)
print(f"[chk] |peak0 - dark_argmax| median={np.median(d01):.3f}px  max={d01.max():.2f}px  (should be ~0)")

m = found & (~gtoff)                                   # valid & on-frame keypoints
top1 = peaks[:, :, 0, :]
err1 = np.linalg.norm(top1 - gt2d, axis=2)             # (F,7) top-1 error orig-640
# distance of each of the K peaks to GT
dist = np.linalg.norm(peaks - gt2d[:, :, None, :], axis=3)   # (F,7,K)

print(f"\n[sanity] argmax >10px tail = {(err1[m]>10).mean()*100:.2f}%  median={np.median(err1[m]):.2f}px  n={int(m.sum())}")

def recover_report(cat_mask, label, secondary_upto):
    """Among catastrophic keypoints (cat_mask), fraction whose ANY secondary peak (idx 1..secondary_upto-1)
    is within thr of GT."""
    n = int(cat_mask.sum())
    if n == 0:
        print(f"  {label:22s} n=0"); return
    # secondary peaks only (exclude idx 0 = the argmax itself)
    sec = dist[:, :, 1:secondary_upto]                 # (F,7,S)
    best_sec = sec.min(axis=2)                          # closest secondary peak to GT
    row = f"  {label:22s} n={n:4d} | "
    for thr in (3, 5, 8):
        rec = (best_sec[cat_mask] <= thr).mean() * 100
        row += f"<={thr}px:{rec:5.1f}%  "
    print(row)

print("\n========== RECOVERABILITY (secondary peak near GT | argmax catastrophic >10px) ==========")
cat = m & (err1 > 10)
print(f"[secondary = top-2 or top-3  (peak idx 1..2)]")
recover_report(cat, "ALL keypoints", 3)
recover_report(cat & np.isin(np.arange(N)[None, :], distal_idx), "distal(l3/6/7/hand)", 3)
recover_report(cat & np.isin(np.arange(N)[None, :], prox_idx), "proximal(l0/2/4)", 3)
for j in distal_idx:
    cj = np.zeros_like(cat); cj[:, j] = cat[:, j]
    recover_report(cj, f"  {TRACK[j]}", 3)
print(f"[secondary = top-2..top-5  (peak idx 1..4) — upper bound]")
recover_report(cat, "ALL keypoints", K)
recover_report(cat & np.isin(np.arange(N)[None, :], distal_idx), "distal(l3/6/7/hand)", K)

# ---- characterization: unimodal-wrong vs bimodal-recoverable ----
mx = pvals[:, :, 0]                                     # (F,7) top peak value
n_sig = (pvals > 0.3 * mx[:, :, None]).sum(axis=2)      # significant modes (val > 0.3*max)
ratio21 = pvals[:, :, 1] / (mx + 1e-9)                  # peak2/peak1 value ratio
clean = m & (err1 <= 3)                                 # clean (argmax accurate)
print("\n========== TAIL CHARACTERIZATION ==========")
print(f"  {'set':22s}{'n':>7s}{'mean n_sig':>12s}{'mean peak2/peak1':>18s}{'%unimodal(nsig=1)':>18s}")
for label, mm in [('catastrophic(>10px)', cat), ('clean(<=3px)', clean)]:
    print(f"  {label:22s}{int(mm.sum()):>7d}{n_sig[mm].mean():>12.2f}{ratio21[mm].mean():>18.3f}{(n_sig[mm]==1).mean()*100:>17.1f}%")
# among catastrophic: how many have a secondary within 5px AND are bimodal
cat_bimodal = cat & (n_sig >= 2)
print(f"  catastrophic & bimodal(nsig>=2): {int(cat_bimodal.sum())} / {int(cat.sum())} = {cat_bimodal.sum()/max(cat.sum(),1)*100:.1f}%")

# ---- mechanism: does the catastrophic argmax snap onto ANOTHER keypoint's GT location? ----
# (link-identity confusion signature, cf. KUKA "90% snap onto another kp GT", gap-doc §14.1/§16.2)
print("\n========== MECHANISM: argmax lands on a WRONG link? ==========")
allgt = gt2d[:, None, :, :]                          # (F,1,7,2) each frame's 7 GT locations
am2 = argmax[:, :, None, :]                          # (F,7,1,2)
d_to_allgt = np.linalg.norm(am2 - allgt, axis=3)     # (F,7,7): argmax_j vs gt_k
own = np.arange(N)
d_to_other = d_to_allgt.copy()
d_to_other[:, own, own] = 1e9                        # mask self
nearest_other = d_to_other.min(axis=2)               # (F,7) dist to nearest OTHER kp GT
snap = nearest_other <= 5.0                          # argmax sits on another kp's GT (<=5px)
print(f"  catastrophic kp whose argmax is within 5px of ANOTHER kp's GT: "
      f"{(snap[cat]).mean()*100:.1f}%  (n={int(cat.sum())})")
print(f"  ... within 8px: {(nearest_other[cat] <= 8).mean()*100:.1f}%")

# ---- bonus: dense heatmap-refiner ADD vs base ----
if 'base_add' in d and 'hm_add' in d:
    ba = d['base_add']; ha = d['hm_add']; v = found
    def frame_auc(a):
        fr = np.array([a[i][v[i]].mean() for i in range(len(a)) if v[i].any()])
        ts = np.arange(0, 0.1, 1e-5); return float((fr[None, :] <= ts[:, None]).mean(1).mean())
    print("\n========== BONUS: dense heatmap-refiner (solve_batch_heatmap) ==========")
    print(f"  base argmax ADD-AUC = {frame_auc(ba):.4f}   dense-heatmap ADD-AUC = {frame_auc(ha):.4f}")

# ---- headroom estimate: GT-2D perfect-distal ceiling on catastrophic frames ----
# fraction of FRAMES that contain >=1 catastrophic distal keypoint
distal_cat_frame = (cat[:, distal_idx]).any(axis=1)
print(f"\n[frames] with >=1 catastrophic distal argmax: {distal_cat_frame.sum()} / {F_} = {distal_cat_frame.mean()*100:.1f}%")
