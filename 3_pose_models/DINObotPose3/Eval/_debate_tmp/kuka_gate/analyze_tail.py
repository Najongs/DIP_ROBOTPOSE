"""Offline characterization of the KUKA catastrophic tail from the gate dump.
Distinguishes SYMMETRY (peak lands on another link's body) vs OCCLUSION/no-evidence,
measures frame concentration, and confidence structure."""
import numpy as np, os
HERE = os.path.dirname(__file__)
R = dict(np.load(os.path.join(HERE, 'kuka_gate_full.npz')))
sa = R['sa_err']; hp = R['hp_err']
cata = sa > 10.0
tw = cata & (hp > 10.0)          # true hard-peak swaps
good = ~cata
print(f"valid kp {len(sa)}, catastrophic {cata.sum()} ({100*cata.mean():.1f}%), true-swap {tw.sum()}")

# --- confidence structure: swapped peaks confident or diffuse? ---
pk = R['peak']
print(f"\npeak-conf: good {pk[good].mean():.2f}  soft-pull {pk[cata&(hp<=10)].mean():.2f}  true-swap {pk[tw].mean():.2f}")
print(f"  true-swap peak percentiles: p10 {np.percentile(pk[tw],10):.2f} p50 {np.percentile(pk[tw],50):.2f} p90 {np.percentile(pk[tw],90):.2f}")
print(f"  good      peak percentiles: p10 {np.percentile(pk[good],10):.2f} p50 {np.percentile(pk[good],50):.2f} p90 {np.percentile(pk[good],90):.2f}")
# What fraction of true swaps are HIGH-conf (would survive conf-gate 0.05 relative)? peak is raw logit.
gate = np.percentile(pk[good], 10)
print(f"  true-swaps with peak >= good-p10 ({gate:.2f}): {100*(pk[tw]>=gate).mean():.1f}%  (confident-wrong -> conf-gate blind)")

# --- swap-radius sweep: does the peak land on ANOTHER link (symmetry) ---
sd = R['swap_dist']
for thr in [10,15,20,30,40,60]:
    print(f"  true-swap hard-argmax within {thr:3d}px of ANOTHER kp GT: {100*(sd[tw]<=thr).mean():5.1f}%")

# --- frame concentration ---
fr = R['frame']
# per-frame catastrophic count
frames = np.unique(fr)
cat_per_frame = np.zeros(int(frames.max())+1, int)
val_per_frame = np.zeros(int(frames.max())+1, int)
np.add.at(cat_per_frame, fr[cata], 1)
np.add.at(val_per_frame, fr, 1)
present = val_per_frame>0
c = cat_per_frame[present]
print(f"\nframe concentration ({present.sum()} frames):")
for k in range(0,7):
    print(f"  frames with exactly {k} catastrophic kp: {100*(c==k).mean():5.1f}%")
print(f"  frames with >=1: {100*(c>=1).mean():.1f}%  >=2: {100*(c>=2).mean():.1f}%  >=3: {100*(c>=3).mean():.1f}%")
print(f"  catastrophic kp in frames with >=3 cata: {100*c[c>=3].sum()/c.sum():.1f}% of all catastrophic")

# --- self-occlusion / folded proxy: is own GT crowded near other kp GTs? ---
# reconstruct per-frame GT to get min inter-kp 2D distance for each kp
gx = R['gt_x']; gy = R['gt_y']; kp = R['kp']
from collections import defaultdict
byf = defaultdict(list)
for i in range(len(fr)):
    byf[fr[i]].append(i)
crowd = np.full(len(fr), np.nan)
for f, idxs in byf.items():
    P = np.stack([gx[idxs], gy[idxs]], 1)
    for a,ia in enumerate(idxs):
        d = np.linalg.norm(P - P[a], axis=1); d[a]=1e9
        crowd[ia] = d.min()
print(f"\nself-crowding (min 2D dist own-GT to another kp GT, px):")
print(f"  good      : median {np.nanmedian(crowd[good]):.1f}")
print(f"  true-swap : median {np.nanmedian(crowd[tw]):.1f}")
for thr in [15,25,40]:
    print(f"  true-swaps with a neighbor kp within {thr}px (folded/occluded): {100*(crowd[tw]<=thr).mean():.1f}%  (vs good {100*(crowd[good]<=thr).mean():.1f}%)")
