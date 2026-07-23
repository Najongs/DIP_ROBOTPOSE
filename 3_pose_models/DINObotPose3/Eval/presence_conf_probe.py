#!/usr/bin/env python3
"""
Presence-signal probe: does the heatmap max (=confidence) ALREADY encode "this keypoint is
off-frame", so that no new border-margin gate is needed?

Motivation: TRAIN/dataset.py:609-611 skips off-frame keypoints when rasterizing the target, so
their heatmap target is ALL ZERO, and TRAIN/train_heatmap.py:232-234 applies no valid mask. The
detector has therefore been explicitly trained to emit NO peak for an off-frame keypoint —
equivalent to RoboPEPP masking off-frame keypoints out of its loss. If that training signal took,
confidence is already a presence detector and the fix is threshold calibration, not new code.

Reads a selfbbox_eval dump carrying conf (B,7) and gtoff (B,7).
"""
import sys

import numpy as np


def roc_auc(scores, labels):
    """P(score[pos] > score[neg]), rank-based (handles ties)."""
    labels = labels.astype(bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return float('nan')
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float('nan')


def main(path, flag_px=10.0):
    d = np.load(path, allow_pickle=True)
    conf, gtoff = d['conf'], d['gtoff'].astype(bool)          # (F,7)
    reproj = d['reproj']                                       # (F,)
    add = np.linalg.norm(d['kp_cam'] - d['gt3d'], axis=-1).mean(axis=1)  # (F,) meters
    F = len(reproj)
    auc = np.maximum(0.0, 1.0 - 10.0 * add).mean()
    print(f"frames={F}  ADD-AUC={auc:.4f}  median ADD={np.median(add)*1000:.1f}mm")
    print(f"off-frame keypoints: {gtoff.sum()}/{gtoff.size} ({100*gtoff.mean():.2f}%)"
          f" | frames with >=1 off-frame kp: {100*gtoff.any(1).mean():.1f}%")

    print("\n== (1) confidence separation: off-frame vs in-frame keypoints ==")
    co, ci = conf[gtoff], conf[~gtoff]
    print(f"  {'set':<12}{'n':>7}{'mean':>9}{'p10':>9}{'p50':>9}{'p90':>9}")
    for nm, a in (("OFF-frame", co), ("in-frame", ci)):
        print(f"  {nm:<12}{len(a):>7}{a.mean():>9.4f}{q(a,10):>9.4f}{q(a,50):>9.4f}{q(a,90):>9.4f}")
    print(f"  ROC-AUC(low conf -> off-frame) = {roc_auc(-conf.ravel(), gtoff.ravel()):.4f}")

    print("\n  separation at candidate conf gates (what fraction each side loses):")
    print(f"  {'gate':>7}{'off-frame dropped':>20}{'in-frame dropped':>19}")
    for g in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        print(f"  {g:>7.2f}{100*(co<g).mean():>19.1f}%{100*(ci<g).mean():>18.1f}%")

    print("\n== (2) the divergent tail ==")
    flag = reproj > flag_px
    print(f"  reproj>{flag_px}px flags {flag.sum()}/{F} ({100*flag.mean():.1f}%)"
          f" | ADD>100mm in flagged {100*(add[flag]>0.1).mean():.1f}%"
          f" vs unflagged {100*(add[~flag]>0.1).mean():.1f}%")
    print(f"  ADD>100mm overall {100*(add>0.1).mean():.1f}% (contributes EXACTLY 0 to AUC)")
    print(f"  P(ADD>100mm | any off-frame kp) = {100*(add[gtoff.any(1)]>0.1).mean():.1f}%"
          f" | P(ADD>100mm | none) = {100*(add[~gtoff.any(1)]>0.1).mean():.1f}%")

    print("\n  confidence of off-frame keypoints INSIDE the divergent tail (the frames that matter):")
    tail = add > 0.1
    for nm, m in (("tail(ADD>100mm)", tail), ("good", ~tail)):
        c_off = conf[m[:, None] & gtoff]
        if len(c_off):
            print(f"    {nm:<18} off-frame kp conf: n={len(c_off):<5} mean={c_off.mean():.4f}"
                  f" p50={q(c_off,50):.4f} p90={q(c_off,90):.4f}"
                  f" frac>0.2={(c_off>0.2).mean():.2f}")

    print("\n  what DOES distinguish the divergent frames (tail vs good):")
    print(f"    {'quantity':<28}{'tail':>10}{'good':>10}")
    for nm, v in (("min conf over 7 kp", conf.min(1)), ("mean conf", conf.mean(1)),
                  ("n off-frame kp", gtoff.sum(1).astype(float)), ("solved reproj px", reproj)):
        print(f"    {nm:<28}{np.median(v[tail]):>10.3f}{np.median(v[~tail]):>10.3f}")

    print("\n== (3) headroom ceiling ==")
    rep = add.copy()
    rep[tail] = np.median(add[~tail])
    print(f"  if every ADD>100mm frame were repaired to the good-frame median:"
          f" AUC {auc:.4f} -> {np.maximum(0,1-10*rep).mean():.4f}"
          f" (+{np.maximum(0,1-10*rep).mean()-auc:.4f})")


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 10.0)
