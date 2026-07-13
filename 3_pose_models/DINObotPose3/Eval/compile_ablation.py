#!/usr/bin/env python3
"""Compile the B leave-one-out table from ablation_logs/ (anchored-regex re-extraction).
Final value: azure = base log (no RC); rs/kinect/orb = RC log, except no_rc = base log."""
import os, re, glob
LOG = os.path.join(os.path.dirname(__file__), 'ablation_logs')
ABL = ['full', 'no_cov', 'no_dark', 'no_rot', 'no_rc', 'gt_bbox', 'no_confgate', 'no_occaug']
CAMS = ['realsense', 'kinect', 'orb', 'azure']
LABEL = {'full': 'Full (deployed)', 'no_cov': '- cov-PnP', 'no_dark': '- DARK',
         'no_rot': '- rot-head', 'no_rc': '- render-compare', 'gt_bbox': 'auto->GT bbox',
         'no_confgate': '- conf-gate', 'no_occaug': '- occ-aug/self-train'}
BASE_RE = re.compile(r'ADD-AUC@100mm[: ]+([0-9]+\.[0-9]+)')
RC_RE   = re.compile(r'render-compare ADD-AUC@100mm[: ]+([0-9]+\.[0-9]+)')

def grab(path, rx):
    if not os.path.exists(path): return None
    vals = rx.findall(open(path, errors='ignore').read())
    return float(vals[-1]) if vals else None

def cell(abl, cam):
    if cam == 'azure':
        if abl == 'no_rc': abl = 'full'          # azure has no RC
        return grab(f'{LOG}/{abl}_azure_base.log', BASE_RE)
    if abl == 'no_rc':                            # base-only
        return grab(f'{LOG}/no_rc_{cam}_base.log', BASE_RE)
    return grab(f'{LOG}/{abl}_{cam}_rc.log', RC_RE)

# full baseline
full = {c: cell('full', c) for c in CAMS}
fmean = sum(v for v in full.values() if v) / 4 if all(full.values()) else None
print(f"{'ablation':<22}{'rs':>8}{'kinect':>8}{'orb':>8}{'azure':>8}{'mean':>8}{'dMean':>8}")
for abl in ABL:
    row = {c: cell(abl, c) for c in CAMS}
    vals = [row[c] for c in CAMS]
    mean = sum(v for v in vals if v is not None) / len([v for v in vals if v is not None]) if any(vals) else None
    d = (mean - fmean) if (mean is not None and fmean is not None and abl != 'full') else None
    fmt = lambda v: f'{v:.4f}' if v is not None else '  --  '
    dstr = f'{d:+.4f}' if d is not None else ('' if abl == 'full' else '  --  ')
    print(f"{LABEL[abl]:<22}{fmt(row['realsense']):>8}{fmt(row['kinect']):>8}{fmt(row['orb']):>8}{fmt(row['azure']):>8}{fmt(mean):>8}{dstr:>8}")
print("\n(Full=deployed 0.804; each row removes one lever. dMean = mean drop vs Full = that lever's contribution.)")
print("note: 'auto->GT bbox' & '- render-compare(azure n/a)' interpreted per deployed policy.")
