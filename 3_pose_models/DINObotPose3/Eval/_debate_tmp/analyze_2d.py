import numpy as np, os, sys
TRACK=['link0','link2','link3','link4','link6','link7','hand']

def stats(dumpfile, label):
    d=np.load(dumpfile, allow_pickle=True)
    if 'kp2d_full' not in d.files:
        print(f'{label}: NO 2D fields'); return None
    kp=d['kp2d_full']; gt=d['gtkp2d']; found=d['found']
    dv=kp-gt
    # convert 512-aniso -> original 640x480 px
    dv640=dv.copy(); dv640[...,0]*=1.25; dv640[...,1]*=0.9375
    e=np.linalg.norm(dv640,axis=2)
    onimg=(gt[...,0]>=0)&(gt[...,0]<512)&(gt[...,1]>=0)&(gt[...,1]<512)
    m=found&onimg
    em=e[m]
    row=dict(label=label, N=int(m.sum()), nframe=len(kp),
             mean=em.mean(), med=np.median(em), p75=np.percentile(em,75),
             p90=np.percentile(em,90), p95=np.percentile(em,95),
             frac2=(em>2).mean(), frac5=(em>5).mean(), frac10=(em>10).mean(), frac20=(em>20).mean(),
             foundrate=found.mean(), onimgrate=onimg.mean())
    perkp=[ (TRACK[j], int(m[:,j].sum()), e[m[:,j],j].mean(), np.median(e[m[:,j],j])) for j in range(7)]
    row['perkp']=perkp
    return row

def pr(r):
    if r is None: return
    print(f"\n{r['label']}  (nframe={r['nframe']}, valid-kp N={r['N']}, found={r['foundrate']:.3f}, onimg={r['onimgrate']:.3f})")
    print(f"  err(orig-640px): mean={r['mean']:.2f}  MEDIAN={r['med']:.2f}  p75={r['p75']:.2f}  p90={r['p90']:.2f}  p95={r['p95']:.2f}")
    print(f"  tail frac >2px={r['frac2']:.3f} >5px={r['frac5']:.3f} >10px={r['frac10']:.3f} >20px={r['frac20']:.3f}")
    print("  per-kp median: "+"  ".join(f"{n}={md:.2f}" for n,_,_,md in r['perkp']))

rows=[]
rows.append(stats('rc_dumps_gf/p1b_ep17_gf.npz','SYNTH  (DR test, shared crop-det)'))
for cam in ['realsense','kinect','orb','azure']:
    f=f'_debate_tmp/real2d_{cam}.npz'
    if os.path.exists(f): rows.append(stats(f,f'REAL {cam}'))
    else: print(f'[pending] {f}')
for r in rows: pr(r)

print("\n=== SUMMARY (median / mean, orig-640 px, valid on-frame keypoints) ===")
for r in rows:
    if r: print(f"  {r['label']:36s} median={r['med']:5.2f}  mean={r['mean']:6.2f}  >10px={r['frac10']*100:4.1f}%")
