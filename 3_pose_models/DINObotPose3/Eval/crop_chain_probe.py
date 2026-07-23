#!/usr/bin/env python
"""크롭 리샘플링 체인 진단: 학습(dataset.py) vs 배포(selfbbox_eval.py) 불일치 측정.

배포 파이프라인은 원본 640x480 을 먼저 512x512 로 (비등방 4:3->1:1) 리사이즈한 뒤
그 텐서에서 roi_align 으로 정사각 크롭을 뜬다. 반면 crop detector 는 dataset.py 에서
원본 해상도의 정사각 크롭을 512 로 리사이즈한 이미지로 학습됐다. 즉

  TRAIN : orig -> square crop (원본 px, 종횡비 보존) -> resize 512      [1회 리샘플]
  DEPLOY: orig -> resize 512x512 (x0.8, y1.0667) -> square roi_align -> 512  [2회 리샘플 + 4:3 왜곡]

배포 크롭은 원본에서 보면 4:3 직사각형을 정사각으로 늘린 것 => 로봇이 세로로 33% 늘어난다.
작은 로봇일수록 업샘플 배율이 커서 2회 리샘플의 블러 손실도 커진다.

체인 A(배포) / B(학습) / C(등방 2회리샘플: 왜곡만 제거) 를 ORACLE GT bbox 로 고정하고
crop detector 의 2D 오차를 원본 프레임 px 로 비교한다. bbox 오차는 세 체인 모두 동일(0)이므로
차이는 순수하게 리샘플링 체인 탓이다.
"""
import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # off-frame GT kp -> 거대 bbox 크롭 허용 (clean 프레임엔 영향 없음)
from torchvision import transforms
from torchvision.ops import roi_align
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'TRAIN'))
sys.path.insert(0, HERE)
from train_heatmap import HeatmapModel          # noqa: E402
from decode_util import dark_decode             # noqa: E402

KP = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4',
      'panda_link6', 'panda_link7', 'panda_hand']
GT_DIR = os.path.join(HERE, '..', 'Dataset', 'Converted_dataset',
                      'DREAM_to_DREAM_syn', 'panda_synth_test_dr')
IMG_DIR = os.path.join(HERE, '..', 'Dataset', 'DREAM_syn', 'panda_synth_test_dr')
W0, H0 = 640, 480
TF = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def square_box(pts, margin):
    """pts (7,2) -> (x0,y0,side) 정사각 bbox. dataset.py / detected_bbox 와 동일 식."""
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(max(x1 - x0, y1 - y0) * margin, 16.0)
    return cx - side / 2, cy - side / 2, side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', default=os.path.join(
        HERE, '..', 'TRAIN', 'outputs_heatmap', 'crop_20260605_010622', 'best_heatmap.pth'))
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--margin', type=float, default=1.5)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--out', default=os.path.join(HERE, 'ablation_logs', 'crop_chain.npz'))
    args = ap.parse_args()

    device = torch.device('cuda')
    S = 512
    model = HeatmapModel(args.model_name, (S, S), unfreeze_blocks=0).to(device).eval()
    sd = torch.load(args.detector, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    msd = model.state_dict()
    keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
    model.load_state_dict(keep, strict=False)
    print(f'detector={args.detector}\n  matched {len(keep)}/{len(msd)} tensors')

    D = np.load(os.path.join(HERE, 'ablation_logs', 'mediocre_band.npz'), allow_pickle=True)
    fids = [str(f) for f in D['fid']]
    add = D['add'] * 1000
    clean = D['n_offframe'] == 0
    exc = clean & (add < 30)
    med = clean & (add >= 30) & (add < 100)
    print(f'frames={len(fids)}  clean={clean.sum()}  excellent={exc.sum()}  mediocre={med.sum()}')

    # ---- GT 2D (원본 프레임) ----
    gt2d = np.zeros((len(fids), 7, 2))
    for i, f in enumerate(fids):
        d = json.load(open(os.path.join(GT_DIR, f'{f}.json')))
        kmap = {k['name']: k for k in d['objects'][0]['keypoints']}
        gt2d[i] = [kmap[n]['projected_location'] for n in KP]

    chains = ['A_deploy', 'B_train', 'C_iso2x', 'D_distort1x']
    pred = {c: np.zeros((len(fids), 7, 2)) for c in chains}

    for b0 in tqdm(range(0, len(fids), args.batch_size), desc='crop chains'):
        b1 = min(b0 + args.batch_size, len(fids))
        imgs = [Image.open(os.path.join(IMG_DIR, f'{fids[i]}.rgb.jpg')).convert('RGB')
                for i in range(b0, b1)]
        B = len(imgs)

        # ---------- 체인 A: 배포 (resize 512x512 먼저 -> roi_align) ----------
        t512 = torch.stack([TF(im) for im in imgs]).to(device)          # (B,3,512,512)
        boxA, metaA = [], []
        for j, i in enumerate(range(b0, b1)):
            p = gt2d[i].copy()
            p[:, 0] *= S / W0; p[:, 1] *= S / H0                        # 512-공간 GT kp
            x0, y0, sd_ = square_box(p, args.margin)
            boxA.append([x0, y0, x0 + sd_, y0 + sd_]); metaA.append((x0, y0, sd_))
        rois = torch.cat([torch.arange(B, device=device).view(-1, 1).float(),
                          torch.tensor(boxA, device=device, dtype=torch.float32)], 1)
        cropA = roi_align(t512, rois, output_size=(S, S), spatial_scale=1.0, aligned=True)

        # ---------- 체인 B: 학습 (원본에서 정사각 크롭 -> resize 512) ----------
        cropB, metaB = [], []
        for j, i in enumerate(range(b0, b1)):
            x0, y0, sd_ = square_box(gt2d[i], args.margin)
            bx0, by0, bs = int(round(x0)), int(round(y0)), int(round(sd_))
            im = imgs[j].crop((bx0, by0, bx0 + bs, by0 + bs))
            cropB.append(TF(im)); metaB.append((bx0, by0, bs))
        cropB = torch.stack(cropB).to(device)

        # ---------- 체인 C: 등방 2회 리샘플 (왜곡만 제거) ----------
        # orig -> 등방 스케일 s=512/480 로 리사이즈(683x512) -> 그 공간에서 정사각 크롭 -> 512
        s_iso = S / H0
        cropC, metaC = [], []
        for j, i in enumerate(range(b0, b1)):
            iw, ih = int(round(W0 * s_iso)), S
            imr = imgs[j].resize((iw, ih), Image.BILINEAR)
            p = gt2d[i] * s_iso
            x0, y0, sd_ = square_box(p, args.margin)
            bx0, by0, bs = int(round(x0)), int(round(y0)), int(round(sd_))
            cropC.append(TF(imr.crop((bx0, by0, bx0 + bs, by0 + bs))))
            metaC.append((bx0, by0, bs))
        cropC = torch.stack(cropC).to(device)

        # ---------- 체인 D: A 와 동일한 기하(4:3 왜곡)이지만 원본에서 1회 리샘플 ----------
        # A 의 정사각(512공간) 박스는 원본에서 보면 w=side/0.8, h=side/1.0667 인 4:3 직사각형이다.
        # 그 직사각형을 원본 해상도에서 바로 잘라 512x512 로 늘린다 => 왜곡은 같고 해상도 손실만 없음.
        # B-D = 순수 왜곡 페널티(학습으로 교정 가능), D-A = 순수 해상도 손실(평가측에서만 교정 가능).
        cropD, metaD = [], []
        for j, i in enumerate(range(b0, b1)):
            x0, y0, sd_ = metaA[j]
            rx0, ry0 = x0 * (W0 / S), y0 * (H0 / S)
            rw, rh = sd_ * (W0 / S), sd_ * (H0 / S)
            bx0, by0 = int(round(rx0)), int(round(ry0))
            bw, bh = max(1, int(round(rw))), max(1, int(round(rh)))
            cropD.append(TF(imgs[j].crop((bx0, by0, bx0 + bw, by0 + bh))))
            metaD.append((bx0, by0, bw, bh))
        cropD = torch.stack(cropD).to(device)

        with torch.no_grad():
            for name, ten in [('A_deploy', cropA), ('B_train', cropB), ('C_iso2x', cropC),
                              ('D_distort1x', cropD)]:
                hm = model(ten).float()
                kp = dark_decode(hm, sigma=2.5).cpu().numpy()          # (B,7,2) crop-512 공간
                for j, i in enumerate(range(b0, b1)):
                    if name == 'A_deploy':
                        x0, y0, sd_ = metaA[j]
                        u = (x0 + kp[j, :, 0] / S * sd_) * (W0 / S)     # 512-공간 -> 원본
                        v = (y0 + kp[j, :, 1] / S * sd_) * (H0 / S)
                    elif name == 'B_train':
                        x0, y0, sd_ = metaB[j]
                        u = x0 + kp[j, :, 0] / S * sd_
                        v = y0 + kp[j, :, 1] / S * sd_
                    elif name == 'D_distort1x':
                        x0, y0, bw, bh = metaD[j]
                        u = x0 + kp[j, :, 0] / S * bw
                        v = y0 + kp[j, :, 1] / S * bh
                    else:
                        x0, y0, sd_ = metaC[j]
                        u = (x0 + kp[j, :, 0] / S * sd_) / s_iso
                        v = (y0 + kp[j, :, 1] / S * sd_) / s_iso
                    pred[name][i] = np.stack([u, v], 1)

    print()
    print('=' * 88)
    print('crop detector 2D 오차 (ORACLE bbox, 원본 640x480 프레임 px)')
    print('=' * 88)
    bbd = D['bbox_diag_px']
    res = {}
    hdr = f'{"chain":12s}{"exc.med":>10s}{"med.med":>10s}{"med.p90":>10s}{"clean.med":>11s}{"clean.p90":>11s}'
    print(hdr)
    for c in chains:
        e = np.linalg.norm(pred[c] - gt2d, axis=2)      # (N,7)
        pf = e.mean(1)                                   # per-frame mean over 7 kp
        res[c] = pf
        print(f'{c:12s}{np.median(pf[exc]):10.2f}{np.median(pf[med]):10.2f}'
              f'{np.percentile(pf[med],90):10.2f}{np.median(pf[clean]):11.2f}'
              f'{np.percentile(pf[clean],90):11.2f}')
    print()
    print('bbox-diag 사분위별 clean-frame 2D 오차 median (px)')
    q = np.percentile(bbd[clean], [25, 50, 75])
    print(f'{"chain":12s}' + ''.join(f'{s:>12s}' for s in
          [f'<{q[0]:.0f}', f'{q[0]:.0f}-{q[1]:.0f}', f'{q[1]:.0f}-{q[2]:.0f}', f'>{q[2]:.0f}']))
    for c in chains:
        row = f'{c:12s}'
        for lo, hi in [(0, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 1e9)]:
            m = clean & (bbd >= lo) & (bbd < hi)
            row += f'{np.median(res[c][m]):12.2f}'
        print(row)

    print()
    print(f'배포(A) 대비 학습체인(B) 개선: mediocre median '
          f'{np.median(res["A_deploy"][med]):.2f} -> {np.median(res["B_train"][med]):.2f} px '
          f'({100*(1-np.median(res["B_train"][med])/np.median(res["A_deploy"][med])):.0f}% 감소)')
    print(f'  왜곡 제거만(C): {np.median(res["C_iso2x"][med]):.2f} px')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, fid=np.array(fids), gt2d=gt2d,
             **{f'pred_{c}': pred[c] for c in chains},
             **{f'e_{c}': res[c] for c in chains})
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
