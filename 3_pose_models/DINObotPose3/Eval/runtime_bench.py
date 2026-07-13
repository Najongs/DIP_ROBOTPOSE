#!/usr/bin/env python3
"""I1 runtime micro-benchmark: per-stage latency of the DINObotPose3 pipeline.
Times the dominant compute (frozen DINOv3 ViT-B/16 forward at 512, done 2x: full-frame + crop)
and the kinematic solver. RC-stage throughput is taken from deployed-eval logs (SAM+nvdiffrast).
Warm GPU, torch.cuda.synchronize, batch=1 latency + batch=16 throughput."""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../TRAIN')); sys.path.insert(0, '.')
from model_v4 import DINOv3Backbone, ViTKeypointHead, panda_forward_kinematics
dev = 'cuda'

bb = DINOv3Backbone('facebook/dinov3-vitb16-pretrain-lvd1689m', unfreeze_blocks=0).to(dev).eval()
kp = ViTKeypointHead(input_dim=bb.model.config.hidden_size, heatmap_size=(512, 512)).to(dev).eval()

def timeit(fn, n=30, warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000  # ms

for B in (1, 16):
    img = torch.randn(B, 3, 512, 512, device=dev)
    with torch.no_grad():
        t_bb = timeit(lambda: bb(img))
        tok = bb(img)
        t_kp = timeit(lambda: kp(tok))
    fps_bb = B / (t_bb / 1000)
    print(f"[batch={B:2d}] DINOv3 backbone fwd: {t_bb:7.1f} ms  ({fps_bb:5.1f} img/s)  | keypoint head: {t_kp:6.1f} ms")

# solver (250 iters) latency — dummy 7-kp
import numpy as np
sys.path.insert(0, '.')
import solve_pose_kinematic as spk
B = 16
kp2d = torch.rand(B, 7, 2, device=dev) * 400 + 50
conf = torch.rand(B, 7, device=dev) * 0.5 + 0.5
K = torch.tensor([[500., 0, 256], [0, 500, 256], [0, 0, 1]], device=dev).unsqueeze(0).repeat(B, 1, 1)
def solve():
    with torch.enable_grad():
        spk.solve_batch(kp2d, conf, K, iters=250, device=dev)
t_solve = timeit(solve, n=5, warm=2)
print(f"[batch={B}] kinematic solver (250 iters): {t_solve:7.1f} ms/batch  ({t_solve/B:6.1f} ms/frame)")

print("\n=== deployed pipeline throughput (from eval logs, RTX 3090) ===")
print("  base (2x backbone + decode + solver):  ~0.42 s/frame  (~2.4 frames/s)")
print("  + render-and-compare (SAM + nvdiffrast opt): ~1.3 s/frame added  (~0.6 fps end-to-end)")
print("  azure ships base-only (RC off) -> ~2.4 fps; RC is per-camera & accuracy-focused, not real-time.")
