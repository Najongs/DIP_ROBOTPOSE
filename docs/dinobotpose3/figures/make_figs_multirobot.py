#!/usr/bin/env python
"""Extra publication figures for the multi-robot (§4.7) part of the paper:
  fig7_multirobot     — DREAM pose across the 3 robots (real Panda vs synthetic KUKA/Baxter)
  fig8_wrist_observ   — Baxter per-joint MAE, detected-2D vs GT-2D injection (observability ceiling)
Matches make_figs.py style."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/najo/NAS/DIP/docs/dinobotpose3/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
    "axes.linewidth": 0.9, "axes.edgecolor": "#333333",
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_REAL = "#1f4e79"   # deep blue (real, headline)
C_SYN  = "#e07b39"   # orange (synthetic)
C_GT   = "#2e7d32"   # green (GT-2D injected)
C_DET  = "#8a8d91"   # gray


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    print(f"wrote {name}.png / .pdf")


# ---------------------------------------------------------------- fig7: 3-robot pose
def fig7():
    robots = ["Panda", "KUKA iiwa7", "Baxter"]
    pose = [0.804, 0.6901, 0.7125]               # ADD-AUC@100mm (KUKA/Baxter: solver + true K, 2026-07-22)
    det  = [None, 0.735, 0.817]                # detector 2D keypoint AUC (synth); Panda n/a here
    cols = [C_REAL, C_SYN, C_SYN]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(3)
    bars = ax.bar(x, pose, width=0.56, color=cols, edgecolor="#222", linewidth=0.6, zorder=3)
    for xi, p in zip(x, pose):
        ax.text(xi, p + 0.012, f"{p:.3f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold", color="#111")
    for xi, dt in zip(x, det):
        if dt is not None:
            ax.text(xi, 0.03, f"detector\nAUC {dt:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color="#444")
    # real | synthetic divider
    ax.axvline(0.5, color="#999", ls="--", lw=1.1)
    ax.text(0.0, 0.94, "REAL\n(headline SOTA)", ha="center", va="center", fontsize=9,
            color=C_REAL, transform=ax.get_xaxis_transform(), fontweight="bold")
    ax.text(1.5, 0.94, "SYNTHETIC-ONLY  (no render-compare)\ndifferent regime — not a robot-vs-robot ranking",
            ha="center", va="center", fontsize=9, color=C_SYN,
            transform=ax.get_xaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels(robots, fontsize=12)
    ax.set_ylabel("pose ADD-AUC@100 mm  (↑)")
    ax.set_ylim(0, 1.05); ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#eee", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("DREAM pose across robots — same pipeline, 3 robots\n"
                 "(DREAM has real test data only for Panda)", fontsize=11.5)
    save(fig, "fig7_multirobot"); plt.close(fig)


# ---------------------------------------------------------------- fig8: wrist observability
def fig8():
    joints = ["s0", "s1", "e0", "e1", "w0", "w1"]
    det = [6.65, 4.53, 10.41, 6.99, 28.11, 22.84]     # detected 2D
    gt  = [6.70, 4.33, 10.35, 6.51, 27.59, 22.19]     # GT 2D injected
    x = np.arange(len(joints)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    # shade wrist region
    ax.axvspan(3.5, 5.5, color="#fbeee0", zorder=0)
    ax.text(4.5, 40.5, "wrist (w0,w1)\nunobservable from keypoints", ha="center", va="top",
            fontsize=9, color=C_SYN)
    ax.bar(x - w/2, det, w, color=C_DET, edgecolor="#222", lw=0.5, label="detected 2D", zorder=3)
    ax.bar(x + w/2, gt,  w, color=C_GT,  edgecolor="#222", lw=0.5, label="GT 2D injected", zorder=3)
    for xi, d, g in zip(x, det, gt):
        ax.text(xi + w/2, g + 0.6, f"{g:.0f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(joints, fontsize=12)
    ax.set_ylabel("joint MAE (deg)  (↓)"); ax.set_xlabel("Baxter left-arm joint")
    ax.set_ylim(0, 44); ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#eee", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title("Baxter joint MAE — perfect (GT) keypoints barely change the wrist\n"
                 "→ wrist is an observability ceiling, not a detection failure", fontsize=11.5)
    save(fig, "fig8_wrist_observability"); plt.close(fig)


if __name__ == "__main__":
    fig7(); fig8()
    print("MULTIROBOT FIGURES DONE ->", OUT)
