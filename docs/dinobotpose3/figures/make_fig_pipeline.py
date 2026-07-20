#!/usr/bin/env python
"""Pipeline overview figure for PAPER_DRAFT §3.1 (fig_pipeline, unnumbered).

Reference draft for the final PPT version: block diagram of the five-stage
DINObotPose3 pipeline with real-image insets (input frame / decoded keypoints /
mesh overlay) and schematic insets (heatmaps, SAM-vs-render IoU).
Sources: FINAL_MODEL.md, architecture/model.md, PAPER_DRAFT §3.1-3.3.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

OUT = "/home/najo/NAS/DIP/docs/dinobotpose3/figures"
QUAL = os.path.join(OUT, "qualitative")
FRAME = ("/home/najo/NAS/DIP/datasets/ICRA_multiview/DREAM_real/"
         "panda-3cam_realsense/panda-3cam_realsense/002700.rgb.jpg")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---- palette (same as make_figs.py) ----
C_FROZEN = "#1f4e79"   # deep blue  = frozen backbone
C_TRAIN  = "#e07b39"   # orange     = trained modules
C_FREE   = "#2e7d32"   # green      = training-free / test-time
F_FROZEN = "#dce7f3"
F_TRAIN  = "#fbe8d8"
F_FREE   = "#e2efe3"
C_EDGE   = "#333333"
F_NEUT   = "#f2f2f2"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print(f"wrote {name}.png / .pdf")


# ---------- helpers ----------

def box(ax, x, y, w, h, fc, ec, title, sub=None, tfs=9.0, sfs=6.8, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    if sub:
        tl = title.count("\n") + 1
        sl = sub.count("\n") + 1
        th, sh, gap = 12.5 * tl, 9.5 * sl, 5
        top = y + h - max((h - (th + sh + gap)) / 2, 4)
        ax.text(x + w / 2, top - th / 2, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color=C_EDGE, zorder=4)
        ax.text(x + w / 2, top - th - gap - sh / 2, sub, ha="center", va="center",
                fontsize=sfs, color="#444444", zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color=C_EDGE, zorder=4)


def arrow(ax, x0, y0, x1, y1, color=C_EDGE, lw=1.6, ls="-", zorder=2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=zorder,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                linestyle=ls, shrinkA=1, shrinkB=1,
                                mutation_scale=14))


def elbow(ax, pts, color=C_EDGE, lw=1.6, ls="-", zorder=2):
    """Orthogonal polyline with an arrowhead on the last segment."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=zorder,
            solid_capstyle="round")
    arrow(ax, xs[-2], ys[-2], xs[-1], ys[-1], color=color, lw=lw, ls=ls,
          zorder=zorder)


def inset(ax, img, x, y, w, h, caption=None, cap_below=True, ec=C_EDGE):
    ax.imshow(img, extent=(x, x + w, y, y + h), zorder=3, interpolation="bilinear")
    ax.add_patch(plt.Rectangle((x, y), w, h, fc="none", ec=ec, lw=1.2, zorder=4))
    if caption:
        cy = y - 11 if cap_below else y + h + 11
        ax.text(x + w / 2, cy, caption, ha="center", va="center",
                fontsize=7.4, color="#444444", zorder=4)


def crop_panel(path, row, col):
    """Crop one panel out of a 2x3 qualitative grid (1152x616, header strips)."""
    im = plt.imread(path)
    ph, pw = 308, 384
    y0, x0 = row * ph, col * pw
    return im[y0 + 24:y0 + ph - 2, x0 + 3:x0 + pw - 3]


def heatmap_schematic(w=120, h=90):
    yy, xx = np.mgrid[0:h, 0:w]
    pts = [(20, 74), (34, 56), (46, 42), (60, 36), (74, 28), (86, 36), (98, 46)]
    hm = np.zeros((h, w))
    for px, py in pts:
        hm += np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * 6.0 ** 2)))
    return hm


ARM = np.array([  # schematic robot-arm silhouette, unit coords
    (0.40, 0.03), (0.62, 0.03), (0.58, 0.30), (0.74, 0.50), (0.68, 0.62),
    (0.54, 0.48), (0.50, 0.70), (0.60, 0.84), (0.52, 0.95), (0.38, 0.82),
    (0.42, 0.55), (0.38, 0.28)])


def arm_poly(x, y, w, h, dx=0.0, dy=0.0, **kw):
    v = ARM.copy()
    v[:, 0] = x + (v[:, 0] + dx) * w
    v[:, 1] = y + (v[:, 1] + dy) * h
    return Polygon(v, closed=True, **kw)


# ---------- figure ----------

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.set_xlim(0, 1350)
    ax.set_ylim(0, 660)
    ax.set_aspect("equal")
    ax.axis("off")

    Y0, Y1 = 420, 510          # main-flow row
    ymid = (Y0 + Y1) / 2

    # --- stage 1: input frame (real inset) ---
    img_in = plt.imread(FRAME)
    inset(ax, img_in, 20, 415, 140, 105, "input RGB frame")
    arrow(ax, 162, ymid, 183, ymid)

    # --- auto-bbox loop ---
    box(ax, 185, Y0, 122, Y1 - Y0, F_TRAIN, C_TRAIN, "Auto-bbox",
        "bbox-from-solved\ndetect $\\rightarrow$ solve $\\rightarrow$ crop")
    arrow(ax, 309, ymid, 320, ymid)

    # --- frozen backbone ---
    box(ax, 322, Y0, 160, Y1 - Y0, F_FROZEN, C_FROZEN, "Frozen DINOv3\nViT-B/16",
        "~86M, frozen throughout\npatch tokens 32$\\times$32$\\times$768", sfs=6.4)
    arrow(ax, 484, ymid, 495, ymid)

    # --- keypoint stage: trained head (top) + free decode (bottom) ---
    box(ax, 497, 462, 150, 48, F_TRAIN, C_TRAIN, "Keypoint head",
        "7 keypoint heatmaps")
    box(ax, 497, 412, 150, 46, F_FREE, C_FREE, "DARK decode",
        "sub-pixel, conf, 2$\\times$2 cov", sfs=6.5)

    # --- heads (trained) ---
    box(ax, 672, 470, 150, 50, F_TRAIN, C_TRAIN, "Angle head",
        "$\\hat{\\theta}$ 6 joints (sin/cos)")
    box(ax, 672, 402, 150, 50, F_TRAIN, C_TRAIN, "Rotation head",
        "6D rot $\\rightarrow$ $R_{init}$")
    arrow(ax, 649, 486, 670, 495)
    arrow(ax, 649, 435, 670, 427)

    # --- solver (training-free) ---
    box(ax, 845, Y0, 195, Y1 - Y0, F_FREE, C_FREE, "cov-PnP + kinematic\nrefinement",
        "EPnP init (top-conf kp)\ndiff. FK + IRLS, 250 iters")
    arrow(ax, 824, 495, 843, 478)
    arrow(ax, 824, 427, 843, 450)

    # --- output ---
    box(ax, 1052, Y0, 110, Y1 - Y0, F_NEUT, C_EDGE, "Output",
        "joint angles $\\theta$\ncamera pose $R, t$")
    arrow(ax, 1042, ymid, 1050, ymid)
    img_mesh = crop_panel(os.path.join(QUAL, "qual_realsense_mesh.png"), 1, 0)
    mh = 150 * img_mesh.shape[0] / img_mesh.shape[1]
    inset(ax, img_mesh, 1188, ymid - mh / 2, 150, mh, "predicted mesh overlay")
    arrow(ax, 1164, ymid, 1186, ymid)

    # --- top band: heatmap schematic + decoded keypoints (real) ---
    hm = heatmap_schematic()
    ax.imshow(hm, extent=(512, 632, 555, 645), cmap="inferno", zorder=3)
    ax.add_patch(plt.Rectangle((512, 555), 120, 90, fc="none", ec=C_EDGE, lw=1.0, zorder=4))
    ax.text(572, 651, "heatmaps (schematic)", ha="center", va="bottom",
            fontsize=7.2, color="#666666", zorder=4)
    ax.plot([572, 572], [512, 553], color="#999999", lw=1.0, ls=":", zorder=2)

    img_kp = crop_panel(os.path.join(QUAL, "qual_realsense_clean.png"), 1, 0)
    kw_ = 132
    kh_ = kw_ * img_kp.shape[0] / img_kp.shape[1]
    inset(ax, img_kp, 690, 645 - kh_, kw_, kh_, None)
    ax.text(756, 651, "decoded 2D keypoints", ha="center", va="bottom",
            fontsize=7.2, color="#666666", zorder=4)
    ax.plot([747, 747], [522, 645 - kh_ - 2], color="#999999", lw=1.0, ls=":", zorder=2)

    # --- bottom band: test-time render-and-compare ---
    ax.add_patch(FancyBboxPatch((470, 60), 700, 290, boxstyle="round,pad=0,rounding_size=10",
                                fc="none", ec=C_FREE, lw=1.4, linestyle=(0, (5, 3)), zorder=2))
    ax.text(482, 334, "Test-time render-and-compare  (depth/scale corrector, training-free)",
            ha="left", va="center", fontsize=9.5, fontweight="bold", color=C_FREE, zorder=4)

    box(ax, 520, 210, 150, 62, F_FREE, C_FREE, "Zero-shot SAM",
        "ViT-B\nrobot foreground mask", tfs=8.6)
    box(ax, 520, 105, 150, 62, F_FREE, C_FREE, "nvdiffrast render",
        "mesh + FK silhouette", tfs=8.6)

    # IoU panel: two offset silhouettes
    px, py, pw_, ph_ = 715, 115, 130, 160
    ax.add_patch(plt.Rectangle((px, py), pw_, ph_, fc="#1a1a1a", ec=C_EDGE, lw=1.0, zorder=3))
    ax.add_patch(arm_poly(px, py, pw_, ph_, dx=0.05, dy=0.02, fc="white",
                          ec="none", alpha=0.85, zorder=4))
    ax.add_patch(arm_poly(px, py, pw_, ph_, dx=-0.05, dy=-0.02, fc=C_FROZEN,
                          ec="none", alpha=0.65, zorder=5))
    ax.text(px + pw_ / 2, py - 13, "SAM mask vs rendered silhouette",
            ha="center", va="center", fontsize=7.2, color="#444444", zorder=4)
    arrow(ax, 672, 241, 713, 220, color=C_FREE)
    arrow(ax, 672, 136, 713, 165, color=C_FREE)

    box(ax, 875, 155, 160, 70, F_FREE, C_FREE, "maximize soft-IoU",
        "optimize depth/scale\n(+ reproj anchor)", tfs=8.6)
    arrow(ax, 847, 190, 873, 190, color=C_FREE)

    # per-camera toggle note
    ax.text(820, 76, "per-camera toggle:  RealSense +0.070 $\\cdot$ Kinect +0.062 $\\cdot$ "
                     "ORB +0.040 $\\cdot$ Azure OFF (near-range)",
            ha="center", va="center", fontsize=7.6, color=C_FREE, zorder=4)

    # pass-1 feedback loop: solved pose -> FK-projected bbox -> re-crop (pass 2)
    elbow(ax, [(880, 418), (880, 370), (246, 370), (246, 418)],
          ls="--", lw=1.3, color=C_TRAIN)
    ax.text(563, 384, "pass-1 solved pose $\\rightarrow$ FK-projected bbox "
                      "(re-detect on crop = pass 2)",
            ha="center", va="center", fontsize=6.8, color=C_TRAIN, zorder=4)

    # branch arrows in/out of RC
    elbow(ax, [(30, 413), (30, 241), (518, 241)], ls="--", lw=1.2, color="#777777")
    ax.text(300, 250, "full image", ha="center", fontsize=7.2, color="#777777", zorder=4)
    elbow(ax, [(942, 418), (942, 300), (495, 300), (495, 136), (517, 136)],
          lw=1.4, color=C_FREE)
    ax.text(770, 310, "initial pose $(\\theta, R, t)$", ha="center", fontsize=7.4,
            color=C_FREE, zorder=4)
    elbow(ax, [(1037, 190), (1107, 190), (1107, 418)], lw=1.4, color=C_FREE)
    ax.text(1118, 250, "corrected $t$", ha="left", fontsize=7.4, color=C_FREE, zorder=4)

    # --- legend ---
    lx, ly = 30, 175
    items = [(F_FROZEN, C_FROZEN, "Frozen (DINOv3 backbone)"),
             (F_TRAIN, C_TRAIN, "Trained (sim-to-real + per-camera self-training,\nlight occlusion aug)"),
             (F_FREE, C_FREE, "Training-free / test-time")]
    ax.text(lx, ly + 40, "Legend", fontsize=8.6, fontweight="bold", color=C_EDGE)
    for i, (fc, ec, lab) in enumerate(items):
        yy = ly - i * 52
        ax.add_patch(plt.Rectangle((lx, yy), 34, 20, fc=fc, ec=ec, lw=1.4, zorder=3))
        ax.text(lx + 44, yy + 10, lab, ha="left", va="center", fontsize=7.8,
                color="#333333", zorder=4)

    save(fig, "fig_pipeline")


if __name__ == "__main__":
    fig_pipeline()
