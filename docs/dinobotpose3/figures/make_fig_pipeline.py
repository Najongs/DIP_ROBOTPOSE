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
    """Two-pass topology, matching Eval/selfbbox_eval.py.

    Corrections vs. the earlier draft (verified against source):
      * pass 1 and pass 2 are SEPARATE weight sets (selfbbox_eval.py:165-169,
        verify_sota.sh:9-12) -- not one network re-entered, so both are drawn.
      * angle and rot heads both branch off the keypoint head's soft-argmax
        output (model_angle.py:296,320-323,341-344); they are siblings, not a
        keypoint->angle chain and not parallel branches off the backbone.
      * DARK / heatmap-covariance re-decode the heatmaps and feed the SOLVER
        only (selfbbox_eval.py:273-285); the heads never see them.
      * pass 1 runs a full kinematic solve, then FK-projects all 7 joints
        (incl. the occluded base) to build the box (selfbbox_eval.py:212-236).
    """
    fig, ax = plt.subplots(figsize=(15.0, 9.2))
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 920)
    ax.set_aspect("equal")
    ax.axis("off")

    # ======================= PASS 1 : full frame =========================
    ax.add_patch(FancyBboxPatch((150, 758), 980, 148,
                                boxstyle="round,pad=0,rounding_size=10",
                                fc="none", ec=C_TRAIN, lw=1.4,
                                linestyle=(0, (5, 3)), zorder=2))
    ax.text(164, 888, "PASS 1 — full frame   (full-frame-trained weights)",
            ha="left", va="center", fontsize=9.5, fontweight="bold",
            color=C_TRAIN, zorder=4)

    P1, P1H = 778, 72
    p1mid = P1 + P1H / 2

    img_in = plt.imread(FRAME)
    inset(ax, img_in, 22, 772, 118, 89, "input RGB frame")
    arrow(ax, 143, p1mid, 252, p1mid)

    box(ax, 255, P1, 135, P1H, F_FROZEN, C_FROZEN, "DINOv3 ViT-B/16",
        "~86M, frozen during\nhead training", tfs=8.4, sfs=6.3)
    arrow(ax, 391, p1mid, 408, p1mid)
    box(ax, 410, P1, 125, P1H, F_TRAIN, C_TRAIN, "Keypoint head",
        "7 heatmaps\n$\\rightarrow$ soft-argmax", tfs=8.4, sfs=6.3)
    arrow(ax, 536, p1mid, 553, p1mid)
    box(ax, 555, P1, 140, P1H, F_TRAIN, C_TRAIN, "Angle + Rot heads",
        "$\\hat{\\theta}$,  $R_{init}$", tfs=8.4, sfs=6.6)
    arrow(ax, 696, p1mid, 713, p1mid)
    box(ax, 715, P1, 150, P1H, F_FREE, C_FREE, "kinematic solve",
        "$\\theta, R, t$  (full frame)", tfs=8.4, sfs=6.3)
    arrow(ax, 866, p1mid, 883, p1mid)
    box(ax, 885, P1, 165, P1H, F_FREE, C_FREE, "FK-project 7 pts",
        "$\\rightarrow$ square bbox\n(occluded base filled)", tfs=8.4, sfs=6.3)

    # pass 1 -> crop (clear channel between the two bands)
    elbow(ax, [(1052, p1mid), (1090, p1mid), (1090, 730), (195, 730), (195, 684)],
          ls="--", lw=1.4, color=C_TRAIN)
    ax.text(660, 743, "solved-pose bbox  (guard: fall back to detected-kp bbox "
                      "if the pass-1 solve diverged)",
            ha="center", va="center", fontsize=6.8, color=C_TRAIN, zorder=4)

    # ======================= PASS 2 : crop ===============================
    ax.add_patch(FancyBboxPatch((90, 400), 1120, 312,
                                boxstyle="round,pad=0,rounding_size=10",
                                fc="none", ec=C_TRAIN, lw=1.4,
                                linestyle=(0, (5, 3)), zorder=2))
    ax.text(340, 698, "PASS 2 — crop   (crop-trained weights; angle + rot heads "
                      "self-trained per camera)",
            ha="left", va="center", fontsize=9.5, fontweight="bold",
            color=C_TRAIN, zorder=4)

    box(ax, 110, 628, 170, 52, F_FREE, C_FREE, "roi_align crop",
        "$K \\rightarrow$ crop-$K$ (focal, pp)", tfs=8.4, sfs=6.3)
    arrow(ax, 195, 626, 195, 620)

    box(ax, 110, 528, 170, 88, F_FROZEN, C_FROZEN, "DINOv3 ViT-B/16",
        "separate weights\npatch tokens 32$\\times$32$\\times$768", tfs=8.4, sfs=6.3)
    arrow(ax, 281, 572, 308, 572)
    box(ax, 310, 528, 140, 88, F_TRAIN, C_TRAIN, "Keypoint head",
        "7 heatmaps", tfs=8.4, sfs=6.6)
    arrow(ax, 451, 572, 478, 572)
    box(ax, 480, 542, 112, 60, F_NEUT, C_EDGE, "soft-argmax",
        "kp2d, conf", tfs=8.2, sfs=6.5)

    # both heads branch off the decoded keypoints (siblings, not a chain)
    box(ax, 632, 590, 150, 58, F_TRAIN, C_TRAIN, "Angle head",
        "$\\hat{\\theta}$ 6 joints (sin/cos)", tfs=8.4, sfs=6.4)
    box(ax, 632, 500, 150, 58, F_TRAIN, C_TRAIN, "Rotation head",
        "6D rot $\\rightarrow$ $R_{init}$", tfs=8.4, sfs=6.4)
    arrow(ax, 593, 580, 630, 615)
    arrow(ax, 593, 564, 630, 533)
    ax.text(707, 574, "tokens @ kp2d  +  bearing geom.",
            ha="center", va="center", fontsize=6.2, color="#666666", zorder=4)

    # DARK / covariance: heatmaps -> solver, bypassing the heads
    box(ax, 330, 420, 250, 54, F_FREE, C_FREE, "DARK re-decode  +  heatmap cov",
        "sub-pixel kp2d, 2$\\times$2 $\\Sigma^{-1}$   (solver only)",
        tfs=8.2, sfs=6.4)
    arrow(ax, 380, 526, 380, 476)
    ax.text(390, 500, "heatmaps", ha="left", va="center", fontsize=6.6,
            color=C_FREE, zorder=4)

    # solver
    box(ax, 850, 500, 185, 118, F_FREE, C_FREE, "cov-PnP + kinematic\nrefinement",
        "EPnP init, $R_{init}$ overrides\ndiff. FK + IRLS, 250 iters",
        tfs=8.8, sfs=6.3)
    arrow(ax, 784, 619, 848, 600)
    arrow(ax, 784, 529, 848, 552)
    elbow(ax, [(582, 447), (895, 447), (895, 498)], lw=1.4, color=C_FREE)

    box(ax, 1065, 528, 120, 88, F_NEUT, C_EDGE, "Output",
        "joint angles $\\theta$\ncamera pose $R, t$", tfs=8.8, sfs=6.6)
    arrow(ax, 1037, 572, 1063, 572)

    img_mesh = crop_panel(os.path.join(QUAL, "qual_realsense_mesh.png"), 1, 0)
    mh = 155 * img_mesh.shape[0] / img_mesh.shape[1]
    inset(ax, img_mesh, 1250, 572 - mh / 2, 155, mh, "predicted mesh overlay")
    arrow(ax, 1212, 572, 1248, 572)

    # ============ bottom band: test-time render-and-compare ==============
    ax.add_patch(FancyBboxPatch((430, 40), 750, 340,
                                boxstyle="round,pad=0,rounding_size=10",
                                fc="none", ec=C_FREE, lw=1.4,
                                linestyle=(0, (5, 3)), zorder=2))
    ax.text(444, 362, "Test-time render-and-compare  (depth/scale corrector, training-free)",
            ha="left", va="center", fontsize=9.5, fontweight="bold",
            color=C_FREE, zorder=4)

    box(ax, 480, 238, 150, 62, F_FREE, C_FREE, "Zero-shot SAM",
        "ViT-B\nrobot foreground mask", tfs=8.6)
    box(ax, 480, 128, 150, 62, F_FREE, C_FREE, "nvdiffrast render",
        "mesh + FK silhouette", tfs=8.6)

    px, py, pw_, ph_ = 678, 132, 130, 160
    ax.add_patch(plt.Rectangle((px, py), pw_, ph_, fc="#1a1a1a", ec=C_EDGE, lw=1.0, zorder=3))
    ax.add_patch(arm_poly(px, py, pw_, ph_, dx=0.05, dy=0.02, fc="white",
                          ec="none", alpha=0.85, zorder=4))
    ax.add_patch(arm_poly(px, py, pw_, ph_, dx=-0.05, dy=-0.02, fc=C_FROZEN,
                          ec="none", alpha=0.65, zorder=5))
    ax.text(px + pw_ / 2, py - 13, "SAM mask vs rendered silhouette",
            ha="center", va="center", fontsize=7.2, color="#444444", zorder=4)
    arrow(ax, 632, 269, 676, 248, color=C_FREE)
    arrow(ax, 632, 159, 676, 186, color=C_FREE)

    box(ax, 850, 178, 160, 70, F_FREE, C_FREE, "maximize soft-IoU",
        "optimize depth/scale\n(+ reproj anchor)", tfs=8.6)
    arrow(ax, 810, 213, 848, 213, color=C_FREE)

    ax.text(805, 62, "per-camera toggle:  RealSense +0.070 $\\cdot$ Kinect +0.062 $\\cdot$ "
                     "ORB +0.040 $\\cdot$ Azure OFF (near-range)",
            ha="center", va="center", fontsize=7.6, color=C_FREE, zorder=4)

    # RC wiring
    elbow(ax, [(14, 790), (14, 269), (478, 269)], ls="--", lw=1.2, color="#777777")
    ax.text(250, 278, "full image (uncropped)", ha="center", fontsize=7.2,
            color="#777777", zorder=4)
    elbow(ax, [(1020, 498), (1020, 330), (446, 330), (446, 159), (478, 159)],
          lw=1.4, color=C_FREE)
    ax.text(760, 340, "initial pose $(\\theta, R, t)$", ha="center", fontsize=7.4,
            color=C_FREE, zorder=4)
    elbow(ax, [(1012, 213), (1125, 213), (1125, 526)], lw=1.4, color=C_FREE)
    ax.text(1148, 320, "corrected $t$", ha="left", fontsize=7.4, color=C_FREE, zorder=4)

    # --- legend ---
    lx, ly = 40, 205
    items = [(F_FROZEN, C_FROZEN, "Frozen DINOv3 backbone (one per pass,\nseparate weights)"),
             (F_TRAIN, C_TRAIN, "Trained (sim-to-real; pass-2 angle + rot\nheads self-trained per camera)"),
             (F_FREE, C_FREE, "Training-free / test-time")]
    ax.text(lx, ly + 46, "Legend", fontsize=8.6, fontweight="bold", color=C_EDGE)
    for i, (fc, ec, lab) in enumerate(items):
        yy = ly - i * 56
        ax.add_patch(plt.Rectangle((lx, yy), 34, 20, fc=fc, ec=ec, lw=1.4, zorder=3))
        ax.text(lx + 44, yy + 10, lab, ha="left", va="center", fontsize=7.6,
                color="#333333", zorder=4)

    ax.text(40, 32, "Pass 1 and pass 2 share the architecture but NOT the\n"
                    "weights; the only quantity carried across is the bbox.",
            ha="left", va="center", fontsize=7.6, style="italic",
            color="#555555", zorder=4)

    save(fig, "fig_pipeline")


if __name__ == "__main__":
    fig_pipeline()
