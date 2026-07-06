#!/usr/bin/env python
"""Publication figures for DINObotPose3 DREAM SOTA (2026-07-06 1000-frame re-lock)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = "/home/najo/NAS/DIP/docs/dinobotpose3/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#333333",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,   # editable text in PDF
    "ps.fonttype": 42,
})

# ---- palette (colorblind-safe) ----
C_OURS = "#1f4e79"     # deep blue
C_PEPP = "#e07b39"     # orange
C_TAG  = "#8a8d91"     # gray
C_ACC  = "#2e7d32"     # green accent

CAMS = ["realsense", "kinect360", "azure", "orb", "mean"]
OURS = [0.8153, 0.8275, 0.7945, 0.7784, 0.8039]
PEPP = [0.805, 0.785, 0.753, 0.775, 0.780]     # RoboPEPP (published, GT-bbox)
TAG  = [0.783, 0.757, 0.831, 0.588, 0.740]     # RoboTAG (published)
PEPP_ORB_AUTO = 0.344                            # RoboPEPP orb under auto-bbox (collapse)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print(f"wrote {name}.png / .pdf")


# ============ FIG 1: per-camera scorecard (grouped bars) ============
def fig_scorecard():
    x = np.arange(len(CAMS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.4))

    b1 = ax.bar(x - w, OURS, w, label="Ours (DINObotPose3)", color=C_OURS, zorder=3)
    b2 = ax.bar(x,     PEPP, w, label="RoboPEPP", color=C_PEPP, zorder=3)
    b3 = ax.bar(x + w, TAG,  w, label="RoboTAG", color=C_TAG, zorder=3)

    # value labels on ours + mean
    for rect, v in zip(b1, OURS):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.2, color=C_OURS, fontweight="bold")

    # orb auto-bbox collapse marker for RoboPEPP (placed in the empty valley left of orb)
    orb_i = CAMS.index("orb")
    ax.plot([x[orb_i]], [PEPP_ORB_AUTO], marker="v", color=C_PEPP, ms=9,
            mec="white", mew=0.8, zorder=6)
    ax.annotate(f"RoboPEPP\nauto-bbox\n{PEPP_ORB_AUTO:.3f}",
                xy=(x[orb_i] - 0.14, PEPP_ORB_AUTO + 0.005), xytext=(x[orb_i] - 0.52, 0.47),
                fontsize=8.0, color=C_PEPP, ha="center", va="center", linespacing=1.25,
                arrowprops=dict(arrowstyle="->", color=C_PEPP, lw=1.0,
                                connectionstyle="arc3,rad=-0.25"))

    # separator before mean
    ax.axvline(x[-1] - 0.5, color="#bbbbbb", lw=0.8, ls=(0, (4, 3)), zorder=1)

    ax.set_ylabel("ADD-AUC @100 mm  (↑)")
    ax.set_ylim(0.30, 0.88)
    ax.set_xlim(-0.6, len(CAMS) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([c if c != "mean" else "MEAN" for c in CAMS])
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(axis="y", which="both", color="#e8e8e8", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), frameon=False,
              fontsize=9.5, ncol=3, columnspacing=1.6, handlelength=1.4)
    ax.set_title("DREAM real (Franka Panda) — ADD-AUC@100mm, 1000-frame re-lock\n"
                 "ours: predicted angles + automatic bbox   vs   published baselines",
                 fontsize=10.5, pad=8)
    save(fig, "fig1_scorecard")


# ============ FIG 2: occlusion robustness curve ============
def fig_occlusion():
    occ = [0, 10, 20, 30, 40]
    ours = [0.812, 0.765, 0.678, 0.575, 0.429]
    pepp = [0.795, 0.730, 0.600, 0.470, 0.351]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))

    ax.plot(occ, ours, "-o", color=C_OURS, lw=2.2, ms=6.5,
            label="Ours (occ-aug head + RC)", zorder=4)
    ax.plot(occ, pepp, "-s", color=C_PEPP, lw=2.2, ms=6, label="RoboPEPP", zorder=3)
    ax.fill_between(occ, pepp, ours, color=C_OURS, alpha=0.08, zorder=1)

    for xo, yo, yp in zip(occ, ours, pepp):
        ax.text(xo, yo + 0.014, f"{yo:.3f}", ha="center", fontsize=7.6,
                color=C_OURS, fontweight="bold")
        d = yo - yp
        ax.text(xo, yp - 0.028, f"+{d:.3f}", ha="center", fontsize=7.2, color=C_ACC)

    ax.set_xlabel("RoI occlusion  (%)")
    ax.set_ylabel("ADD-AUC @100 mm  (↑)")
    ax.set_xticks(occ)
    ax.set_ylim(0.30, 0.87)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(color="#e8e8e8", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.set_title("Occlusion robustness (RoboPEPP Fig. 6 protocol)\nours beats at every level",
                 fontsize=10.5, pad=8)
    save(fig, "fig2_occlusion")


# ============ FIG 3: 800->1000 re-lock stability ============
def fig_relock():
    cams = ["realsense", "kinect360", "azure", "orb", "MEAN"]
    v800 = [0.8165, 0.8303, 0.7953, 0.7726, 0.8037]
    v1000 = OURS
    x = np.arange(len(cams))
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for xi, a, b in zip(x, v800, v1000):
        ax.plot([xi, xi], [a, b], color="#cccccc", lw=1.4, zorder=1)
    ax.scatter(x, v800, s=46, color=C_TAG, label="held-out 800", zorder=3)
    ax.scatter(x, v1000, s=52, color=C_OURS, label="re-lock 1000", zorder=4)
    for xi, b in zip(x, v1000):
        ax.text(xi, b + 0.004, f"{b:.3f}", ha="center", fontsize=7.6,
                color=C_OURS, fontweight="bold")
    ax.axhline(0.780, color=C_PEPP, lw=1.3, ls="--", zorder=2)
    ax.text(x[0] - 0.35, 0.783, "RoboPEPP mean 0.780", fontsize=8, color=C_PEPP)
    ax.set_ylabel("ADD-AUC @100 mm")
    ax.set_ylim(0.74, 0.845)
    ax.set_xticks(x); ax.set_xticklabels(cams)
    ax.grid(axis="y", color="#eeeeee", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_title("Sample-size robustness: 800→1000 re-lock (Δmean +0.0002)",
                 fontsize=10.5, pad=8)
    save(fig, "fig3_relock")


# ============ FIG 4: rendered results table ============
def fig_table():
    rows = [
        ["realsense", "0.8153", "0.805", "0.783", "+0.010"],
        ["kinect360", "0.8275", "0.785", "0.757", "+0.043"],
        ["azure",     "0.7945", "0.753", "0.831*", "+0.042"],
        ["orb",       "0.7784", "0.775", "0.588", "+0.003"],
        ["MEAN",      "0.8039", "0.780", "0.740", "+0.024"],
    ]
    cols = ["Camera", "Ours", "RoboPEPP", "RoboTAG", "Δ vs PEPP"]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.55)
    n = len(rows)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor(C_OURS); cell.set_text_props(color="white", fontweight="bold")
        elif r == n:  # MEAN row
            cell.set_facecolor("#eef2f7"); cell.set_text_props(fontweight="bold")
        if c == 1 and r > 0:
            cell.set_text_props(color=C_OURS, fontweight="bold")
        if c == 4 and r > 0:
            cell.set_text_props(color=C_ACC, fontweight="bold")
    ax.set_title("DREAM 4-real-split ADD-AUC@100mm — 1000-frame re-lock (2026-07-06)\n"
                 "predicted angles + fully-automatic bbox;  * only cell where a baseline beats ours",
                 fontsize=10, pad=6)
    save(fig, "fig4_table")


# ============ LaTeX table ============
def latex_table():
    tex = r"""% DINObotPose3 DREAM results (1000-frame re-lock, 2026-07-06)
\begin{table}[t]
\centering
\caption{ADD-AUC@100mm on the DREAM real Franka-Panda splits under the
predicted-angle + fully-automatic-bbox protocol. Held-out 1000 frames/camera
(azure full split). Best per row in \textbf{bold}.}
\label{tab:dream}
\begin{tabular}{lcccc}
\toprule
Camera & \textbf{Ours} & RoboPEPP & RoboTAG & $\Delta$ \\
\midrule
realsense & \textbf{0.815} & 0.805 & 0.783 & +0.010 \\
kinect360 & \textbf{0.828} & 0.785 & 0.757 & +0.043 \\
azure     & \textbf{0.795} & 0.753 & 0.831 & +0.042 \\
orb       & \textbf{0.778} & 0.775 & 0.588 & +0.003 \\
\midrule
\textbf{MEAN} & \textbf{0.804} & 0.780 & 0.740 & \textbf{+0.024} \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open(f"{OUT}/table_dream.tex", "w") as f:
        f.write(tex)
    print("wrote table_dream.tex")


fig_scorecard()
fig_occlusion()
fig_relock()
fig_table()
latex_table()
print("ALL FIGURES DONE ->", OUT)
