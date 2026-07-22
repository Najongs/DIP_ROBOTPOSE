#!/usr/bin/env python
"""freeze-theta + solve-6DOF ADD-AUC curve vs joint-angle MAE (2026-07-22, 8-point 2-seed sweep).

Data source (all measured, not invented):
  Eval/ablation_logs/freeze_curve/mae*_s*.log  (FULL AUC printed per file, verified),
  docs/dinobotpose3/experiments/2026-07-22_gap_reexamination.md  §21.6 (CLEAN good-frame column),
  §21.5 (deployed joint-solve good-frame CLEAN = 0.7884).
Plots the good-frame (CLEAN) freeze-6DOF curve so that both the curve and the
deployed joint-solve reference line are on the same good-frame subset.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# palette (matches make_figs.py)
C_OURS = "#1f4e79"   # deep blue  - freeze curve
C_PEPP = "#e07b39"   # orange     - RoboPEPP-level angle marker
C_ACC  = "#2e7d32"   # green      - deployed joint-solve reference

# --- measured data (gap_reexamination.md §21.6, good-frame CLEAN) ---
MAE   = np.array([0.0, 1.5, 2.4, 3.0, 3.8, 5.0, 6.0, 7.47])
CLEAN = np.array([0.887, 0.808, 0.737, 0.689, 0.631, 0.548, 0.494, 0.422])

JOINT_SOLVE = 0.7884   # deployed joint-solve, good-frame CLEAN (§21.5)
PEPP_MAE, PEPP_ADD = 3.8, 0.631   # RoboPEPP-level angle accuracy (Table 1 / §21.6 row)

# crossover where the freeze curve meets the deployed joint-solve line
i = np.searchsorted(-CLEAN, -JOINT_SOLVE)  # CLEAN is descending
x0, x1, y0, y1 = MAE[i - 1], MAE[i], CLEAN[i - 1], CLEAN[i]
cross = x0 + (y0 - JOINT_SOLVE) * (x1 - x0) / (y0 - y1)

fig, ax = plt.subplots(figsize=(6.6, 4.2))

# deployed joint-solve reference line
ax.axhline(JOINT_SOLVE, color=C_ACC, lw=1.4, ls=(0, (5, 3)), zorder=2)
ax.text(7.35, JOINT_SOLVE + 0.006, f"deployed joint-solve  {JOINT_SOLVE:.3f}",
        color=C_ACC, ha="right", va="bottom", fontsize=9.2)

# freeze curve
ax.plot(MAE, CLEAN, "-o", color=C_OURS, lw=1.8, ms=6, mec="white", mew=0.8,
        zorder=4, label="freeze-$\\theta$ + solve-6DoF")

# crossover marker
ax.plot([cross], [JOINT_SOLVE], marker="D", color=C_ACC, ms=8, mec="white",
        mew=0.9, zorder=6)
ax.annotate(f"crossover $\\approx${cross:.1f}$^\\circ$",
            xy=(cross, JOINT_SOLVE), xytext=(cross + 1.35, 0.845),
            fontsize=9.2, color=C_ACC, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=C_ACC, lw=1.0,
                            connectionstyle="arc3,rad=0.2"))

# RoboPEPP-level angle marker
ax.plot([PEPP_MAE], [PEPP_ADD], marker="s", color=C_PEPP, ms=9, mec="white",
        mew=0.9, zorder=6)
ax.annotate(f"RoboPEPP-level angle\n{PEPP_MAE:.1f}$^\\circ$  $\\to$  {PEPP_ADD:.3f}",
            xy=(PEPP_MAE, PEPP_ADD), xytext=(2.15, 0.505),
            fontsize=8.8, color=C_PEPP, ha="left", va="center", linespacing=1.25,
            arrowprops=dict(arrowstyle="->", color=C_PEPP, lw=1.0,
                            connectionstyle="arc3,rad=0.2"))

ax.set_xlabel("feed-forward joint-angle MAE  (degrees)")
ax.set_ylabel("freeze-6DoF ADD-AUC @100 mm")
ax.set_xlim(-0.25, 7.9)
ax.set_ylim(0.38, 0.92)
ax.grid(True, which="major", color="#dddddd", lw=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper right", frameon=False, fontsize=9.5)

for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_freeze_curve.{ext}")
plt.close(fig)
print(f"crossover(joint-solve {JOINT_SOLVE}) = {cross:.3f} deg")
print("wrote fig_freeze_curve.pdf / .png")
