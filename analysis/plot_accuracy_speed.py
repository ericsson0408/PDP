#!/usr/bin/env python3
"""
Accuracy-vs-speed figure (Fig. 1): all FOUR pipelines as four points.
Replaces the old 2-point version that drew Classical as a meaningless
horizontal line. Reads the per-case CSVs from ../results/.

x = mean wall-clock per case (s), y = mean Dice (120 cases).

Run (TWCC):
    /opt/ohpc/twcc/conda/24.5.0/miniconda3/bin/python plot_accuracy_speed.py
"""
import csv
import os
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGDIR = os.path.join(HERE, "figures"); os.makedirs(FIGDIR, exist_ok=True)
OUT = os.path.join(FIGDIR, "accuracy_vs_speed.png")


def stats(path, dice_key, time_key):
    d, t = [], []
    with open(os.path.join(RESULTS, path)) as f:
        for r in csv.DictReader(f):
            d.append(float(r[dice_key])); t.append(float(r[time_key]))
    ge = sum(x >= 0.8 for x in d)
    return statistics.mean(t), statistics.mean(d), ge, len(d)

# (label, csv, dice col, time col, colour, (dx,dy) label offset in pts, ha, va)
P = [
    ("Classical",  "classical.csv", "dice",       "total_s",    "#7f7f7f", (12,   0), "left",  "center"),
    ("2.5D U-Net", "unet25d.csv",   "Dice_Score", "Total_Time", "#DD8452", (-12,  0), "right", "center"),
    ("Hybrid",     "hybrid.csv",    "Dice_Score", "Total_Time", "#55A868", (0,  -16), "center", "top"),
    ("3D U-Net",   "unet3d.csv",    "Dice_Score", "Total_Time", "#4C72B0", (9,   11), "left",  "bottom"),
]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for name, fn, dk, tk, col, (dx, dy), ha, va in P:
    x, y, ge, n = stats(fn, dk, tk)
    usable = y >= 0.8
    ax.scatter(x, y, s=140, color=col, zorder=5,
               edgecolor="black", linewidth=0.6,
               marker="o" if usable else "X")
    ax.annotate(f"{name}\n{x:.2f}s/case, Dice {y:.3f}\n{ge}/{n} $\\geq$0.8",
                xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                color=col, fontsize=8.5, ha=ha, va=va, fontweight="bold")

ax.axhline(0.8, ls="--", color="#2ca02c", lw=1.3)
ax.text(0.2, 0.806, "Dice = 0.8 (clinically usable)", color="#2ca02c", fontsize=8.5)

ax.set_xlabel("Mean time per case (s)  --  lower = faster", fontsize=10)
ax.set_ylabel("Mean Dice (120 cases)", fontsize=10)
ax.set_xlim(0, 13)
ax.set_ylim(0.62, 0.97)
ax.set_title("Accuracy vs. speed: four pipelines (120 cases, single V100)",
             fontsize=10.5, pad=10)
ax.grid(True, ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", os.path.normpath(OUT))
for name, fn, dk, tk, *_ in P:
    x, y, ge, n = stats(fn, dk, tk)
    print(f"  {name:11s} {x:.2f}s  Dice {y:.4f}  {ge}/{n}>=0.8")
