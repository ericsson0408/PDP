#!/usr/bin/env python3
"""
Per-voxel inference-cost figure (replaces the old 2-bar 'Per voxel cost.png').

Scatter of per-case ONNX inference time vs total volume (nx*ny*nz) for the 3D
and 2.5D pipelines, with through-origin linear fits whose slopes ARE the
per-voxel cost. Directly visualizes why we normalize by case size: time scales
with volume, and the two slopes give the 3D-vs-2.5D per-voxel gap.

Reads (from ../results/): case_dims.csv, unet3d.csv, unet25d.csv
Writes: analysis/figures/per_voxel_cost.png

Run (TWCC):
    /opt/ohpc/twcc/conda/24.5.0/miniconda3/bin/python plot_pervoxel.py
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGDIR = os.path.join(HERE, "figures"); os.makedirs(FIGDIR, exist_ok=True)
OUT = os.path.join(FIGDIR, "per_voxel_cost.png")

# case -> total voxels
vox = {}
with open(os.path.join(RESULTS, "case_dims.csv")) as f:
    for r in csv.DictReader(f):
        vox[r["Case_ID"]] = int(r["nx"]) * int(r["ny"]) * int(r["nz"])


def load(path):
    xs, ys = [], []
    with open(os.path.join(RESULTS, path)) as f:
        for r in csv.DictReader(f):
            c = r["Case_ID"]
            if c in vox:
                xs.append(vox[c] / 1e8)              # volume, 1e8 voxels
                ys.append(float(r["ONNX_Infer_Time"]))  # seconds
    return xs, ys


def slope_through_origin(xs, ys):
    # least squares y = m x ; xs in 1e8 voxels, slope back to ns/voxel
    num = sum(x * y for x, y in zip(xs, ys))
    den = sum(x * x for x in xs)
    m = num / den                                   # s per 1e8 voxels
    return m, m * 1e9 / 1e8                          # (slope_s_per_1e8, ns/voxel)


x3, y3 = load("unet3d.csv")
x2, y2 = load("unet25d.csv")
m3, pv3 = slope_through_origin(x3, y3)
m2, pv2 = slope_through_origin(x2, y2)

fig, ax = plt.subplots(figsize=(5.2, 3.5))
ax.scatter(x3, y3, s=16, color="#4C72B0", alpha=0.6, label=f"3D U-Net  ({pv3:.0f} ns/voxel)")
ax.scatter(x2, y2, s=16, color="#DD8452", alpha=0.6, label=f"2.5D U-Net  ({pv2:.0f} ns/voxel)")

xmax = max(max(x3), max(x2)) * 1.05
ax.plot([0, xmax], [0, m3 * xmax], color="#4C72B0", lw=1.6)
ax.plot([0, xmax], [0, m2 * xmax], color="#DD8452", lw=1.6)

ax.set_xlabel(r"Case volume  $n_x\!\cdot\!n_y\!\cdot\!n_z$  ($10^8$ voxels)")
ax.set_ylabel("ONNX inference time (s)")
ax.set_xlim(0, xmax)
ax.set_ylim(0, max(max(y3), max(y2)) * 1.05)
ax.set_title(f"Inference time scales with case size\n"
             f"per-voxel cost: 2.5D is {pv3/pv2:.1f}$\\times$ cheaper than 3D",
             fontsize=10)
ax.legend(fontsize=8.5, loc="upper left", frameon=True)
ax.grid(True, ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", os.path.normpath(OUT))
print(f"  3D   slope = {pv3:.1f} ns/voxel  (n={len(x3)})")
print(f"  2.5D slope = {pv2:.1f} ns/voxel  (n={len(x2)})")
print(f"  ratio = {pv3/pv2:.2f}x   volume range = {min(x3+x2)*1e8:.2e}..{max(x3+x2)*1e8:.2e} voxels")
