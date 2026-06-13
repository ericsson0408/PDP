#!/usr/bin/env python3
"""
Per-stage wall-clock breakdown of the four pipelines (mean over 120 cases).
Reads the raw per-case CSVs from ../results/ and writes a stacked horizontal
bar chart to analysis/figures/stage_breakdown.png.

Run with a python that has matplotlib, e.g. on TWCC:
    module load miniconda3/conda24.5.0_py3.9
    /opt/ohpc/twcc/conda/24.5.0/miniconda3/bin/python plot_stage_breakdown.py
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGDIR = os.path.join(HERE, "figures"); os.makedirs(FIGDIR, exist_ok=True)
OUT = os.path.join(FIGDIR, "stage_breakdown.png")

# Stage order (bottom -> top of each stacked bar) and colours.
STAGES = ["Load", "Inference", "Region grow", "Write", "Eval", "Other"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def neural_means(path):
    """Mean per-stage seconds for a neural pipeline CSV."""
    keys = ["Load_Time", "ONNX_Infer_Time", "Region_Grow_Time",
            "Write_Time", "Eval_Time", "Total_Time"]
    acc = {k: 0.0 for k in keys}
    n = 0
    with open(path) as f:
        for row in csv.DictReader(f):
            n += 1
            for k in keys:
                acc[k] += float(row[k])
    m = {k: acc[k] / n for k in keys}
    other = (m["Total_Time"] - m["Load_Time"] - m["ONNX_Infer_Time"]
             - m["Region_Grow_Time"] - m["Write_Time"] - m["Eval_Time"])
    return [m["Load_Time"], m["ONNX_Infer_Time"], m["Region_Grow_Time"],
            m["Write_Time"], m["Eval_Time"], max(other, 0.0)]


def classical_means(path):
    """Mean per-stage seconds for the classical pipeline CSV.
    MPI scatter + GPU Frangi/Gaussian + gather are merged into 'Inference'
    so the bar is comparable to the neural ONNX stage."""
    keys = ["load_s", "mpi_part_s", "gpu_filter_s", "gather_s",
            "region_grow_s", "write_s", "eval_s", "other_s", "total_s"]
    acc = {k: 0.0 for k in keys}
    n = 0
    with open(path) as f:
        for row in csv.DictReader(f):
            n += 1
            for k in keys:
                acc[k] += float(row[k])
    m = {k: acc[k] / n for k in keys}
    filt = m["mpi_part_s"] + m["gpu_filter_s"] + m["gather_s"]
    return [m["load_s"], filt, m["region_grow_s"],
            m["write_s"], m["eval_s"], m["other_s"]]


# Pipelines in plotting order (top bar last). Slowest at top reads well.
data = {
    "2.5D U-Net": neural_means(os.path.join(RESULTS, "unet25d.csv")),
    "Classical":  classical_means(os.path.join(RESULTS, "classical.csv")),
    "Hybrid":     neural_means(os.path.join(RESULTS, "hybrid.csv")),
    "3D U-Net":   neural_means(os.path.join(RESULTS, "unet3d.csv")),
}

labels = list(data.keys())
y = range(len(labels))

fig, ax = plt.subplots(figsize=(6.4, 2.8))
left = [0.0] * len(labels)
for si, stage in enumerate(STAGES):
    vals = [data[p][si] for p in labels]
    ax.barh(y, vals, left=left, color=COLORS[si], label=stage,
            edgecolor="white", height=0.62)
    left = [l + v for l, v in zip(left, vals)]

# Total time annotation at the end of each bar.
for yi, p in zip(y, labels):
    total = sum(data[p])
    ax.text(total + 0.12, yi, f"{total:.2f}s", va="center", ha="left",
            fontsize=9, fontweight="bold")

ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Mean wall-clock per case (s)", fontsize=10)
ax.set_xlim(0, 11)
ax.set_title("Per-stage wall-clock breakdown (120 cases, single V100)",
             fontsize=10)
ax.legend(ncol=6, fontsize=7.5, loc="lower center",
          bbox_to_anchor=(0.5, -0.42), frameon=False, columnspacing=1.0,
          handletextpad=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", os.path.normpath(OUT))
for p in labels:
    print(f"  {p:12s} total={sum(data[p]):.2f}  stages={[round(v,3) for v in data[p]]}")
