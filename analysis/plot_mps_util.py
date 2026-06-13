#!/usr/bin/env python3
"""
Plot the MPS saturation measurement: device SM utilization (%) and aggregate
throughput (cases/s) vs number of co-located inference processes (1..4).

Input : ../results/mps_util.csv  (produced by measure_mps_util.sbatch)
Output: analysis/figures/mps_saturation.png

Run (TWCC):
    /opt/ohpc/twcc/conda/24.5.0/miniconda3/bin/python plot_mps_util.py
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "mps_util.csv")
FIGDIR = os.path.join(HERE, "figures"); os.makedirs(FIGDIR, exist_ok=True)
OUT = os.path.join(FIGDIR, "mps_saturation.png")

n, sm_mean, sm_max, thr = [], [], [], []
with open(SRC) as f:
    for row in csv.DictReader(f):
        n.append(int(row["nranks"]))
        sm_mean.append(float(row["sm_mean_pct"]))
        sm_max.append(float(row["sm_max_pct"]))
        thr.append(float(row["throughput_cases_per_s"]))

fig, ax1 = plt.subplots(figsize=(5.0, 3.2))
ax1.plot(n, sm_mean, "o-", color="#C44E52", label="SM util (mean %)")
ax1.plot(n, sm_max, "o--", color="#C44E52", alpha=0.5, label="SM util (peak %)")
ax1.set_xlabel("Co-located inference processes on one V100")
ax1.set_ylabel("GPU SM utilization (%)", color="#C44E52")
ax1.set_ylim(0, 105)
ax1.set_xticks(n)
ax1.tick_params(axis="y", labelcolor="#C44E52")
ax1.axhline(100, color="gray", ls=":", lw=0.8)

ax2 = ax1.twinx()
ax2.plot(n, thr, "s-", color="#4C72B0", label="throughput")
ax2.set_ylabel("Throughput (cases/s)", color="#4C72B0")
ax2.tick_params(axis="y", labelcolor="#4C72B0")
# anchor the throughput axis at 0 so its flatness is honestly visible
ax2.set_ylim(0, max(thr) * 1.5)

lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="lower right")
ax1.set_title("GPU saturation: SM utilization and throughput vs co-located procs",
              fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", os.path.normpath(OUT))
for i in range(len(n)):
    print(f"  N={n[i]}  sm_mean={sm_mean[i]}%  sm_max={sm_max[i]}%  thr={thr[i]:.3f} cases/s")
