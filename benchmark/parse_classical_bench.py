#!/usr/bin/env python3
# Parse classical (--classical) airway_seg logs into the classical.csv schema
# expected by analysis/plot_stage_breakdown.py (classical_means) and
# plot_accuracy_speed.py. Same "=== Timing (s) ===" block as the neural logs,
# but emitted with the classical column names (lowercase *_s) plus nz / voxels.
# Usage: python benchmark/parse_classical_bench.py <logdir> <out.csv>
import os, re, csv, glob, sys

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "logs/fairbench/classical"
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else "results/classical.csv"

TIMING = {
    "load":        "load_s",
    "mpi part":    "mpi_part_s",
    "gpu filter":  "gpu_filter_s",
    "gather":      "gather_s",
    "region grow": "region_grow_s",
    "write":       "write_s",
    "eval (dice)": "eval_s",
    "other/sync":  "other_s",
    "TOTAL":       "total_s",
}
COLS = ["case", "nz", "method", "load_s", "mpi_part_s", "gpu_filter_s",
        "gather_s", "region_grow_s", "write_s", "eval_s", "other_s",
        "total_s", "dice", "pred_voxels", "gt_voxels"]

def parse_log(path):
    row = {c: "" for c in COLS}
    txt = open(path, errors="ignore").read()
    m = re.search(r"AIIB23_(\d+)", os.path.basename(path))
    row["case"] = m.group(1) if m else os.path.basename(path)
    row["method"] = "v8"
    dims = re.search(r"dims=\d+x\d+x(\d+)", txt)
    if dims:
        row["nz"] = dims.group(1)
    d = re.findall(r"Dice=([0-9.]+)", txt)
    if d:
        row["dice"] = d[-1]
    dv = re.search(r"pred=(\d+)\s+gt=(\d+)", txt)
    if dv:
        row["pred_voxels"], row["gt_voxels"] = dv.group(1), dv.group(2)
    for label, col in TIMING.items():
        mm = re.search(r"^\s*" + re.escape(label) + r"\s*:\s*([0-9]+\.[0-9]+)",
                       txt, re.MULTILINE)
        if mm:
            row[col] = mm.group(1)
    return row

def main():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")),
                  key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1))
                  if re.search(r"(\d+)", os.path.basename(p)) else 0)
    rows = [parse_log(p) for p in logs]
    rows = [r for r in rows if r["total_s"] or r["dice"]]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    def col(name):
        return [float(r[name]) for r in rows if r[name]]
    dices, load, tot = col("dice"), col("load_s"), col("total_s")
    print(f"[csv] wrote {OUT_CSV}  ({len(rows)} cases)")
    if dices:
        print(f"  Dice : mean={sum(dices)/len(dices):.4f}")
    if load:
        print(f"  load : mean={sum(load)/len(load):.3f}s")
    if tot:
        print(f"  TOTAL: mean={sum(tot)/len(tot):.3f}s")

if __name__ == "__main__":
    main()
