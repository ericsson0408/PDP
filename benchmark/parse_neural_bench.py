#!/usr/bin/env python3
# Parse per-case neural airway_seg logs (unet3d / hybrid / unet25d) into the
# per-case benchmark CSV read by analysis/.
# Columns: Case_ID, Dice_Score, Load_Time, MPI_Part_Time, GPU_Filter_Time,
#          Gather_Time, ONNX_Infer_Time, Region_Grow_Time, Write_Time,
#          Eval_Time, Total_Time
# Usage: python benchmark/parse_neural_bench.py <logdir> <out.csv>
import os, re, csv, glob, sys

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "logs/neuralbench"
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else "results/unet3d.csv"

# timing label in the "=== Timing (s) ===" block  ->  CSV column
TIMING = {
    "load":        "Load_Time",
    "mpi part":    "MPI_Part_Time",
    "gpu filter":  "GPU_Filter_Time",
    "gather":      "Gather_Time",
    "onnx infer":  "ONNX_Infer_Time",
    "region grow": "Region_Grow_Time",
    "write":       "Write_Time",
    "eval (dice)": "Eval_Time",
    "TOTAL":       "Total_Time",
}
COLS = ["Case_ID", "Dice_Score", "Load_Time", "MPI_Part_Time", "GPU_Filter_Time",
        "Gather_Time", "ONNX_Infer_Time", "Region_Grow_Time", "Write_Time",
        "Eval_Time", "Total_Time"]

def parse_log(path):
    row = {c: "" for c in COLS}
    txt = open(path, errors="ignore").read()
    m = re.search(r"AIIB23_(\d+)", os.path.basename(path))
    row["Case_ID"] = m.group(1) if m else os.path.basename(path)
    d = re.findall(r"Dice=([0-9.]+)", txt)
    if d:
        row["Dice_Score"] = d[-1]
    for label, col in TIMING.items():
        # lines look like:  "  onnx infer  :   54.928"
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
    rows = [r for r in rows if r["Total_Time"] or r["Dice_Score"]]  # drop empty/failed
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    def col(name):
        return [float(r[name]) for r in rows if r[name]]
    dices = col("Dice_Score"); onnx = col("ONNX_Infer_Time"); tot = col("Total_Time")
    print(f"[csv] wrote {OUT_CSV}  ({len(rows)} cases)")
    if dices:
        print(f"  Dice: mean={sum(dices)/len(dices):.4f}  min={min(dices):.4f}  max={max(dices):.4f}")
    if onnx:
        print(f"  ONNX infer (s): mean={sum(onnx)/len(onnx):.2f}  min={min(onnx):.2f}  max={max(onnx):.2f}")
    if tot:
        print(f"  Total (s): mean={sum(tot)/len(tot):.2f}")

if __name__ == "__main__":
    main()
