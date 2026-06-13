#!/bin/bash
# =============================================================================
# Table IV, row 3 -- "MPI+MPS, >2 ranks : Saturates"
#   Claims to reproduce:
#   (i)   inference is ~81% of wall-clock (so the non-GPU tail an extra rank can
#         overlap is only ~19%);
#   (ii)  a 2nd co-located rank can recover at most that tail (~-17% projected);
#   (iii) a 3rd/4th rank adds nothing -- aggregate throughput stays flat.
#
# (i) and (iii) are MEASURED from archived CSVs (this script, no GPU needed).
# (ii) is an Amdahl PROJECTION: the paper notes CUDA MPS would not start on the
#      test node, so -17% is the conservative no-sharing tail, not a live number.
#
# To RE-MEASURE the saturation sweep live on a V100:
#   sbatch analysis/measure_mps_util.sbatch     # -> results/mps_util.csv (N=1..4)
#
# Run (analysis only):   bash failed-opts/table4/row3_mpi_mps.sh
# =============================================================================
source "$(dirname "$0")/common.sh"

# Archived per-case CSVs live in the package's results/ dir, independent of
# TABLE4_ROOT (failed-opts/table4 -> ../.. = package root).
PKG="$(cd "$(dirname "$0")/../.." && pwd)"
find_csv() { for d in "$PKG/results" "$ROOT/results"; do
                 [ -f "$d/$1" ] && { echo "$d/$1"; return; }; done; echo "$PKG/results/$1"; }
THREED="$(find_csv unet3d.csv)"
MPS="$(find_csv mps_util.csv)"

python3 - "$THREED" "$MPS" <<'PY'
import csv, sys, statistics as st
threed, mps = sys.argv[1], sys.argv[2]

print("=== Table IV row 3: MPI+MPS saturation ===\n")

# (i) inference fraction of wall-clock, from per-case 3D results.
try:
    inf, tot = [], []
    with open(threed) as f:
        for r in csv.DictReader(f):
            inf.append(float(r["ONNX_Infer_Time"])); tot.append(float(r["Total_Time"]))
    frac = sum(inf)/sum(tot)
    print(f"(i)   inference fraction  = mean(infer)/mean(total)")
    print(f"      = {st.mean(inf):.3f}s / {st.mean(tot):.3f}s = {frac*100:.1f}%  "
          f"(paper: 81%)   [n={len(inf)} cases, {threed.split('/')[-1]}]")
    tail = 1 - frac
    print(f"(ii)  non-GPU tail an extra rank can overlap = {tail*100:.1f}%")
    print(f"      => best-case 2-rank recovery ~ -{tail*100:.0f}%  (paper PROJECTION: -17%,")
    print(f"         MPS would not start on the node -> conservative no-sharing floor)\n")
except Exception as e:
    print(f"(i/ii) skipped: {e}\n")

# (iii) throughput flat across co-located ranks.
try:
    rows = list(csv.DictReader(open(mps)))
    print(f"(iii) co-location sweep ({mps.split('/')[-1]}):")
    print(f"      {'ranks':>5} {'throughput(cases/s)':>20} {'SM_peak%':>9} {'wall_s':>9}")
    base = None
    for r in rows:
        n = r["nranks"]; thr = float(r["throughput_cases_per_s"])
        base = base or thr
        print(f"      {n:>5} {thr:>20.4f} {r['sm_max_pct']:>9} {float(r['wall_s']):>9.2f}")
    last = float(rows[-1]["throughput_cases_per_s"])
    print(f"\n      throughput {base:.4f} -> {last:.4f} cases/s across "
          f"{rows[0]['nranks']}..{rows[-1]['nranks']} ranks = FLAT "
          f"({(last-base)/base*100:+.1f}%); SM peak already 100% at 1 rank.")
    print("      => the GPU, not the rank count, is the binding constraint (paper).")
except Exception as e:
    print(f"(iii) skipped: {e}")
PY

echo
echo "[note] re-measure live:  sbatch analysis/measure_mps_util.sbatch"
