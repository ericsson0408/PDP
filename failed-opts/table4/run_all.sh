#!/bin/bash
# =============================================================================
# Reproduce all of paper Table IV ("Evaluated but discarded optimizations").
# Runs the rows that need only the shipped binary + CSVs by default; the rows
# that need an extra ONNX export or a rebuild are gated behind prereq checks
# and will print exactly what to build if their inputs are missing.
#
#   bash failed-opts/table4/run_all.sh                  # smoke set (fast)
#   CASES=all bash failed-opts/table4/run_all.sh        # full 120-case (row4 tail count)
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "########################################################################"
echo "# Table IV reproduction  ($(date))"
echo "########################################################################"

for row in \
    "row3_mpi_mps.sh        (analysis only, no GPU)" \
    "row5_batch_tuning.sh   (CLI sweep, GPU)" \
    "row2_zstd_vs_gzip.sh   (load microbench)" \
    "row1_fp16_io.sh        (needs export_fp16io.py first)" \
    "row4_input_size.sh     (needs export_25d_dynamic.py + rebuild)"
do
    script="${row%% *}"
    echo; echo "======================================================================"
    echo ">>> $row"
    echo "======================================================================"
    bash "$HERE/$script" || echo "[run_all] $script returned non-zero (see message above)"
done

echo
echo "########################################################################"
echo "# Done. Per-row CSVs in: logs/failed_opts/table4/"
echo "########################################################################"
ls -1 "$HERE/../logs/failed_opts/table4/"*.csv 2>/dev/null || true
