#!/bin/bash
# =============================================================================
# failed-opts/table4/common.sh  --  shared helpers for reproducing paper Table IV
# ("Evaluated but discarded optimizations and their rejection rationales").
#
# Sourced by every rowN_*.sh script. Provides:
#   - environment / library-path setup matching the paper's benchmark scripts
#   - case-input resolution (prefer pre-decompressed plain .nii, else .nii.gz)
#   - one-case runner that captures the binary's stage timings + Dice
#   - tiny parsers for the binary's stdout format
#
# Override knobs (env vars):
#   CASES="100 101 168"   # which AIIB23 cases to use (default = a small smoke set)
#   SKIP_ENV=1            # do not `source env.sh` (off-cluster, modules on PATH)
#   NIICACHE=/path        # directory of pre-decompressed AIIB23_*.nii
# =============================================================================
set -uo pipefail

# --- locate the working tree ---------------------------------------------------
# Default = the reproducibility-package root (parent of failed-opts/table4/). On TWCC
# the buildable tree with ONNX Runtime + built binary + models is the MAIN
# project, so set TABLE4_ROOT=/home/u4309334/Project to run there.
ROOT="${TABLE4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

BIN="${BIN:-$ROOT/airway_seg}"
IMGDIR="${IMGDIR:-$ROOT/data/img}"
GTDIR="${GTDIR:-$ROOT/data/gt}"
ORT_LIB_DIR="${ORT_LIB_DIR:-$ROOT/third_party/onnxruntime-linux-x64-gpu-1.19.2/lib}"
PYBIN="${PYBIN:-/opt/ohpc/twcc/conda/24.5.0/miniconda3/bin/python}"
OUTROOT="${OUTROOT:-$ROOT/logs/failed_opts/table4}"
mkdir -p "$OUTROOT"

# --- cluster module loads (guarded so the scripts also run off-cluster) -------
if [ "${SKIP_ENV:-0}" != "1" ] && [ -f "$ROOT/env.sh" ]; then
    # env.sh does `module load ...`; harmless to ignore if modules are absent.
    # shellcheck disable=SC1091
    source "$ROOT/env.sh" 2>/dev/null || echo "[common] note: env.sh load skipped (off-cluster?)"
fi

# Runtime libs: ONNX Runtime + cuDNN (mirrors pipelines/*/run.sh).
export OMPI_MCA_opal_cuda_support=0
export LD_LIBRARY_PATH="$ORT_LIB_DIR:${HOME}/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" OMP_PROC_BIND=close OMP_PLACES=cores

# Default smoke set: a spread of sizes incl. the safety-abort case (168).
# Override with CASES="..." for the full / a custom cohort.
CASES="${CASES:-100 101 110 168}"

# Pre-decompressed plain-.nii cache (the paper's zero-copy fast path).
NIICACHE="${NIICACHE:-/work/${USER}/decompressed_nii}"

# --- resolve a case id -> best available input path ---------------------------
src_for() {
    local c="$1"
    if [ -s "$NIICACHE/AIIB23_${c}.nii" ]; then
        echo "$NIICACHE/AIIB23_${c}.nii"
    else
        echo "$IMGDIR/AIIB23_${c}.nii.gz"
    fi
}

# --- stdout parsers (binary prints: "Dice=0.xxxx", "  TOTAL : N", etc.) --------
parse_dice()  { grep -oE "Dice=[0-9.]+"            "$1" | tail -1 | cut -d= -f2; }
parse_total() { grep -E "^  TOTAL"                 "$1" | grep -oE "[0-9]+\.[0-9]+" | tail -1; }
parse_infer() { grep -E "^  onnx infer"            "$1" | grep -oE "[0-9]+\.[0-9]+" | tail -1; }
parse_load()  { grep -E "^  load"                  "$1" | grep -oE "[0-9]+\.[0-9]+" | head -1; }

# --- run ONE case, return "<dice> <total> <infer> <load>" (NA if missing) ------
# usage: run_case <binary> <case_id> <logfile> <extra airway_seg flags...>
run_case() {
    local bin="$1" c="$2" logf="$3"; shift 3
    local src; src="$(src_for "$c")"
    if [ ! -f "$src" ]; then echo "NA NA NA NA"; return 1; fi
    cat "$src" > /dev/null 2>&1   # warm the page cache (match paper's load timing)
    "$bin" \
        --input  "$src" \
        --output "$OUTROOT/_scratch_${c}.nii.gz" \
        --gt     "$GTDIR/AIIB23_${c}.nii.gz" \
        --omp-threads "$OMP_NUM_THREADS" \
        "$@" > "$logf" 2>&1
    echo "$(parse_dice "$logf") $(parse_total "$logf") $(parse_infer "$logf") $(parse_load "$logf")"
}

# mean of a whitespace/newline list of numbers (ignores NA)
mean() { awk '{ if ($1!="NA" && $1!=""){s+=$1;n++} } END{ if(n)printf "%.4f",s/n; else printf "NA" }'; }

need_bin() {
    [ -x "$BIN" ] || { echo "[ERROR] $BIN not built. Run: cd $ROOT && source env.sh && make"; exit 1; }
}
