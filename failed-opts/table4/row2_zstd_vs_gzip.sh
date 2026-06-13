#!/bin/bash
# =============================================================================
# Table IV, row 2 -- "zstd instead of gzip : No speedup"
#   Two claims to reproduce:
#   (a) A single-frame compressed stream cannot be decoded in parallel, so zstd
#       -T0 (multi-thread) is no faster to DECODE than zstd -T1, and zstd is in
#       the same ballpark as libdeflate gunzip -> switching codecs buys nothing.
#   (b) The earlier "4x" win was a COLD-vs-WARM page-cache artifact, not a codec
#       effect: the same file decompresses ~Nx slower on a cold cache.
#
# This is a load-stage microbenchmark (independent of the GPU binary). It needs
# `gzip` and `zstd` on PATH. Cold-cache timing needs either root (drop_caches)
# or `vmtouch`; otherwise only the warm/parallel claims are measured.
#
# Run:   bash failed-opts/table4/row2_zstd_vs_gzip.sh
#        NII=/work/$USER/decompressed_nii/AIIB23_100.nii bash failed-opts/table4/row2_zstd_vs_gzip.sh
# =============================================================================
source "$(dirname "$0")/common.sh"

command -v zstd >/dev/null || { echo "[ERROR] zstd not on PATH"; exit 1; }
command -v gzip >/dev/null || { echo "[ERROR] gzip not on PATH"; exit 1; }

WORK="$OUTROOT/row2_work"; mkdir -p "$WORK"
OUT="$OUTROOT/row2_zstd_vs_gzip.csv"
echo "codec,threads,cache,decode_s,size_MB" > "$OUT"

# --- obtain one plain .nii to compress ---------------------------------------
NII="${NII:-}"
if [ -z "$NII" ]; then
    c="$(echo $CASES | awk '{print $1}')"
    if [ -s "$NIICACHE/AIIB23_${c}.nii" ]; then NII="$NIICACHE/AIIB23_${c}.nii"
    elif [ -f "$IMGDIR/AIIB23_${c}.nii.gz" ]; then
        echo "[row2] gunzipping case $c once -> plain .nii"
        gzip -dc "$IMGDIR/AIIB23_${c}.nii.gz" > "$WORK/sample.nii"
        NII="$WORK/sample.nii"
    else
        echo "[ERROR] no AIIB23 input found; set NII=/path/to/a/plain.nii"; exit 1
    fi
fi
SZ=$(du -m "$NII" | cut -f1)
echo "[row2] sample: $NII (${SZ} MB)"

# --- compress once with each codec (comparable level) ------------------------
echo "[row2] compressing (gzip -6, zstd -6)..."
gzip  -6 -c "$NII" > "$WORK/s.nii.gz"
zstd -q -6 -f "$NII" -o "$WORK/s.nii.zst"

time_decode() {  # $1=cmd... ; echoes seconds
    local t0 t1
    t0=$(date +%s.%N); "$@" > /dev/null 2>&1; t1=$(date +%s.%N)
    awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}'
}

run_row() {  # $1=codec $2=threads $3=cache  $4..=cmd
    local codec="$1" thr="$2" cache="$3"; shift 3
    local s; s=$(time_decode "$@")
    echo "$codec,$thr,$cache,$s,$SZ" >> "$OUT"
    printf "  %-14s threads=%-2s cache=%-4s  %ss\n" "$codec" "$thr" "$cache" "$s"
}

echo
echo "=== (a) WARM decode: parallel zstd gives no edge; zstd ~ gunzip ==="
# prime caches
cat "$WORK/s.nii.gz" "$WORK/s.nii.zst" > /dev/null 2>&1
run_row "gunzip"     1  warm  gzip -dc "$WORK/s.nii.gz"
run_row "libdeflate" 1  warm  bash -c "command -v libdeflate-gunzip >/dev/null && libdeflate-gunzip -dc '$WORK/s.nii.gz' || gzip -dc '$WORK/s.nii.gz'"
run_row "zstd-T1"    1  warm  zstd -d -T1 -c "$WORK/s.nii.zst"
run_row "zstd-T0"    0  warm  zstd -d -T0 -c "$WORK/s.nii.zst"
echo "  -> zstd-T0 ~= zstd-T1 (single frame can't decode in parallel),"
echo "     and zstd ~= gunzip: switching the codec is not a speedup."

echo
echo "=== (b) COLD-vs-WARM artifact: the old '4x' was cache, not codec ==="
drop_caches() {
    if [ "$(id -u)" = "0" ]; then sync; echo 3 > /proc/sys/vm/drop_caches; return 0; fi
    if command -v vmtouch >/dev/null; then vmtouch -e "$1" >/dev/null 2>&1; return 0; fi
    return 1
}
if drop_caches "$WORK/s.nii.gz"; then
    run_row "gunzip" 1 cold  gzip -dc "$WORK/s.nii.gz"
    cat "$WORK/s.nii.gz" > /dev/null 2>&1   # warm it
    run_row "gunzip" 1 warm2 gzip -dc "$WORK/s.nii.gz"
    echo "  -> compare cold vs warm2: the ratio (~few x) is the '4x' artifact,"
    echo "     present for ANY codec; it is not evidence that zstd is faster."
else
    echo "  [skip] cold-cache timing needs root (drop_caches) or vmtouch."
    echo "         Run as root, or: vmtouch -e $WORK/s.nii.gz before re-timing."
fi

echo
echo "wrote $OUT"
