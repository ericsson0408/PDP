#!/bin/bash
# =============================================================================
# Table IV, row 4 -- "input size 256->224 : Rejected (12% faster, 117->108 >=0.8)"
#   Claim: shrinking the 2.5D model's per-slice input from 256x256 to 224x224 is
#   ~12% faster but drops the usable-case count (Dice>=0.8) from 117 to 108 --
#   a tail-loss that a few sampled cases would hide. Rejected.
#
# This is the ONE Table IV row with NO archived CSV, so reproducing it requires:
#   1) an ONNX that accepts 224x224  (the shipped 2.5D onnx is fixed at 256);
#   2) a SECOND binary built with MODEL25D_H/W = 224 (a compile-time constant in
#      src/onnx_infer.cpp -- not a CLI flag);
#   3) the FULL 120-case set (the 117 vs 108 claim is a tail count).
#
# Steps 1-2 are automated below. Step 3: set CASES="all" (default here) and run
# on a V100. A small CASES= subset reproduces the SPEED delta but, by design,
# "samples hide the tail loss" -- so it will NOT show the 117->108 drop.
#
# Prereq:
#   python failed-opts/table4/export_25d_dynamic.py   # -> model/airway_2p5d_unet_dyn.onnx
#
# Run:   CASES=all bash failed-opts/table4/row4_input_size.sh
# =============================================================================
source "$(dirname "$0")/common.sh"
need_bin

SRC="$ROOT/src/onnx_infer.cpp"
DYNMODEL="${DYNMODEL:-$ROOT/model/airway_2p5d_unet_dyn.onnx}"
STOCK="$ROOT/model/airway_2p5d_unet.onnx"
OUT="$OUTROOT/row4_input_size.csv"
LOGD="$OUTROOT/row4_logs"; mkdir -p "$LOGD"
echo "input_size,mean_total_s,n_ge_0.8,n_cases" > "$OUT"

if [ ! -f "$DYNMODEL" ]; then
    echo "[row4] generating dynamic-spatial 2.5D ONNX from $STOCK ..."
    "$PYBIN" "$(dirname "$0")/export_25d_dynamic.py" --src "$STOCK" --dst "$DYNMODEL" || exit 1
fi

# The 117->108 claim is a 120-case tail count: pass CASES=all for the real test.
# A subset only reproduces the SPEED delta (samples hide the tail loss by design).
if [ "${CASES:-}" = "all" ]; then
    CASES="$(ls "$IMGDIR"/AIIB23_*.nii.gz 2>/dev/null | sed -E 's@.*AIIB23_([0-9]+)\.nii\.gz@\1@' | sort -n)"
    [ -n "$CASES" ] || { echo "[ERROR] no cases in $IMGDIR"; exit 1; }
fi
echo "[row4] cases: $(echo $CASES | wc -w)  (use CASES=all for the 117 vs 108 tail count)"

# --- build a binary for a given MODEL25D size -> $1=size  echoes binary path ---
build_size() {
    local size="$1"
    local bin="$ROOT/airway_seg_25d_${size}"   # NB: separate decl; `${size}` on the
                                               # same `local` line trips set -u.
    if [ "$size" = "256" ]; then
        # stock binary already uses 256
        cp -f "$BIN" "$bin"; echo "$bin"; return 0
    fi
    local stock_bak="$ROOT/airway_seg_25d_256"   # made by the 256 call (= stock binary)
    cp -f "$SRC" "$SRC.t4bak"
    sed -i -E "s/(MODEL25D_H = )[0-9]+/\1${size}/; s/(MODEL25D_W = )[0-9]+/\1${size}/" "$SRC"
    ( cd "$ROOT" && make >/dev/null 2>&1 ) || { mv -f "$SRC.t4bak" "$SRC"; echo "BUILD_FAIL"; return 1; }
    cp -f "$BIN" "$bin"
    # restore stock source + binary (copy back, no slow rebuild needed)
    mv -f "$SRC.t4bak" "$SRC"
    [ -f "$stock_bak" ] && cp -f "$stock_bak" "$BIN"
    echo "$bin"
}

bench_size() {  # $1=size  $2=binary  $3=model
    local size="$1" bin="$2" model="$3" totals="" ge=0 n=0
    for c in $CASES; do
        read -r d t i l <<<"$(run_case "$bin" "$c" "$LOGD/s${size}_c${c}.log" \
                                --model-2p5d --gate --onnx-model "$model")"
        totals="$totals$t"$'\n'; n=$((n+1))
        awk -v x="$d" 'BEGIN{exit !(x>=0.8)}' && ge=$((ge+1))
    done
    local mt; mt="$(printf "%s" "$totals" | mean)"
    echo "$size,$mt,$ge,$n" >> "$OUT"
    printf "  input %-3s  mean_total=%-8s  cases>=0.8: %d/%d\n" "$size" "$mt" "$ge" "$n"
}

echo "=== Table IV row 4: 2.5D input 256 vs 224 ==="
B256="$(build_size 256)"; echo "[build] 256 -> $B256"
B224="$(build_size 224)"; [ "$B224" = "BUILD_FAIL" ] && { echo "[ERROR] 224 build failed"; exit 1; }
echo "[build] 224 -> $B224"

bench_size 256 "$B256" "$STOCK"
bench_size 224 "$B224" "$DYNMODEL"

echo
echo "[interpretation] paper: 224 is ~12% faster but cases>=0.8 fall 117->108."
echo "                 Needs the FULL 120-case run to see the tail drop."
echo "wrote $OUT"
