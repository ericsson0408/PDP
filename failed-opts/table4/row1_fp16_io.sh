#!/bin/bash
# =============================================================================
# Table IV, row 1 -- "FP16 input/output : No speedup"
#   Claim: feeding the model FP16 input/output tensors (instead of FP32 I/O on
#   the same FP16-weight model) changes only the host I/O dtype and yields no
#   speedup -- the weights are already FP16 internally where the GPU win is.
#
# Reproduction: benchmark the SAME 3D pipeline with two models that differ ONLY
# at the I/O boundary:
#   A) FP32-IO  = model/dynunet_retrained_fp16.onnx       (keep_io_types=True)
#   B) FP16-IO  = model/dynunet_retrained_fp16io.onnx     (make with export_fp16io.py)
# The C++ loader auto-detects each model's input dtype, so no rebuild is needed.
# "No speedup" = mean TOTAL for A and B are within noise of each other.
#
# Prereq for variant B (needs the FP32 master; regenerate via export_onnx.py):
#   python failed-opts/table4/export_fp16io.py --src model/dynunet_retrained.onnx \
#                                        --dst model/dynunet_retrained_fp16io.onnx
#
# Run:   bash failed-opts/table4/row1_fp16_io.sh
# =============================================================================
source "$(dirname "$0")/common.sh"
need_bin

# Use the retrained checkpoint (reproduces paper Dice 0.9105). FP32IO is the
# shipped FP16-weight model; FP16IO is its FP16-input/output twin.
FP32IO="${MODEL3D:-$ROOT/model/dynunet_retrained_fp16.onnx}"   # FP32 I/O, FP16 weights
FP16IO="${FP16IO:-$ROOT/model/dynunet_retrained_fp16io.onnx}"  # FP16 I/O, FP16 weights
MASTER_DEFAULT="$ROOT/model/dynunet_retrained.onnx"            # FP32 master to convert
OUT="$OUTROOT/row1_fp16_io.csv"
LOGD="$OUTROOT/row1_logs"; mkdir -p "$LOGD"
echo "io_dtype,model,mean_total_s,mean_dice,n_cases" > "$OUT"

if [ ! -f "$FP16IO" ]; then
    MASTER="${MASTER:-$MASTER_DEFAULT}"
    if [ -f "$MASTER" ]; then
        echo "[row1] generating FP16-IO model from $MASTER ..."
        "$PYBIN" "$(dirname "$0")/export_fp16io.py" --src "$MASTER" --dst "$FP16IO" || exit 1
    else
        echo "[ERROR] $FP16IO missing and no FP32 master at $MASTER."
        echo "        regenerate the master via training/3d_dynunet/export_onnx.py first."
        exit 1
    fi
fi

bench() {  # $1=label  $2=model
    local label="$1" model="$2" totals="" dices=""
    for c in $CASES; do
        read -r d t i l <<<"$(run_case "$BIN" "$c" "$LOGD/${label}_c${c}.log" \
                                --gate --onnx-model "$model")"
        totals="$totals$t"$'\n'; dices="$dices$d"$'\n'
    done
    local mt md n; mt="$(printf "%s" "$totals" | mean)"; md="$(printf "%s" "$dices" | mean)"
    n="$(echo $CASES | wc -w)"
    echo "$label,$(basename "$model"),$mt,$md,$n" >> "$OUT"
    printf "%-9s %-34s total=%-8s dice=%-8s\n" "$label" "$(basename "$model")" "$mt" "$md"
}

echo "=== Table IV row 1: FP16 input/output (no speedup) ==="
echo "cases: $CASES"
bench "FP32-IO" "$FP32IO"
bench "FP16-IO" "$FP16IO"

echo
echo "[interpretation] paper says NO SPEEDUP -> the two mean_total values should"
echo "                 match within run-to-run noise; Dice is unchanged."
echo "wrote $OUT"
