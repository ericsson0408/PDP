#!/bin/bash
# =============================================================================
# Table IV, row 5 -- "per-case batch tuning : No effect"
#   Claim: increasing the ONNX patches-per-Run() batch does not speed up the
#   per-case wall-clock, because the fixed per-case startup cost dominates and
#   the 3D conv scales poorly with the batch dim at the 128^3 patch size.
#
# Reproduction (fully CLI-driven, no rebuild): sweep --onnx-batch over the 3D
# pipeline (--gate) on the same cases and compare mean TOTAL. A "no effect"
# result = the curve is flat (no batch beats batch=1 by a meaningful margin).
#
# Run:   bash failed-opts/table4/row5_batch_tuning.sh
#        CASES="100 101 110 168 124 54" bash failed-opts/table4/row5_batch_tuning.sh
# =============================================================================
source "$(dirname "$0")/common.sh"
need_bin

# The retrained checkpoint is the one that reproduces the paper's 3D Dice 0.9105
# (dynunet_best_model_fp16.onnx is an older, weaker checkpoint ~0.3).
MODEL="${MODEL3D:-$ROOT/model/dynunet_retrained_fp16.onnx}"
BATCHES="${BATCHES:-1 2 4 8}"
OUT="$OUTROOT/row5_batch_tuning.csv"
LOGD="$OUTROOT/row5_logs"; mkdir -p "$LOGD"
echo "batch,mean_total_s,n_cases" > "$OUT"

echo "=== Table IV row 5: per-case batch tuning (3D --gate) ==="
echo "cases: $CASES"
printf "%-8s %-14s\n" "batch" "mean_total(s)"
for b in $BATCHES; do
    totals=""
    for c in $CASES; do
        read -r d t i l <<<"$(run_case "$BIN" "$c" "$LOGD/b${b}_c${c}.log" \
                                --gate --onnx-model "$MODEL" --onnx-batch "$b")"
        totals="$totals$t"$'\n'
    done
    m="$(printf "%s" "$totals" | mean)"
    n="$(echo $CASES | wc -w)"
    echo "$b,$m,$n" >> "$OUT"
    printf "%-8s %-14s\n" "$b" "$m"
done

echo
echo "[interpretation] paper says NO EFFECT -> the mean_total column should be"
echo "                 flat across batch sizes (no batch >~5% faster than b=1)."
echo "wrote $OUT"
