#!/bin/bash
# =============================================================================
# run_pipeline.sh -- run any of the four paper pipelines through ONE C++ binary.
#
# All four pipelines are the same ./airway_seg binary; they differ only in the
# post-processor and which (if any) neural model is loaded. The readable flags
# below are aliases baked into the binary (src/main.cpp) -- the old internal
# version tags still work for anyone reading historical logs:
#
#   pipeline       flag(s)                          old tag   model
#   --------       ------------------------------   -------   ---------------------------
#   classical      --classical                      v8        (none)
#   unet3d         --gate   --gpu-softmax            v16       dynunet_retrained_fp16.onnx
#   hybrid         --hybrid --gpu-softmax            v17       dynunet_retrained_fp16.onnx
#   unet25d        --model-2p5d --gate --gpu-softmax v16-2     airway_2p5d_unet.onnx
#   unet25d_hybrid --model-2p5d --hybrid --gpu-softmax v17-2   airway_2p5d_unet.onnx
#
# Paper results (mean Dice / s per case, 120 AIIB23 cases, one V100):
#   classical 0.6932 / 4.74   unet3d 0.9105 / 7.63
#   hybrid    0.9074 / 7.67   unet25d 0.8634 / 4.01
#
# Usage:
#   bash pipelines/run_pipeline.sh unet3d 110          # one case
#   bash pipelines/run_pipeline.sh hybrid 168          # the single safety-abort case
#   bash pipelines/run_pipeline.sh classical 110 111   # several cases
#   bash pipelines/run_pipeline.sh unet3d all          # full 120-case benchmark
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source env.sh

PIPELINE="${1:-}"; shift || true
if [ -z "$PIPELINE" ]; then
    echo "usage: bash pipelines/run_pipeline.sh <classical|unet3d|hybrid|unet25d|unet25d_hybrid> <case...|all>"
    exit 1
fi

# --- common env: skip CUDA-aware MPI probing, OpenMP 8 threads pinned to cores
export OMPI_MCA_opal_cuda_support=0
export OMPI_MCA_btl_openib_warn_no_device_params_found=0
export OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores

DYNUNET="model/dynunet_retrained_fp16.onnx"
UNET25D="model/airway_2p5d_unet.onnx"

# --- per-pipeline launch command (everything before --input/--output/--gt) ----
case "$PIPELINE" in
    classical)
        # classical uses the MPI Z-slab decomposition (4 ranks); no model.
        LAUNCH=(mpirun -np 4 --bind-to socket --map-by socket ./airway_seg --classical)
        ;;
    unet3d)
        export LD_LIBRARY_PATH="third_party/onnxruntime-linux-x64-gpu-1.19.2/lib:${HOME}/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
        LAUNCH=(./airway_seg --gate --gpu-softmax --no-stream-overlap --onnx-model "$DYNUNET")
        ;;
    hybrid)
        export LD_LIBRARY_PATH="third_party/onnxruntime-linux-x64-gpu-1.19.2/lib:${HOME}/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
        LAUNCH=(./airway_seg --hybrid --gpu-softmax --no-stream-overlap --onnx-model "$DYNUNET")
        ;;
    unet25d)
        export LD_LIBRARY_PATH="third_party/onnxruntime-linux-x64-gpu-1.19.2/lib:${HOME}/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
        LAUNCH=(./airway_seg --model-2p5d --gate --gpu-softmax --onnx-model "$UNET25D")
        ;;
    unet25d_hybrid)
        export LD_LIBRARY_PATH="third_party/onnxruntime-linux-x64-gpu-1.19.2/lib:${HOME}/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
        LAUNCH=(./airway_seg --model-2p5d --hybrid --gpu-softmax --onnx-model "$UNET25D")
        ;;
    *)
        echo "unknown pipeline '$PIPELINE' (classical|unet3d|hybrid|unet25d|unet25d_hybrid)"
        exit 1 ;;
esac

OUT="results/$PIPELINE"
mkdir -p "$OUT"

# --- case selection: explicit ids, or 'all' for the full 120-case cohort -------
if [ "${1:-110}" = "all" ]; then
    CASES=$(ls data/img/AIIB23_*.nii.gz | sed -E 's@.*AIIB23_([0-9]+)\.nii\.gz@\1@' | sort -n)
else
    CASES="${@:-110}"
fi

for c in $CASES; do
    IMG="data/img/AIIB23_${c}.nii.gz"; GT="data/gt/AIIB23_${c}.nii.gz"
    [ -f "$IMG" ] || { echo "skip $c (no image)"; continue; }
    echo "=== $PIPELINE / case $c ==="
    "${LAUNCH[@]}" \
        --input  "$IMG" \
        --output "$OUT/mask_${c}.nii.gz" \
        --gt     "$GT" \
        --omp-threads 8
done
