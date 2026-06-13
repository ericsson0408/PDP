#!/bin/bash
# =============================================================================
# download_models.sh -- fetch the inference ONNX models used by the neural
# pipelines (3D DynUNet + self-trained 2.5D U-Net) into model/.
#
# After this script finishes you should have:
#     model/dynunet_retrained_fp16.onnx   (3D DynUNet, FP16 weights / FP32 I/O)
#     model/airway_2p5d_unet.onnx         (2.5D U-Net, 3-channel in / 2-channel out)
# together ~62 MB. These are loaded by:
#     pipelines/run_pipeline.sh unet3d|hybrid          -> dynunet_retrained_fp16.onnx
#     pipelines/run_pipeline.sh unet25d|unet25d_hybrid -> airway_2p5d_unet.onnx
#
# Usage:
#     bash data/download_models.sh
#
# Requirements: python3 + `pip install gdown`.
# (To rebuild these from scratch instead, see training/3d_dynunet/ and
#  training/2p5d_unet/.)
# =============================================================================
set -euo pipefail

# Repo root = parent of this script's directory.
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

FOLDER_ID="1PbvYLDMz3ETFT11TCaGkAl0Ly1MKjyd4"   # shared Drive folder (3D + 2.5D ONNX)
mkdir -p model

echo "[models] target dir: $HERE/model"
echo "[models] pulling the shared Drive folder with gdown ..."
if command -v gdown >/dev/null 2>&1 || python3 -c "import gdown" 2>/dev/null; then
    python3 -m gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O model || {
        echo "[models] folder pull failed -- install gdown or download manually (see below)."; }
else
    echo "[models] gdown not found. Install it (pip install gdown) and re-run, or"
    echo "         download the two ONNX files manually from:"
    echo "         https://drive.google.com/drive/folders/${FOLDER_ID}"
fi

# gdown --folder may nest files under model/<foldername>/; flatten any *.onnx up.
find model -mindepth 2 -type f -name '*.onnx' -exec mv -n {} model/ \; 2>/dev/null || true
find model -mindepth 1 -type d -empty -delete 2>/dev/null || true

echo "[models] model/ now contains:"
ls -1 model/*.onnx 2>/dev/null || echo "  (no .onnx found -- download manually from the folder above)"
echo
echo "[models] Expected: dynunet_retrained_fp16.onnx + airway_2p5d_unet.onnx"
echo "         If filenames differ, rename them to match what run_pipeline.sh loads."
