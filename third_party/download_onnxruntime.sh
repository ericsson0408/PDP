#!/bin/bash
# =============================================================================
# Download the prebuilt ONNX Runtime GPU 1.19.2 that the C++ engine links
# against. This is ~500 MB and is NOT shipped inside the package, so the engine
# build (`make`) expects it here:
#
#     third_party/onnxruntime-linux-x64-gpu-1.19.2/
#         include/   <- ORT C++ headers   (Makefile: ORT_INC)
#         lib/       <- libonnxruntime.so  (Makefile: ORT_LIB)
#
# Run from the package root OR from third_party/ -- it places the dir correctly.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

VER=1.19.2
TARBALL="onnxruntime-linux-x64-gpu-${VER}.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VER}/${TARBALL}"

if [ -d "onnxruntime-linux-x64-gpu-${VER}" ]; then
    echo "[ort] already present: third_party/onnxruntime-linux-x64-gpu-${VER}"
    exit 0
fi

echo "[ort] downloading $URL ..."
wget -q --show-progress "$URL" -O "$TARBALL"
echo "[ort] extracting ..."
tar -xzf "$TARBALL"
rm -f "$TARBALL"
echo "[ort] done -> third_party/onnxruntime-linux-x64-gpu-${VER}"
