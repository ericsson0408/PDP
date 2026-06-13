#!/usr/bin/env python3
# =============================================================================
# Helper for Table IV row 1 ("FP16 input/output : No speedup").
#
# model/dynunet_retrained_fp16.onnx has FP16 internal weights but FP32
# input/output (keep_io_types=True -> cast nodes at the boundary), so the
# C++ host feeds plain float32 and never converts. This script produces the
# OTHER variant: FP16 weights AND FP16 input/output (keep_io_types=False). The
# C++ loader (src/onnx_infer.cpp) auto-detects the FP16 input type and runs its
# host-side f32->f16 conversion loop instead -- exactly the "only host I/O
# changed" knob the paper rejected.
#
# Needs the FP32 master ONNX as source. If you don't have it, regenerate it
# first with training/3d_dynunet/export_onnx.py (exports model/dynunet_retrained.onnx).
#
# Usage:
#   python failed-opts/table4/export_fp16io.py \
#       --src model/dynunet_retrained.onnx \
#       --dst model/dynunet_retrained_fp16io.onnx
# =============================================================================
import argparse, sys
import onnx
from onnxconverter_common.float16 import convert_float_to_float16

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="model/dynunet_retrained.onnx",
                help="FP32 master ONNX (from training/3d_dynunet/export_onnx.py)")
ap.add_argument("--dst", default="model/dynunet_retrained_fp16io.onnx")
a = ap.parse_args()

print(f"[fp16io] loading {a.src}")
try:
    m = onnx.load(a.src)
except Exception as e:
    sys.exit(f"[fp16io] cannot load {a.src}: {e}\n"
             f"          regenerate it via training/3d_dynunet/export_onnx.py first.")

# keep_io_types=False  ->  the named input/output become FP16 too.
m16 = convert_float_to_float16(m, keep_io_types=False, disable_shape_infer=False)
onnx.save(m16, a.dst)
print(f"[fp16io] wrote {a.dst}  (FP16 input/output)")
print("[fp16io] now benchmark it against the FP32-IO model with row1_fp16_io.sh")
