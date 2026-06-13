#!/usr/bin/env python3
# =============================================================================
# Helper for Table IV row 4 ("input size 256->224 : Rejected").
#
# The shipped model/airway_2p5d_unet.onnx was exported with a fixed 256x256 spatial
# input (only the batch axis is dynamic). To feed 224x224 slices we make the two
# spatial axes dynamic, so ONNX Runtime accepts any HxW (the 2D U-Net is fully
# convolutional; 224 = 32*7 is divisible by the pooling factor). This is a pure
# graph-metadata edit -- no retraining, weights untouched.
#
# Usage:
#   python failed-opts/table4/export_25d_dynamic.py \
#       --src model/airway_2p5d_unet.onnx \
#       --dst model/airway_2p5d_unet_dyn.onnx
# =============================================================================
import argparse, sys
import onnx

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="model/airway_2p5d_unet.onnx")
ap.add_argument("--dst", default="model/airway_2p5d_unet_dyn.onnx")
a = ap.parse_args()

try:
    m = onnx.load(a.src)
except Exception as e:
    sys.exit(f"[25d-dyn] cannot load {a.src}: {e}")

def make_spatial_dynamic(value_info, hname, wname):
    # tensor shape is [B, C, H, W]; free axes 2 (H) and 3 (W).
    dims = value_info.type.tensor_type.shape.dim
    if len(dims) >= 4:
        dims[2].ClearField("dim_value"); dims[2].dim_param = hname
        dims[3].ClearField("dim_value"); dims[3].dim_param = wname

for inp in m.graph.input:
    make_spatial_dynamic(inp, "H", "W")
for out in m.graph.output:
    make_spatial_dynamic(out, "H", "W")

# Drop any baked intermediate shapes so ORT re-infers them at runtime.
del m.graph.value_info[:]

onnx.save(m, a.dst)
print(f"[25d-dyn] wrote {a.dst}  (H,W axes now dynamic)")
print("[25d-dyn] use it with row4_input_size.sh (MODEL25D=224 build)")
