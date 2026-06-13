#!/usr/bin/env python3
# Convert the FP32 DynUNet ONNX to mixed-precision FP16. Keeps the input
# and output as FP32 (keep_io_types=True) so the C++ side can continue to
# feed normalised float32 patches without any host-side conversion. The
# internal weights/activations are FP16 so the V100's TensorCores light
# up — typically 1.5–2× faster than the FP32 model for the same Dice.

import onnx
from onnxconverter_common.float16 import convert_float_to_float16

SRC = "model/dynunet_retrained.onnx"
DST = "model/dynunet_retrained_fp16.onnx"

print(f"[fp16] loading {SRC}")
model = onnx.load(SRC)

# Convert. keep_io_types=True ensures the model's named input/output
# remain FP32 (cast nodes inserted at the boundary). This is the easiest
# integration path with the V11 C++ code which has no FP16 conversion
# loop on the host side.
model_fp16 = convert_float_to_float16(
    model,
    keep_io_types=True,
    disable_shape_infer=False,
)
onnx.save(model_fp16, DST)
print(f"[fp16] wrote {DST}")

# Sanity check via ORT.
try:
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 1, 64, 64, 64).astype(np.float32)
    y = sess.run(None, {"input": x})[0]
    print(f"[verify] FP16 ORT output shape={y.shape} dtype={y.dtype}")
except Exception as e:
    print(f"[verify] skipped ({e})")
