#!/usr/bin/env python3
# =============================================================================
# export_onnx.py  --  Export the trained DynUNet to ONNX for the V11 C++ engine.
#
# Produces two files from one .pth checkpoint:
#   1. FP32 ONNX  (opset 17, dynamic spatial axes)
#   2. FP16 mixed-precision ONNX via onnxconverter_common.float16, with
#      keep_io_types=True so the named input/output stay tensor(float) (FP32)
#      and only the internal weights/activations become tensor(float16).
#      -> the C++ IOBinding path keeps feeding/reading FP32 with no host-side
#         conversion loop; TensorCores light up for the Conv3D/GroupNorm math.
#
# I/O contract pinned to src/onnx_infer.cpp:
#   * input  : "input"   [N, 1, D, H, W]  FP32, normalised to [0,1]
#   * output : "output"  [N, 2, D, H, W]  raw logits (C++ does softmax(ch=2))
#   * C++ binds by INDEX 0 (GetInput/OutputNameAllocated(0, ...)), so the node
#     names are cosmetic -- but we use exactly the names from the spec.
#   * dynamic axes on 0 (batch), 2/3/4 (d/h/w) for sliding-window patch sizes.
#
# By default this writes the canonical names (dynunet_retrained*.onnx);
# model/dynunet_retrained_fp16.onnx is exactly what the C++ pipeline loads.
# The script backs up any existing target to *.bak first so a current model
# is never lost.
# =============================================================================

import os
import shutil
import argparse

import numpy as np
import torch

from monai.networks.nets import DynUNet


def build_model():
    # Identical architecture to train.py (affine instance norm, no res block,
    # deep_supervision off -> single raw-logit head).
    return DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        kernel_size=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        filters=[32, 64, 128, 256, 320],
        norm_name=("INSTANCE", {"affine": True}),
        deep_supervision=False,
        res_block=False,
    )


def backup(path):
    if os.path.exists(path):
        bak = path + ".bak"
        shutil.copy2(path, bak)
        print(f"[backup] {path} -> {bak}")


def export_fp32(model, onnx_path, patch):
    dummy = torch.randn(1, 1, *patch)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"[trace] in={tuple(dummy.shape)} out={tuple(out.shape)}")
    assert out.shape[1] == 2, "model must output 2-channel logits"

    backup(onnx_path)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "d", 3: "h", 4: "w"},
            "output": {0: "batch", 2: "d", 3: "h", 4: "w"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[export] FP32 ONNX -> {onnx_path}")
    return dummy, out


def convert_fp16(fp32_path, fp16_path):
    import onnx
    from onnxconverter_common.float16 import convert_float_to_float16

    print(f"[fp16] loading {fp32_path}")
    model = onnx.load(fp32_path)
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,        # <-- input/output stay FP32 for C++ IOBinding
        disable_shape_infer=False,
    )
    backup(fp16_path)
    onnx.save(model_fp16, fp16_path)
    print(f"[fp16] FP16 ONNX -> {fp16_path}")


def verify(fp32_path, fp16_path, torch_dummy, torch_out):
    try:
        import onnxruntime as ort
    except Exception as e:
        print(f"[verify] onnxruntime unavailable, skipped ({e})")
        return
    x = torch_dummy.numpy().astype(np.float32)

    s32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    i_t = s32.get_inputs()[0].type
    o_t = s32.get_outputs()[0].type
    y32 = s32.run(None, {"input": x})[0]
    diff = np.abs(y32 - torch_out.detach().numpy()).max()
    print(f"[verify-fp32] in={i_t} out={o_t} shape={y32.shape} max|onnx-torch|={diff:.3e}")

    s16 = ort.InferenceSession(fp16_path, providers=["CPUExecutionProvider"])
    i16 = s16.get_inputs()[0].type
    o16 = s16.get_outputs()[0].type
    y16 = s16.run(None, {"input": x})[0]
    d16 = np.abs(y16.astype(np.float32) - y32).max()
    print(f"[verify-fp16] in={i16} out={o16} shape={y16.shape} dtype={y16.dtype} max|fp16-fp32|={d16:.3e}")
    assert i16 == "tensor(float)" and o16 == "tensor(float)", \
        "FP16 model IO must remain tensor(float) (keep_io_types=True)"
    print("[verify] OK: FP16 IO is FP32, internal weights FP16 (C++ IOBinding safe)")


def main():
    ap = argparse.ArgumentParser(description="Export DynUNet to FP32 + FP16 ONNX")
    ap.add_argument("--ckpt", default="/home/u4309334/Project/model/dynunet_retrained_best.pth")
    ap.add_argument("--onnx_fp32", default="/home/u4309334/Project/model/dynunet_retrained.onnx")
    ap.add_argument("--onnx_fp16", default="/home/u4309334/Project/model/dynunet_retrained_fp16.onnx")
    ap.add_argument("--patch", type=int, default=128, help="trace patch size (D=H=W)")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"[fatal] checkpoint not found: {args.ckpt}")

    model = build_model()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    # strict=False: DynUNet exposes the down/up blocks twice via skip_layers;
    # the duplicate keys alias the same Parameters, so a single load populates both.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded {args.ckpt}  missing={len(missing)} unexpected={len(unexpected)}")

    patch = (args.patch, args.patch, args.patch)
    dummy, out = export_fp32(model, args.onnx_fp32, patch)
    convert_fp16(args.onnx_fp32, args.onnx_fp16)
    verify(args.onnx_fp32, args.onnx_fp16, dummy, out)
    print("\n[done] To deploy into the C++ pipeline (backs up the old model):")
    print(f"       python export_onnx.py --ckpt {args.ckpt} \\")
    print(f"           --onnx_fp32 model/dynunet_retrained.onnx \\")
    print(f"           --onnx_fp16 model/dynunet_retrained_fp16.onnx")


if __name__ == "__main__":
    main()
