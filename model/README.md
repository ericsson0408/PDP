# Models

ONNX inference artifacts loaded by the C++ engine (`airway_seg --onnx-model ...`).
FP16 weights with FP32 I/O (`keep_io_types=True`) so the C++ IOBinding path
feeds/reads FP32 with no host-side conversion while TensorCores do the math.

**These files are not bundled** — fetch them with:
```bash
bash data/download_models.sh        # -> model/*.onnx (~62 MB)
```

| file | used by | notes |
|------|---------|-------|
| `dynunet_retrained_fp16.onnx` | `unet3d` + `hybrid` pipelines, and the MPS saturation study | DynUNet, 128³ patch, 2-class logits; reproduces the paper's 0.91 Dice |
| `airway_2p5d_unet.onnx` | `unet25d` + `unet25d_hybrid` pipelines | 2-D U-Net, 3 adjacent axial slices in / 2-channel out |

## Reproducing the weights from scratch

- **3D DynUNet** → [`training/3d_dynunet/`](../training/3d_dynunet/):
  `train.py` (trains + checkpoints) then `export_onnx.py` (→ FP32 + FP16 ONNX).
  `train_retrained_log.json` there is the per-epoch loss / val-Dice history.
- **2.5D U-Net** → [`training/2p5d_unet/`](../training/2p5d_unet/):
  `prep_slices.py` then `train_unet25d.py` (trains + exports ONNX).

## Full checkpoint set (`.pth` / `.pt`)

Only the FP16 inference ONNX files are distributed (via `download_models.sh`).
The full PyTorch checkpoints and FP32 ONNX live alongside the dataset in the
shared Google Drive — they are only needed to resume/fine-tune training, not to
run any pipeline.
