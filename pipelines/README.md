# Pipelines — one binary, four pipelines

All four paper pipelines are the **same** `./airway_seg` binary (built from
`src/`, see the root [`README.md`](../README.md)). They differ only in the
post-processor and which neural model is loaded. A single dispatcher runs any of
them:

```bash
bash pipelines/run_pipeline.sh <pipeline> <case...|all>
```

| pipeline | what it is | flags | model | paper Dice / s·case⁻¹ |
|---|---|---|---|---|
| `classical` | MPI Z-slab + CUDA Frangi + OpenMP region grow (no NN) | `--classical` | — | 0.6932 / 4.74 |
| `unet3d` | DynUNet, 128³ patches, dynamic-threshold gate | `--gate --gpu-softmax --no-stream-overlap` | `dynunet_retrained_fp16.onnx` | **0.9105** / 7.63 |
| `hybrid` | DynUNet oracle + auditable classical region-grow actuator | `--hybrid --gpu-softmax --no-stream-overlap` | `dynunet_retrained_fp16.onnx` | 0.9074 / 7.67 |
| `unet25d` | self-trained 2.5D U-Net (3 adjacent slices), gate post-proc | `--model-2p5d --gate --gpu-softmax` | `airway_2p5d_unet.onnx` | 0.8634 / **4.01** |
| `unet25d_hybrid` | 2.5D oracle + classical actuator (stats only, not a main-table row) | `--model-2p5d --hybrid --gpu-softmax` | `airway_2p5d_unet.onnx` | 0.8695 / 5.42 |

### Readable flags ↔ internal version tags
The flags above are **aliases** added to the binary; the old internal tags still
work, so historical logs (`[v16] gate FIRED ...`, `[v17] SAFETY ABORT ...`) stay
readable:

| readable | old tag | meaning |
|---|---|---|
| `--classical` | `--v8` | classical image-processing pipeline |
| `--gate` | `--v16` | dynamic-threshold gate post-processor |
| `--hybrid` | `--v17` | oracle/actuator hybrid post-processor |
| `--model-2p5d` | `--unet25d` | load the 2.5D model, slice-by-slice inference |
| `--no-stream-overlap` | `--v12-no-overlap` | disable the (no-op) H2D stream overlap |

### Examples
```bash
bash pipelines/run_pipeline.sh classical 110        # one case
bash pipelines/run_pipeline.sh hybrid    168        # the single safety-abort case
bash pipelines/run_pipeline.sh unet3d    all        # full 120-case benchmark
```
Each run prints a per-stage timing block and `Dice=...`; masks land in
`results/<pipeline>/mask_<case>.nii.gz`. Neural pipelines run on a **single rank**
(the paper shows extra MPI ranks are dead work once one rank saturates the GPU).

> Models are fetched by `data/download_models.sh` into `model/`; the dataset by
> `data/download_ct.sh` into `data/img` + `data/gt`. Build the binary first with
> `source env.sh && make`.
