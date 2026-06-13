# High-Throughput 3D Airway Segmentation — Reproducibility Package

**Team 23 · 114-2 Parallel and Distributed Programming (NTU, Spring 2026)**
Paper: *An Evolution From Heuristics to Neural Pipelines.*

This package reproduces **all results in the paper**: four airway-segmentation
pipelines compared on AIIB23 (120 chest-CT cases, one NVIDIA V100) under an
identical I/O / timing harness. **All four pipelines are one C++ binary**
(`airway_seg`) selected by a flag — the only differences are the post-processor
and which (if any) neural model is loaded.

| Pipeline | flag(s) | model | Paper result (Dice / s·case⁻¹) |
|----------|---------|-------|--------------------------------|
| **Classical**  | `--classical`                              | none                                | 0.6932 / 4.74 |
| **3D U-Net**   | `--gate --gpu-softmax --no-stream-overlap` | `model/dynunet_retrained_fp16.onnx` | **0.9105** / 7.63 |
| **Hybrid**     | `--hybrid --gpu-softmax --no-stream-overlap` | `model/dynunet_retrained_fp16.onnx` | 0.9074 / 7.67 |
| **2.5D U-Net** | `--model-2p5d --gate --gpu-softmax`        | `model/airway_2p5d_unet.onnx`       | 0.8634 / **4.01** |

> The readable flags above are aliases baked into the binary. The old internal
> version tags (`--v8/--v16/--v17/--unet25d/--v12-no-overlap`) still work, so
> historical logs (`[v16] gate FIRED`, `[v17] SAFETY ABORT`) remain readable.

---

## What's in this package

```
PDP_Team23_Reproducible/
├── README.md                  ← you are here (master guide)
├── env.sh                     ← cluster module loads (cuda + openmpi)
├── Makefile                   ← builds ./airway_seg from src/
├── src/                       ← SHARED C++/CUDA/MPI/ONNX engine (all 4 pipelines)
│
├── data/
│   ├── download_ct.sh         ← fetch AIIB23 CT + GT -> data/img, data/gt
│   ├── download_models.sh     ← fetch the 3D + 2.5D ONNX models -> model/
│   └── predecompress.sh       ← (optional) .nii.gz -> plain .nii for the fast load path
├── model/                     ← ONNX models (populated by download_models.sh) + README
├── third_party/
│   ├── libdeflate/            ← bundled (NIfTI gunzip)
│   └── download_onnxruntime.sh← fetch ONNX Runtime GPU 1.19.2 (~500 MB, not shipped)
│
├── pipelines/
│   ├── run_pipeline.sh        ← ONE dispatcher: run_pipeline.sh <pipeline> <cases|all>
│   └── README.md
│
├── training/                  ← reproduce the model WEIGHTS from scratch
│   ├── 3d_dynunet/            ← train.py + export_onnx.py (+ SLURM job)
│   └── 2p5d_unet/             ← prep_slices.py + train_unet25d.py (+ SLURM jobs)
│
├── analysis/                  ← CODE for every paper figure/table/stat (reads ../results/)
├── benchmark/                 ← SLURM harnesses for Tables II & III
├── results/                   ← canonical per-case CSVs behind the figures/tables
└── failed-opts/               ← rejected optimizations (Table IV) + negative analyses
```

---

## Quick start

### 0. Prerequisites
- A Linux host with an NVIDIA GPU (paper used a **V100**, `sm_70`; CUDA ≥ 7.0).
- Toolchain: CUDA 12.x, OpenMPI 5.x (+UCX), `mpicxx`, `nvcc`, `zlib`.
  On TWCC Taiwania-2: `source env.sh` (`module load cuda/12.8 openmpi/5.0.2_ucx1.14.1_cuda12.3`).
- cuDNN 9 visible at link time (ships with a pip `torch`; see Makefile note).
- For training/analysis: Python 3.9+ with PyTorch + MONAI + pandas/matplotlib/scipy.

### 1. Get ONNX Runtime, the dataset, and the models
```bash
cd PDP_Team23_Reproducible

bash third_party/download_onnxruntime.sh      # ONNX Runtime GPU 1.19.2 (build + run neural)
pip install gdown
bash data/download_ct.sh                       # AIIB23 -> data/img/, data/gt/  (~50 GB)
bash data/download_models.sh                   # 3D + 2.5D ONNX -> model/        (~62 MB)

# (optional, matches the paper's 0.20 s load) pre-decompress to plain .nii
bash data/predecompress.sh data/img /work/$USER/decompressed_nii
```
Model Drive folder: <https://drive.google.com/drive/folders/1PbvYLDMz3ETFT11TCaGkAl0Ly1MKjyd4>

### 2. Build the engine
```bash
source env.sh                 # load modules (skip off-cluster if mpicxx/nvcc are on PATH)
make                          # -> ./airway_seg
# off-cluster you may need to point the linker at your cuDNN:
#   make CUDNN_RPATH=$(python -c "import nvidia.cudnn,os;print(os.path.dirname(nvidia.cudnn.__file__)+'/lib')")
```

### 3. Run any pipeline
```bash
bash pipelines/run_pipeline.sh classical 110       # Classical, one case
bash pipelines/run_pipeline.sh unet3d    110       # 3D U-Net
bash pipelines/run_pipeline.sh hybrid    168       # Hybrid (case 168 = the safety-abort case)
bash pipelines/run_pipeline.sh unet25d   all       # 2.5D U-Net, full 120-case benchmark
```
Each prints a per-stage timing block and `Dice=...`; masks land in
`results/<pipeline>/`. See [`pipelines/README.md`](pipelines/README.md).

### 4. Regenerate the paper's figures/tables
```bash
cd analysis
python plot_accuracy_speed.py     # Dice vs speed (4 pipelines)
python plot_pervoxel.py           # per-voxel cost (38 vs 15 ns/voxel)
python plot_stage_breakdown.py    # per-stage wall-clock
python plot_mps_util.py           # GPU saturation
python stats_tests.py             # Wilcoxon + Cliff's delta
python stats_heldout_timing.py    # held-out val Dice + timing mean/std/95% CI
```
These read the cleaned per-case CSVs in [`results/`](results/) (our final runs),
so they reproduce 4.74/7.63/7.67/4.01 (Dice 0.693/0.910/0.907/0.863) **without** a
GPU. To re-measure on a GPU see [`benchmark/README.md`](benchmark/README.md)
(Tables II & III). Rejected optimizations (Table IV / "what did not help") are in
[`failed-opts/README.md`](failed-opts/README.md).

---

## Reproducing the model weights (optional)

The neural pipelines load ready-to-run ONNX (`download_models.sh`), but both can
be retrained:

- **3D DynUNet** → [`training/3d_dynunet/README.md`](training/3d_dynunet/README.md)
  (`train.py` → `export_onnx.py`; held-out val Dice 0.8984).
- **2.5D U-Net** → [`training/2p5d_unet/README.md`](training/2p5d_unet/README.md)
  (`prep_slices.py` → `train_unet25d.py`; held-out val Dice 0.841).

---

## What you must supply vs. what is fetched

| Item | In repo? | How to get it |
|------|----------|---------------|
| C++/CUDA engine source + Makefile | ✅ | `src/`, `Makefile` |
| Result CSVs for figures/tables | ✅ | `results/` |
| `libdeflate` | ✅ | `third_party/libdeflate/` |
| 3D + 2.5D **inference** ONNX (fp16) | ❌ (~62 MB) | `data/download_models.sh` |
| **ONNX Runtime GPU 1.19.2** | ❌ (~500 MB) | `third_party/download_onnxruntime.sh` |
| **AIIB23 dataset** | ❌ (~50 GB) | `data/download_ct.sh` |
| Full `.pth`/`.pt` training checkpoints | ❌ | Google Drive (only to resume training) |

---

## Environment used for the paper
NVIDIA V100-SXM2 32 GB, driver 535.161.08 · CUDA 12.8 · OpenMPI 5.0.2 (UCX
1.14.1) · ONNX Runtime 1.19.2 with the TensorRT 10 EP in FP16 · TWCC Taiwania-2
(account `mst114552`). Inputs pre-decompressed to plain `.nii`; Dice evaluated
against expert labels over all 120 AIIB23 cases.
