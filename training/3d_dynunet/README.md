# Reproducing the 3D DynUNet weights

Trains a MONAI **DynUNet** whose ONNX export drops straight into the C++ engine
(`--gate` / `--hybrid`) with no C++ changes. Architecture, intensity normalisation
and 2-class raw-logit output are pinned to match `src/onnx_infer.cpp`.

## Consistency contract with the C++ side
- intensity: `ScaleIntensityRange [-1000, 600] -> [0, 1]`, clip=True
- output: raw logits `[B, 2, D, H, W]` (C++ does softmax(ch=2))
- arch: DynUNet filters `[32,64,128,256,320]`, strides `[1,2,2,2,2]`, instance norm, `deep_supervision=False`

## Environment
Python 3.9 + PyTorch + MONAI (the cluster used `miniconda3/conda24.5.0_py3.9`).
```bash
pip install "torch" monai nibabel onnx onnxconverter-common
```

## Data
Provide AIIB23 at `data/img/AIIB23_*.nii.gz` + `data/gt/AIIB23_*.nii.gz`
(root `data/download_data.sh`). Split: seed-fixed patient-level 108 train / 12 val.

## Run
```bash
# (local) train, then export ONNX
python training/3d_dynunet/train.py --tag retrained --max_epochs 600 --val_interval 10
python training/3d_dynunet/export_onnx.py \
    --ckpt model/dynunet_retrained_best.pth \
    --onnx_fp32 model/dynunet_retrained.onnx \
    --onnx_fp16 model/dynunet_retrained_fp16.onnx

# (SLURM / TWCC) does both in one job:
sbatch training/3d_dynunet/train_airway.sbatch
# quick smoke test:
sbatch -p gtest -t 00:28:00 \
  --export=ALL,EPOCHS=2,VAL_INTERVAL=1,LIMIT_TRAIN=8,LIMIT_VAL=2,SAMPLES=1,GRAD_ACCUM=2,TAG=smoke \
  training/3d_dynunet/train_airway.sbatch
```
Outputs `model/dynunet_<tag>_best.pth` + `model/dynunet_<tag>{,_fp16}.onnx`.
Held-out validation Dice for the paper checkpoint: **0.8984**.

> Note: `train_airway.sbatch` / `train.py` write to `model/` and read `data/`
> at the **repo root**. Run/submit from the package root (the sbatch `cd`s to a
> hard-coded path — edit `cd` and `#SBATCH --account=` for your setup).
