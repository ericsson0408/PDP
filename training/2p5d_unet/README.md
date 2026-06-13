# Reproducing the 2.5D U-Net Training (Team 23, AIIB23 airway)

This package contains **only the code** needed to reproduce our 2.5D U-Net
training. No data, weights, or ONNX files are included (to keep the archive
small). You supply the AIIB23 dataset; the scripts produce the trained weights
(`*_best.pth`) and an exported `*.onnx`.

## 1. Files

```
training/2p5d_unet/
├── README.md                ← this file
├── env.sh                   ← cluster env (module load CUDA/MPI); optional off-cluster
├── prep_slices.py           ← STEP 1: AIIB23 volumes -> 2D-slice memmap
├── train_unet25d.py         ← STEP 2: train the 2.5D U-Net + export ONNX
├── train25d.sbatch          ← SLURM job: quick run (gtest, 28 min)
└── train25d_full.sbatch     ← SLURM job: full run (gp4d, up to 6 h)
```

Run all commands **from the package root** (`PDP_Team23_Reproducible/`); the
SLURM jobs `cd "$SLURM_SUBMIT_DIR"`, so submit them from there too. The scripts
read/write these relative folders at the package root, which you create:

```
data/img/   AIIB23_*.nii.gz          ← input CT volumes (you provide)
data/gt/    AIIB23_*.nii.gz          ← input ground-truth airway masks (you provide)
data/slices25d/                      ← produced by STEP 1 (X.npy, Y.npy, index.csv)
model/                               ← produced by STEP 2 (unet25d_best.pth, unet25d.onnx)
results/                             ← SLURM logs (only if using the .sbatch jobs)
```

## 2. Environment

Python 3.10 with (the exact versions we used):

| package | version |
|---|---|
| torch   | 2.5.1 (+cu121) |
| monai   | 1.5.2 |
| numpy   | 1.26.4 |
| nibabel | 5.4.2 |
| onnx    | 1.21.0 |

```bash
pip install "torch==2.5.1" monai==1.5.2 numpy==1.26.4 nibabel==5.4.2 onnx==1.21.0
```

A CUDA GPU is expected (`--device cuda`); training falls back to CPU but is slow.
`scripts/env.sh` is the TWCC Taiwania-2 module setup (`module load cuda/12.3 ...`)
and is sourced by the `.sbatch` jobs; off-cluster you can ignore it and just use
the `pip`-installed environment.

## 3. How to run

### STEP 0 — download the AIIB23 dataset (~24 GB)
The data is **not** in this archive. Download `AIIB23_Train_T1.zip` (24 GB) from
our Google Drive with `wget`. Because the file is large, Google serves a
virus-scan warning page first, so we fetch the confirm token + uuid, then
download:

```bash
FILEID=1zQnKopDmUY7PGPbQYkBBJRpFKJr_RTxe

# 1) hit the warning page to grab cookies + the confirm token and uuid
wget --quiet --save-cookies /tmp/gdc.txt --keep-session-cookies \
  "https://drive.usercontent.google.com/download?id=${FILEID}&export=download" -O /tmp/gd.html
CONFIRM=$(grep -oE 'name="confirm" value="[^"]+"' /tmp/gd.html | sed -E 's/.*value="([^"]+)".*/\1/')
UUID=$(grep -oE 'name="uuid" value="[^"]+"'       /tmp/gd.html | sed -E 's/.*value="([^"]+)".*/\1/')

# 2) download the real file
wget --load-cookies /tmp/gdc.txt \
  "https://drive.usercontent.google.com/download?id=${FILEID}&export=download&confirm=${CONFIRM}&uuid=${UUID}" \
  -O AIIB23_Train_T1.zip
rm -f /tmp/gdc.txt /tmp/gd.html

# 3) unzip and arrange so the scripts find the volumes/masks at:
#      data/img/AIIB23_*.nii.gz   and   data/gt/AIIB23_*.nii.gz
mkdir -p data
unzip -q AIIB23_Train_T1.zip -d data
# If the zip's internal folder names differ, move them so that CT volumes land in
# data/img/ and ground-truth airway masks in data/gt/ (filenames AIIB23_*.nii.gz).
```

> If Google's token flow changes and the `wget` form breaks, the one-line
> fallback is `pip install gdown && gdown ${FILEID} -O AIIB23_Train_T1.zip`.

### STEP 1 — extract 2D slices (once)
Reads `data/img/AIIB23_*.nii.gz` + `data/gt/`, HU-clips `[-1000,600]→[0,255]`,
resizes each axial slice to `256×256`, and writes a single `uint8` memmap
(`X.npy`, `Y.npy`) plus `index.csv` to `data/slices25d/`.

```bash
python training/2p5d_unet/prep_slices.py --size 256 --out data/slices25d
# (optional smoke test on a few cases: add --limit-cases 4)
```

### STEP 2 — train + export ONNX
Patient-level split (a case's slices never straddle train/val; `seed=42`,
`--val-frac 0.2`). Model: MONAI `BasicUNet` (2D, in=3 adjacent slices,
out=2 channels, features `(32,64,128,256,512,32)`). Loss `DiceCELoss`,
optimizer Adam, `ReduceLROnPlateau` on val Dice, AMP mixed precision, early
stopping. Saves the best checkpoint and exports a dynamic-batch FP32 ONNX.

```bash
python training/2p5d_unet/train_unet25d.py \
  --slices data/slices25d --out-prefix model/unet25d \
  --epochs 80 --patience 12 --batch 32 --val-frac 0.2 --workers 4
```

Outputs: `model/unet25d_best.pth` and `model/unet25d.onnx` (rename/export to
`model/airway_2p5d_unet.onnx` for the C++ pipeline).

### On SLURM (TWCC) instead of the bare commands
Submit **from the package root** (`sbatch` sets `SLURM_SUBMIT_DIR` to where you
submit; the jobs `cd` there, `source training/2p5d_unet/env.sh`, then call the
trainer):

```bash
sbatch training/2p5d_unet/train25d_full.sbatch       # full 80-epoch run on gp4d
# or a short test (pass through trainer args):
sbatch training/2p5d_unet/train25d.sbatch --slices data/slices25d --epochs 3 --out-prefix model/unet25d
```
Edit `#SBATCH --account=...` to your own allocation. Logs land in `results/`.

## 4. What to expect

With the full AIIB23 training set (120 cases, 96 train / 24 val by our split),
training converges in ~20–30 epochs and early-stops. Our run reached a best
**patient-level validation Dice of 0.841** (epoch 17). The exported
`unet25d.onnx` (dynamic batch, FP32 I/O, opset 17) is the file our C++/ONNX
inference pipeline loads for the 2.5D approach.

## 5. Key hyper-parameters (defaults in `train_unet25d.py`)

`slab=3` (adjacent slices as channels) · `batch=32` · `lr=1e-3` ·
`epochs<=80` · `patience=12` · `val-frac=0.2` · `size=256` ·
background-slice down-sampling to 30% of airway slices on the **train** split
only (val uses all airway-containing slices).
