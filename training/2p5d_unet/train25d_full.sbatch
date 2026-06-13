#!/bin/bash
#SBATCH --job-name=train25d_full
#SBATCH --account=ACD115083
#SBATCH --partition=gp4d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=results/train25d_full_%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source training/2p5d_unet/env.sh
module load miniconda3/conda24.5.0_py3.9 2>/dev/null || true
export PATH="$HOME/conda_v9/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/conda_v9/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
echo "[train25d_full] $(hostname) GPUs=$(nvidia-smi -L|wc -l) start=$(date '+%T')"
python -u training/2p5d_unet/train_unet25d.py --slices data/slices25d \
  --epochs 80 --patience 12 --batch 32 --workers 4 --val-frac 0.2 \
  --out-prefix model/unet25d 2> results/train25d_full_${SLURM_JOB_ID}.err
echo "[train25d_full] exit=$? DONE $(date '+%T')"
