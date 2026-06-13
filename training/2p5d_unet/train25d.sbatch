#!/bin/bash
#SBATCH --job-name=train25d
#SBATCH --account=ACD115083
#SBATCH --partition=gtest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:28:00
#SBATCH --output=results/train25d_%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source training/2p5d_unet/env.sh
module load miniconda3/conda24.5.0_py3.9 2>/dev/null || true
export PATH="$HOME/conda_v9/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/conda_v9/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=4
echo "[train25d] host=$(hostname) GPUs=$(nvidia-smi -L|wc -l) args=$*"
# pthread 噪音導向 stderr 檔，python 進度直接進主 log（不過濾、不緩衝）
python -u training/2p5d_unet/train_unet25d.py "$@" 2> results/train25d_${SLURM_JOB_ID}.err
echo "[train25d] exit=$? DONE $(date '+%T')"
