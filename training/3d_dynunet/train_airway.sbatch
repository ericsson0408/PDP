#!/bin/bash
# =============================================================================
# train_airway.sbatch -- run DynUNet airway training + ONNX export on a TWCC
# Taiwania-2 GPU compute node (NOT the login node: its 20 GB per-user cgroup
# OOM-kills the data pipeline). Compute nodes give 8x V100 + 738 GB RAM.
#
# Submit (full run, defaults below):
#   sbatch scripts/train_airway.sbatch
#
# Submit (short validation on the 30-min test partition):
#   sbatch -p gtest -t 00:28:00 \
#     --export=ALL,EPOCHS=2,VAL_INTERVAL=1,LIMIT_TRAIN=8,LIMIT_VAL=2,SAMPLES=1,GRAD_ACCUM=2,TAG=smoke \
#     scripts/train_airway.sbatch
#
# Override any knob with --export=ALL,KEY=VAL,... at submit time.
# =============================================================================
#SBATCH -J airway_train
#SBATCH -A mst114552
#SBATCH -p gp4d
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=90G
#SBATCH -o /home/u4309334/Project/logs/%x_%j.out
#SBATCH -e /home/u4309334/Project/logs/%x_%j.err

set -euo pipefail
cd /home/u4309334/Project

source /etc/profile.d/modules.sh
module purge
module load miniconda3/conda24.5.0_py3.9
PY=/opt/ohpc/twcc/conda/24.5.0/miniconda3/bin/python

# ---- knobs (env-overridable via sbatch --export) ----------------------------
EPOCHS=${EPOCHS:-600}
VAL_INTERVAL=${VAL_INTERVAL:-10}
SAMPLES=${SAMPLES:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
WORKERS=${WORKERS:-4}
TAG=${TAG:-retrained}
LIMIT_TRAIN=${LIMIT_TRAIN:-0}
LIMIT_VAL=${LIMIT_VAL:-0}
DO_EXPORT=${DO_EXPORT:-1}
CACHE_DIR=${CACHE_DIR:-/work/u4309334/airway_cache}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

echo "[sbatch] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-NA} tag=$TAG epochs=$EPOCHS workers=$WORKERS"
echo "[sbatch] python=$($PY --version 2>&1)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

$PY train.py \
    --tag "$TAG" \
    --max_epochs "$EPOCHS" \
    --val_interval "$VAL_INTERVAL" \
    --samples_per_image "$SAMPLES" \
    --grad_accum "$GRAD_ACCUM" \
    --num_workers "$WORKERS" \
    --cache_dir "$CACHE_DIR" \
    --limit_train "$LIMIT_TRAIN" \
    --limit_val "$LIMIT_VAL"

if [ "$DO_EXPORT" = "1" ]; then
    BEST="/home/u4309334/Project/model/dynunet_${TAG}_best.pth"
    echo "[sbatch] exporting ONNX from $BEST"
    $PY export_onnx.py \
        --ckpt "$BEST" \
        --onnx_fp32 "/home/u4309334/Project/model/dynunet_${TAG}.onnx" \
        --onnx_fp16 "/home/u4309334/Project/model/dynunet_${TAG}_fp16.onnx"
fi

echo "[sbatch] DONE"
