#!/bin/bash
# =============================================================================
# Reproduce paper Table IV rows 1-5 on one V100 (TWCC gp1d).
# Runs against the MAIN project tree (built binary + ORT + models + data),
# driving the failed-opts/table4/ scripts via TABLE4_ROOT.
#   sbatch PDP_Team23_Reproducible/failed-opts/table4/run_all.sbatch
# =============================================================================
#SBATCH -J table4
#SBATCH -A mst114552
#SBATCH -p gp1d
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH -o /home/u4309334/Project/logs/%x_%j.out
#SBATCH -e /home/u4309334/Project/logs/%x_%j.err

set -uo pipefail
MAIN=/home/u4309334/Project
cd "$MAIN"
source scripts/env.sh

export TABLE4_ROOT="$MAIN"
export PYBIN=/opt/ohpc/twcc/conda/24.5.0/miniconda3/bin/python
export NIICACHE=/work/u4309334/decompressed_nii
export OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores
export SKIP_ENV=1                       # scripts/env.sh already sourced above
# OpenMPI here is not CUDA-aware; disable the cuda-support request (as the
# reference benchmark scripts do) or MPI_Init aborts.
export OMPI_MCA_opal_cuda_support=0
unset  OMPI_MCA_btl_smcuda_use_cuda_ipc 2>/dev/null || true
T4="$MAIN/PDP_Team23_Reproducible/failed-opts/table4"
SMOKE="100 101 102 110 120 124 130 145 150 158 168 30"

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-NA} ==="
nvidia-smi -L | head -1
mkdir -p "$MAIN/logs"

# ---- 0. smoke: confirm the binary emits Dice/TOTAL before the heavy rows -----
echo; echo "######## SMOKE (1 case, fail-fast) ########"
SRC="$NIICACHE/AIIB23_100.nii"; [ -f "$SRC" ] || SRC="$MAIN/data/img/AIIB23_100.nii.gz"
"$MAIN/airway_seg" --gate --onnx-model model/dynunet_retrained_fp16.onnx \
    --input "$SRC" --output /tmp/t4_smoke.nii.gz --gt data/gt/AIIB23_100.nii.gz \
    --omp-threads 4 > /tmp/t4_smoke.log 2>&1
tail -8 /tmp/t4_smoke.log
SD="$(grep -oE "Dice=[0-9.]+" /tmp/t4_smoke.log | tail -1 | cut -d= -f2)"
[ -n "$SD" ] || { echo "[FATAL] smoke produced no Dice; aborting."; exit 1; }
awk -v d="$SD" 'BEGIN{exit !(d>=0.8)}' || { echo "[FATAL] smoke Dice=$SD < 0.8 (wrong model?); aborting."; exit 1; }
echo "[smoke] OK  Dice=$SD"

# ---- rows --------------------------------------------------------------------
echo; echo "######## ROW 3  (MPI+MPS, analysis) ########"
bash "$T4/row3_mpi_mps.sh"            || echo "[row3] non-zero"

echo; echo "######## ROW 2  (zstd vs gzip, microbench) ########"
bash "$T4/row2_zstd_vs_gzip.sh"      || echo "[row2] non-zero"

echo; echo "######## ROW 5  (per-case batch tuning) ########"
CASES="$SMOKE" bash "$T4/row5_batch_tuning.sh" || echo "[row5] non-zero"

echo; echo "######## ROW 1  (FP16 input/output) ########"
CASES="$SMOKE" bash "$T4/row1_fp16_io.sh"      || echo "[row1] non-zero"

echo; echo "######## ROW 4  (input 256->224, FULL 120 cases) ########"
CASES=all bash "$T4/row4_input_size.sh"        || echo "[row4] non-zero"

echo; echo "######## DONE — per-row CSVs ########"
for f in "$MAIN"/logs/failed_opts/table4/*.csv; do echo "== $f =="; cat "$f"; echo; done
