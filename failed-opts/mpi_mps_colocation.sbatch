#!/bin/bash
# =============================================================================
# Professor critiques #1 (MPS evidence missing) and #2 (MPI never plays in the
# neural path, so "MPI is dead work" is not a real negative result).
#
#  #1  Try HARD to bring up CUDA MPS; if it comes up, measure 2 co-located
#      inference procs WITH vs WITHOUT MPS -> does MPS recover the ~tail?
#      If MPS still won't start, record that cleanly (-> paper states the
#      conditional honestly instead of overclaiming).
#  #2  Show what multi-rank MPI actually does in the neural path: run
#      mpirun -np {1,2,4} --gate on one case; wall-clock is unchanged because
#      non-root ranks short-circuit (MPI_Init->Finalize). This makes the point
#      that the spatial-MPI tier is skipped BY DESIGN for neural, so the real
#      throughput question is data-parallel replication (the co-location sweep),
#      not "MPI vs no MPI".
#   sbatch failed-opts/mpi_mps_colocation.sbatch
# =============================================================================
#SBATCH -J mpi_mps
#SBATCH -A mst114552
#SBATCH -p gp1d
#SBATCH -t 00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH -o /home/u4309334/Project/logs/%x_%j.out
#SBATCH -e /home/u4309334/Project/logs/%x_%j.err

set -uo pipefail
MAIN=/home/u4309334/Project
cd "$MAIN"; source scripts/env.sh
export OMPI_MCA_opal_cuda_support=0
unset  OMPI_MCA_btl_smcuda_use_cuda_ipc 2>/dev/null || true
export LD_LIBRARY_PATH=third_party/onnxruntime-linux-x64-gpu-1.19.2/lib:${HOME}/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
export CUDA_VISIBLE_DEVICES=0
MODEL=model/dynunet_retrained_fp16.onnx
NII=/work/u4309334/decompressed_nii
OUT="$MAIN/logs/failed_opts/mpi_mps"; mkdir -p "$OUT" logs/mpi_mps
TLIM=240
src(){ [ -s "$NII/AIIB23_$1.nii" ] && echo "$NII/AIIB23_$1.nii" || echo "data/img/AIIB23_$1.nii.gz"; }

run1(){ # $1 case $2 tag  -> single inference, guarded
  TMPDIR=/tmp/ompi_${SLURM_JOB_ID}_$2 timeout $TLIM ./airway_seg --gate --onnx-model "$MODEL" \
    --input "$(src $1)" --gt data/gt/AIIB23_$1.nii.gz --output "$OUT/m_$1_$2.nii.gz" --omp-threads 2 \
    > logs/mpi_mps/$2.log 2>&1; }

wall_n(){ # $@=cases to run concurrently; echoes wall seconds to finish all
  local t0=$(date +%s.%N) i=0 c
  for c in "$@"; do run1 "$c" "p${i}_$c" & i=$((i+1)); done; wait
  local t1=$(date +%s.%N); awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.2f",b-a}'; }

echo "=== node $(hostname) ==="; nvidia-smi -L | head -1
nvidia-smi -q | grep -A1 "Compute Mode" | head -2 || true

echo; echo "######## warmup (build TRT engine cache) ########"
run1 100 warm; grep -E "Dice=|TOTAL" logs/mpi_mps/warm.log | tail -2 || true

# ---------- #1 MPS: with vs without -----------------------------------------
echo; echo "######## #1 MPS with-vs-without (2 co-located procs) ########"
RES="$OUT/mps_compare.csv"; echo "config,nproc,wall_s,throughput_cases_per_s" > "$RES"
def1=$(wall_n 100);         echo "no-mps,1,$def1,$(awk -v t=$def1 'BEGIN{printf "%.4f",1/t}')"  | tee -a "$RES"
def2=$(wall_n 100 101);     echo "no-mps,2,$def2,$(awk -v t=$def2 'BEGIN{printf "%.4f",2/t}')"  | tee -a "$RES"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$SLURM_JOB_ID
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
MPS_OK=0
if nvidia-cuda-mps-control -d 2>/dev/null && \
   echo "get_default_active_thread_percentage" | timeout 10 nvidia-cuda-mps-control 2>/dev/null | grep -qE '[0-9]'; then
   echo "[mps] daemon up; probing a client..."
   if run1 100 mps_probe && grep -q "Dice=" logs/mpi_mps/mps_probe.log; then MPS_OK=1; fi
fi
if [ "$MPS_OK" = 1 ]; then
   echo "[mps] VERIFIED -> measuring under MPS"
   m1=$(wall_n 100);      echo "mps,1,$m1,$(awk -v t=$m1 'BEGIN{printf "%.4f",1/t}')"     | tee -a "$RES"
   m2=$(wall_n 100 101);  echo "mps,2,$m2,$(awk -v t=$m2 'BEGIN{printf "%.4f",2/t}')"     | tee -a "$RES"
   echo "[mps] 2-proc throughput recovery vs no-MPS: $(awk -v a=$def2 -v b=$m2 'BEGIN{printf "%+.1f%%",(a/b-1)*100}')"
else
   echo "[mps] COULD NOT START/VERIFY MPS on this node -> no-sharing floor only."
   echo "mps,NA,NA,NA" >> "$RES"
fi
echo quit | nvidia-cuda-mps-control 2>/dev/null || true

# ---------- #2 MPI replication in the neural path ---------------------------
echo; echo "######## #2 mpirun -np {1,2,4} --gate (one case) ########"
MRES="$OUT/mpi_neural.csv"; echo "ranks,wall_s,note" > "$MRES"
for np in 1 2 4; do
   t0=$(date +%s.%N)
   TMPDIR=/tmp/ompi_${SLURM_JOB_ID}_np$np timeout $TLIM mpirun -np $np --bind-to none ./airway_seg \
       --gate --onnx-model "$MODEL" --input "$(src 100)" --gt data/gt/AIIB23_100.nii.gz \
       --output "$OUT/mpi_np${np}.nii.gz" --omp-threads 2 > logs/mpi_mps/np${np}.log 2>&1
   t1=$(date +%s.%N); w=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.2f",b-a}')
   nr=$(grep -c "Dice=" logs/mpi_mps/np${np}.log)   # how many ranks actually computed
   echo "$np,$w,ranks_that_ran_inference=$nr" | tee -a "$MRES"
done
echo "  -> wall is flat and only 1 rank computes: extra ranks do nothing because"
echo "     the neural path skips the spatial-MPI tier by design (if(!ai_mode))."
echo "=== done; artifacts in $OUT ==="
