#!/bin/bash
# =============================================================================
# MPS / co-location saturation measurement for the paper's "MPI+MPS, >2 ranks"
# rejected row.
#
# Launches N concurrent single-rank airway_seg inference processes (each on a
# DISTINCT case) sharing ONE V100, for N = 1,2,3,4. Samples device SM
# utilization with `nvidia-smi dmon -s u` and records aggregate wall-clock +
# throughput. Tries CUDA MPS for true spatial sharing; if MPS does not come up
# cleanly it falls back to default (time-sliced) co-location so the job still
# produces data. EVERY inference run is timeout-guarded so the job can never
# hang the whole allocation.
#
# Why concurrent processes, not `mpirun -np N`: in the neural (--gate) path only
# rank 0 runs inference; non-root ranks short-circuit. The throughput question
# is data-parallel across cases, so we co-locate N independent processes.
#
# Submit:  sbatch analysis/measure_mps_util.sbatch    (writes results/mps_util.csv)
# =============================================================================
#SBATCH -J mps_util
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
cd /home/u4309334/Project
source scripts/env.sh
export OMPI_MCA_opal_cuda_support=0
unset  OMPI_MCA_btl_smcuda_use_cuda_ipc 2>/dev/null || true
export OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores
export CUDA_VISIBLE_DEVICES=0

MODEL=model/dynunet_retrained_fp16.onnx
NIIDIR=/work/u4309334/decompressed_nii          # plain .nii (zero-copy path)
GTDIR=data/gt
OUTDIR=results/mps_util ; mkdir -p "$OUTDIR" logs/mps_util results
RESCSV=results/mps_util.csv
CASES=(100 101 102 103)                          # 4 distinct mid-size cases
TLIMIT=240                                       # per-run wall-clock guard (s)

echo "=== node $(hostname)  job $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
echo "--- GPU compute mode ---"
nvidia-smi -q | grep -A1 "Compute Mode" | head -2 || true

run_one() {  # $1=case  $2=tag   (each process gets its own OMPI session dir)
    local c="$1" tag="$2"
    TMPDIR="/tmp/ompi_${SLURM_JOB_ID}_${tag}" \
    timeout "$TLIMIT" ./airway_seg \
        --input "$NIIDIR/AIIB23_${c}.nii" \
        --output "$OUTDIR/mask_${c}_${tag}.nii.gz" \
        --gt "$GTDIR/AIIB23_${c}.nii.gz" \
        --omp-threads 4 --gate --onnx-model "$MODEL" \
        > "logs/mps_util/${tag}_case${c}.log" 2>&1
    echo $?      # exit status (124 = timeout)
}

# ---------------------------------------------------------------------------
# Stage 0: NO-MPS warmup. Confirms the binary runs at all and warms the shared
#          TensorRT engine cache. If THIS hangs/fails, MPS is not the problem.
# ---------------------------------------------------------------------------
echo "[stage0] no-MPS warmup (build/load engine cache) ..."
st=$(run_one "${CASES[0]}" warmup)
if [ "$st" = "124" ]; then
    echo "[stage0] FATAL: warmup timed out even without MPS -- aborting."
    tail -20 logs/mps_util/warmup_case${CASES[0]}.log 2>/dev/null
    exit 1
fi
echo "[stage0] warmup exit=$st  (engine cache warm)"
grep -E "Dice=|TOTAL" logs/mps_util/warmup_case${CASES[0]}.log | tail -2 || true

# ---------------------------------------------------------------------------
# Stage 1: try to bring up CUDA MPS; verify it actually responds. Fall back to
#          default co-location if anything is off.
# ---------------------------------------------------------------------------
USE_MPS=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$SLURM_JOB_ID
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if nvidia-cuda-mps-control -d 2>/dev/null; then
    sleep 2
    if echo "get_default_active_thread_percentage" | timeout 10 nvidia-cuda-mps-control 2>/dev/null | grep -qE '[0-9]'; then
        # verify a client can actually run under MPS within the time guard
        echo "[stage1] MPS daemon responds; verifying a client run ..."
        vst=$(run_one "${CASES[0]}" mps_probe)
        if [ "$vst" = "0" ]; then
            USE_MPS=1
            echo "[stage1] MPS verified -> using MPS for co-location."
        else
            echo "[stage1] MPS client run failed/timed out (exit=$vst) -> fallback to default co-location."
        fi
    else
        echo "[stage1] MPS control did not respond -> fallback to default co-location."
    fi
else
    echo "[stage1] could not start MPS daemon -> fallback to default co-location."
fi
if [ "$USE_MPS" = "0" ]; then
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
fi

# ---------------------------------------------------------------------------
# Stage 2: N = 1..4 concurrent inference processes, sampling device SM%.
# ---------------------------------------------------------------------------
echo "nranks,use_mps,wall_s,throughput_cases_per_s,sm_mean_pct,sm_max_pct,mem_mean_pct" > "$RESCSV"

for N in 1 2 3 4; do
    echo "########## N=$N concurrent processes (MPS=$USE_MPS) ##########"
    SMLOG="logs/mps_util/dmon_N${N}.txt"
    nvidia-smi dmon -s u -d 1 -o T > "$SMLOG" 2>&1 &
    DMON_PID=$!
    sleep 1

    t0=$(date +%s.%N)
    pids=()
    for i in $(seq 0 $((N-1))); do
        run_one "${CASES[$i]}" "N${N}_p${i}" &
        pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    t1=$(date +%s.%N)

    sleep 1
    kill "$DMON_PID" 2>/dev/null; wait "$DMON_PID" 2>/dev/null

    wall=$(echo "$t1 - $t0" | bc -l)
    thr=$(echo "scale=4; $N / $wall" | bc -l)
    # dmon -s u -o T columns: Time gpu sm mem enc dec  -> sm=$3, mem=$4
    read sm_mean sm_max mem_mean < <(awk '
        !/^#/ && $3 ~ /^[0-9]+$/ {
            ssum+=$3; if($3>smax)smax=$3; msum+=$4; cnt++
        }
        END{ if(cnt>0) printf "%.1f %d %.1f", ssum/cnt, smax, msum/cnt;
             else printf "NA NA NA" }' "$SMLOG")

    echo "${N},${USE_MPS},${wall},${thr},${sm_mean},${sm_max},${mem_mean}" >> "$RESCSV"
    printf "  N=%d wall=%.1fs thr=%s cases/s  sm_mean=%s%%  sm_max=%s%%\n" \
           "$N" "$wall" "$thr" "$sm_mean" "$sm_max"
done

# --- stop MPS if we started it ---
if [ "$USE_MPS" = "1" ]; then
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
fi

echo "################ DONE -> $RESCSV ################"
cat "$RESCSV"
