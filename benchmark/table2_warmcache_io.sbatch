#!/bin/bash
# WARM-CACHE I/O RE-BENCHMARK: run all FOUR pipelines (classical/3D/hybrid/2.5D)
# on the SAME node, SAME current binary, SAME pre-warmed page cache, reading the
# SAME pre-decompressed plain .nii inputs -- so the LOAD stage (and every other
# stage) is measured apples-to-apples. Produces the canonical results/classical.csv
# (Table II classical row). NOTE: the neural rows here run WITHOUT --gpu-softmax,
# so they are the warm-cache LOAD baseline only -- the canonical neural numbers
# (Table II) come from table2_gpu_softmax_3d_hybrid.sbatch / table2_gpu_softmax_25d.sbatch.
#
# Submit full:  sbatch benchmark/table2_warmcache_io.sbatch
# Smoke test:   sbatch --export=ALL,CASES="168 158 63 30 54 124 41 145" benchmark/table2_warmcache_io.sbatch
#SBATCH -J fair_iobench
#SBATCH -A mst114552
#SBATCH -p gp1d
#SBATCH -t 06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=320G
#SBATCH -o /home/u4309334/Project/logs/%x_%j.out
#SBATCH -e /home/u4309334/Project/logs/%x_%j.err

set -uo pipefail
cd /home/u4309334/Project
source scripts/env.sh
export OMPI_MCA_opal_cuda_support=0
unset  OMPI_MCA_btl_smcuda_use_cuda_ipc
export LD_LIBRARY_PATH=third_party/onnxruntime-linux-x64-gpu-1.19.2/lib:${HOME}/.local/lib/python3.9/site-packages/tensorrt_libs:${HOME}/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores

PY=/opt/ohpc/twcc/conda/24.5.0/miniconda3/bin/python
MODEL3D=model/dynunet_retrained_fp16.onnx     # reproduces 3D Dice 0.9105 / hybrid 0.9074
MODEL25=model/airway_2p5d_unet.onnx           # reproduces 2.5D Dice 0.8634
CACHE=/work/u4309334/decompressed_nii
ROOT=logs/fairbench
EXTRA_FLAGS=${EXTRA_FLAGS:-}
mkdir -p results/classical model/trt_cache "$ROOT/classical" "$ROOT/3d" "$ROOT/hybrid" "$ROOT/2p5d" "$ROOT/warmup"

if [ -n "${CASES:-}" ]; then LIST="$CASES"
else LIST=$(ls data/img/AIIB23_*.nii.gz | sed -E 's@.*AIIB23_([0-9]+)\.nii\.gz@\1@' | sort -n); fi

src_for() {
    local c=$1
    if [ -s "$CACHE/AIIB23_${c}.nii" ]; then echo "$CACHE/AIIB23_${c}.nii"
    else echo "data/img/AIIB23_${c}.nii.gz"; fi
}

echo "[fair] node=$(hostname)  cases=$(echo $LIST|wc -w)  OMP=$OMP_NUM_THREADS  cache=$CACHE"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ---- Page-cache pre-warm: pull every input into RAM before any timed run. ----
echo "[prewarm] free before:"; free -g | awk 'NR==1||NR==2'
SECONDS=0
for c in $LIST; do cat "$(src_for $c)" > /dev/null 2>&1; done
echo "[prewarm] warmed $(echo $LIST|wc -w) files in ${SECONDS}s"
echo "[prewarm] free after:";  free -g | awk 'NR==1||NR==2'

# ---- TensorRT engine warm-up (untimed) ----
# The paper measures with a WARM engine cache ("engine builds serialized to a
# cache and reused across all cases and jobs"). The 3D engine (fixed 128^3 patch)
# is already in model/trt_cache, but the 2.5D model uses a dynamic input profile
# whose engines are NOT cached here, so a cold timed run would charge ~45s TRT
# builds to the largest cases. Run 2.5D once over all cases untimed to populate
# the cache; the subsequent timed pass then reuses it (no builds).
echo "[warmup] building 2.5D TRT engines over all cases (untimed) ..."
SECONDS=0
mkdir -p "$ROOT/warmup"
for c in $LIST; do
    src=$(src_for "$c")
    mpirun -np 4 --bind-to none ./airway_seg \
        --model-2p5d --gate --input "$src" --gt data/gt/AIIB23_${c}.nii.gz \
        --onnx-model "$MODEL25" --output "$ROOT/warmup/mask_${c}.nii.gz" \
        --omp-threads 8 $EXTRA_FLAGS > "$ROOT/warmup/AIIB23_${c}.log" 2>&1
done
echo "[warmup] 2.5D engine cache warmed in ${SECONDS}s; trt_cache has $(ls model/trt_cache/*.engine 2>/dev/null | wc -l) engines"

run_one() {  # $1=logdir $2=case  $3..=extra airway_seg args
    local logd=$1 c=$2; shift 2
    local src; src=$(src_for "$c")
    cat "$src" > /dev/null 2>&1            # per-case double-tap -> guaranteed warm
    mpirun -np 4 --bind-to none ./airway_seg \
        --input  "$src" \
        --gt     data/gt/AIIB23_${c}.nii.gz \
        --omp-threads 8 "$@" $EXTRA_FLAGS \
        > "$logd/AIIB23_${c}.log" 2>&1
    local d t l
    d=$(grep -oE "Dice=[0-9.]+" "$logd/AIIB23_${c}.log" | tail -1)
    t=$(grep -E "^  TOTAL" "$logd/AIIB23_${c}.log" | grep -oE "[0-9]+\.[0-9]+")
    l=$(grep -E "^  load" "$logd/AIIB23_${c}.log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    echo "    $d load=${l:-NA}s total=${t:-NA}s"
}

n=0
for c in $LIST; do
    n=$((n+1))
    echo "[fair] ($n) case $c"
    echo -n "  classical:"; run_one "$ROOT/classical" "$c" --classical --output results/classical/mask_${c}.nii.gz
    echo -n "  3d       :"; run_one "$ROOT/3d"        "$c" --gate   --onnx-model "$MODEL3D" --output "$ROOT/3d/mask_${c}.nii.gz"
    echo -n "  hybrid   :"; run_one "$ROOT/hybrid"    "$c" --hybrid --onnx-model "$MODEL3D" --output "$ROOT/hybrid/mask_${c}.nii.gz"
    echo -n "  2.5d     :"; run_one "$ROOT/2p5d"      "$c" --model-2p5d --gate --onnx-model "$MODEL25" --output "$ROOT/2p5d/mask_${c}.nii.gz"
done

echo "[fair] parsing CSVs ..."
# Canonical classical row -> results/classical.csv. The neural CSVs here are the
# warm-cache LOAD baseline (no --gpu-softmax); the canonical Table II neural CSVs
# come from the table2_gpu_softmax_*.sbatch harnesses, so these stay under logs/.
$PY benchmark/parse_classical_bench.py "$ROOT/classical" results/classical.csv
$PY benchmark/parse_neural_bench.py    "$ROOT/3d"        "$ROOT/warmcache_3d_noopt.csv"
$PY benchmark/parse_neural_bench.py    "$ROOT/hybrid"    "$ROOT/warmcache_hybrid_noopt.csv"
$PY benchmark/parse_neural_bench.py    "$ROOT/2p5d"      "$ROOT/warmcache_2p5d_noopt.csv"
echo "[fair] DONE"
