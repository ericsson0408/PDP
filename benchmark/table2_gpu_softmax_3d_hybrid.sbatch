#!/bin/bash
# Re-run the 3D and Hybrid neural pipelines with the GPU softmax kernel as the
# shipped config: --gpu-softmax --no-stream-overlap (overlap is <0.5%, GPU softmax
# saves ~1.5s; both verified bit-identical Dice). Classical and 2.5D are
# unchanged (no GPU softmax in their paths). Warm cache, OMP=8, gpu:4 -- same I/O
# harness as table2_warmcache_io. Writes the canonical results/unet3d.csv + hybrid.csv.
# Submit: sbatch benchmark/table2_gpu_softmax_3d_hybrid.sbatch  |  smoke: --export=ALL,CASES="100 30 168"
#SBATCH -J fair_gpusm
#SBATCH -A mst114552
#SBATCH -p gp1d
#SBATCH -t 04:00:00
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
export OMPI_MCA_opal_cuda_support=0; unset OMPI_MCA_btl_smcuda_use_cuda_ipc
export LD_LIBRARY_PATH=third_party/onnxruntime-linux-x64-gpu-1.19.2/lib:${HOME}/.local/lib/python3.9/site-packages/tensorrt_libs:${HOME}/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores
PY=/opt/ohpc/twcc/conda/24.5.0/miniconda3/bin/python
MODEL=model/dynunet_retrained_fp16.onnx
CACHE=/work/u4309334/decompressed_nii
ROOT=logs/gpusmbench
GSM="--gpu-softmax --no-stream-overlap"   # shipped neural config
mkdir -p results/unet3d results/hybrid model/trt_cache "$ROOT/3d" "$ROOT/hybrid"

if [ -n "${CASES:-}" ]; then LIST="$CASES"
else LIST=$(ls data/img/AIIB23_*.nii.gz | sed -E 's@.*AIIB23_([0-9]+)\.nii\.gz@\1@' | sort -n); fi
src_for(){ if [ -s "$CACHE/AIIB23_$1.nii" ]; then echo "$CACHE/AIIB23_$1.nii"; else echo "data/img/AIIB23_$1.nii.gz"; fi; }

echo "[gpusm] node=$(hostname) cases=$(echo $LIST|wc -w) config='$GSM'"
for c in $LIST; do cat "$(src_for $c)" >/dev/null 2>&1; done   # page-cache prewarm

run_one(){ local logd=$1 c=$2; shift 2; local s; s=$(src_for "$c"); cat "$s">/dev/null 2>&1
  mpirun -np 4 --bind-to none ./airway_seg --input "$s" --gt data/gt/AIIB23_${c}.nii.gz \
    --omp-threads 8 "$@" > "$logd/AIIB23_${c}.log" 2>&1
  local d t; d=$(grep -oE "Dice=[0-9.]+" "$logd/AIIB23_${c}.log"|tail -1); t=$(grep -E "^  TOTAL" "$logd/AIIB23_${c}.log"|grep -oE "[0-9]+\.[0-9]+"); echo "    $d total=${t:-NA}s"; }

n=0
for c in $LIST; do n=$((n+1)); echo "[gpusm] ($n) case $c"
  echo -n "  3d    :"; run_one "$ROOT/3d"     "$c" --gate   --onnx-model "$MODEL" $GSM --output results/unet3d/mask_${c}.nii.gz
  echo -n "  hybrid:"; run_one "$ROOT/hybrid" "$c" --hybrid --onnx-model "$MODEL" $GSM --output results/hybrid/mask_${c}.nii.gz
done
$PY benchmark/parse_neural_bench.py "$ROOT/3d"     results/unet3d.csv
$PY benchmark/parse_neural_bench.py "$ROOT/hybrid" results/hybrid.csv
echo "[gpusm] DONE"
