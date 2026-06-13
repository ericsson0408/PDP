#!/bin/bash
# Re-run the 2.5D pipeline with the GPU softmax kernel too, so 3D/hybrid/2.5D are
# all on equal footing (GPU softmax). Warm cache, OMP=8, gpu:4. Writes the
# canonical results/unet25d.csv.
# Submit: sbatch benchmark/table2_gpu_softmax_25d.sbatch  | smoke: --export=ALL,CASES="100 30"
#SBATCH -J gpusm_25d
#SBATCH -A mst114552
#SBATCH -p gp1d
#SBATCH -t 03:00:00
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
MODEL=model/airway_2p5d_unet.onnx
CACHE=/work/u4309334/decompressed_nii
ROOT=logs/gpusm25d
mkdir -p results/unet25d model/trt_cache "$ROOT"
if [ -n "${CASES:-}" ]; then LIST="$CASES"
else LIST=$(ls data/img/AIIB23_*.nii.gz | sed -E 's@.*AIIB23_([0-9]+)\.nii\.gz@\1@' | sort -n); fi
src_for(){ if [ -s "$CACHE/AIIB23_$1.nii" ]; then echo "$CACHE/AIIB23_$1.nii"; else echo "data/img/AIIB23_$1.nii.gz"; fi; }
echo "[25dgsm] node=$(hostname) cases=$(echo $LIST|wc -w)"
for c in $LIST; do cat "$(src_for $c)" >/dev/null 2>&1; done
# 2.5D TRT engine warmup (untimed) so timed loop reuses it
for c in $LIST; do mpirun -np 4 --bind-to none ./airway_seg --input "$(src_for $c)" \
   --gt data/gt/AIIB23_${c}.nii.gz --omp-threads 8 --model-2p5d --gate --onnx-model "$MODEL" \
   --gpu-softmax --output results/unet25d/warm_${c}.nii.gz >/dev/null 2>&1; done
for c in $LIST; do s=$(src_for $c); cat "$s">/dev/null 2>&1
  mpirun -np 4 --bind-to none ./airway_seg --input "$s" --gt data/gt/AIIB23_${c}.nii.gz \
    --omp-threads 8 --model-2p5d --gate --onnx-model "$MODEL" --gpu-softmax \
    --output results/unet25d/mask_${c}.nii.gz > "$ROOT/AIIB23_${c}.log" 2>&1
  d=$(grep -oE "Dice=[0-9.]+" "$ROOT/AIIB23_${c}.log"|tail -1); t=$(grep -E "^  TOTAL" "$ROOT/AIIB23_${c}.log"|grep -oE "[0-9]+\.[0-9]+")
  echo "[25dgsm] case $c  $d total=${t:-NA}s"
done
$PY benchmark/parse_neural_bench.py "$ROOT" results/unet25d.csv
echo "[25dgsm] DONE"
