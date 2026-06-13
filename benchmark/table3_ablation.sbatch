#!/bin/bash
# Inference-optimization ablation (reproducible Table III).
# All configs run the SAME retrained 3D model + --gate post-processing (so Dice is
# constant ~0.9105), on the warm-cache plain-.nii harness. Each row toggles ONE
# optimization on top of the previous, isolating its wall-clock effect.
#
#   A baseline : cuDNN EP, exhaustive search, CPU softmax, no overlap
#   B +heuristic: cuDNN heuristic search
#   C +gpu-sm  : GPU softmax kernel (half D2H copy, no host exp())
#   D +TRT-FP16: TensorRT FP16 EP (still non-overlap, GPU softmax)
#   E +overlap : double-buffered stream overlap (overlap path; host softmax)
#   D' TRT/CPU-sm/no-ovl : controlled overlap baseline (for the clean D'->E delta)
#   F FP32     : TRT FP32 (--trt-no-fp16) to isolate the FP16 win
#
# Submit: sbatch --export=ALL,CASES="100 30 110 124" benchmark/table3_ablation.sbatch
#         sbatch benchmark/table3_ablation.sbatch   # all 120 (-> results/ablation_summary.txt)
#SBATCH -J abl_bench
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
export OMPI_MCA_opal_cuda_support=0; unset OMPI_MCA_btl_smcuda_use_cuda_ipc
export LD_LIBRARY_PATH=third_party/onnxruntime-linux-x64-gpu-1.19.2/lib:${HOME}/.local/lib/python3.9/site-packages/tensorrt_libs:${HOME}/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores
MODEL=model/dynunet_retrained_fp16.onnx
CACHE=/work/u4309334/decompressed_nii
ROOT=logs/ablbench
mkdir -p results model/trt_cache "$ROOT"

if [ -n "${CASES:-}" ]; then LIST="$CASES"
else LIST=$(ls data/img/AIIB23_*.nii.gz | sed -E 's@.*AIIB23_([0-9]+)\.nii\.gz@\1@' | sort -n); fi
src_for(){ if [ -s "$CACHE/AIIB23_$1.nii" ]; then echo "$CACHE/AIIB23_$1.nii"; else echo "data/img/AIIB23_$1.nii.gz"; fi; }

# config name -> "MODEL_FLAG=... | extra airway_seg flags"
# (MODEL_FLAG selects --gate 3D vs --model-2p5d --gate 2.5D and the onnx model)
M3="--gate --onnx-model $MODEL"
M25="--model-2p5d --gate --onnx-model model/airway_2p5d_unet.onnx"
declare -A CFG=(
  [N_naive_serial_sm]="$M3 | --no-trt --no-stream-overlap --serial-softmax"
  [A_omp_softmax]="$M3 | --no-trt --no-stream-overlap"
  [B_cudnn_heur]="$M3 | --no-trt --cudnn-heuristic --no-stream-overlap"
  [Dp_trt_fp16]="$M3 | --cudnn-heuristic --no-stream-overlap"
  [D_gpu_softmax]="$M3 | --cudnn-heuristic --gpu-softmax --no-stream-overlap"
  [G_switch_2p5d]="$M25 | --cudnn-heuristic --gpu-softmax --no-stream-overlap"
  [E_trt_overlap]="$M3 | --cudnn-heuristic"
  [F_trt_fp32]="$M3 | --cudnn-heuristic --gpu-softmax --no-stream-overlap --trt-no-fp16"
)
ORDER="N_naive_serial_sm A_omp_softmax B_cudnn_heur Dp_trt_fp16 D_gpu_softmax G_switch_2p5d E_trt_overlap F_trt_fp32"

echo "[abl] node=$(hostname) cases=$(echo $LIST|wc -w)"; nvidia-smi --query-gpu=name --format=csv,noheader | head -1
# page-cache prewarm
for c in $LIST; do cat "$(src_for $c)" >/dev/null 2>&1; done
# TRT engine warmup (build once, untimed) so timed TRT rows reuse the cache --
# both the 3D and the 2.5D models need their engines populated.
for c in $LIST; do
  mpirun -np 4 --bind-to none ./airway_seg --input "$(src_for $c)" \
    --gt data/gt/AIIB23_${c}.nii.gz --omp-threads 8 $M3 --cudnn-heuristic \
    --output "$ROOT/warm3d_${c}.nii.gz" >/dev/null 2>&1
  mpirun -np 4 --bind-to none ./airway_seg --input "$(src_for $c)" \
    --gt data/gt/AIIB23_${c}.nii.gz --omp-threads 8 $M25 --cudnn-heuristic \
    --output "$ROOT/warm25_${c}.nii.gz" >/dev/null 2>&1
done

printf "%-26s %8s %8s %8s\n" config mean_onnx mean_tot mean_dice | tee "$ROOT/summary.txt"
for cfg in $ORDER; do
  modelflag="${CFG[$cfg]%%|*}"; flags="${CFG[$cfg]##*|}"; LOGD="$ROOT/$cfg"; mkdir -p "$LOGD"
  for c in $LIST; do
    cat "$(src_for $c)" >/dev/null 2>&1
    mpirun -np 4 --bind-to none ./airway_seg --input "$(src_for $c)" \
      --gt data/gt/AIIB23_${c}.nii.gz --omp-threads 8 \
      $modelflag $flags --output "$LOGD/mask_${c}.nii.gz" \
      > "$LOGD/AIIB23_${c}.log" 2>&1
  done
  # aggregate from the per-stage timing block
  python3 - "$LOGD" "$cfg" <<'PY' | tee -a "$ROOT/summary.txt"
import sys,glob,re,statistics
d,c=sys.argv[1],sys.argv[2]
onnx,tot,dice=[],[],[]
for f in glob.glob(d+"/*.log"):
    t=open(f,errors="ignore").read()
    mo=re.search(r"^\s*onnx infer\s*:\s*([0-9.]+)",t,re.M); mt=re.search(r"^\s*TOTAL\s*:\s*([0-9.]+)",t,re.M)
    md=re.findall(r"Dice=([0-9.]+)",t)
    if mo:onnx.append(float(mo.group(1)))
    if mt:tot.append(float(mt.group(1)))
    if md:dice.append(float(md[-1]))
m=lambda x:statistics.mean(x) if x else float('nan')
print("%-26s %8.3f %8.3f %8.4f"%(c,m(onnx),m(tot),m(dice)))
PY
done
cp "$ROOT/summary.txt" results/ablation_summary.txt
echo "[abl] DONE -> results/ablation_summary.txt"
