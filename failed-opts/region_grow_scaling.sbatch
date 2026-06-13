#!/bin/bash
# Localize the region grower's "+2%": is it the parallel BFS being bandwidth
# bound, or a serial section dominating the stage? Run classical (--classical) at
# OMP = 1,2,4,8 on small/mid/large cases, capturing BOTH the "region grow" STAGE
# time and the in-situ [RG_TIME] wavefront-BFS time. If the wavefront scales but
# the stage doesn't, the cap is a serial section (Amdahl), not bandwidth.
#   sbatch failed-opts/region_grow_scaling.sbatch
#SBATCH -J rg_scaling
#SBATCH -A mst114552
#SBATCH -p gp1d
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH -o /home/u4309334/Project/logs/%x_%j.out
#SBATCH -e /home/u4309334/Project/logs/%x_%j.err
set -uo pipefail
MAIN=/home/u4309334/Project; cd "$MAIN"; source scripts/env.sh
export OMPI_MCA_opal_cuda_support=0
export LD_LIBRARY_PATH=third_party/onnxruntime-linux-x64-gpu-1.19.2/lib:${HOME}/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
NII=/work/u4309334/decompressed_nii
OUT="$MAIN/logs/failed_opts/rg_scaling"; mkdir -p "$OUT" logs/rg_scaling
PIN="numactl --cpunodebind=0 --membind=0"
CASES=${CASES:-145 52 110}     # small / mid / large airway (nz 146 / 461 / 945)
src(){ [ -s "$NII/AIIB23_$1.nii" ] && echo "$NII/AIIB23_$1.nii" || echo "data/img/AIIB23_$1.nii.gz"; }

RES="$OUT/rg_scaling.csv"; echo "case,threads,stage_rg_s,wavefront_bfs_s,processed" > "$RES"
echo "=== node $(hostname) ===" ; make -j8 2>&1 | tail -1
for c in $CASES; do
  for t in 1 2 4 8; do
    L=logs/rg_scaling/c${c}_t${t}.log
    RG_TIME=1 OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores \
      $PIN ./airway_seg --classical --input "$(src $c)" --gt data/gt/AIIB23_${c}.nii.gz \
      --output "$OUT/sc_${c}.nii.gz" --omp-threads $t > "$L" 2>&1
    stage=$(grep -E "^  region grow" "$L" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    wf=$(grep -oE "bfs_s=[0-9.]+" "$L" | tail -1 | cut -d= -f2)
    proc=$(grep -oE "processed=[0-9]+" "$L" | tail -1 | cut -d= -f2)
    echo "$c,$t,${stage:-NA},${wf:-NA},${proc:-NA}" | tee -a "$RES"
  done
done
echo "=== SUMMARY: stage vs wavefront scaling ==="
python3 - "$RES" <<'PY'
import csv,sys,collections
rows=list(csv.DictReader(open(sys.argv[1])))
byc=collections.defaultdict(list)
for r in rows: byc[r['case']].append(r)
for c,rs in byc.items():
    g={int(r['threads']):r for r in rs}
    def fl(x):
        try:return float(x)
        except:return float('nan')
    s1,s8=fl(g[1]['stage_rg_s']),fl(g[8]['stage_rg_s'])
    w1,w8=fl(g[1]['wavefront_bfs_s']),fl(g[8]['wavefront_bfs_s'])
    proc=g[8]['processed']
    print(f"case {c} (processed={proc}):")
    print(f"  STAGE   rg: {s1:.3f}s -> {s8:.3f}s  = {s1/s8:.2f}x  (paper: ~+2% = ~1.0x)")
    print(f"  WAVEFRONT : {w1:.4f}s -> {w8:.4f}s = {w1/w8:.2f}x  (parallel BFS only)")
    print(f"  wavefront is {w8/s8*100:.1f}% of the stage at 8T\n")
print("If STAGE~flat but WAVEFRONT scales and is a small %% -> serial section (Amdahl),")
print("not the BFS being bandwidth bound.")
PY
