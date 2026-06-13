#!/bin/bash
# =============================================================================
# Professor critique #3: is the region grower really MEMORY-BANDWIDTH BOUND, or
# is the +2% thread-scaling actually atomic contention / cache thrashing?
#
# Answers it with a roofline: measure (a) achievable DRAM BW via STREAM, and
# (b) the BFS's achieved BW (LLC-miss x 64B / time) + thread scaling, both
# pinned to ONE socket (the paper's "~85 GB/s socket" claim). If achieved BW
# saturates near STREAM peak and is flat with threads -> bandwidth bound. If it
# sits far below peak -> the ceiling is contention, not bandwidth.
#   sbatch failed-opts/region_grow_roofline.sbatch
# =============================================================================
#SBATCH -J rg_roofline
#SBATCH -A mst114552
#SBATCH -p gp1d
#SBATCH -t 00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH -o /home/u4309334/Project/logs/%x_%j.out
#SBATCH -e /home/u4309334/Project/logs/%x_%j.err

set -uo pipefail
MAIN=/home/u4309334/Project
cd "$MAIN"; source scripts/env.sh
export OMPI_MCA_opal_cuda_support=0
export LD_LIBRARY_PATH=third_party/onnxruntime-linux-x64-gpu-1.19.2/lib:${HOME}/.local/lib/python3.9/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
PR="$MAIN/PDP_Team23_Reproducible/failed-opts"
OUT="$MAIN/logs/failed_opts/roofline"; mkdir -p "$OUT"
NIICACHE=/work/u4309334/decompressed_nii
CASE=${CASE:-100}
PIN="numactl --cpunodebind=0 --membind=0"   # single-socket, per-paper claim

echo "=== node $(hostname) ==="; lscpu | grep -E "Socket|NUMA node\(s\)|Model name|per socket" | head
echo "=== build (main binary w/ RG instrumentation + stream + microbench) ==="
make -j8 2>&1 | tail -2
cc -O3 -fopenmp "$PR/stream.c" -o "$PR/stream"
g++ -O3 -fopenmp "$PR/region_grow_roofline.cpp" -o "$PR/region_grow_roofline"

# ---- 1. dump the REAL interior mask the classical BFS processes -------------
SRC="$NIICACHE/AIIB23_${CASE}.nii"; [ -f "$SRC" ] || SRC="$MAIN/data/img/AIIB23_${CASE}.nii.gz"
DUMP="$OUT/rg_case${CASE}"
echo "=== dump real mask from classical (case $CASE) ==="
RG_DUMP="$DUMP" RG_TIME=1 OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores \
    ./airway_seg --classical --input "$SRC" --gt data/gt/AIIB23_${CASE}.nii.gz \
    --output "$OUT/mask_${CASE}.nii.gz" --omp-threads 8 2>&1 | grep -E "RG_DUMP|RG_TIME|dice|region grow" | head
ls -la "$DUMP".* 2>/dev/null

# ---- 2. STREAM peak BW (1 and 8 threads), single socket ---------------------
echo "=== STREAM peak DRAM bandwidth (single socket) ==="
for t in 1 8; do OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores $PIN "$PR/stream"; done | tee "$OUT/stream.txt"

# ---- 3. BFS achieved BW + thread scaling under perf -------------------------
echo "=== BFS region-grow: thread scaling + achieved BW (perf LLC misses) ==="
RES="$OUT/roofline.csv"; echo "threads,bfs_mean_s,bfs_best_s,processed,LLC_load_miss,LLC_store_miss,cache_miss" > "$RES"
for t in 1 2 4 8; do
    PERFCSV="$OUT/perf_t${t}.csv"
    OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores \
      perf stat -x, -o "$PERFCSV" \
        -e LLC-load-misses,LLC-store-misses,cache-misses,instructions,task-clock \
        $PIN "$PR/region_grow_roofline" "$DUMP" 5 > "$OUT/rg_t${t}.out" 2> "$OUT/rg_t${t}.err"
    read -r TT PROC EDG MEAN BEST <<<"$(cat "$OUT/rg_t${t}.out")"
    llc_l=$(awk -F, '/LLC-load-misses/{print $1}' "$PERFCSV" | head -1)
    llc_s=$(awk -F, '/LLC-store-misses/{print $1}' "$PERFCSV" | head -1)
    cm=$(awk -F, '/cache-misses/{print $1}' "$PERFCSV" | head -1)
    echo "$t,$MEAN,$BEST,$PROC,${llc_l:-NA},${llc_s:-NA},${cm:-NA}" >> "$RES"
    echo "  threads=$t  bfs_mean=${MEAN}s  LLC_l=${llc_l:-NA} LLC_s=${llc_s:-NA} cache_miss=${cm:-NA}"
done

# ---- 4. roofline summary ----------------------------------------------------
echo "=== ROOFLINE SUMMARY ==="
PEAK=$(awk '/threads=8/{print $NF" "$0}' "$OUT/stream.txt" | grep -oE "Triad=[0-9.]+" | cut -d= -f2 | tail -1)
python3 - "$RES" "${PEAK:-85}" <<'PY'
import csv,sys
res,peak=sys.argv[1],float(sys.argv[2])
rows=list(csv.DictReader(open(res)))
print(f"STREAM Triad peak (8T, 1 socket) = {peak:.1f} GB/s\n")
print(f"{'thr':>3} {'bfs_s':>8} {'speedup':>8} {'DRAM_GB':>8} {'achiev_GB/s':>11} {'%peak':>6}")
base=None
for r in rows:
    t=int(r['threads']); s=float(r['bfs_mean_s'])
    base=base or s
    def f(x):
        try:return float(x)
        except:return 0.0
    bytes_dram=(f(r['LLC_load_miss'])+f(r['LLC_store_miss']))*64.0
    # perf counted 5 repeats; BW uses total misses over total BFS time (5*bfs_s)
    bw=bytes_dram/ (5*s) /1e9 if s>0 else 0
    print(f"{t:>3} {s:>8.4f} {base/s:>7.2f}x {bytes_dram/5/1e9:>8.2f} {bw:>11.1f} {bw/peak*100:>5.0f}%")
print("\nIf achieved GB/s is near 100% of peak and flat -> BANDWIDTH BOUND.")
print("If it is well below peak -> the ceiling is contention/latency, not BW.")
PY
echo "=== done; artifacts in $OUT ==="
