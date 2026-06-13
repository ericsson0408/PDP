# failed-opts — optimizations & analyses that did **not** help

The negative results behind the paper's **Table IV** ("Evaluated but discarded
optimizations") and the **"What did not help"** section. Each item here is the
evidence that a tempting optimization gives no measurable single-GPU benefit, so
nobody re-tries it. (The *positive* analyses that back the figures/tables live in
[`../analysis/`](../analysis/); the live MPS-figure data is regenerated there.)

## Contents

| item | backs | what it shows |
|------|-------|---------------|
| `table4/` | **Table IV** (rows 1–5) | re-measures each rejected row from scratch — FP16 I/O, zstd-vs-gzip, MPI+MPS, input 256→224, batch tuning (see [`table4/README.md`](table4/README.md)) |
| `region_grow_roofline.{cpp,sbatch}` + `stream.c` | §"What did not help", Limitation 2 | the lock-free BFS uses <10 GB/s vs a 72 GB/s socket roofline → **not bandwidth-bound** (Amdahl, not a bandwidth wall) |
| `region_grow_scaling.sbatch` | §"What did not help" | the region-grow STAGE barely scales while the BFS scales ~4× → the cap is a **serial section**, and the stage is a sub-second slice of wall-clock |
| `mpi_mps_colocation.sbatch` | Table IV (MPI+MPS), Fig. mps_saturation | one process already pins the SMs at 100%; extra co-located ranks leave throughput flat (MPS would not start on the node) |

## Reproduce (TWCC V100)
```bash
# Table IV rows (see table4/README.md for per-row prereqs)
bash failed-opts/table4/run_all.sh            # smoke set
CASES=all bash failed-opts/table4/run_all.sh  # full 120 (row 4 tail count)

# region-grow roofline + thread scaling, and the co-location sweep
sbatch failed-opts/region_grow_roofline.sbatch
sbatch failed-opts/region_grow_scaling.sbatch
sbatch failed-opts/mpi_mps_colocation.sbatch
```
Flags use the readable aliases (`--classical/--gate/...`); the `.sbatch` files
hard-code our cluster paths (`cd`/`source scripts/env.sh`) — adapt those lines.
Diagnostic outputs land under `logs/failed_opts/`.

> The GPU-saturation **figure** (`mps_saturation.png`) and its CSV
> (`results/mps_util.csv`) are produced by `../analysis/measure_mps_util.sbatch`
> + `../analysis/plot_mps_util.py`, since the figure appears in the paper.
