# Analysis — paper figures, tables & statistics

The **code** that regenerates every figure, the statistics, and the numbers
behind the paper's tables. All scripts read the cleaned per-case CSVs in
[`../results/`](../results/) (one row per AIIB23 case) and write PNGs to
`analysis/figures/`, so they reproduce the paper **without** a GPU or re-running
any benchmark.

## Inputs (in `../results/`)
| CSV | pipeline | columns |
|-----|----------|---------|
| `classical.csv`       | Classical | lowercase (`dice`, `total_s`, per-stage `*_s`) |
| `unet3d.csv`          | 3D U-Net  | `Case_ID, Dice_Score, *_Time` |
| `hybrid.csv`          | Hybrid    | `Case_ID, Dice_Score, *_Time` |
| `unet25d.csv`         | 2.5D U-Net | `Case_ID, Dice_Score, *_Time` |
| `unet25d_hybrid.csv`  | 2.5D Hybrid (stats only) | `Case_ID, Dice_Score, *_Time` |
| `case_dims.csv`       | per-case nx·ny·nz | volume sizes for per-voxel normalization |
| `mps_util.csv`        | GPU saturation sweep | from `measure_mps_util.sbatch` |

## Figures & stats (`pip install pandas numpy matplotlib scipy`)
```bash
cd analysis
python plot_accuracy_speed.py     # Fig: accuracy_vs_speed.png  (Dice vs s/case, 4 pipelines)
python plot_pervoxel.py           # Fig: per_voxel_cost.png     (38 vs 15 ns/voxel slopes)
python plot_stage_breakdown.py    # Fig: stage_breakdown.png    (per-stage wall-clock)
python plot_mps_util.py           # Fig: mps_saturation.png     (SM% & throughput vs #procs)
python stats_tests.py             # Wilcoxon signed-rank + Cliff's delta (paper §Hybrid)
python stats_heldout_timing.py    # held-out (12-case) val Dice + timing mean/std/95% CI
```
PNGs land in `analysis/figures/`; copy them into the paper's `fig/` to update it.

### What reproduces what
- **Table II** (four-pipeline comparison) and **Table III** (inference-opt
  ablation) numbers come from the same `../results/` CSVs; `stats_heldout_timing.py`
  prints the `mean±std` and held-out val-Dice that fill Table II's columns.
- **Figs** accuracy_vs_speed / per_voxel_cost / stage_breakdown / mps_saturation
  are produced by the four `plot_*.py` above.
- **Stats paragraph** (Wilcoxon, Cliff's δ, failed-case overlap) ← `stats_tests.py`.

## GPU saturation data (`mps_util.csv`)
```bash
sbatch analysis/measure_mps_util.sbatch   # N=1..4 co-located inferences on one V100
```
Re-measures the sweep live and writes `../results/mps_util.csv`. (The rejected
optimization this supports — extra ranks/MPS don't help — also lives in
[`../failed-opts/`](../failed-opts/).)
