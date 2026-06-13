# Reproducing Table IV — "Evaluated but discarded optimizations"

Table IV is a **lessons-learned / rejection-rationale** table, not a CSV-driven
results table like Table II/III. Most rows are qualitative ("No speedup", "No
effect"); a few carry numbers. These scripts let you **re-measure each row from
scratch** and check the claim holds, instead of taking the prose on faith.

All scripts live here and write their per-row output to `../../logs/failed_opts/table4/`.
They reuse the same binary, models, and case inputs as the rest of the package.

```bash
cd PDP_Team23_Reproducible
source env.sh && make                       # build ./airway_seg (once)
bash failed-opts/table4/run_all.sh                # smoke set (a few cases)
CASES=all bash failed-opts/table4/run_all.sh      # full 120-case (needed for row 4)
```

Pick the cohort with `CASES="100 101 168 ..."` (default: a small smoke set).
Inputs resolve to pre-decompressed `*.nii` if present (`NIICACHE=`), else `*.nii.gz`.

---

## Row-by-row: what is measured, and how reproducible it is

| # | Table IV row | Script | Reproducibility | Needs |
|---|--------------|--------|-----------------|-------|
| 1 | FP16 input/output → no speedup | `row1_fp16_io.sh` | ⚠️ needs a 2nd ONNX export (FP16-IO) | GPU + `export_fp16io.py` |
| 2 | zstd vs gzip → no speedup; "4×" was cache | `row2_zstd_vs_gzip.sh` | ✅ standalone microbench | `gzip`,`zstd` (cold needs root/vmtouch) |
| 3 | MPI+MPS >2 ranks → saturates (81% / −17% / flat) | `row3_mpi_mps.sh` | ✅ 81% & flat measured; ⚠️ −17% is a projection | CSVs only (sbatch to re-measure) |
| 4 | input 256→224 → 12% faster, 117→108 ≥0.8 | `row4_input_size.sh` | ⚠️ **no archived data**; needs rebuild + full 120 cases | GPU + `export_25d_dynamic.py` + rebuild |
| 5 | per-case batch tuning → no effect | `row5_batch_tuning.sh` | ✅ pure CLI sweep | GPU |

### Row 1 — FP16 input/output
Same FP16-weight model, two I/O dtypes. `dynunet_retrained_fp16.onnx` keeps
**FP32 I/O** (`keep_io_types=True`); `export_fp16io.py` makes the **FP16-IO**
twin. The C++ loader auto-detects the input dtype, so no rebuild — just swap the
model. Expect **equal mean TOTAL** (the GPU win is in the weights, already FP16).
```bash
python failed-opts/table4/export_fp16io.py --src model/dynunet_retrained.onnx \
                                     --dst model/dynunet_retrained_fp16io.onnx
bash failed-opts/table4/row1_fp16_io.sh
```
(If `model/dynunet_retrained.onnx`, the FP32 master, is absent, regenerate it
via `training/3d_dynunet/export_onnx.py`.)

### Row 2 — zstd vs gzip
Load-stage microbench, independent of the GPU binary. Shows (a) `zstd -T0`
(multi-thread) ≈ `zstd -T1` ≈ gunzip on a **single-frame** stream → switching
codec is no speedup; (b) the old **"4×" is a cold-vs-warm cache artifact**
(same file, cold decode ≈ N× warm). Cold timing needs root (`drop_caches`) or
`vmtouch`; otherwise only (a) is measured.
```bash
bash failed-opts/table4/row2_zstd_vs_gzip.sh
```

### Row 3 — MPI+MPS, >2 ranks
Pure analysis over archived CSVs (no GPU):
- **81%** = `mean(ONNX_Infer_Time)/mean(Total_Time)` from `results/unet3d.csv`.
- **flat throughput** from `results/mps_util.csv` (`0.067→0.064 cases/s`, SM peak
  100% at 1 rank).
- **−17%** is an **Amdahl projection** of the ~19% non-GPU tail a 2nd rank could
  overlap — the paper states MPS would not start on the test node, so this is the
  conservative no-sharing floor, not a live measurement.
```bash
bash failed-opts/table4/row3_mpi_mps.sh
sbatch analysis/measure_mps_util.sbatch   # optional: re-measure the sweep live
```

### Row 4 — input size 256 → 224  (the only row with no archived data)
`MODEL25D_H/W` is a **compile-time constant** in `src/onnx_infer.cpp`, and the
shipped 2.5D ONNX is fixed at 256². So the script:
1. `export_25d_dynamic.py` → an ONNX with dynamic H,W (accepts 224²);
2. rebuilds a `airway_seg_25d_224` binary (sed the constant → `make`, then
   restores the stock source);
3. runs both 256 and 224 over the cohort, reporting **mean TOTAL** and
   **#cases ≥ 0.8**.

The 117→108 drop is a **120-case tail count** — you must use `CASES=all`. A
subset reproduces only the ~12% speed delta (by design, "samples hide the tail").
```bash
python failed-opts/table4/export_25d_dynamic.py
CASES=all bash failed-opts/table4/row4_input_size.sh
```

### Row 5 — per-case batch tuning
Sweeps `--onnx-batch ∈ {1,2,4,8}` on the 3D pipeline; "no effect" = mean TOTAL
is flat (fixed per-case startup dominates; the 128³ conv scales poorly with
batch). Fully CLI-driven, no rebuild.
```bash
bash failed-opts/table4/row5_batch_tuning.sh
```

---

## Honest caveats
- Rows **1** and **4** require artifacts that are **not shipped** (a FP16-IO
  export; a dynamic-spatial 2.5D ONNX + a rebuilt binary). The scripts generate
  them, but you need the model/training inputs and a V100.
- Row **3**'s **−17%** is the one number in Table IV that is a **projection, not
  a direct measurement** (MPS would not start on the node). Everything else in
  the table is either measured here or follows from Table II.
- Absolute seconds depend on the GPU (paper: V100-SXM2) and on a warm page
  cache; compare **ratios/counts**, not raw seconds, across machines.
