# Benchmark harnesses — reproducing Tables II & III

All four pipelines are one binary (`./airway_seg`, built from `src/`). Every
number in the paper is measured under **one I/O harness**: pre-decompressed plain
`.nii` read from a **warm page cache**, OMP=8, a `gpu:4` allocation, and a warm
TensorRT engine cache. The `.sbatch` scripts hard-code our TWCC paths (the `cd`
line and `source scripts/env.sh`); adapt those two lines, or just reuse the flag
matrices below with your own loop.

> Flags use the readable aliases (`--classical/--gate/--hybrid/--model-2p5d/
> --no-stream-overlap`); the old `--v8/--v16/--v17/--unet25d/--v12-no-overlap`
> tags still work. See [`../pipelines/README.md`](../pipelines/README.md).

## Table II — four-pipeline comparison (warm-cache, apples-to-apples I/O)

| script | produces | role |
|---|---|---|
| `table2_warmcache_io.sbatch` | `results/classical.csv` | pre-warms the page cache, then times all four; the **classical** row is canonical here |
| `table2_gpu_softmax_3d_hybrid.sbatch` | `results/unet3d.csv`, `results/hybrid.csv` | the shipped neural config (`--gpu-softmax --no-stream-overlap`) — **canonical** 3D + Hybrid |
| `table2_gpu_softmax_25d.sbatch` | `results/unet25d.csv` | the shipped 2.5D config (`--gpu-softmax`) — **canonical** 2.5D |

The GPU softmax kernel saves ~1.5 s and is bit-identical in Dice, so the neural
rows in `table2_warmcache_io.sbatch` (run **without** `--gpu-softmax`) are kept
under `logs/` as a warm-cache load baseline only; the canonical neural CSVs come
from the two `table2_gpu_softmax_*` harnesses.

| Pipeline | flags (on top of `--input … --gt … --omp-threads 8`) | total (s) |
|---|---|---|
| Classical | `--classical` | 4.74 |
| 3D U-Net  | `--gate --gpu-softmax --no-stream-overlap --onnx-model model/dynunet_retrained_fp16.onnx` | 7.63 |
| Hybrid    | `--hybrid --gpu-softmax --no-stream-overlap --onnx-model model/dynunet_retrained_fp16.onnx` | 7.67 |
| 2.5D U-Net| `--model-2p5d --gate --gpu-softmax --onnx-model model/airway_2p5d_unet.onnx` | 4.01 |

## Table III — inference-optimization ablation (reproducible)

`table3_ablation.sbatch` builds the 3D pipeline up from a naive baseline, one
toggle per row, then switches to 2.5D. Mean over all 120 cases; all 3D rows use
`--gate --onnx-model model/dynunet_retrained_fp16.onnx`. Writes
`results/ablation_summary.txt`.

| Row | added flags | total (s) | speedup |
|---|---|---|---|
| serial CPU softmax (naive) | `--no-trt --no-stream-overlap --serial-softmax` | 24.01 | 1.00× |
| + OpenMP-parallel softmax  | `--no-trt --no-stream-overlap` | 12.87 | 1.87× |
| + cuDNN heuristic search    | `--no-trt --cudnn-heuristic --no-stream-overlap` | 11.86 | 2.02× |
| + TensorRT FP16 engine      | `--cudnn-heuristic --no-stream-overlap` | 9.18 | 2.62× |
| + GPU softmax kernel         | `--cudnn-heuristic --gpu-softmax --no-stream-overlap` | 7.63 | 3.15× |
| switch to 2.5D model        | `--model-2p5d --gate --onnx-model model/airway_2p5d_unet.onnx --cudnn-heuristic --gpu-softmax --no-stream-overlap` | 4.01 | 5.99× |

Side checks: TensorRT FP32 (`--trt-no-fp16`) costs ~0.6 s vs FP16; stream overlap
(default on vs `--no-stream-overlap`) changed total by <0.5%.

## Parsers
- `parse_neural_bench.py <logdir> <out.csv>` — neural per-case CSV (Case_ID, …, Total_Time).
- `parse_classical_bench.py <logdir> <out.csv>` — classical CSV (load_s, mpi_part_s, …).

All CSVs land in `../results/` and are consumed by `../analysis/`. Rejected
optimizations (Table IV / "what did not help") live in
[`../failed-opts/`](../failed-opts/).
