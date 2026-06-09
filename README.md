# 3D Airway Segmentation — MPI + CUDA + OpenMP

異質平行 3D 氣道分割管線：MPI Z-slab + CUDA separable Gaussian / Hessian vesselness + OpenMP region growing。針對 MICCAI AIIB23 (512×512×N，~120 cases) 開發；不使用任何學習式模型。

**最新結果**（25 cases，方法 v3）：平均 Dice **0.59**、中位數 **0.70**、13/25 ≥ 0.70。完整 120 cases × 4 方法的逐階段時間與 Dice 見 [results/benchmark_all.csv](results/benchmark_all.csv)。

---

## 目錄結構

```
Project/
├── README.md            ← 本檔
├── INSTRUCTIONS.md      ← 作業原始說明（Step 1–7 規格）
├── FRAMEWORK.md         ← v1：基底管線（MPI/CUDA/OpenMP）的設計與實作
├── FRAMEWORK_V2.md      ← v2：方案 A–G（LUT、金字塔、ROI、自適應、回饋）
├── FRAMEWORK_V3.md      ← v3：Frangi vesselness + 波前分支門檻 + 拓樸回退
├── BENCHMARK.md         ← 25-case 三路對照（v2 / v2.5 / v3）逐 case 表
│
├── Makefile             ← mpicxx + nvcc 編譯；產出 `airway_seg`
├── airway_seg           ← 編好的執行檔（CUDA sm_70 / V100）
│
├── src/                 ← 原始碼（細節見下節）
├── scripts/             ← 啟動 / 訓練 / 基準腳本（細節見下節）
├── data/                ← AIIB23 影像（img/）與 ground truth（gt/）
└── results/             ← 輸出遮罩、LUT、benchmark CSV
```

### `src/` — 原始碼

| 檔案 | 功能 |
|------|------|
| `main.cpp` | 主程式：解析旗標、串接 Load → MPI scatter+halo → CUDA filter → MPI gather → ROI/金字塔 → region grow → 寫 NIfTI → Dice。 |
| `io.h / io.cpp` | 自製 NIfTI-1 `.nii.gz` 讀寫器（zlib，無 nifticlib 依賴）；HU clamp；GT 載入。 |
| `mpi_partition.h / .cpp` | Z 軸切片分區、`MPI_Scatterv`、halo 交換（`MPI_Sendrecv`）、`MPI_Gatherv`。 |
| `cuda_filter.cuh / .cu` | GPU 核心：(1) 可分離 3D Gaussian（X tile + Y/Z flat） + fused 門檻；(2) Gaussian + gradient + **2D LUT 紋理查表**（cudaTextureObject）；(3) **多尺度 Frangi vesselness**（Hessian + 對稱 3×3 特徵值 + atomicMax 自適應 c）。 |
| `region_growing.h / .cpp` | rank-0 CPU 後處理：body bbox、逐 slice 平行 exterior flood-fill、central seeds、O(N) 二值膨脹、coarse 金字塔工具、自適應 coarse 門檻、洩漏 slice 修剪、多種子 OpenMP region growing，以及 v3 的 **wavefront_region_grow**（波前 BFS + 分支門檻 + 拓樸回退）。 |
| `lut.h / .cpp` | 方案 A：2D（亮度×梯度）機率 LUT 的離線訓練/累積/載入。 |
| `*.o` | 編譯產物，可 `make clean` 移除。 |

### `scripts/` — 工具腳本

| 腳本 | 用途 |
|------|------|
| `env.sh` | 載入叢集模組（`cuda/12.8` + `openmpi/5.0.2_ucx1.14.1_cuda12.3`）。**所有腳本都 source 它。** |
| `run.sh <NP> <OMP> [...args]` | 一鍵啟動：`bash scripts/run.sh 4 8 --input ... --pyramid ...` |
| `scaling.sh [case]` | Step 6 強擴展實驗（固定 volume，掃 1/2/4/8 ranks），輸出 `results/scaling_<case>.csv`。 |
| `train_lut.sh out.lut 100 101 ...` | 方案 A：對指定 case 累積 LUT counts。 |
| `benchmark25.sh` | 跑 25 cases (100–124) 的金字塔基準，輸出原始 timing 文字。 |
| `run_all_csv.sh [csv]` | **全資料集基準**：120 cases × 4 方法層級（baseline / pyramid / adaptive / vessel），逐 stage 時間+Dice 寫成 CSV（增量寫入，可中斷續看）。 |

### `data/` — 資料

```
data/img/AIIB23_<id>.nii.gz   ← 輸入 HU 影像（int16 → float32）
data/gt/AIIB23_<id>.nii.gz    ← 對應 ground-truth 氣道遮罩
```
120 個 case（id 介於 30–219，非連續）。`fixed_AIIB23.zip` 與 `*_Train_T1.zip` 是原始壓縮包，可忽略。

### `results/` — 輸出

- `mask_*.nii.gz` — 各次執行寫出的氣道遮罩（uint8 NIfTI）。
- `airway.lut` — 訓練好的 2D LUT（方案 A，可直接給 `--lut` 用）。
- `benchmark_all.csv` — **全資料集基準結果**（480 列：120 cases × 4 方法）。
- `benchmark_all.log` — 全資料集基準的逐筆進度。
- `bench/`、`bench_all/` — 各個遮罩 NIfTI（可大量占空間，不需保留）。

---

## 環境需求

- **叢集**：Taiwania-2（Rocky Linux 8）；Lmod 模組系統
- **編譯器/函式庫**：`mpicxx`（OpenMPI 5.0.2 + UCX 1.14.1 + CUDA 12.3）、`nvcc`（CUDA 12.8）、`zlib`
- **硬體**：CUDA capability ≥ 7.0（V100 用 `sm_70`，已寫死於 Makefile）

只要在叢集上能 `module load cuda/12.8 openmpi/5.0.2_ucx1.14.1_cuda12.3`，本專案即可直接編譯/執行。

---

## 編譯

```bash
cd /home/u4309334/Project
source scripts/env.sh        # 載入 cuda + openmpi 模組
make                          # 產出 ./airway_seg
make clean                    # 移除 src/*.o 與執行檔
```

---

## 執行

### 最簡用法（v3 完整管線、建議設定）

```bash
source scripts/env.sh                    # 一次即可
export OMPI_MCA_opal_cuda_support=0      # OpenMPI 不要嘗試 CUDA-aware 路徑

mpirun -np 4 ./airway_seg \
    --input data/img/AIIB23_110.nii.gz \
    --output results/mask_110.nii.gz \
    --gt    data/gt/AIIB23_110.nii.gz \
    --pyramid --adaptive --feedback --vessel \
    --omp-threads 8
```

或用包好的 wrapper（自動載 modules + 設環境變數）：

```bash
bash scripts/run.sh 4 8 \
    --input data/img/AIIB23_110.nii.gz \
    --output results/mask_110.nii.gz \
    --gt    data/gt/AIIB23_110.nii.gz \
    --pyramid --adaptive --feedback --vessel
```

### 四個方法層級（每升一級會加上前一級的旗標）

| 層級 | 旗標 | 平均 Dice（25 cases） |
|------|------|---------------------|
| baseline (v1) | （無：預設門檻 + ROI + multiseed） | 0.45 |
| pyramid  (v2) | `--pyramid` | 0.45 |
| adaptive (v2.5) | `--pyramid --adaptive --feedback` | 0.49 |
| **vessel  (v3)** | `--pyramid --adaptive --feedback --vessel` | **0.59 (median 0.70)** |

### 常用旗標

| 旗標 | 說明 |
|------|------|
| `--input <X.nii.gz>` | 輸入影像（必填） |
| `--output <Y.nii.gz>` | 輸出遮罩 |
| `--gt <GT.nii.gz>` | 若提供則計算 Dice |
| `--omp-threads <n>` | 每 rank 的 OpenMP threads（建議 8） |
| `--no-roi` | 關閉 body-bbox ROI（回退 v1 序列 3D flood-fill） |
| `--pyramid` | 啟用 ½ 解析度 coarse-trunk ROI 約束 |
| `--coarse-high <HU>` | 金字塔 coarse 上限（預設 −960；`--adaptive` 會自動算） |
| `--adaptive` | 用中央核心平滑空氣統計取代固定 coarse 門檻 |
| `--feedback`  | coarse trunk 洩漏時迭代收緊門檻 + 修剪爆炸 slice |
| `--vessel` | 啟用 GPU Frangi vesselness + 波前分支門檻 + 拓樸回退 |
| `--vmin <v>` | 波前放寬路徑的 vesselness 門檻（預設 0.03） |
| `--thr-periph <HU>` | 波前末梢 HU 上限（預設 −820） |
| `--lut <F.lut> --prob-thresh <p>` | 改用 2D LUT 紋理查表產生候選（方案 B） |
| `--validate` | 跑 GPU vs CPU Gaussian 的小型正確性比對 |

### 重現完整資料集基準

```bash
# 在 tmux 中跑（120 cases × 4 方法 ≈ 1 小時）
tmux new-session -d -s bench "bash scripts/run_all_csv.sh"
tail -f results/benchmark_all.log         # 即時看進度
tmux attach -t bench                      # Ctrl-b d 再次脫離
```

完成後 `results/benchmark_all.csv` 為 481 行（1 表頭 + 480 資料列）。欄位：
```
case, nz, method, load_s, mpi_part_s, gpu_filter_s, gather_s, region_grow_s, total_s, dice, pred_voxels, gt_voxels
```

### 強擴展實驗

```bash
bash scripts/scaling.sh 100               # 在 case 100 上跑 1/2/4/8 ranks
# -> results/scaling_100.csv
```

---

## 進一步閱讀

- 想了解**基底管線怎麼設計**：[FRAMEWORK.md](FRAMEWORK.md)
- 想了解 **LUT 紋理查表、金字塔 ROI、自適應門檻**：[FRAMEWORK_V2.md](FRAMEWORK_V2.md)
- 想了解 **Frangi vesselness、波前分支門檻、拓樸回退**：[FRAMEWORK_V3.md](FRAMEWORK_V3.md)
- 想看 **逐 case 結果對照**：[BENCHMARK.md](BENCHMARK.md)
- 想看 **作業原本要求**：[INSTRUCTIONS.md](INSTRUCTIONS.md)
