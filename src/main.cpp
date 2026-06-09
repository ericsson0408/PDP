// 3D Airway Segmentation — Heterogeneous MPI + CUDA + OpenMP pipeline.
//
// Pipeline:
//   [rank 0] load NIfTI volume, clamp HU
//   [MPI]    Scatterv Z-slabs to all ranks, exchange halo slices
//   [CUDA]   per-rank separable 3D Gaussian + fused air-thresholding
//   [MPI]    Gatherv candidate air mask to rank 0
//   [OpenMP] multi-seed region growing on rank 0 -> airway mask
//   [rank 0] write NIfTI mask, optional Dice vs ground truth
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

#include "io.h"
#include "cuda_filter.cuh"
#include "region_growing.h"
#include "mpi_partition.h"
#include "lut.h"
#include "onnx_infer.h"

struct Args {
    std::string input, output = "results/mask.nii.gz", gt;
    float sigma = 1.0f;
    int radius = 2;
    float tlo = -1100.0f, thi = -920.0f;
    float hu_min = -1024.0f, hu_max = 600.0f;
    int omp_threads = 8;
    bool validate = false;
    bool use_roi = true;          // scheme C/E/F: body-bbox ROI + parallel exterior
    float body_thr = -300.0f;     // HU above this counts as body tissue
    int roi_margin = 5;
    std::string train_lut;        // scheme A: accumulate LUT counts into this file
    std::string lut_file;         // scheme B: use this LUT for candidate generation
    float prob_thresh = 0.5f;     // LUT probability cutoff
    bool pyramid = false;         // scheme D: coarse-trunk -> dilated ROI constraint
    int dilate = 15;              // full-res dilation margin around coarse trunk
    float clo = -1100.f, chi = -960.f;  // coarse-level threshold band (leak-free trunk)
    bool adaptive = false;        // upgrade 招1+招3: data-driven coarse threshold
    bool feedback = false;        // upgrade 招2: prune leaked coarse-trunk slices
    float core_frac = 0.30f;      // central-core fraction for the spatial anchor
    float air_k = 2.5f;           // coarse threshold = mu + air_k * sigma
    bool vessel = false;          // V3: Frangi vesselness + wavefront grower
    float vmin = 0.03f;           // vesselness gate for the relaxed/tubular path
    float thr_periph = -820.f;    // relaxed HU ceiling at the periphery
    int wf_maxlevel = 60;         // BFS level at which threshold reaches periph
    // V5: robustness upgrades on top of vessel mode.
    bool v5 = false;              // implies pyramid+adaptive+feedback+vessel + the below
    float hu_abs_max = -750.f;    // v5: absolute air ceiling for the relaxed path
    float safety_ratio = 2.5f;    // v5: if vessel-mask voxels > ratio * pyramid-mask, revert
    // V6: HU-drift rescue. When the v5 pipeline produces 0 (or near-zero)
    // airway voxels, the per-case HU calibration is the root cause -- the
    // standard air threshold no longer fires inside the lumen (case 82).
    // The rescue is a CPU-side threshold sweep that finds the trachea by
    // geometry instead of by a fixed HU number.
    bool v6 = false;              // v5 + post-grow rescue when airway_voxels==0
    long long rescue_min = 10000; // trigger rescue if airway_voxels < this
                                  // (smallest healthy v5 case is ~30k vox -> 10k is safe)
    float rescue_slack  = 25.f;   // candidate built at (rescue_thr + slack)
    long long rescue_leak_cap = 600000; // if rescue grew > this, revert to baseline
                                        // (largest healthy v5 mask ~ 570k vox)
    // V7: best-of-all blend. v5/v6 used STRICT in-flight guards (vmin 0.06 etc.)
    // which saved category-C target cases in-flight but cost ~0.03-0.10 Dice on
    // ~100 healthy cases (vmin 0.06 rejects partial-volume peripheral airway).
    // v7 reverts to v3 LOOSE in-flight defaults (recovers healthy-case recall)
    // and relies purely on post-grow safety net to catch leaks. To still catch
    // case 120 (vessel/pyramid = 1.91x, below v5's 2.5x trigger), the safety
    // net adds a SECOND rule: revert when ratio > 1.5x AND vessel exceeds an
    // absolute size threshold (typical airway trees are <500k voxels, so >600k
    // alone is suspicious; combined with >1.5x growth it's a leak signature).
    bool v7 = false;              // v6 with loose in-flight + dual-rule safety
    float safety_ratio2 = 1.5f;   // v7 rule (b) relative threshold
    long long safety_abs = 600000;// v7 rule (b) absolute voxel floor for revert
    // V8: parallel-BFS region growers + pinned-pool GPU filter + recurrence
    // Frangi + mmap I/O + parallel Dice. Same algorithmic decisions as v7,
    // same Dice on the same case — only the implementation is faster.
    bool v8 = false;
    // V9: U-Net (DynUNet) ONNX inference replaces upstream candidate/feature
    // construction. Two variants share the same wavefront/safety-net/rescue
    // tail from v7/v8:
    //   --v9-1 (gentle): keep the pyramid hardset; replace Frangi vesselness
    //     with U-Net airway probability so the wavefront relaxed gate follows
    //     the network's tubular score instead of the closed-form Hessian one.
    //   --v9-2 (strong): drop the pyramid entirely. Build the hardset
    //     directly from (prob > thr) and feed it to the wavefront as both
    //     hardset and ROI envelope. The network does the heavy lifting for
    //     the trunk; region growing only refines the periphery.
    bool v9_1 = false;
    bool v9_2 = false;
    // V10: V9-1 base + U-Net-anchored *secondary* safety oracle. The V7/V8
    // dual-rule safety net compares the wavefront result against the pyramid
    // hardset, so when the pyramid itself leaks (case 124: pyramid + rescue
    // -> 494k voxels for a 135k-voxel airway) the safety net never fires.
    // V10 adds a third rule that compares the final mask against the U-Net
    // hardset and reverts to a U-Net-anchored grow when the final mask is
    // implausibly larger than the U-Net thinks the airway is. The U-Net
    // under-predicts the periphery so this rule won't fire on healthy cases
    // (where the V9-1 mask is large but the U-Net mask is *also* sizeable);
    // it only fires when the U-Net mask is anatomically plausible (>= 20k
    // voxels) AND the final mask is much larger (4x by default).
    bool v10 = false;
    long long v10_unet_floor = 20000;   // require U-Net mask >= this to fire
    float     v10_leak_ratio = 10.0f;   // final > ratio * U-Net -> swap.
                                        // Healthy cases sit around 5-7x
                                        // (U-Net under-predicts the
                                        // periphery); true leak cases jump
                                        // to 11x+ (case 124) up to 27x
                                        // (case 30). 10x cleanly separates.
    // V11: V10 + performance optimisations (batched ORT, FP16 model,
    // optional spacing-aware inference) + a multi-feature leak rule that
    // also considers whether the V6 rescue path fired (a strong leak
    // signal on its own). The performance knobs are exposed as flags so
    // V10 can also adopt them for free.
    bool v11 = false;
    // V11 leak rule: trigger if EITHER
    //   (a) rg > v10_leak_ratio * U-Net mask  (the V10 condition), OR
    //   (b) rescue fired AND rg > v11_rescue_leak_ratio * U-Net mask
    //       (catches rescue-induced over-grows even at modest ratios)
    bool      v11_rescue_fired   = false;   // runtime: set by rescue block
    float     v11_rescue_leak_ratio = 4.0f; // tighter ratio when rescue fired
    std::string onnx_model = "model/dynunet_best_model.onnx";
    int   onnx_patch  = 128;   // sliding-window patch size (cube)
    int   onnx_stride = 128;   // stride along each axis (== patch -> no overlap)
    float onnx_hu_lo  = -1000.f;
    float onnx_hu_hi  = 600.f;
    float onnx_thr    = 0.5f;  // probability threshold for the v9-2 hardset
    int   onnx_batch  = 1;     // V11: patches per Run().  1 = legacy V9 behaviour.
    // V11 spacing-aware inference: when target_sp_* > 0, the input HU
    // is resampled to (sx, sy, sz) mm before ORT, and the resulting
    // prob map is resampled back to the original shape. Empirically this
    // can lift U-Net Dice when the case's native spacing is far from the
    // network's training distribution.
    float target_sp_x = 0.f, target_sp_y = 0.f, target_sp_z = 0.f;
    // V12: inherits the full V11 algorithmic stack; adds C++-level inference
    // perf only -- (1) TensorRT EP (kernel fusion at the fixed 128^3 patch)
    // replaces the cuDNN CUDA EP, (2) double-buffered 2-stream H2D/compute
    // overlap hides the per-patch PCIe transfer behind GPU compute. No change
    // to Dice (same model, same softmax, same region-grow tail).
    bool v12 = false;
    bool v12_overlap = true;            // double-buffered stream overlap (off via --v12-no-overlap)
    bool trt_fp16 = true;               // enable TensorRT internal FP16
    std::string trt_cache = "model/trt_cache";  // engine cache dir
    // V13: algorithm hardening + threshold recalibration for the RETRAINED
    // (strong, ~0.90 standalone) model. Inherits the full V12 perf stack and
    // additionally fixes the three things that capped pipeline Dice at ~0.72:
    //   (1) overlapping sliding window (stride 96 = 25% overlap) -- removes the
    //       seam loss that cost ~0.08 Dice at native spacing,
    //   (2) spacing-aware inference at the model's training spacing
    //       [0.6,0.6,1.0] -- the network is trained there (~+0.10 Dice),
    //   (3) tighter leak ratio (10x -> 5x) -- the strong model predicts a
    //       larger/accurate airway, so real leaks now sit at lower rg/U-Net
    //       ratios and slipped past the 10x V10/V11 gate (e.g. case 30 @ 7x).
    bool v13 = false;
    // V14: trust the U-Net. With the retrained ~0.90-Dice DynUNet the entire
    // pyramid / wavefront / safety-net / V6 rescue / V10-V11 leak-rule stack
    // (built to rescue the old ~0.42 model) caps Dice and even regresses some
    // cases. V14 short-circuits all of it: mask = (prob > onnx_thr) filtered
    // to central-seeded connected components. Inherits V12 (TRT EP) + V13's
    // inference fixes (overlap stride 96 + spacing 0.6/0.6/1.0).
    bool v14 = false;
    // V15: V14 + two cheap accuracy refinements aimed at the few low-recall
    // cases (Dice<0.8 in V14: cases 63/118/158/145 -- all spacing outliers
    // with 47-60% recall) without touching the >0.9 majority:
    //   (a) 3D morphological closing of the hardset before central-CC --
    //       bridges 1-voxel-thin gaps so fragmented peripheral branches
    //       survive the connected-components prune;
    //   (b) "local relaxed threshold" -- include prob>v15_thr_lo voxels
    //       only inside a dilate3d envelope of the confident prob>onnx_thr
    //       core, so peripheral airway near a confident prediction is
    //       recovered without admitting far-away false positives.
    // Plus a free speed win: with v14/v15 the upstream MPI Gaussian / GPU
    // filter / gather chain is unused and can be skipped (~2 s/case).
    bool v15 = false;
    // V16: V15's `--onnx-thr 0.4` lifted recall on the low-confidence cases
    // (158: 0.715->0.830) but pulled vessel false-positives into the mask on
    // already-confident cases (case 168: 0.79->0.64). V16 adds a per-case
    // hardware-friendly anomaly check: compare the *count* of voxels at
    // thr 0.4 vs thr 0.5 -- if thr 0.4 inflates the mask by more than
    // `v16_gate_ratio` (default 0.2 = +20%), it's the case-168 over-seg
    // signature, fall back to thr 0.5. Costs two OpenMP reductions over the
    // prob volume (~10 ms), no extra GPU work.
    //   Pseudocode:  if (count(prob>0.4) - count(prob>0.5)) / count(prob>0.5) > 0.2
    //                  -> use thr 0.5  (safety: V14 behaviour)
    //                else
    //                  -> use thr 0.4  (V15 recall win)
    bool  v16 = false;
    // Default 0.5 picked from the 8-case smoke: case 168 (true over-seg) has
    // ratio 1.14, but cases 158/63 (under-seg, thr 0.4 *helps*) have ratios
    // 0.245/0.194 -- a 0.2 default falsely fires on the latter. 0.5 sits in
    // the gap (4.6x margin to 168, 2.0x margin to 158).
    float v16_gate_ratio = 0.5f;
    float v16_thr_lo     = 0.4f;   // threshold used when the gate does NOT fire
    float v16_thr_hi     = 0.5f;   // fallback when over-seg detected
    // V17 HYBRID -- AI-assisted traditional region growing. The U-Net's prob
    // map is the *hard gate*; traditional intensity-bounded BFS does the
    // actual periphery walking only in the AI uncertainty band.
    //   AI confident:    prob > v17_core_thr  -> hardset (always in mask)
    //   AI uncertain:    prob in (v17_prob_gate, v17_core_thr]  -> candidate
    //                    iff hu in [v17_hu_lo, v17_hu_hi] (lumen check)
    //                    -- BFS expands here from the hardset seeds.
    //   AI rejects:      prob <= v17_prob_gate  -> excluded (hard stop)
    bool  v17 = false;
    float v17_core_thr  = 0.5f;
    float v17_prob_gate = 0.1f;
    float v17_hu_lo     = -1000.0f;
    float v17_hu_hi     = -700.0f;   // airway lumen upper bound (sweep-tuned)
    // V17 firmware safety net: a (prob_gate, hu_hi) tuning sweep on 8 cases
    // showed case 168 stays stuck at ~0.74 across every parameter combo --
    // the issue isn't the uncertainty zone, it's that the U-Net CORE itself
    // (prob>0.5) contains FP vessel blobs on that case, and ANY hybrid BFS
    // through ANY uncertainty zone bridges those blobs into the central
    // tree so central-CC can't prune them. Solution: a V16-style anti-
    // over-seg ratio check fires BEFORE hybrid mode; if the U-Net is being
    // too generous, skip hybrid and run pure central-CC on the AI core.
    //   case 168 ratio = 1.30  -> abort, AI-only fallback
    //   case 158 ratio = 0.34  -> hybrid grows as designed
    //   default 0.8 gives 1.6x margin to 168, 2.1x to next-highest non-168 case
    float v17_abort_ratio = 0.8f;
    // Closing + local-relaxed-thr are now OPT-IN (default 0) -- the full
    // 120-case benchmark showed they trade evenly with V14 (mean Dice
    // changed by less than 0.001) while costing ~3 s/case in dilate3d.
    // The clean V15 accuracy win comes from `onnx_thr=0.4` instead of 0.5
    // (sweep on 8 hard cases: +0.024 mean, +0.045 on low-recall cases,
    // high cases unchanged). Keep the knobs for users who want to try
    // closing on case-specific tuning.
    int   v15_close_margin   = 0;     // 0 disables closing
    int   v15_relax_envelope = 0;     // 0 disables relaxed-thr extension
    float v15_thr_lo         = 0.3f;  // lower threshold inside the envelope (if enabled)
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (s == "--input") a.input = next();
        else if (s == "--output") a.output = next();
        else if (s == "--gt") a.gt = next();
        else if (s == "--sigma") a.sigma = atof(next());
        else if (s == "--radius") a.radius = atoi(next());
        else if (s == "--threshold-low") a.tlo = atof(next());
        else if (s == "--threshold-high") a.thi = atof(next());
        else if (s == "--hu-min") a.hu_min = atof(next());
        else if (s == "--hu-max") a.hu_max = atof(next());
        else if (s == "--omp-threads") a.omp_threads = atoi(next());
        else if (s == "--validate") a.validate = true;
        else if (s == "--no-roi") a.use_roi = false;
        else if (s == "--body-thr") a.body_thr = atof(next());
        else if (s == "--train-lut") a.train_lut = next();
        else if (s == "--lut") a.lut_file = next();
        else if (s == "--prob-thresh") a.prob_thresh = atof(next());
        else if (s == "--pyramid") a.pyramid = true;
        else if (s == "--dilate") a.dilate = atoi(next());
        else if (s == "--coarse-low") a.clo = atof(next());
        else if (s == "--coarse-high") a.chi = atof(next());
        else if (s == "--adaptive") a.adaptive = true;
        else if (s == "--feedback") a.feedback = true;
        else if (s == "--core-frac") a.core_frac = atof(next());
        else if (s == "--air-k") a.air_k = atof(next());
        else if (s == "--vessel") a.vessel = true;
        else if (s == "--vmin") a.vmin = atof(next());
        else if (s == "--thr-periph") a.thr_periph = atof(next());
        else if (s == "--wf-maxlevel") a.wf_maxlevel = atoi(next());
        else if (s == "--v5") {
            a.v5 = true; a.pyramid = true; a.adaptive = true;
            a.feedback = true; a.vessel = true;
        }
        else if (s == "--hu-abs-max") a.hu_abs_max = atof(next());
        else if (s == "--safety-ratio") a.safety_ratio = atof(next());
        else if (s == "--v6") {
            a.v6 = true; a.v5 = true; a.pyramid = true;
            a.adaptive = true; a.feedback = true; a.vessel = true;
        }
        else if (s == "--rescue-min") a.rescue_min = atoll(next());
        else if (s == "--rescue-slack") a.rescue_slack = atof(next());
        else if (s == "--rescue-leak-cap") a.rescue_leak_cap = atoll(next());
        else if (s == "--v7") {
            a.v7 = true; a.v6 = true; a.v5 = true; a.pyramid = true;
            a.adaptive = true; a.feedback = true; a.vessel = true;
            // v7 inherits v6 rescue but lifts the trigger floor. With v3-loose
            // in-flight guards the wavefront keeps more voxels, so case 124
            // (v3=0.11 with 12k voxels, v6 rescue gives 0.17) would skip
            // rescue at the old 10k floor. The v6 keepwall (`rescue improves
            // on baseline`) prevents regression even when rescue fires on a
            // small-but-real mask.
            a.rescue_min = 15000;
        }
        else if (s == "--safety-ratio2") a.safety_ratio2 = atof(next());
        else if (s == "--safety-abs") a.safety_abs = atoll(next());
        else if (s == "--v8") {
            // V8 inherits all v7 algorithmic decisions; only the *parallel
            // implementation* of the region growers, GPU filter, Frangi, I/O
            // and Dice differ. Same Dice on the same case, just faster.
            a.v8 = true; a.v7 = true; a.v6 = true; a.v5 = true;
            a.pyramid = true; a.adaptive = true; a.feedback = true;
            a.vessel = true; a.rescue_min = 15000;
        }
        else if (s == "--v9-1") {
            // V9-1 (gentle): inherit the full V8 stack; only swap Frangi
            // vesselness for U-Net airway probability inside the wavefront
            // relaxed-path gate.
            a.v9_1 = true;
            a.v8 = true; a.v7 = true; a.v6 = true; a.v5 = true;
            a.pyramid = true; a.adaptive = true; a.feedback = true;
            a.vessel = true; a.rescue_min = 15000;
        }
        else if (s == "--v9-2") {
            // V9-2 (strong): U-Net replaces both the pyramid hardset *and*
            // the Frangi vesselness. The wavefront refines on top.
            a.v9_2 = true;
            a.v8 = true; a.v7 = true; a.v6 = true; a.v5 = true;
            a.vessel = true; a.rescue_min = 15000;
            // We deliberately do NOT set pyramid/adaptive/feedback here --
            // U-Net replaces that whole stage.
        }
        else if (s == "--v10") {
            // V10: V9-1 base + U-Net post-grow safety oracle. Same in-flight
            // behaviour as V9-1; only adds a final swap rule.
            a.v10 = true;
            a.v9_1 = true;
            a.v8 = true; a.v7 = true; a.v6 = true; a.v5 = true;
            a.pyramid = true; a.adaptive = true; a.feedback = true;
            a.vessel = true; a.rescue_min = 15000;
        }
        else if (s == "--v10-leak-ratio") a.v10_leak_ratio = atof(next());
        else if (s == "--v10-unet-floor") a.v10_unet_floor = atoll(next());
        else if (s == "--v11") {
            // V11: V10 + ONNX FP16 speedup + multi-feature (rescue-aware)
            // leak rule. The FP16 mixed-precision model gives a clean ~1.8x
            // wall-clock reduction over FP32 with no measurable Dice loss
            // (TensorCore math activates on V100 SM_70).
            //
            // Batching ORT patches HURTS on V100 SM_70 (cuDNN's 3D-conv
            // algorithm scales poorly with batch dim for our patch size
            // — single-batch B=1 is consistently 20-50% faster wall-clock
            // than B=4 or B=8). Hence batch defaults to 1 in V11; the
            // --onnx-batch flag exists for experimentation on newer
            // architectures (Ampere/Hopper) where batching may win.
            //
            // Spacing-aware inference is also kept as opt-in: while it
            // brings the model's receptive field closer to its training
            // distribution, the trilinear resample of the probability
            // map back to native shape costs Dice on healthy cases
            // (peripheral airways get blurred). Enable per-case via
            // --onnx-target-spacing if you have a case far from native.
            a.v11 = true;
            a.v10 = true; a.v9_1 = true;
            a.v8 = true; a.v7 = true; a.v6 = true; a.v5 = true;
            a.pyramid = true; a.adaptive = true; a.feedback = true;
            a.vessel = true; a.rescue_min = 15000;
            // FP16 model is the headline V11 change.
            if (a.onnx_model == "model/dynunet_best_model.onnx") {
                a.onnx_model = "model/dynunet_best_model_fp16.onnx";
            }
        }
        else if (s == "--v12") {
            // V12: full V11 algorithmic stack (identical Dice path) + C++
            // inference perf (TensorRT EP + stream overlap). Inherit exactly
            // what --v11 sets, then mark v12 so ort_create swaps the EP.
            a.v12 = true;
            a.v11 = true; a.v10 = true; a.v9_1 = true;
            a.v8 = true; a.v7 = true; a.v6 = true; a.v5 = true;
            a.pyramid = true; a.adaptive = true; a.feedback = true;
            a.vessel = true; a.rescue_min = 15000;
            if (a.onnx_model == "model/dynunet_best_model.onnx") {
                a.onnx_model = "model/dynunet_best_model_fp16.onnx";
            }
        }
        else if (s == "--v12-no-overlap") a.v12_overlap = false;
        else if (s == "--trt-cache") a.trt_cache = next();
        else if (s == "--trt-no-fp16") a.trt_fp16 = false;
        else if (s == "--v14") {
            // V14: trust the U-Net. Inherit V12 (TRT EP + stream overlap),
            // keep V13's overlap fix, but DO NOT set pyramid/vessel/v5..v11
            // -- the region-growing tail is entirely replaced by a central-
            // seeded CC filter on (prob>onnx_thr).
            // Sweep on 6 hard cases (scripts/v14_sweep.sbatch) picked
            // native-spacing + onnx_thr=0.3 + overlap stride 96: lower
            // threshold recovers peripheral airway that thr=0.5 misses, and
            // skipping spacing avoids the trilinear round-trip loss on cases
            // with near-training native spacing (e.g. case 54 0.72->0.69 was
            // largely the resample, not the threshold).
            a.v14 = true;
            a.v12 = true;
            a.v9_1 = true;          // triggers ONNX inference -> unet_prob
            if (a.onnx_model == "model/dynunet_best_model.onnx")
                a.onnx_model = "model/dynunet_best_model_fp16.onnx";
            a.onnx_stride = 96;     // overlap 25%
            // onnx_thr stays at the standard 0.5 (training argmax threshold);
            // the 0.3 hack in an earlier V14 was compensating for the axis-swap
            // bug that's now fixed in onnx_infer.cpp.
        }
        else if (s == "--v15") {
            // V15 final: V14 + lower threshold (0.4) + skipped MPI Gaussian.
            // The lower threshold recovers the low-recall cases (mean Dice on
            // the 4 worst V14 cases: 0.683 -> 0.732) without changing the
            // high-Dice majority. Skipping the unused upstream MPI/GPU chain
            // for the v14/v15 path saves ~2 s/case. Closing + relaxed-thr
            // expansion are opt-in via --v15-close-margin / --v15-relax-envelope
            // -- on the full 120 cases they traded evenly with V14 while
            // costing ~3 s/case in dilate3d, so they're off by default.
            a.v15 = true;
            a.v14 = true;        // reuse the v14 RG short-circuit + skip-MPI gate
            a.v12 = true;
            a.v9_1 = true;
            if (a.onnx_model == "model/dynunet_best_model.onnx")
                a.onnx_model = "model/dynunet_best_model_fp16.onnx";
            a.onnx_stride = 96;
            a.onnx_thr    = 0.4f;
        }
        else if (s == "--v16") {
            // V16: V15 + dynamic gating. Same fast paths (TRT EP, overlap,
            // skip-MPI, central-CC). The threshold is chosen per case at
            // runtime based on a popcount-only over-segmentation check.
            a.v16 = true;
            a.v15 = true; a.v14 = true; a.v12 = true; a.v9_1 = true;
            if (a.onnx_model == "model/dynunet_best_model.onnx")
                a.onnx_model = "model/dynunet_best_model_fp16.onnx";
            a.onnx_stride = 96;
            a.onnx_thr    = 0.5f;   // will be re-chosen at runtime by the gate
        }
        else if (s == "--v16-gate-ratio") a.v16_gate_ratio = atof(next());
        else if (s == "--v16-thr-lo")     a.v16_thr_lo = atof(next());
        else if (s == "--v16-thr-hi")     a.v16_thr_hi = atof(next());
        else if (s == "--v17" || s == "--v17-hybrid") {
            // V17 HYBRID: V14's RG short-circuit + V12 TRT EP + V15 skip-MPI,
            // but replace V14's hardset+CC with the AI-gated traditional
            // wavefront grower (hybrid_grow). No leak rule, no closing, no
            // V16 dynamic gate -- the AI prob map IS the gate.
            a.v17 = true;
            a.v14 = true;       // reuse the V14 RG short-circuit + skip-MPI
            a.v12 = true;
            a.v9_1 = true;
            if (a.onnx_model == "model/dynunet_best_model.onnx")
                a.onnx_model = "model/dynunet_best_model_fp16.onnx";
            a.onnx_stride = 96;
        }
        else if (s == "--v17-core-thr")   a.v17_core_thr  = atof(next());
        else if (s == "--v17-prob-gate")  a.v17_prob_gate = atof(next());
        else if (s == "--v17-hu-lo")      a.v17_hu_lo = atof(next());
        else if (s == "--v17-hu-hi")      a.v17_hu_hi = atof(next());
        else if (s == "--v17-abort-ratio") a.v17_abort_ratio = atof(next());
        else if (s == "--fast") {
            // Convenience: cuts ~3-4 s ONNX time at the cost of ~0.014 mean
            // Dice on the harder cases. Sets stride 128 (vs default 96).
            // Composes with --v14/v15/v16 etc.
            a.onnx_stride = 128;
        }
        else if (s == "--v15-close-margin")   a.v15_close_margin = atoi(next());
        else if (s == "--v15-relax-envelope") a.v15_relax_envelope = atoi(next());
        else if (s == "--v15-thr-lo")         a.v15_thr_lo = atof(next());
        else if (s == "--v13") {
            // V13: inherit the full V12 stack (TensorRT EP + stream overlap +
            // V11 algorithm), then recalibrate for the retrained strong model.
            a.v13 = true;
            a.v12 = true;
            a.v11 = true; a.v10 = true; a.v9_1 = true;
            a.v8 = true; a.v7 = true; a.v6 = true; a.v5 = true;
            a.pyramid = true; a.adaptive = true; a.feedback = true;
            a.vessel = true; a.rescue_min = 15000;
            if (a.onnx_model == "model/dynunet_best_model.onnx")
                a.onnx_model = "model/dynunet_best_model_fp16.onnx";
            // (1) overlapping window, (2) train-spacing inference, (3) tight leak rule.
            a.onnx_stride = 96;
            a.target_sp_x = 0.6f; a.target_sp_y = 0.6f; a.target_sp_z = 1.0f;
            a.v10_leak_ratio = 5.0f;
        }
        else if (s == "--v11-rescue-leak-ratio") a.v11_rescue_leak_ratio = atof(next());
        else if (s == "--onnx-batch") a.onnx_batch = atoi(next());
        else if (s == "--onnx-target-spacing") {
            // e.g. --onnx-target-spacing 0.6,0.6,1.0
            std::string v = next();
            float vx=0, vy=0, vz=0;
            sscanf(v.c_str(), "%f,%f,%f", &vx, &vy, &vz);
            a.target_sp_x = vx; a.target_sp_y = vy; a.target_sp_z = vz;
        }
        else if (s == "--onnx-model")  a.onnx_model = next();
        else if (s == "--onnx-patch")  a.onnx_patch = atoi(next());
        else if (s == "--onnx-stride") a.onnx_stride = atoi(next());
        else if (s == "--onnx-hu-lo")  a.onnx_hu_lo = atof(next());
        else if (s == "--onnx-hu-hi")  a.onnx_hu_hi = atof(next());
        else if (s == "--onnx-thr")    a.onnx_thr = atof(next());
        else if (s.size() && s[0] != '-' && a.input.empty()) a.input = s;
    }
    return a;
}

// Compare GPU vs CPU separable Gaussian on a small synthetic volume.
static int run_validation(const Args& a, int device) {
    int n = 64;
    long long N = (long long)n * n * n;
    std::vector<float> in(N), gpu(N), cpu(N);
    for (long long i = 0; i < N; i++)
        in[i] = sinf(0.05f * i);  // deterministic unit-amplitude test field
    cuda_gaussian_smooth(in.data(), gpu.data(), n, n, n, a.radius, a.sigma, device);
    cpu_gaussian_reference(in.data(), cpu.data(), n, n, n, a.radius, a.sigma);
    double maxerr = 0.0, maxabs = 0.0;
    for (long long i = 0; i < N; i++) {
        maxerr = fmax(maxerr, fabs((double)gpu[i] - cpu[i]));
        maxabs = fmax(maxabs, fabs((double)cpu[i]));
    }
    double rel = maxerr / (maxabs > 0 ? maxabs : 1.0);
    printf("[validate] %d^3 separable Gaussian  max|GPU-CPU|=%.3e  relative=%.3e  (threshold 1e-4) -> %s\n",
           n, maxerr, rel, rel < 1e-4 ? "PASS" : "FAIL");
    return rel < 1e-4 ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    Args a = parse_args(argc, argv);

    int ndev = cuda_device_count();
    int device = (ndev > 0) ? (rank % ndev) : 0;
    if (ndev == 0) {
        if (rank == 0) fprintf(stderr, "[error] no CUDA device visible\n");
        MPI_Finalize(); return 1;
    }

    if (a.validate) {
        int rc = (rank == 0) ? run_validation(a, device) : 0;
        MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return rc;
    }

    if (a.input.empty()) {
        if (rank == 0) fprintf(stderr, "usage: airway_seg --input X.nii.gz --output Y.nii.gz [--gt GT] [--sigma s] [--radius r] [--threshold-low/-high v] [--lut F] [--omp-threads n]\n");
        MPI_Finalize(); return 1;
    }

    // ---- Scheme A: offline LUT training (rank 0, single case) ----
    if (!a.train_lut.empty()) {
        int rc = 0;
        if (rank == 0) {
            Volume v; unsigned char* gt = nullptr; int gx, gy, gz;
            if (a.gt.empty()) { fprintf(stderr, "[train] --gt required\n"); rc = 1; }
            else if (!load_nifti(a.input, v)) rc = 2;
            else if (!(gt = load_label_u8(a.gt, gx, gy, gz)) || gx!=v.nx||gy!=v.ny||gz!=v.nz) {
                fprintf(stderr, "[train] gt load/dim mismatch\n"); rc = 3;
            } else {
                clamp_hu(v.data, v.voxels(), a.hu_min, a.hu_max);
                std::vector<float> smoothed(v.voxels());
                cuda_gaussian_smooth(v.data, smoothed.data(), v.nx, v.ny, v.nz, a.radius, a.sigma, device);
                Lut2D spec;
                if (!lut_train_accumulate(a.train_lut, smoothed.data(), gt, v.nx, v.ny, v.nz, spec)) rc = 4;
            }
            delete[] gt;
        }
        MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return rc;
    }

    // Load LUT for inference (scheme B) on every rank.
    Lut2D lut; bool use_lut = false;
    if (!a.lut_file.empty()) {
        use_lut = lut_load(a.lut_file, lut);
        if (!use_lut) { if (rank==0) fprintf(stderr,"[error] failed to load LUT\n"); MPI_Finalize(); return 1; }
    }

    double t_total0 = MPI_Wtime();

    // ---- Step 1: load + clamp (rank 0) ----
    Volume vol;
    int dims[3] = {0, 0, 0};
    double t_load0 = MPI_Wtime();
    if (rank == 0) {
        if (!load_nifti(a.input, vol)) { MPI_Abort(MPI_COMM_WORLD, 2); }
        clamp_hu(vol.data, vol.voxels(), a.hu_min, a.hu_max);
        dims[0] = vol.nx; dims[1] = vol.ny; dims[2] = vol.nz;
        printf("[load] %s  dims=%dx%dx%d  spacing=%.3f,%.3f,%.3f mm\n",
               a.input.c_str(), vol.nx, vol.ny, vol.nz, vol.dx, vol.dy, vol.dz);
    }
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    int nx = dims[0], ny = dims[1], nz = dims[2];
    long long plane = (long long)nx * ny;
    double t_load = MPI_Wtime() - t_load0;

    // ---- Step 2/3/5a: MPI scatter + GPU Gaussian + Gather (skipped for V14/V15) ----
    // The cuda_gaussian_threshold output feeds only the V5-V13 region-growing
    // tail. V14/V15 short-circuit the tail and read straight from `unet_prob`
    // + `vol.data`, so this whole ~2 s/case chain is dead work for them.
    double t_mpi = 0.0, t_gpu = 0.0, t_gather = 0.0;
    float kms_max = 0;
    std::vector<unsigned char> full;
    if (!a.v14) {
        double t_mpi0 = MPI_Wtime();
        std::vector<int> zc, zd;
        compute_partition(nz, nprocs, zc, zd);
        int local_nz = zc[rank];
        std::vector<float> local((size_t)local_nz * plane);
        scatter_slab(rank == 0 ? vol.data : nullptr, local.data(), nx, ny, zc, zd, rank);

        int halo = a.radius;
        std::vector<float> ext;
        build_extended_with_halo(local.data(), nx, ny, local_nz, halo, rank, nprocs, ext);
        int ext_nz = local_nz + 2 * halo;
        t_mpi = MPI_Wtime() - t_mpi0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t_gpu0 = MPI_Wtime();
        std::vector<unsigned char> ext_mask((size_t)ext_nz * plane);
        float kms;
        if (use_lut) {
            kms = cuda_gaussian_lut(ext.data(), ext_mask.data(), nx, ny, ext_nz,
                                    a.radius, a.sigma, lut.prob.data(),
                                    lut.Ibins, lut.Gbins, lut.Imin, lut.Imax, lut.Gmax,
                                    a.prob_thresh, device);
        } else if (a.v8) {
            kms = cuda_gaussian_threshold_v8(ext.data(), ext_mask.data(),
                                             nx, ny, ext_nz, a.radius, a.sigma,
                                             a.tlo, a.thi, device);
        } else {
            kms = cuda_gaussian_threshold(ext.data(), ext_mask.data(),
                                          nx, ny, ext_nz, a.radius, a.sigma,
                                          a.tlo, a.thi, device);
        }
        if (kms < 0) MPI_Abort(MPI_COMM_WORLD, 3);
        std::vector<unsigned char> local_mask((size_t)local_nz * plane);
        std::memcpy(local_mask.data(), ext_mask.data() + (size_t)halo * plane,
                    (size_t)local_nz * plane);
        t_gpu = MPI_Wtime() - t_gpu0;

        MPI_Reduce(&kms, &kms_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

        double t_gather0 = MPI_Wtime();
        if (rank == 0) full.assign((size_t)nz * plane, 0);
        gather_mask(local_mask.data(), rank == 0 ? full.data() : nullptr, nx, ny, zc, zd, rank);
        t_gather = MPI_Wtime() - t_gather0;
    }

    // ---- Step 5b (v9): U-Net inference on the full gathered volume ----
    // The U-Net runs on the full volume on rank 0 only (the MPI Z-slab
    // partition is purely for the upstream Gaussian filter; we already
    // gathered above). Output is a per-voxel airway probability map shared
    // with either (a) the wavefront's vesselness gate (v9-1) or (b) the
    // hardset + ROI envelope (v9-2).
    double t_onnx = 0.0;
    std::vector<float> unet_prob;
    if (rank == 0 && (a.v9_1 || a.v9_2)) {
        long long N = (long long)nx * ny * nz;
        unet_prob.assign((size_t)N, 0.f);
        double t_onnx0 = MPI_Wtime();
        OrtOptions oo;
        oo.use_trt        = a.v12;               // TensorRT EP only under --v12
        oo.trt_fp16       = a.trt_fp16;
        oo.trt_engine_cache = true;
        oo.trt_cache_path = a.trt_cache;
        oo.stream_overlap = a.v12 && a.v12_overlap;
        OrtPipeline* sess = ort_create(a.onnx_model, device, oo);
        if (!sess) MPI_Abort(MPI_COMM_WORLD, 5);

        // V11 spacing-aware path: resample HU to the model's training
        // spacing before inference, then resample the prob map back to
        // the original shape. Optional — when target_sp_* == 0 we just
        // pass the native volume.
        bool do_resample = (a.target_sp_x > 0.f && a.target_sp_y > 0.f
                            && a.target_sp_z > 0.f);
        float ms = -1.f;
        if (do_resample) {
            // Scale factors to go from src spacing to dst spacing.
            int rnx = (int)std::round(nx * vol.dx / a.target_sp_x);
            int rny = (int)std::round(ny * vol.dy / a.target_sp_y);
            int rnz = (int)std::round(nz * vol.dz / a.target_sp_z);
            // Clamp to multiples of 8 (DynUNet has 4 downsamples / stride 16
            // overall, but we pad inside ORT so any size works; round to
            // 16 to keep patch boundaries aligned).
            auto round16 = [](int v) { return ((v + 7) / 8) * 8; };
            rnx = std::max(16, round16(rnx));
            rny = std::max(16, round16(rny));
            rnz = std::max(16, round16(rnz));
            printf("[v11] resample %dx%dx%d (%.3fx%.3fx%.3f mm) -> "
                   "%dx%dx%d (%.3fx%.3fx%.3f mm)\n",
                   nx, ny, nz, vol.dx, vol.dy, vol.dz,
                   rnx, rny, rnz, a.target_sp_x, a.target_sp_y, a.target_sp_z);

            // Trilinear downsample of HU on CPU (OpenMP). For each
            // resampled voxel (rx,ry,rz), look up the source location
            // and trilerp.
            std::vector<float> hu_r((size_t)rnx * rny * rnz);
            const float sx = (float)(nx - 1) / std::max(1, rnx - 1);
            const float sy = (float)(ny - 1) / std::max(1, rny - 1);
            const float sz = (float)(nz - 1) / std::max(1, rnz - 1);
            #pragma omp parallel for collapse(2) schedule(static)
            for (int rz = 0; rz < rnz; rz++) {
                for (int ry = 0; ry < rny; ry++) {
                    float fz = rz * sz; int z0 = (int)fz; int z1 = std::min(z0+1, nz-1); float tz = fz - z0;
                    float fy = ry * sy; int y0 = (int)fy; int y1 = std::min(y0+1, ny-1); float ty = fy - y0;
                    float* dst = hu_r.data() + ((long long)rz * rny + ry) * rnx;
                    for (int rx = 0; rx < rnx; rx++) {
                        float fx = rx * sx; int x0 = (int)fx; int x1 = std::min(x0+1, nx-1); float tx = fx - x0;
                        const float* p000 = vol.data + (long long)z0*plane + (long long)y0*nx;
                        const float* p001 = vol.data + (long long)z0*plane + (long long)y1*nx;
                        const float* p010 = vol.data + (long long)z1*plane + (long long)y0*nx;
                        const float* p011 = vol.data + (long long)z1*plane + (long long)y1*nx;
                        float c00 = p000[x0]*(1-tx) + p000[x1]*tx;
                        float c01 = p001[x0]*(1-tx) + p001[x1]*tx;
                        float c10 = p010[x0]*(1-tx) + p010[x1]*tx;
                        float c11 = p011[x0]*(1-tx) + p011[x1]*tx;
                        float c0 = c00*(1-ty) + c01*ty;
                        float c1 = c10*(1-ty) + c11*ty;
                        dst[rx] = c0*(1-tz) + c1*tz;
                    }
                }
            }

            std::vector<float> prob_r((size_t)rnx * rny * rnz, 0.f);
            ms = ort_infer_volume(sess, hu_r.data(), prob_r.data(),
                                  rnx, rny, rnz,
                                  a.onnx_patch, a.onnx_patch, a.onnx_patch,
                                  a.onnx_stride, a.onnx_stride, a.onnx_stride,
                                  a.onnx_hu_lo, a.onnx_hu_hi,
                                  a.onnx_batch);

            // Inverse trilerp prob_r back to native shape.
            const float ix = (float)(rnx - 1) / std::max(1, nx - 1);
            const float iy = (float)(rny - 1) / std::max(1, ny - 1);
            const float iz = (float)(rnz - 1) / std::max(1, nz - 1);
            #pragma omp parallel for collapse(2) schedule(static)
            for (int z = 0; z < nz; z++) {
                for (int y = 0; y < ny; y++) {
                    float fz = z * iz; int z0 = (int)fz; int z1 = std::min(z0+1, rnz-1); float tz = fz - z0;
                    float fy = y * iy; int y0 = (int)fy; int y1 = std::min(y0+1, rny-1); float ty = fy - y0;
                    float* dst = unet_prob.data() + (long long)z*plane + (long long)y*nx;
                    for (int x = 0; x < nx; x++) {
                        float fx = x * ix; int x0 = (int)fx; int x1 = std::min(x0+1, rnx-1); float tx = fx - x0;
                        const float* p000 = prob_r.data() + (long long)z0*rnx*rny + (long long)y0*rnx;
                        const float* p001 = prob_r.data() + (long long)z0*rnx*rny + (long long)y1*rnx;
                        const float* p010 = prob_r.data() + (long long)z1*rnx*rny + (long long)y0*rnx;
                        const float* p011 = prob_r.data() + (long long)z1*rnx*rny + (long long)y1*rnx;
                        float c00 = p000[x0]*(1-tx) + p000[x1]*tx;
                        float c01 = p001[x0]*(1-tx) + p001[x1]*tx;
                        float c10 = p010[x0]*(1-tx) + p010[x1]*tx;
                        float c11 = p011[x0]*(1-tx) + p011[x1]*tx;
                        float c0 = c00*(1-ty) + c01*ty;
                        float c1 = c10*(1-ty) + c11*ty;
                        dst[x] = c0*(1-tz) + c1*tz;
                    }
                }
            }
        } else {
            ms = ort_infer_volume(sess, vol.data, unet_prob.data(),
                                  nx, ny, nz,
                                  a.onnx_patch, a.onnx_patch, a.onnx_patch,
                                  a.onnx_stride, a.onnx_stride, a.onnx_stride,
                                  a.onnx_hu_lo, a.onnx_hu_hi,
                                  a.onnx_batch);
        }
        ort_destroy(sess);
        t_onnx = MPI_Wtime() - t_onnx0;
        if (ms < 0) MPI_Abort(MPI_COMM_WORLD, 6);
        // Cheap stats so we can sanity-check the model is producing
        // reasonable probabilities (mean << 1, max ~1 for airway voxels).
        double s = 0.0; float mx = 0.f; long long pos = 0;
        #pragma omp parallel for schedule(static) reduction(+:s,pos) reduction(max:mx)
        for (long long i = 0; i < N; i++) {
            float v = unet_prob[i];
            s += v; if (v > mx) mx = v;
            if (v > 0.5f) pos++;
        }
        printf("[v9] ONNX inference: %.1f ms  mean=%.4f max=%.3f  voxels>0.5=%lld\n",
               ms, s / (double)N, mx, pos);
    }

    // V10: capture the U-Net hardset BEFORE the V9-1 pipeline moves
    // unet_prob into vness. The hardset is a small (uint8 N-voxel) buffer;
    // we hold onto it for the post-grow leak-check.
    std::vector<unsigned char> unet_hardset_v10;
    long long unet_hardset_voxels = 0;
    if (rank == 0 && a.v10) {
        long long N = (long long)nx * ny * nz;
        unet_hardset_v10.assign((size_t)N, 0);
        long long sum = 0;
        #pragma omp parallel for schedule(static) reduction(+:sum)
        for (long long i = 0; i < N; i++) {
            unsigned char b = (unet_prob[i] > a.onnx_thr) ? 1 : 0;
            unet_hardset_v10[i] = b; sum += b;
        }
        unet_hardset_voxels = sum;
        printf("[v10] captured U-Net hardset @ thr=%.2f  voxels=%lld\n",
               a.onnx_thr, unet_hardset_voxels);
    }

    // ---- Step 4: OpenMP multi-seed region growing (rank 0) ----
    double t_rg = 0.0;
    RegionGrowResult rg;
    if (rank == 0) {
        double t_rg0 = MPI_Wtime();
        long long N = (long long)nx * ny * nz;
        std::vector<unsigned char> interior;
        std::vector<Seed> seeds;

        BBox roi;
        if (a.use_roi) {
            roi = compute_body_bbox(vol.data, nx, ny, nz, a.body_thr, a.roi_margin, a.omp_threads);
            printf("[roi] body bbox x[%d,%d] y[%d,%d] z[%d,%d]  %.1f%% of volume\n",
                   roi.x0, roi.x1, roi.y0, roi.y1, roi.z0, roi.z1,
                   100.0 * roi.voxels(nx, ny) / (double)N);
        } else {
            roi.valid = true; roi.x0=0; roi.y0=0; roi.z0=0; roi.x1=nx-1; roi.y1=ny-1; roi.z1=nz-1;
        }

        // ---- V14: trust the U-Net (skip the entire region-growing tail) ----
        // Build the mask as (prob > onnx_thr), then keep only the connected
        // components reachable from the central-axis seeds (drops the network's
        // sporadic small false-positive blobs). Same multiseed_region_grow_v8
        // primitive the V10 leak-swap uses to anchor on the U-Net.
        if (a.v14 && a.v17) {
            // V17 HYBRID short-circuits the V14/V15/V16 hardset+CC. Builds
            // an AI core (prob > core_thr) and an AI uncertainty band, and
            // runs traditional intensity-bounded wavefront BFS through the
            // band hard-stopped at prob <= prob_gate.
            //
            // FIRMWARE SAFETY NET: before invoking the hybrid grow we run a
            // V16-style anti-over-seg check on the U-Net's own confidence
            // distribution. If too much of the volume is in the AI's
            // uncertainty band (count(>prob_gate) - count(>core_thr)) /
            // count(>core_thr) > v17_abort_ratio, the network is being
            // over-generous on this case and the BFS would bridge FP blobs
            // (case-168 failure pattern). In that case we skip the hybrid
            // and use pure central-CC on the AI core only.
            std::vector<unsigned char> hardset((size_t)N, 0);
            long long c_core = 0, c_gate = 0;
            #pragma omp parallel for schedule(static) reduction(+:c_core,c_gate)
            for (long long i = 0; i < N; i++) {
                unsigned char b = (unet_prob[i] > a.v17_core_thr) ? 1 : 0;
                hardset[i] = b; c_core += b;
                c_gate += (unet_prob[i] > a.v17_prob_gate) ? 1 : 0;
            }
            double abort_ratio = c_core > 0
                ? (double)(c_gate - c_core) / (double)c_core
                : 0.0;
            printf("[v17] AI core @ thr=%.2f -> %lld voxels  "
                   "(uncertainty band prob in (%.2f, %.2f], hu in [%.0f,%.0f])  "
                   "abort_ratio=%.3f\n",
                   a.v17_core_thr, c_core,
                   a.v17_prob_gate, a.v17_core_thr,
                   a.v17_hu_lo, a.v17_hu_hi, abort_ratio);

            if (abort_ratio > a.v17_abort_ratio) {
                // Firmware fallback: U-Net is over-generous on this case;
                // hybrid BFS would bridge FP blobs. Drop to pure AI mode.
                printf("[v17] SAFETY ABORT: abort_ratio %.3f > %.2f "
                       "-> AI-only central-CC fallback\n",
                       abort_ratio, a.v17_abort_ratio);
                std::vector<Seed> v17_seeds = central_axis_seeds_robust(
                        hardset.data(), nx, ny, nz, roi);
                if (v17_seeds.empty()) {
                    rg.mask = std::move(hardset);
                    rg.airway_voxels = c_core;
                    rg.interior_voxels = c_core;
                    rg.num_seeds = 0;
                } else {
                    rg = multiseed_region_grow_v8(
                            hardset.data(), nx, ny, nz,
                            v17_seeds, a.omp_threads);
                }
                printf("[v17] AI-only: seeds=%d  airway=%lld voxels\n",
                       rg.num_seeds, rg.airway_voxels);
            } else {
                rg = hybrid_grow(vol.data, unet_prob.data(), hardset,
                                 nx, ny, nz,
                                 a.v17_hu_lo, a.v17_hu_hi, a.v17_prob_gate,
                                 roi, a.omp_threads);
                printf("[v17] hybrid grow: seeds=%d  candidate=%lld voxels  "
                       "final airway=%lld voxels (grew %+lld from core)\n",
                       rg.num_seeds, rg.interior_voxels,
                       rg.airway_voxels, rg.airway_voxels - c_core);
            }
        }
        else if (a.v14) {
            float chosen_thr = a.onnx_thr;
            if (a.v16) {
                // Dynamic gating: two popcount-only OpenMP reductions over the
                // prob volume (~10 ms on 512^3) decide whether the lower V15
                // threshold over-segments this particular case. No GPU work,
                // no extra inference. Pure runtime safety net.
                long long c_lo = 0, c_hi = 0;
                #pragma omp parallel for schedule(static) reduction(+:c_lo,c_hi)
                for (long long i = 0; i < N; i++) {
                    c_lo += (unet_prob[i] > a.v16_thr_lo) ? 1 : 0;
                    c_hi += (unet_prob[i] > a.v16_thr_hi) ? 1 : 0;
                }
                double ratio = (c_hi > 0)
                    ? (double)(c_lo - c_hi) / (double)c_hi
                    : 0.0;
                if (ratio > a.v16_gate_ratio) {
                    chosen_thr = a.v16_thr_hi;
                    printf("[v16] gate FIRED: count(>%.2f)=%lld count(>%.2f)=%lld "
                           "ratio=%.3f > %.2f -> fallback thr=%.2f (anti-over-seg)\n",
                           a.v16_thr_lo, c_lo, a.v16_thr_hi, c_hi,
                           ratio, a.v16_gate_ratio, chosen_thr);
                } else {
                    chosen_thr = a.v16_thr_lo;
                    printf("[v16] gate ok:    count(>%.2f)=%lld count(>%.2f)=%lld "
                           "ratio=%.3f <= %.2f -> thr=%.2f (recall mode)\n",
                           a.v16_thr_lo, c_lo, a.v16_thr_hi, c_hi,
                           ratio, a.v16_gate_ratio, chosen_thr);
                }
            }
            std::vector<unsigned char> hardset((size_t)N, 0);
            long long hs = 0;
            #pragma omp parallel for schedule(static) reduction(+:hs)
            for (long long i = 0; i < N; i++) {
                unsigned char b = (unet_prob[i] > chosen_thr) ? 1 : 0;
                hardset[i] = b; hs += b;
            }
            printf("[v14] U-Net hardset @ thr=%.2f -> %lld voxels\n",
                   chosen_thr, hs);

            if (a.v15) {
                // (a) local relaxed threshold: include prob>v15_thr_lo
                //     voxels only inside a v15_relax_envelope dilation of
                //     the confident core. Adds peripheral airway near the
                //     U-Net's confident prediction without admitting far
                //     false positives.
                if (a.v15_relax_envelope > 0 && a.v15_thr_lo < a.onnx_thr) {
                    std::vector<unsigned char> env = hardset;
                    dilate3d(env, nx, ny, nz, a.v15_relax_envelope, a.omp_threads);
                    long long added = 0;
                    #pragma omp parallel for schedule(static) reduction(+:added)
                    for (long long i = 0; i < N; i++) {
                        if (!hardset[i] && env[i] && unet_prob[i] > a.v15_thr_lo) {
                            hardset[i] = 1; added++;
                        }
                    }
                    hs += added;
                    printf("[v15] relaxed thr=%.2f within %d-dilation envelope: +%lld voxels -> %lld\n",
                           a.v15_thr_lo, a.v15_relax_envelope, added, hs);
                }
                // (b) 3D morphological closing: fill 1-voxel gaps so
                //     fragmented branches survive central_seeded_CC.
                if (a.v15_close_margin > 0) {
                    long long before = hs;
                    close3d(hardset, nx, ny, nz, a.v15_close_margin, a.omp_threads);
                    long long after = 0;
                    #pragma omp parallel for schedule(static) reduction(+:after)
                    for (long long i = 0; i < N; i++) after += hardset[i];
                    printf("[v15] close3d(margin=%d): %lld -> %lld voxels (+%lld)\n",
                           a.v15_close_margin, before, after, after - before);
                    hs = after;
                }
            }

            std::vector<Seed> v14_seeds = central_axis_seeds_robust(
                    hardset.data(), nx, ny, nz, roi);
            if (v14_seeds.empty()) {
                rg.mask = std::move(hardset);
                rg.airway_voxels   = hs;
                rg.interior_voxels = hs;
                rg.num_seeds = 0;
                printf("[v14] no central seed -> raw hardset (no CC filter)\n");
            } else {
                rg = multiseed_region_grow_v8(hardset.data(), nx, ny, nz,
                                              v14_seeds, a.omp_threads);
                printf("[v14] central-seeded CC: seeds=%d  airway=%lld voxels\n",
                       rg.num_seeds, rg.airway_voxels);
            }
        }

        if (!a.v14) {

        // Scheme D: build a coarse airway trunk at half resolution, dilate it
        // into an ROI constraint, and intersect it with the candidate BEFORE the
        // full-res flood-fill -- so the expensive exterior pass runs on a sparse
        // (leak-free) candidate instead of the whole leaked air region.
        const unsigned char* cand = full.data();
        std::vector<unsigned char> cand2, constraint;
        if (a.pyramid) {
            std::vector<float> chu; int dnx, dny, dnz;
            downsample2x_avg(vol.data, nx, ny, nz, chu, dnx, dny, dnz, a.omp_threads);
            long long dN = (long long)dnx * dny * dnz;

            // Upgrade 招1+招3: replace the fixed coarse ceiling with a data-driven
            // threshold from central-core smooth-air statistics (per-case robust).
            float chi = a.chi;
            if (a.adaptive) {
                BBox cbody = compute_body_bbox(chu.data(), dnx, dny, dnz,
                                               a.body_thr, 0, a.omp_threads);
                chi = adaptive_coarse_threshold(chu.data(), dnx, dny, dnz, cbody,
                                                a.air_k, a.core_frac, a.omp_threads);
                printf("[adaptive] coarse air threshold = %.1f HU (static was %.1f)\n",
                       chi, a.chi);
            }

            BBox croi; croi.valid = true; croi.x0=0; croi.y0=0; croi.z0=0;
            croi.x1=dnx-1; croi.y1=dny-1; croi.z1=dnz-1;
            std::vector<unsigned char> ccand(dN), cinterior;
            RegionGrowResult ctrunk;

            // Upgrade 招2: anatomical feedback. A healthy trunk is a small
            // fraction of the volume; a global leak floods the lung. If the
            // coarse trunk explodes past leak_cap, the band is too loose -- back
            // the threshold off and re-grow (cheap at 1/8 resolution).
            long long leak_cap = (long long)(0.012 * (double)dN);  // 1.2% of coarse volume
            // v5: track the smallest non-empty trunk across feedback retries.
            // If the loop never falls under the cap, we still have a useable
            // best instead of the (possibly explosive) last iteration.
            long long best_trunk = -1;
            std::vector<unsigned char> best_mask;
            for (int tries = 0; ; tries++) {
                #pragma omp parallel for schedule(static)
                for (long long i = 0; i < dN; i++)
                    ccand[i] = (chu[i] >= a.clo && chu[i] <= chi) ? 1 : 0;
                std::vector<Seed> cseeds = detect_seeds_parallel(ccand.data(), dnx, dny, dnz,
                                                                 croi, a.omp_threads, cinterior);
                ctrunk = multiseed_region_grow(cinterior.data(), dnx, dny, dnz,
                                               cseeds, a.omp_threads);
                if (a.v5 && ctrunk.airway_voxels > 0 &&
                    (best_trunk < 0 || ctrunk.airway_voxels < best_trunk)) {
                    best_trunk = ctrunk.airway_voxels;
                    best_mask  = ctrunk.mask;
                }
                if (!a.feedback || ctrunk.airway_voxels <= leak_cap || tries >= 8) break;
                printf("[feedback] trunk=%lld > cap=%lld -> tighten chi %.1f -> %.1f\n",
                       ctrunk.airway_voxels, leak_cap, chi, chi - 25.0f);
                chi -= 25.0f;
            }
            if (a.v5 && ctrunk.airway_voxels > leak_cap && best_trunk > 0) {
                printf("[v5] feedback failed to converge; reverting to best trunk=%lld (was %lld)\n",
                       best_trunk, ctrunk.airway_voxels);
                ctrunk.mask = std::move(best_mask);
                ctrunk.airway_voxels = best_trunk;
            }
            // Residual localized leaks (a few exploded slices) -> prune them.
            if (a.feedback) {
                long long before = ctrunk.airway_voxels;
                prune_leaked_slices(ctrunk.mask, dnx, dny, dnz, a.omp_threads);
                long long after = 0;
                #pragma omp parallel for schedule(static) reduction(+:after)
                for (long long i = 0; i < dN; i++) after += ctrunk.mask[i];
                if (after != before)
                    printf("[feedback] pruned leaked slices: trunk %lld -> %lld voxels\n",
                           before, after);
            }
            // Dilate at coarse resolution (1/8 the voxels) by half the margin,
            // then upsample -> ~same full-res envelope at a fraction of the cost.
            dilate3d(ctrunk.mask, dnx, dny, dnz, (a.dilate + 1) / 2, a.omp_threads);
            constraint = upsample_dilate(ctrunk.mask, dnx, dny, dnz, nx, ny, nz, 0, a.omp_threads);
            cand2.resize(N);
            long long kept = 0;
            #pragma omp parallel for schedule(static) reduction(+:kept)
            for (long long i = 0; i < N; i++) { cand2[i] = full[i] & constraint[i]; kept += cand2[i]; }
            cand = cand2.data();
            printf("[pyramid] coarse %dx%dx%d trunk=%lld -> dilated ROI; candidate kept=%lld\n",
                   dnx, dny, dnz, ctrunk.airway_voxels, kept);
        }

        // V9-2: skip the pyramid entirely; build hardset + ROI envelope
        // directly from the U-Net probability. The wavefront then refines
        // edges. We treat (prob > thr) as cand2 ("strict hardset"), and use
        // a slightly dilated version as the growth envelope.
        if (a.v9_2 && !a.pyramid) {
            cand2.assign((size_t)N, 0);
            constraint.assign((size_t)N, 0);
            long long kept = 0;
            #pragma omp parallel for schedule(static) reduction(+:kept)
            for (long long i = 0; i < N; i++) {
                unsigned char b = (unet_prob[i] > a.onnx_thr) ? 1 : 0;
                cand2[i] = b; constraint[i] = b; kept += b;
            }
            // Loose envelope so the wavefront can push 2-3 voxels beyond the
            // U-Net boundary where the network's edge probability rolls off.
            dilate3d(constraint, nx, ny, nz, 3, a.omp_threads);
            printf("[v9-2] U-Net hardset @ thr=%.2f kept=%lld voxels; constraint dilated by 3\n",
                   a.onnx_thr, kept);
            cand = cand2.data();
        }

        if (a.vessel && (a.pyramid || a.v9_2)) {
            // V3 methods 1+2+3: tubular gate + wavefront with depth-relaxed
            // threshold + topological rollback, bounded by the trunk envelope.
            // - V7/V8: tubular score = GPU Frangi vesselness
            // - V9-*:  tubular score = U-Net airway probability
            std::vector<float> vness;
            if (a.v9_1 || a.v9_2) {
                printf("[v9] using U-Net probability as the wavefront tubular gate\n");
                vness = std::move(unet_prob);  // no further use of unet_prob
            } else {
                vness.assign((size_t)N, 0.f);
                float scales[3] = {1.0f, 2.0f, 3.0f};
                float vms = a.v8
                    ? cuda_vesselness_v8(vol.data, vness.data(), nx, ny, nz, scales, 3, device)
                    : cuda_vesselness(vol.data, vness.data(), nx, ny, nz, scales, 3, device);
                if (vms < 0) MPI_Abort(MPI_COMM_WORLD, 4);
                printf("[vessel] Frangi vesselness: %.1f ms (3 scales%s)\n",
                       vms, a.v8 ? ", recurrence" : "");
            }
            // Seeds = central voxels of the strict trunk candidate (definite lumen).
            // v5: robust seed search guarantees we don't return an empty mask
            // for patients whose central air is missing (case 82/124 in v3).
            if (a.v5)
                seeds = central_axis_seeds_robust(cand2.data(), nx, ny, nz, roi);
            else
                seeds = central_axis_seeds(cand2.data(), nx, ny, nz, roi);

            // v5: pre-compute the pyramid-only mask as a safety fallback. If the
            // vessel wavefront catastrophically explodes (e.g. walks into the
            // pulmonary vasculature), we revert to this conservative result
            // instead of returning a leaked mask.
            RegionGrowResult rg_pyramid;
            if (a.v5) {
                std::vector<unsigned char> pinterior = cand2;
                std::vector<Seed> pseeds = central_axis_seeds_robust(
                        pinterior.data(), nx, ny, nz, roi);
                rg_pyramid = a.v8
                    ? multiseed_region_grow_v8(pinterior.data(), nx, ny, nz,
                                                pseeds, a.omp_threads)
                    : multiseed_region_grow(pinterior.data(), nx, ny, nz,
                                             pseeds, a.omp_threads);
                printf("[v%s] pyramid-only fallback mask: %lld voxels\n",
                       a.v8 ? "8" : "5", rg_pyramid.airway_voxels);
            }

            WavefrontParams wp;
            wp.thr_periph = a.thr_periph;
            wp.vmin       = a.vmin;
            wp.max_level  = a.wf_maxlevel;
            if (a.v5 && !a.v7) {
                // v5: in-flight leak guards. These caught case 120 in-flight
                // (no post-grow revert) and shrank case 49/74 leaks enough that
                // the result stays in the airway. Looser values (factor 2.5,
                // ratio 2.0) lost case 120 back to v3 levels, so we keep these.
                wp.hu_abs_max = a.hu_abs_max;      // -750 HU: hard air ceiling
                if (a.vmin == 0.03f) wp.vmin = 0.06f;
                wp.leak_factor = 2.0f;
                wp.leak_base   = 400;
                wp.leak_ratio  = 1.5f;
            }
            // v7 path: keep v3 defaults (vmin 0.03, leak_factor 3.0, ratio 3.0)
            // -- the post-grow safety net (rule a + rule b below) is what
            // catches category-C leaks, not the in-flight guards.
            rg = a.v8
                ? wavefront_region_grow_v8(vol.data, vness.data(), cand2.data(),
                                           constraint.empty() ? nullptr : constraint.data(),
                                           nx, ny, nz, seeds, wp, a.omp_threads)
                : wavefront_region_grow(vol.data, vness.data(), cand2.data(),
                                        constraint.empty() ? nullptr : constraint.data(),
                                        nx, ny, nz, seeds, wp, a.omp_threads);

            // v5/v7: post-grow safety net. Revert to the pyramid mask when the
            // wavefront result is a runaway leak. The pyramid floor (>= 80k
            // voxels) is always required so undersegmented pyramids cannot
            // trigger a false-positive revert. Two leak signatures fire:
            //   Rule (a): vessel > safety_ratio (2.5x) * pyramid
            //             Catches catastrophic blow-ups (cases 54, 63, 96)
            //   Rule (b, v7 only): vessel > safety_ratio2 (1.5x) * pyramid
            //                      AND vessel > safety_abs (600k voxels)
            //             Catches "slow" leaks too small to trigger rule (a)
            //             but anatomically implausible in absolute size.
            //             Specifically saves case 120 (1.91x, 894k voxels).
            const long long SAFETY_PYRAMID_FLOOR = 80000;
            bool rule_a = rg_pyramid.airway_voxels >= SAFETY_PYRAMID_FLOOR &&
                          rg.airway_voxels > (long long)(a.safety_ratio *
                                              (double)rg_pyramid.airway_voxels);
            bool rule_b = a.v7 &&
                          rg_pyramid.airway_voxels >= SAFETY_PYRAMID_FLOOR &&
                          rg.airway_voxels > a.safety_abs &&
                          rg.airway_voxels > (long long)(a.safety_ratio2 *
                                              (double)rg_pyramid.airway_voxels);
            if (a.v5 && (rule_a || rule_b)) {
                printf("[v%d-safety] vessel=%lld pyramid=%lld (rule_a=%d rule_b=%d) -> revert to pyramid\n",
                       a.v7 ? 7 : 5, rg.airway_voxels, rg_pyramid.airway_voxels,
                       (int)rule_a, (int)rule_b);
                rg = std::move(rg_pyramid);
            }
        } else if (a.pyramid) {
            // The dilated trunk already isolates the airway; region growing only
            // follows connectivity from central seeds, so the full-res exterior
            // flood-fill is redundant. Use the constrained candidate directly.
            interior = std::move(cand2);
            if (a.v5)
                seeds = central_axis_seeds_robust(interior.data(), nx, ny, nz, roi);
            else
                seeds = central_axis_seeds(interior.data(), nx, ny, nz, roi);
            rg = a.v8
                ? multiseed_region_grow_v8(interior.data(), nx, ny, nz, seeds, a.omp_threads)
                : multiseed_region_grow(interior.data(), nx, ny, nz, seeds, a.omp_threads);
        } else {
            if (a.use_roi)
                seeds = detect_seeds_parallel(cand, nx, ny, nz, roi, a.omp_threads, interior);
            else
                seeds = detect_seeds(cand, nx, ny, nz, interior);
            rg = a.v8
                ? multiseed_region_grow_v8(interior.data(), nx, ny, nz, seeds, a.omp_threads)
                : multiseed_region_grow(interior.data(), nx, ny, nz, seeds, a.omp_threads);
        }
        // V6 RESCUE: HU-drift cases (case 82-class) leave the pipeline with an
        // empty mask because every upstream stage trusts the -920 HU air
        // threshold that no longer fires in the lumen. The rescue is
        // *independent* of that threshold: it sweeps HU on the upper-central
        // body window, finds the trachea by geometry (Z-elongated, compact),
        // and re-grows from the located component using a relaxed threshold.
        // It only runs when the pipeline already failed, so healthy cases pay
        // nothing.
        if (a.v6 && rg.airway_voxels < a.rescue_min) {
            long long before = rg.airway_voxels;
            BBox rb = roi;
            if (!rb.valid) {
                // The body bbox can over-include when HU is drifted; recompute
                // with a looser tissue threshold so we still get a window even
                // when -300 HU labels almost the whole volume as body.
                rb = compute_body_bbox(vol.data, nx, ny, nz, 200.f,
                                       a.roi_margin, a.omp_threads);
                if (!rb.valid) {
                    rb.valid = true; rb.x0 = 0; rb.y0 = 0; rb.z0 = 0;
                    rb.x1 = nx - 1; rb.y1 = ny - 1; rb.z1 = nz - 1;
                }
            }
            printf("[v6] pipeline returned %lld voxels (< %lld) -> rescue\n",
                   before, a.rescue_min);
            TracheaCandidate rc = rescue_threshold_sweep(vol.data, nx, ny, nz,
                                                         rb, a.omp_threads);
            if (rc.valid) {
                // Build a CPU candidate at (rescued thr + slack). Slack lets
                // wall partial-volume voxels (slightly above lumen HU) be
                // included so growth doesn't stall at the carina.
                long long N2 = (long long)nx * ny * nz;
                std::vector<unsigned char> rcand((size_t)N2);
                float rthr = rc.threshold + a.rescue_slack;
                #pragma omp parallel for schedule(static)
                for (long long i = 0; i < N2; i++)
                    rcand[i] = (vol.data[i] <= rthr) ? 1 : 0;

                // Detect exterior air from the lateral edges of the body bbox
                // -- removes the background and isolates the airway lumen.
                std::vector<unsigned char> rinterior;
                std::vector<Seed> auto_seeds = detect_seeds_parallel(
                        rcand.data(), nx, ny, nz, rb, a.omp_threads, rinterior);

                std::vector<Seed> all_seeds = rc.seeds;
                for (const Seed& s : auto_seeds) all_seeds.push_back(s);

                RegionGrowResult rg2 = a.v8
                    ? multiseed_region_grow_v8(rinterior.data(), nx, ny, nz, all_seeds, a.omp_threads)
                    : multiseed_region_grow(rinterior.data(), nx, ny, nz, all_seeds, a.omp_threads);
                printf("[v6] rescue grew %lld voxels at thr=%.0f HU (was %lld)\n",
                       rg2.airway_voxels, rthr, before);

                // Post-grow keepwall: accept rescue only when BOTH true:
                //   (i) rescue grew at least as many voxels as baseline (the
                //       rescue interior can be empty when seeds land just
                //       outside the lumen -- accepting that would replace a
                //       small-but-real mask with 0; previously broke cases
                //       61/89/93/144 in the old v6 binary)
                //   (ii) rescue is <= rescue_leak_cap (leak guard, see below)
                // The geometry filter can mis-identify a tubular component as
                // the trachea (e.g. a large gas-filled stomach pocket) and the
                // subsequent grow leaks into surrounding lung. If the rescue
                // is *vastly* larger than typical airway trees, drop back.
                if (rg2.airway_voxels > a.rescue_leak_cap) {
                    printf("[v6] rescue grow %lld > leak_cap %lld -> keep baseline\n",
                           rg2.airway_voxels, a.rescue_leak_cap);
                } else if (rg2.airway_voxels <= before) {
                    printf("[v6] rescue grow %lld <= baseline %lld -> keep baseline\n",
                           rg2.airway_voxels, before);
                } else {
                    rg = std::move(rg2);
                    a.v11_rescue_fired = true;   // mark for V11 leak rule
                }
            } else {
                printf("[v6] rescue could not locate a trachea-shaped component; "
                       "keeping original mask (%lld voxels)\n", before);
            }
        }

        // V10: U-Net-anchored secondary safety oracle.
        // Why this is needed: V7/V8's dual-rule safety net compares the
        // wavefront result against the *pyramid* mask. When the pyramid
        // *also* leaks (case 124: pyramid envelope swallows the entire
        // lung, rescue then grows 494k voxels for a 135k-voxel airway),
        // both arms of the dual-rule look "consistent" and no revert
        // fires. V10 adds a third comparator: the U-Net-anchored mask.
        //
        // Crucially, we don't compare to the *raw* (prob > thr) voxel count
        // — that includes vessels and other scattered false-positive CCs
        // the network lights up (case 124 raw: 208k vox; central
        // connected component: 44k). We compute the central-seeded grow
        // of the U-Net hardset and compare against that.
        //
        // Trigger (both required):
        //   (i)  central-grown U-Net mask has >= v10_unet_floor voxels
        //        (rejects degenerate near-empty network outputs)
        //   (ii) final rg has > v10_leak_ratio * U-Net-anchored mask
        //        (catches the >=4x oversegmentations)
        // On a fire, replace rg with the U-Net-anchored grow itself.
        if (a.v10 && unet_hardset_voxels >= a.v10_unet_floor) {
            std::vector<Seed> v10_seeds = central_axis_seeds_robust(
                    unet_hardset_v10.data(), nx, ny, nz, roi);
            RegionGrowResult rg_unet;
            if (v10_seeds.empty()) {
                // No central seed lands inside the U-Net hardset; fall back
                // to the raw hardset count (largest CC ≈ the hardset itself
                // when seeds can't anchor it).
                rg_unet.mask = unet_hardset_v10;
                rg_unet.airway_voxels = unet_hardset_voxels;
                rg_unet.interior_voxels = unet_hardset_voxels;
                rg_unet.num_seeds = 0;
            } else {
                rg_unet = multiseed_region_grow_v8(
                        unet_hardset_v10.data(), nx, ny, nz,
                        v10_seeds, a.omp_threads);
            }
            printf("[v10] U-Net-anchored grow: seeds=%d airway=%lld "
                   "(rg=%lld ratio=%.2fx)\n",
                   rg_unet.num_seeds, rg_unet.airway_voxels,
                   rg.airway_voxels,
                   rg_unet.airway_voxels > 0
                       ? (double)rg.airway_voxels / rg_unet.airway_voxels
                       : 0.0);
            // V10 single-feature rule: rg > 10x U-Net mask.
            bool fire_v10 = (rg_unet.airway_voxels >= a.v10_unet_floor &&
                rg.airway_voxels > (long long)(a.v10_leak_ratio *
                                  (double)rg_unet.airway_voxels));
            // V11 additional feature: when rescue fired in this case
            // (V6 rescue grew the mask from a tiny baseline), be much
            // more suspicious of the result -- a 4x oversegmentation is
            // already a strong leak signal because rescue's relaxed HU
            // ceiling routinely walks into the lung.
            bool fire_v11_rescue = (a.v11 && a.v11_rescue_fired &&
                rg_unet.airway_voxels >= a.v10_unet_floor &&
                rg.airway_voxels > (long long)(a.v11_rescue_leak_ratio *
                                  (double)rg_unet.airway_voxels));
            if (fire_v10 || fire_v11_rescue) {
                printf("[v10/v11] LEAK detected (rg=%lld U-Net=%lld ratio=%.2fx, "
                       "rescue=%d) -> swap\n",
                       rg.airway_voxels, rg_unet.airway_voxels,
                       (double)rg.airway_voxels / std::max<long long>(1, rg_unet.airway_voxels),
                       (int)a.v11_rescue_fired);
                rg = std::move(rg_unet);
            }
        }

        }  // end of: if (!a.v14)

        t_rg = MPI_Wtime() - t_rg0;
        printf("[regiongrow] seeds=%d  interior_air=%lld  airway=%lld voxels (omp_threads=%d)\n",
               rg.num_seeds, rg.interior_voxels, rg.airway_voxels, a.omp_threads);
    }

    // ---- Step 5b: write output + Dice (rank 0) ----
    double t_write = 0.0, t_eval = 0.0;
    if (rank == 0) {
        double t_write0 = MPI_Wtime();
        if (!write_mask_nifti(a.output, vol, rg.mask.data()))
            fprintf(stderr, "[warn] failed to write %s\n", a.output.c_str());
        else
            printf("[write] %s\n", a.output.c_str());
        t_write = MPI_Wtime() - t_write0;

        if (!a.gt.empty()) {
            double t_eval0 = MPI_Wtime();
            int gx, gy, gz;
            unsigned char* gt = load_label_u8(a.gt, gx, gy, gz);
            if (gt && gx == nx && gy == ny && gz == nz) {
                long long inter = 0, sp = 0, sg = 0;
                long long N = (long long)nx * ny * nz;
                // V8: parallel reduction. Each thread keeps a private partial
                // count; final sum is computed in O(threads) at the barrier.
                // Embarrassingly parallel, cuts ~150 ms on a 512³ volume.
                #pragma omp parallel for schedule(static) reduction(+:inter,sp,sg)
                for (long long i = 0; i < N; i++) {
                    int p = rg.mask[i] ? 1 : 0, g = gt[i] ? 1 : 0;
                    inter += p & g; sp += p; sg += g;
                }
                double dice = (sp + sg) ? (2.0 * inter) / (double)(sp + sg) : 1.0;
                printf("[dice] pred=%lld gt=%lld inter=%lld  Dice=%.4f\n", sp, sg, inter, dice);
            } else {
                fprintf(stderr, "[warn] gt load/dim mismatch, skipping Dice\n");
            }
            delete[] gt;
            t_eval = MPI_Wtime() - t_eval0;
        }
    }

    double t_total = MPI_Wtime() - t_total0;
    double t_other = t_total - (t_load + t_mpi + t_gpu + t_gather + t_onnx + t_rg + t_write + t_eval);
    if (rank == 0) {
        printf("\n=== Timing (s) | nprocs=%d omp=%d ndev=%d ===\n", nprocs, a.omp_threads, ndev);
        printf("  load        : %8.3f\n", t_load);
        printf("  mpi part    : %8.3f\n", t_mpi);
        printf("  gpu filter  : %8.3f  (max kernel %.2f ms)\n", t_gpu, kms_max);
        printf("  gather      : %8.3f\n", t_gather);
        if (a.v9_1 || a.v9_2)
            printf("  onnx infer  : %8.3f\n", t_onnx);
        printf("  region grow : %8.3f\n", t_rg);
        printf("  write       : %8.3f\n", t_write);
        printf("  eval (dice) : %8.3f\n", t_eval);
        printf("  other/sync  : %8.3f\n", t_other);
        printf("  TOTAL       : %8.3f\n", t_total);
    }

    MPI_Finalize();
    return 0;
}
