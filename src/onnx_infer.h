#pragma once
#include <string>

// Thin wrapper around ONNX Runtime for the AIIB23 DynUNet airway model.
// The model takes a 5D tensor [N, 1, D, H, W] of normalized CT (HU mapped
// to [0,1] via `hu_norm`) and returns logits [N, 2, D, H, W]. We softmax on
// the channel axis and return channel-1 (airway) probability.
//
// V11: the implementation auto-detects the model's input dtype (FP32 or
// FP16), allocates a persistent IOBinding on the CUDA device, and runs
// patches in batches to amortise launch overhead. Same external API as
// V9, plus an optional batch_size argument.

struct OrtPipeline;  // opaque

// V12 execution-provider / pipelining options.
//   use_trt        : register the TensorRT EP (kernel fusion at the fixed
//                    128^3 patch) ahead of the CUDA EP. The CUDA EP stays
//                    registered as a fallback for any node TRT can't fuse.
//   trt_fp16       : enable TensorRT internal FP16 (TensorCore math on V100).
//   trt_engine_cache / trt_cache_path : serialise built engines to disk so
//                    only the first patch of the first case pays the build.
//   stream_overlap : double-buffered, two-CUDA-stream H2D/compute pipeline so
//                    patch N+1's PCIe upload overlaps patch N's GPU compute.
struct OrtOptions {
    bool use_trt          = false;
    bool trt_fp16         = true;
    bool trt_engine_cache = true;
    std::string trt_cache_path = "model/trt_cache";
    bool stream_overlap   = false;
};

// Construct a session. `gpu_device >= 0` enables the GPU EP(s) on that device;
// `gpu_device < 0` forces CPU. Returns nullptr on failure.
OrtPipeline* ort_create(const std::string& model_path, int gpu_device,
                        const OrtOptions& opt = OrtOptions());

// Destroy the session and free all GPU resources.
void ort_destroy(OrtPipeline*);

// Run the model over the entire volume by sliding non-overlapping (or
// optionally overlapping) patches of `patch{D,H,W}`. The output `prob`
// receives the airway probability (softmax channel 1) at every voxel.
//
//   hu, prob: nx*ny*nz, layout idx = z*ny*nx + y*nx + x.
//   stride{D,H,W}: typically equal to patch size for non-overlap. Overlap
//     regions are averaged.
//   hu_lo, hu_hi: HU clip range used for [0,1] normalization (matches the
//     model's training preprocessing).
//   batch_size: how many patches to send to the GPU in a single Run.
//     >=2 amortises kernel-launch overhead and improves SM occupancy.
//     V100 32GB fits batch=8 at patch=128 with FP32 weights comfortably.
//
// Returns: wallclock ms of the inference loop (>=0) on success, -1.0f on
// failure (stderr-printed).
float ort_infer_volume(OrtPipeline* p,
                       const float* hu, float* prob,
                       int nx, int ny, int nz,
                       int patch_d, int patch_h, int patch_w,
                       int stride_d, int stride_h, int stride_w,
                       float hu_lo, float hu_hi,
                       int batch_size = 1);
