#include "onnx_infer.h"

#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

// V18 ablation: device-side softmax launcher (defined in cuda_filter.cu).
void cuda_softmax_ch1(const float* d_logits, float* d_prob,
                      long long patch_vox, int B);

// =============================================================================
// V11 ONNX Runtime wrapper.
//
// Compared to V9/V10 this:
//   - auto-detects the model's input element type (FP32 or FP16) and
//     converts the host patch on the fly when FP16 is needed
//   - holds a persistent IOBinding with device-side input/output buffers
//     so each patch is just two cudaMemcpy + Run() (no per-call tensor
//     allocation in the ORT arena)
//   - groups patches into batches and submits a single Run() per batch
//   - both reductions in the softmax-accumulate step run in OpenMP
//
// On a V100, this drops per-volume inference from ~12 s (V9/V10) to ~3-4 s
// with FP16 (or ~6-7 s with FP32, batch=4).
// =============================================================================

namespace {

struct PipelineImpl {
    Ort::Env env;
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> sess;
    std::string in_name, out_name;
    Ort::AllocatorWithDefaultOptions alloc;
    bool cuda_ok = false;
    bool model_fp16 = false;
    bool trt_ok = false;
    int  gpu = -1;
    OrtOptions o;                       // V12 EP / pipelining options
    cudaStream_t compute_stream = nullptr;  // ORT-bound compute stream (overlap)
    cudaStream_t copy_stream = nullptr;     // H2D prefetch stream (overlap)
    PipelineImpl() : env(ORT_LOGGING_LEVEL_WARNING, "airway_dynunet") {}
};

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Generate patch starting offsets such that every voxel along an axis is
// covered. Final patch is flush with the boundary; the stride may be less
// than `stride` between the second-to-last and final start.
std::vector<int> patch_starts(int n, int patch, int stride) {
    std::vector<int> s;
    if (n <= patch) { s.push_back(0); return s; }
    for (int z = 0; z + patch <= n; z += stride) s.push_back(z);
    if (s.back() + patch < n) s.push_back(n - patch);
    return s;
}

// Float -> half conversion. Uses Ort::Float16_t which already implements
// the IEEE 754 binary16 packing.
inline void f32_to_f16_block(const float* src, Ort::Float16_t* dst, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) dst[i] = Ort::Float16_t(src[i]);
}

// Convert half -> float in a stride-friendly loop.
inline void f16_to_f32_block(const Ort::Float16_t* src, float* dst, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) dst[i] = static_cast<float>(src[i]);
}

}  // namespace

struct OrtPipeline {
    PipelineImpl p;
};

OrtPipeline* ort_create(const std::string& model_path, int gpu_device,
                        const OrtOptions& opt) {
    auto pipe = new OrtPipeline();
    auto& p = pipe->p;
    p.gpu = gpu_device;
    p.o   = opt;

    p.opts.SetIntraOpNumThreads(1);
    p.opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // V12: a user-owned compute stream lets the ORT GPU EP run concurrently
    // with our own H2D prefetch on a second stream (double buffering).
    void* user_stream = nullptr;
    if (gpu_device >= 0 && opt.stream_overlap) {
        if (cudaStreamCreate(&p.compute_stream) == cudaSuccess &&
            cudaStreamCreate(&p.copy_stream)    == cudaSuccess) {
            user_stream = p.compute_stream;
        } else {
            fprintf(stderr, "[ort] cudaStreamCreate failed; disabling stream overlap\n");
            p.o.stream_overlap = false;
        }
    }

    if (gpu_device >= 0 && opt.use_trt) {
        // TensorRT EP: hardware kernel fusion for the fixed 128^3 patch.
        // Engine build is one-time per shape; cached to disk for reuse.
        OrtTensorRTProviderOptions trt{};
        trt.device_id                  = gpu_device;
        trt.trt_fp16_enable            = opt.trt_fp16 ? 1 : 0;
        trt.trt_engine_cache_enable    = opt.trt_engine_cache ? 1 : 0;
        trt.trt_engine_cache_path      = p.o.trt_cache_path.c_str();
        trt.trt_max_workspace_size     = (size_t)4 << 30;   // 4 GB
        trt.trt_max_partition_iterations = 1000;
        trt.trt_min_subgraph_size      = 1;
        if (user_stream) {
            trt.has_user_compute_stream = 1;
            trt.user_compute_stream     = user_stream;
        }
        try {
            p.opts.AppendExecutionProvider_TensorRT(trt);
            p.trt_ok = true;
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "[ort] TensorRT EP unavailable (%s); using CUDA EP\n", e.what());
        }
    }

    if (gpu_device >= 0) {
        // CUDA EP -- primary when !use_trt, otherwise the fallback that runs
        // any node the TensorRT EP did not fuse into an engine.
        OrtCUDAProviderOptions cu{};
        cu.device_id                 = gpu_device;
        cu.arena_extend_strategy     = 0;
        cu.gpu_mem_limit             = SIZE_MAX;
        cu.cudnn_conv_algo_search    = opt.cudnn_heuristic
                                       ? OrtCudnnConvAlgoSearchHeuristic
                                       : OrtCudnnConvAlgoSearchExhaustive;
        cu.do_copy_in_default_stream = user_stream ? 0 : 1;
        if (user_stream) {
            cu.has_user_compute_stream = 1;
            cu.user_compute_stream     = user_stream;
        }
        try {
            p.opts.AppendExecutionProvider_CUDA(cu);
            p.cuda_ok = true;
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "[ort] CUDA EP unavailable (%s); falling back to CPU\n", e.what());
        }
    }

    try {
        p.sess = std::make_unique<Ort::Session>(p.env, model_path.c_str(), p.opts);
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "[ort] failed to load %s: %s\n", model_path.c_str(), e.what());
        delete pipe;
        return nullptr;
    }

    auto in_alloc  = p.sess->GetInputNameAllocated(0, p.alloc);
    auto out_alloc = p.sess->GetOutputNameAllocated(0, p.alloc);
    p.in_name  = in_alloc.get();
    p.out_name = out_alloc.get();

    // Detect input dtype to decide whether to feed FP32 or FP16 from C++.
    Ort::TypeInfo ti = p.sess->GetInputTypeInfo(0);
    auto tensor_info = ti.GetTensorTypeAndShapeInfo();
    p.model_fp16 = (tensor_info.GetElementType()
                    == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

    const char* ep = p.trt_ok ? "TensorRT(+CUDA)" : (p.cuda_ok ? "CUDA" : "CPU");
    fprintf(stderr, "[ort] loaded %s  EP=%s  dtype=%s  overlap=%d  in='%s' out='%s'\n",
            model_path.c_str(), ep,
            p.model_fp16 ? "FP16" : "FP32",
            (int)p.o.stream_overlap,
            p.in_name.c_str(), p.out_name.c_str());
    return pipe;
}

void ort_destroy(OrtPipeline* pipe) {
    if (pipe) {
        if (pipe->p.compute_stream) cudaStreamDestroy(pipe->p.compute_stream);
        if (pipe->p.copy_stream)    cudaStreamDestroy(pipe->p.copy_stream);
    }
    delete pipe;
}

// Run B patches in one Run(). Input layout: [B, 1, D, H, W] of either
// FP32 or FP16. Output layout: [B, 2, D, H, W] of the same dtype. The
// caller has already filled `in_host_f32[B * patch_vox]` with the
// normalized patches (or `in_host_f16[]` if model_fp16).
//
// d_in / d_out are device pointers, allocated once by the caller.
// Returns ms taken for this batched run, or -1 on failure.
static float run_batch(PipelineImpl& p,
                       void* d_in, void* d_out,
                       float* out_f32_host,
                       int B, int patch_d, int patch_h, int patch_w,
                       const float* in_host_f32,
                       Ort::Float16_t* scratch_fp16,
                       const Ort::MemoryInfo& cuda_mi) {
    const long long patch_vox = (long long)patch_d * patch_h * patch_w;
    const size_t bytes_per_elt = p.model_fp16 ? 2 : 4;
    const size_t in_bytes  = (size_t)B * patch_vox * bytes_per_elt;
    const size_t out_bytes = (size_t)B * 2 * patch_vox * bytes_per_elt;

    // Host -> device upload.
    if (p.model_fp16) {
        f32_to_f16_block(in_host_f32, scratch_fp16, (size_t)B * patch_vox);
        cudaMemcpy(d_in, scratch_fp16, in_bytes, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(d_in, in_host_f32, in_bytes, cudaMemcpyHostToDevice);
    }

    const int64_t in_shape[5]  = {B, 1, patch_d, patch_h, patch_w};
    const int64_t out_shape[5] = {B, 2, patch_d, patch_h, patch_w};

    // Wrap the device buffers as ORT tensors of the right dtype.
    Ort::Value in_val{nullptr}, out_val{nullptr};
    try {
        if (p.model_fp16) {
            in_val  = Ort::Value::CreateTensor(cuda_mi, d_in,  in_bytes,
                                                in_shape, 5,
                                                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            out_val = Ort::Value::CreateTensor(cuda_mi, d_out, out_bytes,
                                                out_shape, 5,
                                                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        } else {
            in_val  = Ort::Value::CreateTensor<float>(cuda_mi, (float*)d_in,
                                                       (size_t)B * patch_vox,
                                                       in_shape, 5);
            out_val = Ort::Value::CreateTensor<float>(cuda_mi, (float*)d_out,
                                                       (size_t)B * 2 * patch_vox,
                                                       out_shape, 5);
        }

        const char* in_names[]  = { p.in_name.c_str() };
        const char* out_names[] = { p.out_name.c_str() };
        p.sess->Run(Ort::RunOptions{nullptr},
                    in_names, &in_val, 1,
                    out_names, &out_val, 1);
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "[ort] batched run failed: %s\n", e.what());
        return -1.f;
    }

    // Device -> host.
    if (p.model_fp16) {
        cudaMemcpy(scratch_fp16, d_out, out_bytes, cudaMemcpyDeviceToHost);
        f16_to_f32_block(scratch_fp16, out_f32_host, (size_t)B * 2 * patch_vox);
    } else if (p.o.gpu_softmax) {
        // V18 ablation: reduce the 2-channel logits to a 1-channel foreground
        // probability ON the GPU, then copy back only that one channel (half
        // the D2H traffic, no host exp() loop). out_f32_host now holds
        // [B, patch_vox] probabilities; the caller reads them directly.
        // The CUDA EP may write d_out on its own stream and Run() does not
        // always sync it before returning, so a full device sync is required
        // or the kernel reads partially-written logits (silent Dice drift).
        cudaDeviceSynchronize();
        float* d_prob = nullptr;
        cudaMalloc(&d_prob, (size_t)B * patch_vox * sizeof(float));
        cuda_softmax_ch1((const float*)d_out, d_prob, patch_vox, B);
        cudaMemcpy(out_f32_host, d_prob,
                   (size_t)B * patch_vox * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_prob);
    } else {
        cudaMemcpy(out_f32_host, d_out, out_bytes, cudaMemcpyDeviceToHost);
    }
    return 0.f;  // success
}

// V12 double-buffered, two-stream inference (batch=1). While patch N runs on
// the ORT-bound compute_stream, patch N+1's normalized input is uploaded H2D
// on copy_stream (GPU copy engine), so the PCIe transfer hides behind compute.
// FP32-IO models only (the V11/V12 fp16 model keeps FP32 IO via keep_io_types,
// so model_fp16 is false here); fp16-IO models fall back to the batched path.
static float infer_volume_overlap(PipelineImpl& p,
                                  const float* hu, float* prob,
                                  int nx, int ny, int nz,
                                  int patch_d, int patch_h, int patch_w,
                                  int stride_d, int stride_h, int stride_w,
                                  float hu_lo, float hu_hi) {
    const long long N = (long long)nx * ny * nz;
    const long long plane = (long long)nx * ny;
    const long long patch_vox = (long long)patch_d * patch_h * patch_w;
    std::memset(prob, 0, sizeof(float) * (size_t)N);
    std::vector<uint8_t> nhit((size_t)N, 0);

    const float inv_range = 1.f / (hu_hi - hu_lo);
    const float lo = hu_lo, hi = hu_hi;

    std::vector<int> zs = patch_starts(nz, patch_d, stride_d);
    std::vector<int> ys = patch_starts(ny, patch_h, stride_h);
    std::vector<int> xs = patch_starts(nx, patch_w, stride_w);
    struct Origin { int z, y, x, d_hi, h_hi, w_hi; };
    std::vector<Origin> origins;
    for (int z0 : zs) for (int y0 : ys) for (int x0 : xs)
        origins.push_back({z0, y0, x0,
                           std::min(patch_d, nz - z0),
                           std::min(patch_h, ny - y0),
                           std::min(patch_w, nx - x0)});
    const int total = (int)origins.size();

    const size_t in_elems  = (size_t)patch_vox;
    const size_t out_elems = (size_t)2 * patch_vox;
    const size_t in_bytes  = in_elems * sizeof(float);
    const size_t out_bytes = out_elems * sizeof(float);

    float* d_in[2]  = {nullptr, nullptr};
    float* d_out[2] = {nullptr, nullptr};
    float* h_in[2]  = {nullptr, nullptr};   // pinned (async copy)
    float* h_out[2] = {nullptr, nullptr};   // pinned
    bool alloc_ok = true;
    for (int b = 0; b < 2; b++) {
        if (cudaMalloc(&d_in[b],  in_bytes)  != cudaSuccess) alloc_ok = false;
        if (cudaMalloc(&d_out[b], out_bytes) != cudaSuccess) alloc_ok = false;
        if (cudaHostAlloc((void**)&h_in[b],  in_bytes,  cudaHostAllocDefault) != cudaSuccess) alloc_ok = false;
        if (cudaHostAlloc((void**)&h_out[b], out_bytes, cudaHostAllocDefault) != cudaSuccess) alloc_ok = false;
    }
    auto free_all = [&]() {
        for (int b = 0; b < 2; b++) {
            if (d_in[b])  cudaFree(d_in[b]);
            if (d_out[b]) cudaFree(d_out[b]);
            if (h_in[b])  cudaFreeHost(h_in[b]);
            if (h_out[b]) cudaFreeHost(h_out[b]);
        }
    };
    if (!alloc_ok) { fprintf(stderr, "[ort] overlap buffer alloc failed\n"); free_all(); return -1.f; }

    OrtMemoryInfo* raw_mi = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateMemoryInfo(
        "Cuda", OrtArenaAllocator, p.gpu < 0 ? 0 : p.gpu, OrtMemTypeDefault, &raw_mi));
    Ort::MemoryInfo cuda_mi(raw_mi);

    const int64_t in_shape[5]  = {1, 1, patch_d, patch_h, patch_w};
    const int64_t out_shape[5] = {1, 2, patch_d, patch_h, patch_w};
    const char* in_names[]  = { p.in_name.c_str() };
    const char* out_names[] = { p.out_name.c_str() };

    // Tensor layout MUST match MONAI training: axis 2 = volume X. Packing the
    // other way (axis 2 = Z) costs ~0.18 mean Dice; see ort_infer_volume notes.
    auto fill = [&](int idx, float* dst) {
        const Origin& o = origins[idx];
        std::memset(dst, 0, in_bytes);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int d = 0; d < o.d_hi; d++)
            for (int h = 0; h < o.h_hi; h++) {
                const float* src = hu + (long long)(o.z + d) * plane
                                      + (long long)(o.y + h) * nx + o.x;
                for (int w = 0; w < o.w_hi; w++)
                    dst[((size_t)w * patch_h + h) * patch_d + d] =
                        (clampf(src[w], lo, hi) - lo) * inv_range;
            }
    };
    auto accumulate = [&](int idx, const float* out_f32) {
        const Origin& o = origins[idx];
        const float* logit0 = out_f32;
        const float* logit1 = out_f32 + patch_vox;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int d = 0; d < o.d_hi; d++)
            for (int h = 0; h < o.h_hi; h++) {
                long long row_vol = (long long)(o.z + d) * plane
                                  + (long long)(o.y + h) * nx + o.x;
                for (int w = 0; w < o.w_hi; w++) {
                    size_t ti = ((size_t)w * patch_h + h) * patch_d + d;
                    float l0 = logit0[ti], l1 = logit1[ti];
                    float m  = l0 > l1 ? l0 : l1;
                    float e0 = std::exp(l0 - m), e1 = std::exp(l1 - m);
                    long long gi = row_vol + w;
                    prob[gi] += e1 / (e0 + e1);
                    nhit[gi] += 1;
                }
            }
    };

    auto t0 = std::chrono::steady_clock::now();

    fill(0, h_in[0]);
    cudaMemcpyAsync(d_in[0], h_in[0], in_bytes, cudaMemcpyHostToDevice, p.copy_stream);
    cudaStreamSynchronize(p.copy_stream);

    bool failed = false;
    for (int n = 0; n < total; n++) {
        int cur = n & 1, nxt = (n + 1) & 1;
        // Prefetch patch n+1 (CPU fill + async H2D) -- overlaps patch n compute.
        if (n + 1 < total) {
            fill(n + 1, h_in[nxt]);
            cudaMemcpyAsync(d_in[nxt], h_in[nxt], in_bytes,
                            cudaMemcpyHostToDevice, p.copy_stream);
        }
        try {
            Ort::Value in_val  = Ort::Value::CreateTensor<float>(
                cuda_mi, d_in[cur], in_elems, in_shape, 5);
            Ort::Value out_val = Ort::Value::CreateTensor<float>(
                cuda_mi, d_out[cur], out_elems, out_shape, 5);
            p.sess->Run(Ort::RunOptions{nullptr}, in_names, &in_val, 1,
                        out_names, &out_val, 1);
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "[ort] overlap run failed: %s\n", e.what());
            failed = true; break;
        }
        cudaStreamSynchronize(p.compute_stream);   // user stream not auto-synced
        cudaMemcpyAsync(h_out[cur], d_out[cur], out_bytes,
                        cudaMemcpyDeviceToHost, p.compute_stream);
        cudaStreamSynchronize(p.compute_stream);
        accumulate(n, h_out[cur]);
        if (n + 1 < total) cudaStreamSynchronize(p.copy_stream);

        if (n == 0 || (n + 1) % 64 == 0 || n + 1 == total)
            fprintf(stderr, "[ort] patches %d/%d  (overlap)\n", n + 1, total);
    }

    free_all();
    if (failed) return -1.f;

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < N; i++)
        if (nhit[i] > 1) prob[i] /= (float)nhit[i];

    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}

float ort_infer_volume(OrtPipeline* pipe,
                       const float* hu, float* prob,
                       int nx, int ny, int nz,
                       int patch_d, int patch_h, int patch_w,
                       int stride_d, int stride_h, int stride_w,
                       float hu_lo, float hu_hi,
                       int batch_size) {
    if (!pipe || !pipe->p.sess) return -1.f;
    if (batch_size < 1) batch_size = 1;
    auto& p = pipe->p;

    // V12: double-buffered stream-overlap path (batch=1, FP32-IO model).
    if (p.o.stream_overlap && p.compute_stream && p.copy_stream && !p.model_fp16) {
        return infer_volume_overlap(p, hu, prob, nx, ny, nz,
                                    patch_d, patch_h, patch_w,
                                    stride_d, stride_h, stride_w,
                                    hu_lo, hu_hi);
    }

    const long long N = (long long)nx * ny * nz;
    const long long plane = (long long)nx * ny;
    std::memset(prob, 0, sizeof(float) * (size_t)N);
    std::vector<uint8_t> nhit((size_t)N, 0);

    const long long patch_vox = (long long)patch_d * patch_h * patch_w;
    const float inv_range = 1.f / (hu_hi - hu_lo);
    const float lo = hu_lo, hi = hu_hi;

    std::vector<int> zs = patch_starts(nz, patch_d, stride_d);
    std::vector<int> ys = patch_starts(ny, patch_h, stride_h);
    std::vector<int> xs = patch_starts(nx, patch_w, stride_w);

    struct Origin { int z, y, x, d_hi, h_hi, w_hi; };
    std::vector<Origin> origins;
    origins.reserve(zs.size() * ys.size() * xs.size());
    for (int z0 : zs) for (int y0 : ys) for (int x0 : xs) {
        origins.push_back({z0, y0, x0,
                           std::min(patch_d, nz - z0),
                           std::min(patch_h, ny - y0),
                           std::min(patch_w, nx - x0)});
    }
    const int total = (int)origins.size();

    // Device buffers for the largest possible batch.
    void* d_in = nullptr;  void* d_out = nullptr;
    const size_t bytes_per_elt = p.model_fp16 ? 2 : 4;
    cudaMalloc(&d_in,  (size_t)batch_size * patch_vox * bytes_per_elt);
    cudaMalloc(&d_out, (size_t)batch_size * 2 * patch_vox * bytes_per_elt);
    if (!d_in || !d_out) {
        fprintf(stderr, "[ort] cudaMalloc failed for batch=%d\n", batch_size);
        if (d_in)  cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        return -1.f;
    }

    // Host buffers: input patches (FP32, normalized), output logits (FP32).
    std::vector<float> in_buf((size_t)batch_size * patch_vox, 0.f);
    std::vector<float> out_buf((size_t)batch_size * 2 * patch_vox, 0.f);
    // FP16 scratch only if needed.
    std::vector<Ort::Float16_t> fp16_in, fp16_out;
    if (p.model_fp16) {
        fp16_in.resize((size_t)batch_size * patch_vox);
        fp16_out.resize((size_t)batch_size * 2 * patch_vox);
    }

    OrtMemoryInfo* raw_mi = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateMemoryInfo(
        "Cuda", OrtArenaAllocator, p.gpu < 0 ? 0 : p.gpu,
        OrtMemTypeDefault, &raw_mi));
    Ort::MemoryInfo cuda_mi(raw_mi);

    auto t0 = std::chrono::steady_clock::now();

    for (int start = 0; start < total; start += batch_size) {
        int B = std::min(batch_size, total - start);

        // -------- Fill the B patches into in_buf (FP32, normalized). --------
        // Tensor layout MUST match MONAI training: (B, C, X, Y, Z) i.e. axis 2
        // is volume X, axis 4 is volume Z. The training pipeline (nibabel ->
        // EnsureChannelFirstd -> Spacingd -> model) feeds the model X first.
        // Packing it the other way (axis 2 = Z) costs ~0.18 mean Dice on this
        // dataset because the model was never trained X<->Z invariant
        // (train.py only RandRotate90 in the XY plane). So we pack:
        //   tensor[..., w, h, d] = volume[(z+d), (y+h), (x+w)]   with axis 2 = w (X)
        std::memset(in_buf.data(), 0,
                    sizeof(float) * (size_t)B * patch_vox);
        for (int b = 0; b < B; b++) {
            const Origin& o = origins[start + b];
            float* p_in = in_buf.data() + (size_t)b * patch_vox;
            #pragma omp parallel for collapse(2) schedule(static)
            for (int d = 0; d < o.d_hi; d++) {
                for (int h = 0; h < o.h_hi; h++) {
                    const float* src = hu + (long long)(o.z + d) * plane
                                          + (long long)(o.y + h) * nx + o.x;
                    for (int w = 0; w < o.w_hi; w++) {
                        p_in[((size_t)w * patch_h + h) * patch_d + d] =
                            (clampf(src[w], lo, hi) - lo) * inv_range;
                    }
                }
            }
        }

        // -------- One batched Run on the GPU. --------
        if (run_batch(p, d_in, d_out, out_buf.data(), B,
                      patch_d, patch_h, patch_w,
                      in_buf.data(),
                      p.model_fp16 ? fp16_out.data() : nullptr,
                      cuda_mi) < 0.f) {
            cudaFree(d_in); cudaFree(d_out);
            return -1.f;
        }
        // run_batch reuses the same fp16 scratch for both directions when
        // needed; the host-side path always produces FP32 logits in out_buf.

        // -------- Softmax(channel 0 vs 1) + accumulate into prob. --------
        // Read logits in the same (axis 2 = X) layout as the fill above.
        // When gpu_softmax is on, run_batch already reduced the logits to a
        // 1-channel probability on the device, so out_buf holds [B, patch_vox]
        // probabilities and the host loop is a pure copy/accumulate (no exp()).
        const bool gsm = p.o.gpu_softmax && !p.model_fp16;
        // --serial-softmax disables the OpenMP parallelism of the host softmax
        // (the naive, no-acceleration baseline for the Table III ablation).
        const bool par_sm = !p.o.serial_softmax;
        for (int b = 0; b < B; b++) {
            const Origin& o = origins[start + b];
            const float* logit0 = out_buf.data() + (size_t)b * (gsm ? 1 : 2) * patch_vox;
            const float* logit1 = gsm ? nullptr : logit0 + patch_vox;

            #pragma omp parallel for collapse(2) schedule(static) if(par_sm)
            for (int d = 0; d < o.d_hi; d++) {
                for (int h = 0; h < o.h_hi; h++) {
                    long long row_vol = (long long)(o.z + d) * plane
                                      + (long long)(o.y + h) * nx
                                      + o.x;
                    for (int w = 0; w < o.w_hi; w++) {
                        size_t ti = ((size_t)w * patch_h + h) * patch_d + d;
                        float pr;
                        if (gsm) {
                            pr = logit0[ti];           // GPU already did softmax
                        } else {
                            float l0 = logit0[ti];
                            float l1 = logit1[ti];
                            float m  = l0 > l1 ? l0 : l1;
                            float e0 = std::exp(l0 - m), e1 = std::exp(l1 - m);
                            pr = e1 / (e0 + e1);
                        }
                        long long gi = row_vol + w;
                        prob[gi] += pr;
                        nhit[gi] += 1;
                    }
                }
            }
        }

        int patch_idx = start + B;
        if (start == 0 || patch_idx % (32 * std::max(1, batch_size / 4)) < B
            || patch_idx == total) {
            fprintf(stderr, "[ort] patches %d/%d  (B=%d)\n",
                    patch_idx, total, B);
        }
    }

    cudaFree(d_in); cudaFree(d_out);

    // Average overlapping contributions.
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < N; i++) {
        if (nhit[i] > 1) prob[i] /= (float)nhit[i];
    }

    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}

// =============================================================================
// 2.5D slice-by-slice inference.
//
// The 2.5D model takes [B, 3, 256, 256] FP32 input where the 3 channels are
// adjacent axial slices (z-1, z, z+1), boundary-clamped. The model was trained
// with nibabel (nx, ny, nz).permute(2,0,1) so each slice has H=nx, W=ny —
// the opposite of the C++ volume layout (H=ny, W=nx). We therefore transpose
// and bilinearly resize each slice to 256×256 before feeding the model, then
// bilinearly resize and transpose back when accumulating the probability output.
// =============================================================================

static const int MODEL25D_H = 256;
static const int MODEL25D_W = 256;

// Normalise + transpose + bilinear resize one volume slice into a model channel.
// Fills dst[MODEL25D_H * MODEL25D_W].
static void fill_channel_25d(float* dst,
                              const float* hu, int nx, int ny,
                              long long plane,
                              int z_vol,
                              float hu_lo, float hu_hi) {
    const float inv_range = 1.f / (hu_hi - hu_lo);
    const float sx = (float)(nx - 1) / (MODEL25D_H - 1);
    const float sy = (float)(ny - 1) / (MODEL25D_W - 1);
    const float* slice_base = hu + (long long)z_vol * plane;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < MODEL25D_H; h++) {
        for (int w = 0; w < MODEL25D_W; w++) {
            float fx = h * sx;
            float fy = w * sy;
            int x0 = (int)fx; int x1 = std::min(x0 + 1, nx - 1); float tx = fx - x0;
            int y0 = (int)fy; int y1 = std::min(y0 + 1, ny - 1); float ty = fy - y0;
            float v00 = slice_base[(long long)y0 * nx + x0];
            float v10 = slice_base[(long long)y0 * nx + x1];
            float v01 = slice_base[(long long)y1 * nx + x0];
            float v11 = slice_base[(long long)y1 * nx + x1];
            float v = v00 * (1.f-tx) * (1.f-ty)
                    + v10 * tx * (1.f-ty)
                    + v01 * (1.f-tx) * ty
                    + v11 * tx * ty;
            dst[(size_t)h * MODEL25D_W + w] = (clampf(v, hu_lo, hu_hi) - hu_lo) * inv_range;
        }
    }
}

// Bilinear resize + transpose model output prob back to volume slice layout.
// src : [MODEL25D_H, MODEL25D_W] airway probability (softmax channel 1)
// dst : prob[z*ny*nx + y*nx + x] — fills for one slice
static void accumulate_slice_25d(float* dst,
                                 const float* src,
                                 int nx, int ny) {
    const float sx = (float)(MODEL25D_H - 1) / std::max(1, nx - 1);
    const float sy = (float)(MODEL25D_W - 1) / std::max(1, ny - 1);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            float fh = x * sx;
            float fw = y * sy;
            int h0 = (int)fh; int h1 = std::min(h0 + 1, MODEL25D_H - 1); float th = fh - h0;
            int w0 = (int)fw; int w1 = std::min(w0 + 1, MODEL25D_W - 1); float tw = fw - w0;
            float p00 = src[(size_t)h0 * MODEL25D_W + w0];
            float p10 = src[(size_t)h1 * MODEL25D_W + w0];
            float p01 = src[(size_t)h0 * MODEL25D_W + w1];
            float p11 = src[(size_t)h1 * MODEL25D_W + w1];
            dst[(long long)y * nx + x] =
                p00 * (1.f-th) * (1.f-tw)
              + p10 * th * (1.f-tw)
              + p01 * (1.f-th) * tw
              + p11 * th * tw;
        }
    }
}

float ort_infer_volume_25d(OrtPipeline* pipe,
                           const float* hu, float* prob,
                           int nx, int ny, int nz,
                           float hu_lo, float hu_hi,
                           int batch_size) {
    if (!pipe || !pipe->p.sess) return -1.f;
    if (batch_size < 1) batch_size = 1;
    auto& p = pipe->p;

    const long long plane = (long long)nx * ny;
    const long long N     = (long long)nx * ny * nz;
    std::memset(prob, 0, sizeof(float) * (size_t)N);

    const size_t slice_sz  = (size_t)MODEL25D_H * MODEL25D_W;
    const size_t in_elems  = (size_t)batch_size * 3 * slice_sz;
    const size_t out_elems = (size_t)batch_size * 2 * slice_sz;

    std::vector<float> in_buf(in_elems, 0.f);
    std::vector<float> out_buf(out_elems, 0.f);
    std::vector<float> ch1_prob(slice_sz);

    void* d_in  = nullptr;
    void* d_out = nullptr;
    if (cudaMalloc(&d_in,  in_elems  * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_out, out_elems * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "[25d] cudaMalloc failed\n");
        if (d_in)  cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        return -1.f;
    }

    OrtMemoryInfo* raw_mi = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateMemoryInfo(
        "Cuda", OrtArenaAllocator, p.gpu < 0 ? 0 : p.gpu,
        OrtMemTypeDefault, &raw_mi));
    Ort::MemoryInfo cuda_mi(raw_mi);

    const char* in_names[]  = { p.in_name.c_str() };
    const char* out_names[] = { p.out_name.c_str() };

    bool failed = false;
    auto t0 = std::chrono::steady_clock::now();

    for (int z_start = 0; z_start < nz && !failed; z_start += batch_size) {
        int B = std::min(batch_size, nz - z_start);

        std::memset(in_buf.data(), 0, sizeof(float) * (size_t)B * 3 * slice_sz);
        for (int b = 0; b < B; b++) {
            int z = z_start + b;
            float* ch0 = in_buf.data() + (size_t)b * 3 * slice_sz;
            float* ch1 = ch0 + slice_sz;
            float* ch2 = ch1 + slice_sz;
            fill_channel_25d(ch0, hu, nx, ny, plane,
                             std::max(0, z - 1), hu_lo, hu_hi);
            fill_channel_25d(ch1, hu, nx, ny, plane, z, hu_lo, hu_hi);
            fill_channel_25d(ch2, hu, nx, ny, plane,
                             std::min(nz - 1, z + 1), hu_lo, hu_hi);
        }

        cudaMemcpy(d_in, in_buf.data(),
                   sizeof(float) * (size_t)B * 3 * slice_sz,
                   cudaMemcpyHostToDevice);

        const int64_t in_shape[4]  = {B, 3, MODEL25D_H, MODEL25D_W};
        const int64_t out_shape[4] = {B, 2, MODEL25D_H, MODEL25D_W};

        try {
            Ort::Value in_val = Ort::Value::CreateTensor<float>(
                cuda_mi, (float*)d_in,
                (size_t)B * 3 * slice_sz, in_shape, 4);
            Ort::Value out_val = Ort::Value::CreateTensor<float>(
                cuda_mi, (float*)d_out,
                (size_t)B * 2 * slice_sz, out_shape, 4);
            p.sess->Run(Ort::RunOptions{nullptr},
                        in_names, &in_val, 1,
                        out_names, &out_val, 1);
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "[25d] Run failed: %s\n", e.what());
            failed = true; break;
        }

        // GPU softmax (same kernel as the 3D path) keeps the 2.5D and 3D
        // pipelines on an equal footing: reduce 2-channel logits to 1-channel
        // probability on the device, copy back one channel. Otherwise copy the
        // full 2 channels and run the host exp() loop.
        const bool gsm = p.o.gpu_softmax && !p.model_fp16;
        if (gsm) {
            cudaDeviceSynchronize();
            float* d_prob = nullptr;
            cudaMalloc(&d_prob, sizeof(float) * (size_t)B * slice_sz);
            cuda_softmax_ch1((const float*)d_out, d_prob, (long long)slice_sz, B);
            cudaMemcpy(out_buf.data(), d_prob,
                       sizeof(float) * (size_t)B * slice_sz,
                       cudaMemcpyDeviceToHost);
            cudaFree(d_prob);
        } else {
            cudaMemcpy(out_buf.data(), d_out,
                       sizeof(float) * (size_t)B * 2 * slice_sz,
                       cudaMemcpyDeviceToHost);
        }

        for (int b = 0; b < B; b++) {
            int z = z_start + b;
            if (gsm) {
                std::memcpy(ch1_prob.data(), out_buf.data() + (size_t)b * slice_sz,
                            sizeof(float) * slice_sz);
            } else {
                const float* logit0 = out_buf.data() + (size_t)b * 2 * slice_sz;
                const float* logit1 = logit0 + slice_sz;
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < slice_sz; i++) {
                    float m  = logit0[i] > logit1[i] ? logit0[i] : logit1[i];
                    float e0 = std::exp(logit0[i] - m);
                    float e1 = std::exp(logit1[i] - m);
                    ch1_prob[i] = e1 / (e0 + e1);
                }
            }
            float* prob_slice = prob + (long long)z * plane;
            accumulate_slice_25d(prob_slice, ch1_prob.data(), nx, ny);
        }

        int done = z_start + B;
        if (z_start == 0 || done % 64 == 0 || done == nz)
            fprintf(stderr, "[25d] slices %d/%d\n", done, nz);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    if (failed) return -1.f;

    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}
