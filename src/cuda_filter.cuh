#pragma once

// CUDA separable 3D Gaussian smoothing + fused thresholding.
// Operates on one MPI rank's extended slab (local_nz already includes halo).
// Produces a uint8 air-candidate mask (1 where threshold_low <= smoothed <= threshold_high).

// Number of visible CUDA devices (0 if none / driver error).
int cuda_device_count();

// Fill a normalized 1D Gaussian kernel of length (2*radius+1).
void generate_gaussian_kernel(float* k, int radius, float sigma);

// Run X->Y->Z separable Gaussian then fused threshold on the GPU.
//   in       : host float volume, size nx*ny*nz (extended slab)
//   mask_out : host uint8 buffer, size nx*ny*nz (caller-allocated)
// Returns kernel-only GPU time in milliseconds (excludes H2D/D2H), or -1 on error.
float cuda_gaussian_threshold(const float* in, unsigned char* mask_out,
                              int nx, int ny, int nz,
                              int radius, float sigma,
                              float threshold_low, float threshold_high,
                              int device_id);

// Scheme B: separable Gaussian, then per-voxel gradient magnitude + a 2D
// (intensity x gradient) probability lookup bound to CUDA texture memory
// (hardware bilinear interpolation). Writes mask=1 where prob >= prob_thresh.
// lut_prob is row-major prob[g*Ibins + i]. Returns kernel ms, or -1 on error.
float cuda_gaussian_lut(const float* in, unsigned char* mask_out,
                        int nx, int ny, int nz, int radius, float sigma,
                        const float* lut_prob, int Ibins, int Gbins,
                        float Imin, float Imax, float Gmax,
                        float prob_thresh, int device_id);

// Smoothing-only GPU path (float output, no threshold) for validation.
float cuda_gaussian_smooth(const float* in, float* out,
                           int nx, int ny, int nz,
                           int radius, float sigma, int device_id);

// Method 1: multiscale Frangi "dark-tube" vesselness on the GPU. For each scale
// it Gaussian-smooths, builds the 3D Hessian (scale-normalized), solves the
// symmetric-3x3 eigenvalues (lambda1<=lambda2<=lambda3 by magnitude), and forms
// the Frangi vesselness for a dark tube on bright background (lambda2,3 > 0).
// Output is the per-voxel max vesselness across scales in [0,1].
//   in        : host float volume (clamped HU), size nx*ny*nz
//   vness_out : host float buffer, size nx*ny*nz (caller-allocated)
//   scales    : array of Gaussian sigmas (voxel units), nscales entries
// Returns kernel-only GPU time in ms, or -1 on error.
float cuda_vesselness(const float* in, float* vness_out,
                      int nx, int ny, int nz,
                      const float* scales, int nscales, int device_id);

// Serial CPU reference for the separable Gaussian (no threshold), for validation.
void cpu_gaussian_reference(const float* in, float* out,
                            int nx, int ny, int nz, int radius, float sigma);

// V8: stream-overlapped variant. Chunks the slab into NCHUNK Z-slices and
// pipelines H2D / kernel / D2H across 2 streams with double-buffering. The
// Z-direction convolution still uses clamp-to-edge at chunk boundaries WITHIN
// the rank (the outer-slab halo from MPI is already accounted for at the
// caller level; this function operates on the already-extended slab).
// Pinned host buffers are pulled from an internal pool (no alloc per call).
float cuda_gaussian_threshold_v8(const float* in, unsigned char* mask_out,
                                 int nx, int ny, int nz,
                                 int radius, float sigma,
                                 float threshold_low, float threshold_high,
                                 int device_id);

// V8: Frangi vesselness with sigma-recurrence — instead of smoothing the
// original volume to each scale independently, smooth incrementally with
// G_σi+1 = G_sqrt(σi+1² - σi²) ∗ G_σi.  Saves ~17% of separable-conv work for
// the {1,2,3}-scale default. Identical numerical result up to floating-point
// roundoff (well within Frangi's response amplitude tolerance).
float cuda_vesselness_v8(const float* in, float* vness_out,
                         int nx, int ny, int nz,
                         const float* scales, int nscales, int device_id);

// V18 ablation: device-side softmax of 2-channel logits -> 1-channel foreground
// probability. d_logits/d_prob are DEVICE pointers; launches on the default
// stream (caller's subsequent D2H copy is ordered after). See cuda_filter.cu.
void cuda_softmax_ch1(const float* d_logits, float* d_prob,
                      long long patch_vox, int B);
