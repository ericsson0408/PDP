#pragma once
#include <string>
#include <vector>

// Scheme A/B: a 2D (intensity x gradient-magnitude) airway-probability table.
// Trained offline from cases with ground truth, then bound to CUDA texture
// memory at inference time for O(1), hardware-interpolated lookups.

struct Lut2D {
    int Ibins = 256, Gbins = 128;
    float Imin = -1100.f, Imax = -600.f;  // smoothed-intensity (HU) axis
    float Gmax = 300.f;                     // gradient-magnitude axis (0..Gmax)
    std::vector<float> prob;                // Ibins*Gbins, prob[g*Ibins + i]
};

// Central-difference gradient magnitude of a (smoothed) volume, clamp boundary.
void gradient_magnitude(const float* vol, float* grad, int nx, int ny, int nz);

// Accumulate (airway,total) counts from one labelled case into a binary counts
// file (created if absent, summed if present). `smoothed` and `gt` are full
// volumes of size nx*ny*nz. Returns false on I/O error.
bool lut_train_accumulate(const std::string& counts_path,
                          const float* smoothed, const unsigned char* gt,
                          int nx, int ny, int nz, const Lut2D& spec);

// Build a normalized probability LUT from a counts file. Bins with fewer than
// `min_total` samples are set to 0. Returns false if the file is missing.
bool lut_load(const std::string& counts_path, Lut2D& out, long long min_total = 20);
