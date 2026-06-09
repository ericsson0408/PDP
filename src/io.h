#pragma once
#include <cstdint>
#include <string>

// Minimal self-contained NIfTI-1 (.nii / .nii.gz) handling.
// Reads scalar 3D volumes, converts to float32 (HU), and writes uint8 masks
// reusing the source header geometry. Uses zlib for gzip (.nii.gz) support.

struct Volume {
    int nx = 0, ny = 0, nz = 0;          // spatial dims
    float dx = 1, dy = 1, dz = 1;        // voxel spacing (mm)
    float* data = nullptr;               // nx*ny*nz float32 (HU after scaling)
    unsigned char header[352];           // raw NIfTI-1 header (348) + 4 ext flag

    long long voxels() const { return (long long)nx * ny * nz; }
    ~Volume();
};

// Load a NIfTI image; converts any supported datatype to float32 with
// scl_slope/scl_inter applied. Exits on error.
bool load_nifti(const std::string& path, Volume& vol);

// Clamp HU values into [hu_min, hu_max] in place.
void clamp_hu(float* data, long long N, float hu_min, float hu_max);

// Write a uint8 mask (mask[N], 0/1) as .nii.gz, reusing ref's header geometry.
bool write_mask_nifti(const std::string& path, const Volume& ref,
                      const unsigned char* mask);

// Load a uint8 label volume (e.g. ground-truth mask) into a freshly allocated
// buffer. Returns nullptr on failure; caller frees with delete[].
unsigned char* load_label_u8(const std::string& path, int& nx, int& ny, int& nz);
