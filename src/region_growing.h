#pragma once
#include <vector>
#include <cstdint>

struct Seed { int x, y, z; };

struct BBox {
    int x0 = 0, y0 = 0, z0 = 0, x1 = 0, y1 = 0, z1 = 0;  // inclusive bounds
    bool valid = false;
    long long voxels(int /*nx*/, int /*ny*/) const {
        return valid ? (long long)(x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1) : 0;
    }
};

struct RegionGrowResult {
    std::vector<unsigned char> mask;  // nx*ny*nz airway mask (0/1)
    int num_seeds = 0;
    long long interior_voxels = 0;
    long long airway_voxels = 0;
};

// Scheme C/F: tight body bounding box from soft-tissue voxels (HU > body_thr).
// OpenMP-parallel reduction over Z slices. Used as an ROI to isolate the body
// from couch/background and to bound all downstream work.
BBox compute_body_bbox(const float* hu, int nx, int ny, int nz,
                       float body_thr, int margin, int num_threads);

// Detect trachea/central-airway seeds automatically from a candidate air mask.
// Strategy: flood-fill exterior air from the volume borders, keep only interior
// air, then take the interior-air voxel nearest the central axis on each slice.
std::vector<Seed> detect_seeds(const unsigned char* candidate,
                               int nx, int ny, int nz,
                               std::vector<unsigned char>& interior_out,
                               float central_frac = 0.10f);

// Scheme E/F: ROI-bounded, OpenMP per-slice-parallel exterior detection.
// Each Z slice is flooded independently (2D, from its lateral edges) so the
// work is embarrassingly parallel; interior air = candidate inside `roi` and
// not reachable from any slice border. Returns central-axis seeds.
std::vector<Seed> detect_seeds_parallel(const unsigned char* candidate,
                                        int nx, int ny, int nz,
                                        const BBox& roi, int num_threads,
                                        std::vector<unsigned char>& interior_out,
                                        float central_frac = 0.10f);

// Scheme D: 2x average-pool downsample of a volume (Gaussian-pyramid level 1).
void downsample2x_avg(const float* hu, int nx, int ny, int nz,
                      std::vector<float>& out, int& dnx, int& dny, int& dnz,
                      int num_threads);

// Scheme D: in-place separable 3D binary box-dilation by `margin` voxels, O(N)
// per axis (radius-independent). Cheap when run at coarse resolution.
void dilate3d(std::vector<unsigned char>& mask, int nx, int ny, int nz,
              int margin, int num_threads);

// V15: in-place 3D binary morphological CLOSING (dilate by `margin`, then
// erode by `margin`). Fills concave gaps up to `margin` voxels wide without
// expanding the overall boundary -- bridges thin-disconnected airway branches
// so they survive the central-seeded CC step. O(N) per axis (uses dilate3d
// twice via the complement trick: erode(M) = NOT dilate(NOT M)).
void close3d(std::vector<unsigned char>& mask, int nx, int ny, int nz,
             int margin, int num_threads);

// V17 HYBRID: AI-assisted traditional region growing. Builds a candidate mask
// = hardset ∪ {voxels where prob > prob_gate AND hu in [hu_lo, hu_hi]},
// finds central-axis seeds inside the hardset (the AI-confident core), and
// runs the existing parallel multiseed BFS (multiseed_region_grow_v8) within
// the candidate. The candidate construction is the "AI hard gate": growth
// physically cannot escape voxels where the U-Net's probability is below
// prob_gate or whose HU is outside the airway lumen range. The BFS itself is
// pure traditional connectivity, parallelised over seeds with lock-free
// atomic voxel claiming -- so the result is a real traditional region-grown
// mask, bounded by the AI as a safety net.
//
//   hu, prob: native-resolution volumes (size nx*ny*nz).
//   hardset : AI core (prob > core_thr), zero elsewhere.
//   hu_lo/hi: airway lumen HU bounds (defaults -1000, -500 -> air-ish).
//   prob_gate: hard AI floor (default 0.1).
//   roi    : body-bbox to constrain seed search.
//   The returned RegionGrowResult.mask is the AI-core ∪ traditionally-grown
//   uncertainty-zone voxels reachable from the core.
RegionGrowResult hybrid_grow(const float* hu, const float* prob,
                             const std::vector<unsigned char>& hardset,
                             int nx, int ny, int nz,
                             float hu_lo, float hu_hi, float prob_gate,
                             const BBox& roi, int num_threads);

// Scheme D: nearest-neighbour upsample a coarse mask to full resolution, then
// dilate by `margin` voxels (separable 3D max filter). Produces an ROI that
// loosely envelopes the coarse airway trunk to constrain fine region growing.
std::vector<unsigned char> upsample_dilate(const std::vector<unsigned char>& coarse,
                                           int dnx, int dny, int dnz,
                                           int nx, int ny, int nz,
                                           int margin, int num_threads);

// Pick the interior voxel nearest the central axis on each slice (within ROI).
std::vector<Seed> central_axis_seeds(const unsigned char* interior,
                                     int nx, int ny, int nz, const BBox& roi,
                                     float central_frac = 0.10f);

// v5: robust seed search. If the standard central-axis search yields too few
// seeds (interior air absent in the central R for most slices -- patient with
// off-axis trachea, tube artefact, or aspiration), progressively widen the
// search radius and finally fall back to ANY interior-air voxel per slice.
// Guarantees at least one seed if the interior mask has any voxel at all.
std::vector<Seed> central_axis_seeds_robust(const unsigned char* interior,
                                            int nx, int ny, int nz, const BBox& roi,
                                            float central_frac = 0.10f);

// Upgrade 招1+招3: data-driven coarse air threshold. Within the central core
// of the body bbox (anatomical trachea location, 招3), accumulate mean/std of
// "smooth air" voxels -- low gradient-magnitude lumen, not wall edges (招1) --
// and return mu + k*sigma as the upper HU bound for the coarse trunk band.
// Robust to per-case HU drift that breaks a fixed threshold. `core_frac` is the
// fraction of the bbox X/Y extent kept around the center.
float adaptive_coarse_threshold(const float* chu, int dnx, int dny, int dnz,
                                const BBox& body, float k, float core_frac,
                                int num_threads);

// Upgrade 招2: anatomical feedback. Zero out coarse-trunk slices that "exploded"
// into the lung -- detected as a per-slice area spike (vs the trunk median) with
// a sprawling, low-fill bounding box (an octopus shape, not a compact lumen).
// Run on the coarse trunk before dilation to keep leaks out of the ROI.
void prune_leaked_slices(std::vector<unsigned char>& trunk,
                         int nx, int ny, int nz, int num_threads);

// Tunables for the wavefront grower (methods 2+3).
struct WavefrontParams {
    float thr_trunk  = -950.f;  // relaxed-path HU ceiling near the trunk (level 0)
    float thr_periph = -720.f;  // relaxed-path HU ceiling at the periphery
    float vmin       = 0.20f;   // vesselness gate for the relaxed/tubular path
    int   max_level  = 60;      // BFS level at which the ceiling reaches thr_periph
    float leak_factor= 3.0f;    // a layer larger than leak_factor*recent => explosion
    long long leak_base = 800;  // absolute layer size below which no explosion is flagged
    float leak_ratio = 3.0f;    // catastrophic guard: if relaxed-path adds > ratio*|hardset|
                                // voxels, the lung is flooding -> disable relaxed growth
    // v5: anatomical brake. Pulmonary vessels are tubular (high vesselness) but
    // soft-tissue (HU ~ 0). Without a hard HU ceiling, Frangi happily walks the
    // relaxed wavefront down a vessel that touches a thin airway wall. Reject
    // relaxed voxels with hu > hu_abs_max regardless of BFS depth.
    float hu_abs_max = -500.f;  // absolute air ceiling for the relaxed path
};

// v6: rescue plan when the v5 pipeline produces an empty airway mask. The
// failure mode (case 82) is HU drift -- the global air threshold (-1100..-920)
// finds zero candidates in the trachea region because lumen voxels actually
// sit around -700..-800 HU. The rescue is independent of the GPU air mask: it
// sweeps a per-case HU threshold over the upper-central column of the body
// bbox, runs 3D connected components at each step, and accepts the FIRST
// component that matches trachea geometry (Z-elongated, compact XY, central).
// Returns seeds + the threshold at which that component appeared, so the
// caller can re-binarise and re-grow the airway without trusting -920 HU.
struct TracheaCandidate {
    std::vector<Seed> seeds;
    float threshold = 0.f;
    long long voxels = 0;
    bool valid = false;
    // Full-volume binary mask of the rescued trachea component (only the
    // voxels that belong to the connected component the sweep selected, in
    // the original nx*ny*nz coordinate system). Empty when valid=false.
    // Used by main.cpp to build a dilated growth envelope so the rescue's
    // multiseed grow doesn't leak into the lung (case 124).
    std::vector<unsigned char> trunk_mask;
};
TracheaCandidate rescue_threshold_sweep(const float* hu, int nx, int ny, int nz,
                                        const BBox& body, int num_threads);

// Methods 2+3: level-synchronized wavefront region growing.
//  - `hardset` is the definite-airway candidate (the smoothed-air mask). It is
//    always accepted, so growth reproduces the robust baseline connectivity.
//  - Branch-wise local adaptive threshold (招2): on TOP of hardset, the relaxed
//    HU ceiling rises from thr_trunk toward thr_periph as the BFS advances, so
//    faint peripheral branches that the air threshold missed can be recovered.
//  - Vesselness gate (招1): a relaxed-path voxel must also be tubular
//    (vness >= vmin), so growth follows tubes not the spongy lung.
//  - Topological rollback (招3): if a wavefront layer explodes (size jumps past
//    leak_factor * recent mean, i.e. burst into the lung), that layer is
//    re-filtered to hardset-only; the relaxed voxels are rolled back.
// `roi` (may be null) bounds all growth. Returns the airway mask + counts.
RegionGrowResult wavefront_region_grow(const float* hu, const float* vness,
                                       const unsigned char* hardset,
                                       const unsigned char* roi,
                                       int nx, int ny, int nz,
                                       const std::vector<Seed>& seeds,
                                       const WavefrontParams& p, int num_threads);

// OpenMP multi-seed 6-connected region growing within the interior-air mask.
// Different seeds grow concurrently; voxels are claimed atomically (lock-free
// union) so the result is the union of all reachable components.
RegionGrowResult multiseed_region_grow(const unsigned char* interior,
                                        int nx, int ny, int nz,
                                        const std::vector<Seed>& seeds,
                                        int num_threads);

// V8: level-parallel BFS variants. The v7 versions are single-threaded inside
// the BFS main loop (per-seed parallelism only for multiseed). V8 expands each
// BFS layer in parallel: each thread takes a chunk of the current frontier,
// expands 6-neighbours, and atomically claims voxels. Thread-local "next"
// frontiers are merged after the layer. Leak detection / commit / rollback
// semantics are byte-identical to v7 — the layer size (and therefore the
// leak_factor / recent_mean decision) depends only on which voxels are
// reachable, not on the order they were enqueued in a layer.
RegionGrowResult wavefront_region_grow_v8(const float* hu, const float* vness,
                                          const unsigned char* hardset,
                                          const unsigned char* roi,
                                          int nx, int ny, int nz,
                                          const std::vector<Seed>& seeds,
                                          const WavefrontParams& p, int num_threads);
RegionGrowResult multiseed_region_grow_v8(const unsigned char* interior,
                                          int nx, int ny, int nz,
                                          const std::vector<Seed>& seeds,
                                          int num_threads);
