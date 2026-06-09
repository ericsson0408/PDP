#include "region_growing.h"
#include <omp.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace {
inline long long lin(int x, int y, int z, int nx, int ny) {
    return (long long)z * ny * nx + (long long)y * nx + x;
}
} // namespace

std::vector<Seed> detect_seeds(const unsigned char* candidate,
                               int nx, int ny, int nz,
                               std::vector<unsigned char>& interior,
                               float central_frac) {
    long long N = (long long)nx * ny * nz;
    long long plane = (long long)nx * ny;

    // 1) Flood-fill EXTERIOR air starting from every border air voxel.
    std::vector<unsigned char> exterior(N, 0);
    std::vector<long long> stack;
    stack.reserve(1 << 20);

    auto push_if = [&](int x, int y, int z) {
        long long i = lin(x, y, z, nx, ny);
        if (candidate[i] && !exterior[i]) { exterior[i] = 1; stack.push_back(i); }
    };
    // Seed ONLY from the lateral (X/Y) faces. The surrounding background air
    // touches these faces, but the trachea opens through the Z (top/bottom)
    // slices of the scan -- seeding those would flood the whole airway tree as
    // "exterior" and erase it.
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++) { push_if(0, y, z); push_if(nx - 1, y, z); }
    for (int z = 0; z < nz; z++)
        for (int x = 0; x < nx; x++) { push_if(x, 0, z); push_if(x, ny - 1, z); }

    while (!stack.empty()) {
        long long cur = stack.back(); stack.pop_back();
        int x = cur % nx, y = (cur / nx) % ny, z = cur / plane;
        const int dx[6]={1,-1,0,0,0,0}, dy[6]={0,0,1,-1,0,0}, dz[6]={0,0,0,0,1,-1};
        for (int d = 0; d < 6; d++) {
            int xx=x+dx[d], yy=y+dy[d], zz=z+dz[d];
            if (xx<0||xx>=nx||yy<0||yy>=ny||zz<0||zz>=nz) continue;
            long long ni = lin(xx, yy, zz, nx, ny);
            if (candidate[ni] && !exterior[ni]) { exterior[ni]=1; stack.push_back(ni); }
        }
    }

    // 2) Interior air = candidate AND not exterior.
    interior.assign(N, 0);
    for (long long i = 0; i < N; i++)
        interior[i] = (candidate[i] && !exterior[i]) ? 1 : 0;

    // 3) Seeds: nearest interior voxel to the central axis on each slice.
    int cx = nx / 2, cy = ny / 2;
    int R = (int)(central_frac * (nx < ny ? nx : ny));
    std::vector<Seed> seeds;
    for (int z = 0; z < nz; z++) {
        long long best = -1; int bestd = R * R + 1;
        for (int y = cy - R; y <= cy + R; y++) {
            if (y < 0 || y >= ny) continue;
            for (int x = cx - R; x <= cx + R; x++) {
                if (x < 0 || x >= nx) continue;
                long long i = lin(x, y, z, nx, ny);
                if (!interior[i]) continue;
                int dd = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                if (dd < bestd) { bestd = dd; best = i; }
            }
        }
        if (best >= 0)
            seeds.push_back({(int)(best % nx), (int)((best / nx) % ny), (int)(best / plane)});
    }
    return seeds;
}

BBox compute_body_bbox(const float* hu, int nx, int ny, int nz,
                       float body_thr, int margin, int num_threads) {
    long long plane = (long long)nx * ny;
    int gx0 = nx, gy0 = ny, gz0 = nz, gx1 = -1, gy1 = -1, gz1 = -1;
    omp_set_num_threads(num_threads);

    // Per-slice min/max of soft-tissue voxels, reduced across threads.
    #pragma omp parallel for schedule(static) \
        reduction(min:gx0,gy0,gz0) reduction(max:gx1,gy1,gz1)
    for (int z = 0; z < nz; z++) {
        const float* sl = hu + (long long)z * plane;
        bool any = false;
        int lx0 = nx, ly0 = ny, lx1 = -1, ly1 = -1;
        for (int y = 0; y < ny; y++) {
            const float* row = sl + (long long)y * nx;
            for (int x = 0; x < nx; x++) {
                if (row[x] > body_thr) {
                    any = true;
                    if (x < lx0) lx0 = x; if (x > lx1) lx1 = x;
                    if (y < ly0) ly0 = y; if (y > ly1) ly1 = y;
                }
            }
        }
        if (any) {
            if (lx0 < gx0) gx0 = lx0; if (lx1 > gx1) gx1 = lx1;
            if (ly0 < gy0) gy0 = ly0; if (ly1 > gy1) gy1 = ly1;
            if (z < gz0) gz0 = z;     if (z > gz1) gz1 = z;
        }
    }

    BBox b;
    if (gx1 < 0) return b;  // no body found
    b.valid = true;
    b.x0 = gx0 > margin ? gx0 - margin : 0;
    b.y0 = gy0 > margin ? gy0 - margin : 0;
    b.z0 = gz0 > margin ? gz0 - margin : 0;
    b.x1 = gx1 + margin < nx ? gx1 + margin : nx - 1;
    b.y1 = gy1 + margin < ny ? gy1 + margin : ny - 1;
    b.z1 = gz1 + margin < nz ? gz1 + margin : nz - 1;
    return b;
}

std::vector<Seed> detect_seeds_parallel(const unsigned char* candidate,
                                        int nx, int ny, int nz,
                                        const BBox& roi, int num_threads,
                                        std::vector<unsigned char>& interior,
                                        float central_frac) {
    long long N = (long long)nx * ny * nz;
    long long plane = (long long)nx * ny;
    interior.assign(N, 0);
    omp_set_num_threads(num_threads);

    BBox b = roi;
    if (!b.valid) { b.x0=0; b.y0=0; b.z0=0; b.x1=nx-1; b.y1=ny-1; b.z1=nz-1; b.valid=true; }

    // Per-slice 2D exterior flood-fill, parallel over Z (slices are independent
    // because background air touches every slice's lateral border). Each thread
    // owns a disjoint set of slices, so writes to `interior` never race.
    #pragma omp parallel
    {
        std::vector<int> ext(plane);          // per-slice scratch (reused)
        std::vector<long long> stk;
        stk.reserve(1 << 14);
        #pragma omp for schedule(dynamic, 4)
        for (int z = b.z0; z <= b.z1; z++) {
            const unsigned char* cs = candidate + (long long)z * plane;
            std::fill(ext.begin(), ext.end(), 0);
            stk.clear();
            auto push2d = [&](int x, int y) {
                long long i = (long long)y * nx + x;
                if (cs[i] && !ext[i]) { ext[i] = 1; stk.push_back(i); }
            };
            // Seed from the ROI's lateral edges on this slice.
            for (int y = b.y0; y <= b.y1; y++) { push2d(b.x0, y); push2d(b.x1, y); }
            for (int x = b.x0; x <= b.x1; x++) { push2d(x, b.y0); push2d(x, b.y1); }
            while (!stk.empty()) {
                long long c = stk.back(); stk.pop_back();
                int x = c % nx, y = c / nx;
                if (x > b.x0) push2d(x - 1, y);
                if (x < b.x1) push2d(x + 1, y);
                if (y > b.y0) push2d(x, y - 1);
                if (y < b.y1) push2d(x, y + 1);
            }
            // interior = candidate inside ROI and not flagged exterior.
            unsigned char* is = interior.data() + (long long)z * plane;
            for (int y = b.y0; y <= b.y1; y++)
                for (int x = b.x0; x <= b.x1; x++) {
                    long long i = (long long)y * nx + x;
                    is[i] = (cs[i] && !ext[i]) ? 1 : 0;
                }
        }
    }

    return central_axis_seeds(interior.data(), nx, ny, nz, b, central_frac);
}

std::vector<Seed> central_axis_seeds_robust(const unsigned char* interior,
                                            int nx, int ny, int nz, const BBox& roi,
                                            float central_frac) {
    long long plane = (long long)nx * ny;
    BBox b = roi;
    if (!b.valid) { b.x0=0; b.y0=0; b.z0=0; b.x1=nx-1; b.y1=ny-1; b.z1=nz-1; }

    // Try progressively widening central radii. The trachea normally sits in
    // the central 10% of the slice, but in cases with tube/aspiration artefact
    // the only interior air may be off-axis: widen until we find something.
    const float fracs[] = { central_frac, central_frac * 1.5f,
                            central_frac * 2.5f, 0.50f };
    std::vector<Seed> seeds;
    for (float frac : fracs) {
        seeds = central_axis_seeds(interior, nx, ny, nz, b, frac);
        // Heuristic: a healthy trachea spans most of the body's Z range. If the
        // search finds at least ~25% of the slices we're done.
        int z_span = b.z1 - b.z0 + 1;
        if ((int)seeds.size() >= std::max(8, z_span / 4)) {
            if (frac > central_frac + 1e-6f)
                printf("[seeds-robust] widened central_frac to %.2f -> %zu seeds\n",
                       frac, seeds.size());
            return seeds;
        }
    }

    // Final fallback: scan the ENTIRE ROI for any interior-air voxel per slice
    // (no centrality preference). Better an off-axis seed than none.
    if (seeds.empty()) {
        for (int z = b.z0; z <= b.z1; z++) {
            for (int y = b.y0; y <= b.y1; y++) {
                bool done = false;
                for (int x = b.x0; x <= b.x1; x++) {
                    long long i = (long long)z * plane + (long long)y * nx + x;
                    if (interior[i]) {
                        seeds.push_back({x, y, z});
                        done = true; break;
                    }
                }
                if (done) break;
            }
        }
        printf("[seeds-robust] full-ROI fallback -> %zu seeds\n", seeds.size());
    }
    return seeds;
}

std::vector<Seed> central_axis_seeds(const unsigned char* interior,
                                     int nx, int ny, int nz, const BBox& roi,
                                     float central_frac) {
    long long plane = (long long)nx * ny;
    BBox b = roi;
    if (!b.valid) { b.x0=0; b.y0=0; b.z0=0; b.x1=nx-1; b.y1=ny-1; b.z1=nz-1; }
    int cx = nx / 2, cy = ny / 2;
    int R = (int)(central_frac * (nx < ny ? nx : ny));
    std::vector<Seed> seeds;
    for (int z = b.z0; z <= b.z1; z++) {
        long long best = -1; int bestd = R * R + 1;
        for (int y = cy - R; y <= cy + R; y++) {
            if (y < b.y0 || y > b.y1) continue;
            for (int x = cx - R; x <= cx + R; x++) {
                if (x < b.x0 || x > b.x1) continue;
                long long i = (long long)z * plane + (long long)y * nx + x;
                if (!interior[i]) continue;
                int dd = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                if (dd < bestd) { bestd = dd; best = i; }
            }
        }
        if (best >= 0)
            seeds.push_back({(int)(best % nx), (int)((best / nx) % ny), (int)(best / plane)});
    }
    return seeds;
}

float adaptive_coarse_threshold(const float* chu, int dnx, int dny, int dnz,
                                const BBox& body, float k, float core_frac,
                                int num_threads) {
    long long plane = (long long)dnx * dny;
    omp_set_num_threads(num_threads);

    // 招3: spatial anchor. The trachea trunk runs through the centre of the
    // chest, so restrict statistics to the central core of the body bbox.
    int bx0 = body.valid ? body.x0 : 0, bx1 = body.valid ? body.x1 : dnx - 1;
    int by0 = body.valid ? body.y0 : 0, by1 = body.valid ? body.y1 : dny - 1;
    int bz0 = body.valid ? body.z0 : 0, bz1 = body.valid ? body.z1 : dnz - 1;
    int ccx = (bx0 + bx1) / 2, ccy = (by0 + by1) / 2;
    int hwx = (int)(0.5f * core_frac * (bx1 - bx0 + 1)) + 1;
    int hwy = (int)(0.5f * core_frac * (by1 - by0 + 1)) + 1;
    int cx0 = std::max(1, ccx - hwx),       cx1 = std::min(dnx - 2, ccx + hwx);
    int cy0 = std::max(1, ccy - hwy),       cy1 = std::min(dny - 2, ccy + hwy);
    int cz0 = std::max(1, bz0),             cz1 = std::min(dnz - 2, bz1);

    // 招1: semantic weighting. Keep only "smooth air" -- HU below a loose air
    // ceiling (excludes soft tissue) AND gradient-magnitude below grad_max
    // (excludes airway walls / lung interfaces). The survivors are flat air.
    const float air_ceiling = -600.f;
    const float grad_max    = 80.f;

    // Collect smooth-air HU values in the core (single pass; the core is small).
    std::vector<float> vals;
    vals.reserve((size_t)(cx1 - cx0 + 1) * (cy1 - cy0 + 1));
    for (int z = cz0; z <= cz1; z++)
      for (int y = cy0; y <= cy1; y++)
        for (int x = cx0; x <= cx1; x++) {
            long long i = (long long)z * plane + (long long)y * dnx + x;
            float v = chu[i];
            if (v >= air_ceiling) continue;
            float gx = chu[i + 1]     - chu[i - 1];
            float gy = chu[i + dnx]   - chu[i - dnx];
            float gz = chu[i + plane] - chu[i - plane];
            float g = 0.5f * sqrtf(gx * gx + gy * gy + gz * gz);
            if (g > grad_max) continue;
            vals.push_back(v);
        }

    if (vals.size() < 50) return -960.f;  // degenerate -> static fallback

    // 招3: percentile anchor. The 5th-percentile HU is the airway-lumen floor
    // for THIS case (robust to scanner/HU drift). The trunk band sits a margin
    // above it; `k` scales that margin so the lumen is captured but lung
    // parenchyma (which lies higher) is excluded.
    std::sort(vals.begin(), vals.end());
    float lumen_floor = vals[(size_t)(0.05 * vals.size())];
    float thr = lumen_floor + 30.0f * k;   // k=2.5 -> +75 HU above the floor
    if (thr < -1010.f) thr = -1010.f;
    if (thr > -905.f)  thr = -905.f;
    return thr;
}

void prune_leaked_slices(std::vector<unsigned char>& trunk,
                         int nx, int ny, int nz, int num_threads) {
    long long plane = (long long)nx * ny;
    omp_set_num_threads(num_threads);

    std::vector<long long> area(nz, 0);
    std::vector<int> bw(nz, 0), bh(nz, 0);
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < nz; z++) {
        const unsigned char* s = trunk.data() + (long long)z * plane;
        int x0 = nx, y0 = ny, x1 = -1, y1 = -1; long long a = 0;
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
                if (s[(long long)y * nx + x]) {
                    a++;
                    if (x < x0) x0 = x; if (x > x1) x1 = x;
                    if (y < y0) y0 = y; if (y > y1) y1 = y;
                }
        area[z] = a;
        bw[z] = x1 >= x0 ? x1 - x0 + 1 : 0;
        bh[z] = y1 >= y0 ? y1 - y0 + 1 : 0;
    }

    // Median area of non-empty slices = the "normal" trunk cross-section.
    std::vector<long long> nz_areas;
    for (int z = 0; z < nz; z++) if (area[z] > 0) nz_areas.push_back(area[z]);
    if (nz_areas.empty()) return;
    std::sort(nz_areas.begin(), nz_areas.end());
    long long med = nz_areas[nz_areas.size() / 2];

    // A leaked slice: area >> median (sudden explosion) AND either a sprawling
    // bbox (spans half the slice) or low bbox-fill (octopus, not compact disk).
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < nz; z++) {
        if (area[z] == 0) continue;
        long long bbox_area = (long long)bw[z] * bh[z];
        double fill = bbox_area > 0 ? (double)area[z] / (double)bbox_area : 0.0;
        bool huge   = area[z] > 8 * med + 50;
        bool sprawl = (bw[z] > nx / 2 || bh[z] > ny / 2);
        if (huge && (sprawl || fill < 0.5)) {
            unsigned char* s = trunk.data() + (long long)z * plane;
            std::fill(s, s + plane, (unsigned char)0);
        }
    }
}

void downsample2x_avg(const float* hu, int nx, int ny, int nz,
                      std::vector<float>& out, int& dnx, int& dny, int& dnz,
                      int num_threads) {
    dnx = nx / 2; dny = ny / 2; dnz = nz / 2;
    out.assign((size_t)dnx * dny * dnz, 0.f);
    long long dplane = (long long)dnx * dny;
    long long plane = (long long)nx * ny;
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < dnz; z++)
        for (int y = 0; y < dny; y++)
            for (int x = 0; x < dnx; x++) {
                float s = 0;
                for (int dz = 0; dz < 2; dz++)
                  for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++)
                      s += hu[(long long)(2*z+dz)*plane + (long long)(2*y+dy)*nx + (2*x+dx)];
                out[(long long)z*dplane + (long long)y*dnx + x] = s * 0.125f;
            }
}

namespace {
// One line of binary box-dilation by `margin`: dst[j] = 1 iff any src within
// distance margin. Two passes (forward/backward nearest-set distance), O(L).
inline void dilate_line(const unsigned char* src, unsigned char* dst,
                        long long base, int L, long long stride, int margin) {
    int d = margin + 1;
    long long off = base;
    // forward: distance back to the previous set pixel (capped at margin+1)
    static thread_local std::vector<int> fwd;
    if ((int)fwd.size() < L) fwd.resize(L);
    for (int j = 0; j < L; j++, off += stride) {
        if (src[off]) d = 0; else if (d <= margin) d++;
        fwd[j] = d;
    }
    d = margin + 1;
    off = base + (long long)(L - 1) * stride;
    for (int j = L - 1; j >= 0; j--, off -= stride) {
        if (src[off]) d = 0; else if (d <= margin) d++;
        int dd = fwd[j] < d ? fwd[j] : d;
        dst[off] = (dd <= margin) ? 1 : 0;
    }
}
} // namespace

std::vector<unsigned char> upsample_dilate(const std::vector<unsigned char>& coarse,
                                           int dnx, int dny, int dnz,
                                           int nx, int ny, int nz,
                                           int margin, int num_threads) {
    long long N = (long long)nx * ny * nz;
    long long plane = (long long)nx * ny;
    long long dplane = (long long)dnx * dny;
    std::vector<unsigned char> up(N, 0);
    omp_set_num_threads(num_threads);

    // Nearest-neighbour upsample (each coarse voxel -> a 2x2x2 block).
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < nz; z++) {
        int cz = z / 2; if (cz >= dnz) cz = dnz - 1;
        for (int y = 0; y < ny; y++) {
            int cy = y / 2; if (cy >= dny) cy = dny - 1;
            for (int x = 0; x < nx; x++) {
                int cx = x / 2; if (cx >= dnx) cx = dnx - 1;
                up[(long long)z*plane + (long long)y*nx + x] =
                    coarse[(long long)cz*dplane + (long long)cy*dnx + cx];
            }
        }
    }
    if (margin > 0) dilate3d(up, nx, ny, nz, margin, num_threads);
    return up;
}

// V15: morphological closing = dilate(M, r) then erode(M, r). The erosion is
// implemented via the complement trick (erode(M) = !dilate(!M)), so we reuse
// the existing separable O(N)/axis dilate3d twice. Net effect on a binary
// mask: 1-voxel-thin internal gaps get filled (the dilation merges across the
// gap and the erosion can't reopen what's now a contiguous region), while
// the outer boundary returns to roughly its original shape. The cost is one
// scratch buffer of the same size as `mask`.
void close3d(std::vector<unsigned char>& mask, int nx, int ny, int nz,
             int margin, int num_threads) {
    if (margin <= 0) return;
    // (1) dilate in place
    dilate3d(mask, nx, ny, nz, margin, num_threads);
    // (2) erode via complement
    const size_t N = mask.size();
    std::vector<unsigned char> comp(N);
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (size_t i = 0; i < N; i++) comp[i] = mask[i] ? 0 : 1;
    dilate3d(comp, nx, ny, nz, margin, num_threads);
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (size_t i = 0; i < N; i++) mask[i] = comp[i] ? 0 : 1;
}

// V17 HYBRID: AI-assisted traditional region growing. See header doc.
//
// The architectural intent: the U-Net acts as the *register-bound hard gate*
// for a traditional 3D wavefront region-grower. Where the AI is confident
// (prob > core_thr, default 0.5) we trust it directly. Where the AI is
// uncertain (prob in (prob_gate, core_thr]) we let traditional intensity-
// gated connectivity decide -- but only voxels reachable from the AI's
// confident core can be claimed. Where the AI says "no airway" (prob <=
// prob_gate, default 0.1) the candidate is zero so the BFS *physically*
// cannot expand into it. No leak rule, no safety net needed -- the leak
// surface IS the gate.
//
// This is exactly the firmware safety net story: AI provides a high-level
// probability map; traditional algorithm walks it deterministically with
// hardware-friendly per-voxel checks (one float compare for HU, one for
// prob, no branches in the inner loop), bounded above and below.
RegionGrowResult hybrid_grow(const float* hu, const float* prob,
                             const std::vector<unsigned char>& hardset,
                             int nx, int ny, int nz,
                             float hu_lo, float hu_hi, float prob_gate,
                             const BBox& roi, int num_threads) {
    const long long N = (long long)nx * ny * nz;
    // (1) Build the AI-gated candidate region. Vectorisable, no branches
    //     other than the prob_gate / HU pair (`omp simd` would help on
    //     AVX-512; here we just rely on the compiler).
    std::vector<unsigned char> candidate((size_t)N, 0);
    long long n_core = 0, n_uncertain = 0;
    #pragma omp parallel for schedule(static) num_threads(num_threads) \
            reduction(+:n_core,n_uncertain)
    for (long long i = 0; i < N; i++) {
        if (hardset[i]) {
            candidate[i] = 1;
            n_core++;
        } else {
            // Uncertainty band: AI prob > floor AND HU within airway lumen.
            float p = prob[i];
            float v = hu[i];
            if (p > prob_gate && v >= hu_lo && v <= hu_hi) {
                candidate[i] = 1;
                n_uncertain++;
            }
        }
    }
    fprintf(stderr,
            "[v17] candidate built: core=%lld  uncertainty=%lld  total=%lld voxels "
            "(hu in [%.0f,%.0f], prob > %.2f)\n",
            n_core, n_uncertain, n_core + n_uncertain,
            hu_lo, hu_hi, prob_gate);

    // (2) Seed from central axis of the AI core. The seeds anchor the BFS
    //     to the confident centre of the airway tree; orphan blobs in the
    //     uncertainty zone (vessel false-positives etc.) are unreachable
    //     and stay out of the final mask.
    std::vector<Seed> seeds = central_axis_seeds_robust(
            hardset.data(), nx, ny, nz, roi);
    if (seeds.empty()) {
        // Degenerate case: no central seed inside the AI core. Fall back to
        // the raw core -- this only happens when the network gives almost
        // nothing on a case, where region growing into the uncertainty zone
        // would be guessing.
        RegionGrowResult rg;
        rg.mask = hardset;
        rg.airway_voxels   = n_core;
        rg.interior_voxels = n_core + n_uncertain;
        rg.num_seeds = 0;
        return rg;
    }

    // (3) Parallel multi-seed BFS within the AI-gated candidate. Lock-free
    //     atomic voxel claiming (multiseed_region_grow_v8 internals) handles
    //     the load imbalance of asymmetric airway branches; OpenMP dynamic
    //     scheduling over the seed list keeps short-branch threads from
    //     stalling long-branch threads.
    RegionGrowResult rg = multiseed_region_grow_v8(
            candidate.data(), nx, ny, nz, seeds, num_threads);
    rg.interior_voxels = n_core + n_uncertain;   // expose candidate size in print
    return rg;
}

void dilate3d(std::vector<unsigned char>& mask, int nx, int ny, int nz,
              int margin, int num_threads) {
    if (margin <= 0) return;
    long long N = (long long)nx * ny * nz;
    long long plane = (long long)nx * ny;
    omp_set_num_threads(num_threads);

    // Separable 3D binary box-dilation in O(N) per axis (independent of radius)
    // via a two-pass nearest-set-pixel distance along each line: a voxel is set
    // iff a set pixel lies within `margin` either forward or backward. Exact for
    // binary dilation, avoiding the O(N*(2r+1)) max-filter cost.
    std::vector<unsigned char> tmp(N);
    #pragma omp parallel for schedule(static) collapse(2)
    for (int z = 0; z < nz; z++)
      for (int y = 0; y < ny; y++)
        dilate_line(mask.data(), tmp.data(), (long long)z*plane + (long long)y*nx, nx, 1, margin);
    #pragma omp parallel for schedule(static) collapse(2)
    for (int z = 0; z < nz; z++)
      for (int x = 0; x < nx; x++)
        dilate_line(tmp.data(), mask.data(), (long long)z*plane + x, ny, nx, margin);
    #pragma omp parallel for schedule(static)
    for (long long xy = 0; xy < plane; xy++)
        dilate_line(mask.data(), tmp.data(), xy, nz, plane, margin);
    mask.swap(tmp);
}

TracheaCandidate rescue_threshold_sweep(const float* hu, int nx, int ny, int nz,
                                        const BBox& body, int num_threads) {
    TracheaCandidate r;
    long long plane = (long long)nx * ny;
    omp_set_num_threads(num_threads);

    // Search window: upper 40% of the body in Z (cervical airway is here even
    // when the chest scan extends well below the carina), central 40% in XY.
    // Restricting the sweep to this window keeps the 3D connected-components
    // pass O(2.5% of the volume) and avoids labelling lung parenchyma when the
    // threshold is high.
    BBox b = body;
    if (!b.valid) { b.x0=0; b.y0=0; b.z0=0; b.x1=nx-1; b.y1=ny-1; b.z1=nz-1; }
    // The trachea enters at the *low Z* end in head-first scans (NIfTI's +Z is
    // typically superior). We don't know orientation here, so take BOTH ends
    // and let the geometry filter pick the right one. In practice we take the
    // upper 40% from each end -> top of dataset (z near nz-1) AND bottom.
    // Implementation: search the cervical-side window directly. Most AIIB23
    // cases are oriented with the trachea at low Z; the geometry filter keeps
    // false matches out, so over-inclusion at the boundary is safe.
    int z_span = b.z1 - b.z0 + 1;
    int sz0 = b.z0;
    int sz1 = b.z0 + std::max(20, (int)(0.40f * z_span)) - 1;
    if (sz1 > b.z1) sz1 = b.z1;
    int scx = (b.x0 + b.x1) / 2;
    int scy = (b.y0 + b.y1) / 2;
    int hwx = std::max(40, (int)(0.20f * (b.x1 - b.x0 + 1)));
    int hwy = std::max(40, (int)(0.20f * (b.y1 - b.y0 + 1)));
    int sx0 = std::max(0, scx - hwx);
    int sx1 = std::min(nx - 1, scx + hwx);
    int sy0 = std::max(0, scy - hwy);
    int sy1 = std::min(ny - 1, scy + hwy);
    int Wx = sx1 - sx0 + 1, Wy = sy1 - sy0 + 1, Wz = sz1 - sz0 + 1;
    long long Wplane = (long long)Wx * Wy;
    long long Wsize = Wplane * Wz;

    std::vector<unsigned char> mask((size_t)Wsize);
    std::vector<int> label((size_t)Wsize);
    std::vector<long long> stk;
    stk.reserve(1 << 16);

    // Sweep the upper end of the HU range. Step 25 HU is fine enough to land
    // close to the lumen floor for any per-case drift we've seen (<= ~250 HU).
    // We stop the first time the geometry filter accepts a component -- with
    // a sweep direction from low HU -> high HU, that's the lowest threshold
    // that resolves the trachea as a distinct connected component, which is
    // also the threshold least contaminated by surrounding tissue.
    for (float thr = -950.f; thr <= -550.f; thr += 25.f) {
        #pragma omp parallel for schedule(static)
        for (int z = 0; z < Wz; z++) {
            for (int y = 0; y < Wy; y++) {
                long long base_g = (long long)(z + sz0) * plane
                                 + (long long)(y + sy0) * nx + sx0;
                long long base_l = (long long)z * Wplane + (long long)y * Wx;
                for (int x = 0; x < Wx; x++)
                    mask[(size_t)(base_l + x)] = (hu[base_g + x] <= thr) ? 1 : 0;
            }
        }

        std::fill(label.begin(), label.end(), 0);
        int next_label = 1;
        std::vector<long long> sz_, xs_, ys_, zs_;
        std::vector<int> zmin_, zmax_, xmin_, xmax_, ymin_, ymax_;
        for (long long start = 0; start < Wsize; start++) {
            if (!mask[(size_t)start] || label[(size_t)start]) continue;
            int L = next_label++;
            sz_.push_back(0); xs_.push_back(0); ys_.push_back(0); zs_.push_back(0);
            zmin_.push_back(Wz); zmax_.push_back(-1);
            xmin_.push_back(Wx); xmax_.push_back(-1);
            ymin_.push_back(Wy); ymax_.push_back(-1);
            label[(size_t)start] = L;
            stk.clear(); stk.push_back(start);
            while (!stk.empty()) {
                long long cur = stk.back(); stk.pop_back();
                int x = cur % Wx;
                int y = (cur / Wx) % Wy;
                int z = cur / Wplane;
                int idx = L - 1;
                sz_[idx]++; xs_[idx] += x; ys_[idx] += y; zs_[idx] += z;
                if (z < zmin_[idx]) zmin_[idx] = z;
                if (z > zmax_[idx]) zmax_[idx] = z;
                if (x < xmin_[idx]) xmin_[idx] = x;
                if (x > xmax_[idx]) xmax_[idx] = x;
                if (y < ymin_[idx]) ymin_[idx] = y;
                if (y > ymax_[idx]) ymax_[idx] = y;
                const int dx[6]={1,-1,0,0,0,0}, dy[6]={0,0,1,-1,0,0}, dz[6]={0,0,0,0,1,-1};
                for (int d = 0; d < 6; d++) {
                    int xx=x+dx[d], yy=y+dy[d], zz=z+dz[d];
                    if (xx<0||xx>=Wx||yy<0||yy>=Wy||zz<0||zz>=Wz) continue;
                    long long ni = (long long)zz*Wplane + (long long)yy*Wx + xx;
                    if (mask[(size_t)ni] && !label[(size_t)ni]) {
                        label[(size_t)ni] = L; stk.push_back(ni);
                    }
                }
            }
        }

        // Geometry filter. The trachea is the only structure in this window
        // that is simultaneously: (i) Z-elongated (long thin tube), (ii) XY
        // compact (< ~60 voxels diameter), (iii) sized like a lumen (>= ~200
        // voxels, not the global background air which floods the box), (iv)
        // near the local X/Y centre. We score with aspect * sqrt(size) and
        // penalise distance from the window centre.
        int best = -1; double best_score = 0;
        long long size_cap = Wsize / 3;   // background-air guard
        // Geometric bounds for "this is a trachea-shaped tube":
        //   s >= 200             (must be a real structure, not noise)
        //   Lz >= 8              (must span several axial slices)
        //   Dxy <= 60            (up to ~3-4 cm including a halo of bronchi)
        //   aspect = Lz/Dxy >= 1.5  (clearly elongated along Z)
        // These admit short cervical-trachea-only components (case 82, Lz=45)
        // *and* longer trunk+bronchi components (case 124, Lz=152). The
        // downstream leak cap catches the rare case where a wide component
        // is accepted and the subsequent grow leaks into the lung.
        for (int i = 0; i < (int)sz_.size(); i++) {
            long long s = sz_[i];
            if (s < 200 || s > size_cap) continue;
            int Lz  = zmax_[i] - zmin_[i] + 1;
            int Dx  = xmax_[i] - xmin_[i] + 1;
            int Dy  = ymax_[i] - ymin_[i] + 1;
            int Dxy = Dx > Dy ? Dx : Dy;
            if (Lz < 8 || Dxy > 60) continue;
            double aspect = (double)Lz / (double)Dxy;
            if (aspect < 1.5) continue;
            double cx = (double)xs_[i] / (double)s;
            double cy = (double)ys_[i] / (double)s;
            double dc = fabs(cx - Wx * 0.5) + fabs(cy - Wy * 0.5);
            double score = aspect * sqrt((double)s) / (1.0 + 0.05 * dc);
            if (score > best_score) { best_score = score; best = i; }
        }
        if (best < 0) continue;

        int target = best + 1;
        // One seed per Z slice within this component, taken at the slice
        // centroid (most central voxel of the component on that slice).
        double ccx = (double)xs_[best] / (double)sz_[best];
        double ccy = (double)ys_[best] / (double)sz_[best];
        for (int z = zmin_[best]; z <= zmax_[best]; z++) {
            long long pick = -1; double bestd = 1e18;
            for (int y = ymin_[best]; y <= ymax_[best]; y++)
                for (int x = xmin_[best]; x <= xmax_[best]; x++) {
                    long long i = (long long)z * Wplane + (long long)y * Wx + x;
                    if (label[(size_t)i] != target) continue;
                    double d = (x - ccx)*(x - ccx) + (y - ccy)*(y - ccy);
                    if (d < bestd) { bestd = d; pick = i; }
                }
            if (pick >= 0) {
                int lx = pick % Wx;
                int ly = (pick / Wx) % Wy;
                int lz = pick / Wplane;
                r.seeds.push_back({lx + sx0, ly + sy0, lz + sz0});
            }
        }
        r.threshold = thr;
        r.voxels    = sz_[best];
        r.valid     = true;
        // Materialise the trachea component into a full-volume binary mask so
        // the caller can dilate it into a growth envelope.
        long long Nfull = (long long)nx * ny * nz;
        r.trunk_mask.assign((size_t)Nfull, 0);
        for (int z = zmin_[best]; z <= zmax_[best]; z++) {
            for (int y = ymin_[best]; y <= ymax_[best]; y++) {
                long long base_l = (long long)z * Wplane + (long long)y * Wx;
                long long base_g = (long long)(z + sz0) * plane
                                 + (long long)(y + sy0) * nx + sx0;
                for (int x = xmin_[best]; x <= xmax_[best]; x++) {
                    if (label[(size_t)(base_l + x)] == target)
                        r.trunk_mask[(size_t)(base_g + x)] = 1;
                }
            }
        }
        int Lz  = zmax_[best] - zmin_[best] + 1;
        int Dx  = xmax_[best] - xmin_[best] + 1;
        int Dy  = ymax_[best] - ymin_[best] + 1;
        printf("[v6-rescue] sweep accepted thr=%.0f HU  trachea: %lld vox  "
               "Lz=%d  Dxy=%dx%d  aspect=%.2f  seeds=%zu\n",
               thr, sz_[best], Lz, Dx, Dy,
               (double)Lz / (double)std::max(Dx, Dy), r.seeds.size());
        return r;
    }
    printf("[v6-rescue] sweep found no trachea-shaped component in window "
           "x[%d,%d] y[%d,%d] z[%d,%d]\n", sx0, sx1, sy0, sy1, sz0, sz1);
    return r;
}

RegionGrowResult wavefront_region_grow(const float* hu, const float* vness,
                                       const unsigned char* hardset,
                                       const unsigned char* roi,
                                       int nx, int ny, int nz,
                                       const std::vector<Seed>& seeds,
                                       const WavefrontParams& p, int num_threads) {
    (void)num_threads;
    long long N = (long long)nx * ny * nz;
    long long plane = (long long)nx * ny;
    RegionGrowResult res;
    res.mask.assign(N, 0);
    res.num_seeds = (int)seeds.size();
    // result state codes:
    //   0 = free, 1 = committed, 2 = pending, 3 = rolled-back-relaxed
    // (v4: state 3 prevents re-trying the relaxed path on a leaked voxel; only
    // hardset/definite-airway voxels can reclaim it.)
    unsigned char* result = res.mask.data();

    long long hardset_total = 0;
    if (hardset) {
        #pragma omp parallel for schedule(static) reduction(+:hardset_total)
        for (long long i = 0; i < N; i++) hardset_total += (hardset[i] && (!roi || roi[i])) ? 1 : 0;
    }
    long long relaxed_cap = (long long)(p.leak_ratio * (double)hardset_total) + 5000;
    long long relaxed_added = 0;
    bool relaxed_off = false;

    // v4: per-level relaxed HU ceiling table. Hoists the (L / max_level) divide
    // out of the per-voxel hot path; ~bounds-check free since the table is
    // sized for any plausible BFS depth.
    int max_levels_tabled = std::max(p.max_level, 1) + 4096;
    std::vector<float> thr_by_level((size_t)max_levels_tabled + 1);
    {
        float span = p.thr_periph - p.thr_trunk;
        float inv_maxL = 1.0f / (float)std::max(p.max_level, 1);
        for (int L = 0; L <= max_levels_tabled; L++) {
            float t = (float)L * inv_maxL; if (t > 1.f) t = 1.f;
            thr_by_level[(size_t)L] = p.thr_trunk + t * span;
        }
    }
    const float vmin = p.vmin;
    const float hu_abs_max = p.hu_abs_max;  // v5 anatomical brake

    std::vector<long long> frontier, tentative, kept;
    frontier.reserve(1 << 16); tentative.reserve(1 << 16);
    for (const Seed& s : seeds) {
        long long i = (long long)s.z * plane + (long long)s.y * nx + s.x;
        if (result[i] != 0) continue;
        bool ok;
        if (roi && !roi[i]) ok = false;
        else if (hardset && hardset[i]) ok = true;
        else ok = (hu[i] <= thr_by_level[0] && hu[i] <= hu_abs_max
                   && vness[i] >= vmin);
        if (ok) { result[i] = 1; frontier.push_back(i); }
    }

    const int dx[6]={1,-1,0,0,0,0}, dy[6]={0,0,1,-1,0,0}, dz[6]={0,0,0,0,1,-1};
    double recent[5] = {0,0,0,0,0}; int nrec = 0;
    long long n_leak_layers = 0;

    for (int L = 1; !frontier.empty(); L++) {
        tentative.clear();
        const float thr_L = thr_by_level[(size_t)std::min(L, max_levels_tabled)];

        for (long long cur : frontier) {
            int x = cur % nx, y = (cur / nx) % ny, z = cur / plane;
            for (int d = 0; d < 6; d++) {
                int xx=x+dx[d], yy=y+dy[d], zz=z+dz[d];
                if (xx<0||xx>=nx||yy<0||yy>=ny||zz<0||zz>=nz) continue;
                long long ni = (long long)zz*plane + (long long)yy*nx + xx;
                unsigned char st = result[ni];
                // Common case (st==0) falls straight through; the rolled-back
                // (st==3) path is rare and only re-enters via hardset.
                if (st && !(st == 3 && hardset && hardset[ni])) continue;
                if (roi && !roi[ni]) continue;
                if (hardset && hardset[ni]) {
                    // hardset path: always accept
                } else if (relaxed_off || hu[ni] > thr_L || vness[ni] < vmin
                           || hu[ni] > hu_abs_max) {
                    // v5: hu_abs_max is the *absolute* air ceiling. Vessels
                    // are tubular (Frangi happy) but soft-tissue (HU ~ 0), so
                    // without this gate the relaxed wavefront walks down them.
                    continue;
                }
                result[ni] = 2; tentative.push_back(ni);
            }
        }

        double baseline = 0;
        if (nrec > 0) { for (int i=0;i<nrec;i++) baseline += recent[i]; baseline /= nrec; }
        long long sz = (long long)tentative.size();
        bool explode = (sz > p.leak_base) && (baseline <= 0 || sz > p.leak_factor * baseline);

        if (explode) {
            n_leak_layers++;
            kept.clear();
            for (long long ni : tentative) {
                if (hardset && hardset[ni]) { result[ni] = 1; kept.push_back(ni); }
                else result[ni] = 3;          // v4: mark as rejected, not free
            }
            tentative.swap(kept);
        } else {
            long long added_here = 0;
            for (long long ni : tentative) {
                result[ni] = 1;
                if (!(hardset && hardset[ni])) added_here++;
            }
            relaxed_added += added_here;
            if (!relaxed_off && relaxed_added > relaxed_cap) relaxed_off = true;
        }

        recent[nrec % 5] = (double)tentative.size();
        if (nrec < 5) nrec++;
        frontier.swap(tentative);
    }

    // Collapse {1,3} -> {1,0} so the output mask is binary.
    long long aw = 0;
    #pragma omp parallel for schedule(static) reduction(+:aw)
    for (long long i = 0; i < N; i++) {
        unsigned char v = result[i];
        if (v == 1) aw++;
        else if (v) result[i] = 0;
    }
    res.airway_voxels = aw;
    res.interior_voxels = aw;
    if (n_leak_layers || relaxed_off)
        printf("[wavefront] rolled-back layers=%lld  relaxed-path %s (added=%lld cap=%lld)\n",
               n_leak_layers, relaxed_off ? "DISABLED(leak guard)" : "ok",
               relaxed_added, relaxed_cap);
    return res;
}

RegionGrowResult multiseed_region_grow(const unsigned char* interior,
                                       int nx, int ny, int nz,
                                       const std::vector<Seed>& seeds,
                                       int num_threads) {
    long long N = (long long)nx * ny * nz;
    long long plane = (long long)nx * ny;
    RegionGrowResult res;
    res.mask.assign(N, 0);
    res.num_seeds = (int)seeds.size();

    unsigned char* result = res.mask.data();
    omp_set_num_threads(num_threads);

    long long inter = 0;
    #pragma omp parallel for schedule(static) reduction(+:inter)
    for (long long i = 0; i < N; i++) inter += interior[i];
    res.interior_voxels = inter;

    // Each seed grows its own BFS; voxels are claimed atomically so the union
    // is computed lock-free with no per-seed mask duplication.
    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < (int)seeds.size(); s++) {
        long long start = lin(seeds[s].x, seeds[s].y, seeds[s].z, nx, ny);
        if (!interior[start]) continue;
        unsigned char prev = __atomic_exchange_n(&result[start], (unsigned char)1, __ATOMIC_RELAXED);
        if (prev) continue;  // already claimed by another seed's growth

        std::vector<long long> frontier;
        frontier.reserve(1 << 16);
        frontier.push_back(start);
        const int dx[6]={1,-1,0,0,0,0}, dy[6]={0,0,1,-1,0,0}, dz[6]={0,0,0,0,1,-1};

        while (!frontier.empty()) {
            long long cur = frontier.back(); frontier.pop_back();
            int x = cur % nx, y = (cur / nx) % ny, z = cur / plane;
            for (int d = 0; d < 6; d++) {
                int xx=x+dx[d], yy=y+dy[d], zz=z+dz[d];
                if (xx<0||xx>=nx||yy<0||yy>=ny||zz<0||zz>=nz) continue;
                long long ni = lin(xx, yy, zz, nx, ny);
                if (!interior[ni]) continue;
                unsigned char p = __atomic_exchange_n(&result[ni], (unsigned char)1, __ATOMIC_RELAXED);
                if (!p) frontier.push_back(ni);
            }
        }
    }

    long long aw = 0;
    #pragma omp parallel for schedule(static) reduction(+:aw)
    for (long long i = 0; i < N; i++) aw += result[i];
    res.airway_voxels = aw;
    return res;
}

// ============================================================================
// V8: level-parallel BFS variants
// ============================================================================
// Design notes:
//  - One global frontier per level (not per-seed). All threads expand the same
//    layer concurrently; each thread accumulates a thread-local "next" vector
//    and merges it into the global next-frontier at the end of the level via
//    an offset reservation (single critical section, O(threads)).
//  - Voxel claiming uses __atomic_compare_exchange_n so only one thread wins
//    when multiple frontier voxels expand to the same neighbour in the same
//    level (common at branch points).
//  - For the wavefront variant, leak detection (size > leak_factor * baseline)
//    is computed AFTER the parallel expansion, before commit/rollback. The
//    rollback path is also parallelized — each tentative voxel is written
//    independently. The recent[] sliding window stays in sync because it's
//    updated by the main loop, not inside the parallel region.

RegionGrowResult multiseed_region_grow_v8(const unsigned char* interior,
                                          int nx, int ny, int nz,
                                          const std::vector<Seed>& seeds,
                                          int num_threads) {
    long long N = (long long)nx * ny * nz;
    long long plane = (long long)nx * ny;
    RegionGrowResult res;
    res.mask.assign(N, 0);
    res.num_seeds = (int)seeds.size();
    unsigned char* result = res.mask.data();
    omp_set_num_threads(num_threads);

    long long inter = 0;
    #pragma omp parallel for schedule(static) reduction(+:inter)
    for (long long i = 0; i < N; i++) inter += interior[i];
    res.interior_voxels = inter;

    // Initial frontier: all seeds that land on interior voxels. atomic-claim
    // them so duplicates collapse.
    std::vector<long long> frontier;
    frontier.reserve(seeds.size());
    for (const Seed& s : seeds) {
        long long idx = lin(s.x, s.y, s.z, nx, ny);
        if (!interior[idx]) continue;
        unsigned char prev = __atomic_exchange_n(&result[idx], (unsigned char)1, __ATOMIC_RELAXED);
        if (!prev) frontier.push_back(idx);
    }

    const int dx[6]={1,-1,0,0,0,0}, dy[6]={0,0,1,-1,0,0}, dz[6]={0,0,0,0,1,-1};
    std::vector<long long> next_frontier;
    next_frontier.reserve(1 << 18);

    while (!frontier.empty()) {
        next_frontier.clear();
        size_t frontier_sz = frontier.size();

        #pragma omp parallel
        {
            std::vector<long long> local;
            local.reserve(4096);
            #pragma omp for schedule(static, 1024) nowait
            for (long long i = 0; i < (long long)frontier_sz; i++) {
                long long cur = frontier[i];
                int x = cur % nx, y = (cur / nx) % ny, z = cur / plane;
                for (int d = 0; d < 6; d++) {
                    int xx=x+dx[d], yy=y+dy[d], zz=z+dz[d];
                    if (xx<0||xx>=nx||yy<0||yy>=ny||zz<0||zz>=nz) continue;
                    long long ni = lin(xx, yy, zz, nx, ny);
                    if (!interior[ni]) continue;
                    unsigned char prev = __atomic_exchange_n(&result[ni], (unsigned char)1, __ATOMIC_RELAXED);
                    if (!prev) local.push_back(ni);
                }
            }
            if (!local.empty()) {
                size_t off;
                #pragma omp critical (v8_msg_merge)
                { off = next_frontier.size(); next_frontier.resize(off + local.size()); }
                std::memcpy(next_frontier.data() + off, local.data(),
                            local.size() * sizeof(long long));
            }
        }
        frontier.swap(next_frontier);
    }

    long long aw = 0;
    #pragma omp parallel for schedule(static) reduction(+:aw)
    for (long long i = 0; i < N; i++) aw += result[i];
    res.airway_voxels = aw;
    return res;
}

RegionGrowResult wavefront_region_grow_v8(const float* hu, const float* vness,
                                          const unsigned char* hardset,
                                          const unsigned char* roi,
                                          int nx, int ny, int nz,
                                          const std::vector<Seed>& seeds,
                                          const WavefrontParams& p, int num_threads) {
    long long N = (long long)nx * ny * nz;
    long long plane = (long long)nx * ny;
    RegionGrowResult res;
    res.mask.assign(N, 0);
    res.num_seeds = (int)seeds.size();
    unsigned char* result = res.mask.data();
    omp_set_num_threads(num_threads);

    // state codes match v7:  0 free, 1 committed, 2 pending, 3 rolled-back
    long long hardset_total = 0;
    if (hardset) {
        #pragma omp parallel for schedule(static) reduction(+:hardset_total)
        for (long long i = 0; i < N; i++)
            hardset_total += (hardset[i] && (!roi || roi[i])) ? 1 : 0;
    }
    long long relaxed_cap = (long long)(p.leak_ratio * (double)hardset_total) + 5000;
    long long relaxed_added = 0;
    bool relaxed_off = false;

    int max_levels_tabled = std::max(p.max_level, 1) + 4096;
    std::vector<float> thr_by_level((size_t)max_levels_tabled + 1);
    {
        float span = p.thr_periph - p.thr_trunk;
        float inv_maxL = 1.0f / (float)std::max(p.max_level, 1);
        for (int L = 0; L <= max_levels_tabled; L++) {
            float t = (float)L * inv_maxL; if (t > 1.f) t = 1.f;
            thr_by_level[(size_t)L] = p.thr_trunk + t * span;
        }
    }
    const float vmin = p.vmin;
    const float hu_abs_max = p.hu_abs_max;

    // Seed initial frontier (sequential — tiny).
    std::vector<long long> frontier, tentative;
    frontier.reserve(1 << 16);
    tentative.reserve(1 << 18);
    for (const Seed& s : seeds) {
        long long i = (long long)s.z * plane + (long long)s.y * nx + s.x;
        if (result[i] != 0) continue;
        bool ok;
        if (roi && !roi[i]) ok = false;
        else if (hardset && hardset[i]) ok = true;
        else ok = (hu[i] <= thr_by_level[0] && hu[i] <= hu_abs_max
                   && vness[i] >= vmin);
        if (ok) { result[i] = 1; frontier.push_back(i); }
    }

    const int dx[6]={1,-1,0,0,0,0}, dy[6]={0,0,1,-1,0,0}, dz[6]={0,0,0,0,1,-1};
    double recent[5] = {0,0,0,0,0}; int nrec = 0;
    long long n_leak_layers = 0;

    for (int L = 1; !frontier.empty(); L++) {
        tentative.clear();
        const float thr_L = thr_by_level[(size_t)std::min(L, max_levels_tabled)];
        long long fsz = (long long)frontier.size();

        // Parallel expansion. Each thread accumulates locally then merges via
        // offset reservation. The atomic CAS on result[ni] ensures only one
        // thread claims each voxel even though many frontier voxels may share
        // a 6-neighbour (branch points / dense lumen).
        #pragma omp parallel
        {
            std::vector<long long> local;
            local.reserve(4096);
            #pragma omp for schedule(static, 1024) nowait
            for (long long i = 0; i < fsz; i++) {
                long long cur = frontier[i];
                int x = cur % nx, y = (cur / nx) % ny, z = cur / plane;
                for (int d = 0; d < 6; d++) {
                    int xx=x+dx[d], yy=y+dy[d], zz=z+dz[d];
                    if (xx<0||xx>=nx||yy<0||yy>=ny||zz<0||zz>=nz) continue;
                    long long ni = (long long)zz*plane + (long long)yy*nx + xx;
                    // First, snapshot the current state. If it's committed (1)
                    // or pending (2), skip. If rolled-back (3), allow ONLY via
                    // hardset path (matches v7 semantics).
                    unsigned char st = __atomic_load_n(&result[ni], __ATOMIC_RELAXED);
                    if (st == 1 || st == 2) continue;
                    if (st == 3 && !(hardset && hardset[ni])) continue;
                    if (roi && !roi[ni]) continue;
                    bool on_hardset = (hardset && hardset[ni]);
                    if (!on_hardset) {
                        if (relaxed_off) continue;
                        if (hu[ni] > thr_L || vness[ni] < vmin
                            || hu[ni] > hu_abs_max) continue;
                    }
                    // Atomic claim: transition (st) -> 2 (pending). Only one
                    // thread succeeds even on race.
                    unsigned char expected = st;
                    if (!__atomic_compare_exchange_n(&result[ni], &expected,
                                                    (unsigned char)2, false,
                                                    __ATOMIC_RELAXED,
                                                    __ATOMIC_RELAXED))
                        continue;
                    local.push_back(ni);
                }
            }
            if (!local.empty()) {
                size_t off;
                #pragma omp critical (v8_wf_merge)
                { off = tentative.size(); tentative.resize(off + local.size()); }
                std::memcpy(tentative.data() + off, local.data(),
                            local.size() * sizeof(long long));
            }
        }

        // Leak detection — identical to v7.
        double baseline = 0;
        if (nrec > 0) { for (int i=0;i<nrec;i++) baseline += recent[i]; baseline /= nrec; }
        long long sz = (long long)tentative.size();
        bool explode = (sz > p.leak_base) && (baseline <= 0 || sz > p.leak_factor * baseline);

        if (explode) {
            n_leak_layers++;
            // Parallel rollback: hardset voxels become committed (1); others
            // become rolled-back (3). Compact kept voxels via per-thread filter
            // + merge.
            std::vector<long long> kept;
            kept.reserve(tentative.size() / 4 + 16);
            #pragma omp parallel
            {
                std::vector<long long> local_kept;
                local_kept.reserve(1024);
                #pragma omp for schedule(static, 1024) nowait
                for (long long i = 0; i < (long long)tentative.size(); i++) {
                    long long ni = tentative[i];
                    if (hardset && hardset[ni]) {
                        result[ni] = 1;
                        local_kept.push_back(ni);
                    } else {
                        result[ni] = 3;
                    }
                }
                if (!local_kept.empty()) {
                    size_t off;
                    #pragma omp critical (v8_wf_kept)
                    { off = kept.size(); kept.resize(off + local_kept.size()); }
                    std::memcpy(kept.data() + off, local_kept.data(),
                                local_kept.size() * sizeof(long long));
                }
            }
            tentative.swap(kept);
        } else {
            // Parallel commit: all tentative voxels become committed; count
            // relaxed-path additions via reduction.
            long long added_here = 0;
            #pragma omp parallel for schedule(static, 1024) reduction(+:added_here)
            for (long long i = 0; i < (long long)tentative.size(); i++) {
                long long ni = tentative[i];
                result[ni] = 1;
                if (!(hardset && hardset[ni])) added_here++;
            }
            relaxed_added += added_here;
            if (!relaxed_off && relaxed_added > relaxed_cap) relaxed_off = true;
        }

        recent[nrec % 5] = (double)tentative.size();
        if (nrec < 5) nrec++;
        frontier.swap(tentative);
    }

    // Collapse {1,3} -> {1,0}.
    long long aw = 0;
    #pragma omp parallel for schedule(static) reduction(+:aw)
    for (long long i = 0; i < N; i++) {
        unsigned char v = result[i];
        if (v == 1) aw++;
        else if (v) result[i] = 0;
    }
    res.airway_voxels = aw;
    res.interior_voxels = aw;
    if (n_leak_layers || relaxed_off)
        printf("[wavefront-v8] rolled-back layers=%lld  relaxed-path %s (added=%lld cap=%lld)\n",
               n_leak_layers, relaxed_off ? "DISABLED(leak guard)" : "ok",
               relaxed_added, relaxed_cap);
    return res;
}
