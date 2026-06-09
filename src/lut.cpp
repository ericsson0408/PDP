#include "lut.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace {
constexpr int LUT_MAGIC = 0x4C555432;  // 'LUT2'

inline int ibin(float v, float lo, float hi, int n) {
    int b = (int)((v - lo) / (hi - lo) * n);
    return b < 0 ? 0 : (b >= n ? n - 1 : b);
}
} // namespace

void gradient_magnitude(const float* vol, float* grad, int nx, int ny, int nz) {
    long long plane = (long long)nx * ny;
    auto cl = [](int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); };
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                long long i = (long long)z*plane + (long long)y*nx + x;
                float gx = 0.5f*(vol[(long long)z*plane+(long long)y*nx+cl(x+1,0,nx-1)]
                               - vol[(long long)z*plane+(long long)y*nx+cl(x-1,0,nx-1)]);
                float gy = 0.5f*(vol[(long long)z*plane+(long long)cl(y+1,0,ny-1)*nx+x]
                               - vol[(long long)z*plane+(long long)cl(y-1,0,ny-1)*nx+x]);
                float gz = 0.5f*(vol[(long long)cl(z+1,0,nz-1)*plane+(long long)y*nx+x]
                               - vol[(long long)cl(z-1,0,nz-1)*plane+(long long)y*nx+x]);
                grad[i] = sqrtf(gx*gx + gy*gy + gz*gz);
            }
        }
    }
}

bool lut_train_accumulate(const std::string& counts_path,
                          const float* smoothed, const unsigned char* gt,
                          int nx, int ny, int nz, const Lut2D& spec) {
    int Ib = spec.Ibins, Gb = spec.Gbins;
    long long B = (long long)Ib * Gb;
    std::vector<double> airway(B, 0.0), total(B, 0.0);

    // Load existing counts (if any) so multiple cases accumulate.
    if (FILE* f = fopen(counts_path.c_str(), "rb")) {
        int magic, ib, gb; float imin, imax, gmax;
        if (fread(&magic,4,1,f)==1 && fread(&ib,4,1,f)==1 && fread(&gb,4,1,f)==1 &&
            fread(&imin,4,1,f)==1 && fread(&imax,4,1,f)==1 && fread(&gmax,4,1,f)==1 &&
            magic==LUT_MAGIC && ib==Ib && gb==Gb) {
            fread(airway.data(), sizeof(double), B, f);
            fread(total.data(),  sizeof(double), B, f);
        } else {
            fprintf(stderr, "[lut] existing counts incompatible, overwriting\n");
        }
        fclose(f);
    }

    long long N = (long long)nx * ny * nz;
    std::vector<float> grad(N);
    gradient_magnitude(smoothed, grad.data(), nx, ny, nz);

    for (long long i = 0; i < N; i++) {
        int bi = ibin(smoothed[i], spec.Imin, spec.Imax, Ib);
        int bg = ibin(grad[i], 0.f, spec.Gmax, Gb);
        long long b = (long long)bg * Ib + bi;
        total[b] += 1.0;
        if (gt[i]) airway[b] += 1.0;
    }

    FILE* f = fopen(counts_path.c_str(), "wb");
    if (!f) { fprintf(stderr, "[lut] cannot write %s\n", counts_path.c_str()); return false; }
    int magic = LUT_MAGIC;
    fwrite(&magic,4,1,f); fwrite(&Ib,4,1,f); fwrite(&Gb,4,1,f);
    fwrite(&spec.Imin,4,1,f); fwrite(&spec.Imax,4,1,f); fwrite(&spec.Gmax,4,1,f);
    fwrite(airway.data(), sizeof(double), B, f);
    fwrite(total.data(),  sizeof(double), B, f);
    fclose(f);

    double sa=0, st=0; for (long long b=0;b<B;b++){sa+=airway[b];st+=total[b];}
    printf("[lut-train] %s  airway=%.0f total=%.0f (this case added)\n",
           counts_path.c_str(), sa, st);
    return true;
}

bool lut_load(const std::string& counts_path, Lut2D& out, long long min_total) {
    FILE* f = fopen(counts_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "[lut] cannot open %s\n", counts_path.c_str()); return false; }
    int magic, ib, gb; float imin, imax, gmax;
    if (fread(&magic,4,1,f)!=1 || fread(&ib,4,1,f)!=1 || fread(&gb,4,1,f)!=1 ||
        fread(&imin,4,1,f)!=1 || fread(&imax,4,1,f)!=1 || fread(&gmax,4,1,f)!=1 ||
        magic != LUT_MAGIC) {
        fprintf(stderr, "[lut] bad LUT file\n"); fclose(f); return false;
    }
    out.Ibins = ib; out.Gbins = gb; out.Imin = imin; out.Imax = imax; out.Gmax = gmax;
    long long B = (long long)ib * gb;
    std::vector<double> airway(B), total(B);
    fread(airway.data(), sizeof(double), B, f);
    fread(total.data(),  sizeof(double), B, f);
    fclose(f);

    out.prob.assign(B, 0.f);
    long long active = 0;
    for (long long b = 0; b < B; b++) {
        if (total[b] >= min_total) { out.prob[b] = (float)(airway[b] / total[b]); active++; }
    }
    printf("[lut] loaded %dx%d  I[%.0f,%.0f] Gmax=%.0f  active_bins=%lld/%lld\n",
           ib, gb, imin, imax, gmax, active, B);
    return true;
}
