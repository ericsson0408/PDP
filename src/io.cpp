#include "io.h"
#include "../third_party/libdeflate/libdeflate.h"
#include <zlib.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

Volume::~Volume() { delete[] data; }

// NIfTI-1 header field offsets (little-endian assumed; AIIB23 data is LE).
namespace {
constexpr int OFF_DIM      = 40;   // short[8]
constexpr int OFF_DATATYPE = 70;   // short
constexpr int OFF_BITPIX   = 72;   // short
constexpr int OFF_PIXDIM   = 76;   // float[8]
constexpr int OFF_VOXOFF   = 108;  // float
constexpr int OFF_SCLSLOPE = 112;  // float
constexpr int OFF_SCLINTER = 116;  // float
constexpr int OFF_CALMAX   = 124;  // float
constexpr int OFF_CALMIN   = 128;  // float
constexpr int OFF_MAGIC    = 344;  // char[4]

template <typename T> T rd(const unsigned char* h, int off) {
    T v; std::memcpy(&v, h + off, sizeof(T)); return v;
}
template <typename T> void wr(unsigned char* h, int off, T v) {
    std::memcpy(h + off, &v, sizeof(T));
}

// v4: read the whole .nii.gz into memory in one shot (sequential, fastest).
bool read_file_to_vec(const std::string& path, std::vector<uint8_t>& out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    if (std::fseek(fp, 0, SEEK_END) != 0) { std::fclose(fp); return false; }
    long sz = std::ftell(fp);
    if (sz < 0) { std::fclose(fp); return false; }
    std::fseek(fp, 0, SEEK_SET);
    out.resize((size_t)sz);
    bool ok = (std::fread(out.data(), 1, (size_t)sz, fp) == (size_t)sz);
    std::fclose(fp);
    return ok;
}

// V8: mmap-based read. Returns a pointer + length; the caller passes the
// pointer to libdeflate directly, saving one full file-size memcpy (and the
// associated heap alloc). The kernel maps the file's page cache straight into
// our address space; on warm cache this is essentially free.
//
// We MADV_SEQUENTIAL so the kernel prefetches aggressively for the
// libdeflate forward scan.
struct MmapFile {
    int fd = -1;
    void* ptr = nullptr;
    size_t sz = 0;
    ~MmapFile() {
        if (ptr) munmap(ptr, sz);
        if (fd >= 0) close(fd);
    }
};
bool mmap_file(const std::string& path, MmapFile& out) {
    out.fd = open(path.c_str(), O_RDONLY);
    if (out.fd < 0) return false;
    struct stat st;
    if (fstat(out.fd, &st) != 0) { close(out.fd); out.fd = -1; return false; }
    out.sz = (size_t)st.st_size;
    out.ptr = mmap(nullptr, out.sz, PROT_READ, MAP_PRIVATE, out.fd, 0);
    if (out.ptr == MAP_FAILED) {
        out.ptr = nullptr;
        close(out.fd); out.fd = -1;
        return false;
    }
    // Hint the kernel: we will scan forward, so prefetch.
    madvise(out.ptr, out.sz, MADV_SEQUENTIAL);
    return true;
}

// v4: one-shot gzip decode with libdeflate (AVX2/CLMUL kernels, ~2-3x faster
// than zlib). NIfTI CT typically compresses 4-6x; start with 8x and grow on
// INSUFFICIENT_SPACE so unusual cases still succeed.
bool gunzip_to_vec(const uint8_t* in, size_t in_sz, std::vector<uint8_t>& out) {
    libdeflate_decompressor* dec = libdeflate_alloc_decompressor();
    if (!dec) return false;
    size_t cap = std::max<size_t>(in_sz * 8, (size_t)1 << 20);
    for (int tries = 0; tries < 6; tries++) {
        out.resize(cap);
        size_t actual = 0;
        auto r = libdeflate_gzip_decompress(dec, in, in_sz, out.data(), cap, &actual);
        if (r == LIBDEFLATE_SUCCESS) {
            out.resize(actual);
            libdeflate_free_decompressor(dec);
            return true;
        }
        if (r != LIBDEFLATE_INSUFFICIENT_SPACE) {
            libdeflate_free_decompressor(dec);
            return false;
        }
        cap *= 4;
    }
    libdeflate_free_decompressor(dec);
    return false;
}
} // namespace

bool load_nifti(const std::string& path, Volume& vol) {
    // V8: mmap the file instead of fread+heap. libdeflate scans forward only,
    // so the mapped pages stream through L1/L2 once; no double-buffer.
    MmapFile mm;
    if (!mmap_file(path, mm)) {
        fprintf(stderr, "[io] cannot mmap %s\n", path.c_str()); return false;
    }
    // V16: detect gzip via the 0x1f 0x8b magic bytes (more robust than
    // filename extension). If the file is a plain .nii, we read the NIfTI
    // header directly out of the mmap'd buffer -- saving ~3-4 s/case of
    // libdeflate CPU work on big volumes. (Pre-decompress your .nii.gz
    // dataset once into /work and pass the .nii path here to enable.)
    std::vector<uint8_t> dec;
    const uint8_t* buf;
    size_t buf_sz;
    bool is_gz = (mm.sz >= 2
                  && ((const uint8_t*)mm.ptr)[0] == 0x1f
                  && ((const uint8_t*)mm.ptr)[1] == 0x8b);
    if (is_gz) {
        if (!gunzip_to_vec((const uint8_t*)mm.ptr, mm.sz, dec)) {
            fprintf(stderr, "[io] gunzip failed: %s\n", path.c_str()); return false;
        }
        buf = dec.data(); buf_sz = dec.size();
    } else {
        buf = (const uint8_t*)mm.ptr; buf_sz = mm.sz;
    }
    // mm released at function return after vol.data has been copied out.

    if (buf_sz < 348) {
        fprintf(stderr, "[io] short header: %s\n", path.c_str()); return false;
    }
    std::memcpy(vol.header, buf, 348);
    std::memset(vol.header + 348, 0, 4);

    short* dim   = (short*)(vol.header + OFF_DIM);
    short dtype  = rd<short>(vol.header, OFF_DATATYPE);
    float voxoff = rd<float>(vol.header, OFF_VOXOFF);
    float slope  = rd<float>(vol.header, OFF_SCLSLOPE);
    float inter  = rd<float>(vol.header, OFF_SCLINTER);
    float* pix   = (float*)(vol.header + OFF_PIXDIM);

    vol.nx = dim[1]; vol.ny = dim[2]; vol.nz = (dim[0] >= 3) ? dim[3] : 1;
    vol.dx = pix[1]; vol.dy = pix[2]; vol.dz = pix[3];
    if (slope == 0.0f) slope = 1.0f;

    long long N = vol.voxels();
    if (N <= 0) { fprintf(stderr, "[io] bad dims\n"); return false; }

    size_t data_off = (size_t)voxoff;
    vol.data = new float[N];

    // Convert raw datatype -> float32 in parallel, directly from the
    // decompressed buffer (no extra raw-typed allocation as in the zlib path).
    auto convert_typed = [&](auto* src) {
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < N; i++)
            vol.data[i] = (float)src[i] * slope + inter;
    };

    bool ok = true;
    switch (dtype) {
        case 2: {
            if (buf_sz < data_off + (size_t)N) { ok=false; break; }
            convert_typed((const uint8_t*)(buf + data_off)); break;
        }
        case 4: {
            if (buf_sz < data_off + (size_t)(2*N)) { ok=false; break; }
            convert_typed((const int16_t*)(buf + data_off)); break;
        }
        case 8: {
            if (buf_sz < data_off + (size_t)(4*N)) { ok=false; break; }
            convert_typed((const int32_t*)(buf + data_off)); break;
        }
        case 16: {
            if (buf_sz < data_off + (size_t)(4*N)) { ok=false; break; }
            const float* src = (const float*)(buf + data_off);
            #pragma omp parallel for schedule(static)
            for (long long i=0; i<N; i++) vol.data[i] = src[i]*slope + inter;
            break;
        }
        case 256: {
            if (buf_sz < data_off + (size_t)N) { ok=false; break; }
            convert_typed((const int8_t*)(buf + data_off)); break;
        }
        case 512: {
            if (buf_sz < data_off + (size_t)(2*N)) { ok=false; break; }
            convert_typed((const uint16_t*)(buf + data_off)); break;
        }
        default:
            fprintf(stderr, "[io] unsupported datatype %d\n", dtype); ok = false;
    }
    if (!ok) {
        fprintf(stderr, "[io] data read failed: %s\n", path.c_str());
        delete[] vol.data; vol.data = nullptr;
        return false;
    }
    printf("[load] %s %.1f MB via %s\n", path.c_str(),
           (double)buf_sz / (1<<20),
           is_gz ? "libdeflate" : "mmap (plain .nii)");
    return true;
}

void clamp_hu(float* data, long long N, float hu_min, float hu_max) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < N; i++) {
        float v = data[i];
        data[i] = v < hu_min ? hu_min : (v > hu_max ? hu_max : v);
    }
}

bool write_mask_nifti(const std::string& path, const Volume& ref,
                      const unsigned char* mask) {
    unsigned char hdr[352];
    std::memcpy(hdr, ref.header, 352);

    wr<short>(hdr, OFF_DATATYPE, 2);   // DT_UINT8
    wr<short>(hdr, OFF_BITPIX, 8);
    wr<float>(hdr, OFF_VOXOFF, 352.0f);
    wr<float>(hdr, OFF_SCLSLOPE, 1.0f);
    wr<float>(hdr, OFF_SCLINTER, 0.0f);
    wr<float>(hdr, OFF_CALMIN, 0.0f);
    wr<float>(hdr, OFF_CALMAX, 1.0f);
    std::memcpy(hdr + OFF_MAGIC, "n+1\0", 4);

    // Level-1 compression: the mask is sparse so the file stays small while
    // compression runs several times faster than the default level.
    gzFile f = gzopen(path.c_str(), "wb1");
    if (!f) { fprintf(stderr, "[io] cannot write %s\n", path.c_str()); return false; }
    gzwrite(f, hdr, 352);

    long long N = ref.voxels();
    const unsigned char* p = mask;
    while (N > 0) {
        unsigned chunk = (unsigned)(N > (1 << 30) ? (1 << 30) : N);
        if (gzwrite(f, p, chunk) != (int)chunk) { gzclose(f); return false; }
        p += chunk; N -= chunk;
    }
    gzclose(f);
    return true;
}

unsigned char* load_label_u8(const std::string& path, int& nx, int& ny, int& nz) {
    // V8: mmap path (saves ~150ms on warm cache for typical GT label).
    MmapFile mm;
    std::vector<uint8_t> dec;
    if (!mmap_file(path, mm)) return nullptr;
    if (!gunzip_to_vec((const uint8_t*)mm.ptr, mm.sz, dec)) return nullptr;
    if (dec.size() < 348) return nullptr;

    const unsigned char* header = dec.data();
    const short* dim   = (const short*)(header + OFF_DIM);
    short dtype        = rd<short>(header, OFF_DATATYPE);
    float voxoff       = rd<float>(header, OFF_VOXOFF);
    nx = dim[1]; ny = dim[2]; nz = (dim[0] >= 3) ? dim[3] : 1;
    long long N = (long long)nx * ny * nz;
    if (N <= 0) return nullptr;
    size_t data_off = (size_t)voxoff;

    auto* out = new unsigned char[N];
    bool ok = true;
    if (dtype == 2 || dtype == 256) {
        if (dec.size() < data_off + (size_t)N) ok = false;
        else std::memcpy(out, dec.data() + data_off, (size_t)N);
    } else if (dtype == 4) {
        if (dec.size() < data_off + (size_t)(2*N)) ok = false;
        else {
            const int16_t* src = (const int16_t*)(dec.data() + data_off);
            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < N; i++) out[i] = src[i] ? 1 : 0;
        }
    } else { ok = false; }
    if (!ok) { delete[] out; return nullptr; }
    return out;
}
