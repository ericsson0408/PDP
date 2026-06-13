#include "mpi_partition.h"
#include <mpi.h>
#include <cstring>
#include <vector>

void compute_partition(int nz, int nprocs,
                       std::vector<int>& z_counts, std::vector<int>& z_displs) {
    z_counts.assign(nprocs, 0);
    z_displs.assign(nprocs, 0);
    int base = nz / nprocs, rem = nz % nprocs, off = 0;
    for (int r = 0; r < nprocs; r++) {
        z_counts[r] = base + (r < rem ? 1 : 0);  // spread remainder over first ranks
        z_displs[r] = off;
        off += z_counts[r];
    }
}

void scatter_slab(const float* full, float* local, int nx, int ny,
                  const std::vector<int>& z_counts,
                  const std::vector<int>& z_displs, int rank) {
    int nprocs = (int)z_counts.size();
    long long plane = (long long)nx * ny;
    std::vector<int> sc(nprocs), sd(nprocs);
    for (int r = 0; r < nprocs; r++) { sc[r] = z_counts[r] * (int)plane; sd[r] = z_displs[r] * (int)plane; }
    MPI_Scatterv(full, sc.data(), sd.data(), MPI_FLOAT,
                 local, z_counts[rank] * (int)plane, MPI_FLOAT,
                 0, MPI_COMM_WORLD);
}

void build_extended_with_halo(const float* local, int nx, int ny, int local_nz,
                              int halo, int rank, int nprocs,
                              std::vector<float>& ext) {
    long long plane = (long long)nx * ny;
    int ext_nz = local_nz + 2 * halo;
    ext.assign((size_t)ext_nz * plane, 0.0f);

    // Copy own slab into the centre.
    std::memcpy(ext.data() + (size_t)halo * plane, local,
                (size_t)local_nz * plane * sizeof(float));

    int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;          // smaller z
    int down = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL; // larger z

    std::vector<float> recv_top((size_t)halo * plane);    // ghost from `up`
    std::vector<float> recv_bot((size_t)halo * plane);    // ghost from `down`

    // Send my first `halo` slices up; receive my bottom ghost from `down`.
    MPI_Sendrecv(local, halo * (int)plane, MPI_FLOAT, up, 0,
                 recv_bot.data(), halo * (int)plane, MPI_FLOAT, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Send my last `halo` slices down; receive my top ghost from `up`.
    MPI_Sendrecv(local + (size_t)(local_nz - halo) * plane, halo * (int)plane, MPI_FLOAT, down, 1,
                 recv_top.data(), halo * (int)plane, MPI_FLOAT, up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Place top ghost (or replicate edge at global boundary).
    if (up != MPI_PROC_NULL)
        std::memcpy(ext.data(), recv_top.data(), (size_t)halo * plane * sizeof(float));
    else
        for (int h = 0; h < halo; h++)
            std::memcpy(ext.data() + (size_t)h * plane, local, plane * sizeof(float));

    // Place bottom ghost (or replicate edge).
    if (down != MPI_PROC_NULL)
        std::memcpy(ext.data() + (size_t)(halo + local_nz) * plane, recv_bot.data(),
                    (size_t)halo * plane * sizeof(float));
    else
        for (int h = 0; h < halo; h++)
            std::memcpy(ext.data() + (size_t)(halo + local_nz + h) * plane,
                        local + (size_t)(local_nz - 1) * plane, plane * sizeof(float));
}

void gather_mask(const unsigned char* local_mask, unsigned char* full,
                 int nx, int ny, const std::vector<int>& z_counts,
                 const std::vector<int>& z_displs, int rank) {
    int nprocs = (int)z_counts.size();
    long long plane = (long long)nx * ny;
    std::vector<int> rc(nprocs), rd(nprocs);
    for (int r = 0; r < nprocs; r++) { rc[r] = z_counts[r] * (int)plane; rd[r] = z_displs[r] * (int)plane; }
    MPI_Gatherv(local_mask, z_counts[rank] * (int)plane, MPI_UNSIGNED_CHAR,
                full, rc.data(), rd.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);
}
