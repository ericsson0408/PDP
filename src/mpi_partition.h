#pragma once
#include <vector>

// Z-axis slab decomposition + halo (ghost) exchange for the 3D stencil.

// Slice counts/offsets per rank (handles uneven nz/nprocs).
void compute_partition(int nz, int nprocs,
                       std::vector<int>& z_counts, std::vector<int>& z_displs);

// Scatter a full float volume held on rank 0 into per-rank slabs.
//   full  : valid only on rank 0, size nx*ny*nz
//   local : receive buffer, size nx*ny*z_counts[rank]
void scatter_slab(const float* full, float* local, int nx, int ny,
                  const std::vector<int>& z_counts,
                  const std::vector<int>& z_displs, int rank);

// Build an extended slab = local slab + `halo` ghost slices on each Z side.
// Ghost data comes from neighbour ranks via MPI_Sendrecv; missing neighbours
// (global boundary) are filled by edge replication. Valid region begins at
// slice offset `halo` in `ext`.
void build_extended_with_halo(const float* local, int nx, int ny, int local_nz,
                              int halo, int rank, int nprocs,
                              std::vector<float>& ext);

// Gather per-rank uint8 mask slabs back into the full volume on rank 0.
void gather_mask(const unsigned char* local_mask, unsigned char* full,
                 int nx, int ny, const std::vector<int>& z_counts,
                 const std::vector<int>& z_displs, int rank);
