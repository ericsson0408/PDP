// rg_roofline.cpp -- standalone replay of the V8 lock-free wavefront BFS
// (the region grower) on a REAL interior mask dumped from the pipeline, so we
// can measure thread scaling + achieved DRAM bandwidth under perf counters.
//
// The kernel below is a VERBATIM copy of multiseed_region_grow_v8() from
// src/region_growing.cpp (same atomics, same schedule(static,1024) nowait, same
// 6-neighbour probe), so the memory behaviour is identical to production.
//
//   g++ -O3 -fopenmp rg_roofline.cpp -o rg_roofline
//   OMP_NUM_THREADS=8 ./rg_roofline <dump_prefix> <repeats>
// Wrap with: perf stat -e LLC-load-misses,LLC-store-misses,cache-misses ...
//
// Prints per-thread-count: processed voxels, edges, mean BFS seconds.
#include <omp.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

struct Seed { int x, y, z; };
static inline long long lin(int x,int y,int z,int nx,int ny){
    return (long long)z*ny*nx + (long long)y*nx + x;
}

// ---- VERBATIM kernel (mirror of src/region_growing.cpp) ---------------------
static long long bfs_v8(const unsigned char* interior, unsigned char* result,
                        int nx,int ny,int nz, const std::vector<Seed>& seeds,
                        int num_threads) {
    long long N = (long long)nx*ny*nz, plane = (long long)nx*ny;
    std::memset(result, 0, (size_t)N);
    omp_set_num_threads(num_threads);

    std::vector<long long> frontier; frontier.reserve(seeds.size());
    for (const Seed& s : seeds) {
        long long idx = lin(s.x,s.y,s.z,nx,ny);
        if (idx<0||idx>=N||!interior[idx]) continue;
        unsigned char prev = __atomic_exchange_n(&result[idx],(unsigned char)1,__ATOMIC_RELAXED);
        if (!prev) frontier.push_back(idx);
    }
    const int dx[6]={1,-1,0,0,0,0}, dy[6]={0,0,1,-1,0,0}, dz[6]={0,0,0,0,1,-1};
    std::vector<long long> next_frontier; next_frontier.reserve(1<<18);
    while (!frontier.empty()) {
        next_frontier.clear();
        size_t frontier_sz = frontier.size();
        #pragma omp parallel
        {
            std::vector<long long> local; local.reserve(4096);
            #pragma omp for schedule(static,1024) nowait
            for (long long i=0;i<(long long)frontier_sz;i++){
                long long cur = frontier[i];
                int x=cur%nx, y=(cur/nx)%ny, z=cur/plane;
                for (int d=0;d<6;d++){
                    int xx=x+dx[d],yy=y+dy[d],zz=z+dz[d];
                    if (xx<0||xx>=nx||yy<0||yy>=ny||zz<0||zz>=nz) continue;
                    long long ni=lin(xx,yy,zz,nx,ny);
                    if (!interior[ni]) continue;
                    unsigned char prev=__atomic_exchange_n(&result[ni],(unsigned char)1,__ATOMIC_RELAXED);
                    if (!prev) local.push_back(ni);
                }
            }
            if (!local.empty()){
                size_t off;
                #pragma omp critical (merge)
                { off=next_frontier.size(); next_frontier.resize(off+local.size()); }
                std::memcpy(next_frontier.data()+off, local.data(), local.size()*sizeof(long long));
            }
        }
        frontier.swap(next_frontier);
    }
    long long aw=0;
    #pragma omp parallel for schedule(static) reduction(+:aw)
    for (long long i=0;i<N;i++) aw+=result[i];
    return aw;
}

int main(int argc, char** argv){
    if (argc<2){ fprintf(stderr,"usage: %s <dump_prefix> [repeats]\n",argv[0]); return 1; }
    std::string pre=argv[1];
    int repeats = argc>2 ? atoi(argv[2]) : 5;

    int nx,ny,nz; size_t nseed;
    { FILE* f=fopen((pre+".meta").c_str(),"r");
      if(!f){ fprintf(stderr,"no %s.meta\n",pre.c_str()); return 1; }
      fscanf(f,"%d %d %d %zu",&nx,&ny,&nz,&nseed); fclose(f); }
    long long N=(long long)nx*ny*nz;

    std::vector<unsigned char> interior((size_t)N);
    { FILE* f=fopen((pre+".interior").c_str(),"rb");
      if(!f){ fprintf(stderr,"no %s.interior\n",pre.c_str()); return 1; }
      size_t got=fread(interior.data(),1,(size_t)N,f); fclose(f);
      if(got!=(size_t)N){ fprintf(stderr,"short read\n"); return 1; } }
    std::vector<Seed> seeds;
    { FILE* f=fopen((pre+".seeds").c_str(),"r"); Seed s;
      while(f && fscanf(f,"%d %d %d",&s.x,&s.y,&s.z)==3) seeds.push_back(s);
      if(f) fclose(f); }

    std::vector<unsigned char> result((size_t)N);
    long long inter=0; for(long long i=0;i<N;i++) inter+=interior[i];
    int T = omp_get_max_threads();

    // Warm pages, then time `repeats` BFS runs (perf wraps the whole process; the
    // one-time load is amortised across repeats and reported separately by N=0).
    long long processed=0; double tsum=0, tbest=1e30;
    for (int r=0;r<repeats;r++){
        double t=omp_get_wtime();
        processed=bfs_v8(interior.data(),result.data(),nx,ny,nz,seeds,T);
        t=omp_get_wtime()-t; tsum+=t; if(t<tbest)tbest=t;
    }
    double tmean=tsum/repeats;
    fprintf(stderr,
      "[rg_roofline] threads=%d dims=%dx%dx%d inter=%lld processed=%lld edges=%lld "
      "bfs_mean_s=%.5f bfs_best_s=%.5f repeats=%d\n",
      T,nx,ny,nz,inter,processed,6*processed,tmean,tbest,repeats);
    // machine-readable line for the parser
    printf("%d %lld %lld %.6f %.6f\n", T, processed, 6*processed, tmean, tbest);
    return 0;
}
