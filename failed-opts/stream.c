/* Compact OpenMP STREAM (Triad/Copy) to measure achievable DRAM bandwidth =
 * the "peak" line of the region-grow roofline. Arrays sized well past LLC.
 * Reports the standard STREAM "best rate" (MB/s) per kernel.
 *   cc -O3 -fopenmp -DN=80000000 stream.c -o stream && OMP_NUM_THREADS=8 ./stream
 */
#include <stdio.h>
#include <omp.h>
#ifndef N
#define N 80000000          /* 80M doubles/array -> 1.9 GB total, >> LLC */
#endif
static double a[N], b[N], c[N];
int main(void) {
    const double scalar = 3.0;
    #pragma omp parallel for
    for (long i = 0; i < N; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }
    double best_copy = 1e30, best_triad = 1e30;
    for (int r = 0; r < 10; r++) {
        double t = omp_get_wtime();
        #pragma omp parallel for
        for (long i = 0; i < N; i++) c[i] = a[i];
        t = omp_get_wtime() - t;
        if (t < best_copy) best_copy = t;

        t = omp_get_wtime();
        #pragma omp parallel for
        for (long i = 0; i < N; i++) a[i] = b[i] + scalar * c[i];
        t = omp_get_wtime() - t;
        if (t < best_triad) best_triad = t;
    }
    double gb_copy  = 2.0 * sizeof(double) * N / 1e9;   /* read b, write c  */
    double gb_triad = 3.0 * sizeof(double) * N / 1e9;   /* read b,c, write a */
    printf("threads=%d  Copy=%.1f GB/s  Triad=%.1f GB/s\n",
           omp_get_max_threads(), gb_copy / best_copy, gb_triad / best_triad);
    return 0;
}
