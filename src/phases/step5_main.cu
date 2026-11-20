// Step 5: Write main()
// Allocates host and device memory, copies inputs to GPU,
// launches and times each kernel, verifies results, and prints the summary.
// Default sizes: Transpose N=4096, Matmul N=1024.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#ifndef TILE
#define TILE 32
#endif
#ifndef BLK
#define BLK 16
#endif

// main allocates host and device memory, copies the input to the GPU,
// launches and times each kernel, copies the result back, verifies it, and prints the table.
int main(int argc, char **argv)
{
    int NT = 4096; // transpose size (NT x NT)
    int NM = 1024; // matmul size (NM x NM)

    if (argc > 1) NT = atoi(argv[1]);
    if (argc > 2) NM = atoi(argv[2]);

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s (compute capability %d.%d)\n",
           prop.name, prop.major, prop.minor);

    // ================= PART A: TRANSPOSE =================

    size_t bytesT = (size_t)NT * NT * sizeof(float);

    float *h_in = (float *)malloc(bytesT);
    float *h_out = (float *)malloc(bytesT);

    for (size_t i = 0; i < (size_t)NT * NT; i++)
        h_in[i] = (float)(i % 1000);

    float *d_in, *d_out;

    CHECK(cudaMalloc(&d_in, bytesT));
    CHECK(cudaMalloc(&d_out, bytesT));

    CHECK(cudaMemcpy(d_in, h_in, bytesT, cudaMemcpyHostToDevice));

    dim3 blockT(TILE, TILE);
    dim3 gridT((NT + TILE - 1) / TILE, (NT + TILE - 1) / TILE);

    float tN = timeTranspose(transposeNaive, gridT, blockT, d_in, d_out, NT, 20);
    CHECK(cudaMemcpy(h_out, d_out, bytesT, cudaMemcpyDeviceToHost));
    bool okN = checkTranspose(h_in, h_out, NT);

    float tC = timeTranspose(transposeCoalesced, gridT, blockT, d_in, d_out, NT, 20);
    CHECK(cudaMemcpy(h_out, d_out, bytesT, cudaMemcpyDeviceToHost));
    bool okC = checkTranspose(h_in, h_out, NT);

    double gbN = 2.0 * bytesT / (tN * 1e6); // read + write, GB/s
    double gbC = 2.0 * bytesT / (tC * 1e6);

    printf("\n=== PART A: Matrix transpose (N = %d) ===\n", NT);
    printf("Uncoalesced : %8.3f ms | %7.1f GB/s | %s\n", tN, gbN, okN ? "PASS" : "FAIL");
    printf("Coalesced   : %8.3f ms | %7.1f GB/s | %s\n", tC, gbC, okC ? "PASS" : "FAIL");
    printf("Speedup (uncoalesced / coalesced) = %.2fx\n", tN / tC);

    // ================= PART B: NAIVE MATMUL =================

    size_t bytesM = (size_t)NM * NM * sizeof(float);

    float *hA = (float *)malloc(bytesM);
    float *hB = (float *)malloc(bytesM);
    float *hC = (float *)malloc(bytesM);

    for (size_t i = 0; i < (size_t)NM * NM; i++) {
        hA[i] = rand() / (float)RAND_MAX;
        hB[i] = rand() / (float)RAND_MAX;
    }

    float *dA, *dB, *dC;

    CHECK(cudaMalloc(&dA, bytesM));
    CHECK(cudaMalloc(&dB, bytesM));
    CHECK(cudaMalloc(&dC, bytesM));

    CHECK(cudaMemcpy(dA, hA, bytesM, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytesM, cudaMemcpyHostToDevice));

    dim3 block(BLK, BLK);
    dim3 grid((NM + BLK - 1) / BLK, (NM + BLK - 1) / BLK);

    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s));
    CHECK(cudaEventCreate(&e));

    matMulNaive<<<grid, block>>>(dA, dB, dC, NM); // warm-up
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    int reps = 5;
    CHECK(cudaEventRecord(s));
    for (int r = 0; r < reps; r++)
        matMulNaive<<<grid, block>>>(dA, dB, dC, NM);
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, s, e));
    float tM = ms / reps;

    CHECK(cudaMemcpy(hC, dC, bytesM, cudaMemcpyDeviceToHost));
    bool okM = checkMatMul(hA, hB, hC, NM);

    double gflops = 2.0 * NM * NM * (double)NM / (tM * 1e6);

    printf("\n=== PART B: Naive matrix multiplication (N = %d) ===\n", NM);
    printf("Naive matmul: %8.3f ms | %7.1f GFLOPS | %s\n", tM, gflops, okM ? "PASS" : "FAIL");

    // ================= SUMMARY =================

    printf("\n=== SUMMARY ===\n");
    printf("Transpose uncoalesced : %.3f ms\n", tN);
    printf("Transpose coalesced    : %.3f ms\n", tC);
    printf("Naive matmul           : %.3f ms\n", tM);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(h_in);
    free(h_out);
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
