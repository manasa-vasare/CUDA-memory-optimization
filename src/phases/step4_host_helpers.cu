// Step 4: Add host helper functions (timing and verification)
// Kernel launches are asynchronous, so we time them with CUDA events.
// The first launch is a warm-up and is not timed.
// The transpose is repeated 20 times and the average is reported.
// Checkers compare GPU outputs with single-threaded CPU references.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#ifndef CHECK
#define CHECK(call) do { cudaError_t _cuda_status = (call); \
    if (_cuda_status != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                cudaGetErrorString(_cuda_status), __FILE__, __LINE__); \
        exit(1); } } while (0)
#endif

typedef void (*TKernel)(const float *, float *, int);

float timeTranspose(TKernel k, dim3 g, dim3 b,
                    const float *in, float *out, int N, int reps)
{
    cudaEvent_t s, e;

    CHECK(cudaEventCreate(&s));
    CHECK(cudaEventCreate(&e));

    CHECK(cudaMemset(out, 0, (size_t)N * N * sizeof(float)));

    // Warm-up launch (not timed)
    k<<<g, b>>>(in, out, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Timed repetitions
    CHECK(cudaEventRecord(s));

    for (int r = 0; r < reps; r++)
        k<<<g, b>>>(in, out, N);

    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, s, e));

    CHECK(cudaEventDestroy(s));
    CHECK(cudaEventDestroy(e));

    return ms / reps; // average launch time
}

bool checkTranspose(const float *in, const float *out, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (out[j * N + i] != in[i * N + j])
                return false;

    return true;
}

bool checkMatMul(const float *A, const float *B,
                 const float *C, int N)
{
    for (int t = 0; t < 20; t++) { // spot-check 20 random elements
        int i = rand() % N;
        int j = rand() % N;

        double ref = 0.0;

        for (int k = 0; k < N; k++)
            ref += (double)A[i * N + k] * B[k * N + j];

        if (fabs(ref - C[i * N + j]) > 1e-3 * fabs(ref) + 1e-3)
            return false;
    }

    return true;
}
