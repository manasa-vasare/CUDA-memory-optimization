// Step 3: Write the naive matrix-multiplication kernel
// Thread (row, col) owns one element of C.
// The grid is 2D with 16 x 16 blocks; dim3 block(16, 16) and dim3 grid((N+15)/16, (N+15)/16).

#ifndef BLK
#define BLK 16
#endif

// ---------- Kernel 3: NAIVE matrix multiplication C = A x B ----------
// Each thread computes one element of C using a loop of N iterations;
// every iteration performs 2 global-memory reads and 1 multiply-add.
// Memory-bound: no data is reused through shared memory.
__global__ void matMulNaive(const float *A, const float *B, float *C, int N)
{
    int row = blockIdx.y * BLK + threadIdx.y;
    int col = blockIdx.x * BLK + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < N; k++) // 2 global reads + 1 multiply-add per iteration
            sum += A[row * N + k] * B[k * N + col];

        C[row * N + col] = sum;
    }
}
