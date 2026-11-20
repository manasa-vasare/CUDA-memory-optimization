// Step 1: Create the file, headers and error-check macro
// TILE and BLK are the block dimensions for 2D kernels.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32 // transpose block = 32 x 32 threads
#define BLK 16  // matmul block = 16 x 16 threads

#define CHECK(call) do { cudaError_t _cuda_status = (call); \
    if (_cuda_status != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                cudaGetErrorString(_cuda_status), __FILE__, __LINE__); \
        exit(1); } } while (0)
