// Step 2: Write the two transpose kernels
// Note the differences: the coalesced kernel stages data in tile[][],
// calls __syncthreads(), and swaps block coordinates before writing.

#ifndef TILE
#define TILE 32
#endif

// ---------- Kernel 1: UNCOALESCED transpose (one element per thread) ----------
// Each thread copies one element. Consecutive threads read consecutive
// elements of a row (coalesced), but write column-wise with stride N (uncoalesced).
__global__ void transposeNaive(const float *in, float *out, int N)
{
    int x = blockIdx.x * TILE + threadIdx.x; // column index
    int y = blockIdx.y * TILE + threadIdx.y; // row index

    if (x < N && y < N)
        out[x * N + y] = in[y * N + x];      // read: row-wise (good), write: column-wise (stride N)
}

// ---------- Kernel 2: COALESCED transpose (shared-memory tile) ----------
// A block loads a 32 x 32 tile into shared memory with coalesced reads,
// synchronises, then writes the tile out transposed with coalesced writes.
// Tile is declared [TILE][TILE + 1]: the +1 padding eliminates shared-memory bank conflicts.
__global__ void transposeCoalesced(const float *in, float *out, int N)
{
    __shared__ float tile[TILE][TILE + 1];  // +1 padding avoids shared memory bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x]; // coalesced read from global memory

    __syncthreads();                                    // wait until tile is loaded by all threads

    x = blockIdx.y * TILE + threadIdx.x;                // swap block coordinates for transposed write
    y = blockIdx.x * TILE + threadIdx.y;

    if (x < N && y < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y]; // coalesced write to global memory
}
