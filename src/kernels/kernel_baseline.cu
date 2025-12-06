#include <math.h>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

__global__ void baseline_copy_kernel(float *output, const float *input) {
  int x = TILE_DIM * blockIdx.x + threadIdx.x;
  int y = TILE_DIM * blockIdx.y + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    int index = (y + j) * width + x;
    output[index] = input[index];
  }
}

// --- Baseline Kernel (Naive, Unoptimized) --- 161 us ---
// it inherently coalesces reads but not writes

__global__ void baseline_transpose_kernel(float *output, const float *input) {

  int x = TILE_DIM * blockIdx.x + threadIdx.x;
  int y = TILE_DIM * blockIdx.y + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    output[x*width + (y+j)] = input[(y+j)*width + x];
}

// --- Write Coalesced Kernel --- 143 us ---
// 11% decrease by coalescing writes and not coalescing reads

__global__ void write_coalesced_matrix_transpose_kernel(float *output,
                                                        const float *input,
                                                        int N) {
  int threadIdX = blockDim.x * blockIdx.x + threadIdx.x;
  int threadIdY = blockDim.y * blockIdx.y + threadIdx.y;

  if (threadIdX >= N || threadIdY >= N)
    return;

  int oldSpot = threadIdX + N * threadIdY;
  int newSpot = threadIdY + N * threadIdX;

  output[oldSpot] = input[newSpot];
}

// --- Shared memory optimized kernel

// __global__ void smem_optimized_matrix_transpose_kernel(float *output,
//                                                        const float *input,
//                                                        int N) {
//   __shared__ float tile[TILE_DIM][TILE_DIM];

//   int x = blockIdx.x * TILE_DIM + threadIdx.x;
//   int y = blockIdx.y * TILE_DIM + threadIdx.y;

//   for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
//     tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * N + x];

//   __syncthreads();

//   // now do the optimization
// }