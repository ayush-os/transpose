#include <math.h>

const int TILE_DIM = 16;
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
    output[x * width + (y + j)] = input[(y + j) * width + x];
}

// --- Shared memory optimized kernel

__global__ void smem_transpose_kernel(float *output, const float *input) {
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int x = TILE_DIM * blockIdx.x + threadIdx.x;
  int y = TILE_DIM * blockIdx.y + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  #pragma unroll
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];

  __syncthreads();

  x = TILE_DIM * blockIdx.y + threadIdx.x;
  y = TILE_DIM * blockIdx.x + threadIdx.y;

  #pragma unroll
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    output[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void smem_copy_kernel(float *output, const float *input) {
  __shared__ float tile[TILE_DIM * TILE_DIM];

  int x = TILE_DIM * blockIdx.x + threadIdx.x;
  int y = TILE_DIM * blockIdx.y + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x] =
        input[(y + j) * width + x];

  __syncthreads();

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    output[(y + j) * width + x] =
        tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x];
}