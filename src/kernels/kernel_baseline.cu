#include <cstdio>
#include <math.h>

// --- Baseline Kernel (Naive, Unoptimized) ---

__global__ void baseline_matrix_transpose_kernel(float *output,
                                                 const float *input, int N) {

  __shared__ float block[256];

  int threadIdX = blockDim.x * blockIdx.x + threadIdx.x;
  int threadIdY = blockDim.y * blockIdx.y + threadIdx.y;

  if (threadIdX >= N || threadIdY >= N)
    return;

  int oldSpot = threadIdX + N * threadIdY;
  int newSpot = threadIdY + N * threadIdX;

  block[threadIdx.x + threadIdx.y * 16] = input[oldSpot];

  output[newSpot] = block[threadIdx.x + threadIdx.y * 16];
  // output[oldSpot] = input[newSpot];
}