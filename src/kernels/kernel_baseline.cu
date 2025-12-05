#include <cstdio>
#include <math.h>

// --- Baseline Kernel (Naive, Unoptimized) ---

__global__ void baseline_matrix_transpose_kernel(float *output,
                                                 const float *input, int N) {

  int threadIdX = blockDim.x * blockIdx.x + threadIdx.x;
  int threadIdY = blockDim.y * blockIdx.y + threadIdx.y;

  if (threadIdX >= N || threadIdY >= N)
    return;

  int oldSpot = threadIdX + N * threadIdY;
  int newSpot = threadIdY + N * threadIdX;

  // if (threadIdX <= 6 && threadIdY <= 6) {
  //   printf("===============threadIdX: %d, threadIdY: "
  //          "%d===============\noldSpot: %d\nnewSpot: %d\n===============\n",
  //          threadIdX, threadIdY, oldSpot, newSpot);
  // }

  // output[newSpot] = input[oldSpot];
  output[oldSpot] = input[newSpot];
}