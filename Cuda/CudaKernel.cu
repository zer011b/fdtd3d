#include "CudaKernelInterface.h"
#include "Kernels.h"

#include "cstdio"

#define cudaCheckErrors(cmd) \
  { \
    (cmd); \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) \
    { \
      fprintf(stderr, "Fatal error: %s at %s:%d\n", cudaGetErrorString(__err), __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n"); \
      exit(1); \
    } \
  } while (0)

__global__ void fdtd_step_Ez (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy,
                              FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                              FieldValue gridTimeStep, FieldValue gridStep, int sx, int sy, int t)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  int index1 = i * sy + j;
  int index2 = (i - 1) * sy + j;
  int index3 = i * sy + j - 1;

  /*printf ("Cuda block #(x=%d,y=%d) of size #(%d,%d), thread #(x=%d, y=%d) = %d %d. Index = %d\n",
    blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y, i, j, index1);*/

  if (i < 0 || j < 0 || i >= sx || j >= sy)
  {
    return;
  }

  if (i == 0 || j == 0)
  {
    Ez[index1] = Ez_prev[index1];
    return;
  }

  Ez[index1] = calculateEz_2D_TMz (Ez_prev[index1],
                                   Hx_prev[index3],
                                   Hx_prev[index1],
                                   Hy_prev[index1],
                                   Hy_prev[index2],
                                   gridTimeStep,
                                   gridStep,
                                   1.0 * 0.0000000000088541878176203892);

  if (i == sx / 2 && j == sy / 2)
  {
    Ez[index1] = cos (t * 3.1415 / 12);
  }

  /*printf ("Cuda block #(x=%d,y=%d) of size #(%d,%d), thread #(x=%d, y=%d) = %d %d %d %d. Val = %f\n",
    blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y, i, j, sx, sy, Hx_prev[index1]);*/

  Ez_prev[index1] = Ez[index1];
}

__global__ void fdtd_step_Hx (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy,
                              FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                              FieldValue gridTimeStep, FieldValue gridStep, int sx, int sy)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  int index1 = i * sy + j;
  int index2 = i * sy + j + 1;

  if (i < 0 || j < 0 || i >= sx || j >= sy)
  {
    return;
  }

  if (i == 0 || j == sy - 1)
  {
    Hx[index1] = Hx_prev[index1];
    return;
  }

  Hx[index1] = calculateHx_2D_TMz (Hx_prev[index1],
                                   Ez_prev[index1],
                                   Ez_prev[index2],
                                   gridTimeStep,
                                   gridStep,
                                   1.0 * 0.0000012566370614359173);

  Hx_prev[index1] = Hx[index1];
}

__global__ void fdtd_step_Hy (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy,
                              FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                              FieldValue gridTimeStep, FieldValue gridStep, int sx, int sy)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  int index1 = i * sy + j;
  int index2 = (i + 1) * sy + j;

  if (i < 0 || j < 0 || i >= sx || j >= sy)
  {
    return;
  }

  if (i == sx - 1 || j == 0)
  {
    Hy[index1] = Hy_prev[index1];
    return;
  }

  Hy[index1] = calculateHx_2D_TMz (Hy_prev[index1],
                                   Ez_prev[index2],
                                   Ez_prev[index1],
                                   gridTimeStep,
                                   gridStep,
                                   1.0 * 0.0000012566370614359173);

  Hy_prev[index1] = Hy[index1];
}

void executeTMz (FieldValue *tmp_Ez, FieldValue *tmp_Hx, FieldValue *tmp_Hy,
                 FieldValue *tmp_Ez_prev, FieldValue *tmp_Hx_prev, FieldValue *tmp_Hy_prev,
                 int sx, int sy, FieldValue gridTimeStep, FieldValue gridStep, int totalStep)
{
  FieldValue *Ez_cuda;
  FieldValue *Hx_cuda;
  FieldValue *Hy_cuda;

  FieldValue *Ez_cuda_prev;
  FieldValue *Hx_cuda_prev;
  FieldValue *Hy_cuda_prev;

  int size = sx * sy * sizeof (FieldValue);

  cudaCheckErrors (cudaMalloc ((void **) &Ez_cuda, size));
  cudaCheckErrors (cudaMalloc ((void **) &Hx_cuda, size));
  cudaCheckErrors (cudaMalloc ((void **) &Hy_cuda, size));

  cudaCheckErrors (cudaMalloc ((void **) &Ez_cuda_prev, size));
  cudaCheckErrors (cudaMalloc ((void **) &Hx_cuda_prev, size));
  cudaCheckErrors (cudaMalloc ((void **) &Hy_cuda_prev, size));

  cudaCheckErrors (cudaMemcpy (Ez_cuda, tmp_Ez, size, cudaMemcpyHostToDevice));
  cudaCheckErrors (cudaMemcpy (Hx_cuda, tmp_Hx, size, cudaMemcpyHostToDevice));
  cudaCheckErrors (cudaMemcpy (Hy_cuda, tmp_Hy, size, cudaMemcpyHostToDevice));

  cudaCheckErrors (cudaMemcpy (Ez_cuda_prev, tmp_Ez_prev, size, cudaMemcpyHostToDevice));
  cudaCheckErrors (cudaMemcpy (Hx_cuda_prev, tmp_Hx_prev, size, cudaMemcpyHostToDevice));
  cudaCheckErrors (cudaMemcpy (Hy_cuda_prev, tmp_Hy_prev, size, cudaMemcpyHostToDevice));

  int NN = 32;

  dim3 N (sx / NN, sy / NN);
  dim3 N1 (NN, NN);

  for (int t = 0; t < totalStep; ++t)
  {
    fdtd_step_Ez <<< N, N1 >>> (Ez_cuda, Hx_cuda, Hy_cuda, Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev, gridTimeStep, gridStep, sx, sy, t);
    fdtd_step_Hx <<< N, N1 >>> (Ez_cuda, Hx_cuda, Hy_cuda, Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev, gridTimeStep, gridStep, sx, sy);
    fdtd_step_Hy <<< N, N1 >>> (Ez_cuda, Hx_cuda, Hy_cuda, Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev, gridTimeStep, gridStep, sx, sy);
  }

  cudaCheckErrors (cudaMemcpy (tmp_Ez, Ez_cuda, size, cudaMemcpyDeviceToHost));
  cudaCheckErrors (cudaMemcpy (tmp_Hx, Hx_cuda, size, cudaMemcpyDeviceToHost));
  cudaCheckErrors (cudaMemcpy (tmp_Hy, Hy_cuda, size, cudaMemcpyDeviceToHost));

  cudaCheckErrors (cudaMemcpy (tmp_Ez_prev, Ez_cuda_prev, size, cudaMemcpyDeviceToHost));
  cudaCheckErrors (cudaMemcpy (tmp_Hx_prev, Hx_cuda_prev, size, cudaMemcpyDeviceToHost));
  cudaCheckErrors (cudaMemcpy (tmp_Hy_prev, Hy_cuda_prev, size, cudaMemcpyDeviceToHost));

  cudaCheckErrors (cudaFree (Ez_cuda));
  cudaCheckErrors (cudaFree (Hx_cuda));
  cudaCheckErrors (cudaFree (Hy_cuda));

  cudaCheckErrors (cudaFree (Ez_cuda_prev));
  cudaCheckErrors (cudaFree (Hx_cuda_prev));
  cudaCheckErrors (cudaFree (Hy_cuda_prev));
}
