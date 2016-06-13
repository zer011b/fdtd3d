#include "CudaKernelInterface.h"

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

// __global__ void fdtd_step_Ez (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy,
//                               FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
//                               FieldValue gridTimeStep, FieldValue gridStep, int sx, int sy, int t)
// {
//   int i = (blockIdx.x * blockDim.x) + threadIdx.x;
//   int j = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//   int index1 = i * sy + j;
//   int index2 = (i - 1) * sy + j;
//   int index3 = i * sy + j - 1;
//
//   if (i < 0 || j < 0 || i >= sx || j >= sy)
//   {
//     return;
//   }
//
//   if (i == 0 || j == 0)
//   {
//     Ez[index1] = Ez_prev[index1];
//     return;
//   }
//
//   Ez[index1] = calculateEz_2D_TMz (Ez_prev[index1],
//                                    Hx_prev[index3],
//                                    Hx_prev[index1],
//                                    Hy_prev[index1],
//                                    Hy_prev[index2],
//                                    gridTimeStep,
//                                    gridStep,
//                                    1.0 * 0.0000000000088541878176203892);
//
//   if (i == sx / 2 && j == sy / 2)
//   {
//     Ez[index1] = cos (t * 3.1415 / 12);
//   }
//
//   /*printf ("Cuda block #(x=%d,y=%d) of size #(%d,%d), thread #(x=%d, y=%d) = %d %d %d %d. Val = %f\n",
//     blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y, i, j, sx, sy, Hx_prev[index1]);*/
//
//   Ez_prev[index1] = Ez[index1];
// }
//
// __global__ void fdtd_step_Hx (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy,
//                               FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
//                               FieldValue gridTimeStep, FieldValue gridStep, int sx, int sy)
// {
//   int i = (blockIdx.x * blockDim.x) + threadIdx.x;
//   int j = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//   int index1 = i * sy + j;
//   int index2 = i * sy + j + 1;
//
//   if (i < 0 || j < 0 || i >= sx || j >= sy)
//   {
//     return;
//   }
//
//   if (i == 0 || j == sy - 1)
//   {
//     Hx[index1] = Hx_prev[index1];
//     return;
//   }
//
//   Hx[index1] = calculateHx_2D_TMz (Hx_prev[index1],
//                                    Ez_prev[index1],
//                                    Ez_prev[index2],
//                                    gridTimeStep,
//                                    gridStep,
//                                    1.0 * 0.0000012566370614359173);
//
//   Hx_prev[index1] = Hx[index1];
// }
//
// __global__ void fdtd_step_Hy (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy,
//                               FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
//                               FieldValue gridTimeStep, FieldValue gridStep, int sx, int sy)
// {
//   int i = (blockIdx.x * blockDim.x) + threadIdx.x;
//   int j = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//   int index1 = i * sy + j;
//   int index2 = (i + 1) * sy + j;
//
//   if (i < 0 || j < 0 || i >= sx || j >= sy)
//   {
//     return;
//   }
//
//   if (i == sx - 1 || j == 0)
//   {
//     Hy[index1] = Hy_prev[index1];
//     return;
//   }
//
//   Hy[index1] = calculateHx_2D_TMz (Hy_prev[index1],
//                                    Ez_prev[index2],
//                                    Ez_prev[index1],
//                                    gridTimeStep,
//                                    gridStep,
//                                    1.0 * 0.0000012566370614359173);
//
//   Hy_prev[index1] = Hy[index1];
// }
//
// void executeTMz (FieldValue *tmp_Ez, FieldValue *tmp_Hx, FieldValue *tmp_Hy,
//                  FieldValue *tmp_Ez_prev, FieldValue *tmp_Hx_prev, FieldValue *tmp_Hy_prev,
//                  int sx, int sy, FieldValue gridTimeStep, FieldValue gridStep, int totalStep)
// {
//   FieldValue *Ez_cuda;
//   FieldValue *Hx_cuda;
//   FieldValue *Hy_cuda;
//
//   FieldValue *Ez_cuda_prev;
//   FieldValue *Hx_cuda_prev;
//   FieldValue *Hy_cuda_prev;
//
//   int size = sx * sy * sizeof (FieldValue);
//
//   cudaCheckErrors (cudaMalloc ((void **) &Ez_cuda, size));
//   cudaCheckErrors (cudaMalloc ((void **) &Hx_cuda, size));
//   cudaCheckErrors (cudaMalloc ((void **) &Hy_cuda, size));
//
//   cudaCheckErrors (cudaMalloc ((void **) &Ez_cuda_prev, size));
//   cudaCheckErrors (cudaMalloc ((void **) &Hx_cuda_prev, size));
//   cudaCheckErrors (cudaMalloc ((void **) &Hy_cuda_prev, size));
//
//   cudaCheckErrors (cudaMemcpy (Ez_cuda, tmp_Ez, size, cudaMemcpyHostToDevice));
//   cudaCheckErrors (cudaMemcpy (Hx_cuda, tmp_Hx, size, cudaMemcpyHostToDevice));
//   cudaCheckErrors (cudaMemcpy (Hy_cuda, tmp_Hy, size, cudaMemcpyHostToDevice));
//
//   cudaCheckErrors (cudaMemcpy (Ez_cuda_prev, tmp_Ez_prev, size, cudaMemcpyHostToDevice));
//   cudaCheckErrors (cudaMemcpy (Hx_cuda_prev, tmp_Hx_prev, size, cudaMemcpyHostToDevice));
//   cudaCheckErrors (cudaMemcpy (Hy_cuda_prev, tmp_Hy_prev, size, cudaMemcpyHostToDevice));
//
//   int NN = 32;
//
//   dim3 N (sx / NN, sy / NN);
//   dim3 N1 (NN, NN);
//
//   for (int t = 0; t < totalStep; ++t)
//   {
//     fdtd_step_Ez <<< N, N1 >>> (Ez_cuda, Hx_cuda, Hy_cuda, Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev, gridTimeStep, gridStep, sx, sy, t);
//     fdtd_step_Hx <<< N, N1 >>> (Ez_cuda, Hx_cuda, Hy_cuda, Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev, gridTimeStep, gridStep, sx, sy);
//     fdtd_step_Hy <<< N, N1 >>> (Ez_cuda, Hx_cuda, Hy_cuda, Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev, gridTimeStep, gridStep, sx, sy);
//   }
//
//   cudaCheckErrors (cudaMemcpy (tmp_Ez, Ez_cuda, size, cudaMemcpyDeviceToHost));
//   cudaCheckErrors (cudaMemcpy (tmp_Hx, Hx_cuda, size, cudaMemcpyDeviceToHost));
//   cudaCheckErrors (cudaMemcpy (tmp_Hy, Hy_cuda, size, cudaMemcpyDeviceToHost));
//
//   cudaCheckErrors (cudaMemcpy (tmp_Ez_prev, Ez_cuda_prev, size, cudaMemcpyDeviceToHost));
//   cudaCheckErrors (cudaMemcpy (tmp_Hx_prev, Hx_cuda_prev, size, cudaMemcpyDeviceToHost));
//   cudaCheckErrors (cudaMemcpy (tmp_Hy_prev, Hy_cuda_prev, size, cudaMemcpyDeviceToHost));
//
//   cudaCheckErrors (cudaFree (Ez_cuda));
//   cudaCheckErrors (cudaFree (Hx_cuda));
//   cudaCheckErrors (cudaFree (Hy_cuda));
//
//   cudaCheckErrors (cudaFree (Ez_cuda_prev));
//   cudaCheckErrors (cudaFree (Hx_cuda_prev));
//   cudaCheckErrors (cudaFree (Hy_cuda_prev));
// }





__global__ CudaExitStatus cudaCalculateTMzSteps (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy,
                                                 FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                                                 FieldValue gridTimeStep, FieldValue gridStep,
                                                 grid_coord sx, grid_coord sy,
                                                 time_step stepStart, time_step stepEnd)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i >= sx || j >= sy)
  {
    return CUDA_ERROR;
  }

  grid_coord indexEz1 = i * sy + j;
  grid_coord indexEz2 = (i - 1) * sy + j;
  grid_coord indexEz3 = i * sy + j - 1;

  grid_coord indexHx1 = i * sy + j;
  grid_coord indexHx2 = i * sy + j + 1;

  grid_coord indexHy1 = i * sy + j;
  grid_coord indexHy2 = (i + 1) * sy + j;

  for (time_step t = stepStart; t < stepEnd; ++t)
  {
    // Shift border values of Ez
    if (i == 0 || j == 0)
    {
      Ez[indexEz1] = Ez_prev[indexEz1];
    }
    else
    {
      Ez[index1] = calculateEz_2D_TMz (Ez_prev[indexEz1],
                                       Hx_prev[indexEz3],
                                       Hx_prev[indexEz1],
                                       Hy_prev[indexEz1],
                                       Hy_prev[indexEz2],
                                       gridTimeStep,
                                       gridStep,
                                       1.0 * 0.0000000000088541878176203892);
    }

    if (i == sx / 2 && j == sy / 2)
    {
      Ez[indexEz1] = cos (t * 3.1415 / 12);
    }

    // Shift border values of Hx
    if (i == 0 || j == sy - 1)
    {
      Hx[indexHx1] = Hx_prev[indexHx1];
    }
    else
    {
      Hx[index1] = calculateHx_2D_TMz (Hx_prev[index1],
                                       Ez_prev[index1],
                                       Ez_prev[index2],
                                       gridTimeStep,
                                       gridStep,
                                       1.0 * 0.0000012566370614359173);
    }

    // Shift border values of Hy
    if (i == sx - 1 || j == 0)
    {
      Hy[index1] = Hy_prev[index1];
      return;
    }
    else
    {
      Hy[index1] = calculateHx_2D_TMz (Hy_prev[index1],
                                       Ez_prev[index2],
                                       Ez_prev[index1],
                                       gridTimeStep,
                                       gridStep,
                                       1.0 * 0.0000012566370614359173);
    }

    Ez_prev[indexEz1] = Ez[indexEz1];
    Hx_prev[indexHx1] = Hx[indexHx1];
    Hy_prev[indexHy1] = Hy[indexHy1];
  }

  return CUDA_OK;
}

CudaExitStatus cudaExecuteTMzSteps (FieldValue *Ez, FieldValue *Hx, FieldValue *Hy,
                                    FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                                    FieldValue gridTimeStep, FieldValue gridStep,
                                    grid_coord sx, grid_coord sy,
                                    time_step stepStart, time_step stepEnd,
                                    uint32_t blocksX, uin32_t blocksY, uint32_t threadsX, uin32_t threadsY)
{
  FieldValue *Ez_cuda;
  FieldValue *Hx_cuda;
  FieldValue *Hy_cuda;

  FieldValue *Ez_cuda_prev;
  FieldValue *Hx_cuda_prev;
  FieldValue *Hy_cuda_prev;

  grid_coord size = sx * sy * sizeof (FieldValue);

  cudaCheckErrors (cudaMalloc ((void **) &Ez_cuda, size));
  cudaCheckErrors (cudaMalloc ((void **) &Hx_cuda, size));
  cudaCheckErrors (cudaMalloc ((void **) &Hy_cuda, size));

  cudaCheckErrors (cudaMalloc ((void **) &Ez_cuda_prev, size));
  cudaCheckErrors (cudaMalloc ((void **) &Hx_cuda_prev, size));
  cudaCheckErrors (cudaMalloc ((void **) &Hy_cuda_prev, size));

  cudaCheckErrors (cudaMemcpy (Ez_cuda, Ez, size, cudaMemcpyHostToDevice));
  cudaCheckErrors (cudaMemcpy (Hx_cuda, Hx, size, cudaMemcpyHostToDevice));
  cudaCheckErrors (cudaMemcpy (Hy_cuda, Hy, size, cudaMemcpyHostToDevice));

  cudaCheckErrors (cudaMemcpy (Ez_cuda_prev, Ez_prev, size, cudaMemcpyHostToDevice));
  cudaCheckErrors (cudaMemcpy (Hx_cuda_prev, Hx_prev, size, cudaMemcpyHostToDevice));
  cudaCheckErrors (cudaMemcpy (Hy_cuda_prev, Hy_prev, size, cudaMemcpyHostToDevice));

  dim3 blocks (blocksX, blocksY);
  dim3 threads (threadsX, threadsY);

  CudaExitStatus exitStatus;
  exitStatus = cudaCalculateTMzSteps <<< blocks, threads >>> (Ez_cuda, Hx_cuda, Hy_cuda,
                                                              Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev,
                                                              gridTimeStep, gridStep, sx, sy, stepStart, stepEnd);

  if (exitStatus != CUDA_OK)
  {
    cudaCheckErrors (void);

    return CUDA_ERROR;
  }

  cudaCheckErrors (cudaMemcpy (Ez, Ez_cuda, size, cudaMemcpyDeviceToHost));
  cudaCheckErrors (cudaMemcpy (Hx, Hx_cuda, size, cudaMemcpyDeviceToHost));
  cudaCheckErrors (cudaMemcpy (Hy, Hy_cuda, size, cudaMemcpyDeviceToHost));

  cudaCheckErrors (cudaMemcpy (Ez_prev, Ez_cuda_prev, size, cudaMemcpyDeviceToHost));
  cudaCheckErrors (cudaMemcpy (Hx_prev, Hx_cuda_prev, size, cudaMemcpyDeviceToHost));
  cudaCheckErrors (cudaMemcpy (Hy_prev, Hy_cuda_prev, size, cudaMemcpyDeviceToHost));

  cudaCheckErrors (cudaFree (Ez_cuda));
  cudaCheckErrors (cudaFree (Hx_cuda));
  cudaCheckErrors (cudaFree (Hy_cuda));

  cudaCheckErrors (cudaFree (Ez_cuda_prev));
  cudaCheckErrors (cudaFree (Hx_cuda_prev));
  cudaCheckErrors (cudaFree (Hy_cuda_prev));

  return CUDA_OK;
}
