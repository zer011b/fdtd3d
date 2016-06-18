#include "CudaKernelInterface.h"

#include "cstdio"

#define cudaCheckError() \
  { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) \
    { \
      fprintf(stderr, "Fatal error: %s at %s:%d\n", cudaGetErrorString(__err), __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n"); \
      exit(1); \
    } \
  }

#define cudaCheckErrorCmd(cmd) \
  { \
    (cmd); \
    cudaCheckError(); \
  }

__global__ void cudaCalculateTMzESteps (CudaExitStatus *retval,
                                        FieldValue *Ez,
                                        FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                                        FieldValue gridTimeStep, FieldValue gridStep,
                                        grid_coord sx, grid_coord sy, time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i >= sx || j >= sy)
  {
    *retval = CUDA_ERROR;
    return;
  }

  grid_coord indexEz1 = i * sy + j;
  grid_coord indexEz2 = (i - 1) * sy + j;
  grid_coord indexEz3 = i * sy + j - 1;

  // Shift border values of Ez
  if (i == 0 || j == 0)
  {
    Ez[indexEz1] = Ez_prev[indexEz1];
  }
  else
  {
    Ez[indexEz1] = calculateEz_2D_TMz (Ez_prev[indexEz1],
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

  Ez_prev[indexEz1] = Ez[indexEz1];

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculateTMzHSteps (CudaExitStatus *retval,
                                        FieldValue *Hx, FieldValue *Hy,
                                        FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                                        FieldValue gridTimeStep, FieldValue gridStep,
                                        grid_coord sx, grid_coord sy, time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i >= sx || j >= sy)
  {
    *retval = CUDA_ERROR;
    return;
  }

  grid_coord indexHx1 = i * sy + j;
  grid_coord indexHx2 = i * sy + j + 1;

  grid_coord indexHy1 = i * sy + j;
  grid_coord indexHy2 = (i + 1) * sy + j;

  // Shift border values of Hx
  if (i == 0 || j == sy - 1)
  {
    Hx[indexHx1] = Hx_prev[indexHx1];
  }
  else
  {
    Hx[indexHx1] = calculateHx_2D_TMz (Hx_prev[indexHx1],
                                       Ez_prev[indexHx1],
                                       Ez_prev[indexHx2],
                                       gridTimeStep,
                                       gridStep,
                                       1.0 * 0.0000012566370614359173);
  }

  // Shift border values of Hy
  if (i == sx - 1 || j == 0)
  {
    Hy[indexHy1] = Hy_prev[indexHy1];
  }
  else
  {
    Hy[indexHy1] = calculateHx_2D_TMz (Hy_prev[indexHy1],
                                       Ez_prev[indexHy2],
                                       Ez_prev[indexHy1],
                                       gridTimeStep,
                                       gridStep,
                                       1.0 * 0.0000012566370614359173);
  }

  Hx_prev[indexHx1] = Hx[indexHx1];
  Hy_prev[indexHy1] = Hy[indexHy1];

  *retval = CUDA_OK;
  return;
}

void cudaExecuteTMzSteps (CudaExitStatus *retval,
                          FieldValue *Ez, FieldValue *Hx, FieldValue *Hy,
                          FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                          FieldValue gridTimeStep, FieldValue gridStep,
                          grid_coord sx, grid_coord sy,
                          time_step stepStart, time_step stepEnd,
                          uint32_t blocksX, uint32_t blocksY, uint32_t threadsX, uint32_t threadsY)
{
  FieldValue *Ez_cuda;
  FieldValue *Hx_cuda;
  FieldValue *Hy_cuda;

  FieldValue *Ez_cuda_prev;
  FieldValue *Hx_cuda_prev;
  FieldValue *Hy_cuda_prev;

  grid_iter size = (grid_iter) sx * sy * sizeof (FieldValue);
  //printf ("%llu=%ld*%ld*%lld", size, sx, sy, sizeof (FieldValue));

  cudaCheckErrorCmd (cudaMalloc ((void **) &Ez_cuda, size));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hx_cuda, size));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hy_cuda, size));

  cudaCheckErrorCmd (cudaMalloc ((void **) &Ez_cuda_prev, size));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hx_cuda_prev, size));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hy_cuda_prev, size));

  cudaCheckErrorCmd (cudaMemcpy (Ez_cuda, Ez, size, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hx_cuda, Hx, size, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hy_cuda, Hy, size, cudaMemcpyHostToDevice));

  cudaCheckErrorCmd (cudaMemcpy (Ez_cuda_prev, Ez_prev, size, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hx_cuda_prev, Hx_prev, size, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hy_cuda_prev, Hy_prev, size, cudaMemcpyHostToDevice));

  dim3 blocks (blocksX, blocksY);
  dim3 threads (threadsX, threadsY);

  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  for (time_step t = stepStart; t < stepEnd; ++t)
  {
    cudaCalculateTMzESteps <<< blocks, threads >>> (exitStatusCuda,
                                                    Ez_cuda,
                                                    Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev,
                                                    gridTimeStep, gridStep, sx, sy, t);

    cudaCheckErrorCmd (cudaMemcpy (&exitStatus, exitStatusCuda, sizeof (CudaExitStatus), cudaMemcpyDeviceToHost));

    if (exitStatus != CUDA_OK)
    {
      cudaCheckError ();

      *retval = CUDA_ERROR;
      return;
    }

    cudaCalculateTMzHSteps <<< blocks, threads >>> (exitStatusCuda,
                                                    Hx_cuda, Hy_cuda,
                                                    Ez_cuda_prev, Hx_cuda_prev, Hy_cuda_prev,
                                                    gridTimeStep, gridStep, sx, sy, t);

    cudaCheckErrorCmd (cudaMemcpy (&exitStatus, exitStatusCuda, sizeof (CudaExitStatus), cudaMemcpyDeviceToHost));

    if (exitStatus != CUDA_OK)
    {
      cudaCheckError ();

      *retval = CUDA_ERROR;
      return;
    }
  }

  cudaCheckErrorCmd (cudaMemcpy (Ez, Ez_cuda, size, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (Hx, Hx_cuda, size, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (Hy, Hy_cuda, size, cudaMemcpyDeviceToHost));

  cudaCheckErrorCmd (cudaMemcpy (Ez_prev, Ez_cuda_prev, size, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (Hx_prev, Hx_cuda_prev, size, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (Hy_prev, Hy_cuda_prev, size, cudaMemcpyDeviceToHost));

  cudaCheckErrorCmd (cudaFree (Ez_cuda));
  cudaCheckErrorCmd (cudaFree (Hx_cuda));
  cudaCheckErrorCmd (cudaFree (Hy_cuda));

  cudaCheckErrorCmd (cudaFree (Ez_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Hx_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Hy_cuda_prev));

  *retval = CUDA_OK;
  return;
}
