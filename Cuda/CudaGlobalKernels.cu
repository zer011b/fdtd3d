#include "CudaGlobalKernels.h"

__global__ void cudaCalculateTMzEStep (CudaExitStatus *retval,
                                       FieldValue *Ez,
                                       FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                                       FieldValue *eps,
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
                                       eps[indexEz1]);
  }

  Ez_prev[indexEz1] = Ez[indexEz1];

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculateTMzESource (CudaExitStatus *retval,
                                         FieldValue *Ez_prev,
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

  if (i == sx / 2 && j == sy / 2)
  {
    Ez_prev[indexEz1] = cos (t * 3.1415 / 12);
  }

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculateTMzHStep (CudaExitStatus *retval,
                                       FieldValue *Hx, FieldValue *Hy,
                                       FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                                       FieldValue *mu,
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
                                       mu[indexHx1]);
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
                                       mu[indexHy1]);
  }

  Hx_prev[indexHx1] = Hx[indexHx1];
  Hy_prev[indexHy1] = Hy[indexHy1];

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculateTMzHSource (CudaExitStatus *retval,
                                         FieldValue *Hx_prev, FieldValue *Hy_prev,
                                         grid_coord sx, grid_coord sy, time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i >= sx || j >= sy)
  {
    *retval = CUDA_ERROR;
    return;
  }

  *retval = CUDA_OK;
  return;
}
