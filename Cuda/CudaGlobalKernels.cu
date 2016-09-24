#include "CudaGlobalKernels.h"

__global__ void cudaCalculateTMzEzStep (CudaExitStatus *retval,
                                        FieldValue *Ez,
                                        FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                                        FieldValue *eps,
                                        FieldValue gridTimeStep, FieldValue gridStep,
                                        grid_coord Ez_startX, grid_coord Ez_startY,
                                        grid_coord Ez_endX, grid_coord Ez_endY,
                                        grid_coord sx_Ez, grid_coord sy_Ez,
                                        time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i < Ez_startX || j < Ez_startY || i >= Ez_endX || j >= Ez_endY)
  {
    *retval = CUDA_ERROR;
    return;
  }

  grid_coord indexEz1 = i * sy_Ez + j;
  grid_coord indexEz2 = (i - 1) * sy_Ez + j;
  grid_coord indexEz3 = i * sy_Ez + j - 1;

  Ez[indexEz1] = calculateEz_2D_TMz (Ez_prev[indexEz1],
                                     Hx_prev[indexEz3],
                                     Hx_prev[indexEz1],
                                     Hy_prev[indexEz1],
                                     Hy_prev[indexEz2],
                                     gridTimeStep,
                                     gridStep,
                                     eps[indexEz1]);

  Ez_prev[indexEz1] = Ez[indexEz1];

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculateTMzEzSource (CudaExitStatus *retval,
                                          FieldValue *Ez_prev,
                                          grid_coord Ez_startX, grid_coord Ez_startY,
                                          grid_coord Ez_endX, grid_coord Ez_endY,
                                          grid_coord sx_Ez, grid_coord sy_Ez,
                                          time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i < Ez_startX || j < Ez_startY || i >= Ez_endX || j >= Ez_endY)
  {
    *retval = CUDA_ERROR;
    return;
  }

  grid_coord indexEz1 = i * sy_Ez + j;

  if (i == sx_Ez / 2 && j == sy_Ez / 2)
  {
    Ez_prev[indexEz1] = cos (t * 3.1415 / 12);
  }

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculateTMzHxStep (CudaExitStatus *retval,
                                        FieldValue *Hx,
                                        FieldValue *Ez_prev, FieldValue *Hx_prev,
                                        FieldValue *mu,
                                        FieldValue gridTimeStep, FieldValue gridStep,
                                        grid_coord Hx_startX, grid_coord Hx_startY,
                                        grid_coord Hx_endX, grid_coord Hx_endY,
                                        grid_coord sx_Hx, grid_coord sy_Hx,
                                        time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i < Hx_startX || j < Hx_startY || i >= Hx_endX || j >= Hx_endY)
  {
    *retval = CUDA_ERROR;
    return;
  }

  grid_coord indexHx1 = i * sy_Hx + j;
  grid_coord indexHx2 = i * sy_Hx + j + 1;

  Hx[indexHx1] = calculateHx_2D_TMz (Hx_prev[indexHx1],
                                     Ez_prev[indexHx1],
                                     Ez_prev[indexHx2],
                                     gridTimeStep,
                                     gridStep,
                                     mu[indexHx1]);

  Hx_prev[indexHx1] = Hx[indexHx1];

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculateTMzHyStep (CudaExitStatus *retval,
                                        FieldValue *Hy,
                                        FieldValue *Ez_prev, FieldValue *Hy_prev,
                                        FieldValue *mu,
                                        FieldValue gridTimeStep, FieldValue gridStep,
                                        grid_coord Hy_startX, grid_coord Hy_startY,
                                        grid_coord Hy_endX, grid_coord Hy_endY,
                                        grid_coord sx_Hy, grid_coord sy_Hy,
                                        time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i < Hy_startX || j < Hy_startY || i >= Hy_endX || j >= Hy_endY)
  {
    *retval = CUDA_ERROR;
    return;
  }

  grid_coord indexHy1 = i * sy_Hy + j;
  grid_coord indexHy2 = (i + 1) * sy_Hy + j;

  Hy[indexHy1] = calculateHx_2D_TMz (Hy_prev[indexHy1],
                                     Ez_prev[indexHy2],
                                     Ez_prev[indexHy1],
                                     gridTimeStep,
                                     gridStep,
                                     mu[indexHy1]);

  Hy_prev[indexHy1] = Hy[indexHy1];

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculateTMzHxSource (CudaExitStatus *retval,
                                          FieldValue *Hx_prev,
                                          grid_coord Hx_startX, grid_coord Hx_startY,
                                          grid_coord Hx_endX, grid_coord Hx_endY,
                                          grid_coord sx_Hx, grid_coord sy_Hx,
                                          time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i < Hx_startX || j < Hx_startY || i >= Hx_endX || j >= Hx_endY)
  {
    *retval = CUDA_ERROR;
    return;
  }

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculateTMzHySource (CudaExitStatus *retval,
                                          FieldValue *Hy_prev,
                                          grid_coord Hy_startX, grid_coord Hy_startY,
                                          grid_coord Hy_endX, grid_coord Hy_endY,
                                          grid_coord sx_Hy, grid_coord sy_Hy,
                                          time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i < Hy_startX || j < Hy_startY || i >= Hy_endX || j >= Hy_endY)
  {
    *retval = CUDA_ERROR;
    return;
  }

  *retval = CUDA_OK;
  return;
}
