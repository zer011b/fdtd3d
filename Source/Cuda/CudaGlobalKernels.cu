#include "CudaGlobalKernels.h"

#include "PhysicsConst.h"

static FieldValue CUDA_DEVICE calcField (FieldValue prev, FieldValue oppositeField12, FieldValue oppositeField11,
                      FieldValue oppositeField22, FieldValue oppositeField21, FieldValue prevRightSide,
                      FPValue Ca, FPValue Cb, FPValue delta)
{
  YeeGridLayout<static_cast<uint8_t>(SchemeType::Dim3), GridCoordinate3DTemplate, LayoutType::E_CENTERED> *yeeLayout = NULLPTR;

  FieldValue tmp = oppositeField12 - oppositeField11 - oppositeField22 + oppositeField21 + prevRightSide * delta;
  return Ca * prev + Cb * tmp;
}

//
// __global__ void cudaCalculateTMzEzStep (CudaExitStatus *retval,
//                                         FieldValue *Ez,
//                                         FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
//                                         FieldValue *eps,
//                                         FPValue gridTimeStep, FPValue gridStep,
//                                         GridCoordinate3D Ez_start,
//                                         grid_coord Ez_endX, grid_coord Ez_endY,
//                                         grid_coord sx_Ez, grid_coord sy_Ez,
//                                         time_step t)
// {
//   grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
//   grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//   if (i < Ez_start.get1 () || j < Ez_start.get2 () || i >= Ez_endX || j >= Ez_endY)
//   {
//     *retval = CUDA_OK;
//     return;
//   }
//
//   grid_coord indexEz1 = i * sy_Ez + j;
//   grid_coord indexEz2 = (i - 1) * sy_Ez + j;
//   grid_coord indexEz3 = i * sy_Ez + j - 1;
//
//   Ez[indexEz1] = calcField (Ez_prev[indexEz1],
//                                  Hy_prev[indexEz1],
//                                  Hy_prev[indexEz2],
//                                  Hx_prev[indexEz1],
//                                  Hx_prev[indexEz3],
//                                  0.0,
//                                  1.0,
//                                  gridTimeStep / (eps[indexEz1] * PhysicsConst::Eps0 * gridStep),
//                                  gridStep);
//
//   Ez_prev[indexEz1] = Ez[indexEz1];
//
//   *retval = CUDA_OK;
//   return;
// }
//
// __global__ void cudaCalculateTMzEzSource (CudaExitStatus *retval,
//                                           FieldValue *Ez_prev,
//                                           grid_coord Ez_startX, grid_coord Ez_startY,
//                                           grid_coord Ez_endX, grid_coord Ez_endY,
//                                           grid_coord sx_Ez, grid_coord sy_Ez,
//                                           time_step t,
//                                           int processId)
// {
//   grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
//   grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//   if (i < Ez_startX || j < Ez_startY || i >= Ez_endX || j >= Ez_endY)
//   {
//     *retval = CUDA_OK;
//     return;
//   }
//
//   grid_coord indexEz1 = i * sy_Ez + j;
//
//   if (processId == 0)
//   {
//     if (i == sx_Ez / 2 && j == sy_Ez / 2)
//     {
//       FPValue val = cos (t * 3.1415 / 12);
//
// #ifdef COMPLEX_FIELD_VALUES
//       Ez_prev[indexEz1].real = val;
//       Ez_prev[indexEz1].imag = 0;
// #else /* COMPLEX_FIELD_VALUES */
//       Ez_prev[indexEz1] = val;
// #endif /* !COMPLEX_FIELD_VALUES */
//     }
//   }
//
//   *retval = CUDA_OK;
//   return;
// }
//
// __global__ void cudaCalculateTMzHxStep (CudaExitStatus *retval,
//                                         FieldValue *Hx,
//                                         FieldValue *Ez_prev, FieldValue *Hx_prev,
//                                         FieldValue *mu,
//                                         FPValue gridTimeStep, FPValue gridStep,
//                                         grid_coord Hx_startX, grid_coord Hx_startY,
//                                         grid_coord Hx_endX, grid_coord Hx_endY,
//                                         grid_coord sx_Hx, grid_coord sy_Hx,
//                                         time_step t)
// {
//   grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
//   grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//   if (i < Hx_startX || j < Hx_startY || i >= Hx_endX || j >= Hx_endY)
//   {
//     *retval = CUDA_OK;
//     return;
//   }
//
//   grid_coord indexHx1 = i * sy_Hx + j;
//
//   /*
//    * FIXME: is this correct ?
//    */
//   grid_coord indexHx2 = i * sy_Hx + j + 1;
//
//   Hx[indexHx1] = calculateHx_Dim2_TMz (Hx_prev[indexHx1],
//                                      Ez_prev[indexHx2],
//                                      Ez_prev[indexHx1],
//                                      gridTimeStep,
//                                      gridStep,
//                                      mu[indexHx1] * PhysicsConst::Mu0);
//
//   Hx_prev[indexHx1] = Hx[indexHx1];
//
//   *retval = CUDA_OK;
//   return;
// }
//
// __global__ void cudaCalculateTMzHyStep (CudaExitStatus *retval,
//                                         FieldValue *Hy,
//                                         FieldValue *Ez_prev, FieldValue *Hy_prev,
//                                         FieldValue *mu,
//                                         FPValue gridTimeStep, FPValue gridStep,
//                                         grid_coord Hy_startX, grid_coord Hy_startY,
//                                         grid_coord Hy_endX, grid_coord Hy_endY,
//                                         grid_coord sx_Hy, grid_coord sy_Hy,
//                                         time_step t)
// {
//   grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
//   grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//   if (i < Hy_startX || j < Hy_startY || i >= Hy_endX || j >= Hy_endY)
//   {
//     *retval = CUDA_OK;
//     return;
//   }
//
//   grid_coord indexHy1 = i * sy_Hy + j;
//   grid_coord indexHy2 = (i + 1) * sy_Hy + j;
//
//   Hy[indexHy1] = calculateHy_Dim2_TMz (Hy_prev[indexHy1],
//                                      Ez_prev[indexHy2],
//                                      Ez_prev[indexHy1],
//                                      gridTimeStep,
//                                      gridStep,
//                                      mu[indexHy1] * PhysicsConst::Mu0);
//
//   Hy_prev[indexHy1] = Hy[indexHy1];
//
//   *retval = CUDA_OK;
//   return;
// }
//
// __global__ void cudaCalculateTMzHxSource (CudaExitStatus *retval,
//                                           FieldValue *Hx_prev,
//                                           grid_coord Hx_startX, grid_coord Hx_startY,
//                                           grid_coord Hx_endX, grid_coord Hx_endY,
//                                           grid_coord sx_Hx, grid_coord sy_Hx,
//                                           time_step t)
// {
//   grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
//   grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//   if (i < Hx_startX || j < Hx_startY || i >= Hx_endX || j >= Hx_endY)
//   {
//     *retval = CUDA_OK;
//     return;
//   }
//
//   *retval = CUDA_OK;
//   return;
// }
//
// __global__ void cudaCalculateTMzHySource (CudaExitStatus *retval,
//                                           FieldValue *Hy_prev,
//                                           grid_coord Hy_startX, grid_coord Hy_startY,
//                                           grid_coord Hy_endX, grid_coord Hy_endY,
//                                           grid_coord sx_Hy, grid_coord sy_Hy,
//                                           time_step t)
// {
//   grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
//   grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//   if (i < Hy_startX || j < Hy_startY || i >= Hy_endX || j >= Hy_endY)
//   {
//     *retval = CUDA_OK;
//     return;
//   }
//
//   *retval = CUDA_OK;
//   return;
// }

__global__ void cudaCalculate3DExStep (CudaExitStatus *retval,
                                       FieldValue *Ex,
                                       FieldValue *Ex_prev, FieldValue *Hy_prev, FieldValue *Hz_prev,
                                       FieldValue *eps,
                                       FPValue gridTimeStep, FPValue gridStep,
                                       grid_coord Ex_startX, grid_coord Ex_startY, grid_coord Ex_startZ,
                                       grid_coord Ex_endX, grid_coord Ex_endY, grid_coord Ex_endZ,
                                       grid_coord sx_Ex, grid_coord sy_Ex, grid_coord sz_Ex,
                                       time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
  grid_coord k = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (i < Ex_startX || j < Ex_startY || k < Ex_startZ
      || i >= Ex_endX || j >= Ex_endY || k >= Ex_endZ)
  {
    *retval = CUDA_OK;
    return;
  }

  grid_coord indexEx1 = i * sy_Ex * sz_Ex + j * sz_Ex + k;

  grid_coord indexEx2 = i * sy_Ex * sz_Ex + (j - 1) * sz_Ex + k;
  grid_coord indexEx3 = i * sy_Ex * sz_Ex + j * sz_Ex + k - 1;

  Ex[indexEx1] = calcField (Ex_prev[indexEx1],
                                 Hz_prev[indexEx1],
                                 Hz_prev[indexEx2],
                                 Hy_prev[indexEx1],
                                 Hy_prev[indexEx3],
                                 0.0,
                                 1.0,
                                 gridTimeStep / (eps[indexEx1] * PhysicsConst::Eps0 * gridStep),
                                 gridStep);

  Ex_prev[indexEx1] = Ex[indexEx1];

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculate3DEyStep (CudaExitStatus *retval,
                                       FieldValue *Ey,
                                       FieldValue *Ey_prev, FieldValue *Hx_prev, FieldValue *Hz_prev,
                                       FieldValue *eps,
                                       FPValue gridTimeStep, FPValue gridStep,
                                       grid_coord Ey_startX, grid_coord Ey_startY, grid_coord Ey_startZ,
                                       grid_coord Ey_endX, grid_coord Ey_endY, grid_coord Ey_endZ,
                                       grid_coord sx_Ey, grid_coord sy_Ey, grid_coord sz_Ey,
                                       time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
  grid_coord k = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (i < Ey_startX || j < Ey_startY || k < Ey_startZ
      || i >= Ey_endX || j >= Ey_endY || k >= Ey_endZ)
  {
    *retval = CUDA_OK;
    return;
  }

  grid_coord indexEy1 = i * sy_Ey * sz_Ey + j * sz_Ey + k;

  grid_coord indexEy2 = (i - 1) * sy_Ey * sz_Ey + j * sz_Ey + k;
  grid_coord indexEy3 = i * sy_Ey * sz_Ey + j * sz_Ey + k - 1;

  Ey[indexEy1] = calcField (Ey_prev[indexEy1],
                                 Hx_prev[indexEy1],
                                 Hx_prev[indexEy3],
                                 Hz_prev[indexEy1],
                                 Hz_prev[indexEy3],
                                 0.0,
                                 1.0,
                                 gridTimeStep / (eps[indexEy1] * PhysicsConst::Eps0 * gridStep),
                                 gridStep);

  Ey_prev[indexEy1] = Ey[indexEy1];

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculate3DEzStep (CudaExitStatus *retval,
                                       FieldValue *Ez,
                                       FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                                       FieldValue *eps,
                                       FPValue gridTimeStep, FPValue gridStep,
                                       grid_coord Ez_startX, grid_coord Ez_startY, grid_coord Ez_startZ,
                                       grid_coord Ez_endX, grid_coord Ez_endY, grid_coord Ez_endZ,
                                       grid_coord sx_Ez, grid_coord sy_Ez, grid_coord sz_Ez,
                                       time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
  grid_coord k = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (i < Ez_startX || j < Ez_startY || k < Ez_startZ
      || i >= Ez_endX || j >= Ez_endY || k >= Ez_endZ)
  {
    *retval = CUDA_OK;
    return;
  }

  grid_coord indexEz1 = i * sy_Ez * sz_Ez + j * sz_Ez + k;

  grid_coord indexEz2 = (i - 1) * sy_Ez * sz_Ez + j * sz_Ez + k;
  grid_coord indexEz3 = i * sy_Ez * sz_Ez + (j - 1) * sz_Ez + k;

  Ez[indexEz1] = calcField (Ez_prev[indexEz1],
                                 Hy_prev[indexEz1],
                                 Hy_prev[indexEz2],
                                 Hx_prev[indexEz1],
                                 Hx_prev[indexEz3],
                                 0.0,
                                 1.0,
                                 gridTimeStep / (eps[indexEz1] * PhysicsConst::Eps0 * gridStep),
                                 gridStep);

  Ez_prev[indexEz1] = Ez[indexEz1];

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculate3DHxStep (CudaExitStatus *retval,
                                       FieldValue *Hx,
                                       FieldValue *Hx_prev, FieldValue *Ey_prev, FieldValue *Ez_prev,
                                       FieldValue *mu,
                                       FPValue gridTimeStep, FPValue gridStep,
                                       grid_coord Hx_startX, grid_coord Hx_startY, grid_coord Hx_startZ,
                                       grid_coord Hx_endX, grid_coord Hx_endY, grid_coord Hx_endZ,
                                       grid_coord sx_Hx, grid_coord sy_Hx, grid_coord sz_Hx,
                                       time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
  grid_coord k = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (i < Hx_startX || j < Hx_startY || k < Hx_startZ
      || i >= Hx_endX || j >= Hx_endY || k >= Hx_endZ)
  {
    *retval = CUDA_OK;
    return;
  }

  grid_coord indexHx1 = i * sy_Hx * sz_Hx + j * sz_Hx + k;

  grid_coord indexHx2 = i * sy_Hx * sz_Hx + (j + 1) * sz_Hx + k;
  grid_coord indexHx3 = i * sy_Hx * sz_Hx + j * sz_Hx + k + 1;

  Hx[indexHx1] = calcField (Hx_prev[indexHx1],
                                 Ey_prev[indexHx3],
                                 Ey_prev[indexHx1],
                                 Ez_prev[indexHx2],
                                 Ez_prev[indexHx1],
                                 0.0,
                                 1.0,
                                 gridTimeStep / (mu[indexHx1] * PhysicsConst::Mu0 * gridStep),
                                 gridStep);

  Hx_prev[indexHx1] = Hx[indexHx1];

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculate3DHyStep (CudaExitStatus *retval,
                                       FieldValue *Hy,
                                       FieldValue *Hy_prev, FieldValue *Ex_prev, FieldValue *Ez_prev,
                                       FieldValue *mu,
                                       FPValue gridTimeStep, FPValue gridStep,
                                       grid_coord Hy_startX, grid_coord Hy_startY, grid_coord Hy_startZ,
                                       grid_coord Hy_endX, grid_coord Hy_endY, grid_coord Hy_endZ,
                                       grid_coord sx_Hy, grid_coord sy_Hy, grid_coord sz_Hy,
                                       time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
  grid_coord k = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (i < Hy_startX || j < Hy_startY || k < Hy_startZ
      || i >= Hy_endX || j >= Hy_endY || k >= Hy_endZ)
  {
    *retval = CUDA_OK;
    return;
  }

  grid_coord indexHy1 = i * sy_Hy * sz_Hy + j * sz_Hy + k;

  grid_coord indexHy2 = (i + 1) * sy_Hy * sz_Hy + j * sz_Hy + k;
  grid_coord indexHy3 = i * sy_Hy * sz_Hy + j * sz_Hy + k + 1;

  Hy[indexHy1] = calcField (Hy_prev[indexHy1],
                                 Ez_prev[indexHy2],
                                 Ez_prev[indexHy1],
                                 Ex_prev[indexHy3],
                                 Ex_prev[indexHy1],
                                 0.0,
                                 1.0,
                                 gridTimeStep / (mu[indexHy1] * PhysicsConst::Mu0 * gridStep),
                                 gridStep);

  Hy_prev[indexHy1] = Hy[indexHy1];

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculate3DHzStep (CudaExitStatus *retval,
                                       FieldValue *Hz,
                                       FieldValue *Hz_prev, FieldValue *Ex_prev, FieldValue *Ey_prev,
                                       FieldValue *mu,
                                       FPValue gridTimeStep, FPValue gridStep,
                                       grid_coord Hz_startX, grid_coord Hz_startY, grid_coord Hz_startZ,
                                       grid_coord Hz_endX, grid_coord Hz_endY, grid_coord Hz_endZ,
                                       grid_coord sx_Hz, grid_coord sy_Hz, grid_coord sz_Hz,
                                       time_step t)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
  grid_coord k = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (i < Hz_startX || j < Hz_startY || k < Hz_startZ
      || i >= Hz_endX || j >= Hz_endY || k >= Hz_endZ)
  {
    *retval = CUDA_OK;
    return;
  }

  grid_coord indexHz1 = i * sy_Hz * sz_Hz + j * sz_Hz + k;

  grid_coord indexHz2 = (i + 1) * sy_Hz * sz_Hz + j * sz_Hz + k;
  grid_coord indexHz3 = i * sy_Hz * sz_Hz + (j + 1) * sz_Hz + k;

  Hz[indexHz1] = calcField (Hz_prev[indexHz1],
                                 Ex_prev[indexHz3],
                                 Ex_prev[indexHz1],
                                 Ey_prev[indexHz2],
                                 Ey_prev[indexHz1],
                                 0.0,
                                 1.0,
                                 gridTimeStep / (mu[indexHz1] * PhysicsConst::Mu0 * gridStep),
                                 gridStep);

  Hz_prev[indexHz1] = Hz[indexHz1];

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculate3DExSource (CudaExitStatus *retval,
                                         FieldValue *Ex_prev,
                                         grid_coord Ex_startX, grid_coord Ex_startY, grid_coord Ex_startZ,
                                         grid_coord Ex_endX, grid_coord Ex_endY, grid_coord Ex_endZ,
                                         grid_coord sx_Ex, grid_coord sy_Ex, grid_coord sz_Ex,
                                         time_step t,
                                         int processId)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
  grid_coord k = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (i < Ex_startX || j < Ex_startY || k < Ex_startZ
      || i >= Ex_endX || j >= Ex_endY || k >= Ex_endZ)
  {
    *retval = CUDA_OK;
    return;
  }

  grid_coord indexEx1 = i * sy_Ex * sz_Ex + j * sz_Ex + k;

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculate3DEySource (CudaExitStatus *retval,
                                         FieldValue *Ey_prev,
                                         grid_coord Ey_startX, grid_coord Ey_startY, grid_coord Ey_startZ,
                                         grid_coord Ey_endX, grid_coord Ey_endY, grid_coord Ey_endZ,
                                         grid_coord sx_Ey, grid_coord sy_Ey, grid_coord sz_Ey,
                                         time_step t,
                                         int processId)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
  grid_coord k = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (i < Ey_startX || j < Ey_startY || k < Ey_startZ
      || i >= Ey_endX || j >= Ey_endY || k >= Ey_endZ)
  {
    *retval = CUDA_OK;
    return;
  }

  grid_coord indexEy1 = i * sy_Ey * sz_Ey + j * sz_Ey + k;

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculate3DEzSource (CudaExitStatus *retval,
                                         FieldValue *Ez_prev,
                                         grid_coord Ez_startX, grid_coord Ez_startY, grid_coord Ez_startZ,
                                         grid_coord Ez_endX, grid_coord Ez_endY, grid_coord Ez_endZ,
                                         grid_coord sx_Ez, grid_coord sy_Ez, grid_coord sz_Ez,
                                         time_step t,
                                         int processId)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
  grid_coord k = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (i < Ez_startX || j < Ez_startY || k < Ez_startZ
      || i >= Ez_endX || j >= Ez_endY || k >= Ez_endZ)
  {
    *retval = CUDA_OK;
    return;
  }

  grid_coord indexEz1 = i * sy_Ez * sz_Ez + j * sz_Ez + k;

  if (processId == 0)
  {
    if (i == sx_Ez / 2 && j == sy_Ez / 2)
    {
      FPValue val = cos (t * 3.1415 / 12);

#ifdef COMPLEX_FIELD_VALUES
      Ez_prev[indexEz1].real = val;
      Ez_prev[indexEz1].imag = 0;
#else /* COMPLEX_FIELD_VALUES */
      Ez_prev[indexEz1] = val;
#endif /* !COMPLEX_FIELD_VALUES */
    }
  }

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculate3DHxSource (CudaExitStatus *retval,
                                         FieldValue *Hx_prev,
                                         grid_coord Hx_startX, grid_coord Hx_startY, grid_coord Hx_startZ,
                                         grid_coord Hx_endX, grid_coord Hx_endY, grid_coord Hx_endZ,
                                         grid_coord sx_Hx, grid_coord sy_Hx, grid_coord sz_Hx,
                                         time_step t,
                                         int processId)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
  grid_coord k = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (i < Hx_startX || j < Hx_startY || k < Hx_startZ
      || i >= Hx_endX || j >= Hx_endY || k >= Hx_endZ)
  {
    *retval = CUDA_OK;
    return;
  }

  grid_coord indexHx1 = i * sy_Hx * sz_Hx + j * sz_Hx + k;

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculate3DHySource (CudaExitStatus *retval,
                                         FieldValue *Hy_prev,
                                         grid_coord Hy_startX, grid_coord Hy_startY, grid_coord Hy_startZ,
                                         grid_coord Hy_endX, grid_coord Hy_endY, grid_coord Hy_endZ,
                                         grid_coord sx_Hy, grid_coord sy_Hy, grid_coord sz_Hy,
                                         time_step t,
                                         int processId)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
  grid_coord k = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (i < Hy_startX || j < Hy_startY || k < Hy_startZ
      || i >= Hy_endX || j >= Hy_endY || k >= Hy_endZ)
  {
    *retval = CUDA_OK;
    return;
  }

  grid_coord indexHy1 = i * sy_Hy * sz_Hy + j * sz_Hy + k;

  *retval = CUDA_OK;
  return;
}

__global__ void cudaCalculate3DHzSource (CudaExitStatus *retval,
                                         FieldValue *Hz_prev,
                                         grid_coord Hz_startX, grid_coord Hz_startY, grid_coord Hz_startZ,
                                         grid_coord Hz_endX, grid_coord Hz_endY, grid_coord Hz_endZ,
                                         grid_coord sx_Hz, grid_coord sy_Hz, grid_coord sz_Hz,
                                         time_step t,
                                         int processId)
{
  grid_coord i = (blockIdx.x * blockDim.x) + threadIdx.x;
  grid_coord j = (blockIdx.y * blockDim.y) + threadIdx.y;
  grid_coord k = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (i < Hz_startX || j < Hz_startY || k < Hz_startZ
      || i >= Hz_endX || j >= Hz_endY || k >= Hz_endZ)
  {
    *retval = CUDA_OK;
    return;
  }

  grid_coord indexHz1 = i * sy_Hz * sz_Hz + j * sz_Hz + k;

  *retval = CUDA_OK;
  return;
}
