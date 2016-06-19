#include "CudaInterface.h"
#include "CudaGlobalKernels.h"

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
    cudaCheckExitStatus (cudaCalculateTMzEStep <<< blocks, threads >>> (exitStatusCuda,
                                                                        Ez_cuda,
                                                                        Ez_cuda_prev,
                                                                        Hx_cuda_prev,
                                                                        Hy_cuda_prev,
                                                                        gridTimeStep,
                                                                        gridStep,
                                                                        sx,
                                                                        sy,
                                                                        t));

    cudaCheckExitStatus (cudaCalculateTMzESource <<< blocks, threads >>> (exitStatusCuda,
                                                                          Ez_cuda_prev,
                                                                          sx,
                                                                          sy,
                                                                          t));

    cudaCheckExitStatus (cudaCalculateTMzHStep <<< blocks, threads >>> (exitStatusCuda,
                                                                        Hx_cuda,
                                                                        Hy_cuda,
                                                                        Ez_cuda_prev,
                                                                        Hx_cuda_prev,
                                                                        Hy_cuda_prev,
                                                                        gridTimeStep,
                                                                        gridStep,
                                                                        sx,
                                                                        sy,
                                                                        t));

    cudaCheckExitStatus (cudaCalculateTMzHSource <<< blocks, threads >>> (exitStatusCuda,
                                                                          Hx_cuda_prev,
                                                                          Hy_cuda_prev,
                                                                          sx,
                                                                          sy,
                                                                          t));
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
