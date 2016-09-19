#include "CudaInterface.h"
#include "CudaGlobalKernels.h"

void cudaExecute2DTMzSteps (CudaExitStatus *retval,
                            FieldValue *Ez, FieldValue *Hx, FieldValue *Hy,
                            FieldValue *Ez_prev, FieldValue *Hx_prev, FieldValue *Hy_prev,
                            FieldValue *eps, FieldValue *mu,
                            FieldValue gridTimeStep, FieldValue gridStep,
                            grid_coord sx_Ez, grid_coord sy_Ez,
                            grid_coord sx_Hx, grid_coord sy_Hx,
                            grid_coord sx_Hy, grid_coord sy_Hy,
                            grid_iter sizeEps, grid_iter sizeMu,
                            time_step stepStart, time_step stepEnd,
                            uint32_t blocksX, uint32_t blocksY, uint32_t threadsX, uint32_t threadsY)
{
  FieldValue *Ez_cuda;
  FieldValue *Hx_cuda;
  FieldValue *Hy_cuda;

  FieldValue *Ez_cuda_prev;
  FieldValue *Hx_cuda_prev;
  FieldValue *Hy_cuda_prev;

  FieldValue *eps_cuda;
  FieldValue *mu_cuda;

  grid_iter sizeEz = (grid_iter) sx_Ez * sy_Ez * sizeof (FieldValue);
  grid_iter sizeHx = (grid_iter) sx_Hx * sy_Hx * sizeof (FieldValue);
  grid_iter sizeHy = (grid_iter) sx_Hy * sy_Hy * sizeof (FieldValue);
  //printf ("%llu=%ld*%ld*%lld", size, sx, sy, sizeof (FieldValue));

  cudaCheckErrorCmd (cudaMalloc ((void **) &Ez_cuda, sizeEz));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hx_cuda, sizeHx));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hy_cuda, sizeHy));

  cudaCheckErrorCmd (cudaMalloc ((void **) &Ez_cuda_prev, sizeEz));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hx_cuda_prev, sizeHx));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hy_cuda_prev, sizeHy));

  cudaCheckErrorCmd (cudaMalloc ((void **) &eps_cuda, sizeEps));
  cudaCheckErrorCmd (cudaMalloc ((void **) &mu_cuda, sizeMu));

  cudaCheckErrorCmd (cudaMemcpy (Ez_cuda, Ez, sizeEz, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hx_cuda, Hx, sizeHx, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hy_cuda, Hy, sizeHy, cudaMemcpyHostToDevice));

  cudaCheckErrorCmd (cudaMemcpy (Ez_cuda_prev, Ez_prev, sizeEz, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hx_cuda_prev, Hx_prev, sizeHx, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hy_cuda_prev, Hy_prev, sizeHy, cudaMemcpyHostToDevice));

  cudaCheckErrorCmd (cudaMemcpy (eps_cuda, eps, sizeEps, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (mu_cuda, mu, sizeMu, cudaMemcpyHostToDevice));

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
                                                                        eps_cuda,
                                                                        gridTimeStep,
                                                                        gridStep,
                                                                        sx_Ez,
                                                                        sy_Ez,
                                                                        t));

    cudaCheckExitStatus (cudaCalculateTMzESource <<< blocks, threads >>> (exitStatusCuda,
                                                                          Ez_cuda_prev,
                                                                          sx_Ez,
                                                                          sy_Ez,
                                                                          t));

    cudaCheckExitStatus (cudaCalculateTMzHStep <<< blocks, threads >>> (exitStatusCuda,
                                                                        Hx_cuda,
                                                                        Hy_cuda,
                                                                        Ez_cuda_prev,
                                                                        Hx_cuda_prev,
                                                                        Hy_cuda_prev,
                                                                        mu_cuda,
                                                                        gridTimeStep,
                                                                        gridStep,
                                                                        sx_Hx,
                                                                        sy_Hx,
                                                                        sx_Hy,
                                                                        sy_Hy,
                                                                        t));

    cudaCheckExitStatus (cudaCalculateTMzHSource <<< blocks, threads >>> (exitStatusCuda,
                                                                          Hx_cuda_prev,
                                                                          Hy_cuda_prev,
                                                                          sx_Hx,
                                                                          sy_Hx,
                                                                          sx_Hy,
                                                                          sy_Hy,
                                                                          t));
  }

  cudaCheckErrorCmd (cudaMemcpy (Ez, Ez_cuda, sizeEz, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (Hx, Hx_cuda, sizeHx, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (Hy, Hy_cuda, sizeHy, cudaMemcpyDeviceToHost));

  cudaCheckErrorCmd (cudaMemcpy (Ez_prev, Ez_cuda_prev, sizeEz, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (Hx_prev, Hx_cuda_prev, sizeHx, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (Hy_prev, Hy_cuda_prev, sizeHy, cudaMemcpyDeviceToHost));

  cudaCheckErrorCmd (cudaMemcpy (eps, eps_cuda, sizeEps, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (mu, mu_cuda, sizeMu, cudaMemcpyDeviceToHost));

  cudaCheckErrorCmd (cudaFree (Ez_cuda));
  cudaCheckErrorCmd (cudaFree (Hx_cuda));
  cudaCheckErrorCmd (cudaFree (Hy_cuda));

  cudaCheckErrorCmd (cudaFree (Ez_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Hx_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Hy_cuda_prev));

  cudaCheckErrorCmd (cudaFree (eps_cuda));
  cudaCheckErrorCmd (cudaFree (mu_cuda));

  *retval = CUDA_OK;
  return;
}
