#include "CudaInterface.h"
#include "CudaGlobalKernels.h"

#ifdef PARALLEL_GRID
void cudaExecute2DTMzSteps (CudaExitStatus *retval,
                            YeeGridLayout &yeeLayout,
                            FieldValue gridTimeStep, FieldValue gridStep,
                            ParallelGrid &Ez,
                            ParallelGrid &Hx,
                            ParallelGrid &Hy,
                            ParallelGrid &Eps,
                            ParallelGrid &Mu,
                            time_step stepStart, time_step stepEnd)
{
  ParallelGridCoordinate EzSizeCoord = Ez.getSize ();
  ParallelGridCoordinate HxSizeCoord = Hx.getSize ();
  ParallelGridCoordinate HySizeCoord = Hy.getSize ();
  ParallelGridCoordinate EpsSizeCoord = Eps.getSize ();
  ParallelGridCoordinate MuSizeCoord = Mu.getSize ();

  grid_iter sizeEz = EzSizeCoord.calculateTotalCoord();
  grid_iter sizeHx = HxSizeCoord.calculateTotalCoord();
  grid_iter sizeHy = HySizeCoord.calculateTotalCoord();
  grid_iter sizeEps = EpsSizeCoord.calculateTotalCoord();
  grid_iter sizeMu = MuSizeCoord.calculateTotalCoord();

  grid_iter sizeEzRaw = (grid_iter) sizeEz * sizeof (FieldValue);
  grid_iter sizeHxRaw = (grid_iter) sizeHx * sizeof (FieldValue);
  grid_iter sizeHyRaw = (grid_iter) sizeHy * sizeof (FieldValue);
  grid_iter sizeEpsRaw = (grid_iter) sizeEps * sizeof (FieldValue);
  grid_iter sizeMuRaw = (grid_iter) sizeMu * sizeof (FieldValue);

  FieldValue *tmp_Ez = new FieldValue [sizeEz];
  FieldValue *tmp_Hx = new FieldValue [sizeHx];
  FieldValue *tmp_Hy = new FieldValue [sizeHy];

  FieldValue *tmp_Ez_prev = new FieldValue [sizeEz];
  FieldValue *tmp_Hx_prev = new FieldValue [sizeHx];
  FieldValue *tmp_Hy_prev = new FieldValue [sizeHy];

  FieldValue *tmp_eps = new FieldValue [sizeEps];
  FieldValue *tmp_mu = new FieldValue [sizeMu];

  for (grid_iter i = 0; i < sizeEz; ++i)
  {
    FieldPointValue* valEz = Ez.getFieldPointValue (i);
    tmp_Ez[i] = valEz->getCurValue ();
    tmp_Ez_prev[i] = valEz->getPrevValue ();
  }

  for (grid_iter i = 0; i < sizeHx; ++i)
  {
    FieldPointValue* valHx = Hx.getFieldPointValue (i);
    tmp_Hx[i] = valHx->getCurValue ();
    tmp_Hx_prev[i] = valHx->getPrevValue ();
  }

  for (grid_iter i = 0; i < sizeHy; ++i)
  {
    FieldPointValue* valHy = Hy.getFieldPointValue (i);
    tmp_Hy[i] = valHy->getCurValue ();
    tmp_Hy_prev[i] = valHy->getPrevValue ();
  }

  for (grid_iter i = 0; i < sizeEps; ++i)
  {
    FieldPointValue *valEps = Eps.getFieldPointValue (i);
    tmp_eps[i] = valEps->getCurValue ();
  }

  for (grid_iter i = 0; i < sizeMu; ++i)
  {
    FieldPointValue *valMu = Mu.getFieldPointValue (i);
    tmp_mu[i] = valMu->getCurValue ();
  }

  FieldValue *Ez_cuda;
  FieldValue *Hx_cuda;
  FieldValue *Hy_cuda;

  FieldValue *Ez_cuda_prev;
  FieldValue *Hx_cuda_prev;
  FieldValue *Hy_cuda_prev;

  FieldValue *eps_cuda;
  FieldValue *mu_cuda;

  cudaCheckErrorCmd (cudaMalloc ((void **) &Ez_cuda, sizeEzRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hx_cuda, sizeHxRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hy_cuda, sizeHyRaw));

  cudaCheckErrorCmd (cudaMalloc ((void **) &Ez_cuda_prev, sizeEzRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hx_cuda_prev, sizeHxRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hy_cuda_prev, sizeHyRaw));

  cudaCheckErrorCmd (cudaMalloc ((void **) &eps_cuda, sizeEpsRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &mu_cuda, sizeMuRaw));

  cudaCheckErrorCmd (cudaMemcpy (Ez_cuda, tmp_Ez, sizeEzRaw, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hx_cuda, tmp_Hx, sizeHxRaw, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hy_cuda, tmp_Hy, sizeHyRaw, cudaMemcpyHostToDevice));

  cudaCheckErrorCmd (cudaMemcpy (Ez_cuda_prev, tmp_Ez_prev, sizeEzRaw, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hx_cuda_prev, tmp_Hx_prev, sizeHxRaw, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hy_cuda_prev, tmp_Hy_prev, sizeHyRaw, cudaMemcpyHostToDevice));

  cudaCheckErrorCmd (cudaMemcpy (eps_cuda, tmp_eps, sizeEpsRaw, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (mu_cuda, tmp_mu, sizeMuRaw, cudaMemcpyHostToDevice));

  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  dim3 blocksEz (EzSizeCoord.getX () / 16, EzSizeCoord.getY () / 16);
  dim3 threadsEz (16, 16);

  dim3 blocksHx (HxSizeCoord.getX () / 16, HxSizeCoord.getY () / 16);
  dim3 threadsHx (16, 16);

  dim3 blocksHy (HySizeCoord.getX () / 16, HySizeCoord.getY () / 16);
  dim3 threadsHy (16, 16);

  for (time_step t = stepStart; t < stepEnd; ++t)
  {
    GridCoordinate3D EzStart = yeeLayout.getEzStart (Ez.getStart ());
    GridCoordinate3D EzEnd = yeeLayout.getEzEnd (Ez.getEnd ());

    GridCoordinate3D HxStart = yeeLayout.getHxStart (Hx.getStart ());
    GridCoordinate3D HxEnd = yeeLayout.getHxEnd (Hx.getEnd ());

    GridCoordinate3D HyStart = yeeLayout.getHyStart (Hy.getStart ());
    GridCoordinate3D HyEnd = yeeLayout.getHyEnd (Hy.getEnd ());

    cudaCheckExitStatus (cudaCalculateTMzEzStep <<< blocksEz, threadsEz >>> (exitStatusCuda,
                                                                             Ez_cuda,
                                                                             Ez_cuda_prev,
                                                                             Hx_cuda_prev,
                                                                             Hy_cuda_prev,
                                                                             eps_cuda,
                                                                             gridTimeStep,
                                                                             gridStep,
                                                                             EzStart.getX (),
                                                                             EzStart.getY (),
                                                                             EzEnd.getX (),
                                                                             EzEnd.getY (),
                                                                             EzSizeCoord.getX (),
                                                                             EzSizeCoord.getY (),
                                                                             t));

    cudaCheckExitStatus (cudaCalculateTMzEzSource <<< blocksEz, threadsEz >>> (exitStatusCuda,
                                                                               Ez_cuda_prev,
                                                                               EzStart.getX (),
                                                                               EzStart.getY (),
                                                                               EzEnd.getX (),
                                                                               EzEnd.getY (),
                                                                               EzSizeCoord.getX (),
                                                                               EzSizeCoord.getY (),
                                                                               t));

    Ez.nextShareStep ();

    cudaCheckExitStatus (cudaCalculateTMzHxStep <<< blocksHx, threadsHx >>> (exitStatusCuda,
                                                                             Hx_cuda,
                                                                             Ez_cuda_prev,
                                                                             Hx_cuda_prev,
                                                                             mu_cuda,
                                                                             gridTimeStep,
                                                                             gridStep,
                                                                             HxStart.getX (),
                                                                             HxStart.getY (),
                                                                             HxEnd.getX (),
                                                                             HxEnd.getY (),
                                                                             HxSizeCoord.getX (),
                                                                             HxSizeCoord.getY (),
                                                                             t));

    cudaCheckExitStatus (cudaCalculateTMzHyStep <<< blocksHy, threadsHy >>> (exitStatusCuda,
                                                                             Hy_cuda,
                                                                             Ez_cuda_prev,
                                                                             Hy_cuda_prev,
                                                                             mu_cuda,
                                                                             gridTimeStep,
                                                                             gridStep,
                                                                             HyStart.getX (),
                                                                             HyStart.getY (),
                                                                             HyEnd.getX (),
                                                                             HyEnd.getY (),
                                                                             HySizeCoord.getX (),
                                                                             HySizeCoord.getY (),
                                                                             t));

    cudaCheckExitStatus (cudaCalculateTMzHxSource <<< blocksHx, threadsHx >>> (exitStatusCuda,
                                                                               Hx_cuda_prev,
                                                                               HxStart.getX (),
                                                                               HxStart.getY (),
                                                                               HxEnd.getX (),
                                                                               HxEnd.getY (),
                                                                               HxSizeCoord.getX (),
                                                                               HxSizeCoord.getY (),
                                                                               t));

    cudaCheckExitStatus (cudaCalculateTMzHySource <<< blocksHy, threadsHy >>> (exitStatusCuda,
                                                                               Hy_cuda_prev,
                                                                               HyStart.getX (),
                                                                               HyStart.getY (),
                                                                               HyEnd.getX (),
                                                                               HyEnd.getY (),
                                                                               HySizeCoord.getX (),
                                                                               HySizeCoord.getY (),
                                                                               t));

    Hx.nextShareStep ();
    Hy.nextShareStep ();
  }

  cudaCheckErrorCmd (cudaMemcpy (tmp_Ez, Ez_cuda, sizeEzRaw, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (tmp_Hx, Hx_cuda, sizeHxRaw, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (tmp_Hy, Hy_cuda, sizeHyRaw, cudaMemcpyDeviceToHost));

  cudaCheckErrorCmd (cudaMemcpy (tmp_Ez_prev, Ez_cuda_prev, sizeEzRaw, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (tmp_Hx_prev, Hx_cuda_prev, sizeHxRaw, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (tmp_Hy_prev, Hy_cuda_prev, sizeHyRaw, cudaMemcpyDeviceToHost));

  cudaCheckErrorCmd (cudaMemcpy (tmp_eps, eps_cuda, sizeEpsRaw, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (tmp_mu, mu_cuda, sizeMuRaw, cudaMemcpyDeviceToHost));

  cudaCheckErrorCmd (cudaFree (Ez_cuda));
  cudaCheckErrorCmd (cudaFree (Hx_cuda));
  cudaCheckErrorCmd (cudaFree (Hy_cuda));

  cudaCheckErrorCmd (cudaFree (Ez_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Hx_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Hy_cuda_prev));

  cudaCheckErrorCmd (cudaFree (eps_cuda));
  cudaCheckErrorCmd (cudaFree (mu_cuda));

  for (grid_iter i = 0; i < sizeEz; ++i)
  {
    FieldPointValue* valEz = Ez.getFieldPointValue (i);
    valEz->setCurValue (tmp_Ez[i]);
    valEz->setPrevValue (tmp_Ez_prev[i]);
  }

  for (grid_iter i = 0; i < sizeHx; ++i)
  {
    FieldPointValue* valHx = Hx.getFieldPointValue (i);
    valHx->setCurValue (tmp_Hx[i]);
    valHx->setPrevValue (tmp_Hx_prev[i]);
  }

  for (grid_iter i = 0; i < sizeHy; ++i)
  {
    FieldPointValue* valHy = Hy.getFieldPointValue (i);
    valHy->setCurValue (tmp_Hy[i]);
    valHy->setPrevValue (tmp_Hy_prev[i]);
  }

  delete[] tmp_Ez;
  delete[] tmp_Hx;
  delete[] tmp_Hy;

  delete[] tmp_Ez_prev;
  delete[] tmp_Hx_prev;
  delete[] tmp_Hy_prev;

  delete[] tmp_eps;
  delete[] tmp_mu;

  *retval = CUDA_OK;
  return;
}

#else

void cudaExecute2DTMzSteps (CudaExitStatus *retval,
                            YeeGridLayout &yeeLayout,
                            FieldValue gridTimeStep, FieldValue gridStep,
                            Grid<GridCoordinate2D> &Ez,
                            Grid<GridCoordinate2D> &Hx,
                            Grid<GridCoordinate2D> &Hy,
                            Grid<GridCoordinate2D> &Eps,
                            Grid<GridCoordinate2D> &Mu,
                            time_step stepStart, time_step stepEnd)
{
  GridCoordinate2D EzSizeCoord = Ez.getSize ();
  GridCoordinate2D HxSizeCoord = Hx.getSize ();
  GridCoordinate2D HySizeCoord = Hy.getSize ();
  GridCoordinate2D EpsSizeCoord = Eps.getSize ();
  GridCoordinate2D MuSizeCoord = Mu.getSize ();

  grid_iter sizeEz = EzSizeCoord.calculateTotalCoord();
  grid_iter sizeHx = HxSizeCoord.calculateTotalCoord();
  grid_iter sizeHy = HySizeCoord.calculateTotalCoord();
  grid_iter sizeEps = EpsSizeCoord.calculateTotalCoord();
  grid_iter sizeMu = MuSizeCoord.calculateTotalCoord();

  grid_iter sizeEzRaw = (grid_iter) sizeEz * sizeof (FieldValue);
  grid_iter sizeHxRaw = (grid_iter) sizeHx * sizeof (FieldValue);
  grid_iter sizeHyRaw = (grid_iter) sizeHy * sizeof (FieldValue);
  grid_iter sizeEpsRaw = (grid_iter) sizeEps * sizeof (FieldValue);
  grid_iter sizeMuRaw = (grid_iter) sizeMu * sizeof (FieldValue);

  FieldValue *tmp_Ez = new FieldValue [sizeEz];
  FieldValue *tmp_Hx = new FieldValue [sizeHx];
  FieldValue *tmp_Hy = new FieldValue [sizeHy];

  FieldValue *tmp_Ez_prev = new FieldValue [sizeEz];
  FieldValue *tmp_Hx_prev = new FieldValue [sizeHx];
  FieldValue *tmp_Hy_prev = new FieldValue [sizeHy];

  FieldValue *tmp_eps = new FieldValue [sizeEps];
  FieldValue *tmp_mu = new FieldValue [sizeMu];

  for (grid_iter i = 0; i < sizeEz; ++i)
  {
    FieldPointValue* valEz = Ez.getFieldPointValue (i);
    tmp_Ez[i] = valEz->getCurValue ();
    tmp_Ez_prev[i] = valEz->getPrevValue ();
  }

  for (grid_iter i = 0; i < sizeHx; ++i)
  {
    FieldPointValue* valHx = Hx.getFieldPointValue (i);
    tmp_Hx[i] = valHx->getCurValue ();
    tmp_Hx_prev[i] = valHx->getPrevValue ();
  }

  for (grid_iter i = 0; i < sizeHy; ++i)
  {
    FieldPointValue* valHy = Hy.getFieldPointValue (i);
    tmp_Hy[i] = valHy->getCurValue ();
    tmp_Hy_prev[i] = valHy->getPrevValue ();
  }

  for (grid_iter i = 0; i < sizeEps; ++i)
  {
    FieldPointValue *valEps = Eps.getFieldPointValue (i);
    tmp_eps[i] = valEps->getCurValue ();
  }

  for (grid_iter i = 0; i < sizeMu; ++i)
  {
    FieldPointValue *valMu = Mu.getFieldPointValue (i);
    tmp_mu[i] = valMu->getCurValue ();
  }

  FieldValue *Ez_cuda;
  FieldValue *Hx_cuda;
  FieldValue *Hy_cuda;

  FieldValue *Ez_cuda_prev;
  FieldValue *Hx_cuda_prev;
  FieldValue *Hy_cuda_prev;

  FieldValue *eps_cuda;
  FieldValue *mu_cuda;

  cudaCheckErrorCmd (cudaMalloc ((void **) &Ez_cuda, sizeEzRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hx_cuda, sizeHxRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hy_cuda, sizeHyRaw));

  cudaCheckErrorCmd (cudaMalloc ((void **) &Ez_cuda_prev, sizeEzRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hx_cuda_prev, sizeHxRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hy_cuda_prev, sizeHyRaw));

  cudaCheckErrorCmd (cudaMalloc ((void **) &eps_cuda, sizeEpsRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &mu_cuda, sizeMuRaw));

  cudaCheckErrorCmd (cudaMemcpy (Ez_cuda, tmp_Ez, sizeEzRaw, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hx_cuda, tmp_Hx, sizeHxRaw, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hy_cuda, tmp_Hy, sizeHyRaw, cudaMemcpyHostToDevice));

  cudaCheckErrorCmd (cudaMemcpy (Ez_cuda_prev, tmp_Ez_prev, sizeEzRaw, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hx_cuda_prev, tmp_Hx_prev, sizeHxRaw, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (Hy_cuda_prev, tmp_Hy_prev, sizeHyRaw, cudaMemcpyHostToDevice));

  cudaCheckErrorCmd (cudaMemcpy (eps_cuda, tmp_eps, sizeEpsRaw, cudaMemcpyHostToDevice));
  cudaCheckErrorCmd (cudaMemcpy (mu_cuda, tmp_mu, sizeMuRaw, cudaMemcpyHostToDevice));

  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  GridCoordinate3D EzStart = yeeLayout.getEzStart (Ez.getStart ());
  GridCoordinate3D EzEnd = yeeLayout.getEzEnd (Ez.getEnd ());

  GridCoordinate3D HxStart = yeeLayout.getHxStart (Hx.getStart ());
  GridCoordinate3D HxEnd = yeeLayout.getHxEnd (Hx.getEnd ());

  GridCoordinate3D HyStart = yeeLayout.getHyStart (Hy.getStart ());
  GridCoordinate3D HyEnd = yeeLayout.getHyEnd (Hy.getEnd ());

  dim3 blocksEz (EzSizeCoord.getX () / 16, EzSizeCoord.getY () / 16);
  dim3 threadsEz (16, 16);

  dim3 blocksHx (HxSizeCoord.getX () / 16, HxSizeCoord.getY () / 16);
  dim3 threadsHx (16, 16);

  dim3 blocksHy (HySizeCoord.getX () / 16, HySizeCoord.getY () / 16);
  dim3 threadsHy (16, 16);

  for (time_step t = stepStart; t < stepEnd; ++t)
  {
    cudaCheckExitStatus (cudaCalculateTMzEzStep <<< blocksEz, threadsEz >>> (exitStatusCuda,
                                                                             Ez_cuda,
                                                                             Ez_cuda_prev,
                                                                             Hx_cuda_prev,
                                                                             Hy_cuda_prev,
                                                                             eps_cuda,
                                                                             gridTimeStep,
                                                                             gridStep,
                                                                             EzStart.getX (),
                                                                             EzStart.getY (),
                                                                             EzEnd.getX (),
                                                                             EzEnd.getY (),
                                                                             EzSizeCoord.getX (),
                                                                             EzSizeCoord.getY (),
                                                                             t));

    cudaCheckExitStatus (cudaCalculateTMzEzSource <<< blocksEz, threadsEz >>> (exitStatusCuda,
                                                                               Ez_cuda_prev,
                                                                               EzStart.getX (),
                                                                               EzStart.getY (),
                                                                               EzEnd.getX (),
                                                                               EzEnd.getY (),
                                                                               EzSizeCoord.getX (),
                                                                               EzSizeCoord.getY (),
                                                                               t));

    cudaCheckExitStatus (cudaCalculateTMzHxStep <<< blocksHx, threadsHx >>> (exitStatusCuda,
                                                                             Hx_cuda,
                                                                             Ez_cuda_prev,
                                                                             Hx_cuda_prev,
                                                                             mu_cuda,
                                                                             gridTimeStep,
                                                                             gridStep,
                                                                             HxStart.getX (),
                                                                             HxStart.getY (),
                                                                             HxEnd.getX (),
                                                                             HxEnd.getY (),
                                                                             HxSizeCoord.getX (),
                                                                             HxSizeCoord.getY (),
                                                                             t));

    cudaCheckExitStatus (cudaCalculateTMzHyStep <<< blocksHy, threadsHy >>> (exitStatusCuda,
                                                                             Hy_cuda,
                                                                             Ez_cuda_prev,
                                                                             Hy_cuda_prev,
                                                                             mu_cuda,
                                                                             gridTimeStep,
                                                                             gridStep,
                                                                             HyStart.getX (),
                                                                             HyStart.getY (),
                                                                             HyEnd.getX (),
                                                                             HyEnd.getY (),
                                                                             HySizeCoord.getX (),
                                                                             HySizeCoord.getY (),
                                                                             t));

    cudaCheckExitStatus (cudaCalculateTMzHxSource <<< blocksHx, threadsHx >>> (exitStatusCuda,
                                                                               Hx_cuda_prev,
                                                                               HxStart.getX (),
                                                                               HxStart.getY (),
                                                                               HxEnd.getX (),
                                                                               HxEnd.getY (),
                                                                               HxSizeCoord.getX (),
                                                                               HxSizeCoord.getY (),
                                                                               t));

    cudaCheckExitStatus (cudaCalculateTMzHySource <<< blocksHy, threadsHy >>> (exitStatusCuda,
                                                                               Hy_cuda_prev,
                                                                               HyStart.getX (),
                                                                               HyStart.getY (),
                                                                               HyEnd.getX (),
                                                                               HyEnd.getY (),
                                                                               HySizeCoord.getX (),
                                                                               HySizeCoord.getY (),
                                                                               t));
  }

  cudaCheckErrorCmd (cudaMemcpy (tmp_Ez, Ez_cuda, sizeEzRaw, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (tmp_Hx, Hx_cuda, sizeHxRaw, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (tmp_Hy, Hy_cuda, sizeHyRaw, cudaMemcpyDeviceToHost));

  cudaCheckErrorCmd (cudaMemcpy (tmp_Ez_prev, Ez_cuda_prev, sizeEzRaw, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (tmp_Hx_prev, Hx_cuda_prev, sizeHxRaw, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (tmp_Hy_prev, Hy_cuda_prev, sizeHyRaw, cudaMemcpyDeviceToHost));

  cudaCheckErrorCmd (cudaMemcpy (tmp_eps, eps_cuda, sizeEpsRaw, cudaMemcpyDeviceToHost));
  cudaCheckErrorCmd (cudaMemcpy (tmp_mu, mu_cuda, sizeMuRaw, cudaMemcpyDeviceToHost));

  cudaCheckErrorCmd (cudaFree (Ez_cuda));
  cudaCheckErrorCmd (cudaFree (Hx_cuda));
  cudaCheckErrorCmd (cudaFree (Hy_cuda));

  cudaCheckErrorCmd (cudaFree (Ez_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Hx_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Hy_cuda_prev));

  cudaCheckErrorCmd (cudaFree (eps_cuda));
  cudaCheckErrorCmd (cudaFree (mu_cuda));

  for (grid_iter i = 0; i < sizeEz; ++i)
  {
    FieldPointValue* valEz = Ez.getFieldPointValue (i);
    valEz->setCurValue (tmp_Ez[i]);
    valEz->setPrevValue (tmp_Ez_prev[i]);
  }

  for (grid_iter i = 0; i < sizeHx; ++i)
  {
    FieldPointValue* valHx = Hx.getFieldPointValue (i);
    valHx->setCurValue (tmp_Hx[i]);
    valHx->setPrevValue (tmp_Hx_prev[i]);
  }

  for (grid_iter i = 0; i < sizeHy; ++i)
  {
    FieldPointValue* valHy = Hy.getFieldPointValue (i);
    valHy->setCurValue (tmp_Hy[i]);
    valHy->setPrevValue (tmp_Hy_prev[i]);
  }

  delete[] tmp_Ez;
  delete[] tmp_Hx;
  delete[] tmp_Hy;

  delete[] tmp_Ez_prev;
  delete[] tmp_Hx_prev;
  delete[] tmp_Hy_prev;

  delete[] tmp_eps;
  delete[] tmp_mu;

  *retval = CUDA_OK;
  return;
}

#endif
