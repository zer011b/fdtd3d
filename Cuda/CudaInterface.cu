#include "CudaInterface.h"
#include "CudaGlobalKernels.h"

extern int cudaThreadsX;
extern int cudaThreadsY;
extern int cudaThreadsZ;

#ifdef PARALLEL_GRID
void cudaExecute2DTMzSteps (CudaExitStatus *retval,
                            YeeGridLayout &yeeLayout,
                            FieldValue gridTimeStep, FieldValue gridStep,
                            ParallelGrid &Ez,
                            ParallelGrid &Hx,
                            ParallelGrid &Hy,
                            ParallelGrid &Eps,
                            ParallelGrid &Mu,
                            time_step totalStep,
                            int processId)
#else
void cudaExecute2DTMzSteps (CudaExitStatus *retval,
                            YeeGridLayout &yeeLayout,
                            FieldValue gridTimeStep, FieldValue gridStep,
                            Grid<GridCoordinate2D> &Ez,
                            Grid<GridCoordinate2D> &Hx,
                            Grid<GridCoordinate2D> &Hy,
                            Grid<GridCoordinate2D> &Eps,
                            Grid<GridCoordinate2D> &Mu,
                            time_step totalStep,
                            int processId)
#endif
{
  time_step t = 0;

#ifdef PARALLEL_GRID
  GridCoordinate2D bufSize = Ez.getBufferSize ();

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  time_step shareStep = bufSize.getX ();
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  time_step shareStep = bufSize.getY ();
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  time_step shareStep = bufSize.getZ ();
#endif
#else
  time_step shareStep = totalStep;
#endif

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

  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  dim3 blocksEz (EzSizeCoord.getX () / cudaThreadsX, EzSizeCoord.getY () / cudaThreadsY);
  dim3 threadsEz (cudaThreadsX, cudaThreadsY);

  dim3 blocksHx (HxSizeCoord.getX () / cudaThreadsX, HxSizeCoord.getY () / cudaThreadsY);
  dim3 threadsHx (cudaThreadsX, cudaThreadsY);

  dim3 blocksHy (HySizeCoord.getX () / cudaThreadsX, HySizeCoord.getY () / cudaThreadsY);
  dim3 threadsHy (cudaThreadsX, cudaThreadsY);

  while (t < totalStep)
  {
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

    cudaCheckErrorCmd (cudaMemcpy (Ez_cuda, tmp_Ez, sizeEzRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (Hx_cuda, tmp_Hx, sizeHxRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (Hy_cuda, tmp_Hy, sizeHyRaw, cudaMemcpyHostToDevice));

    cudaCheckErrorCmd (cudaMemcpy (Ez_cuda_prev, tmp_Ez_prev, sizeEzRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (Hx_cuda_prev, tmp_Hx_prev, sizeHxRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (Hy_cuda_prev, tmp_Hy_prev, sizeHyRaw, cudaMemcpyHostToDevice));

    cudaCheckErrorCmd (cudaMemcpy (eps_cuda, tmp_eps, sizeEpsRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (mu_cuda, tmp_mu, sizeMuRaw, cudaMemcpyHostToDevice));

#if not defined (PARALLEL_GRID)
    GridCoordinate3D EzStart = yeeLayout.getEzStart (Ez.getStart ());
    GridCoordinate3D EzEnd = yeeLayout.getEzEnd (Ez.getEnd ());

    GridCoordinate3D HxStart = yeeLayout.getHxStart (Hx.getStart ());
    GridCoordinate3D HxEnd = yeeLayout.getHxEnd (Hx.getEnd ());

    GridCoordinate3D HyStart = yeeLayout.getHyStart (Hy.getStart ());
    GridCoordinate3D HyEnd = yeeLayout.getHyEnd (Hy.getEnd ());
#endif

    for (time_step stepEnd = t + shareStep; t < stepEnd; ++t)
    {
#if defined (PARALLEL_GRID)
      GridCoordinate3D EzStart = yeeLayout.getEzStart (Ez.getStart ());
      GridCoordinate3D EzEnd = yeeLayout.getEzEnd (Ez.getEnd ());

      GridCoordinate3D HxStart = yeeLayout.getHxStart (Hx.getStart ());
      GridCoordinate3D HxEnd = yeeLayout.getHxEnd (Hx.getEnd ());

      GridCoordinate3D HyStart = yeeLayout.getHyStart (Hy.getStart ());
      GridCoordinate3D HyEnd = yeeLayout.getHyEnd (Hy.getEnd ());
#endif

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
                                                                                 t,
                                                                                 processId));

#if defined (PARALLEL_GRID)
      Ez.nextShareStep ();
#endif

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

#if defined (PARALLEL_GRID)
      Hx.nextShareStep ();
      Hy.nextShareStep ();
#endif
    }

    cudaCheckErrorCmd (cudaMemcpy (tmp_Ez, Ez_cuda, sizeEzRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_Hx, Hx_cuda, sizeHxRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_Hy, Hy_cuda, sizeHyRaw, cudaMemcpyDeviceToHost));

    cudaCheckErrorCmd (cudaMemcpy (tmp_Ez_prev, Ez_cuda_prev, sizeEzRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_Hx_prev, Hx_cuda_prev, sizeHxRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_Hy_prev, Hy_cuda_prev, sizeHyRaw, cudaMemcpyDeviceToHost));

    cudaCheckErrorCmd (cudaMemcpy (tmp_eps, eps_cuda, sizeEpsRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_mu, mu_cuda, sizeMuRaw, cudaMemcpyDeviceToHost));

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

#if defined (PARALLEL_GRID)
    Ez.zeroShareStep ();
    Ez.share ();

    Hx.zeroShareStep ();
    Hx.share ();

    Hy.zeroShareStep ();
    Hy.share ();
#endif /* PARALLEL_GRID */
  }

  cudaCheckErrorCmd (cudaFree (Ez_cuda));
  cudaCheckErrorCmd (cudaFree (Hx_cuda));
  cudaCheckErrorCmd (cudaFree (Hy_cuda));

  cudaCheckErrorCmd (cudaFree (Ez_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Hx_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Hy_cuda_prev));

  cudaCheckErrorCmd (cudaFree (eps_cuda));
  cudaCheckErrorCmd (cudaFree (mu_cuda));

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

#ifdef PARALLEL_GRID
void cudaExecute3DSteps (CudaExitStatus *retval,
                         YeeGridLayout &yeeLayout,
                         FieldValue gridTimeStep, FieldValue gridStep,
                         ParallelGrid &Ex,
                         ParallelGrid &Ey,
                         ParallelGrid &Ez,
                         ParallelGrid &Hx,
                         ParallelGrid &Hy,
                         ParallelGrid &Hz,
                         ParallelGrid &Eps,
                         ParallelGrid &Mu,
                         time_step totalStep,
                         int processId)
#else
void cudaExecute3DSteps (CudaExitStatus *retval,
                         YeeGridLayout &yeeLayout,
                         FieldValue gridTimeStep, FieldValue gridStep,
                         Grid<GridCoordinate3D> &Ex,
                         Grid<GridCoordinate3D> &Ey,
                         Grid<GridCoordinate3D> &Ez,
                         Grid<GridCoordinate3D> &Hx,
                         Grid<GridCoordinate3D> &Hy,
                         Grid<GridCoordinate3D> &Hz,
                         Grid<GridCoordinate3D> &Eps,
                         Grid<GridCoordinate3D> &Mu,
                         time_step totalStep,
                         int processId)
#endif
{
  time_step t = 0;

#ifdef PARALLEL_GRID
  GridCoordinate3D bufSize = Ex.getBufferSize ();

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  time_step shareStep = bufSize.getX ();
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  time_step shareStep = bufSize.getY ();
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  time_step shareStep = bufSize.getZ ();
#endif
#else
  time_step shareStep = totalStep;
#endif

  GridCoordinate3D ExSizeCoord = Ex.getSize ();
  GridCoordinate3D EySizeCoord = Ey.getSize ();
  GridCoordinate3D EzSizeCoord = Ez.getSize ();
  GridCoordinate3D HxSizeCoord = Hx.getSize ();
  GridCoordinate3D HySizeCoord = Hy.getSize ();
  GridCoordinate3D HzSizeCoord = Hz.getSize ();
  GridCoordinate3D EpsSizeCoord = Eps.getSize ();
  GridCoordinate3D MuSizeCoord = Mu.getSize ();

  grid_iter sizeEx = ExSizeCoord.calculateTotalCoord();
  grid_iter sizeEy = EySizeCoord.calculateTotalCoord();
  grid_iter sizeEz = EzSizeCoord.calculateTotalCoord();
  grid_iter sizeHx = HxSizeCoord.calculateTotalCoord();
  grid_iter sizeHy = HySizeCoord.calculateTotalCoord();
  grid_iter sizeHz = HzSizeCoord.calculateTotalCoord();
  grid_iter sizeEps = EpsSizeCoord.calculateTotalCoord();
  grid_iter sizeMu = MuSizeCoord.calculateTotalCoord();

  grid_iter sizeExRaw = (grid_iter) sizeEx * sizeof (FieldValue);
  grid_iter sizeEyRaw = (grid_iter) sizeEy * sizeof (FieldValue);
  grid_iter sizeEzRaw = (grid_iter) sizeEz * sizeof (FieldValue);
  grid_iter sizeHxRaw = (grid_iter) sizeHx * sizeof (FieldValue);
  grid_iter sizeHyRaw = (grid_iter) sizeHy * sizeof (FieldValue);
  grid_iter sizeHzRaw = (grid_iter) sizeHz * sizeof (FieldValue);
  grid_iter sizeEpsRaw = (grid_iter) sizeEps * sizeof (FieldValue);
  grid_iter sizeMuRaw = (grid_iter) sizeMu * sizeof (FieldValue);

  FieldValue *tmp_Ex = new FieldValue [sizeEx];
  FieldValue *tmp_Ey = new FieldValue [sizeEy];
  FieldValue *tmp_Ez = new FieldValue [sizeEz];
  FieldValue *tmp_Hx = new FieldValue [sizeHx];
  FieldValue *tmp_Hy = new FieldValue [sizeHy];
  FieldValue *tmp_Hz = new FieldValue [sizeHz];

  FieldValue *tmp_Ex_prev = new FieldValue [sizeEx];
  FieldValue *tmp_Ey_prev = new FieldValue [sizeEy];
  FieldValue *tmp_Ez_prev = new FieldValue [sizeEz];
  FieldValue *tmp_Hx_prev = new FieldValue [sizeHx];
  FieldValue *tmp_Hy_prev = new FieldValue [sizeHy];
  FieldValue *tmp_Hz_prev = new FieldValue [sizeHz];

  FieldValue *tmp_eps = new FieldValue [sizeEps];
  FieldValue *tmp_mu = new FieldValue [sizeMu];

  FieldValue *Ex_cuda;
  FieldValue *Ey_cuda;
  FieldValue *Ez_cuda;
  FieldValue *Hx_cuda;
  FieldValue *Hy_cuda;
  FieldValue *Hz_cuda;

  FieldValue *Ex_cuda_prev;
  FieldValue *Ey_cuda_prev;
  FieldValue *Ez_cuda_prev;
  FieldValue *Hx_cuda_prev;
  FieldValue *Hy_cuda_prev;
  FieldValue *Hz_cuda_prev;

  FieldValue *eps_cuda;
  FieldValue *mu_cuda;

  cudaCheckErrorCmd (cudaMalloc ((void **) &Ex_cuda, sizeExRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Ey_cuda, sizeEyRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Ez_cuda, sizeEzRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hx_cuda, sizeHxRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hy_cuda, sizeHyRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hz_cuda, sizeHzRaw));

  cudaCheckErrorCmd (cudaMalloc ((void **) &Ex_cuda_prev, sizeExRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Ey_cuda_prev, sizeEyRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Ez_cuda_prev, sizeEzRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hx_cuda_prev, sizeHxRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hy_cuda_prev, sizeHyRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &Hz_cuda_prev, sizeHzRaw));

  cudaCheckErrorCmd (cudaMalloc ((void **) &eps_cuda, sizeEpsRaw));
  cudaCheckErrorCmd (cudaMalloc ((void **) &mu_cuda, sizeMuRaw));

  CudaExitStatus exitStatus;
  CudaExitStatus *exitStatusCuda;
  cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));

  dim3 blocksEx (ExSizeCoord.getX () / cudaThreadsX, ExSizeCoord.getY () / cudaThreadsY, ExSizeCoord.getZ () / cudaThreadsZ);
  dim3 threadsEx (cudaThreadsX, cudaThreadsY, cudaThreadsZ);

  dim3 blocksEy (EySizeCoord.getX () / cudaThreadsX, EySizeCoord.getY () / cudaThreadsY, EySizeCoord.getZ () / cudaThreadsZ);
  dim3 threadsEy (cudaThreadsX, cudaThreadsY, cudaThreadsZ);

  dim3 blocksEz (EzSizeCoord.getX () / cudaThreadsX, EzSizeCoord.getY () / cudaThreadsY, EzSizeCoord.getZ () / cudaThreadsZ);
  dim3 threadsEz (cudaThreadsX, cudaThreadsY, cudaThreadsZ);

  dim3 blocksHx (HxSizeCoord.getX () / cudaThreadsX, HxSizeCoord.getY () / cudaThreadsY, HxSizeCoord.getZ () / cudaThreadsZ);
  dim3 threadsHx (cudaThreadsX, cudaThreadsY, cudaThreadsZ);

  dim3 blocksHy (HySizeCoord.getX () / cudaThreadsX, HySizeCoord.getY () / cudaThreadsY, HySizeCoord.getZ () / cudaThreadsZ);
  dim3 threadsHy (cudaThreadsX, cudaThreadsY, cudaThreadsZ);

  dim3 blocksHz (HzSizeCoord.getX () / cudaThreadsX, HzSizeCoord.getY () / cudaThreadsY, HzSizeCoord.getZ () / cudaThreadsZ);
  dim3 threadsHz (cudaThreadsX, cudaThreadsY, cudaThreadsZ);

  while (t < totalStep)
  {
    for (grid_iter i = 0; i < sizeEx; ++i)
    {
      FieldPointValue* valEx = Ex.getFieldPointValue (i);
      tmp_Ex[i] = valEx->getCurValue ();
      tmp_Ex_prev[i] = valEx->getPrevValue ();
    }

    for (grid_iter i = 0; i < sizeEy; ++i)
    {
      FieldPointValue* valEy = Ey.getFieldPointValue (i);
      tmp_Ey[i] = valEy->getCurValue ();
      tmp_Ey_prev[i] = valEy->getPrevValue ();
    }

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

    for (grid_iter i = 0; i < sizeHz; ++i)
    {
      FieldPointValue* valHz = Hz.getFieldPointValue (i);
      tmp_Hz[i] = valHz->getCurValue ();
      tmp_Hz_prev[i] = valHz->getPrevValue ();
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

    cudaCheckErrorCmd (cudaMemcpy (Ex_cuda, tmp_Ex, sizeExRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (Ey_cuda, tmp_Ey, sizeEyRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (Ez_cuda, tmp_Ez, sizeEzRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (Hx_cuda, tmp_Hx, sizeHxRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (Hy_cuda, tmp_Hy, sizeHyRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (Hz_cuda, tmp_Hz, sizeHzRaw, cudaMemcpyHostToDevice));

    cudaCheckErrorCmd (cudaMemcpy (Ex_cuda_prev, tmp_Ex_prev, sizeExRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (Ey_cuda_prev, tmp_Ey_prev, sizeEyRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (Ez_cuda_prev, tmp_Ez_prev, sizeEzRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (Hx_cuda_prev, tmp_Hx_prev, sizeHxRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (Hy_cuda_prev, tmp_Hy_prev, sizeHyRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (Hz_cuda_prev, tmp_Hz_prev, sizeHzRaw, cudaMemcpyHostToDevice));

    cudaCheckErrorCmd (cudaMemcpy (eps_cuda, tmp_eps, sizeEpsRaw, cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (mu_cuda, tmp_mu, sizeMuRaw, cudaMemcpyHostToDevice));

#if not defined (PARALLEL_GRID)
    GridCoordinate3D ExStart = yeeLayout.getExStart (Ex.getStart ());
    GridCoordinate3D ExEnd = yeeLayout.getExEnd (Ex.getSize ());

    GridCoordinate3D EyStart = yeeLayout.getEyStart (Ey.getStart ());
    GridCoordinate3D EyEnd = yeeLayout.getEyEnd (Ey.getSize ());

    GridCoordinate3D EzStart = yeeLayout.getEzStart (Ez.getStart ());
    GridCoordinate3D EzEnd = yeeLayout.getEzEnd (Ez.getSize ());

    GridCoordinate3D HxStart = yeeLayout.getHxStart (Hx.getStart ());
    GridCoordinate3D HxEnd = yeeLayout.getHxEnd (Hx.getSize ());

    GridCoordinate3D HyStart = yeeLayout.getHyStart (Hy.getStart ());
    GridCoordinate3D HyEnd = yeeLayout.getHyEnd (Hy.getSize ());

    GridCoordinate3D HzStart = yeeLayout.getHzStart (Hz.getStart ());
    GridCoordinate3D HzEnd = yeeLayout.getHzEnd (Hz.getSize ());
#endif

    for (time_step stepEnd = t + shareStep; t < stepEnd; ++t)
    {
#if defined (PARALLEL_GRID)
      GridCoordinate3D ExStart = yeeLayout.getExStart (Ex.getStart ());
      GridCoordinate3D ExEnd = yeeLayout.getExEnd (Ex.getSize ());

      GridCoordinate3D EyStart = yeeLayout.getEyStart (Ey.getStart ());
      GridCoordinate3D EyEnd = yeeLayout.getEyEnd (Ey.getSize ());

      GridCoordinate3D EzStart = yeeLayout.getEzStart (Ez.getStart ());
      GridCoordinate3D EzEnd = yeeLayout.getEzEnd (Ez.getSize ());

      GridCoordinate3D HxStart = yeeLayout.getHxStart (Hx.getStart ());
      GridCoordinate3D HxEnd = yeeLayout.getHxEnd (Hx.getSize ());

      GridCoordinate3D HyStart = yeeLayout.getHyStart (Hy.getStart ());
      GridCoordinate3D HyEnd = yeeLayout.getHyEnd (Hy.getSize ());

      GridCoordinate3D HzStart = yeeLayout.getHzStart (Hz.getStart ());
      GridCoordinate3D HzEnd = yeeLayout.getHzEnd (Hz.getSize ());
#endif

      cudaCheckExitStatus (cudaCalculate3DExStep <<< blocksEx, threadsEx >>> (exitStatusCuda,
                                                                              Ex_cuda,
                                                                              Ex_cuda_prev,
                                                                              Hy_cuda_prev,
                                                                              Hz_cuda_prev,
                                                                              eps_cuda,
                                                                              gridTimeStep,
                                                                              gridStep,
                                                                              ExStart.getX (),
                                                                              ExStart.getY (),
                                                                              ExStart.getZ (),
                                                                              ExEnd.getX (),
                                                                              ExEnd.getY (),
                                                                              ExEnd.getZ (),
                                                                              ExSizeCoord.getX (),
                                                                              ExSizeCoord.getY (),
                                                                              ExSizeCoord.getZ (),
                                                                              t));

      cudaCheckExitStatus (cudaCalculate3DEyStep <<< blocksEy, threadsEy >>> (exitStatusCuda,
                                                                              Ey_cuda,
                                                                              Ey_cuda_prev,
                                                                              Hx_cuda_prev,
                                                                              Hz_cuda_prev,
                                                                              eps_cuda,
                                                                              gridTimeStep,
                                                                              gridStep,
                                                                              EyStart.getX (),
                                                                              EyStart.getY (),
                                                                              EyStart.getZ (),
                                                                              EyEnd.getX (),
                                                                              EyEnd.getY (),
                                                                              EyEnd.getZ (),
                                                                              EySizeCoord.getX (),
                                                                              EySizeCoord.getY (),
                                                                              EySizeCoord.getZ (),
                                                                              t));

      cudaCheckExitStatus (cudaCalculate3DEzStep <<< blocksEz, threadsEz >>> (exitStatusCuda,
                                                                              Ez_cuda,
                                                                              Ez_cuda_prev,
                                                                              Hx_cuda_prev,
                                                                              Hy_cuda_prev,
                                                                              eps_cuda,
                                                                              gridTimeStep,
                                                                              gridStep,
                                                                              EzStart.getX (),
                                                                              EzStart.getY (),
                                                                              EzStart.getZ (),
                                                                              EzEnd.getX (),
                                                                              EzEnd.getY (),
                                                                              EzEnd.getZ (),
                                                                              EzSizeCoord.getX (),
                                                                              EzSizeCoord.getY (),
                                                                              EzSizeCoord.getZ (),
                                                                              t));

      cudaCheckExitStatus (cudaCalculate3DExSource <<< blocksEx, threadsEx >>> (exitStatusCuda,
                                                                                Ex_cuda_prev,
                                                                                ExStart.getX (),
                                                                                ExStart.getY (),
                                                                                ExStart.getZ (),
                                                                                ExEnd.getX (),
                                                                                ExEnd.getY (),
                                                                                ExEnd.getZ (),
                                                                                ExSizeCoord.getX (),
                                                                                ExSizeCoord.getY (),
                                                                                ExSizeCoord.getZ (),
                                                                                t,
                                                                                processId));

      cudaCheckExitStatus (cudaCalculate3DEySource <<< blocksEy, threadsEy >>> (exitStatusCuda,
                                                                                Ey_cuda_prev,
                                                                                EyStart.getX (),
                                                                                EyStart.getY (),
                                                                                EyStart.getZ (),
                                                                                EyEnd.getX (),
                                                                                EyEnd.getY (),
                                                                                EyEnd.getZ (),
                                                                                EySizeCoord.getX (),
                                                                                EySizeCoord.getY (),
                                                                                EySizeCoord.getZ (),
                                                                                t,
                                                                                processId));

      cudaCheckExitStatus (cudaCalculate3DEzSource <<< blocksEz, threadsEz >>> (exitStatusCuda,
                                                                                Ez_cuda_prev,
                                                                                EzStart.getX (),
                                                                                EzStart.getY (),
                                                                                EzStart.getZ (),
                                                                                EzEnd.getX (),
                                                                                EzEnd.getY (),
                                                                                EzEnd.getZ (),
                                                                                EzSizeCoord.getX (),
                                                                                EzSizeCoord.getY (),
                                                                                EzSizeCoord.getZ (),
                                                                                t,
                                                                                processId));

#if defined (PARALLEL_GRID)
      Ex.nextShareStep ();
      Ey.nextShareStep ();
      Ez.nextShareStep ();
#endif

      cudaCheckExitStatus (cudaCalculate3DHxStep <<< blocksHx, threadsHx >>> (exitStatusCuda,
                                                                              Hx_cuda,
                                                                              Hx_cuda_prev,
                                                                              Ey_cuda_prev,
                                                                              Ez_cuda_prev,
                                                                              mu_cuda,
                                                                              gridTimeStep,
                                                                              gridStep,
                                                                              HxStart.getX (),
                                                                              HxStart.getY (),
                                                                              HxStart.getZ (),
                                                                              HxEnd.getX (),
                                                                              HxEnd.getY (),
                                                                              HxEnd.getZ (),
                                                                              HxSizeCoord.getX (),
                                                                              HxSizeCoord.getY (),
                                                                              HxSizeCoord.getZ (),
                                                                              t));

      cudaCheckExitStatus (cudaCalculate3DHyStep <<< blocksHy, threadsHy >>> (exitStatusCuda,
                                                                              Hy_cuda,
                                                                              Hy_cuda_prev,
                                                                              Ex_cuda_prev,
                                                                              Ez_cuda_prev,
                                                                              mu_cuda,
                                                                              gridTimeStep,
                                                                              gridStep,
                                                                              HyStart.getX (),
                                                                              HyStart.getY (),
                                                                              HyStart.getZ (),
                                                                              HyEnd.getX (),
                                                                              HyEnd.getY (),
                                                                              HyEnd.getZ (),
                                                                              HySizeCoord.getX (),
                                                                              HySizeCoord.getY (),
                                                                              HySizeCoord.getZ (),
                                                                              t));

      cudaCheckExitStatus (cudaCalculate3DHzStep <<< blocksHz, threadsHz >>> (exitStatusCuda,
                                                                              Hz_cuda,
                                                                              Hz_cuda_prev,
                                                                              Ex_cuda_prev,
                                                                              Ey_cuda_prev,
                                                                              mu_cuda,
                                                                              gridTimeStep,
                                                                              gridStep,
                                                                              HzStart.getX (),
                                                                              HzStart.getY (),
                                                                              HzStart.getZ (),
                                                                              HzEnd.getX (),
                                                                              HzEnd.getY (),
                                                                              HzEnd.getZ (),
                                                                              HzSizeCoord.getX (),
                                                                              HzSizeCoord.getY (),
                                                                              HzSizeCoord.getZ (),
                                                                              t));

      cudaCheckExitStatus (cudaCalculate3DHxSource <<< blocksHx, threadsHx >>> (exitStatusCuda,
                                                                                Hx_cuda_prev,
                                                                                HxStart.getX (),
                                                                                HxStart.getY (),
                                                                                HxStart.getZ (),
                                                                                HxEnd.getX (),
                                                                                HxEnd.getY (),
                                                                                HxEnd.getZ (),
                                                                                HxSizeCoord.getX (),
                                                                                HxSizeCoord.getY (),
                                                                                HxSizeCoord.getZ (),
                                                                                t,
                                                                                processId));

      cudaCheckExitStatus (cudaCalculate3DHySource <<< blocksHy, threadsHy >>> (exitStatusCuda,
                                                                                Hy_cuda_prev,
                                                                                HyStart.getX (),
                                                                                HyStart.getY (),
                                                                                HyStart.getZ (),
                                                                                HyEnd.getX (),
                                                                                HyEnd.getY (),
                                                                                HyEnd.getZ (),
                                                                                HySizeCoord.getX (),
                                                                                HySizeCoord.getY (),
                                                                                HySizeCoord.getZ (),
                                                                                t,
                                                                                processId));

      cudaCheckExitStatus (cudaCalculate3DHzSource <<< blocksHz, threadsHz >>> (exitStatusCuda,
                                                                                Hz_cuda_prev,
                                                                                HzStart.getX (),
                                                                                HzStart.getY (),
                                                                                HzStart.getZ (),
                                                                                HzEnd.getX (),
                                                                                HzEnd.getY (),
                                                                                HzEnd.getZ (),
                                                                                HzSizeCoord.getX (),
                                                                                HzSizeCoord.getY (),
                                                                                HzSizeCoord.getZ (),
                                                                                t,
                                                                                processId));

#if defined (PARALLEL_GRID)
      Hx.nextShareStep ();
      Hy.nextShareStep ();
      Hz.nextShareStep ();
#endif
    }

    cudaCheckErrorCmd (cudaMemcpy (tmp_Ex, Ex_cuda, sizeExRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_Ey, Ey_cuda, sizeEyRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_Ez, Ez_cuda, sizeEzRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_Hx, Hx_cuda, sizeHxRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_Hy, Hy_cuda, sizeHyRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_Hz, Hz_cuda, sizeHzRaw, cudaMemcpyDeviceToHost));

    cudaCheckErrorCmd (cudaMemcpy (tmp_Ex_prev, Ex_cuda_prev, sizeExRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_Ey_prev, Ey_cuda_prev, sizeEyRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_Ez_prev, Ez_cuda_prev, sizeEzRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_Hx_prev, Hx_cuda_prev, sizeHxRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_Hy_prev, Hy_cuda_prev, sizeHyRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_Hz_prev, Hz_cuda_prev, sizeHzRaw, cudaMemcpyDeviceToHost));

    cudaCheckErrorCmd (cudaMemcpy (tmp_eps, eps_cuda, sizeEpsRaw, cudaMemcpyDeviceToHost));
    cudaCheckErrorCmd (cudaMemcpy (tmp_mu, mu_cuda, sizeMuRaw, cudaMemcpyDeviceToHost));

    for (grid_iter i = 0; i < sizeEx; ++i)
    {
      FieldPointValue* valEx = Ex.getFieldPointValue (i);
      valEx->setCurValue (tmp_Ex[i]);
      valEx->setPrevValue (tmp_Ex_prev[i]);
    }

    for (grid_iter i = 0; i < sizeEy; ++i)
    {
      FieldPointValue* valEy = Ey.getFieldPointValue (i);
      valEy->setCurValue (tmp_Ey[i]);
      valEy->setPrevValue (tmp_Ey_prev[i]);
    }

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

    for (grid_iter i = 0; i < sizeHz; ++i)
    {
      FieldPointValue* valHz = Hz.getFieldPointValue (i);
      valHz->setCurValue (tmp_Hz[i]);
      valHz->setPrevValue (tmp_Hz_prev[i]);
    }

#if defined (PARALLEL_GRID)
    Ex.zeroShareStep ();
    Ex.share ();

    Ey.zeroShareStep ();
    Ey.share ();

    Ez.zeroShareStep ();
    Ez.share ();

    Hx.zeroShareStep ();
    Hx.share ();

    Hy.zeroShareStep ();
    Hy.share ();

    Hz.zeroShareStep ();
    Hz.share ();
#endif /* PARALLEL_GRID */
  }

  cudaCheckErrorCmd (cudaFree (Ex_cuda));
  cudaCheckErrorCmd (cudaFree (Ey_cuda));
  cudaCheckErrorCmd (cudaFree (Ez_cuda));
  cudaCheckErrorCmd (cudaFree (Hx_cuda));
  cudaCheckErrorCmd (cudaFree (Hy_cuda));
  cudaCheckErrorCmd (cudaFree (Hz_cuda));

  cudaCheckErrorCmd (cudaFree (Ex_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Ey_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Ez_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Hx_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Hy_cuda_prev));
  cudaCheckErrorCmd (cudaFree (Hz_cuda_prev));

  cudaCheckErrorCmd (cudaFree (eps_cuda));
  cudaCheckErrorCmd (cudaFree (mu_cuda));

  delete[] tmp_Ex;
  delete[] tmp_Ey;
  delete[] tmp_Ez;
  delete[] tmp_Hx;
  delete[] tmp_Hy;
  delete[] tmp_Hz;

  delete[] tmp_Ex_prev;
  delete[] tmp_Ey_prev;
  delete[] tmp_Ez_prev;
  delete[] tmp_Hx_prev;
  delete[] tmp_Hy_prev;
  delete[] tmp_Hz_prev;

  delete[] tmp_eps;
  delete[] tmp_mu;

  *retval = CUDA_OK;
  return;
}

void cudaInit (int rank)
{
  cudaCheckErrorCmd (cudaSetDevice(rank));
}

void cudaInfo ()
{
  int cudaDevicesCount;

  cudaCheckErrorCmd (cudaGetDeviceCount (&cudaDevicesCount));

  for (int i = 0; i < cudaDevicesCount; i++)
  {
    cudaDeviceProp prop;
    cudaCheckErrorCmd (cudaGetDeviceProperties(&prop, i));
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Concurrent kernels number: %d\n", prop.concurrentKernels);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Total global mem: %llu\n", (long long unsigned) prop.totalGlobalMem);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}
