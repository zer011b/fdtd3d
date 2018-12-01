#include "CudaInterface.h"
#include "CudaGlobalKernels.h"

extern int cudaThreadsX;
extern int cudaThreadsY;
extern int cudaThreadsZ;
//
// #ifdef PARALLEL_GRID
// void cudaExecute2DTMzSteps (CudaExitStatus *retval,
//                             YeeGridLayout *yeeLayout,
//                             FPValue gridTimeStep, FPValue gridStep,
//                             ParallelGrid &Ez,
//                             ParallelGrid &Hx,
//                             ParallelGrid &Hy,
//                             ParallelGrid &Eps,
//                             ParallelGrid &Mu,
//                             time_step totalStep,
//                             int processId)
// #else
// void cudaExecute2DTMzSteps (CudaExitStatus *retval,
//                             YeeGridLayout *yeeLayout,
//                             FPValue gridTimeStep, FPValue gridStep,
//                             Grid<GridCoordinate2D> &Ez,
//                             Grid<GridCoordinate2D> &Hx,
//                             Grid<GridCoordinate2D> &Hy,
//                             Grid<GridCoordinate2D> &Eps,
//                             Grid<GridCoordinate2D> &Mu,
//                             time_step totalStep,
//                             int processId)
// #endif
// {
//   time_step t = 0;
//
// #ifdef PARALLEL_GRID
//   GridCoordinate2D bufSize = Ez.getBufferSize ();
//
// #if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
//     defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
//   time_step shareStep = bufSize.get1 ();
// #endif
// #if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
//   time_step shareStep = bufSize.get2 ();
// #endif
// #if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
//   time_step shareStep = bufSize.get3 ();
// #endif
// #else
//   time_step shareStep = totalStep;
// #endif
//
//   GridCoordinate2D EzSizeCoord = Ez.getSize ();
//   GridCoordinate2D HxSizeCoord = Hx.getSize ();
//   GridCoordinate2D HySizeCoord = Hy.getSize ();
//   GridCoordinate2D EpsSizeCoord = Eps.getSize ();
//   GridCoordinate2D MuSizeCoord = Mu.getSize ();
//
//   grid_coord sizeEz = EzSizeCoord.calculateTotalCoord();
//   grid_coord sizeHx = HxSizeCoord.calculateTotalCoord();
//   grid_coord sizeHy = HySizeCoord.calculateTotalCoord();
//   grid_coord sizeEps = EpsSizeCoord.calculateTotalCoord();
//   grid_coord sizeMu = MuSizeCoord.calculateTotalCoord();
//
//   grid_coord sizeEzRaw = (grid_coord) sizeEz * sizeof (FieldValue);
//   grid_coord sizeHxRaw = (grid_coord) sizeHx * sizeof (FieldValue);
//   grid_coord sizeHyRaw = (grid_coord) sizeHy * sizeof (FieldValue);
//   grid_coord sizeEpsRaw = (grid_coord) sizeEps * sizeof (FieldValue);
//   grid_coord sizeMuRaw = (grid_coord) sizeMu * sizeof (FieldValue);
//
//   FieldValue *tmp_Ez = new FieldValue [sizeEz];
//   FieldValue *tmp_Hx = new FieldValue [sizeHx];
//   FieldValue *tmp_Hy = new FieldValue [sizeHy];
//
//   FieldValue *tmp_Ez_prev = new FieldValue [sizeEz];
//   FieldValue *tmp_Hx_prev = new FieldValue [sizeHx];
//   FieldValue *tmp_Hy_prev = new FieldValue [sizeHy];
//
//   FieldValue *tmp_eps = new FieldValue [sizeEps];
//   FieldValue *tmp_mu = new FieldValue [sizeMu];
//
//   FieldValue *Ez_cuda;
//   FieldValue *Hx_cuda;
//   FieldValue *Hy_cuda;
//
//   FieldValue *Ez_cuda_prev;
//   FieldValue *Hx_cuda_prev;
//   FieldValue *Hy_cuda_prev;
//
//   FieldValue *eps_cuda;
//   FieldValue *mu_cuda;
//
//   cudaCheckErrorCmd (cudaMalloc ((void **) &Ez_cuda, sizeEzRaw));
//   cudaCheckErrorCmd (cudaMalloc ((void **) &Hx_cuda, sizeHxRaw));
//   cudaCheckErrorCmd (cudaMalloc ((void **) &Hy_cuda, sizeHyRaw));
//
//   cudaCheckErrorCmd (cudaMalloc ((void **) &Ez_cuda_prev, sizeEzRaw));
//   cudaCheckErrorCmd (cudaMalloc ((void **) &Hx_cuda_prev, sizeHxRaw));
//   cudaCheckErrorCmd (cudaMalloc ((void **) &Hy_cuda_prev, sizeHyRaw));
//
//   cudaCheckErrorCmd (cudaMalloc ((void **) &eps_cuda, sizeEpsRaw));
//   cudaCheckErrorCmd (cudaMalloc ((void **) &mu_cuda, sizeMuRaw));
//
//   CudaExitStatus exitStatus;
//   CudaExitStatus *exitStatusCuda;
//   cudaCheckErrorCmd (cudaMalloc ((void **) &exitStatusCuda, sizeof (CudaExitStatus)));
//
//   dim3 blocksEz (EzSizeCoord.get1 () / cudaThreadsX, EzSizeCoord.get2 () / cudaThreadsY);
//   dim3 threadsEz (cudaThreadsX, cudaThreadsY);
//
//   dim3 blocksHx (HxSizeCoord.get1 () / cudaThreadsX, HxSizeCoord.get2 () / cudaThreadsY);
//   dim3 threadsHx (cudaThreadsX, cudaThreadsY);
//
//   dim3 blocksHy (HySizeCoord.get1 () / cudaThreadsX, HySizeCoord.get2 () / cudaThreadsY);
//   dim3 threadsHy (cudaThreadsX, cudaThreadsY);
//
//   while (t < totalStep)
//   {
//     for (grid_coord i = 0; i < sizeEz; ++i)
//     {
//       FieldPointValue* valEz = Ez.getFieldPointValue (i);
//       tmp_Ez[i] = valEz->getCurValue ();
//       tmp_Ez_prev[i] = valEz->getPrevValue ();
//     }
//
//     for (grid_coord i = 0; i < sizeHx; ++i)
//     {
//       FieldPointValue* valHx = Hx.getFieldPointValue (i);
//       tmp_Hx[i] = valHx->getCurValue ();
//       tmp_Hx_prev[i] = valHx->getPrevValue ();
//     }
//
//     for (grid_coord i = 0; i < sizeHy; ++i)
//     {
//       FieldPointValue* valHy = Hy.getFieldPointValue (i);
//       tmp_Hy[i] = valHy->getCurValue ();
//       tmp_Hy_prev[i] = valHy->getPrevValue ();
//     }
//
//     for (grid_coord i = 0; i < sizeEps; ++i)
//     {
//       FieldPointValue *valEps = Eps.getFieldPointValue (i);
//       tmp_eps[i] = valEps->getCurValue ();
//     }
//
//     for (grid_coord i = 0; i < sizeMu; ++i)
//     {
//       FieldPointValue *valMu = Mu.getFieldPointValue (i);
//       tmp_mu[i] = valMu->getCurValue ();
//     }
//
//     cudaCheckErrorCmd (cudaMemcpy (Ez_cuda, tmp_Ez, sizeEzRaw, cudaMemcpyHostToDevice));
//     cudaCheckErrorCmd (cudaMemcpy (Hx_cuda, tmp_Hx, sizeHxRaw, cudaMemcpyHostToDevice));
//     cudaCheckErrorCmd (cudaMemcpy (Hy_cuda, tmp_Hy, sizeHyRaw, cudaMemcpyHostToDevice));
//
//     cudaCheckErrorCmd (cudaMemcpy (Ez_cuda_prev, tmp_Ez_prev, sizeEzRaw, cudaMemcpyHostToDevice));
//     cudaCheckErrorCmd (cudaMemcpy (Hx_cuda_prev, tmp_Hx_prev, sizeHxRaw, cudaMemcpyHostToDevice));
//     cudaCheckErrorCmd (cudaMemcpy (Hy_cuda_prev, tmp_Hy_prev, sizeHyRaw, cudaMemcpyHostToDevice));
//
//     cudaCheckErrorCmd (cudaMemcpy (eps_cuda, tmp_eps, sizeEpsRaw, cudaMemcpyHostToDevice));
//     cudaCheckErrorCmd (cudaMemcpy (mu_cuda, tmp_mu, sizeMuRaw, cudaMemcpyHostToDevice));
//
// #if not defined (PARALLEL_GRID)
//     GridCoordinate3D EzStart = Ez.getComputationStart (yeeLayout->getEzStartDiff ());
//     GridCoordinate3D EzEnd = Ez.getComputationEnd (yeeLayout->getEzEndDiff ());
//
//     GridCoordinate3D HxStart = Hx.getComputationStart (yeeLayout->getHxStartDiff ());
//     GridCoordinate3D HxEnd = Hx.getComputationEnd (yeeLayout->getHxEndDiff ());
//
//     GridCoordinate3D HyStart = Hy.getComputationStart (yeeLayout->getHyStartDiff ());
//     GridCoordinate3D HyEnd = Hy.getComputationEnd (yeeLayout->getHyEndDiff ());
// #endif
//
//     for (time_step stepEnd = t + shareStep; t < stepEnd; ++t)
//     {
// #if defined (PARALLEL_GRID)
//       GridCoordinate3D EzStart = Ez.getComputationStart (yeeLayout->getEzStartDiff ());
//       GridCoordinate3D EzEnd = Ez.getComputationEnd (yeeLayout->getEzEndDiff ());
//
//       GridCoordinate3D HxStart = Hx.getComputationStart (yeeLayout->getHxStartDiff ());
//       GridCoordinate3D HxEnd = Hx.getComputationEnd (yeeLayout->getHxEndDiff ());
//
//       GridCoordinate3D HyStart = Hy.getComputationStart (yeeLayout->getHyStartDiff ());
//       GridCoordinate3D HyEnd = Hy.getComputationEnd (yeeLayout->getHyEndDiff ());
// #endif
//
//       cudaCheckExitStatus (cudaCalculateTMzEzStep <<< blocksEz, threadsEz >>> (exitStatusCuda,
//                                                                                Ez_cuda,
//                                                                                Ez_cuda_prev,
//                                                                                Hx_cuda_prev,
//                                                                                Hy_cuda_prev,
//                                                                                eps_cuda,
//                                                                                gridTimeStep,
//                                                                                gridStep,
//                                                                                EzStart,
//                                                                                EzEnd.get1 (),
//                                                                                EzEnd.get2 (),
//                                                                                EzSizeCoord.get1 (),
//                                                                                EzSizeCoord.get2 (),
//                                                                                t));
//
//       cudaCheckExitStatus (cudaCalculateTMzEzSource <<< blocksEz, threadsEz >>> (exitStatusCuda,
//                                                                                  Ez_cuda_prev,
//                                                                                  EzStart.get1 (),
//                                                                                  EzStart.get2 (),
//                                                                                  EzEnd.get1 (),
//                                                                                  EzEnd.get2 (),
//                                                                                  EzSizeCoord.get1 (),
//                                                                                  EzSizeCoord.get2 (),
//                                                                                  t,
//                                                                                  processId));
//
// #if defined (PARALLEL_GRID)
//       Ez.nextShareStep ();
// #endif
//
//       cudaCheckExitStatus (cudaCalculateTMzHxStep <<< blocksHx, threadsHx >>> (exitStatusCuda,
//                                                                                Hx_cuda,
//                                                                                Ez_cuda_prev,
//                                                                                Hx_cuda_prev,
//                                                                                mu_cuda,
//                                                                                gridTimeStep,
//                                                                                gridStep,
//                                                                                HxStart.get1 (),
//                                                                                HxStart.get2 (),
//                                                                                HxEnd.get1 (),
//                                                                                HxEnd.get2 (),
//                                                                                HxSizeCoord.get1 (),
//                                                                                HxSizeCoord.get2 (),
//                                                                                t));
//
//       cudaCheckExitStatus (cudaCalculateTMzHyStep <<< blocksHy, threadsHy >>> (exitStatusCuda,
//                                                                                Hy_cuda,
//                                                                                Ez_cuda_prev,
//                                                                                Hy_cuda_prev,
//                                                                                mu_cuda,
//                                                                                gridTimeStep,
//                                                                                gridStep,
//                                                                                HyStart.get1 (),
//                                                                                HyStart.get2 (),
//                                                                                HyEnd.get1 (),
//                                                                                HyEnd.get2 (),
//                                                                                HySizeCoord.get1 (),
//                                                                                HySizeCoord.get2 (),
//                                                                                t));
//
//       cudaCheckExitStatus (cudaCalculateTMzHxSource <<< blocksHx, threadsHx >>> (exitStatusCuda,
//                                                                                  Hx_cuda_prev,
//                                                                                  HxStart.get1 (),
//                                                                                  HxStart.get2 (),
//                                                                                  HxEnd.get1 (),
//                                                                                  HxEnd.get2 (),
//                                                                                  HxSizeCoord.get1 (),
//                                                                                  HxSizeCoord.get2 (),
//                                                                                  t));
//
//       cudaCheckExitStatus (cudaCalculateTMzHySource <<< blocksHy, threadsHy >>> (exitStatusCuda,
//                                                                                  Hy_cuda_prev,
//                                                                                  HyStart.get1 (),
//                                                                                  HyStart.get2 (),
//                                                                                  HyEnd.get1 (),
//                                                                                  HyEnd.get2 (),
//                                                                                  HySizeCoord.get1 (),
//                                                                                  HySizeCoord.get2 (),
//                                                                                  t));
//
// #if defined (PARALLEL_GRID)
//       Hx.nextShareStep ();
//       Hy.nextShareStep ();
// #endif
//     }
//
//     cudaCheckErrorCmd (cudaMemcpy (tmp_Ez, Ez_cuda, sizeEzRaw, cudaMemcpyDeviceToHost));
//     cudaCheckErrorCmd (cudaMemcpy (tmp_Hx, Hx_cuda, sizeHxRaw, cudaMemcpyDeviceToHost));
//     cudaCheckErrorCmd (cudaMemcpy (tmp_Hy, Hy_cuda, sizeHyRaw, cudaMemcpyDeviceToHost));
//
//     cudaCheckErrorCmd (cudaMemcpy (tmp_Ez_prev, Ez_cuda_prev, sizeEzRaw, cudaMemcpyDeviceToHost));
//     cudaCheckErrorCmd (cudaMemcpy (tmp_Hx_prev, Hx_cuda_prev, sizeHxRaw, cudaMemcpyDeviceToHost));
//     cudaCheckErrorCmd (cudaMemcpy (tmp_Hy_prev, Hy_cuda_prev, sizeHyRaw, cudaMemcpyDeviceToHost));
//
//     cudaCheckErrorCmd (cudaMemcpy (tmp_eps, eps_cuda, sizeEpsRaw, cudaMemcpyDeviceToHost));
//     cudaCheckErrorCmd (cudaMemcpy (tmp_mu, mu_cuda, sizeMuRaw, cudaMemcpyDeviceToHost));
//
//     for (grid_coord i = 0; i < sizeEz; ++i)
//     {
//       FieldPointValue* valEz = Ez.getFieldPointValue (i);
//       valEz->setCurValue (tmp_Ez[i]);
//       valEz->setPrevValue (tmp_Ez_prev[i]);
//     }
//
//     for (grid_coord i = 0; i < sizeHx; ++i)
//     {
//       FieldPointValue* valHx = Hx.getFieldPointValue (i);
//       valHx->setCurValue (tmp_Hx[i]);
//       valHx->setPrevValue (tmp_Hx_prev[i]);
//     }
//
//     for (grid_coord i = 0; i < sizeHy; ++i)
//     {
//       FieldPointValue* valHy = Hy.getFieldPointValue (i);
//       valHy->setCurValue (tmp_Hy[i]);
//       valHy->setPrevValue (tmp_Hy_prev[i]);
//     }
//
// #if defined (PARALLEL_GRID)
//     Ez.zeroShareStep ();
//     Ez.share ();
//
//     Hx.zeroShareStep ();
//     Hx.share ();
//
//     Hy.zeroShareStep ();
//     Hy.share ();
// #endif /* PARALLEL_GRID */
//   }
//
//   cudaCheckErrorCmd (cudaFree (Ez_cuda));
//   cudaCheckErrorCmd (cudaFree (Hx_cuda));
//   cudaCheckErrorCmd (cudaFree (Hy_cuda));
//
//   cudaCheckErrorCmd (cudaFree (Ez_cuda_prev));
//   cudaCheckErrorCmd (cudaFree (Hx_cuda_prev));
//   cudaCheckErrorCmd (cudaFree (Hy_cuda_prev));
//
//   cudaCheckErrorCmd (cudaFree (eps_cuda));
//   cudaCheckErrorCmd (cudaFree (mu_cuda));
//
//   delete[] tmp_Ez;
//   delete[] tmp_Hx;
//   delete[] tmp_Hy;
//
//   delete[] tmp_Ez_prev;
//   delete[] tmp_Hx_prev;
//   delete[] tmp_Hy_prev;
//
//   delete[] tmp_eps;
//   delete[] tmp_mu;
//
//   *retval = CUDA_OK;
//   return;
// }

// #ifdef PARALLEL_GRID
// void cudaExecute3DSteps (CudaExitStatus *retval,
//                          YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED> *yeeLayout,
//                          FPValue gridTimeStep, FPValue gridStep,
//                          ParallelGrid &Ex,
//                          ParallelGrid &Ey,
//                          ParallelGrid &Ez,
//                          ParallelGrid &Hx,
//                          ParallelGrid &Hy,
//                          ParallelGrid &Hz,
//                          ParallelGrid &Eps,
//                          ParallelGrid &Mu,
//                          time_step totalStep,
//                          int processId)
// #else
void cudaExecute3DSteps (bool useParallel,
                         CudaExitStatus *retval,
                         YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED> *yeeLayout,
                         FPValue gridTimeStep, FPValue gridStep,
                         Grid<GridCoordinate3D> *Ex,
                         Grid<GridCoordinate3D> *Ey,
                         Grid<GridCoordinate3D> *Ez,
                         Grid<GridCoordinate3D> *Hx,
                         Grid<GridCoordinate3D> *Hy,
                         Grid<GridCoordinate3D> *Hz,
                         Grid<GridCoordinate3D> *Eps,
                         Grid<GridCoordinate3D> *Mu,
                         time_step totalStep,
                         int processId)
// #endif
{
  time_step t = 0;

#ifdef PARALLEL_GRID
  GridCoordinate3D bufSize = ((ParallelGrid *)Ex)->getBufferSize ();

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  time_step shareStep = bufSize.get1 ();
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  time_step shareStep = bufSize.get2 ();
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  time_step shareStep = bufSize.get3 ();
#endif
#else
  time_step shareStep = totalStep;
#endif

  GridCoordinate3D ExSizeCoord = Ex->getSize ();
  GridCoordinate3D EySizeCoord = Ey->getSize ();
  GridCoordinate3D EzSizeCoord = Ez->getSize ();
  GridCoordinate3D HxSizeCoord = Hx->getSize ();
  GridCoordinate3D HySizeCoord = Hy->getSize ();
  GridCoordinate3D HzSizeCoord = Hz->getSize ();
  GridCoordinate3D EpsSizeCoord = Eps->getSize ();
  GridCoordinate3D MuSizeCoord = Mu->getSize ();

  grid_coord sizeEx = ExSizeCoord.calculateTotalCoord();
  grid_coord sizeEy = EySizeCoord.calculateTotalCoord();
  grid_coord sizeEz = EzSizeCoord.calculateTotalCoord();
  grid_coord sizeHx = HxSizeCoord.calculateTotalCoord();
  grid_coord sizeHy = HySizeCoord.calculateTotalCoord();
  grid_coord sizeHz = HzSizeCoord.calculateTotalCoord();
  grid_coord sizeEps = EpsSizeCoord.calculateTotalCoord();
  grid_coord sizeMu = MuSizeCoord.calculateTotalCoord();

  grid_coord sizeExRaw = (grid_coord) sizeEx * sizeof (FieldValue);
  grid_coord sizeEyRaw = (grid_coord) sizeEy * sizeof (FieldValue);
  grid_coord sizeEzRaw = (grid_coord) sizeEz * sizeof (FieldValue);
  grid_coord sizeHxRaw = (grid_coord) sizeHx * sizeof (FieldValue);
  grid_coord sizeHyRaw = (grid_coord) sizeHy * sizeof (FieldValue);
  grid_coord sizeHzRaw = (grid_coord) sizeHz * sizeof (FieldValue);
  grid_coord sizeEpsRaw = (grid_coord) sizeEps * sizeof (FieldValue);
  grid_coord sizeMuRaw = (grid_coord) sizeMu * sizeof (FieldValue);

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

  dim3 blocksEx (ExSizeCoord.get1 () / cudaThreadsX, ExSizeCoord.get2 () / cudaThreadsY, ExSizeCoord.get3 () / cudaThreadsZ);
  dim3 threadsEx (cudaThreadsX, cudaThreadsY, cudaThreadsZ);

  dim3 blocksEy (EySizeCoord.get1 () / cudaThreadsX, EySizeCoord.get2 () / cudaThreadsY, EySizeCoord.get3 () / cudaThreadsZ);
  dim3 threadsEy (cudaThreadsX, cudaThreadsY, cudaThreadsZ);

  dim3 blocksEz (EzSizeCoord.get1 () / cudaThreadsX, EzSizeCoord.get2 () / cudaThreadsY, EzSizeCoord.get3 () / cudaThreadsZ);
  dim3 threadsEz (cudaThreadsX, cudaThreadsY, cudaThreadsZ);

  dim3 blocksHx (HxSizeCoord.get1 () / cudaThreadsX, HxSizeCoord.get2 () / cudaThreadsY, HxSizeCoord.get3 () / cudaThreadsZ);
  dim3 threadsHx (cudaThreadsX, cudaThreadsY, cudaThreadsZ);

  dim3 blocksHy (HySizeCoord.get1 () / cudaThreadsX, HySizeCoord.get2 () / cudaThreadsY, HySizeCoord.get3 () / cudaThreadsZ);
  dim3 threadsHy (cudaThreadsX, cudaThreadsY, cudaThreadsZ);

  dim3 blocksHz (HzSizeCoord.get1 () / cudaThreadsX, HzSizeCoord.get2 () / cudaThreadsY, HzSizeCoord.get3 () / cudaThreadsZ);
  dim3 threadsHz (cudaThreadsX, cudaThreadsY, cudaThreadsZ);

  while (t < totalStep)
  {
    for (grid_coord i = 0; i < sizeEx; ++i)
    {
      FieldPointValue* valEx = Ex->getFieldPointValue (i);
      tmp_Ex[i] = valEx->getCurValue ();
      tmp_Ex_prev[i] = valEx->getPrevValue ();
    }

    for (grid_coord i = 0; i < sizeEy; ++i)
    {
      FieldPointValue* valEy = Ey->getFieldPointValue (i);
      tmp_Ey[i] = valEy->getCurValue ();
      tmp_Ey_prev[i] = valEy->getPrevValue ();
    }

    for (grid_coord i = 0; i < sizeEz; ++i)
    {
      FieldPointValue* valEz = Ez->getFieldPointValue (i);
      tmp_Ez[i] = valEz->getCurValue ();
      tmp_Ez_prev[i] = valEz->getPrevValue ();
    }

    for (grid_coord i = 0; i < sizeHx; ++i)
    {
      FieldPointValue* valHx = Hx->getFieldPointValue (i);
      tmp_Hx[i] = valHx->getCurValue ();
      tmp_Hx_prev[i] = valHx->getPrevValue ();
    }

    for (grid_coord i = 0; i < sizeHy; ++i)
    {
      FieldPointValue* valHy = Hy->getFieldPointValue (i);
      tmp_Hy[i] = valHy->getCurValue ();
      tmp_Hy_prev[i] = valHy->getPrevValue ();
    }

    for (grid_coord i = 0; i < sizeHz; ++i)
    {
      FieldPointValue* valHz = Hz->getFieldPointValue (i);
      tmp_Hz[i] = valHz->getCurValue ();
      tmp_Hz_prev[i] = valHz->getPrevValue ();
    }

    for (grid_coord i = 0; i < sizeEps; ++i)
    {
      FieldPointValue *valEps = Eps->getFieldPointValue (i);
      tmp_eps[i] = valEps->getCurValue ();
    }

    for (grid_coord i = 0; i < sizeMu; ++i)
    {
      FieldPointValue *valMu = Mu->getFieldPointValue (i);
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
    GridCoordinate3D ExStart = Ex->getComputationStart (yeeLayout->getExStartDiff ());
    GridCoordinate3D ExEnd = Ex->getComputationEnd (yeeLayout->getExEndDiff ());

    GridCoordinate3D EyStart = Ey->getComputationStart (yeeLayout->getEyStartDiff ());
    GridCoordinate3D EyEnd = Ey->getComputationEnd (yeeLayout->getEyEndDiff ());

    GridCoordinate3D EzStart = Ez->getComputationStart (yeeLayout->getEzStartDiff ());
    GridCoordinate3D EzEnd = Ez->getComputationEnd (yeeLayout->getEzEndDiff ());

    GridCoordinate3D HxStart = Hx->getComputationStart (yeeLayout->getHxStartDiff ());
    GridCoordinate3D HxEnd = Hx->getComputationEnd (yeeLayout->getHxEndDiff ());

    GridCoordinate3D HyStart = Hy->getComputationStart (yeeLayout->getHyStartDiff ());
    GridCoordinate3D HyEnd = Hy->getComputationEnd (yeeLayout->getHyEndDiff ());

    GridCoordinate3D HzStart = Hz->getComputationStart (yeeLayout->getHzStartDiff ());
    GridCoordinate3D HzEnd = Hz->getComputationEnd (yeeLayout->getHzEndDiff ());
#endif

    for (time_step stepEnd = t + shareStep; t < stepEnd; ++t)
    {
#if defined (PARALLEL_GRID)
      GridCoordinate3D ExStart = ((ParallelGrid *)Ex)->getComputationStart (yeeLayout->getExStartDiff ());
      GridCoordinate3D ExEnd = ((ParallelGrid *)Ex)->getComputationEnd (yeeLayout->getExEndDiff ());

      GridCoordinate3D EyStart = ((ParallelGrid *)Ey)->getComputationStart (yeeLayout->getEyStartDiff ());
      GridCoordinate3D EyEnd = ((ParallelGrid *)Ey)->getComputationEnd (yeeLayout->getEyEndDiff ());

      GridCoordinate3D EzStart = ((ParallelGrid *)Ez)->getComputationStart (yeeLayout->getEzStartDiff ());
      GridCoordinate3D EzEnd = ((ParallelGrid *)Ez)->getComputationEnd (yeeLayout->getEzEndDiff ());

      GridCoordinate3D HxStart = ((ParallelGrid *)Hx)->getComputationStart (yeeLayout->getHxStartDiff ());
      GridCoordinate3D HxEnd = ((ParallelGrid *)Hx)->getComputationEnd (yeeLayout->getHxEndDiff ());

      GridCoordinate3D HyStart = ((ParallelGrid *)Hy)->getComputationStart (yeeLayout->getHyStartDiff ());
      GridCoordinate3D HyEnd = ((ParallelGrid *)Hy)->getComputationEnd (yeeLayout->getHyEndDiff ());

      GridCoordinate3D HzStart = ((ParallelGrid *)Hz)->getComputationStart (yeeLayout->getHzStartDiff ());
      GridCoordinate3D HzEnd = ((ParallelGrid *)Hz)->getComputationEnd (yeeLayout->getHzEndDiff ());
#endif

      if (useParallel && solverSettings.getDoUseDynamicGrid ())
      {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
        ParallelGrid::getParallelCore ()->StartCalcClock ();
#else
        ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
      }

      cudaCheckExitStatus (cudaCalculate3DExStep <<< blocksEx, threadsEx >>> (exitStatusCuda,
                                                                              Ex_cuda,
                                                                              Ex_cuda_prev,
                                                                              Hy_cuda_prev,
                                                                              Hz_cuda_prev,
                                                                              eps_cuda,
                                                                              gridTimeStep,
                                                                              gridStep,
                                                                              ExStart.get1 (),
                                                                              ExStart.get2 (),
                                                                              ExStart.get3 (),
                                                                              ExEnd.get1 (),
                                                                              ExEnd.get2 (),
                                                                              ExEnd.get3 (),
                                                                              ExSizeCoord.get1 (),
                                                                              ExSizeCoord.get2 (),
                                                                              ExSizeCoord.get3 (),
                                                                              t));

      cudaCheckExitStatus (cudaCalculate3DEyStep <<< blocksEy, threadsEy >>> (exitStatusCuda,
                                                                              Ey_cuda,
                                                                              Ey_cuda_prev,
                                                                              Hx_cuda_prev,
                                                                              Hz_cuda_prev,
                                                                              eps_cuda,
                                                                              gridTimeStep,
                                                                              gridStep,
                                                                              EyStart.get1 (),
                                                                              EyStart.get2 (),
                                                                              EyStart.get3 (),
                                                                              EyEnd.get1 (),
                                                                              EyEnd.get2 (),
                                                                              EyEnd.get3 (),
                                                                              EySizeCoord.get1 (),
                                                                              EySizeCoord.get2 (),
                                                                              EySizeCoord.get3 (),
                                                                              t));

      cudaCheckExitStatus (cudaCalculate3DEzStep <<< blocksEz, threadsEz >>> (exitStatusCuda,
                                                                              Ez_cuda,
                                                                              Ez_cuda_prev,
                                                                              Hx_cuda_prev,
                                                                              Hy_cuda_prev,
                                                                              eps_cuda,
                                                                              gridTimeStep,
                                                                              gridStep,
                                                                              EzStart.get1 (),
                                                                              EzStart.get2 (),
                                                                              EzStart.get3 (),
                                                                              EzEnd.get1 (),
                                                                              EzEnd.get2 (),
                                                                              EzEnd.get3 (),
                                                                              EzSizeCoord.get1 (),
                                                                              EzSizeCoord.get2 (),
                                                                              EzSizeCoord.get3 (),
                                                                              t));

      if (useParallel && solverSettings.getDoUseDynamicGrid ())
      {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
        ParallelGrid::getParallelCore ()->StopCalcClock ();
#else
        ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
      }

      cudaCheckExitStatus (cudaCalculate3DExSource <<< blocksEx, threadsEx >>> (exitStatusCuda,
                                                                                Ex_cuda_prev,
                                                                                ExStart.get1 (),
                                                                                ExStart.get2 (),
                                                                                ExStart.get3 (),
                                                                                ExEnd.get1 (),
                                                                                ExEnd.get2 (),
                                                                                ExEnd.get3 (),
                                                                                ExSizeCoord.get1 (),
                                                                                ExSizeCoord.get2 (),
                                                                                ExSizeCoord.get3 (),
                                                                                t,
                                                                                processId));

      cudaCheckExitStatus (cudaCalculate3DEySource <<< blocksEy, threadsEy >>> (exitStatusCuda,
                                                                                Ey_cuda_prev,
                                                                                EyStart.get1 (),
                                                                                EyStart.get2 (),
                                                                                EyStart.get3 (),
                                                                                EyEnd.get1 (),
                                                                                EyEnd.get2 (),
                                                                                EyEnd.get3 (),
                                                                                EySizeCoord.get1 (),
                                                                                EySizeCoord.get2 (),
                                                                                EySizeCoord.get3 (),
                                                                                t,
                                                                                processId));

      cudaCheckExitStatus (cudaCalculate3DEzSource <<< blocksEz, threadsEz >>> (exitStatusCuda,
                                                                                Ez_cuda_prev,
                                                                                EzStart.get1 (),
                                                                                EzStart.get2 (),
                                                                                EzStart.get3 (),
                                                                                EzEnd.get1 (),
                                                                                EzEnd.get2 (),
                                                                                EzEnd.get3 (),
                                                                                EzSizeCoord.get1 (),
                                                                                EzSizeCoord.get2 (),
                                                                                EzSizeCoord.get3 (),
                                                                                t,
                                                                                processId));

#if defined (PARALLEL_GRID)
      ((ParallelGrid *)Ex)->nextShareStep ();
      ((ParallelGrid *)Ey)->nextShareStep ();
      ((ParallelGrid *)Ez)->nextShareStep ();
#endif

      if (useParallel && solverSettings.getDoUseDynamicGrid ())
      {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
        ParallelGrid::getParallelCore ()->StartCalcClock ();
#else
        ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
      }

      cudaCheckExitStatus (cudaCalculate3DHxStep <<< blocksHx, threadsHx >>> (exitStatusCuda,
                                                                              Hx_cuda,
                                                                              Hx_cuda_prev,
                                                                              Ey_cuda_prev,
                                                                              Ez_cuda_prev,
                                                                              mu_cuda,
                                                                              gridTimeStep,
                                                                              gridStep,
                                                                              HxStart.get1 (),
                                                                              HxStart.get2 (),
                                                                              HxStart.get3 (),
                                                                              HxEnd.get1 (),
                                                                              HxEnd.get2 (),
                                                                              HxEnd.get3 (),
                                                                              HxSizeCoord.get1 (),
                                                                              HxSizeCoord.get2 (),
                                                                              HxSizeCoord.get3 (),
                                                                              t));

      cudaCheckExitStatus (cudaCalculate3DHyStep <<< blocksHy, threadsHy >>> (exitStatusCuda,
                                                                              Hy_cuda,
                                                                              Hy_cuda_prev,
                                                                              Ex_cuda_prev,
                                                                              Ez_cuda_prev,
                                                                              mu_cuda,
                                                                              gridTimeStep,
                                                                              gridStep,
                                                                              HyStart.get1 (),
                                                                              HyStart.get2 (),
                                                                              HyStart.get3 (),
                                                                              HyEnd.get1 (),
                                                                              HyEnd.get2 (),
                                                                              HyEnd.get3 (),
                                                                              HySizeCoord.get1 (),
                                                                              HySizeCoord.get2 (),
                                                                              HySizeCoord.get3 (),
                                                                              t));

      cudaCheckExitStatus (cudaCalculate3DHzStep <<< blocksHz, threadsHz >>> (exitStatusCuda,
                                                                              Hz_cuda,
                                                                              Hz_cuda_prev,
                                                                              Ex_cuda_prev,
                                                                              Ey_cuda_prev,
                                                                              mu_cuda,
                                                                              gridTimeStep,
                                                                              gridStep,
                                                                              HzStart.get1 (),
                                                                              HzStart.get2 (),
                                                                              HzStart.get3 (),
                                                                              HzEnd.get1 (),
                                                                              HzEnd.get2 (),
                                                                              HzEnd.get3 (),
                                                                              HzSizeCoord.get1 (),
                                                                              HzSizeCoord.get2 (),
                                                                              HzSizeCoord.get3 (),
                                                                              t));

      if (useParallel && solverSettings.getDoUseDynamicGrid ())
      {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
        ParallelGrid::getParallelCore ()->StopCalcClock ();
#else
        ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
      }

      cudaCheckExitStatus (cudaCalculate3DHxSource <<< blocksHx, threadsHx >>> (exitStatusCuda,
                                                                                Hx_cuda_prev,
                                                                                HxStart.get1 (),
                                                                                HxStart.get2 (),
                                                                                HxStart.get3 (),
                                                                                HxEnd.get1 (),
                                                                                HxEnd.get2 (),
                                                                                HxEnd.get3 (),
                                                                                HxSizeCoord.get1 (),
                                                                                HxSizeCoord.get2 (),
                                                                                HxSizeCoord.get3 (),
                                                                                t,
                                                                                processId));

      cudaCheckExitStatus (cudaCalculate3DHySource <<< blocksHy, threadsHy >>> (exitStatusCuda,
                                                                                Hy_cuda_prev,
                                                                                HyStart.get1 (),
                                                                                HyStart.get2 (),
                                                                                HyStart.get3 (),
                                                                                HyEnd.get1 (),
                                                                                HyEnd.get2 (),
                                                                                HyEnd.get3 (),
                                                                                HySizeCoord.get1 (),
                                                                                HySizeCoord.get2 (),
                                                                                HySizeCoord.get3 (),
                                                                                t,
                                                                                processId));

      cudaCheckExitStatus (cudaCalculate3DHzSource <<< blocksHz, threadsHz >>> (exitStatusCuda,
                                                                                Hz_cuda_prev,
                                                                                HzStart.get1 (),
                                                                                HzStart.get2 (),
                                                                                HzStart.get3 (),
                                                                                HzEnd.get1 (),
                                                                                HzEnd.get2 (),
                                                                                HzEnd.get3 (),
                                                                                HzSizeCoord.get1 (),
                                                                                HzSizeCoord.get2 (),
                                                                                HzSizeCoord.get3 (),
                                                                                t,
                                                                                processId));

#if defined (PARALLEL_GRID)
      ((ParallelGrid *)Hx)->nextShareStep ();
      ((ParallelGrid *)Hy)->nextShareStep ();
      ((ParallelGrid *)Hz)->nextShareStep ();
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

    for (grid_coord i = 0; i < sizeEx; ++i)
    {
      FieldPointValue* valEx = Ex->getFieldPointValue (i);
      valEx->setCurValue (tmp_Ex[i]);
      valEx->setPrevValue (tmp_Ex_prev[i]);
    }

    for (grid_coord i = 0; i < sizeEy; ++i)
    {
      FieldPointValue* valEy = Ey->getFieldPointValue (i);
      valEy->setCurValue (tmp_Ey[i]);
      valEy->setPrevValue (tmp_Ey_prev[i]);
    }

    for (grid_coord i = 0; i < sizeEz; ++i)
    {
      FieldPointValue* valEz = Ez->getFieldPointValue (i);
      valEz->setCurValue (tmp_Ez[i]);
      valEz->setPrevValue (tmp_Ez_prev[i]);
    }

    for (grid_coord i = 0; i < sizeHx; ++i)
    {
      FieldPointValue* valHx = Hx->getFieldPointValue (i);
      valHx->setCurValue (tmp_Hx[i]);
      valHx->setPrevValue (tmp_Hx_prev[i]);
    }

    for (grid_coord i = 0; i < sizeHy; ++i)
    {
      FieldPointValue* valHy = Hy->getFieldPointValue (i);
      valHy->setCurValue (tmp_Hy[i]);
      valHy->setPrevValue (tmp_Hy_prev[i]);
    }

    for (grid_coord i = 0; i < sizeHz; ++i)
    {
      FieldPointValue* valHz = Hz->getFieldPointValue (i);
      valHz->setCurValue (tmp_Hz[i]);
      valHz->setPrevValue (tmp_Hz_prev[i]);
    }

#if defined (PARALLEL_GRID)
    ((ParallelGrid *)Ex)->zeroShareStep ();
    ((ParallelGrid *)Ex)->share ();

    ((ParallelGrid *)Ey)->zeroShareStep ();
    ((ParallelGrid *)Ey)->share ();

    ((ParallelGrid *)Ez)->zeroShareStep ();
    ((ParallelGrid *)Ez)->share ();

    ((ParallelGrid *)Hx)->zeroShareStep ();
    ((ParallelGrid *)Hx)->share ();

    ((ParallelGrid *)Hy)->zeroShareStep ();
    ((ParallelGrid *)Hy)->share ();

    ((ParallelGrid *)Hz)->zeroShareStep ();
    ((ParallelGrid *)Hz)->share ();

    //additionalUpdateOfGrids
    {
      if (useParallel && solverSettings.getDoUseDynamicGrid ())
      {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
        //if (false && t % solverSettings.getRebalanceStep () == 0)
        time_step diffT = solverSettings.getRebalanceStep ();
        if (t % diffT == 0 && t > 0)
        {
          if (ParallelGrid::getParallelCore ()->getProcessId () == 0)
          {
            DPRINTF (LOG_LEVEL_STAGES, "Try rebalance on step %u, steps elapsed after previous %u\n", t, diffT);
          }

          //ASSERT (isParallelLayout);

          ParallelYeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim3), E_CENTERED> *parallelYeeLayout =
            (ParallelYeeGridLayout<static_cast<SchemeType_t> (SchemeType::Dim3), E_CENTERED> *) yeeLayout;

          if (parallelYeeLayout->Rebalance (diffT))
          {
            DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Rebalancing for process %d!\n", ParallelGrid::getParallelCore ()->getProcessId ());

            ((ParallelGrid *) Eps)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
            ((ParallelGrid *) Mu)->Resize (parallelYeeLayout->getMuSizeForCurNode ());

            //if (doNeedEx)
            {
              ((ParallelGrid *) Ex)->Resize (parallelYeeLayout->getExSizeForCurNode ());
            }
            //if (doNeedEy)
            {
              ((ParallelGrid *) Ey)->Resize (parallelYeeLayout->getEySizeForCurNode ());
            }
            //if (doNeedEz)
            {
              ((ParallelGrid *) Ez)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
            }

            //if (doNeedHx)
            {
              ((ParallelGrid *) Hx)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
            }
            //if (doNeedHy)
            {
              ((ParallelGrid *) Hy)->Resize (parallelYeeLayout->getHySizeForCurNode ());
            }
            //if (doNeedHz)
            {
              ((ParallelGrid *) Hz)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
            }

            // if (solverSettings.getDoUsePML ())
            // {
            //   if (doNeedEx)
            //   {
            //     ((ParallelGrid *) Dx)->Resize (parallelYeeLayout->getExSizeForCurNode ());
            //   }
            //   if (doNeedEy)
            //   {
            //     ((ParallelGrid *) Dy)->Resize (parallelYeeLayout->getEySizeForCurNode ());
            //   }
            //   if (doNeedEz)
            //   {
            //     ((ParallelGrid *) Dz)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
            //   }
            //
            //   if (doNeedHx)
            //   {
            //     ((ParallelGrid *) Bx)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
            //   }
            //   if (doNeedHy)
            //   {
            //     ((ParallelGrid *) By)->Resize (parallelYeeLayout->getHySizeForCurNode ());
            //   }
            //   if (doNeedHz)
            //   {
            //     ((ParallelGrid *) Bz)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
            //   }
            //
            //   if (solverSettings.getDoUseMetamaterials ())
            //   {
            //     if (doNeedEx)
            //     {
            //       ((ParallelGrid *) D1x)->Resize (parallelYeeLayout->getExSizeForCurNode ());
            //     }
            //     if (doNeedEy)
            //     {
            //       ((ParallelGrid *) D1y)->Resize (parallelYeeLayout->getEySizeForCurNode ());
            //     }
            //     if (doNeedEz)
            //     {
            //       ((ParallelGrid *) D1z)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
            //     }
            //
            //     if (doNeedHx)
            //     {
            //       ((ParallelGrid *) B1x)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
            //     }
            //     if (doNeedHy)
            //     {
            //       ((ParallelGrid *) B1y)->Resize (parallelYeeLayout->getHySizeForCurNode ());
            //     }
            //     if (doNeedHz)
            //     {
            //       ((ParallelGrid *) B1z)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
            //     }
            //   }
            //
            //   if (doNeedSigmaX)
            //   {
            //     ((ParallelGrid *) SigmaX)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
            //   }
            //   if (doNeedSigmaY)
            //   {
            //     ((ParallelGrid *) SigmaY)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
            //   }
            //   if (doNeedSigmaZ)
            //   {
            //     ((ParallelGrid *) SigmaZ)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
            //   }
            // }

            // if (solverSettings.getDoUseAmplitudeMode ())
            // {
            //   if (doNeedEx)
            //   {
            //     ((ParallelGrid *) ExAmplitude)->Resize (parallelYeeLayout->getExSizeForCurNode ());
            //   }
            //   if (doNeedEy)
            //   {
            //     ((ParallelGrid *) EyAmplitude)->Resize (parallelYeeLayout->getEySizeForCurNode ());
            //   }
            //   if (doNeedEz)
            //   {
            //     ((ParallelGrid *) EzAmplitude)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
            //   }
            //
            //   if (doNeedHx)
            //   {
            //     ((ParallelGrid *) HxAmplitude)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
            //   }
            //   if (doNeedHy)
            //   {
            //     ((ParallelGrid *) HyAmplitude)->Resize (parallelYeeLayout->getHySizeForCurNode ());
            //   }
            //   if (doNeedHz)
            //   {
            //     ((ParallelGrid *) HzAmplitude)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
            //   }
            // }

            // if (solverSettings.getDoUseMetamaterials ())
            // {
            //   ((ParallelGrid *) OmegaPE)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
            //   ((ParallelGrid *) GammaE)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
            //   ((ParallelGrid *) OmegaPM)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
            //   ((ParallelGrid *) GammaM)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
            // }

            //diffT += 1;
            //diffT *= 2;
          }
        }
    #else
        ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
    #endif
      }
    }
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
