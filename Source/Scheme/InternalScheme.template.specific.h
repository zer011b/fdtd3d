template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
InternalScheme<Type, TCoord, layout_type>::~InternalScheme ()
{
#define GRID_NAME(x) \
  delete x;
#include "Grids.inc.h"
#undef GRID_NAME

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    delete EInc;
    delete HInc;
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalScheme<Type, TCoord, layout_type>::init (YeeGridLayout<Type, TCoord, layout_type> *layout,
                                                           bool parallel)
{
  yeeLayout = layout;
  useParallel = parallel;

  initCoordTypes ();

  if (SOLVER_SETTINGS.getDoUseNTFF ())
  {
    leftNTFF = TC::initAxesCoordinate (SOLVER_SETTINGS.getNTFFSizeX (), SOLVER_SETTINGS.getNTFFSizeY (), SOLVER_SETTINGS.getNTFFSizeZ (),
                                       ct1, ct2, ct3);
    rightNTFF = layout->getEzSize () - leftNTFF + TC (1, 1, 1
#ifdef DEBUG_INFO
                                                      , ct1, ct2, ct3
#endif
                                                      );
  }

  if (useParallel)
  {
#if defined (PARALLEL_GRID) && ! defined (__CUDA_ARCH__)
    allocateParallelGrids ();
#else
    ALWAYS_ASSERT (false);
#endif
  }
  else
  {
    allocateGrids ();
  }

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    allocateGridsInc ();
  }

  isInitialized = true;
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelper::allocateGrids (InternalScheme<Type, TCoord, layout_type> *intScheme, YeeGridLayout<Type, TCoord, layout_type> *layout)
{
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

  int storedSteps = 3;

#define GRID_NAME(x, y, steps) \
  intScheme->x = intScheme->doNeed ## y ? new Grid<TC> (layout->get ## y ## Size (), 0, steps, #x) : NULLPTR;
#define GRID_NAME_NO_CHECK(x, y, steps) \
  intScheme->x = new Grid<TC> (layout->get ## y ## Size (), 0, steps, #x);
#include "Grids2.inc.h"
#undef GRID_NAME
#undef GRID_NAME_NO_CHECK
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelper::allocateGridsInc (InternalScheme<Type, TCoord, layout_type> *intScheme, YeeGridLayout<Type, TCoord, layout_type> *layout)
{
  intScheme->EInc = new Grid<GridCoordinate1D> (GRID_COORDINATE_1D (500*(layout->getSize ().get1 ()), CoordinateType::X), 0, 2, "EInc");
  intScheme->HInc = new Grid<GridCoordinate1D> (GRID_COORDINATE_1D (500*(layout->getSize ().get1 ()), CoordinateType::X), 0, 2, "HInc");
}

#ifdef PARALLEL_GRID

template <SchemeType_t Type, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelper::allocateParallelGrids (InternalScheme<Type, ParallelGridCoordinateTemplate, layout_type> *intScheme)
{
  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (SOLVER_SETTINGS.getBufferSize (),
                                                                               SOLVER_SETTINGS.getBufferSize (),
                                                                               SOLVER_SETTINGS.getBufferSize (),
                                                                               ct1, ct2, ct3);

  ParallelYeeGridLayout<Type, layout_type> *pLayout = intScheme->yeeLayout;

  int storedSteps = 3;

#define GRID_NAME(x, y, steps) \
  intScheme->x = intScheme->doNeed ## y ? new ParallelGrid (pLayout->get ## y ## Size (), bufSize, 0, pLayout->get ## y ## SizeForCurNode (), steps, #x) : NULLPTR;
#define GRID_NAME_NO_CHECK(x, y, steps) \
  intScheme->x = new ParallelGrid (pLayout->get ## y ## Size (), bufSize, 0, pLayout->get ## y ## SizeForCurNode (), steps, #x);
#include "Grids2.inc.h"
#undef GRID_NAME
#undef GRID_NAME_NO_CHECK
}

#endif /* PARALLEL_GRID */

#ifdef CUDA_ENABLED

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelperGPU::allocateGridsFromCPU (InternalSchemeGPU<Type, TCoord, layout_type> *intScheme,
                                               InternalScheme<Type, TCoord, layout_type> *cpuScheme, TCoord<grid_coord, true> blockSize, TCoord<grid_coord, true> bufSize)
{
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

#define GRID_NAME(x, y, steps) \
  intScheme->x = intScheme->doNeed ## y ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->x) : NULLPTR;
#define GRID_NAME_NO_CHECK(x, y, steps) \
  intScheme->x = new CudaGrid<TC> (blockSize, bufSize, cpuScheme->x);
#include "Grids2.inc.h"
#undef GRID_NAME
#undef GRID_NAME_NO_CHECK

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    GridCoordinate1D one = GRID_COORDINATE_1D (1, CoordinateType::X);
    intScheme->EInc = new CudaGrid<GridCoordinate1D> (GRID_COORDINATE_1D (500*(cpuScheme->yeeLayout->getSize ().get1 ()), CoordinateType::X), one, cpuScheme->EInc);
    intScheme->HInc = new CudaGrid<GridCoordinate1D> (GRID_COORDINATE_1D (500*(cpuScheme->yeeLayout->getSize ().get1 ()), CoordinateType::X), one, cpuScheme->HInc);
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelperGPU::freeGridsFromCPU (InternalSchemeGPU<Type, TCoord, layout_type> *intScheme)
{
#define GRID_NAME(x, y, steps) \
  delete intScheme->x; \
  intScheme->x = NULLPTR;
#define GRID_NAME_NO_CHECK(x, y, steps) \
  GRID_NAME(x, y, steps)
#include "Grids2.inc.h"
#undef GRID_NAME
#undef GRID_NAME_NO_CHECK

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    delete intScheme->EInc;
    delete intScheme->HInc;
    intScheme->EInc = NULLPTR;
    intScheme->HInc = NULLPTR;
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
InternalSchemeGPU<Type, TCoord, layout_type>::~InternalSchemeGPU ()
{
#define GRID_NAME(x, y, steps) \
  ASSERT (x == NULLPTR);
#define GRID_NAME_NO_CHECK(x, y, steps) \
  GRID_NAME(x, y, steps)
#include "Grids2.inc.h"
#undef GRID_NAME
#undef GRID_NAME_NO_CHECK

  ASSERT (EInc == NULLPTR);
  ASSERT (HInc == NULLPTR);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelperGPU::allocateGridsOnGPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme)
{
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

#define GRID_NAME(x, y, steps) \
  if (gpuScheme->doNeed ## y) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->x, sizeof(CudaGrid<TC>))); }
#define GRID_NAME_NO_CHECK(x, y, steps) \
  cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->x, sizeof(CudaGrid<TC>)));
#include "Grids2.inc.h"
#undef GRID_NAME
#undef GRID_NAME_NO_CHECK

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->EInc, sizeof(CudaGrid<GridCoordinate1D>)));
    cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->HInc, sizeof(CudaGrid<GridCoordinate1D>)));
  }
}


template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelperGPU::freeGridsOnGPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme)
{
#define GRID_NAME(x, y, steps) \
  if (gpuScheme->doNeed ## y) { cudaCheckErrorCmd (cudaFree (gpuScheme->x)); gpuScheme->x = NULLPTR; }
#define GRID_NAME_NO_CHECK(x, y, steps) \
  cudaCheckErrorCmd (cudaFree (gpuScheme->x)); gpuScheme->x = NULLPTR;
#include "Grids2.inc.h"
#undef GRID_NAME
#undef GRID_NAME_NO_CHECK

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    cudaCheckErrorCmd (cudaFree (gpuScheme->EInc));
    cudaCheckErrorCmd (cudaFree (gpuScheme->HInc));
    gpuScheme->EInc = NULLPTR;
    gpuScheme->HInc = NULLPTR;
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelperGPU::copyGridsFromCPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                  TCoord<grid_coord, true> start,
                  TCoord<grid_coord, true> end)
{
#define GRID_NAME(x, y, steps) \
  if (gpuScheme->doNeed ## y) { gpuScheme->x->copyFromCPU (start, end); }
#define GRID_NAME_NO_CHECK(x, y, steps) \
  gpuScheme->x->copyFromCPU (start, end);
#include "Grids2.inc.h"
#undef GRID_NAME
#undef GRID_NAME_NO_CHECK

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    GridCoordinate1D zero = GRID_COORDINATE_1D (0, CoordinateType::X);
    gpuScheme->EInc->copyFromCPU (zero, GRID_COORDINATE_1D (500*(gpuScheme->yeeLayout->getSize ().get1 ()), CoordinateType::X));
    gpuScheme->HInc->copyFromCPU (zero, GRID_COORDINATE_1D (500*(gpuScheme->yeeLayout->getSize ().get1 ()), CoordinateType::X));
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelperGPU::copyGridsToGPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                         InternalSchemeGPU<Type, TCoord, layout_type> *intScheme)
{
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

#define GRID_NAME(x, y, steps) \
  if (gpuScheme->doNeed ## y) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->x, intScheme->x, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
#define GRID_NAME_NO_CHECK(x, y, steps) \
  cudaCheckErrorCmd (cudaMemcpy (gpuScheme->x, intScheme->x, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice));
#include "Grids2.inc.h"
#undef GRID_NAME
#undef GRID_NAME_NO_CHECK

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    cudaCheckErrorCmd (cudaMemcpy (gpuScheme->EInc, intScheme->EInc, sizeof(CudaGrid<GridCoordinate1D>), cudaMemcpyHostToDevice));
    cudaCheckErrorCmd (cudaMemcpy (gpuScheme->HInc, intScheme->HInc, sizeof(CudaGrid<GridCoordinate1D>), cudaMemcpyHostToDevice));
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelperGPU::copyGridsBackToCPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme,
                                             time_step N,
                                             bool finalCopy) /**< used for grid, which should be copied back from CPU only once, i.e. TFSF */
{
  if (gpuScheme->doNeedEx)
  {
    ASSERT (gpuScheme->Ex->getShareStep () == N);
    gpuScheme->Ex->copyToCPU ();
    gpuScheme->Ex->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (gpuScheme->Dx->getShareStep () == N);
      gpuScheme->Dx->copyToCPU ();
      gpuScheme->Dx->zeroShareStep ();

      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        ASSERT (gpuScheme->D1x->getShareStep () == N);
        gpuScheme->D1x->copyToCPU ();
        gpuScheme->D1x->zeroShareStep ();
      }
    }

    if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
    {
      ASSERT (gpuScheme->ExAmplitude->getShareStep () == N);
      gpuScheme->ExAmplitude->copyToCPU ();
      gpuScheme->ExAmplitude->zeroShareStep ();
    }
  }

  if (gpuScheme->doNeedEy)
  {
    ASSERT (gpuScheme->Ey->getShareStep () == N);
    gpuScheme->Ey->copyToCPU ();
    gpuScheme->Ey->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (gpuScheme->Dy->getShareStep () == N);
      gpuScheme->Dy->copyToCPU ();
      gpuScheme->Dy->zeroShareStep ();

      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        ASSERT (gpuScheme->D1y->getShareStep () == N);
        gpuScheme->D1y->copyToCPU ();
        gpuScheme->D1y->zeroShareStep ();
      }
    }

    if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
    {
      ASSERT (gpuScheme->EyAmplitude->getShareStep () == N);
      gpuScheme->EyAmplitude->copyToCPU ();
      gpuScheme->EyAmplitude->zeroShareStep ();
    }
  }

  if (gpuScheme->doNeedEz)
  {
    ASSERT (gpuScheme->Ez->getShareStep () == N);
    gpuScheme->Ez->copyToCPU ();
    gpuScheme->Ez->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (gpuScheme->Dz->getShareStep () == N);
      gpuScheme->Dz->copyToCPU ();
      gpuScheme->Dz->zeroShareStep ();

      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        ASSERT (gpuScheme->D1z->getShareStep () == N);
        gpuScheme->D1z->copyToCPU ();
        gpuScheme->D1z->zeroShareStep ();
      }
    }

    if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
    {
      ASSERT (gpuScheme->EzAmplitude->getShareStep () == N);
      gpuScheme->EzAmplitude->copyToCPU ();
      gpuScheme->EzAmplitude->zeroShareStep ();
    }
  }

  if (gpuScheme->doNeedHx)
  {
    ASSERT (gpuScheme->Hx->getShareStep () == N);
    gpuScheme->Hx->copyToCPU ();
    gpuScheme->Hx->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (gpuScheme->Bx->getShareStep () == N);
      gpuScheme->Bx->copyToCPU ();
      gpuScheme->Bx->zeroShareStep ();

      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        ASSERT (gpuScheme->B1x->getShareStep () == N);
        gpuScheme->B1x->copyToCPU ();
        gpuScheme->B1x->zeroShareStep ();
      }
    }

    if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
    {
      ASSERT (gpuScheme->HxAmplitude->getShareStep () == N);
      gpuScheme->HxAmplitude->copyToCPU ();
      gpuScheme->HxAmplitude->zeroShareStep ();
    }
  }

  if (gpuScheme->doNeedHy)
  {
    ASSERT (gpuScheme->Hy->getShareStep () == N);
    gpuScheme->Hy->copyToCPU ();
    gpuScheme->Hy->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (gpuScheme->By->getShareStep () == N);
      gpuScheme->By->copyToCPU ();
      gpuScheme->By->zeroShareStep ();

      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        ASSERT (gpuScheme->B1y->getShareStep () == N);
        gpuScheme->B1y->copyToCPU ();
        gpuScheme->B1y->zeroShareStep ();
      }
    }

    if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
    {
      ASSERT (gpuScheme->HyAmplitude->getShareStep () == N);
      gpuScheme->HyAmplitude->copyToCPU ();
      gpuScheme->HyAmplitude->zeroShareStep ();
    }
  }

  if (gpuScheme->doNeedHz)
  {
    ASSERT (gpuScheme->Hz->getShareStep () == N);
    gpuScheme->Hz->copyToCPU ();
    gpuScheme->Hz->zeroShareStep ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      ASSERT (gpuScheme->Bz->getShareStep () == N);
      gpuScheme->Bz->copyToCPU ();
      gpuScheme->Bz->zeroShareStep ();

      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        ASSERT (gpuScheme->B1z->getShareStep () == N);
        gpuScheme->B1z->copyToCPU ();
        gpuScheme->B1z->zeroShareStep ();
      }
    }

    if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
    {
      ASSERT (gpuScheme->HzAmplitude->getShareStep () == N);
      gpuScheme->HzAmplitude->copyToCPU ();
      gpuScheme->HzAmplitude->zeroShareStep ();
    }
  }

  if (finalCopy)
  {
    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      gpuScheme->EInc->copyToCPU ();
      gpuScheme->HInc->copyToCPU ();
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeGPU<Type, TCoord, layout_type>::initFromCPU (InternalScheme<Type, TCoord, layout_type> *cpuScheme,
                                                                  TC blockSize,
                                                                  TC bufSize)
{
  ASSERT (cpuScheme->isInitialized);

  yeeLayout = cpuScheme->yeeLayout;
  initScheme (cpuScheme->gridStep, cpuScheme->sourceWaveLength);

  initCoordTypes ();

  TC one (1, 1, 1
#ifdef DEBUG_INFO
          , ct1, ct2, ct3
#endif
          );

  if (SOLVER_SETTINGS.getDoUseNTFF ())
  {
    leftNTFF = TC::initAxesCoordinate (SOLVER_SETTINGS.getNTFFSizeX (), SOLVER_SETTINGS.getNTFFSizeY (), SOLVER_SETTINGS.getNTFFSizeZ (),
                                       ct1, ct2, ct3);
    rightNTFF = cpuScheme->yeeLayout->getEzSize () - leftNTFF + one;
  }

  allocateGridsFromCPU (cpuScheme, blockSize, bufSize);
  
  cudaCheckErrorCmd (cudaMalloc ((void **) &d_norm, sizeof(6 * FPValue)));

  isInitialized = true;
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeGPU<Type, TCoord, layout_type>::uninitFromCPU ()
{
  ASSERT (isInitialized);

  freeGridsFromCPU ();
  
  cudaCheckErrorCmd (cudaFree (d_norm));
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeGPU<Type, TCoord, layout_type>::initOnGPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme)
{
  yeeLayout = gpuScheme->yeeLayout;
  initScheme (gpuScheme->gridStep, gpuScheme->sourceWaveLength);

  initCoordTypes ();

  TC one (1, 1, 1
#ifdef DEBUG_INFO
          , ct1, ct2, ct3
#endif
          );

  if (SOLVER_SETTINGS.getDoUseNTFF ())
  {
    leftNTFF = TC::initAxesCoordinate (SOLVER_SETTINGS.getNTFFSizeX (), SOLVER_SETTINGS.getNTFFSizeY (), SOLVER_SETTINGS.getNTFFSizeZ (),
                                       ct1, ct2, ct3);
    rightNTFF = gpuScheme->yeeLayout->getEzSize () - leftNTFF + one;
  }

  cudaCheckErrorCmd (cudaMalloc ((void **) &yeeLayout, sizeof(YeeGridLayout<Type, TCoord, layout_type>)));
  cudaCheckErrorCmd (cudaMemcpy (yeeLayout, gpuScheme->yeeLayout, sizeof(YeeGridLayout<Type, TCoord, layout_type>), cudaMemcpyHostToDevice));

  allocateGridsOnGPU ();

  isInitialized = true;
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeGPU<Type, TCoord, layout_type>::uninitOnGPU ()
{
  ASSERT (isInitialized);

  cudaCheckErrorCmd (cudaFree (yeeLayout));

  freeGridsOnGPU ();
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeGPU<Type, TCoord, layout_type>::copyFromCPU (TCoord<grid_coord, true> start,
                                                                   TCoord<grid_coord, true> end)
{
  ASSERT (isInitialized);

  copyGridsFromCPU (start, end);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeGPU<Type, TCoord, layout_type>::copyToGPU (InternalSchemeGPU<Type, TCoord, layout_type> *gpuScheme)
{
  ASSERT (isInitialized);
  ASSERT (gpuScheme->isInitialized);

  copyGridsToGPU (gpuScheme);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeGPU<Type, TCoord, layout_type>::copyBackToCPU (time_step N,
                                                             bool finalCopy)
{
  ASSERT (isInitialized);

  copyGridsBackToCPU (N, finalCopy);
}

#endif /* CUDA_ENABLED */
