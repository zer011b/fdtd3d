/*
 * Copyright (C) 2018 Gleb Balykov
 *
 * This file is part of fdtd3d.
 *
 * fdtd3d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * fdtd3d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with fdtd3d; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

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

  /*
   * Minimum number of stores steps is 2.
   * TODO: optimize to be able to allocate single step for non-PML modes.
   */
  int storedSteps = 2;

  if (SOLVER_SETTINGS.getDoUseMetamaterials () && SOLVER_SETTINGS.getDoUsePML ())
  {
    storedSteps = 3;
  }

#define GRID_NAME(x, y, steps, time_offset) \
  intScheme->x = intScheme->doNeed ## y ? new Grid<TC> (layout->get ## y ## Size (), steps, #x) : NULLPTR;
#define GRID_NAME_NO_CHECK(x, y, steps, time_offset) \
  intScheme->x = new Grid<TC> (layout->get ## y ## Size (), steps, #x);
#include "Grids2.inc.h"
#undef GRID_NAME
#undef GRID_NAME_NO_CHECK
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelper::allocateGridsInc (InternalScheme<Type, TCoord, layout_type> *intScheme, YeeGridLayout<Type, TCoord, layout_type> *layout)
{
  // TODO: allocate considering number of time steps
  intScheme->EInc = new Grid<GridCoordinate1D> (GRID_COORDINATE_1D (500*(layout->getSize ().get1 ()), CoordinateType::X), 2, "EInc");
  intScheme->HInc = new Grid<GridCoordinate1D> (GRID_COORDINATE_1D (500*(layout->getSize ().get1 ()), CoordinateType::X), 2, "HInc");
}

#ifdef PARALLEL_GRID

template <SchemeType_t Type, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelper::allocateParallelGrids1D (InternalScheme<Type, GridCoordinate1DTemplate, layout_type> *intScheme)
{
#ifndef GRID_1D
  ALWAYS_ASSERT_MESSAGE ("Solver is not compiled with support of 1D parallel grids.");
#else /* !GRID_1D */
  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (SOLVER_SETTINGS.getBufferSize (),
                                                                               SOLVER_SETTINGS.getBufferSize (),
                                                                               SOLVER_SETTINGS.getBufferSize (),
                                                                               intScheme->ct1, intScheme->ct2, intScheme->ct3);

  ParallelYeeGridLayout<Type, layout_type> *pLayout = (ParallelYeeGridLayout<Type, layout_type> *) intScheme->yeeLayout;

  int storedSteps = 3;

#define GRID_NAME(x, y, steps, time_offset) \
  intScheme->x = intScheme->doNeed ## y ? new ParallelGrid (pLayout->get ## y ## Size (), bufSize, 1, pLayout->get ## y ## SizeForCurNode (), steps, time_offset, #x) : NULLPTR;
#define GRID_NAME_NO_CHECK(x, y, steps, time_offset) \
  intScheme->x = new ParallelGrid (pLayout->get ## y ## Size (), bufSize, 1, pLayout->get ## y ## SizeForCurNode (), steps, time_offset, #x);
#include "Grids2.inc.h"
#undef GRID_NAME
#undef GRID_NAME_NO_CHECK

#endif /* GRID_1D */
}

template <SchemeType_t Type, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelper::allocateParallelGrids2D (InternalScheme<Type, GridCoordinate2DTemplate, layout_type> *intScheme)
{
#ifndef GRID_2D
  ALWAYS_ASSERT_MESSAGE ("Solver is not compiled with support of 2D parallel grids.");
#else /* !GRID_1D */
  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (SOLVER_SETTINGS.getBufferSize (),
                                                                               SOLVER_SETTINGS.getBufferSize (),
                                                                               SOLVER_SETTINGS.getBufferSize (),
                                                                               intScheme->ct1, intScheme->ct2, intScheme->ct3);

  ParallelYeeGridLayout<Type, layout_type> *pLayout = (ParallelYeeGridLayout<Type, layout_type> *) intScheme->yeeLayout;

  int storedSteps = 3;

#define GRID_NAME(x, y, steps, time_offset) \
  intScheme->x = intScheme->doNeed ## y ? new ParallelGrid (pLayout->get ## y ## Size (), bufSize, 1, pLayout->get ## y ## SizeForCurNode (), steps, time_offset, #x) : NULLPTR;
#define GRID_NAME_NO_CHECK(x, y, steps, time_offset) \
  intScheme->x = new ParallelGrid (pLayout->get ## y ## Size (), bufSize, 1, pLayout->get ## y ## SizeForCurNode (), steps, time_offset, #x);
#include "Grids2.inc.h"
#undef GRID_NAME
#undef GRID_NAME_NO_CHECK

#endif /* GRID_2D */
}

template <SchemeType_t Type, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelper::allocateParallelGrids3D (InternalScheme<Type, GridCoordinate3DTemplate, layout_type> *intScheme)
{
#ifndef GRID_3D
  ALWAYS_ASSERT_MESSAGE ("Solver is not compiled with support of 3D parallel grids.");
#else /* !GRID_3D */
  ParallelGridCoordinate bufSize = ParallelGridCoordinate::initAxesCoordinate (SOLVER_SETTINGS.getBufferSize (),
                                                                               SOLVER_SETTINGS.getBufferSize (),
                                                                               SOLVER_SETTINGS.getBufferSize (),
                                                                               intScheme->ct1, intScheme->ct2, intScheme->ct3);

  ParallelYeeGridLayout<Type, layout_type> *pLayout = (ParallelYeeGridLayout<Type, layout_type> *) intScheme->yeeLayout;

  int storedSteps = 3;

#define GRID_NAME(x, y, steps, time_offset) \
  intScheme->x = intScheme->doNeed ## y ? new ParallelGrid (pLayout->get ## y ## Size (), bufSize, 1, pLayout->get ## y ## SizeForCurNode (), steps, time_offset, #x) : NULLPTR;
#define GRID_NAME_NO_CHECK(x, y, steps, time_offset) \
  intScheme->x = new ParallelGrid (pLayout->get ## y ## Size (), bufSize, 1, pLayout->get ## y ## SizeForCurNode (), steps, time_offset, #x);
#include "Grids2.inc.h"
#undef GRID_NAME
#undef GRID_NAME_NO_CHECK

#endif /* GRID_3D */
}

#endif /* PARALLEL_GRID */

#ifdef CUDA_ENABLED
#ifdef CUDA_SOURCES

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

#define GRID_NAME(x, y, steps, time_offset) \
  intScheme->x = intScheme->doNeed ## y ? new CudaGrid<TC> (blockSize, bufSize, cpuScheme->x) : NULLPTR;
#define GRID_NAME_NO_CHECK(x, y, steps, time_offset) \
  intScheme->x = new CudaGrid<TC> (blockSize, bufSize, cpuScheme->x);
#include "Grids2.inc.h"
#undef GRID_NAME
#undef GRID_NAME_NO_CHECK

  if (SOLVER_SETTINGS.getDoUseTFSF ())
  {
    GridCoordinate1D zero = GRID_COORDINATE_1D (0, CoordinateType::X);
    intScheme->EInc = new CudaGrid<GridCoordinate1D> (GRID_COORDINATE_1D (500*(cpuScheme->yeeLayout->getSize ().get1 ()), CoordinateType::X), zero, cpuScheme->EInc);
    intScheme->HInc = new CudaGrid<GridCoordinate1D> (GRID_COORDINATE_1D (500*(cpuScheme->yeeLayout->getSize ().get1 ()), CoordinateType::X), zero, cpuScheme->HInc);
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
CUDA_HOST
void
InternalSchemeHelperGPU::freeGridsFromCPU (InternalSchemeGPU<Type, TCoord, layout_type> *intScheme)
{
#define GRID_NAME(x, y, steps, time_offset) \
  delete intScheme->x; \
  intScheme->x = NULLPTR;
#define GRID_NAME_NO_CHECK(x, y, steps, time_offset) \
  GRID_NAME(x, y, steps, time_offset)
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
#define GRID_NAME(x, y, steps, time_offset) \
  ASSERT (x == NULLPTR);
#define GRID_NAME_NO_CHECK(x, y, steps, time_offset) \
  GRID_NAME(x, y, steps, time_offset)
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

#define GRID_NAME(x, y, steps, time_offset) \
  if (gpuScheme->doNeed ## y) { cudaCheckErrorCmd (cudaMalloc ((void **) &gpuScheme->x, sizeof(CudaGrid<TC>))); }
#define GRID_NAME_NO_CHECK(x, y, steps, time_offset) \
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
#define GRID_NAME(x, y, steps, time_offset) \
  if (gpuScheme->doNeed ## y) { cudaCheckErrorCmd (cudaFree (gpuScheme->x)); gpuScheme->x = NULLPTR; }
#define GRID_NAME_NO_CHECK(x, y, steps, time_offset) \
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
#define GRID_NAME(x, y, steps, time_offset) \
  if (gpuScheme->doNeed ## y) { gpuScheme->x->copyFromCPU (start, end); }
#define GRID_NAME_NO_CHECK(x, y, steps, time_offset) \
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

#define GRID_NAME(x, y, steps, time_offset) \
  if (gpuScheme->doNeed ## y) { cudaCheckErrorCmd (cudaMemcpy (gpuScheme->x, intScheme->x, sizeof(CudaGrid<TC>), cudaMemcpyHostToDevice)); }
#define GRID_NAME_NO_CHECK(x, y, steps, time_offset) \
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

  allocateGridsFromCPU (cpuScheme, blockSize, bufSize);

  cudaCheckErrorCmd (cudaMalloc ((void **) &d_norm, 6 * sizeof(FPValue)));

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

  d_norm = gpuScheme->d_norm;

  initCoordTypes ();

  TC one (1, 1, 1
#ifdef DEBUG_INFO
          , ct1, ct2, ct3
#endif
          );

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

#endif /* CUDA_SOURCES */
#endif /* CUDA_ENABLED */
