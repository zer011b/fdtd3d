#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"
#include "TXTDumper.h"
#include "TXTLoader.h"
#include "Kernels.h"
#include "Settings.h"
#include "Scheme.h"
#include "Approximation.h"

#if defined (PARALLEL_GRID)
#include <mpi.h>
#endif

#include <cmath>

#if defined (CUDA_ENABLED)
#include "CudaInterface.h"
#endif

#define _NAME(A,B) A ##B

#define SPECIALIZE_TEMPLATE_FUNC(RET, STYPE, COORD, LAYOUT_TYPE, NAME, ARGSND, ARGS, NAME_HELPER) \
  template <> \
  RET \
  Scheme<static_cast<SchemeType_t> (SchemeType::STYPE), COORD, LAYOUT_TYPE>::NAME ARGSND \
  { \
    return SchemeHelper::NAME_HELPER ARGS; \
  }

#define SPECIALIZE_TEMPLATE(RET1D, RET2D, RET3D, NAME, ARGS1D, ARGS2D, ARGS3D, ARGS) \
  SPECIALIZE_TEMPLATE_FUNC(RET1D, Dim1_ExHy, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, _NAME(NAME, 1D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET1D, Dim1_ExHz, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, _NAME(NAME, 1D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET1D, Dim1_EyHx, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, _NAME(NAME, 1D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET1D, Dim1_EyHz, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, _NAME(NAME, 1D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET1D, Dim1_EzHx, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, _NAME(NAME, 1D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET1D, Dim1_EzHy, GridCoordinate1DTemplate, E_CENTERED, NAME, ARGS1D, ARGS, _NAME(NAME, 1D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET2D, Dim2_TEx, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, _NAME(NAME, 2D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET2D, Dim2_TEy, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, _NAME(NAME, 2D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET2D, Dim2_TEz, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, _NAME(NAME, 2D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET2D, Dim2_TMx, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, _NAME(NAME, 2D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET2D, Dim2_TMy, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, _NAME(NAME, 2D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET2D, Dim2_TMz, GridCoordinate2DTemplate, E_CENTERED, NAME, ARGS2D, ARGS, _NAME(NAME, 2D)) \
  SPECIALIZE_TEMPLATE_FUNC(RET3D, Dim3, GridCoordinate3DTemplate, E_CENTERED, NAME, ARGS3D, ARGS, _NAME(NAME, 3D))

SPECIALIZE_TEMPLATE(GridCoordinate1D, GridCoordinate2D, GridCoordinate3D,
                    getStartCoordRes,
                    (OrthogonalAxis orthogonalAxis, GridCoordinate1D start, GridCoordinate1D size),
                    (OrthogonalAxis orthogonalAxis, GridCoordinate2D start, GridCoordinate2D size),
                    (OrthogonalAxis orthogonalAxis, GridCoordinate3D start, GridCoordinate3D size),
                    (orthogonalAxis, start, size))

SPECIALIZE_TEMPLATE(GridCoordinate1D, GridCoordinate2D, GridCoordinate3D,
                    getEndCoordRes,
                    (OrthogonalAxis orthogonalAxis, GridCoordinate1D end, GridCoordinate1D size),
                    (OrthogonalAxis orthogonalAxis, GridCoordinate2D end, GridCoordinate2D size),
                    (OrthogonalAxis orthogonalAxis, GridCoordinate3D end, GridCoordinate3D size),
                    (orthogonalAxis, end, size))

SPECIALIZE_TEMPLATE(NPair, NPair, NPair,
                    ntffN,
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate1D> *curEz, Grid<GridCoordinate1D> *curHx, Grid<GridCoordinate1D> *curHy, Grid<GridCoordinate1D> *curHz),
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate2D> *curEz, Grid<GridCoordinate2D> *curHx, Grid<GridCoordinate2D> *curHy, Grid<GridCoordinate2D> *curHz),
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *curEz, Grid<GridCoordinate3D> *curHx, Grid<GridCoordinate3D> *curHy, Grid<GridCoordinate3D> *curHz),
                    (angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, HInc, curEz, curHx, curHy, curHz)) // TODO: check sourceWaveLengthNumerical here

SPECIALIZE_TEMPLATE(NPair, NPair, NPair,
                    ntffL,
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate1D> *curEx, Grid<GridCoordinate1D> *curEy, Grid<GridCoordinate1D> *curEz),
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate2D> *curEx, Grid<GridCoordinate2D> *curEy, Grid<GridCoordinate2D> *curEz),
                    (FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *curEx, Grid<GridCoordinate3D> *curEy, Grid<GridCoordinate3D> *curEz),
                    (angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, EInc, curEx, curEy, curEz)) // TODO: check sourceWaveLengthNumerical here

SPECIALIZE_TEMPLATE(bool, bool, bool,
                    doSkipMakeScattered,
                    (GridCoordinateFP1D pos),
                    (GridCoordinateFP2D pos),
                    (GridCoordinateFP3D pos),
                    (pos, yeeLayout->getLeftBorderTFSF (), yeeLayout->getRightBorderTFSF ()))

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
Scheme<Type, TCoord, layout_type>::Scheme (YeeGridLayout<Type, TCoord, layout_type> *layout,
                                           bool parallelLayout,
                                           const TC& totSize,
                                           time_step tStep)
  : useParallel (false)
  , internalScheme ()
#ifdef CUDA_ENABLED
  , intGPUScheme ()
  , d_intGPUScheme (NULLPTR)
#endif /* CUDA_ENABLED */
  , totalEx (NULLPTR)
  , totalEy (NULLPTR)
  , totalEz (NULLPTR)
  , totalHx (NULLPTR)
  , totalHy (NULLPTR)
  , totalHz (NULLPTR)
  , totalInitialized (false)
  , totalEps (NULLPTR)
  , totalMu (NULLPTR)
  , totalOmegaPE (NULLPTR)
  , totalOmegaPM (NULLPTR)
  , totalGammaE (NULLPTR)
  , totalGammaM (NULLPTR)
  , totalStep (tStep)
  , process (-1)
  , numProcs (-1)
{
  ASSERT (!SOLVER_SETTINGS.getDoUseTFSF ()
          || (SOLVER_SETTINGS.getDoUseTFSF ()
              && (yeeLayout->getLeftBorderTFSF () != TC (0, 0, 0, ct1, ct2, ct3)
                  || yeeLayout->getRightBorderTFSF () != yeeLayout->getSize ())));

  ASSERT (!SOLVER_SETTINGS.getDoUsePML ()
          || (SOLVER_SETTINGS.getDoUsePML () && (yeeLayout->getSizePML () != TC (0, 0, 0, ct1, ct2, ct3))));

  ASSERT (!SOLVER_SETTINGS.getDoUseAmplitudeMode ()
          || SOLVER_SETTINGS.getDoUseAmplitudeMode () && SOLVER_SETTINGS.getNumAmplitudeSteps () != 0);

#ifdef COMPLEX_FIELD_VALUES
  ASSERT (!SOLVER_SETTINGS.getDoUseAmplitudeMode ());
#endif /* COMPLEX_FIELD_VALUES */

  if (SOLVER_SETTINGS.getDoUseParallelGrid ())
  {
#ifndef PARALLEL_GRID
    ALWAYS_ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.")
#endif

    ALWAYS_ASSERT (parallelLayout);
#ifdef PARALLEL_GRID
    ALWAYS_ASSERT ((TCoord<grid_coord, false>::dimension == ParallelGridCoordinateTemplate<grid_coord, false>::dimension));
#endif

    useParallel = true;
  }

  internalScheme.init (layout, useParallel);

#ifdef CUDA_ENABLED
  intGPUScheme.initFromCPU (&internalScheme);
#endif /* CUDA_ENABLED */

  if (!useParallel)
  {
    totalEps = internalScheme.Eps;
    totalMu = internalScheme.Mu;
    totalOmegaPE = internalScheme.OmegaPE;
    totalOmegaPM = internalScheme.OmegaPM;
    totalGammaE = internalScheme.GammaE;
    totalGammaM = internalScheme.GammaM;
  }
  else
  {
    /*
     * In parallel mode total grids will be allocated if required
     */
  }

  if (SOLVER_SETTINGS.getDoSaveAsBMP ())
  {
    PaletteType palette = PaletteType::PALETTE_GRAY;
    OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;

    if (SOLVER_SETTINGS.getDoUsePaletteGray ())
    {
      palette = PaletteType::PALETTE_GRAY;
    }
    else if (SOLVER_SETTINGS.getDoUsePaletteRGB ())
    {
      palette = PaletteType::PALETTE_BLUE_GREEN_RED;
    }

    if (SOLVER_SETTINGS.getDoUseOrthAxisX ())
    {
      orthogonalAxis = OrthogonalAxis::X;
    }
    else if (SOLVER_SETTINGS.getDoUseOrthAxisY ())
    {
      orthogonalAxis = OrthogonalAxis::Y;
    }
    else if (SOLVER_SETTINGS.getDoUseOrthAxisZ ())
    {
      orthogonalAxis = OrthogonalAxis::Z;
    }

    dumper[FILE_TYPE_BMP] = new BMPDumper<TC> ();
    ((BMPDumper<TC> *) dumper[FILE_TYPE_BMP])->initializeHelper (palette, orthogonalAxis);

    dumper1D[FILE_TYPE_BMP] = new BMPDumper<GridCoordinate1D> ();
    ((BMPDumper<GridCoordinate1D> *) dumper1D[FILE_TYPE_BMP])->initializeHelper (palette, orthogonalAxis);
  }
  else
  {
    dumper[FILE_TYPE_BMP] = NULLPTR;
    dumper1D[FILE_TYPE_BMP] = NULLPTR;
  }

  if (SOLVER_SETTINGS.getDoSaveAsDAT ())
  {
    dumper[FILE_TYPE_DAT] = new DATDumper<TC> ();
    dumper1D[FILE_TYPE_DAT] = new DATDumper<GridCoordinate1D> ();
  }
  else
  {
    dumper[FILE_TYPE_DAT] = NULLPTR;
    dumper1D[FILE_TYPE_DAT] = NULLPTR;
  }

  if (SOLVER_SETTINGS.getDoSaveAsTXT ())
  {
    dumper[FILE_TYPE_TXT] = new TXTDumper<TC> ();
    dumper1D[FILE_TYPE_TXT] = new TXTDumper<GridCoordinate1D> ();
  }
  else
  {
    dumper[FILE_TYPE_TXT] = NULLPTR;
    dumper1D[FILE_TYPE_TXT] = NULLPTR;
  }

  if (!SOLVER_SETTINGS.getEpsFileName ().empty ()
      || !SOLVER_SETTINGS.getMuFileName ().empty ()
      || !SOLVER_SETTINGS.getOmegaPEFileName ().empty ()
      || !SOLVER_SETTINGS.getOmegaPMFileName ().empty ()
      || !SOLVER_SETTINGS.getGammaEFileName ().empty ()
      || !SOLVER_SETTINGS.getGammaMFileName ().empty ())
  {
    {
      loader[FILE_TYPE_BMP] = new BMPLoader<TC> ();

      PaletteType palette = PaletteType::PALETTE_GRAY;
      OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;

      if (SOLVER_SETTINGS.getDoUsePaletteGray ())
      {
        palette = PaletteType::PALETTE_GRAY;
      }
      else if (SOLVER_SETTINGS.getDoUsePaletteRGB ())
      {
        palette = PaletteType::PALETTE_BLUE_GREEN_RED;
      }

      if (SOLVER_SETTINGS.getDoUseOrthAxisX ())
      {
        orthogonalAxis = OrthogonalAxis::X;
      }
      else if (SOLVER_SETTINGS.getDoUseOrthAxisY ())
      {
        orthogonalAxis = OrthogonalAxis::Y;
      }
      else if (SOLVER_SETTINGS.getDoUseOrthAxisZ ())
      {
        orthogonalAxis = OrthogonalAxis::Z;
      }

      ((BMPLoader<TC> *) loader[FILE_TYPE_BMP])->initializeHelper (palette, orthogonalAxis);
    }
    {
      loader[FILE_TYPE_DAT] = new DATLoader<TC> ();
    }
    {
      loader[FILE_TYPE_TXT] = new TXTLoader<TC> ();
    }
  }
  else
  {
    loader[FILE_TYPE_BMP] = NULLPTR;
    loader[FILE_TYPE_DAT] = NULLPTR;
    loader[FILE_TYPE_TXT] = NULLPTR;
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
Scheme<Type, TCoord, layout_type>::~Scheme ()
{
  if (totalInitialized)
  {
    delete totalEx;
    delete totalEy;
    delete totalEz;

    delete totalHx;
    delete totalHy;
    delete totalHz;
  }

  if (useParallel)
  {
#if defined (PARALLEL_GRID)
    delete totalEps;
    delete totalMu;

    delete totalOmegaPE;
    delete totalOmegaPM;
    delete totalGammaE;
    delete totalGammaM;
#else /* PARALLEL_GRID */
    UNREACHABLE;
#endif /* !PARALLEL_GRID */
  }

  delete dumper[FILE_TYPE_BMP];
  delete dumper[FILE_TYPE_DAT];
  delete dumper[FILE_TYPE_TXT];

  delete loader[FILE_TYPE_BMP];
  delete loader[FILE_TYPE_DAT];
  delete loader[FILE_TYPE_TXT];

  delete dumper1D[FILE_TYPE_BMP];
  delete dumper1D[FILE_TYPE_DAT];
  delete dumper1D[FILE_TYPE_TXT];
}

/*
 * Specialization for Sigma
 */
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaZ);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaY);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaZ);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaX);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaY);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaX);
};

template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaY);
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaZ);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaX);
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaZ);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaX);
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaY);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaY);
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaZ);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaX);
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaZ);
};
template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaX);
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaY);
};

template <>
void
Scheme<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED>::initSigmas ()
{
  SchemeHelper::initSigmaX<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaX);
  SchemeHelper::initSigmaY<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaY);
  SchemeHelper::initSigmaZ<static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED> (yeeLayout, gridStep, internalScheme.SigmaZ);
};

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::performNSteps (time_step startStep, time_step numberTimeSteps)
{
  time_step diffT = SOLVER_SETTINGS.getRebalanceStep ();

  int processId = 0;

  time_step stepLimit = startStep + numberTimeSteps;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  if (processId == 0)
  {
    DPRINTF (LOG_LEVEL_STAGES, "Performing computations for [%u,%u] time steps.\n", startStep, stepLimit);
  }

  TC zero (0, 0, 0
#ifdef DEBUG_INFO
           , ct1, ct2, ct3
#endif /* DEBUG_INFO */
           );

  for (time_step t = startStep; t < stepLimit; ++t)
  {
    if (processId == 0)
    {
      DPRINTF (LOG_LEVEL_STAGES, "Calculating time step %u...\n", t);
    }

    TC ExStart = internalScheme.doNeedEx ? internalScheme.Ex->getComputationStart (yeeLayout->getExStartDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                         , ct1, ct2, ct3
#endif
                                                                                         );
    TC ExEnd = internalScheme.doNeedEx ? internalScheme.Ex->getComputationEnd (yeeLayout->getExEndDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                   , ct1, ct2, ct3
#endif
                                                                                   );

    TC EyStart = internalScheme.doNeedEy ? internalScheme.Ey->getComputationStart (yeeLayout->getEyStartDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                         , ct1, ct2, ct3
#endif
                                                                                         );
    TC EyEnd = internalScheme.doNeedEy ? internalScheme.Ey->getComputationEnd (yeeLayout->getEyEndDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                   , ct1, ct2, ct3
#endif
                                                                                   );

    TC EzStart = internalScheme.doNeedEz ? internalScheme.Ez->getComputationStart (yeeLayout->getEzStartDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                         , ct1, ct2, ct3
#endif
                                                                                         );
    TC EzEnd = internalScheme.doNeedEz ? internalScheme.Ez->getComputationEnd (yeeLayout->getEzEndDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                   , ct1, ct2, ct3
#endif
                                                                                   );

    TC HxStart = internalScheme.doNeedHx ? internalScheme.Hx->getComputationStart (yeeLayout->getHxStartDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                         , ct1, ct2, ct3
#endif
                                                                                         );
    TC HxEnd = internalScheme.doNeedHx ? internalScheme.Hx->getComputationEnd (yeeLayout->getHxEndDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                   , ct1, ct2, ct3
#endif
                                                                                   );

    TC HyStart = internalScheme.doNeedHy ? internalScheme.Hy->getComputationStart (yeeLayout->getHyStartDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                         , ct1, ct2, ct3
#endif
                                                                                         );
    TC HyEnd = internalScheme.doNeedHy ? internalScheme.Hy->getComputationEnd (yeeLayout->getHyEndDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                   , ct1, ct2, ct3
#endif
                                                                                   );

    TC HzStart = internalScheme.doNeedHz ? internalScheme.Hz->getComputationStart (yeeLayout->getHzStartDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                         , ct1, ct2, ct3
#endif
                                                                                         );
    TC HzEnd = internalScheme.doNeedHz ? internalScheme.Hz->getComputationEnd (yeeLayout->getHzEndDiff ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                   , ct1, ct2, ct3
#endif
                                                                                   );

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D (0
#ifdef DEBUG_INFO
                               , CoordinateType::X
#endif /* DEBUG_INFO */
                               );

      performPlaneWaveESteps (t, zero1D, intScheme.EInc->getSize ());
      internalScheme.EInc->shiftInTime (zero1D, EInc->getSize ());
      internalScheme.EInc->nextTimeStep ();
    }

    if (useParallel && SOLVER_SETTINGS.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StartCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
                      "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
#endif
    }

    if (internalScheme.doNeedEx)
    {
      internalScheme.performFieldSteps<static_cast<uint8_t> (GridType::EX)> (t, ExStart, ExEnd);
    }
    if (internalScheme.doNeedEy)
    {
      internalScheme.performFieldSteps<static_cast<uint8_t> (GridType::EY)> (t, EyStart, EyEnd);
    }
    if (internalScheme.doNeedEz)
    {
      internalScheme.performFieldSteps<static_cast<uint8_t> (GridType::EZ)> (t, EzStart, EzEnd);
    }

    if (useParallel && SOLVER_SETTINGS.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StopCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
                      "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
#endif
    }

    if (internalScheme.doNeedEx)
    {
      internalScheme.Ex->shiftInTime (zero, internalscheme.Ex->getSize ());
      internalScheme.Ex->nextTimeStep ();
    }
    if (internalScheme.doNeedEy)
    {
      internalScheme.Ey->shiftInTime (zero, internalscheme.Ey->getSize ());
      internalScheme.Ey->nextTimeStep ();
    }
    if (internalScheme.doNeedEz)
    {
      internalScheme.Ez->shiftInTime (zero, internalscheme.Ez->getSize ());
      internalScheme.Ez->nextTimeStep ();
    }

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      if (internalScheme.doNeedEx)
      {
        internalScheme.Dx->shiftInTime (zero, internalscheme.Dx->getSize ());
        internalScheme.Dx->nextTimeStep ();
      }
      if (internalScheme.doNeedEy)
      {
        internalScheme.Dy->shiftInTime (zero, internalscheme.Dy->getSize ());
        internalScheme.Dy->nextTimeStep ();
      }
      if (internalScheme.doNeedEz)
      {
        internalScheme.Dz->shiftInTime (zero, internalscheme.Dz->getSize ());
        internalScheme.Dz->nextTimeStep ();
      }
    }

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      if (internalScheme.doNeedEx)
      {
        internalScheme.D1x->shiftInTime (zero, internalscheme.D1x->getSize ());
        internalScheme.D1x->nextTimeStep ();
      }
      if (internalScheme.doNeedEy)
      {
        internalScheme.D1y->shiftInTime (zero, internalscheme.D1y->getSize ());
        internalScheme.D1y->nextTimeStep ();
      }
      if (internalScheme.doNeedEz)
      {
        internalScheme.D1z->shiftInTime (zero, internalscheme.D1z->getSize ());
        internalScheme.D1z->nextTimeStep ();
      }
    }

    if (SOLVER_SETTINGS.getDoUseTFSF ())
    {
      GridCoordinate1D zero1D (0
#ifdef DEBUG_INFO
                               , CoordinateType::X
#endif /* DEBUG_INFO */
                               );

      performPlaneWaveHSteps (t, zero1D, internalscheme.HInc->getSize ());

      internalScheme.HInc->shiftInTime (zero1D, internalscheme.HInc->getSize ());
      internalScheme.HInc->nextTimeStep ();
    }

    if (useParallel && SOLVER_SETTINGS.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StartCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
                      "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
#endif
    }

    if (internalScheme.doNeedHx)
    {
      internalScheme.performFieldSteps<static_cast<uint8_t> (GridType::HX)> (t, HxStart, HxEnd);
    }
    if (internalScheme.doNeedHy)
    {
      internalScheme.performFieldSteps<static_cast<uint8_t> (GridType::HY)> (t, HyStart, HyEnd);
    }
    if (internalScheme.doNeedHz)
    {
      internalScheme.performFieldSteps<static_cast<uint8_t> (GridType::HZ)> (t, HzStart, HzEnd);
    }

    if (useParallel && SOLVER_SETTINGS.getDoUseDynamicGrid ())
    {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
      ParallelGrid::getParallelCore ()->StopCalcClock ();
#else
      ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
                      "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
#endif
    }

    if (internalScheme.doNeedHx)
    {
      internalScheme.Hx->shiftInTime (zero, internalscheme.Hx->getSize ());
      internalScheme.Hx->nextTimeStep ();
    }
    if (internalScheme.doNeedHy)
    {
      internalScheme.Hy->shiftInTime (zero, internalscheme.Hy->getSize ());
      internalScheme.Hy->nextTimeStep ();
    }
    if (internalScheme.doNeedHz)
    {
      internalScheme.Hz->shiftInTime (zero, internalscheme.Hz->getSize ());
      internalScheme.Hz->nextTimeStep ();
    }

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      if (internalScheme.doNeedHx)
      {
        internalScheme.Bx->shiftInTime (zero, internalscheme.Bx->getSize ());
        internalScheme.Bx->nextTimeStep ();
      }
      if (internalScheme.doNeedHy)
      {
        internalScheme.By->shiftInTime (zero, internalscheme.By->getSize ());
        internalScheme.By->nextTimeStep ();
      }
      if (internalScheme.doNeedHz)
      {
        internalScheme.Bz->shiftInTime (zero, internalscheme.Bz->getSize ());
        internalScheme.Bz->nextTimeStep ();
      }
    }

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      if (internalScheme.doNeedHx)
      {
        internalScheme.B1x->shiftInTime (zero, internalscheme.B1x->getSize ());
        internalScheme.B1x->nextTimeStep ();
      }
      if (internalScheme.doNeedHy)
      {
        internalScheme.B1y->shiftInTime (zero, internalscheme.B1y->getSize ());
        internalScheme.B1y->nextTimeStep ();
      }
      if (internalScheme.doNeedHz)
      {
        internalScheme.B1z->shiftInTime (zero, internalscheme.B1z->getSize ());
        internalScheme.B1z->nextTimeStep ();
      }
    }

    if (SOLVER_SETTINGS.getDoSaveIntermediateRes ()
        && t % SOLVER_SETTINGS.getIntermediateSaveStep () == 0)
    {
      gatherFieldsTotal (SOLVER_SETTINGS.getDoSaveScatteredFieldIntermediate ());
      saveGrids (t);
    }

    if (SOLVER_SETTINGS.getDoUseNTFF ()
        && t > 0 && t % SOLVER_SETTINGS.getIntermediateNTFFStep () == 0)
    {
      saveNTFF (SOLVER_SETTINGS.getDoCalcReverseNTFF (), t);
    }

    additionalUpdateOfGrids (t, diffT);
  }

  if (SOLVER_SETTINGS.getDoSaveRes ())
  {
    gatherFieldsTotal (SOLVER_SETTINGS.getDoSaveScatteredFieldRes ());
    saveGrids (stepLimit);
  }
}
//
// template <SchemeType_t Type, template <typename, bool> class TCoord, typename Layout>
// void
// Scheme<Type, TCoord, Layout>::performAmplitudeSteps (time_step startStep)
// {
// #ifdef COMPLEX_FIELD_VALUES
//   UNREACHABLE;
// #else /* COMPLEX_FIELD_VALUES */
//
//   ASSERT_MESSAGE ("Temporary unsupported");
//
//   int processId = 0;
//
//   if (SOLVER_SETTINGS.getDoUseParallelGrid ())
//   {
// #ifdef PARALLEL_GRID
//     processId = ParallelGrid::getParallelCore ()->getProcessId ();
// #else /* PARALLEL_GRID */
//     ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
// #endif /* !PARALLEL_GRID */
//   }
//
//   int is_stable_state = 0;
//
//   GridCoordinate3D EzSize = internalScheme.Ez->getSize ();
//
//   time_step t = startStep;
//
//   while (is_stable_state == 0 && t < SOLVER_SETTINGS.getNumAmplitudeSteps ())
//   {
//     FPValue maxAccuracy = -1;
//
//     //is_stable_state = 1;
//
//     GridCoordinate3D ExStart = internalScheme.Ex->getComputationStart (yeeLayout->getExStartDiff ());
//     GridCoordinate3D ExEnd = internalScheme.Ex->getComputationEnd (yeeLayout->getExEndDiff ());
//
//     GridCoordinate3D EyStart = internalScheme.Ey->getComputationStart (yeeLayout->getEyStartDiff ());
//     GridCoordinate3D EyEnd = internalScheme.Ey->getComputationEnd (yeeLayout->getEyEndDiff ());
//
//     GridCoordinate3D EzStart = internalScheme.Ez->getComputationStart (yeeLayout->getEzStartDiff ());
//     GridCoordinate3D EzEnd = internalScheme.Ez->getComputationEnd (yeeLayout->getEzEndDiff ());
//
//     GridCoordinate3D HxStart = internalScheme.Hx->getComputationStart (yeeLayout->getHxStartDiff ());
//     GridCoordinate3D HxEnd = internalScheme.Hx->getComputationEnd (yeeLayout->getHxEndDiff ());
//
//     GridCoordinate3D HyStart = internalScheme.Hy->getComputationStart (yeeLayout->getHyStartDiff ());
//     GridCoordinate3D HyEnd = internalScheme.Hy->getComputationEnd (yeeLayout->getHyEndDiff ());
//
//     GridCoordinate3D HzStart = internalScheme.Hz->getComputationStart (yeeLayout->getHzStartDiff ());
//     GridCoordinate3D HzEnd = internalScheme.Hz->getComputationEnd (yeeLayout->getHzEndDiff ());
//
//     if (SOLVER_SETTINGS.getDoUseTFSF ())
//     {
//       performPlaneWaveESteps (t);
//     }
//
//     performFieldSteps<static_cast<uint8_t> (GridType::EX)> (t, ExStart, ExEnd);
//     performFieldSteps<static_cast<uint8_t> (GridType::EY)> (t, EyStart, EyEnd);
//     performFieldSteps<static_cast<uint8_t> (GridType::EZ)> (t, EzStart, EzEnd);
//
//     for (int i = ExStart.get1 (); i < ExEnd.get1 (); ++i)
//     {
//       for (int j = ExStart.get2 (); j < ExEnd.get2 (); ++j)
//       {
//         for (int k = ExStart.get3 (); k < ExEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isExInPML (internalScheme.Ex->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = internalScheme.Ex->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = internalScheme.ExAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getExCoordFP (internalScheme.Ex->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     for (int i = EyStart.get1 (); i < EyEnd.get1 (); ++i)
//     {
//       for (int j = EyStart.get2 (); j < EyEnd.get2 (); ++j)
//       {
//         for (int k = EyStart.get3 (); k < EyEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isEyInPML (internalScheme.Ey->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = internalScheme.Ey->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = internalScheme.EyAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getEyCoordFP (internalScheme.Ey->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     for (int i = EzStart.get1 (); i < EzEnd.get1 (); ++i)
//     {
//       for (int j = EzStart.get2 (); j < EzEnd.get2 (); ++j)
//       {
//         for (int k = EzStart.get3 (); k < EzEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isEzInPML (internalScheme.Ez->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = internalScheme.Ez->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = internalScheme.EzAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getEzCoordFP (internalScheme.Ez->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     internalScheme.Ex->nextTimeStep ();
//     internalScheme.Ey->nextTimeStep ();
//     internalScheme.Ez->nextTimeStep ();
//
//     if (SOLVER_SETTINGS.getDoUsePML ())
//     {
//       internalScheme.Dx->nextTimeStep ();
//       internalScheme.Dy->nextTimeStep ();
//       internalScheme.Dz->nextTimeStep ();
//     }
//
//     if (SOLVER_SETTINGS.getDoUseTFSF ())
//     {
//       performPlaneWaveHSteps (t);
//     }
//
//     performFieldSteps<static_cast<uint8_t> (GridType::HX)> (t, HxStart, HxEnd);
//     performFieldSteps<static_cast<uint8_t> (GridType::HY)> (t, HyStart, HyEnd);
//     performFieldSteps<static_cast<uint8_t> (GridType::HZ)> (t, HzStart, HzEnd);
//
//     for (int i = HxStart.get1 (); i < HxEnd.get1 (); ++i)
//     {
//       for (int j = HxStart.get2 (); j < HxEnd.get2 (); ++j)
//       {
//         for (int k = HxStart.get3 (); k < HxEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isHxInPML (internalScheme.Hx->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = internalScheme.Hx->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = internalScheme.HxAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getHxCoordFP (internalScheme.Hx->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     for (int i = HyStart.get1 (); i < HyEnd.get1 (); ++i)
//     {
//       for (int j = HyStart.get2 (); j < HyEnd.get2 (); ++j)
//       {
//         for (int k = HyStart.get3 (); k < HyEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isHyInPML (internalScheme.Hy->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = internalScheme.Hy->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = internalScheme.HyAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getHyCoordFP (internalScheme.Hy->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     for (int i = HzStart.get1 (); i < HzEnd.get1 (); ++i)
//     {
//       for (int j = HzStart.get2 (); j < HzEnd.get2 (); ++j)
//       {
//         for (int k = HzStart.get3 (); k < HzEnd.get3 (); ++k)
//         {
//           GridCoordinate3D pos (i, j, k);
//
//           if (!yeeLayout->isHzInPML (internalScheme.Hz->getTotalPosition (pos)))
//           {
//             FieldPointValue* tmp = internalScheme.Hz->getFieldPointValue (pos);
//             FieldPointValue* tmpAmp = internalScheme.HzAmplitude->getFieldPointValue (pos);
//
//             GridCoordinateFP3D realCoord = yeeLayout->getHzCoordFP (internalScheme.Hz->getTotalPosition (pos));
//
//             GridCoordinateFP3D leftBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getLeftBorderTFSF ());
//             GridCoordinateFP3D rightBorder = GridCoordinateFP3D (0, 0, 0) + convertCoord (yeeLayout->getRightBorderTFSF ());
//
//             FPValue val = tmp->getCurValue ();
//
//             if (updateAmplitude (val, tmpAmp, &maxAccuracy) == 0)
//             {
//               is_stable_state = 0;
//             }
//           }
//         }
//       }
//     }
//
//     internalScheme.Hx->nextTimeStep ();
//     internalScheme.Hy->nextTimeStep ();
//     internalScheme.Hz->nextTimeStep ();
//
//     if (SOLVER_SETTINGS.getDoUsePML ())
//     {
//       internalScheme.Bx->nextTimeStep ();
//       internalScheme.By->nextTimeStep ();
//       internalScheme.Bz->nextTimeStep ();
//     }
//
//     ++t;
//
//     if (maxAccuracy < 0)
//     {
//       is_stable_state = 0;
//     }
//
//     DPRINTF (LOG_LEVEL_STAGES, "%d amplitude calculation step: max accuracy " FP_MOD ". \n", t, maxAccuracy);
//   }
//
//   if (is_stable_state == 0)
//   {
//     ASSERT_MESSAGE ("Stable state is not reached. Increase number of steps.");
//   }
//
// #endif /* !COMPLEX_FIELD_VALUES */
// }

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
int
Scheme<Type, TCoord, layout_type>::updateAmplitude (FPValue val, FieldPointValue *amplitudeValue, FPValue *maxAccuracy)
{
#ifdef COMPLEX_FIELD_VALUES
  UNREACHABLE;
#else /* COMPLEX_FIELD_VALUES */

  int is_stable_state = 1;

  FPValue valAmp = amplitudeValue->getCurValue ();

  val = val >= 0 ? val : -val;

  if (val >= valAmp)
  {
    FPValue accuracy = val - valAmp;
    if (valAmp != 0)
    {
      accuracy /= valAmp;
    }
    else if (val != 0)
    {
      accuracy /= val;
    }

    if (accuracy > PhysicsConst::accuracy)
    {
      is_stable_state = 0;

      amplitudeValue->setCurValue (val);
    }

    if (accuracy > *maxAccuracy)
    {
      *maxAccuracy = accuracy;
    }
  }

  return is_stable_state;
#endif /* !COMPLEX_FIELD_VALUES */
}

#ifdef CUDA_ENABLED
template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::performCudaSteps ()
{
  ALWAYS_ASSERT (0);
}

template <>
void
Scheme<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>::performCudaSteps ()
{
  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  if (SOLVER_SETTINGS.getDoUsePML ()
      || SOLVER_SETTINGS.getDoUseTFSF ()
      || SOLVER_SETTINGS.getDoUseAmplitudeMode ()
      || SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    ASSERT_MESSAGE ("Cuda GPU calculations with these parameters are not implemented");
  }

  CudaExitStatus status;

  cudaExecute3DSteps (useParallel, &status, yeeLayout, gridTimeStep, gridStep, internalScheme.Ex, internalScheme.Ey, internalScheme.Ez, internalScheme.Hx, internalScheme.Hy, internalScheme.Hz, internalScheme.Eps, internalScheme.Mu, totalStep, processId);

  ASSERT (status == CUDA_OK);

  if (SOLVER_SETTINGS.getDoSaveRes ())
  {
    gatherFieldsTotal (SOLVER_SETTINGS.getDoSaveScatteredFieldRes ());
    saveGrids (totalStep);
  }
}
#endif

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::performSteps ()
{
#if defined (CUDA_ENABLED)

  performCudaSteps ();

#else /* CUDA_ENABLED */

  if (SOLVER_SETTINGS.getDoUseMetamaterials () && !SOLVER_SETTINGS.getDoUsePML ())
  {
    ASSERT_MESSAGE ("Metamaterials without pml are not implemented");
  }

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
    {
      ASSERT_MESSAGE ("Parallel amplitude mode is not implemented");
    }
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  performNSteps (0, totalStep);

  if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
  {
    UNREACHABLE;
    //performAmplitudeSteps (totalStep);
  }

#endif /* !CUDA_ENABLED */
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initScheme (FPValue dx, FPValue sourceWaveLen)
{
  sourceWaveLength = sourceWaveLen;
  sourceFrequency = PhysicsConst::SpeedOfLight / sourceWaveLength;

  gridStep = dx;
  courantNum = SOLVER_SETTINGS.getCourantNum ();
  gridTimeStep = gridStep * courantNum / PhysicsConst::SpeedOfLight;

  FPValue N_lambda = sourceWaveLength / gridStep;
  ALWAYS_ASSERT (SQR (round (N_lambda) - N_lambda) < Approximation::getAccuracy ());

  FPValue phaseVelocity0 = Approximation::phaseVelocityIncidentWave (gridStep, sourceWaveLength, courantNum, N_lambda, PhysicsConst::Pi / 2, 0);
  FPValue phaseVelocity = Approximation::phaseVelocityIncidentWave (gridStep, sourceWaveLength, courantNum, N_lambda, yeeLayout->getIncidentWaveAngle1 (), yeeLayout->getIncidentWaveAngle2 ());
  FPValue k = 2 * PhysicsConst::Pi * PhysicsConst::SpeedOfLight / sourceWaveLength / phaseVelocity0;

  relPhaseVelocity = phaseVelocity0 / phaseVelocity;
  sourceWaveLengthNumerical = 2 * PhysicsConst::Pi / k;

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "initScheme: "
                                      "\n\tphase velocity relation -> %f "
                                      "\n\tphase velosity 0 -> %f "
                                      "\n\tphase velocity -> %f "
                                      "\n\tanalytical wave number -> %.20f "
                                      "\n\tnumerical wave number -> %.20f"
                                      "\n\tanalytical wave length -> %.20f"
                                      "\n\tnumerical wave length -> %.20f"
                                      "\n\tnumerical grid step -> %.20f"
                                      "\n\tnumerical time step -> %.20f"
                                      "\n\twave length -> %.20f"
                                      "\n",
           relPhaseVelocity, phaseVelocity0, phaseVelocity, 2*PhysicsConst::Pi/sourceWaveLength, k,
           sourceWaveLength, sourceWaveLengthNumerical, gridStep, gridTimeStep, sourceFrequency);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initCallBacks ()
{
#ifndef COMPLEX_FIELD_VALUES
  if (SOLVER_SETTINGS.getDoUsePolinom1BorderCondition ())
  {
    EzBorder = CallBack::polinom1_ez;
    HyBorder = CallBack::polinom1_hy;
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom2BorderCondition ())
  {
    ExBorder = CallBack::polinom2_ex;
    EyBorder = CallBack::polinom2_ey;
    EzBorder = CallBack::polinom2_ez;

    HxBorder = CallBack::polinom2_hx;
    HyBorder = CallBack::polinom2_hy;
    HzBorder = CallBack::polinom2_hz;
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom3BorderCondition ())
  {
    EzBorder = CallBack::polinom3_ez;
    HyBorder = CallBack::polinom3_hy;
  }
  else if (SOLVER_SETTINGS.getDoUseSin1BorderCondition ())
  {
    EzBorder = CallBack::sin1_ez;
    HyBorder = CallBack::sin1_hy;
  }

  if (SOLVER_SETTINGS.getDoUsePolinom1StartValues ())
  {
    EzInitial = CallBack::polinom1_ez;
    HyInitial = CallBack::polinom1_hy;
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom2StartValues ())
  {
    ExInitial = CallBack::polinom2_ex;
    EyInitial = CallBack::polinom2_ey;
    EzInitial = CallBack::polinom2_ez;

    HxInitial = CallBack::polinom2_hx;
    HyInitial = CallBack::polinom2_hy;
    HzInitial = CallBack::polinom2_hz;
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom3StartValues ())
  {
    EzInitial = CallBack::polinom3_ez;
    HyInitial = CallBack::polinom3_hy;
  }
  else if (SOLVER_SETTINGS.getDoUseSin1StartValues ())
  {
    EzInitial = CallBack::sin1_ez;
    HyInitial = CallBack::sin1_hy;
  }

  if (SOLVER_SETTINGS.getDoUsePolinom1RightSide ())
  {
    Jz = CallBack::polinom1_jz;
    My = CallBack::polinom1_my;
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom2RightSide ())
  {
    Jx = CallBack::polinom2_jx;
    Jy = CallBack::polinom2_jy;
    Jz = CallBack::polinom2_jz;

    Mx = CallBack::polinom2_mx;
    My = CallBack::polinom2_my;
    Mz = CallBack::polinom2_mz;
  }
  else if (SOLVER_SETTINGS.getDoUsePolinom3RightSide ())
  {
    Jz = CallBack::polinom3_jz;
    My = CallBack::polinom3_my;
  }

  if (SOLVER_SETTINGS.getDoCalculatePolinom1DiffNorm ())
  {
    EzExact = CallBack::polinom1_ez;
    HyExact = CallBack::polinom1_hy;
  }
  else if (SOLVER_SETTINGS.getDoCalculatePolinom2DiffNorm ())
  {
    ExExact = CallBack::polinom2_ex;
    EyExact = CallBack::polinom2_ey;
    EzExact = CallBack::polinom2_ez;

    HxExact = CallBack::polinom2_hx;
    HyExact = CallBack::polinom2_hy;
    HzExact = CallBack::polinom2_hz;
  }
  else if (SOLVER_SETTINGS.getDoCalculatePolinom3DiffNorm ())
  {
    EzExact = CallBack::polinom3_ez;
    HyExact = CallBack::polinom3_hy;
  }
  else if (SOLVER_SETTINGS.getDoCalculateSin1DiffNorm ())
  {
    EzExact = CallBack::sin1_ez;
    HyExact = CallBack::sin1_hy;
  }
#endif

  if (SOLVER_SETTINGS.getDoCalculateExp1ExHyDiffNorm ())
  {
    ExExact = CallBack::exp1_ex_exhy;
    HyExact = CallBack::exp1_hy_exhy;
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2ExHyDiffNorm ())
  {
    ExExact = CallBack::exp2_ex_exhy;
    HyExact = CallBack::exp2_hy_exhy;
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3ExHyDiffNorm ())
  {
    ExExact = CallBack::exp3_ex_exhy;
    HyExact = CallBack::exp3_hy_exhy;
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1ExHzDiffNorm ())
  {
    ExExact = CallBack::exp1_ex_exhz;
    HzExact = CallBack::exp1_hz_exhz;
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2ExHzDiffNorm ())
  {
    ExExact = CallBack::exp2_ex_exhz;
    HzExact = CallBack::exp2_hz_exhz;
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3ExHzDiffNorm ())
  {
    ExExact = CallBack::exp3_ex_exhz;
    HzExact = CallBack::exp3_hz_exhz;
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1EyHxDiffNorm ())
  {
    EyExact = CallBack::exp1_ey_eyhx;
    HxExact = CallBack::exp1_hx_eyhx;
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2EyHxDiffNorm ())
  {
    EyExact = CallBack::exp2_ey_eyhx;
    HxExact = CallBack::exp2_hx_eyhx;
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3EyHxDiffNorm ())
  {
    EyExact = CallBack::exp3_ey_eyhx;
    HxExact = CallBack::exp3_hx_eyhx;
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1EyHzDiffNorm ())
  {
    EyExact = CallBack::exp1_ey_eyhz;
    HzExact = CallBack::exp1_hz_eyhz;
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2EyHzDiffNorm ())
  {
    EyExact = CallBack::exp2_ey_eyhz;
    HzExact = CallBack::exp2_hz_eyhz;
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3EyHzDiffNorm ())
  {
    EyExact = CallBack::exp3_ey_eyhz;
    HzExact = CallBack::exp3_hz_eyhz;
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1EzHxDiffNorm ())
  {
    EzExact = CallBack::exp1_ez_ezhx;
    HxExact = CallBack::exp1_hx_ezhx;
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2EzHxDiffNorm ())
  {
    EzExact = CallBack::exp2_ez_ezhx;
    HxExact = CallBack::exp2_hx_ezhx;
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3EzHxDiffNorm ())
  {
    EzExact = CallBack::exp3_ez_ezhx;
    HxExact = CallBack::exp3_hx_ezhx;
  }

  if (SOLVER_SETTINGS.getDoCalculateExp1EzHyDiffNorm ())
  {
    EzExact = CallBack::exp1_ez_ezhy;
    HyExact = CallBack::exp1_hy_ezhy;
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp2EzHyDiffNorm ())
  {
    EzExact = CallBack::exp2_ez_ezhy;
    HyExact = CallBack::exp2_hy_ezhy;
  }
  else if (SOLVER_SETTINGS.getDoCalculateExp3EzHyDiffNorm ())
  {
    EzExact = CallBack::exp3_ez_ezhy;
    HyExact = CallBack::exp3_hy_ezhy;
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initMaterialFromFile (GridType gridType, Grid<TC> *grid, Grid<TC> *totalGrid)
{
  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  std::string filename;

  switch (gridType)
  {
    case GridType::EPS:
    {
      filename = SOLVER_SETTINGS.getEpsFileName ();
      break;
    }
    case GridType::MU:
    {
      filename = SOLVER_SETTINGS.getMuFileName ();
      break;
    }
    case GridType::OMEGAPE:
    {
      filename = SOLVER_SETTINGS.getOmegaPEFileName ();
      break;
    }
    case GridType::OMEGAPM:
    {
      filename = SOLVER_SETTINGS.getOmegaPMFileName ();
      break;
    }
    case GridType::GAMMAE:
    {
      filename = SOLVER_SETTINGS.getGammaEFileName ();
      break;
    }
    case GridType::GAMMAM:
    {
      filename = SOLVER_SETTINGS.getGammaMFileName ();
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  if (filename.empty ())
  {
    return;
  }

  FileType type = GridFileManager::getFileType (filename);
  loader[type]->initManual (0, CURRENT, processId, filename, "", "");
  loader[type]->loadGrid (totalGrid);

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    for (grid_coord i = 0; i < grid->getSize ().calculateTotalCoord (); ++i)
    {
      TC pos = grid->calculatePositionFromIndex (i);
      TC posAbs = grid->getTotalPosition (pos);

      FieldPointValue *val = grid->getFieldPointValue (pos);
      *val = *totalGrid->getFieldPointValue (posAbs);
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initGridWithInitialVals (GridType gridType, Grid<TC> *grid, FPValue timestep)
{
  SourceCallBack cb = NULLPTR;

  switch (gridType)
  {
    case GridType::EX:
    {
      cb = ExInitial;
      break;
    }
    case GridType::EY:
    {
      cb = EyInitial;
      break;
    }
    case GridType::EZ:
    {
      cb = EzInitial;
      break;
    }
    case GridType::HX:
    {
      cb = HxInitial;
      break;
    }
    case GridType::HY:
    {
      cb = HyInitial;
      break;
    }
    case GridType::HZ:
    {
      cb = HzInitial;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  if (cb == NULLPTR)
  {
    return;
  }

  for (grid_coord i = 0; i < grid->getSize ().calculateTotalCoord (); ++i)
  {
    TC pos = grid->calculatePositionFromIndex (i);
    TC posAbs = grid->getTotalPosition (pos);
    TCFP realCoord;

    switch (gridType)
    {
      case GridType::EX:
      {
        realCoord = yeeLayout->getExCoordFP (posAbs);
        break;
      }
      case GridType::EY:
      {
        realCoord = yeeLayout->getEyCoordFP (posAbs);
        break;
      }
      case GridType::EZ:
      {
        realCoord = yeeLayout->getEzCoordFP (posAbs);
        break;
      }
      case GridType::HX:
      {
        realCoord = yeeLayout->getHxCoordFP (posAbs);
        break;
      }
      case GridType::HY:
      {
        realCoord = yeeLayout->getHyCoordFP (posAbs);
        break;
      }
      case GridType::HZ:
      {
        realCoord = yeeLayout->getHzCoordFP (posAbs);
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    grid->getFieldPointValue (pos)->setCurValue (cb (expandTo3D (realCoord * gridStep, ct1, ct2, ct3), timestep));
  }
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids1D (internalScheme.Eps, totalEps,
                                         internalScheme.Mu, totalMu,
                                         internalScheme.OmegaPE, totalOmegaPE,
                                         internalScheme.OmegaPM, totalOmegaPM,
                                         internalScheme.GammaE, totalGammaE,
                                         internalScheme.GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids1D (internalScheme.Eps, totalEps,
                                         internalScheme.Mu, totalMu,
                                         internalScheme.OmegaPE, totalOmegaPE,
                                         internalScheme.OmegaPM, totalOmegaPM,
                                         internalScheme.GammaE, totalGammaE,
                                         internalScheme.GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids1D (internalScheme.Eps, totalEps,
                                         internalScheme.Mu, totalMu,
                                         internalScheme.OmegaPE, totalOmegaPE,
                                         internalScheme.OmegaPM, totalOmegaPM,
                                         internalScheme.GammaE, totalGammaE,
                                         internalScheme.GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids1D (internalScheme.Eps, totalEps,
                                         internalScheme.Mu, totalMu,
                                         internalScheme.OmegaPE, totalOmegaPE,
                                         internalScheme.OmegaPM, totalOmegaPM,
                                         internalScheme.GammaE, totalGammaE,
                                         internalScheme.GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids1D (internalScheme.Eps, totalEps,
                                         internalScheme.Mu, totalMu,
                                         internalScheme.OmegaPE, totalOmegaPE,
                                         internalScheme.OmegaPM, totalOmegaPM,
                                         internalScheme.GammaE, totalGammaE,
                                         internalScheme.GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids1D (internalScheme.Eps, totalEps,
                                         internalScheme.Mu, totalMu,
                                         internalScheme.OmegaPE, totalOmegaPE,
                                         internalScheme.OmegaPM, totalOmegaPM,
                                         internalScheme.GammaE, totalGammaE,
                                         internalScheme.GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids2D (internalScheme.Eps, totalEps,
                                         internalScheme.Mu, totalMu,
                                         internalScheme.OmegaPE, totalOmegaPE,
                                         internalScheme.OmegaPM, totalOmegaPM,
                                         internalScheme.GammaE, totalGammaE,
                                         internalScheme.GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids2D (internalScheme.Eps, totalEps,
                                         internalScheme.Mu, totalMu,
                                         internalScheme.OmegaPE, totalOmegaPE,
                                         internalScheme.OmegaPM, totalOmegaPM,
                                         internalScheme.GammaE, totalGammaE,
                                         internalScheme.GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids2D (internalScheme.Eps, totalEps,
                                         internalScheme.Mu, totalMu,
                                         internalScheme.OmegaPE, totalOmegaPE,
                                         internalScheme.OmegaPM, totalOmegaPM,
                                         internalScheme.GammaE, totalGammaE,
                                         internalScheme.GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids2D (internalScheme.Eps, totalEps,
                                         internalScheme.Mu, totalMu,
                                         internalScheme.OmegaPE, totalOmegaPE,
                                         internalScheme.OmegaPM, totalOmegaPM,
                                         internalScheme.GammaE, totalGammaE,
                                         internalScheme.GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids2D (internalScheme.Eps, totalEps,
                                         internalScheme.Mu, totalMu,
                                         internalScheme.OmegaPE, totalOmegaPE,
                                         internalScheme.OmegaPM, totalOmegaPM,
                                         internalScheme.GammaE, totalGammaE,
                                         internalScheme.GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids2D (internalScheme.Eps, totalEps,
                                         internalScheme.Mu, totalMu,
                                         internalScheme.OmegaPE, totalOmegaPE,
                                         internalScheme.OmegaPM, totalOmegaPM,
                                         internalScheme.GammaE, totalGammaE,
                                         internalScheme.GammaM, totalGammaM);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>::initFullMaterialGrids ()
{
  SchemeHelper::initFullMaterialGrids3D (internalScheme.Eps, totalEps,
                                         internalScheme.Mu, totalMu,
                                         internalScheme.OmegaPE, totalOmegaPE,
                                         internalScheme.OmegaPM, totalOmegaPM,
                                         internalScheme.GammaE, totalGammaE,
                                         internalScheme.GammaM, totalGammaM);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::initGrids ()
{
  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else /* PARALLEL_GRID */
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif /* !PARALLEL_GRID */
  }

  internalScheme.Eps->initialize (getFieldValueRealOnly (1.0));
  initMaterialFromFile (GridType::EPS, internalScheme.Eps, totalEps);

  if (SOLVER_SETTINGS.getEpsSphere () != 1)
  {
    for (grid_coord i = 0; i < internalScheme.Eps->getSize ().calculateTotalCoord (); ++i)
    {
      TC pos = internalScheme.Eps->calculatePositionFromIndex (i);
      TCFP posAbs = yeeLayout->getEpsCoordFP (internalScheme.Eps->getTotalPosition (pos));
      FieldPointValue *val = internalScheme.Eps->getFieldPointValue (pos);

      FieldValue epsVal = getFieldValueRealOnly (SOLVER_SETTINGS.getEpsSphere ());

      FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

      TCFP center = TCFP::initAxesCoordinate(SOLVER_SETTINGS.getEpsSphereCenterX (),
                                             SOLVER_SETTINGS.getEpsSphereCenterY (),
                                             SOLVER_SETTINGS.getEpsSphereCenterZ (),
                                             ct1, ct2, ct3);
      val->setCurValue (Approximation::approximateSphereAccurate (posAbs,
                                                                  center * modifier + TCFP (0.5, 0.5, 0.5
#ifdef DEBUG_INFO
                                                                                                        , ct1, ct2, ct3
#endif
                                                                                                        ),
                                                                  SOLVER_SETTINGS.getEpsSphereRadius () * modifier,
                                                                  epsVal,
                                                                  getFieldValueRealOnly (1.0)));
    }
  }
  if (SOLVER_SETTINGS.getUseEpsAllNorm ())
  {
    for (grid_coord i = 0; i < internalScheme.Eps->getSize ().calculateTotalCoord (); ++i)
    {
      FieldPointValue *val = internalScheme.Eps->getFieldPointValue (i);
      val->setCurValue (getFieldValueRealOnly (FPValue(1.0) / PhysicsConst::Eps0));
    }
  }

  internalScheme.Mu->initialize (getFieldValueRealOnly (1.0));
  initMaterialFromFile (GridType::MU, internalScheme.Mu, totalMu);

  if (SOLVER_SETTINGS.getMuSphere () != 1)
  {
    for (grid_coord i = 0; i < internalScheme.Mu->getSize ().calculateTotalCoord (); ++i)
    {
      TC pos = internalScheme.Mu->calculatePositionFromIndex (i);
      TCFP posAbs = yeeLayout->getMuCoordFP (internalScheme.Mu->getTotalPosition (pos));
      FieldPointValue *val = internalScheme.Mu->getFieldPointValue (pos);

      FieldValue muVal = getFieldValueRealOnly (SOLVER_SETTINGS.getMuSphere ());

      FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

      TCFP center = TCFP::initAxesCoordinate(SOLVER_SETTINGS.getMuSphereCenterX (),
                                             SOLVER_SETTINGS.getMuSphereCenterY (),
                                             SOLVER_SETTINGS.getMuSphereCenterZ (),
                                             ct1, ct2, ct3);
      val->setCurValue (Approximation::approximateSphereAccurate (posAbs,
                                                                  center * modifier + TCFP (0.5, 0.5, 0.5
  #ifdef DEBUG_INFO
                                                                                                        , ct1, ct2, ct3
  #endif
                                                                                                        ),
                                                                  SOLVER_SETTINGS.getMuSphereRadius () * modifier,
                                                                  muVal,
                                                                  getFieldValueRealOnly (1.0)));
    }
  }
  if (SOLVER_SETTINGS.getUseMuAllNorm ())
  {
    for (grid_coord i = 0; i < internalScheme.Mu->getSize ().calculateTotalCoord (); ++i)
    {
      FieldPointValue *val = internalScheme.Mu->getFieldPointValue (i);
      val->setCurValue (getFieldValueRealOnly (FPValue(1.0) / PhysicsConst::Mu0));
    }
  }

  if (SOLVER_SETTINGS.getDoUseMetamaterials ())
  {
    initMaterialFromFile (GridType::OMEGAPE, internalScheme.OmegaPE, totalOmegaPE);

    if (SOLVER_SETTINGS.getOmegaPESphere () != 0)
    {
      for (grid_coord i = 0; i < internalScheme.OmegaPE->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = internalScheme.OmegaPE->calculatePositionFromIndex (i);
        TCFP posAbs = yeeLayout->getEpsCoordFP (internalScheme.OmegaPE->getTotalPosition (pos));
        FieldPointValue *val = internalScheme.OmegaPE->getFieldPointValue (pos);

        FieldValue omegapeVal = getFieldValueRealOnly (SOLVER_SETTINGS.getOmegaPESphere () * 2 * PhysicsConst::Pi * sourceFrequency);

        FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

        TCFP center = TCFP::initAxesCoordinate (SOLVER_SETTINGS.getOmegaPESphereCenterX (),
                                                SOLVER_SETTINGS.getOmegaPESphereCenterY (),
                                                SOLVER_SETTINGS.getOmegaPESphereCenterZ (),
                                                ct1, ct2, ct3);
        val->setCurValue (Approximation::approximateSphereAccurate (posAbs,
                                                                    center * modifier + TCFP (0.5, 0.5, 0.5
#ifdef DEBUG_INFO
                                                                                                          , ct1, ct2, ct3
#endif
                                                                                                          ),
                                                                    SOLVER_SETTINGS.getOmegaPESphereRadius () * modifier,
                                                                    omegapeVal,
                                                                    getFieldValueRealOnly (0.0)));
      }
    }

    initMaterialFromFile (GridType::OMEGAPM, internalScheme.OmegaPM, totalOmegaPM);

    if (SOLVER_SETTINGS.getOmegaPMSphere () != 0)
    {
      for (grid_coord i = 0; i < internalScheme.OmegaPM->getSize ().calculateTotalCoord (); ++i)
      {
        TC pos = internalScheme.OmegaPM->calculatePositionFromIndex (i);
        TCFP posAbs = yeeLayout->getEpsCoordFP (internalScheme.OmegaPM->getTotalPosition (pos));
        FieldPointValue *val = internalScheme.OmegaPM->getFieldPointValue (pos);

        FieldValue omegapmVal = getFieldValueRealOnly (SOLVER_SETTINGS.getOmegaPMSphere () * 2 * PhysicsConst::Pi * sourceFrequency);

        FPValue modifier = (yeeLayout->getIsDoubleMaterialPrecision () ? 2 : 1);

        TCFP center = TCFP::initAxesCoordinate (SOLVER_SETTINGS.getOmegaPMSphereCenterX (),
                                                SOLVER_SETTINGS.getOmegaPMSphereCenterY (),
                                                SOLVER_SETTINGS.getOmegaPMSphereCenterZ (),
                                                ct1, ct2, ct3);
        val->setCurValue (Approximation::approximateSphereAccurate (posAbs,
                                                                    center * modifier + TCFP (0.5, 0.5, 0.5
#ifdef DEBUG_INFO
                                                                                                          , ct1, ct2, ct3
#endif
                                                                                                          ),
                                                                    SOLVER_SETTINGS.getOmegaPMSphereRadius () * modifier,
                                                                    omegapmVal,
                                                                    getFieldValueRealOnly (0.0)));
      }
    }

    initMaterialFromFile (GridType::GAMMAE, internalScheme.GammaE, totalGammaE);

    initMaterialFromFile (GridType::GAMMAM, internalScheme.GammaM, totalGammaM);
  }

  if (SOLVER_SETTINGS.getDoUsePML ())
  {
    initSigmas ();
  }

  for (int type = FILE_TYPE_BMP; type < FILE_TYPE_COUNT; ++type)
  {
    if (!dumper[type])
    {
      continue;
    }

    if (SOLVER_SETTINGS.getDoSaveMaterials ())
    {
      if (useParallel)
      {
        initFullMaterialGrids ();
      }

      if (processId == 0)
      {
        TC startEps, startMu, startOmegaPE, startOmegaPM, startGammaE, startGammaM;
        TC endEps, endMu, endOmegaPE, endOmegaPM, endGammaE, endGammaM;

        if (SOLVER_SETTINGS.getDoUseManualStartEndDumpCoord ())
        {
          TC start = TC::initAxesCoordinate (SOLVER_SETTINGS.getSaveStartCoordX (),
                                            SOLVER_SETTINGS.getSaveStartCoordY (),
                                            SOLVER_SETTINGS.getSaveStartCoordZ (),
                                            ct1, ct2, ct3);
          TC end = TC::initAxesCoordinate (SOLVER_SETTINGS.getSaveEndCoordX (),
                                          SOLVER_SETTINGS.getSaveEndCoordY (),
                                          SOLVER_SETTINGS.getSaveEndCoordZ (),
                                          ct1, ct2, ct3);
          startEps = startMu = startOmegaPE = startOmegaPM = startGammaE = startGammaM = start;
          endEps = endMu = endOmegaPE = endOmegaPM = endGammaE = endGammaM = end;
        }
        else
        {
          startEps = getStartCoord (GridType::EPS, totalEps->getSize ());
          endEps = getEndCoord (GridType::EPS, totalEps->getSize ());

          startMu = getStartCoord (GridType::MU, totalMu->getSize ());
          endMu = getEndCoord (GridType::MU, totalMu->getSize ());

          if (SOLVER_SETTINGS.getDoUseMetamaterials ())
          {
            startOmegaPE = getStartCoord (GridType::OMEGAPE, totalOmegaPE->getSize ());
            endOmegaPE = getEndCoord (GridType::OMEGAPE, totalOmegaPE->getSize ());

            startOmegaPM = getStartCoord (GridType::OMEGAPM, totalOmegaPM->getSize ());
            endOmegaPM = getEndCoord (GridType::OMEGAPM, totalOmegaPM->getSize ());

            startGammaE = getStartCoord (GridType::GAMMAE, totalGammaE->getSize ());
            endGammaE = getEndCoord (GridType::GAMMAE, totalGammaE->getSize ());

            startGammaM = getStartCoord (GridType::GAMMAM, totalGammaM->getSize ());
            endGammaM = getEndCoord (GridType::GAMMAM, totalGammaM->getSize ());
          }
        }

        dumper[type]->init (0, CURRENT, processId, "Eps");
        dumper[type]->dumpGrid (totalEps,
                                startEps,
                                endEps);

        dumper[type]->init (0, CURRENT, processId, "Mu");
        dumper[type]->dumpGrid (totalMu,
                                startMu,
                                endMu);

        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          dumper[type]->init (0, CURRENT, processId, "OmegaPE");
          dumper[type]->dumpGrid (totalOmegaPE,
                                  startOmegaPE,
                                  endOmegaPE);

          dumper[type]->init (0, CURRENT, processId, "OmegaPM");
          dumper[type]->dumpGrid (totalOmegaPM,
                                  startOmegaPM,
                                  endOmegaPM);

          dumper[type]->init (0, CURRENT, processId, "GammaE");
          dumper[type]->dumpGrid (totalGammaE,
                                  startGammaE,
                                  endGammaE);

          dumper[type]->init (0, CURRENT, processId, "GammaM");
          dumper[type]->dumpGrid (totalGammaM,
                                  startGammaM,
                                  endGammaM);
        }
        //
        // if (SOLVER_SETTINGS.getDoUsePML ())
        // {
        //   dumper[type]->init (0, CURRENT, processId, "internalScheme.SigmaX");
        //   dumper[type]->dumpGrid (internalScheme.SigmaX,
        //                           GridCoordinate3D (0, 0, internalScheme.SigmaX->getSize ().get3 () / 2),
        //                           GridCoordinate3D (internalScheme.SigmaX->getSize ().get1 (), internalScheme.SigmaX->getSize ().get2 (), internalScheme.SigmaX->getSize ().get3 () / 2 + 1));
        //
        //   dumper[type]->init (0, CURRENT, processId, "internalScheme.SigmaY");
        //   dumper[type]->dumpGrid (internalScheme.SigmaY,
        //                           GridCoordinate3D (0, 0, internalScheme.SigmaY->getSize ().get3 () / 2),
        //                           GridCoordinate3D (internalScheme.SigmaY->getSize ().get1 (), internalScheme.SigmaY->getSize ().get2 (), internalScheme.SigmaY->getSize ().get3 () / 2 + 1));
        //
        //   dumper[type]->init (0, CURRENT, processId, "internalScheme.SigmaZ");
        //   dumper[type]->dumpGrid (internalScheme.SigmaZ,
        //                           GridCoordinate3D (0, 0, internalScheme.SigmaZ->getSize ().get3 () / 2),
        //                           GridCoordinate3D (internalScheme.SigmaZ->getSize ().get1 (), internalScheme.SigmaZ->getSize ().get2 (), internalScheme.SigmaZ->getSize ().get3 () / 2 + 1));
        // }
      }
    }
  }

  if (internalScheme.doNeedEx)
  {
    initGridWithInitialVals (GridType::EX, internalScheme.Ex, 0.5 * gridTimeStep);
  }
  if (internalScheme.doNeedEy)
  {
    initGridWithInitialVals (GridType::EY, internalScheme.Ey, 0.5 * gridTimeStep);
  }
  if (internalScheme.doNeedEz)
  {
    initGridWithInitialVals (GridType::EZ, internalScheme.Ez, 0.5 * gridTimeStep);
  }

  if (internalScheme.doNeedHx)
  {
    initGridWithInitialVals (GridType::HX, internalScheme.Hx, gridTimeStep);
  }
  if (internalScheme.doNeedHy)
  {
    initGridWithInitialVals (GridType::HY, internalScheme.Hy, gridTimeStep);
  }
  if (internalScheme.doNeedHz)
  {
    initGridWithInitialVals (GridType::HZ, internalScheme.Hz, gridTimeStep);
  }

  if (useParallel)
  {
#if defined (PARALLEL_GRID)
    MPI_Barrier (ParallelGrid::getParallelCore ()->getCommunicator ());

    ((ParallelGrid *) internalScheme.Eps)->share ();
    ((ParallelGrid *) internalScheme.Mu)->share ();

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      if (internalScheme.doNeedSigmaX)
      {
        ((ParallelGrid *) internalScheme.SigmaX)->share ();
      }
      if (internalScheme.doNeedSigmaY)
      {
        ((ParallelGrid *) internalScheme.SigmaY)->share ();
      }
      if (internalScheme.doNeedSigmaZ)
      {
        ((ParallelGrid *) internalScheme.SigmaZ)->share ();
      }
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
FPValue
Scheme<Type, TCoord, layout_type>::Pointing_scat (FPValue angleTeta, FPValue anglePhi, Grid<TC> *curEx, Grid<TC> *curEy, Grid<TC> *curEz,
                       Grid<TC> *curHx, Grid<TC> *curHy, Grid<TC> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  FPValue k = 2 * PhysicsConst::Pi / sourceWaveLength; // TODO: check numerical here

  NPair N = ntffN (angleTeta, anglePhi, curEz, curHx, curHy, curHz);
  NPair L = ntffL (angleTeta, anglePhi, curEx, curEy, curEz);

  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();

    FieldValue tmpArray[4];
    FieldValue tmpArrayRes[4];
    const int count = 4;

    tmpArray[0] = N.nTeta;
    tmpArray[1] = N.nPhi;
    tmpArray[2] = L.nTeta;
    tmpArray[3] = L.nPhi;

    // gather all sum_teta and sum_phi on 0 node
    MPI_Reduce (tmpArray, tmpArrayRes, count, MPI_FPVALUE, MPI_SUM, 0, ParallelGrid::getParallelCore ()->getCommunicator ());

    if (processId == 0)
    {
      N.nTeta = FieldValue (tmpArrayRes[0]);
      N.nPhi = FieldValue (tmpArrayRes[1]);

      L.nTeta = FieldValue (tmpArrayRes[2]);
      L.nPhi = FieldValue (tmpArrayRes[3]);
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  if (processId == 0)
  {
    FPValue n0 = sqrt (PhysicsConst::Mu0 / PhysicsConst::Eps0);

    FieldValue first = -L.nPhi + N.nTeta * n0;
    FieldValue second = -L.nTeta - N.nPhi * n0;

    FPValue first_abs2 = SQR (first.real ()) + SQR (first.imag ());
    FPValue second_abs2 = SQR (second.real ()) + SQR (second.imag ());

    return SQR(k) / (8 * PhysicsConst::Pi * n0) * (first_abs2 + second_abs2);
  }
  else
  {
    return 0.0;
  }
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
FPValue
Scheme<Type, TCoord, layout_type>::Pointing_inc (FPValue angleTeta, FPValue anglePhi)
{
  return sqrt (PhysicsConst::Eps0 / PhysicsConst::Mu0);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::makeGridScattered (Grid<TC> *grid, GridType gridType)
{
  for (grid_coord i = 0; i < grid->getSize ().calculateTotalCoord (); ++i)
  {
    FieldPointValue *val = grid->getFieldPointValue (i);

    TC pos = grid->calculatePositionFromIndex (i);
    TC posAbs = grid->getTotalPosition (pos);

    TCFP realCoord;
    switch (gridType)
    {
      case GridType::EX:
      {
        realCoord = yeeLayout->getExCoordFP (posAbs);
        break;
      }
      case GridType::EY:
      {
        realCoord = yeeLayout->getEyCoordFP (posAbs);
        break;
      }
      case GridType::EZ:
      {
        realCoord = yeeLayout->getEzCoordFP (posAbs);
        break;
      }
      case GridType::HX:
      {
        realCoord = yeeLayout->getHxCoordFP (posAbs);
        break;
      }
      case GridType::HY:
      {
        realCoord = yeeLayout->getHyCoordFP (posAbs);
        break;
      }
      case GridType::HZ:
      {
        realCoord = yeeLayout->getHzCoordFP (posAbs);
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    if (doSkipMakeScattered (realCoord))
    {
      continue;
    }

    YeeGridLayout<Type, TCoord, layout_type> *layout = Scheme<Type, TCoord, layout_type>::yeeLayout;

    FieldValue iVal;
    if (gridType == GridType::EX
        || gridType == GridType::EY
        || gridType == GridType::EZ)
    {
      iVal = internalScheme.approximateIncidentWaveE (realCoord);
    }
    else if (gridType == GridType::HX
             || gridType == GridType::HY
             || gridType == GridType::HZ)
    {
      iVal = internalScheme.approximateIncidentWaveH (realCoord);
    }
    else
    {
      UNREACHABLE;
    }

    FieldValue incVal;
    switch (gridType)
    {
      case GridType::EX:
      {
        incVal = yeeLayout->getExFromIncidentE (iVal);
        break;
      }
      case GridType::EY:
      {
        incVal = yeeLayout->getEyFromIncidentE (iVal);
        break;
      }
      case GridType::EZ:
      {
        incVal = yeeLayout->getEzFromIncidentE (iVal);
        break;
      }
      case GridType::HX:
      {
        incVal = yeeLayout->getHxFromIncidentH (iVal);
        break;
      }
      case GridType::HY:
      {
        incVal = yeeLayout->getHyFromIncidentH (iVal);
        break;
      }
      case GridType::HZ:
      {
        incVal = yeeLayout->getHzFromIncidentH (iVal);
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    val->setCurValue (val->getCurValue () - incVal);
  }
}


template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), GridCoordinate1DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids1D (&totalInitialized, internalScheme.doNeedEx, &internalScheme.Ex, &totalEx, internalScheme.doNeedEy, &internalScheme.Ey, &totalEy, internalScheme.doNeedEz, &internalScheme.Ez, &totalEz,
                                      internalScheme.doNeedHx, &internalScheme.Hx, &totalHx, internalScheme.doNeedHy, &internalScheme.Hy, &totalHy, internalScheme.doNeedHz, &internalScheme.Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), GridCoordinate1DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids1D (&totalInitialized, internalScheme.doNeedEx, &internalScheme.Ex, &totalEx, internalScheme.doNeedEy, &internalScheme.Ey, &totalEy, internalScheme.doNeedEz, &internalScheme.Ez, &totalEz,
                                      internalScheme.doNeedHx, &internalScheme.Hx, &totalHx, internalScheme.doNeedHy, &internalScheme.Hy, &totalHy, internalScheme.doNeedHz, &internalScheme.Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), GridCoordinate1DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids1D (&totalInitialized, internalScheme.doNeedEx, &internalScheme.Ex, &totalEx, internalScheme.doNeedEy, &internalScheme.Ey, &totalEy, internalScheme.doNeedEz, &internalScheme.Ez, &totalEz,
                                      internalScheme.doNeedHx, &internalScheme.Hx, &totalHx, internalScheme.doNeedHy, &internalScheme.Hy, &totalHy, internalScheme.doNeedHz, &internalScheme.Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), GridCoordinate1DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids1D (&totalInitialized, internalScheme.doNeedEx, &internalScheme.Ex, &totalEx, internalScheme.doNeedEy, &internalScheme.Ey, &totalEy, internalScheme.doNeedEz, &internalScheme.Ez, &totalEz,
                                      internalScheme.doNeedHx, &internalScheme.Hx, &totalHx, internalScheme.doNeedHy, &internalScheme.Hy, &totalHy, internalScheme.doNeedHz, &internalScheme.Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), GridCoordinate1DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids1D (&totalInitialized, internalScheme.doNeedEx, &internalScheme.Ex, &totalEx, internalScheme.doNeedEy, &internalScheme.Ey, &totalEy, internalScheme.doNeedEz, &internalScheme.Ez, &totalEz,
                                      internalScheme.doNeedHx, &internalScheme.Hx, &totalHx, internalScheme.doNeedHy, &internalScheme.Hy, &totalHy, internalScheme.doNeedHz, &internalScheme.Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), GridCoordinate1DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids1D (&totalInitialized, internalScheme.doNeedEx, &internalScheme.Ex, &totalEx, internalScheme.doNeedEy, &internalScheme.Ey, &totalEy, internalScheme.doNeedEz, &internalScheme.Ez, &totalEz,
                                      internalScheme.doNeedHx, &internalScheme.Hx, &totalHx, internalScheme.doNeedHy, &internalScheme.Hy, &totalHy, internalScheme.doNeedHz, &internalScheme.Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), GridCoordinate2DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids2D (&totalInitialized, internalScheme.doNeedEx, &internalScheme.Ex, &totalEx, internalScheme.doNeedEy, &internalScheme.Ey, &totalEy, internalScheme.doNeedEz, &internalScheme.Ez, &totalEz,
                                      internalScheme.doNeedHx, &internalScheme.Hx, &totalHx, internalScheme.doNeedHy, &internalScheme.Hy, &totalHy, internalScheme.doNeedHz, &internalScheme.Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), GridCoordinate2DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids2D (&totalInitialized, internalScheme.doNeedEx, &internalScheme.Ex, &totalEx, internalScheme.doNeedEy, &internalScheme.Ey, &totalEy, internalScheme.doNeedEz, &internalScheme.Ez, &totalEz,
                                      internalScheme.doNeedHx, &internalScheme.Hx, &totalHx, internalScheme.doNeedHy, &internalScheme.Hy, &totalHy, internalScheme.doNeedHz, &internalScheme.Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), GridCoordinate2DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids2D (&totalInitialized, internalScheme.doNeedEx, &internalScheme.Ex, &totalEx, internalScheme.doNeedEy, &internalScheme.Ey, &totalEy, internalScheme.doNeedEz, &internalScheme.Ez, &totalEz,
                                      internalScheme.doNeedHx, &internalScheme.Hx, &totalHx, internalScheme.doNeedHy, &internalScheme.Hy, &totalHy, internalScheme.doNeedHz, &internalScheme.Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), GridCoordinate2DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids2D (&totalInitialized, internalScheme.doNeedEx, &internalScheme.Ex, &totalEx, internalScheme.doNeedEy, &internalScheme.Ey, &totalEy, internalScheme.doNeedEz, &internalScheme.Ez, &totalEz,
                                      internalScheme.doNeedHx, &internalScheme.Hx, &totalHx, internalScheme.doNeedHy, &internalScheme.Hy, &totalHy, internalScheme.doNeedHz, &internalScheme.Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), GridCoordinate2DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids2D (&totalInitialized, internalScheme.doNeedEx, &internalScheme.Ex, &totalEx, internalScheme.doNeedEy, &internalScheme.Ey, &totalEy, internalScheme.doNeedEz, &internalScheme.Ez, &totalEz,
                                      internalScheme.doNeedHx, &internalScheme.Hx, &totalHx, internalScheme.doNeedHy, &internalScheme.Hy, &totalHy, internalScheme.doNeedHz, &internalScheme.Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), GridCoordinate2DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids2D (&totalInitialized, internalScheme.doNeedEx, &internalScheme.Ex, &totalEx, internalScheme.doNeedEy, &internalScheme.Ey, &totalEy, internalScheme.doNeedEz, &internalScheme.Ez, &totalEz,
                                      internalScheme.doNeedHx, &internalScheme.Hx, &totalHx, internalScheme.doNeedHy, &internalScheme.Hy, &totalHy, internalScheme.doNeedHz, &internalScheme.Hz, &totalHz);
}

template <>
void
Scheme< (static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, E_CENTERED>::initFullFieldGrids ()
{
  SchemeHelper::initFullFieldGrids3D (&totalInitialized, internalScheme.doNeedEx, &internalScheme.Ex, &totalEx, internalScheme.doNeedEy, &internalScheme.Ey, &totalEy, internalScheme.doNeedEz, &internalScheme.Ez, &totalEz,
                                      internalScheme.doNeedHx, &internalScheme.Hx, &totalHx, internalScheme.doNeedHy, &internalScheme.Hy, &totalHy, internalScheme.doNeedHz, &internalScheme.Hz, &totalHz);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::gatherFieldsTotal (bool scattered)
{
  if (useParallel)
  {
    initFullFieldGrids ();
  }
  else
  {
    if (totalInitialized)
    {
      if (internalScheme.doNeedEx)
      {
        *totalEx = *internalScheme.Ex;
      }
      if (internalScheme.doNeedEy)
      {
        *totalEy = *internalScheme.Ey;
      }
      if (internalScheme.doNeedEz)
      {
        *totalEz = *internalScheme.Ez;
      }

      if (internalScheme.doNeedHx)
      {
        *totalHx = *internalScheme.Hx;
      }
      if (internalScheme.doNeedHy)
      {
        *totalHy = *internalScheme.Hy;
      }
      if (internalScheme.doNeedHz)
      {
        *totalHz = *internalScheme.Hz;
      }
    }
    else
    {
      if (scattered)
      {
        if (internalScheme.doNeedEx)
        {
          totalEx = new Grid<TC> (yeeLayout->getExSize (), 0, "Ex");
          *totalEx = *internalScheme.Ex;
        }
        if (internalScheme.doNeedEy)
        {
          totalEy = new Grid<TC> (yeeLayout->getEySize (), 0, "Ey");
          *totalEy = *internalScheme.Ey;
        }
        if (internalScheme.doNeedEz)
        {
          totalEz = new Grid<TC> (yeeLayout->getEzSize (), 0, "Ez");
          *totalEz = *internalScheme.Ez;
        }

        if (internalScheme.doNeedHx)
        {
          totalHx = new Grid<TC> (yeeLayout->getHxSize (), 0, "Hx");
          *totalHx = *internalScheme.Hx;
        }
        if (internalScheme.doNeedHy)
        {
          totalHy = new Grid<TC> (yeeLayout->getHySize (), 0, "Hy");
          *totalHy = *internalScheme.Hy;
        }
        if (internalScheme.doNeedHz)
        {
          totalHz = new Grid<TC> (yeeLayout->getHzSize (), 0, "Hz");
          *totalHz = *internalScheme.Hz;
        }

        totalInitialized = true;
      }
      else
      {
        if (internalScheme.doNeedEx)
        {
          totalEx = internalScheme.Ex;
        }
        if (internalScheme.doNeedEy)
        {
          totalEy = internalScheme.Ey;
        }
        if (internalScheme.doNeedEz)
        {
          totalEz = internalScheme.Ez;
        }

        if (internalScheme.doNeedHx)
        {
          totalHx = internalScheme.Hx;
        }
        if (internalScheme.doNeedHy)
        {
          totalHy = internalScheme.Hy;
        }
        if (internalScheme.doNeedHz)
        {
          totalHz = internalScheme.Hz;
        }
      }
    }
  }

  if (scattered)
  {
    if (internalScheme.doNeedEx)
    {
      makeGridScattered (totalEx, GridType::EX);
    }
    if (internalScheme.doNeedEy)
    {
      makeGridScattered (totalEy, GridType::EY);
    }
    if (internalScheme.doNeedEz)
    {
      makeGridScattered (totalEz, GridType::EZ);
    }

    if (internalScheme.doNeedHx)
    {
      makeGridScattered (totalHx, GridType::HX);
    }
    if (internalScheme.doNeedHy)
    {
      makeGridScattered (totalHy, GridType::HY);
    }
    if (internalScheme.doNeedHz)
    {
      makeGridScattered (totalHz, GridType::HZ);
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::saveGrids (time_step t)
{
  int processId = 0;
  if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else
    UNREACHABLE;
#endif
  }

  TC startEx;
  TC endEx;
  TC startEy;
  TC endEy;
  TC startEz;
  TC endEz;
  TC startHx;
  TC endHx;
  TC startHy;
  TC endHy;
  TC startHz;
  TC endHz;

  if (SOLVER_SETTINGS.getDoUseManualStartEndDumpCoord ())
  {
    TC start = TC::initAxesCoordinate (SOLVER_SETTINGS.getSaveStartCoordX (),
                                       SOLVER_SETTINGS.getSaveStartCoordY (),
                                       SOLVER_SETTINGS.getSaveStartCoordZ (),
                                       ct1, ct2, ct3);
    TC end = TC::initAxesCoordinate (SOLVER_SETTINGS.getSaveEndCoordX (),
                                     SOLVER_SETTINGS.getSaveEndCoordY (),
                                     SOLVER_SETTINGS.getSaveEndCoordZ (),
                                     ct1, ct2, ct3);

    startEx = startEy = startEz = startHx = startHy = startHz = start;
    endEx = endEy = endEz = endHx = endHy = endHz = end;
  }
  else
  {
    startEx = internalScheme.doNeedEx ? getStartCoord (GridType::EX, internalScheme.Ex->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                 , ct1, ct2, ct3
#endif
                                                                                 );
    endEx = internalScheme.doNeedEx ? getEndCoord (GridType::EX, internalScheme.Ex->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                             , ct1, ct2, ct3
#endif
                                                                             );

    startEy = internalScheme.doNeedEy ? getStartCoord (GridType::EY, internalScheme.Ey->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                 , ct1, ct2, ct3
#endif
                                                                                 );
    endEy = internalScheme.doNeedEy ? getEndCoord (GridType::EY, internalScheme.Ey->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                             , ct1, ct2, ct3
#endif
                                                                             );

    startEz = internalScheme.doNeedEz ? getStartCoord (GridType::EZ, internalScheme.Ez->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                 , ct1, ct2, ct3
#endif
                                                                                 );
    endEz = internalScheme.doNeedEz ? getEndCoord (GridType::EZ, internalScheme.Ez->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                             , ct1, ct2, ct3
#endif
                                                                             );

    startHx = internalScheme.doNeedHx ? getStartCoord (GridType::HX, internalScheme.Hx->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                 , ct1, ct2, ct3
#endif
                                                                                 );
    endHx = internalScheme.doNeedHx ? getEndCoord (GridType::HX, internalScheme.Hx->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                             , ct1, ct2, ct3
#endif
                                                                             );

    startHy = internalScheme.doNeedHy ? getStartCoord (GridType::HY, internalScheme.Hy->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                 , ct1, ct2, ct3
#endif
                                                                                 );
    endHy = internalScheme.doNeedHy ? getEndCoord (GridType::HY, internalScheme.Hy->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                             , ct1, ct2, ct3
#endif
                                                                             );

    startHz = internalScheme.doNeedHz ? getStartCoord (GridType::HZ, internalScheme.Hz->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                                 , ct1, ct2, ct3
#endif
                                                                                 );
    endHz = internalScheme.doNeedHz ? getEndCoord (GridType::HZ, internalScheme.Hz->getTotalSize ()) : TC (0, 0, 0
#ifdef DEBUG_INFO
                                                                             , ct1, ct2, ct3
#endif
                                                                             );
  }

  TC zero (0, 0, 0
#ifdef DEBUG_INFO
           , ct1, ct2, ct3
#endif
           );

  for (int type = FILE_TYPE_BMP; type < FILE_TYPE_COUNT; ++type)
  {
    if (!dumper[type])
    {
      continue;
    }

    if (internalScheme.doNeedEx)
    {
      dumper[type]->init (t, CURRENT, processId, totalEx->getName ());

      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (internalScheme.Ex, zero, internalScheme.Ex->getSize ());
      }
      else
      {
        dumper[type]->dumpGrid (totalEx, startEx, endEx);
      }
    }

    if (internalScheme.doNeedEy)
    {
      dumper[type]->init (t, CURRENT, processId, totalEy->getName ());

      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (internalScheme.Ey, zero, internalScheme.Ey->getSize ());
      }
      else
      {
        dumper[type]->dumpGrid (totalEy, startEy, endEy);
      }
    }

    if (internalScheme.doNeedEz)
    {
      dumper[type]->init (t, CURRENT, processId, totalEz->getName ());

      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (internalScheme.Ez, zero, internalScheme.Ez->getSize ());
      }
      else
      {
        dumper[type]->dumpGrid (totalEz, startEz, endEz);
      }
    }

    if (internalScheme.doNeedHx)
    {
      dumper[type]->init (t, CURRENT, processId, totalHx->getName ());

      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (internalScheme.Hx, zero, internalScheme.Hx->getSize ());
      }
      else
      {
        dumper[type]->dumpGrid (totalHx, startHx, endHx);
      }
    }

    if (internalScheme.doNeedHy)
    {
      dumper[type]->init (t, CURRENT, processId, totalHy->getName ());

      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (internalScheme.Hy, zero, internalScheme.Hy->getSize ());
      }
      else
      {
        dumper[type]->dumpGrid (totalHy, startHy, endHy);
      }
    }

    if (internalScheme.doNeedHz)
    {
      dumper[type]->init (t, CURRENT, processId, totalHz->getName ());

      if (SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        dumper[type]->dumpGrid (internalScheme.Hz, zero, internalScheme.Hz->getSize ());
      }
      else
      {
        dumper[type]->dumpGrid (totalHz, startHz, endHz);
      }
    }

    if (SOLVER_SETTINGS.getDoSaveTFSFEInc ())
    {
      if (!dumper1D[type])
      {
        continue;
      }

      dumper1D[type]->init (t, CURRENT, processId, "EInc");
      dumper1D[type]->dumpGrid (EInc, GridCoordinate1D (0
#ifdef DEBUG_INFO
                                                        , CoordinateType::X
#endif
                                                        ),
                                EInc->getSize ());
    }

    if (SOLVER_SETTINGS.getDoSaveTFSFHInc ())
    {
      if (!dumper1D[type])
      {
        continue;
      }

      dumper1D[type]->init (t, CURRENT, processId, "HInc");
      dumper1D[type]->dumpGrid (HInc, GridCoordinate1D (0
#ifdef DEBUG_INFO
                                                        , CoordinateType::X
#endif
                                                        ),
                                HInc->getSize ());
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::saveNTFF (bool isReverse, time_step t)
{
  int processId = 0;

  if (useParallel)
  {
#ifdef PARALLEL_GRID
    processId = ParallelGrid::getParallelCore ()->getProcessId ();
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  std::ofstream outfile;
  std::ostream *outs;
  const char *strName;
  FPValue start;
  FPValue end;
  FPValue step;

  if (isReverse)
  {
    strName = "Reverse diagram";
    start = yeeLayout->getIncidentWaveAngle2 ();
    end = yeeLayout->getIncidentWaveAngle2 ();
    step = 1.0;
  }
  else
  {
    strName = "Forward diagram";
    start = 0.0;
    end = 2 * PhysicsConst::Pi + PhysicsConst::Pi / 180;
    step = PhysicsConst::Pi * SOLVER_SETTINGS.getAngleStepNTFF () / 180;
  }

  if (processId == 0)
  {
    if (SOLVER_SETTINGS.getDoSaveNTFFToStdout ())
    {
      outs = &std::cout;
    }
    else
    {
      outfile.open (SOLVER_SETTINGS.getFileNameNTFF ().c_str ());
      outs = &outfile;
    }
    (*outs) << strName << std::endl << std::endl;
  }

  for (FPValue angle = start; angle <= end; angle += step)
  {
    FPValue val = Pointing_scat (yeeLayout->getIncidentWaveAngle1 (),
                                 angle,
                                 internalScheme.Ex,
                                 internalScheme.Ey,
                                 internalScheme.Ez,
                                 internalScheme.Hx,
                                 internalScheme.Hy,
                                 internalScheme.Hz) / Pointing_inc (yeeLayout->getIncidentWaveAngle1 (), angle);

    if (processId == 0)
    {
      (*outs) << "timestep = "
              << t
              << ", incident wave angle=("
              << yeeLayout->getIncidentWaveAngle1 () << ","
              << yeeLayout->getIncidentWaveAngle2 () << ","
              << yeeLayout->getIncidentWaveAngle3 () << ","
              << "), angle NTFF = "
              << angle
              << ", NTFF value = "
              << val
              << std::endl;
    }
  }

  if (processId == 0)
  {
    if (!SOLVER_SETTINGS.getDoSaveNTFFToStdout ())
    {
      outfile.close ();
    }
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
void
Scheme<Type, TCoord, layout_type>::additionalUpdateOfGrids (time_step t, time_step &diffT)
{
  if (useParallel && SOLVER_SETTINGS.getDoUseDynamicGrid ())
  {
#if defined (PARALLEL_GRID) && defined (DYNAMIC_GRID)
    //if (false && t % SOLVER_SETTINGS.getRebalanceStep () == 0)
    if (t % diffT == 0 && t > 0)
    {
      if (ParallelGrid::getParallelCore ()->getProcessId () == 0)
      {
        DPRINTF (LOG_LEVEL_STAGES, "Try rebalance on step %u, steps elapsed after previous %u\n", t, diffT);
      }

      ASSERT (isParallelLayout);

      ParallelYeeGridLayout<Type, layout_type> *parallelYeeLayout = (ParallelYeeGridLayout<Type, layout_type> *) yeeLayout;

      if (parallelYeeLayout->Rebalance (diffT))
      {
        DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Rebalancing for process %d!\n", ParallelGrid::getParallelCore ()->getProcessId ());

        ((ParallelGrid *) internalScheme.Eps)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
        ((ParallelGrid *) internalScheme.Mu)->Resize (parallelYeeLayout->getMuSizeForCurNode ());

        if (internalScheme.doNeedEx)
        {
          ((ParallelGrid *) internalScheme.Ex)->Resize (parallelYeeLayout->getExSizeForCurNode ());
        }
        if (internalScheme.doNeedEy)
        {
          ((ParallelGrid *) internalScheme.Ey)->Resize (parallelYeeLayout->getEySizeForCurNode ());
        }
        if (internalScheme.doNeedEz)
        {
          ((ParallelGrid *) internalScheme.Ez)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
        }

        if (internalScheme.doNeedHx)
        {
          ((ParallelGrid *) internalScheme.Hx)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
        }
        if (internalScheme.doNeedHy)
        {
          ((ParallelGrid *) internalScheme.Hy)->Resize (parallelYeeLayout->getHySizeForCurNode ());
        }
        if (internalScheme.doNeedHz)
        {
          ((ParallelGrid *) internalScheme.Hz)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
        }

        if (SOLVER_SETTINGS.getDoUsePML ())
        {
          if (internalScheme.doNeedEx)
          {
            ((ParallelGrid *) internalScheme.Dx)->Resize (parallelYeeLayout->getExSizeForCurNode ());
          }
          if (internalScheme.doNeedEy)
          {
            ((ParallelGrid *) internalScheme.Dy)->Resize (parallelYeeLayout->getEySizeForCurNode ());
          }
          if (internalScheme.doNeedEz)
          {
            ((ParallelGrid *) internalScheme.Dz)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
          }

          if (internalScheme.doNeedHx)
          {
            ((ParallelGrid *) internalScheme.Bx)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
          }
          if (internalScheme.doNeedHy)
          {
            ((ParallelGrid *) internalScheme.By)->Resize (parallelYeeLayout->getHySizeForCurNode ());
          }
          if (internalScheme.doNeedHz)
          {
            ((ParallelGrid *) internalScheme.Bz)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
          }

          if (SOLVER_SETTINGS.getDoUseMetamaterials ())
          {
            if (internalScheme.doNeedEx)
            {
              ((ParallelGrid *) internalScheme.D1x)->Resize (parallelYeeLayout->getExSizeForCurNode ());
            }
            if (internalScheme.doNeedEy)
            {
              ((ParallelGrid *) internalScheme.D1y)->Resize (parallelYeeLayout->getEySizeForCurNode ());
            }
            if (internalScheme.doNeedEz)
            {
              ((ParallelGrid *) internalScheme.D1z)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
            }

            if (internalScheme.doNeedHx)
            {
              ((ParallelGrid *) internalScheme.B1x)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
            }
            if (internalScheme.doNeedHy)
            {
              ((ParallelGrid *) internalScheme.B1y)->Resize (parallelYeeLayout->getHySizeForCurNode ());
            }
            if (internalScheme.doNeedHz)
            {
              ((ParallelGrid *) internalScheme.B1z)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
            }
          }

          if (internalScheme.doNeedSigmaX)
          {
            ((ParallelGrid *) internalScheme.SigmaX)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          }
          if (internalScheme.doNeedSigmaY)
          {
            ((ParallelGrid *) internalScheme.SigmaY)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          }
          if (internalScheme.doNeedSigmaZ)
          {
            ((ParallelGrid *) internalScheme.SigmaZ)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          }
        }

        if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
        {
          if (internalScheme.doNeedEx)
          {
            ((ParallelGrid *) internalScheme.ExAmplitude)->Resize (parallelYeeLayout->getExSizeForCurNode ());
          }
          if (internalScheme.doNeedEy)
          {
            ((ParallelGrid *) internalScheme.EyAmplitude)->Resize (parallelYeeLayout->getEySizeForCurNode ());
          }
          if (internalScheme.doNeedEz)
          {
            ((ParallelGrid *) internalScheme.EzAmplitude)->Resize (parallelYeeLayout->getEzSizeForCurNode ());
          }

          if (internalScheme.doNeedHx)
          {
            ((ParallelGrid *) internalScheme.HxAmplitude)->Resize (parallelYeeLayout->getHxSizeForCurNode ());
          }
          if (internalScheme.doNeedHy)
          {
            ((ParallelGrid *) internalScheme.HyAmplitude)->Resize (parallelYeeLayout->getHySizeForCurNode ());
          }
          if (internalScheme.doNeedHz)
          {
            ((ParallelGrid *) internalScheme.HzAmplitude)->Resize (parallelYeeLayout->getHzSizeForCurNode ());
          }
        }

        if (SOLVER_SETTINGS.getDoUseMetamaterials ())
        {
          ((ParallelGrid *) internalScheme.OmegaPE)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          ((ParallelGrid *) internalScheme.GammaE)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          ((ParallelGrid *) internalScheme.OmegaPM)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
          ((ParallelGrid *) internalScheme.GammaM)->Resize (parallelYeeLayout->getEpsSizeForCurNode ());
        }

        //diffT += 1;
        //diffT *= 2;
      }
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. "
                    "Recompile it with -DPARALLEL_GRID=ON and -DDYNAMIC_GRID=ON.");
#endif
  }
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
typename Scheme<Type, TCoord, layout_type>::TC
Scheme<Type, TCoord, layout_type>::getStartCoord (GridType gridType, TC size)
{
  TC start (0, 0, 0
#ifdef DEBUG_INFO
            , ct1, ct2, ct3
#endif
            );

  if (SOLVER_SETTINGS.getDoSaveWithoutPML ()
      && SOLVER_SETTINGS.getDoUsePML ())
  {
    TCFP leftBorder = convertCoord (yeeLayout->getLeftBorderPML ());
    TCFP min;

    switch (gridType)
    {
      case GridType::EX:
      {
        min = yeeLayout->getMinExCoordFP ();
        break;
      }
      case GridType::EY:
      {
        min = yeeLayout->getMinEyCoordFP ();
        break;
      }
      case GridType::EZ:
      {
        min = yeeLayout->getMinEzCoordFP ();
        break;
      }
      case GridType::HX:
      {
        min = yeeLayout->getMinHxCoordFP ();
        break;
      }
      case GridType::HY:
      {
        min = yeeLayout->getMinHyCoordFP ();
        break;
      }
      case GridType::HZ:
      {
        min = yeeLayout->getMinHzCoordFP ();
        break;
      }
      default:
      {
        // do nothing
      }
    }

    start = convertCoord (expandTo3D (leftBorder - min, ct1, ct2, ct3)) + GridCoordinate3D (1, 1, 1
#ifdef DEBUG_INFO
                                                                                            , ct1, ct2, ct3
#endif
                                                                                            );
  }

  OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;
  if (SOLVER_SETTINGS.getDoUseOrthAxisX ())
  {
    orthogonalAxis = OrthogonalAxis::X;
  }
  else if (SOLVER_SETTINGS.getDoUseOrthAxisY ())
  {
    orthogonalAxis = OrthogonalAxis::Y;
  }
  else if (SOLVER_SETTINGS.getDoUseOrthAxisZ ())
  {
    orthogonalAxis = OrthogonalAxis::Z;
  }

  return getStartCoordRes (orthogonalAxis, start, size);
}

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
typename Scheme<Type, TCoord, layout_type>::TC
Scheme<Type, TCoord, layout_type>::getEndCoord (GridType gridType, TC size)
{
  TC end = size;
  if (SOLVER_SETTINGS.getDoSaveWithoutPML ()
      && SOLVER_SETTINGS.getDoUsePML ())
  {
    TCFP rightBorder = convertCoord (yeeLayout->getRightBorderPML ());
    TCFP min;

    switch (gridType)
    {
      case GridType::EX:
      {
        min = yeeLayout->getMinExCoordFP ();
        break;
      }
      case GridType::EY:
      {
        min = yeeLayout->getMinEyCoordFP ();
        break;
      }
      case GridType::EZ:
      {
        min = yeeLayout->getMinEzCoordFP ();
        break;
      }
      case GridType::HX:
      {
        min = yeeLayout->getMinHxCoordFP ();
        break;
      }
      case GridType::HY:
      {
        min = yeeLayout->getMinHyCoordFP ();
        break;
      }
      case GridType::HZ:
      {
        min = yeeLayout->getMinHzCoordFP ();
        break;
      }
      default:
      {
        // do nothing
      }
    }

    end = convertCoord (expandTo3D (rightBorder - min, ct1, ct2, ct3));
  }

  OrthogonalAxis orthogonalAxis = OrthogonalAxis::Z;
  if (SOLVER_SETTINGS.getDoUseOrthAxisX ())
  {
    orthogonalAxis = OrthogonalAxis::X;
  }
  else if (SOLVER_SETTINGS.getDoUseOrthAxisY ())
  {
    orthogonalAxis = OrthogonalAxis::Y;
  }
  else if (SOLVER_SETTINGS.getDoUseOrthAxisZ ())
  {
    orthogonalAxis = OrthogonalAxis::Z;
  }

  return getEndCoordRes (orthogonalAxis, end, size);
}

template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim1_ExHy), GridCoordinate1DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim1_ExHz), GridCoordinate1DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim1_EyHx), GridCoordinate1DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim1_EyHz), GridCoordinate1DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim1_EzHx), GridCoordinate1DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim1_EzHy), GridCoordinate1DTemplate, E_CENTERED >;

template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim2_TEx), GridCoordinate2DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim2_TEy), GridCoordinate2DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim2_TEz), GridCoordinate2DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim2_TMx), GridCoordinate2DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim2_TMy), GridCoordinate2DTemplate, E_CENTERED >;
template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim2_TMz), GridCoordinate2DTemplate, E_CENTERED >;

template class
Scheme< static_cast<SchemeType_t> (SchemeType::Dim3), GridCoordinate3DTemplate, E_CENTERED >;
