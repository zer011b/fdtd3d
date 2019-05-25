#ifndef SCHEME_HELPER_H
#define SCHEME_HELPER_H

#include "GridInterface.h"
#include "PhysicsConst.h"
#include "YeeGridLayout.h"
#include "ParallelYeeGridLayout.h"
#include "CallBack.h"

#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"
#include "TXTDumper.h"
#include "TXTLoader.h"

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
class Scheme;

struct NPair
{
  FieldValue nTeta;
  FieldValue nPhi;

  NPair (FieldValue n_teta = 0, FieldValue n_phi = 0)
    : nTeta (n_teta)
  , nPhi (n_phi)
  {
  }

  NPair operator+ (const NPair &right)
  {
    return NPair (nTeta + right.nTeta, nPhi + right.nPhi);
  }
};

class SchemeHelper
{
public:

  template <SchemeType_t Type, LayoutType layout_type>
  static
  void performNSteps1D (Scheme<Type, GridCoordinate1DTemplate, layout_type> *scheme,
                        time_step tStart,
                        time_step N)
  {
    for (grid_coord c1 = 0; c1 < scheme->blockCount.get1 (); ++c1)
    {
      GridCoordinate1D blockIdx = GRID_COORDINATE_1D (c1, scheme->ct1);

      // TODO: save block to prev blocks storage
      scheme->performNStepsForBlock (tStart, N, blockIdx);
    }

#ifdef CUDA_ENABLED
    scheme->shareE ();
    scheme->shareH ();
#endif
    scheme->rebalance ();

    if (SOLVER_SETTINGS.getDoUseNTFF ()
        && ((tStart) / SOLVER_SETTINGS.getIntermediateNTFFStep () < (tStart + N) / SOLVER_SETTINGS.getIntermediateNTFFStep ()))
    {
      scheme->saveNTFF (SOLVER_SETTINGS.getDoCalcReverseNTFF (), tStart + N);
    }

    if (SOLVER_SETTINGS.getDoSaveIntermediateRes ()
        && ((tStart) / SOLVER_SETTINGS.getIntermediateSaveStep () < (tStart + N) / SOLVER_SETTINGS.getIntermediateSaveStep ()))
    {
      if (!SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        scheme->gatherFieldsTotal (SOLVER_SETTINGS.getDoSaveScatteredFieldIntermediate ());
      }
      else
      {
        ASSERT (!SOLVER_SETTINGS.getDoSaveScatteredFieldRes ());
      }

      scheme->saveGrids (tStart + N);
    }
  }

  template <SchemeType_t Type, LayoutType layout_type>
  static
  void performNSteps2D (Scheme<Type, GridCoordinate2DTemplate, layout_type> *scheme,
                        time_step tStart,
                        time_step N)
  {
    for (grid_coord c1 = 0; c1 < scheme->blockCount.get1 (); ++c1)
    {
      for (grid_coord c2 = 0; c2 < scheme->blockCount.get2 (); ++c2)
      {
        GridCoordinate2D blockIdx = GRID_COORDINATE_2D (c1, c2, scheme->ct1, scheme->ct2);

        // TODO: save block to prev blocks storage
        scheme->performNStepsForBlock (tStart, N, blockIdx);
      }
    }

#ifdef CUDA_ENABLED
    scheme->shareE ();
    scheme->shareH ();
#endif
    scheme->rebalance ();

    if (SOLVER_SETTINGS.getDoUseNTFF ()
        && ((tStart) / SOLVER_SETTINGS.getIntermediateNTFFStep () < (tStart + N) / SOLVER_SETTINGS.getIntermediateNTFFStep ()))
    {
      scheme->saveNTFF (SOLVER_SETTINGS.getDoCalcReverseNTFF (), tStart + N);
    }

    if (SOLVER_SETTINGS.getDoSaveIntermediateRes ()
        && ((tStart) / SOLVER_SETTINGS.getIntermediateSaveStep () < (tStart + N) / SOLVER_SETTINGS.getIntermediateSaveStep ()))
    {
      if (!SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        scheme->gatherFieldsTotal (SOLVER_SETTINGS.getDoSaveScatteredFieldIntermediate ());
      }
      else
      {
        ASSERT (!SOLVER_SETTINGS.getDoSaveScatteredFieldRes ());
      }

      scheme->saveGrids (tStart + N);
    }
  }

  template <SchemeType_t Type, LayoutType layout_type>
  static
  void performNSteps3D (Scheme<Type, GridCoordinate3DTemplate, layout_type> *scheme,
                        time_step tStart,
                        time_step N)
  {
    for (grid_coord c1 = 0; c1 < scheme->blockCount.get1 (); ++c1)
    {
      for (grid_coord c2 = 0; c2 < scheme->blockCount.get2 (); ++c2)
      {
        for (grid_coord c3 = 0; c3 < scheme->blockCount.get3 (); ++c3)
        {
          GridCoordinate3D blockIdx = GRID_COORDINATE_3D (c1, c2, c3, scheme->ct1, scheme->ct2, scheme->ct3);

          // TODO: save block to prev blocks storage
          scheme->performNStepsForBlock (tStart, N, blockIdx);
        }
      }
    }

#ifdef CUDA_ENABLED
    scheme->shareE ();
    scheme->shareH ();
#endif
    scheme->rebalance ();

    if (SOLVER_SETTINGS.getDoUseNTFF ()
        && ((tStart) / SOLVER_SETTINGS.getIntermediateNTFFStep () < (tStart + N) / SOLVER_SETTINGS.getIntermediateNTFFStep ()))
    {
      scheme->saveNTFF (SOLVER_SETTINGS.getDoCalcReverseNTFF (), tStart + N);
    }

    if (SOLVER_SETTINGS.getDoSaveIntermediateRes ()
        && ((tStart) / SOLVER_SETTINGS.getIntermediateSaveStep () < (tStart + N) / SOLVER_SETTINGS.getIntermediateSaveStep ()))
    {
      if (!SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        scheme->gatherFieldsTotal (SOLVER_SETTINGS.getDoSaveScatteredFieldIntermediate ());
      }
      else
      {
        ASSERT (!SOLVER_SETTINGS.getDoSaveScatteredFieldRes ());
      }

      scheme->saveGrids (tStart + N);
    }
  }

  static
  void initFullMaterialGrids1D (Grid<GridCoordinate1D> *Eps, Grid<GridCoordinate1D> *totalEps,
                                Grid<GridCoordinate1D> *Mu, Grid<GridCoordinate1D> *totalMu,
                                Grid<GridCoordinate1D> *OmegaPE, Grid<GridCoordinate1D> *totalOmegaPE,
                                Grid<GridCoordinate1D> *OmegaPM, Grid<GridCoordinate1D> *totalOmegaPM,
                                Grid<GridCoordinate1D> *GammaE, Grid<GridCoordinate1D> *totalGammaE,
                                Grid<GridCoordinate1D> *GammaM, Grid<GridCoordinate1D> *totalGammaM)
  {
#ifdef PARALLEL_GRID
#ifdef GRID_1D
    ((ParallelGrid *) Eps)->gatherFullGridPlacement (totalEps);
    ((ParallelGrid *) Mu)->gatherFullGridPlacement (totalMu);

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ((ParallelGrid *) OmegaPE)->gatherFullGridPlacement (totalOmegaPE);
      ((ParallelGrid *) OmegaPM)->gatherFullGridPlacement (totalOmegaPM);
      ((ParallelGrid *) GammaE)->gatherFullGridPlacement (totalGammaE);
      ((ParallelGrid *) GammaM)->gatherFullGridPlacement (totalGammaM);
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                    "Recompile it with -DPARALLEL_GRID_DIMENSION=1.");
#endif
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  static
  void initFullMaterialGrids2D (Grid<GridCoordinate2D> *Eps, Grid<GridCoordinate2D> *totalEps,
                                Grid<GridCoordinate2D> *Mu, Grid<GridCoordinate2D> *totalMu,
                                Grid<GridCoordinate2D> *OmegaPE, Grid<GridCoordinate2D> *totalOmegaPE,
                                Grid<GridCoordinate2D> *OmegaPM, Grid<GridCoordinate2D> *totalOmegaPM,
                                Grid<GridCoordinate2D> *GammaE, Grid<GridCoordinate2D> *totalGammaE,
                                Grid<GridCoordinate2D> *GammaM, Grid<GridCoordinate2D> *totalGammaM)
  {
#ifdef PARALLEL_GRID
#ifdef GRID_2D
    ((ParallelGrid *) Eps)->gatherFullGridPlacement (totalEps);
    ((ParallelGrid *) Mu)->gatherFullGridPlacement (totalMu);

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ((ParallelGrid *) OmegaPE)->gatherFullGridPlacement (totalOmegaPE);
      ((ParallelGrid *) OmegaPM)->gatherFullGridPlacement (totalOmegaPM);
      ((ParallelGrid *) GammaE)->gatherFullGridPlacement (totalGammaE);
      ((ParallelGrid *) GammaM)->gatherFullGridPlacement (totalGammaM);
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                    "Recompile it with -DPARALLEL_GRID_DIMENSION=2.");
#endif
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  static
  void initFullMaterialGrids3D (Grid<GridCoordinate3D> *Eps, Grid<GridCoordinate3D> *totalEps,
                                Grid<GridCoordinate3D> *Mu, Grid<GridCoordinate3D> *totalMu,
                                Grid<GridCoordinate3D> *OmegaPE, Grid<GridCoordinate3D> *totalOmegaPE,
                                Grid<GridCoordinate3D> *OmegaPM, Grid<GridCoordinate3D> *totalOmegaPM,
                                Grid<GridCoordinate3D> *GammaE, Grid<GridCoordinate3D> *totalGammaE,
                                Grid<GridCoordinate3D> *GammaM, Grid<GridCoordinate3D> *totalGammaM)
  {
#ifdef PARALLEL_GRID
#ifdef GRID_3D
    ((ParallelGrid *) Eps)->gatherFullGridPlacement (totalEps);
    ((ParallelGrid *) Mu)->gatherFullGridPlacement (totalMu);

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      ((ParallelGrid *) OmegaPE)->gatherFullGridPlacement (totalOmegaPE);
      ((ParallelGrid *) OmegaPM)->gatherFullGridPlacement (totalOmegaPM);
      ((ParallelGrid *) GammaE)->gatherFullGridPlacement (totalGammaE);
      ((ParallelGrid *) GammaM)->gatherFullGridPlacement (totalGammaM);
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                    "Recompile it with -DPARALLEL_GRID_DIMENSION=3.");
#endif
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  static
  void initFullFieldGrids1D (bool *totalInitialized,
                             bool doNeedEx, Grid<GridCoordinate1D> *Ex, Grid<GridCoordinate1D> **totalEx,
                             bool doNeedEy, Grid<GridCoordinate1D> *Ey, Grid<GridCoordinate1D> **totalEy,
                             bool doNeedEz, Grid<GridCoordinate1D> *Ez, Grid<GridCoordinate1D> **totalEz,
                             bool doNeedHx, Grid<GridCoordinate1D> *Hx, Grid<GridCoordinate1D> **totalHx,
                             bool doNeedHy, Grid<GridCoordinate1D> *Hy, Grid<GridCoordinate1D> **totalHy,
                             bool doNeedHz, Grid<GridCoordinate1D> *Hz, Grid<GridCoordinate1D> **totalHz)
  {
#ifdef PARALLEL_GRID
#ifdef GRID_1D
    if (*totalInitialized)
    {
      if (doNeedEx)
      {
        *totalEx = ((ParallelGrid *) Ex)->gatherFullGridPlacement (*totalEx);
      }
      if (doNeedEy)
      {
        *totalEy = ((ParallelGrid *) Ey)->gatherFullGridPlacement (*totalEy);
      }
      if (doNeedEz)
      {
        *totalEz = ((ParallelGrid *) Ez)->gatherFullGridPlacement (*totalEz);
      }

      if (doNeedHx)
      {
        *totalHx = ((ParallelGrid *) Hx)->gatherFullGridPlacement (*totalHx);
      }
      if (doNeedHy)
      {
        *totalHy = ((ParallelGrid *) Hy)->gatherFullGridPlacement (*totalHy);
      }
      if (doNeedHz)
      {
        *totalHz = ((ParallelGrid *) Hz)->gatherFullGridPlacement (*totalHz);
      }
    }
    else
    {
      if (doNeedEx)
      {
        *totalEx = ((ParallelGrid *) Ex)->gatherFullGrid ();
      }
      if (doNeedEy)
      {
        *totalEy = ((ParallelGrid *) Ey)->gatherFullGrid ();
      }
      if (doNeedEz)
      {
        *totalEz = ((ParallelGrid *) Ez)->gatherFullGrid ();
      }

      if (doNeedHx)
      {
        *totalHx = ((ParallelGrid *) Hx)->gatherFullGrid ();
      }
      if (doNeedHy)
      {
        *totalHy = ((ParallelGrid *) Hy)->gatherFullGrid ();
      }
      if (doNeedHz)
      {
        *totalHz = ((ParallelGrid *) Hz)->gatherFullGrid ();
      }

      *totalInitialized = true;
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                    "Recompile it with -DPARALLEL_GRID_DIMENSION=1.");
#endif
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  static
  void initFullFieldGrids2D (bool *totalInitialized,
                             bool doNeedEx, Grid<GridCoordinate2D> *Ex, Grid<GridCoordinate2D> **totalEx,
                             bool doNeedEy, Grid<GridCoordinate2D> *Ey, Grid<GridCoordinate2D> **totalEy,
                             bool doNeedEz, Grid<GridCoordinate2D> *Ez, Grid<GridCoordinate2D> **totalEz,
                             bool doNeedHx, Grid<GridCoordinate2D> *Hx, Grid<GridCoordinate2D> **totalHx,
                             bool doNeedHy, Grid<GridCoordinate2D> *Hy, Grid<GridCoordinate2D> **totalHy,
                             bool doNeedHz, Grid<GridCoordinate2D> *Hz, Grid<GridCoordinate2D> **totalHz)
  {
#ifdef PARALLEL_GRID
#ifdef GRID_2D
    if (*totalInitialized)
    {
      if (doNeedEx)
      {
        *totalEx = ((ParallelGrid *) Ex)->gatherFullGridPlacement (*totalEx);
      }
      if (doNeedEy)
      {
        *totalEy = ((ParallelGrid *) Ey)->gatherFullGridPlacement (*totalEy);
      }
      if (doNeedEz)
      {
        *totalEz = ((ParallelGrid *) Ez)->gatherFullGridPlacement (*totalEz);
      }

      if (doNeedHx)
      {
        *totalHx = ((ParallelGrid *) Hx)->gatherFullGridPlacement (*totalHx);
      }
      if (doNeedHy)
      {
        *totalHy = ((ParallelGrid *) Hy)->gatherFullGridPlacement (*totalHy);
      }
      if (doNeedHz)
      {
        *totalHz = ((ParallelGrid *) Hz)->gatherFullGridPlacement (*totalHz);
      }
    }
    else
    {
      if (doNeedEx)
      {
        *totalEx = ((ParallelGrid *) Ex)->gatherFullGrid ();
      }
      if (doNeedEy)
      {
        *totalEy = ((ParallelGrid *) Ey)->gatherFullGrid ();
      }
      if (doNeedEz)
      {
        *totalEz = ((ParallelGrid *) Ez)->gatherFullGrid ();
      }

      if (doNeedHx)
      {
        *totalHx = ((ParallelGrid *) Hx)->gatherFullGrid ();
      }
      if (doNeedHy)
      {
        *totalHy = ((ParallelGrid *) Hy)->gatherFullGrid ();
      }
      if (doNeedHz)
      {
        *totalHz = ((ParallelGrid *) Hz)->gatherFullGrid ();
      }

      *totalInitialized = true;
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                    "Recompile it with -DPARALLEL_GRID_DIMENSION=2.");
#endif
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  static
  void initFullFieldGrids3D (bool *totalInitialized,
                             bool doNeedEx, Grid<GridCoordinate3D> *Ex, Grid<GridCoordinate3D> **totalEx,
                             bool doNeedEy, Grid<GridCoordinate3D> *Ey, Grid<GridCoordinate3D> **totalEy,
                             bool doNeedEz, Grid<GridCoordinate3D> *Ez, Grid<GridCoordinate3D> **totalEz,
                             bool doNeedHx, Grid<GridCoordinate3D> *Hx, Grid<GridCoordinate3D> **totalHx,
                             bool doNeedHy, Grid<GridCoordinate3D> *Hy, Grid<GridCoordinate3D> **totalHy,
                             bool doNeedHz, Grid<GridCoordinate3D> *Hz, Grid<GridCoordinate3D> **totalHz)
  {
#ifdef PARALLEL_GRID
#ifdef GRID_3D
    if (*totalInitialized)
    {
      if (doNeedEx)
      {
        *totalEx = ((ParallelGrid *) Ex)->gatherFullGridPlacement (*totalEx);
      }
      if (doNeedEy)
      {
        *totalEy = ((ParallelGrid *) Ey)->gatherFullGridPlacement (*totalEy);
      }
      if (doNeedEz)
      {
        *totalEz = ((ParallelGrid *) Ez)->gatherFullGridPlacement (*totalEz);
      }

      if (doNeedHx)
      {
        *totalHx = ((ParallelGrid *) Hx)->gatherFullGridPlacement (*totalHx);
      }
      if (doNeedHy)
      {
        *totalHy = ((ParallelGrid *) Hy)->gatherFullGridPlacement (*totalHy);
      }
      if (doNeedHz)
      {
        *totalHz = ((ParallelGrid *) Hz)->gatherFullGridPlacement (*totalHz);
      }
    }
    else
    {
      if (doNeedEx)
      {
        *totalEx = ((ParallelGrid *) Ex)->gatherFullGrid ();
      }
      if (doNeedEy)
      {
        *totalEy = ((ParallelGrid *) Ey)->gatherFullGrid ();
      }
      if (doNeedEz)
      {
        *totalEz = ((ParallelGrid *) Ez)->gatherFullGrid ();
      }

      if (doNeedHx)
      {
        *totalHx = ((ParallelGrid *) Hx)->gatherFullGrid ();
      }
      if (doNeedHy)
      {
        *totalHy = ((ParallelGrid *) Hy)->gatherFullGrid ();
      }
      if (doNeedHz)
      {
        *totalHz = ((ParallelGrid *) Hz)->gatherFullGrid ();
      }

      *totalInitialized = true;
    }
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid for this dimension. "
                    "Recompile it with -DPARALLEL_GRID_DIMENSION=3.");
#endif
#else
    ASSERT_MESSAGE ("Solver is not compiled with support of parallel grid. Recompile it with -DPARALLEL_GRID=ON.");
#endif
  }

  static void initSigma (FieldValue *fieldValue, FPValue dist, FPValue boundary, FPValue gridStep)
  {
    FPValue eps0 = PhysicsConst::Eps0;
    FPValue mu0 = PhysicsConst::Mu0;

    uint32_t exponent = 6;
    FPValue R_err = 1e-16;
    FPValue sigma_max_1 = -log (R_err) * (exponent + 1.0) / (2.0 * sqrt (mu0 / eps0) * boundary);
    FPValue boundaryFactor = sigma_max_1 / (gridStep * (pow (boundary, exponent)) * (exponent + 1));

    FPValue x1 = (dist + 1) * gridStep; // upper bounds for point i
    FPValue x2 = dist * gridStep;       // lower bounds for point i

    FPValue val = boundaryFactor * (pow (x1, (exponent + 1)) - pow (x2, (exponent + 1)));    //   polynomial grading
    //FPValue val = sigma_max_1 * pow((dist*gridStep/boundary), exponent);
    //printf("SIGMA: val:%.20f, sigma_max_1:%.20f, dist:%.20f, boundary:%.20f\n", val, sigma_max_1, dist, boundary);

    *fieldValue = getFieldValueRealOnly (val);
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  static
  void initSigmaX (YeeGridLayout<Type, TCoord, layout_type> *layout, FPValue dx, Grid< TCoord<grid_coord, true> > *sigma)
  {
    TCoord<grid_coord, true> PMLSize = layout->getLeftBorderPML () * (layout->getIsDoubleMaterialPrecision () ? 2 : 1);
    FPValue PMLSizeX = FPValue (PMLSize.get1 ());
    FPValue boundary = PMLSizeX * dx;

    for (grid_coord i = 0; i < sigma->getSize ().calculateTotalCoord (); ++i)
    {
      TCoord<grid_coord, true> pos = sigma->calculatePositionFromIndex (i);

      FieldValue valSigma;
      TCoord<FPValue, true> posAbs = layout->getEpsCoordFP (sigma->getTotalPosition (pos));

      TCoord<FPValue, true> size = convertCoord (sigma->getTotalSize ());

      /*
       * TODO: add layout coordinates for material: sigma, eps, etc.
       */
      ASSERT (FPValue (grid_coord (posAbs.get1 () - FPValue (0.5))) == posAbs.get1 () - FPValue (0.5));
      if (posAbs.get1 () < PMLSizeX)
      {
        FPValue dist = (PMLSizeX - posAbs.get1 ());
        SchemeHelper::initSigma (&valSigma, dist, boundary, dx);
      }
      else if (posAbs.get1 () >= size.get1 () - PMLSizeX)
      {
        FPValue dist = (posAbs.get1 () - (size.get1 () - PMLSizeX));
        SchemeHelper::initSigma (&valSigma, dist, boundary, dx);
      }
      else
      {
        valSigma = getFieldValueRealOnly (FPValue (0));
      }

      sigma->setFieldValue (valSigma, pos, 0);
    }
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  static
  void initSigmaY (YeeGridLayout<Type, TCoord, layout_type> *layout, FPValue dx, Grid< TCoord<grid_coord, true> > *sigma)
  {
    TCoord<grid_coord, true> PMLSize = layout->getLeftBorderPML () * (layout->getIsDoubleMaterialPrecision () ? 2 : 1);
    FPValue PMLSizeY = FPValue (PMLSize.get2 ());
    FPValue boundary = PMLSizeY * dx;

    for (grid_coord i = 0; i < sigma->getSize ().calculateTotalCoord (); ++i)
    {
      TCoord<grid_coord, true> pos = sigma->calculatePositionFromIndex (i);

      FieldValue valSigma;
      TCoord<FPValue, true> posAbs = layout->getEpsCoordFP (sigma->getTotalPosition (pos));

      TCoord<FPValue, true> size = layout->getEpsCoordFP (sigma->getTotalSize ());

      /*
       * TODO: add layout coordinates for material: sigma, eps, etc.
       */
      ASSERT (FPValue (grid_coord (posAbs.get2 () - FPValue (0.5))) == posAbs.get2 () - FPValue (0.5));
      if (posAbs.get2 () < PMLSizeY)
      {
        grid_coord dist = (grid_coord) (PMLSizeY - posAbs.get2 ());
        SchemeHelper::initSigma (&valSigma, dist, boundary, dx);
      }
      else if (posAbs.get2 () >= size.get2 () - PMLSizeY)
      {
        grid_coord dist = (grid_coord) (posAbs.get2 () - (size.get2 () - PMLSizeY));
        SchemeHelper::initSigma (&valSigma, dist, boundary, dx);
      }
      else
      {
        valSigma = getFieldValueRealOnly (FPValue (0));
      }

      sigma->setFieldValue (valSigma, pos, 0);
    }
  }

  template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
  static
  void initSigmaZ (YeeGridLayout<Type, TCoord, layout_type> *layout, FPValue dx, Grid< TCoord<grid_coord, true> > *sigma)
  {
    TCoord<grid_coord, true> PMLSize = layout->getLeftBorderPML () * (layout->getIsDoubleMaterialPrecision () ? 2 : 1);
    FPValue PMLSizeZ = FPValue (PMLSize.get3 ());
    FPValue boundary = PMLSizeZ * dx;

    for (grid_coord i = 0; i < sigma->getSize ().calculateTotalCoord (); ++i)
    {
      TCoord<grid_coord, true> pos = sigma->calculatePositionFromIndex (i);

      FieldValue valSigma;
      TCoord<FPValue, true> posAbs = layout->getEpsCoordFP (sigma->getTotalPosition (pos));

      TCoord<FPValue, true> size = layout->getEpsCoordFP (sigma->getTotalSize ());

      /*
       * TODO: add layout coordinates for material: sigma, eps, etc.
       */
      ASSERT (FPValue (grid_coord (posAbs.get3 () - FPValue (0.5))) == posAbs.get3 () - FPValue (0.5));
      if (posAbs.get3 () < PMLSizeZ)
      {
        grid_coord dist = (grid_coord) (PMLSizeZ - posAbs.get3 ());
        SchemeHelper::initSigma (&valSigma, dist, boundary, dx);
      }
      else if (posAbs.get3 () >= size.get3 () - PMLSizeZ)
      {
        grid_coord dist = (grid_coord) (posAbs.get3 () - (size.get3 () - PMLSizeZ));
        SchemeHelper::initSigma (&valSigma, dist, boundary, dx);
      }
      else
      {
        valSigma = getFieldValueRealOnly (FPValue (0));
      }

      sigma->setFieldValue (valSigma, pos, 0);
    }
  }

  template <LayoutType layout_type>
  static
  NPair ntffN3D_x (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *,
                   FPValue, FPValue,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *);
  template <LayoutType layout_type>
  static
  NPair ntffN3D_y (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *,
                   FPValue, FPValue,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *);
  template <LayoutType layout_type>
  static
  NPair ntffN3D_z (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *,
                   FPValue, FPValue,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *);
  template <LayoutType layout_type>
  static
  NPair ntffN3D (FPValue, FPValue,
                 GridCoordinate3D, GridCoordinate3D,
                 YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *,
                 FPValue, FPValue,
                 Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
                 Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
                 Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);

  template <SchemeType_t Type, LayoutType layout_type>
  static
  NPair ntffN2D (FPValue, FPValue,
                 GridCoordinate2D, GridCoordinate2D,
                 YeeGridLayout<Type, GridCoordinate2DTemplate, layout_type> *,
                 FPValue, FPValue,
                 Grid<GridCoordinate2D> *, Grid<GridCoordinate2D> *,
                 Grid<GridCoordinate2D> *, Grid<GridCoordinate2D> *,
                 Grid<GridCoordinate2D> *, Grid<GridCoordinate2D> *)
  {
    UNREACHABLE;
    return NPair ();
  }

  template <SchemeType_t Type, LayoutType layout_type>
  static
  NPair ntffN1D (FPValue, FPValue,
                 GridCoordinate1D, GridCoordinate1D,
                 YeeGridLayout<Type, GridCoordinate1DTemplate, layout_type> *,
                 FPValue, FPValue,
                 Grid<GridCoordinate1D> *, Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate1D> *, Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate1D> *, Grid<GridCoordinate1D> *)
  {
    UNREACHABLE;
    return NPair ();
  }

  template <LayoutType layout_type>
  static
  NPair ntffL3D_x (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *,
                   FPValue, FPValue,
                   Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  template <LayoutType layout_type>
  static
  NPair ntffL3D_y (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *,
                   FPValue, FPValue,
                   Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  template <LayoutType layout_type>
  static
  NPair ntffL3D_z (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *,
                   FPValue, FPValue,
                   Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  template <LayoutType layout_type>
  static
  NPair ntffL3D (FPValue, FPValue,
                 GridCoordinate3D, GridCoordinate3D,
                 YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *,
                 FPValue, FPValue,
                 Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
                 Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
                 Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);

  template <SchemeType_t Type, LayoutType layout_type>
  static
  NPair ntffL2D (FPValue, FPValue,
                 GridCoordinate2D, GridCoordinate2D,
                 YeeGridLayout<Type, GridCoordinate2DTemplate, layout_type> *,
                 FPValue, FPValue,
                 Grid<GridCoordinate2D> *, Grid<GridCoordinate2D> *,
                 Grid<GridCoordinate2D> *, Grid<GridCoordinate2D> *,
                 Grid<GridCoordinate2D> *, Grid<GridCoordinate2D> *)
  {
    UNREACHABLE;
    return NPair ();
  }

  template <SchemeType_t Type, LayoutType layout_type>
  static
  NPair ntffL1D (FPValue, FPValue,
                 GridCoordinate1D, GridCoordinate1D,
                 YeeGridLayout<Type, GridCoordinate1DTemplate, layout_type> *,
                 FPValue, FPValue,
                 Grid<GridCoordinate1D> *, Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate1D> *, Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate1D> *, Grid<GridCoordinate1D> *)
  {
    UNREACHABLE;
    return NPair ();
  }

  template <typename TCoord>
  static grid_coord getStartCoordOrthX (TCoord size)
  {
    return size.get1 () / 2;
  }
  template <typename TCoord>
  static grid_coord getStartCoordOrthY (TCoord size)
  {
    return size.get2 () / 2;
  }
  template <typename TCoord>
  static grid_coord getStartCoordOrthZ (TCoord size)
  {
    return size.get3 () / 2;
  }

  template <typename TCoord>
  static grid_coord getEndCoordOrthX (TCoord size)
  {
    return size.get1 () / 2 + 1;
  }
  template <typename TCoord>
  static grid_coord getEndCoordOrthY (TCoord size)
  {
    return size.get2 () / 2 + 1;
  }
  template <typename TCoord>
  static grid_coord getEndCoordOrthZ (TCoord size)
  {
    return size.get3 () / 2 + 1;
  }

  static bool doSkipMakeScattered1D (GridCoordinateFP1D pos, GridCoordinate1D left, GridCoordinate1D right)
  {
    GridCoordinateFP1D leftTFSF = convertCoord (left);
    GridCoordinateFP1D rightTFSF = convertCoord (right);
    return pos.get1 () < leftTFSF.get1 () || pos.get1 () > rightTFSF.get1 ();
  }
  static bool doSkipMakeScattered2D (GridCoordinateFP2D pos, GridCoordinate2D left, GridCoordinate2D right)
  {
    GridCoordinateFP2D leftTFSF = convertCoord (left);
    GridCoordinateFP2D rightTFSF = convertCoord (right);
    return pos.get1 () < leftTFSF.get1 () || pos.get1 () > rightTFSF.get1 ()
           || pos.get2 () < leftTFSF.get2 () || pos.get2 () > rightTFSF.get2 ();
  }
  static bool doSkipMakeScattered3D (GridCoordinateFP3D pos, GridCoordinate3D left, GridCoordinate3D right)
  {
    GridCoordinateFP3D leftTFSF = convertCoord (left);
    GridCoordinateFP3D rightTFSF = convertCoord (right);
    return pos.get1 () < leftTFSF.get1 () || pos.get1 () > rightTFSF.get1 ()
           || pos.get2 () < leftTFSF.get2 () || pos.get2 () > rightTFSF.get2 ()
           || pos.get3 () < leftTFSF.get3 () || pos.get3 () > rightTFSF.get3 ();
  }

  static GridCoordinate1D getStartCoordRes1D (OrthogonalAxis orthogonalAxis, GridCoordinate1D start, GridCoordinate1D size)
  {
    return start;
  }
  static GridCoordinate1D getEndCoordRes1D (OrthogonalAxis orthogonalAxis, GridCoordinate1D end, GridCoordinate1D size)
  {
    return end;
  }

  static GridCoordinate2D getStartCoordRes2D (OrthogonalAxis orthogonalAxis, GridCoordinate2D start, GridCoordinate2D size)
  {
    return start;
  }
  static GridCoordinate2D getEndCoordRes2D (OrthogonalAxis orthogonalAxis, GridCoordinate2D end, GridCoordinate2D size)
  {
    return end;
  }

  static GridCoordinate3D getStartCoordRes3D (OrthogonalAxis orthogonalAxis, GridCoordinate3D start, GridCoordinate3D size)
  {
    if (orthogonalAxis == OrthogonalAxis::Z)
    {
      return GridCoordinate3D (start.get1 (), start.get2 (), SchemeHelper::getStartCoordOrthZ (size)
#ifdef DEBUG_INFO
             , start.getType1 (), start.getType2 (), start.getType3 ()
#endif
             );
    }
    else if (orthogonalAxis == OrthogonalAxis::Y)
    {
      return GridCoordinate3D (start.get1 (), SchemeHelper::getStartCoordOrthY (size), start.get3 ()
#ifdef DEBUG_INFO
             , start.getType1 (), start.getType2 (), start.getType3 ()
#endif
             );
    }
    else if (orthogonalAxis == OrthogonalAxis::X)
    {
      return GridCoordinate3D (SchemeHelper::getStartCoordOrthX (size), start.get2 (), start.get3 ()
#ifdef DEBUG_INFO
             , start.getType1 (), start.getType2 (), start.getType3 ()
#endif
             );
    }

    UNREACHABLE;
    return GRID_COORDINATE_3D (0, 0, 0, start.getType1 (), start.getType2 (), start.getType3 ());
  }
  static GridCoordinate3D getEndCoordRes3D (OrthogonalAxis orthogonalAxis, GridCoordinate3D end, GridCoordinate3D size)
  {
    if (orthogonalAxis == OrthogonalAxis::Z)
    {
      return GridCoordinate3D (end.get1 (), end.get2 (), SchemeHelper::getEndCoordOrthZ (size)
#ifdef DEBUG_INFO
             , end.getType1 (), end.getType2 (), end.getType3 ()
#endif
             );
    }
    else if (orthogonalAxis == OrthogonalAxis::Y)
    {
      return GridCoordinate3D (end.get1 (), SchemeHelper::getEndCoordOrthY (size), end.get3 ()
#ifdef DEBUG_INFO
             , end.getType1 (), end.getType2 (), end.getType3 ()
#endif
             );
    }
    else if (orthogonalAxis == OrthogonalAxis::X)
    {
      return GridCoordinate3D (SchemeHelper::getEndCoordOrthX (size), end.get2 (), end.get3 ()
#ifdef DEBUG_INFO
             , end.getType1 (), end.getType2 (), end.getType3 ()
#endif
             );
    }

    UNREACHABLE;
    return GRID_COORDINATE_3D (0, 0, 0, end.getType1 (), end.getType2 (), end.getType3 ());
  }

#ifdef PARALLEL_GRID

  template <SchemeType_t Type, LayoutType layout_type>
  static void initParallelBlocks1D (Scheme<Type, GridCoordinate1DTemplate, layout_type> *scheme)
  {
#ifndef GRID_1D
    ALWAYS_ASSERT_MESSAGE ("Solver is not compiled with support of 1D parallel grids.");
#else /* !GRID_1D */
    ParallelYeeGridLayout<Type, layout_type> *parallelYeeLayout = (ParallelYeeGridLayout<Type, layout_type> *) scheme->yeeLayout;
    scheme->blockSize = parallelYeeLayout->getSizeForCurNode ();
#endif /* GRID_1D */
  }

  template <SchemeType_t Type, LayoutType layout_type>
  static void initParallelBlocks2D (Scheme<Type, GridCoordinate2DTemplate, layout_type> *scheme)
  {
#ifndef GRID_2D
    ALWAYS_ASSERT_MESSAGE ("Solver is not compiled with support of 2D parallel grids.");
#else /* !GRID_2D */
    ParallelYeeGridLayout<Type, layout_type> *parallelYeeLayout = (ParallelYeeGridLayout<Type, layout_type> *) scheme->yeeLayout;
    scheme->blockSize = parallelYeeLayout->getSizeForCurNode ();
#endif /* GRID_2D */
  }

  template <SchemeType_t Type, LayoutType layout_type>
  static void initParallelBlocks3D (Scheme<Type, GridCoordinate3DTemplate, layout_type> *scheme)
  {
#ifndef GRID_3D
    ALWAYS_ASSERT_MESSAGE ("Solver is not compiled with support of 3D parallel grids.");
#else /* !GRID_3D */
    ParallelYeeGridLayout<Type, layout_type> *parallelYeeLayout = (ParallelYeeGridLayout<Type, layout_type> *) scheme->yeeLayout;
    scheme->blockSize = parallelYeeLayout->getSizeForCurNode ();
#endif /* GRID_3D */
  }

#endif /* PARALLEL_GRID */
};


/**
 * Compute N for +-x0 on time step t+0.5 (i.e. E is used as is, H as is averaged for t and t+1)
 */
template <LayoutType layout_type>
NPair
SchemeHelper::ntffN3D_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate3D> *curEx,
                         Grid<GridCoordinate3D> *curEy,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHx,
                         Grid<GridCoordinate3D> *curHy,
                         Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (yeeLayout->getSize ().get1 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get2 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get3 () % 2 == 0);
  FPValue diffx0 = yeeLayout->getSize ().get1 () / 2;
  FPValue diffy0 = yeeLayout->getSize ().get2 () / 2;
  FPValue diffz0 = yeeLayout->getSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart = GRID_COORDINATE_FP_3D (x0, leftNTFF.get2 () + 0.5, leftNTFF.get3 () + 0.5, ct1, ct2, ct3);
  GridCoordinateFP3D coordEnd = GRID_COORDINATE_FP_3D (x0, rightNTFF.get2 () - 0.5, rightNTFF.get3 () - 0.5, ct1, ct2, ct3);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      FieldValue Hz1;
      FieldValue Hz2;
      FieldValue Hy1;
      FieldValue Hy2;

      if (layout_type == E_CENTERED)
      {
        GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (x0, coordY - 0.5, coordZ, ct1, ct2, ct3);
        GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (x0, coordY + 0.5, coordZ, ct1, ct2, ct3);
        GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (x0, coordY, coordZ - 0.5, ct1, ct2, ct3);
        GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (x0, coordY, coordZ + 0.5, ct1, ct2, ct3);

        pos1 = pos1 - yeeLayout->getMinHzCoordFP ();
        pos2 = pos2 - yeeLayout->getMinHzCoordFP ();

        pos3 = pos3 - yeeLayout->getMinHyCoordFP ();
        pos4 = pos4 - yeeLayout->getMinHyCoordFP ();

        GridCoordinate3D pos11 = convertCoord (pos1);
        GridCoordinate3D pos21 = convertCoord (pos2);
        GridCoordinate3D pos31 = convertCoord (pos3);
        GridCoordinate3D pos41 = convertCoord (pos4);

        FieldValue *valHz1 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
        FieldValue *valHz2 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);

        FieldValue *valHy1 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
        FieldValue *valHy2 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
        if (valHz1 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos11)
            || valHz2 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos21)
            || valHy1 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos31)
            || valHy2 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos41))
        {
          continue;
        }
#endif

        FieldValue *valHz1_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos11);
        FieldValue *valHz2_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos21);

        FieldValue *valHy1_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos31);
        FieldValue *valHy2_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos41);

        ASSERT (valHz1 != NULLPTR && valHz2 != NULLPTR && valHy1 != NULLPTR && valHy2 != NULLPTR
                && valHz1_prev != NULLPTR && valHz2_prev != NULLPTR && valHy1_prev != NULLPTR && valHy2_prev != NULLPTR);

        Hz1 = (*valHz1 + *valHz1_prev) / FPValue (2);
        Hz2 = (*valHz2 + *valHz2_prev) / FPValue (2);
        Hy1 = (*valHy1 + *valHy1_prev) / FPValue (2);
        Hy2 = (*valHy2 + *valHy2_prev) / FPValue (2);
      }
      else if (layout_type == H_CENTERED)
      {
        GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (x0, coordY, coordZ - 0.5, ct1, ct2, ct3);
        GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (x0, coordY, coordZ + 0.5, ct1, ct2, ct3);
        GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (x0, coordY - 0.5, coordZ, ct1, ct2, ct3);
        GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (x0, coordY + 0.5, coordZ, ct1, ct2, ct3);

        pos1 = pos1 - yeeLayout->getMinHzCoordFP ();
        pos2 = pos2 - yeeLayout->getMinHzCoordFP ();

        pos3 = pos3 - yeeLayout->getMinHyCoordFP ();
        pos4 = pos4 - yeeLayout->getMinHyCoordFP ();

        GridCoordinate3D pos11 = convertCoord (pos1 - GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
        GridCoordinate3D pos12 = convertCoord (pos1 + GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
        GridCoordinate3D pos21 = convertCoord (pos2 - GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
        GridCoordinate3D pos22 = convertCoord (pos2 + GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));

        GridCoordinate3D pos31 = convertCoord (pos3 - GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
        GridCoordinate3D pos32 = convertCoord (pos3 + GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
        GridCoordinate3D pos41 = convertCoord (pos4 - GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
        GridCoordinate3D pos42 = convertCoord (pos4 + GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));

        FieldValue *valHz11 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
        FieldValue *valHz12 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos12);
        FieldValue *valHz21 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);
        FieldValue *valHz22 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos22);

        FieldValue *valHy11 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
        FieldValue *valHy12 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos32);
        FieldValue *valHy21 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);
        FieldValue *valHy22 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
        if (valHz11 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos11)
            || valHz12 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos11)
            || valHz21 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos21)
            || valHz22 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos22)
            || valHy11 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos31)
            || valHy12 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos32)
            || valHy21 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos41)
            || valHy22 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos42))
        {
          continue;
        }
#endif

        FieldValue *valHz11_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos11);
        FieldValue *valHz12_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos12);
        FieldValue *valHz21_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos21);
        FieldValue *valHz22_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos22);

        FieldValue *valHy11_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos31);
        FieldValue *valHy12_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos32);
        FieldValue *valHy21_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos41);
        FieldValue *valHy22_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos42);

        ASSERT (valHz11 != NULLPTR && valHz12 != NULLPTR && valHz21 != NULLPTR && valHz22 != NULLPTR
                && valHy11 != NULLPTR && valHy12 != NULLPTR && valHy21 != NULLPTR && valHy22 != NULLPTR
                && valHz11_prev != NULLPTR && valHz12_prev != NULLPTR && valHz21_prev != NULLPTR && valHz22_prev != NULLPTR
                && valHy11_prev != NULLPTR && valHy12_prev != NULLPTR && valHy21_prev != NULLPTR && valHy22_prev != NULLPTR);

        Hz1 = (*valHz11 + *valHz12 + *valHz11_prev + *valHz12_prev) / FPValue(4.0);
        Hz2 = (*valHz21 + *valHz22 + *valHz21_prev + *valHz22_prev) / FPValue(4.0);
        Hy1 = (*valHy11 + *valHy12 + *valHy11_prev + *valHy12_prev) / FPValue(4.0);
        Hy2 = (*valHy21 + *valHy22 + *valHy21_prev + *valHy22_prev) / FPValue(4.0);
      }
      else
      {
        UNREACHABLE;
      }

      FPValue arg = (x0 - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += ((Hz1 + Hz2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))
                   + (Hy1 + Hy2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent * SQR (gridStep) * (-1) * (x0==rightNTFF.get1 ()?1:-1);

      sum_phi += ((Hz1 + Hz2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent * SQR (gridStep) * (-1) * (x0==rightNTFF.get1 ()?1:-1);
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

template <LayoutType layout_type>
NPair
SchemeHelper::ntffN3D_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate3D> *curEx,
                         Grid<GridCoordinate3D> *curEy,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHx,
                         Grid<GridCoordinate3D> *curHy,
                         Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (yeeLayout->getSize ().get1 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get2 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get3 () % 2 == 0);
  FPValue diffx0 = yeeLayout->getSize ().get1 () / 2;
  FPValue diffy0 = yeeLayout->getSize ().get2 () / 2;
  FPValue diffz0 = yeeLayout->getSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart = GRID_COORDINATE_FP_3D (leftNTFF.get1 () + 0.5, y0, leftNTFF.get3 () + 0.5, ct1, ct2, ct3);
  GridCoordinateFP3D coordEnd = GRID_COORDINATE_FP_3D (rightNTFF.get1 () - 0.5, y0, rightNTFF.get3 () - 0.5, ct1, ct2, ct3);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      FieldValue Hz1;
      FieldValue Hz2;
      FieldValue Hx1;
      FieldValue Hx2;

      if (layout_type == E_CENTERED)
      {
        GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (coordX - 0.5, y0, coordZ, ct1, ct2, ct3);
        GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (coordX + 0.5, y0, coordZ, ct1, ct2, ct3);
        GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (coordX, y0, coordZ - 0.5, ct1, ct2, ct3);
        GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (coordX, y0, coordZ + 0.5, ct1, ct2, ct3);

        pos1 = pos1 - yeeLayout->getMinHzCoordFP ();
        pos2 = pos2 - yeeLayout->getMinHzCoordFP ();

        pos3 = pos3 - yeeLayout->getMinHxCoordFP ();
        pos4 = pos4 - yeeLayout->getMinHxCoordFP ();

        GridCoordinate3D pos11 = convertCoord (pos1);
        GridCoordinate3D pos21 = convertCoord (pos2);
        GridCoordinate3D pos31 = convertCoord (pos3);
        GridCoordinate3D pos41 = convertCoord (pos4);

        FieldValue *valHz1 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
        FieldValue *valHz2 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);

        FieldValue *valHx1 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
        FieldValue *valHx2 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
        if (valHz1 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos11)
            || valHz2 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos21)
            || valHx1 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos31)
            || valHx2 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos41))
        {
          continue;
        }
#endif

        FieldValue *valHz1_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos11);
        FieldValue *valHz2_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos21);

        FieldValue *valHx1_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos31);
        FieldValue *valHx2_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos41);

        ASSERT (valHz1 != NULLPTR && valHz2 != NULLPTR && valHx1 != NULLPTR && valHx2 != NULLPTR
                && valHz1_prev != NULLPTR && valHz2_prev != NULLPTR && valHx1_prev != NULLPTR && valHx2_prev != NULLPTR);

        Hz1 = (*valHz1 + *valHz1_prev) / FPValue (2);
        Hz2 = (*valHz2 + *valHz2_prev) / FPValue (2);
        Hx1 = (*valHx1 + *valHx1_prev) / FPValue (2);
        Hx2 = (*valHx2 + *valHx2_prev) / FPValue (2);
      }
      else if (layout_type == H_CENTERED)
      {
        GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (coordX, y0, coordZ - 0.5, ct1, ct2, ct3);
        GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (coordX, y0, coordZ + 0.5, ct1, ct2, ct3);
        GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (coordX - 0.5, y0, coordZ, ct1, ct2, ct3);
        GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (coordX + 0.5, y0, coordZ, ct1, ct2, ct3);

        pos1 = pos1 - yeeLayout->getMinHzCoordFP ();
        pos2 = pos2 - yeeLayout->getMinHzCoordFP ();

        pos3 = pos3 - yeeLayout->getMinHxCoordFP ();
        pos4 = pos4 - yeeLayout->getMinHxCoordFP ();

        GridCoordinate3D pos11 = convertCoord (pos1 - GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
        GridCoordinate3D pos12 = convertCoord (pos1 + GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
        GridCoordinate3D pos21 = convertCoord (pos2 - GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
        GridCoordinate3D pos22 = convertCoord (pos2 + GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));

        GridCoordinate3D pos31 = convertCoord (pos3 - GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
        GridCoordinate3D pos32 = convertCoord (pos3 + GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
        GridCoordinate3D pos41 = convertCoord (pos4 - GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
        GridCoordinate3D pos42 = convertCoord (pos4 + GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));

        FieldValue *valHz11 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
        FieldValue *valHz12 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos12);
        FieldValue *valHz21 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);
        FieldValue *valHz22 = curHz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos22);

        FieldValue *valHx11 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
        FieldValue *valHx12 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos32);
        FieldValue *valHx21 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);
        FieldValue *valHx22 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
        if (valHz11 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos11)
            || valHz12 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos11)
            || valHz21 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos21)
            || valHz22 == NULLPTR || ((ParallelGrid*) curHz)->isBufferLeftPosition (pos22)
            || valHx11 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos31)
            || valHx12 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos32)
            || valHx21 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos41)
            || valHx22 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos42))
        {
          continue;
        }
#endif

        FieldValue *valHz11_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos11);
        FieldValue *valHz12_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos12);
        FieldValue *valHz21_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos21);
        FieldValue *valHz22_prev = curHz->getFieldValuePreviousAfterShiftByAbsolutePos (pos22);

        FieldValue *valHx11_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos31);
        FieldValue *valHx12_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos32);
        FieldValue *valHx21_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos41);
        FieldValue *valHx22_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos42);

        ASSERT (valHz11 != NULLPTR && valHz12 != NULLPTR && valHz21 != NULLPTR && valHz22 != NULLPTR
                && valHx11 != NULLPTR && valHx12 != NULLPTR && valHx21 != NULLPTR && valHx22 != NULLPTR
                && valHz11_prev != NULLPTR && valHz12_prev != NULLPTR && valHz21_prev != NULLPTR && valHz22_prev != NULLPTR
                && valHx11_prev != NULLPTR && valHx12_prev != NULLPTR && valHx21_prev != NULLPTR && valHx22_prev != NULLPTR);

        Hz1 = (*valHz11 + *valHz12 + *valHz11_prev + *valHz12_prev) / FPValue(4.0);
        Hz2 = (*valHz21 + *valHz22 + *valHz21_prev + *valHz22_prev) / FPValue(4.0);
        Hx1 = (*valHx11 + *valHx12 + *valHx11_prev + *valHx12_prev) / FPValue(4.0);
        Hx2 = (*valHx21 + *valHx22 + *valHx21_prev + *valHx22_prev) / FPValue(4.0);
      }
      else
      {
        UNREACHABLE;
      }

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (y0 - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += ((Hz1 + Hz2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                   + (Hx1 + Hx2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent * SQR (gridStep) * (y0==rightNTFF.get2 ()?1:-1);

      sum_phi += ((Hz1 + Hz2)/FPValue(2.0) * FPValue (sin (anglePhi))) * exponent * SQR (gridStep) * (-1) * (y0==rightNTFF.get2 ()?1:-1);
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

template <LayoutType layout_type>
NPair
SchemeHelper::ntffN3D_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate3D> *curEx,
                         Grid<GridCoordinate3D> *curEy,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHx,
                         Grid<GridCoordinate3D> *curHy,
                         Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (yeeLayout->getSize ().get1 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get2 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get3 () % 2 == 0);
  FPValue diffx0 = yeeLayout->getSize ().get1 () / 2;
  FPValue diffy0 = yeeLayout->getSize ().get2 () / 2;
  FPValue diffz0 = yeeLayout->getSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart = GRID_COORDINATE_FP_3D (leftNTFF.get1 () + 0.5, leftNTFF.get2 () + 0.5, z0, ct1, ct2, ct3);
  GridCoordinateFP3D coordEnd = GRID_COORDINATE_FP_3D (rightNTFF.get1 () - 0.5, rightNTFF.get2 () - 0.5, z0, ct1, ct2, ct3);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
    {
      FieldValue Hy1;
      FieldValue Hy2;
      FieldValue Hx1;
      FieldValue Hx2;

      if (layout_type == E_CENTERED)
      {
        GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (coordX - 0.5, coordY, z0, ct1, ct2, ct3);
        GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (coordX + 0.5, coordY, z0, ct1, ct2, ct3);
        GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (coordX, coordY - 0.5, z0, ct1, ct2, ct3);
        GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (coordX, coordY + 0.5, z0, ct1, ct2, ct3);

        pos1 = pos1 - yeeLayout->getMinHyCoordFP ();
        pos2 = pos2 - yeeLayout->getMinHyCoordFP ();

        pos3 = pos3 - yeeLayout->getMinHxCoordFP ();
        pos4 = pos4 - yeeLayout->getMinHxCoordFP ();

        GridCoordinate3D pos11 = convertCoord (pos1);
        GridCoordinate3D pos21 = convertCoord (pos2);
        GridCoordinate3D pos31 = convertCoord (pos3);
        GridCoordinate3D pos41 = convertCoord (pos4);

        FieldValue *valHy1 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
        FieldValue *valHy2 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);

        FieldValue *valHx1 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
        FieldValue *valHx2 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
        if (valHy1 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos11)
            || valHy2 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos21)
            || valHx1 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos31)
            || valHx2 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos41))
        {
          continue;
        }
#endif

        FieldValue *valHy1_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos11);
        FieldValue *valHy2_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos21);

        FieldValue *valHx1_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos31);
        FieldValue *valHx2_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos41);

        ASSERT (valHy1 != NULLPTR && valHy2 != NULLPTR && valHx1 != NULLPTR && valHx2 != NULLPTR
                && valHy1_prev != NULLPTR && valHy2_prev != NULLPTR && valHx1_prev != NULLPTR && valHx2_prev != NULLPTR);

        Hy1 = (*valHy1 + *valHy1_prev) / FPValue (2);
        Hy2 = (*valHy2 + *valHy2_prev) / FPValue (2);
        Hx1 = (*valHx1 + *valHx1_prev) / FPValue (2);
        Hx2 = (*valHx2 + *valHx2_prev) / FPValue (2);
      }
      else if (layout_type == H_CENTERED)
      {
        GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (coordX, coordY - 0.5, z0, ct1, ct2, ct3);
        GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (coordX, coordY - 0.5, z0, ct1, ct2, ct3);
        GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (coordX - 0.5, coordY, z0, ct1, ct2, ct3);
        GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (coordX + 0.5, coordY, z0, ct1, ct2, ct3);

        pos1 = pos1 - yeeLayout->getMinHyCoordFP ();
        pos2 = pos2 - yeeLayout->getMinHyCoordFP ();

        pos3 = pos3 - yeeLayout->getMinHxCoordFP ();
        pos4 = pos4 - yeeLayout->getMinHxCoordFP ();

        GridCoordinate3D pos11 = convertCoord (pos1 - GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
        GridCoordinate3D pos12 = convertCoord (pos1 + GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
        GridCoordinate3D pos21 = convertCoord (pos2 - GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
        GridCoordinate3D pos22 = convertCoord (pos2 + GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));

        GridCoordinate3D pos31 = convertCoord (pos3 - GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
        GridCoordinate3D pos32 = convertCoord (pos3 + GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
        GridCoordinate3D pos41 = convertCoord (pos4 - GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
        GridCoordinate3D pos42 = convertCoord (pos4 + GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));

        FieldValue *valHy11 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
        FieldValue *valHy12 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos12);
        FieldValue *valHy21 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);
        FieldValue *valHy22 = curHy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos22);

        FieldValue *valHx11 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
        FieldValue *valHx12 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos32);
        FieldValue *valHx21 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);
        FieldValue *valHx22 = curHx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
        if (valHy11 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos11)
            || valHy12 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos11)
            || valHy21 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos21)
            || valHy22 == NULLPTR || ((ParallelGrid*) curHy)->isBufferLeftPosition (pos22)
            || valHx11 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos31)
            || valHx12 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos32)
            || valHx21 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos41)
            || valHx22 == NULLPTR || ((ParallelGrid*) curHx)->isBufferLeftPosition (pos42))
        {
          continue;
        }
#endif

        FieldValue *valHy11_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos11);
        FieldValue *valHy12_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos12);
        FieldValue *valHy21_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos21);
        FieldValue *valHy22_prev = curHy->getFieldValuePreviousAfterShiftByAbsolutePos (pos22);

        FieldValue *valHx11_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos31);
        FieldValue *valHx12_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos32);
        FieldValue *valHx21_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos41);
        FieldValue *valHx22_prev = curHx->getFieldValuePreviousAfterShiftByAbsolutePos (pos42);

        ASSERT (valHy11 != NULLPTR && valHy12 != NULLPTR && valHy21 != NULLPTR && valHy22 != NULLPTR
                && valHx11 != NULLPTR && valHx12 != NULLPTR && valHx21 != NULLPTR && valHx22 != NULLPTR
                && valHy11_prev != NULLPTR && valHy12_prev != NULLPTR && valHy21_prev != NULLPTR && valHy22_prev != NULLPTR
                && valHx11_prev != NULLPTR && valHx12_prev != NULLPTR && valHx21_prev != NULLPTR && valHx22_prev != NULLPTR);

        Hy1 = (*valHy11 + *valHy12 + *valHy11_prev + *valHy12_prev) / FPValue(4.0);
        Hy2 = (*valHy21 + *valHy22 + *valHy21_prev + *valHy22_prev) / FPValue(4.0);
        Hx1 = (*valHx11 + *valHx12 + *valHx11_prev + *valHx12_prev) / FPValue(4.0);
        Hx2 = (*valHx21 + *valHx22 + *valHx21_prev + *valHx22_prev) / FPValue(4.0);
      }
      else
      {
        UNREACHABLE;
      }

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (z0 - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += (-(Hy1 + Hy2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                   + (Hx1 + Hx2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))) * exponent * SQR (gridStep) * (z0==rightNTFF.get3 ()?1:-1);

      sum_phi += ((Hy1 + Hy2)/FPValue(2.0) * FPValue (sin (anglePhi))
                  + (Hx1 + Hx2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent * SQR (gridStep) * (z0==rightNTFF.get3 ()?1:-1);
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

template <LayoutType layout_type>
NPair
SchemeHelper::ntffL3D_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate3D> *curEx,
                         Grid<GridCoordinate3D> *curEy,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHx,
                         Grid<GridCoordinate3D> *curHy,
                         Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (yeeLayout->getSize ().get1 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get2 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get3 () % 2 == 0);
  FPValue diffx0 = yeeLayout->getSize ().get1 () / 2;
  FPValue diffy0 = yeeLayout->getSize ().get2 () / 2;
  FPValue diffz0 = yeeLayout->getSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart = GRID_COORDINATE_FP_3D (x0, leftNTFF.get2 () + 0.5, leftNTFF.get3 () + 0.5, ct1, ct2, ct3);
  GridCoordinateFP3D coordEnd = GRID_COORDINATE_FP_3D (x0, rightNTFF.get2 () - 0.5, rightNTFF.get3 () - 0.5, ct1, ct2, ct3);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      FieldValue Ey1;
      FieldValue Ey2;
      FieldValue Ez1;
      FieldValue Ez2;

      if (layout_type == E_CENTERED)
      {
        GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (x0, coordY - 0.5, coordZ, ct1, ct2, ct3);
        GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (x0, coordY + 0.5, coordZ, ct1, ct2, ct3);
        GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (x0, coordY, coordZ - 0.5, ct1, ct2, ct3);
        GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (x0, coordY, coordZ + 0.5, ct1, ct2, ct3);

        pos1 = pos1 - yeeLayout->getMinEyCoordFP ();
        pos2 = pos2 - yeeLayout->getMinEyCoordFP ();

        pos3 = pos3 - yeeLayout->getMinEzCoordFP ();
        pos4 = pos4 - yeeLayout->getMinEzCoordFP ();

        GridCoordinate3D pos11 = convertCoord (pos1 - GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
        GridCoordinate3D pos12 = convertCoord (pos1 + GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
        GridCoordinate3D pos21 = convertCoord (pos2 - GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
        GridCoordinate3D pos22 = convertCoord (pos2 + GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));

        GridCoordinate3D pos31 = convertCoord (pos3 - GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
        GridCoordinate3D pos32 = convertCoord (pos3 + GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
        GridCoordinate3D pos41 = convertCoord (pos4 - GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));
        GridCoordinate3D pos42 = convertCoord (pos4 + GRID_COORDINATE_FP_3D (0.5, 0, 0, ct1, ct2, ct3));

        FieldValue *valEy11 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
        FieldValue *valEy12 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos12);
        FieldValue *valEy21 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);
        FieldValue *valEy22 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos22);

        FieldValue *valEz11 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
        FieldValue *valEz12 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos32);
        FieldValue *valEz21 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);
        FieldValue *valEz22 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
        if (valEy11 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos11)
            || valEy12 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos11)
            || valEy21 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos21)
            || valEy22 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos22)
            || valEz11 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos31)
            || valEz12 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos32)
            || valEz21 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos41)
            || valEz22 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos42))
        {
          continue;
        }
#endif

        ASSERT (valEy11 != NULLPTR && valEy12 != NULLPTR && valEy21 != NULLPTR && valEy22 != NULLPTR
                && valEz11 != NULLPTR && valEz12 != NULLPTR && valEz21 != NULLPTR && valEz22 != NULLPTR);

        Ey1 = (*valEy11 + *valEy12) / FPValue(2.0);
        Ey2 = (*valEy21 + *valEy22) / FPValue(2.0);
        Ez1 = (*valEz11 + *valEz12) / FPValue(2.0);
        Ez2 = (*valEz21 + *valEz22) / FPValue(2.0);
      }
      else if (layout_type == H_CENTERED)
      {
        GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (x0, coordY, coordZ - 0.5, ct1, ct2, ct3);
        GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (x0, coordY, coordZ + 0.5, ct1, ct2, ct3);
        GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (x0, coordY - 0.5, coordZ, ct1, ct2, ct3);
        GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (x0, coordY + 0.5, coordZ, ct1, ct2, ct3);

        pos1 = pos1 - yeeLayout->getMinEyCoordFP ();
        pos2 = pos2 - yeeLayout->getMinEyCoordFP ();

        pos3 = pos3 - yeeLayout->getMinEzCoordFP ();
        pos4 = pos4 - yeeLayout->getMinEzCoordFP ();

        GridCoordinate3D pos11 = convertCoord (pos1);
        GridCoordinate3D pos21 = convertCoord (pos2);

        GridCoordinate3D pos31 = convertCoord (pos3);
        GridCoordinate3D pos41 = convertCoord (pos4);

        FieldValue *valEy11 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
        FieldValue *valEy21 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);

        FieldValue *valEz11 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
        FieldValue *valEz21 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
        if (valEy11 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos11)
            || valEy21 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos21)
            || valEz11 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos31)
            || valEz21 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos41))
        {
          continue;
        }
#endif

        ASSERT (valEy11 != NULLPTR && valEy21 != NULLPTR && valEz11 != NULLPTR && valEz21 != NULLPTR);

        Ey1 = (*valEy11);
        Ey2 = (*valEy21);
        Ez1 = (*valEz11);
        Ez2 = (*valEz21);
      }
      else
      {
        UNREACHABLE;
      }

      FPValue arg = (x0 - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))
                   + (Ey1 + Ey2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent * SQR (gridStep) * (x0==rightNTFF.get1 ()?1:-1);

      sum_phi += ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent * SQR (gridStep) * (x0==rightNTFF.get1 ()?1:-1);
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

template <LayoutType layout_type>
NPair
SchemeHelper::ntffL3D_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate3D> *curEx,
                         Grid<GridCoordinate3D> *curEy,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHx,
                         Grid<GridCoordinate3D> *curHy,
                         Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (yeeLayout->getSize ().get1 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get2 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get3 () % 2 == 0);
  FPValue diffx0 = yeeLayout->getSize ().get1 () / 2;
  FPValue diffy0 = yeeLayout->getSize ().get2 () / 2;
  FPValue diffz0 = yeeLayout->getSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart = GRID_COORDINATE_FP_3D (leftNTFF.get1 () + 0.5, y0, leftNTFF.get3 () + 0.5, ct1, ct2, ct3);
  GridCoordinateFP3D coordEnd = GRID_COORDINATE_FP_3D (rightNTFF.get1 () - 0.5, y0, rightNTFF.get3 () - 0.5, ct1, ct2, ct3);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordZ = coordStart.get3 (); coordZ <= coordEnd.get3 (); ++coordZ)
    {
      FieldValue Ex1;
      FieldValue Ex2;
      FieldValue Ez1;
      FieldValue Ez2;

      if (layout_type == E_CENTERED)
      {
        GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (coordX - 0.5, y0, coordZ, ct1, ct2, ct3);
        GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (coordX + 0.5, y0, coordZ, ct1, ct2, ct3);
        GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (coordX, y0, coordZ - 0.5, ct1, ct2, ct3);
        GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (coordX, y0, coordZ + 0.5, ct1, ct2, ct3);

        pos1 = pos1 - yeeLayout->getMinExCoordFP ();
        pos2 = pos2 - yeeLayout->getMinExCoordFP ();

        pos3 = pos3 - yeeLayout->getMinEzCoordFP ();
        pos4 = pos4 - yeeLayout->getMinEzCoordFP ();

        GridCoordinate3D pos11 = convertCoord (pos1 - GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
        GridCoordinate3D pos12 = convertCoord (pos1 + GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
        GridCoordinate3D pos21 = convertCoord (pos2 - GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
        GridCoordinate3D pos22 = convertCoord (pos2 + GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));

        GridCoordinate3D pos31 = convertCoord (pos3 - GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
        GridCoordinate3D pos32 = convertCoord (pos3 + GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
        GridCoordinate3D pos41 = convertCoord (pos4 - GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));
        GridCoordinate3D pos42 = convertCoord (pos4 + GRID_COORDINATE_FP_3D (0, 0.5, 0, ct1, ct2, ct3));

        FieldValue *valEx11 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
        FieldValue *valEx12 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos12);
        FieldValue *valEx21 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);
        FieldValue *valEx22 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos22);

        FieldValue *valEz11 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
        FieldValue *valEz12 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos32);
        FieldValue *valEz21 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);
        FieldValue *valEz22 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
        if (valEx11 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos11)
            || valEx12 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos12)
            || valEx21 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos21)
            || valEx22 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos22)
            || valEz11 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos31)
            || valEz12 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos32)
            || valEz21 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos41)
            || valEz22 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos42))
        {
          continue;
        }
#endif

        ASSERT (valEx11 != NULLPTR && valEx12 != NULLPTR && valEx21 != NULLPTR && valEx22 != NULLPTR
                && valEz11 != NULLPTR && valEz12 != NULLPTR && valEz21 != NULLPTR && valEz22 != NULLPTR);

        Ex1 = (*valEx11 + *valEx12) / FPValue(2.0);
        Ex2 = (*valEx21 + *valEx22) / FPValue(2.0);
        Ez1 = (*valEz11 + *valEz12) / FPValue(2.0);
        Ez2 = (*valEz21 + *valEz22) / FPValue(2.0);
      }
      else if (layout_type == H_CENTERED)
      {
        GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (coordX, y0, coordZ - 0.5, ct1, ct2, ct3);
        GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (coordX, y0, coordZ + 0.5, ct1, ct2, ct3);
        GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (coordX - 0.5, y0, coordZ, ct1, ct2, ct3);
        GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (coordX + 0.5, y0, coordZ, ct1, ct2, ct3);

        pos1 = pos1 - yeeLayout->getMinExCoordFP ();
        pos2 = pos2 - yeeLayout->getMinExCoordFP ();

        pos3 = pos3 - yeeLayout->getMinEzCoordFP ();
        pos4 = pos4 - yeeLayout->getMinEzCoordFP ();

        GridCoordinate3D pos11 = convertCoord (pos1);
        GridCoordinate3D pos21 = convertCoord (pos2);

        GridCoordinate3D pos31 = convertCoord (pos3);
        GridCoordinate3D pos41 = convertCoord (pos4);

        FieldValue *valEx11 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
        FieldValue *valEx21 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);

        FieldValue *valEz11 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
        FieldValue *valEz21 = curEz->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
        if (valEx11 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos11)
            || valEx21 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos21)
            || valEz11 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos31)
            || valEz21 == NULLPTR || ((ParallelGrid*) curEz)->isBufferLeftPosition (pos41))
        {
          continue;
        }
#endif

        ASSERT (valEx11 != NULLPTR && valEx21 != NULLPTR && valEz11 != NULLPTR && valEz21 != NULLPTR);

        Ex1 = (*valEx11);
        Ex2 = (*valEx21);
        Ez1 = (*valEz11);
        Ez2 = (*valEz21);
      }
      else
      {
        UNREACHABLE;
      }

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (y0 - diffy0) * sin(angleTeta)*sin(anglePhi) + (coordZ - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += ((Ez1 + Ez2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                   + (Ex1 + Ex2)/FPValue(2.0) * FPValue (sin (angleTeta))) * exponent * SQR (gridStep) * (-1) * (y0==rightNTFF.get2 ()?1:-1);

      sum_phi += ((Ez1 + Ez2)/FPValue(2.0) * FPValue (sin (anglePhi))) * exponent * SQR (gridStep) * (y0==rightNTFF.get2 ()?1:-1);
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

template <LayoutType layout_type>
NPair
SchemeHelper::ntffL3D_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi,
                         GridCoordinate3D leftNTFF,
                         GridCoordinate3D rightNTFF,
                         YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *yeeLayout,
                         FPValue gridStep,
                         FPValue sourceWaveLength,
                         Grid<GridCoordinate3D> *curEx,
                         Grid<GridCoordinate3D> *curEy,
                         Grid<GridCoordinate3D> *curEz,
                         Grid<GridCoordinate3D> *curHx,
                         Grid<GridCoordinate3D> *curHy,
                         Grid<GridCoordinate3D> *curHz)
{
#ifdef COMPLEX_FIELD_VALUES
  ASSERT (yeeLayout->getSize ().get1 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get2 () % 2 == 0);
  ASSERT (yeeLayout->getSize ().get3 () % 2 == 0);
  FPValue diffx0 = yeeLayout->getSize ().get1 () / 2;
  FPValue diffy0 = yeeLayout->getSize ().get2 () / 2;
  FPValue diffz0 = yeeLayout->getSize ().get3 () / 2;

  CoordinateType ct1, ct2, ct3;
#ifdef DEBUG_INFO
  ct1 = leftNTFF.getType1 ();
  ct2 = leftNTFF.getType2 ();
  ct3 = leftNTFF.getType3 ();
#endif

  GridCoordinateFP3D coordStart = GRID_COORDINATE_FP_3D (leftNTFF.get1 () + 0.5, leftNTFF.get2 () + 0.5, z0, ct1, ct2, ct3);
  GridCoordinateFP3D coordEnd = GRID_COORDINATE_FP_3D (rightNTFF.get1 () - 0.5, rightNTFF.get2 () - 0.5, z0, ct1, ct2, ct3);

  FieldValue sum_teta (0.0, 0.0);
  FieldValue sum_phi (0.0, 0.0);

  for (FPValue coordX = coordStart.get1 (); coordX <= coordEnd.get1 (); ++coordX)
  {
    for (FPValue coordY = coordStart.get2 (); coordY <= coordEnd.get2 (); ++coordY)
    {
      FieldValue Ex1;
      FieldValue Ex2;
      FieldValue Ey1;
      FieldValue Ey2;

      if (layout_type == E_CENTERED)
      {
        GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (coordX - 0.5, coordY, z0, ct1, ct2, ct3);
        GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (coordX + 0.5, coordY, z0, ct1, ct2, ct3);
        GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (coordX, coordY - 0.5, z0, ct1, ct2, ct3);
        GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (coordX, coordY + 0.5, z0, ct1, ct2, ct3);

        pos1 = pos1 - yeeLayout->getMinExCoordFP ();
        pos2 = pos2 - yeeLayout->getMinExCoordFP ();

        pos3 = pos3 - yeeLayout->getMinEyCoordFP ();
        pos4 = pos4 - yeeLayout->getMinEyCoordFP ();

        GridCoordinate3D pos11 = convertCoord (pos1 - GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
        GridCoordinate3D pos12 = convertCoord (pos1 + GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
        GridCoordinate3D pos21 = convertCoord (pos2 - GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
        GridCoordinate3D pos22 = convertCoord (pos2 + GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));

        GridCoordinate3D pos31 = convertCoord (pos3 - GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
        GridCoordinate3D pos32 = convertCoord (pos3 + GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
        GridCoordinate3D pos41 = convertCoord (pos4 - GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));
        GridCoordinate3D pos42 = convertCoord (pos4 + GRID_COORDINATE_FP_3D (0, 0, 0.5, ct1, ct2, ct3));

        FieldValue *valEx11 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
        FieldValue *valEx12 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos12);
        FieldValue *valEx21 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);
        FieldValue *valEx22 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos22);

        FieldValue *valEy11 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
        FieldValue *valEy12 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos32);
        FieldValue *valEy21 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);
        FieldValue *valEy22 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos42);

#ifdef PARALLEL_GRID
        if (valEx11 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos11)
            || valEx12 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos12)
            || valEx21 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos21)
            || valEx22 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos22)
            || valEy11 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos31)
            || valEy12 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos32)
            || valEy21 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos41)
            || valEy22 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos42))
        {
          continue;
        }
#endif

        ASSERT (valEx11 != NULLPTR && valEx12 != NULLPTR && valEx21 != NULLPTR && valEx22 != NULLPTR
                && valEy11 != NULLPTR && valEy12 != NULLPTR && valEy21 != NULLPTR && valEy22 != NULLPTR);

        Ex1 = (*valEx11 + *valEx12) / FPValue(2.0);
        Ex2 = (*valEx21 + *valEx22) / FPValue(2.0);
        Ey1 = (*valEy11 + *valEy12) / FPValue(2.0);
        Ey2 = (*valEy21 + *valEy22) / FPValue(2.0);
      }
      else if (layout_type == H_CENTERED)
      {
        GridCoordinateFP3D pos1 = GRID_COORDINATE_FP_3D (coordX, coordY - 0.5, z0, ct1, ct2, ct3);
        GridCoordinateFP3D pos2 = GRID_COORDINATE_FP_3D (coordX, coordY - 0.5, z0, ct1, ct2, ct3);
        GridCoordinateFP3D pos3 = GRID_COORDINATE_FP_3D (coordX - 0.5, coordY, z0, ct1, ct2, ct3);
        GridCoordinateFP3D pos4 = GRID_COORDINATE_FP_3D (coordX + 0.5, coordY, z0, ct1, ct2, ct3);

        pos1 = pos1 - yeeLayout->getMinExCoordFP ();
        pos2 = pos2 - yeeLayout->getMinExCoordFP ();

        pos3 = pos3 - yeeLayout->getMinEyCoordFP ();
        pos4 = pos4 - yeeLayout->getMinEyCoordFP ();

        GridCoordinate3D pos11 = convertCoord (pos1);
        GridCoordinate3D pos21 = convertCoord (pos2);

        GridCoordinate3D pos31 = convertCoord (pos3);
        GridCoordinate3D pos41 = convertCoord (pos4);

        FieldValue *valEx11 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos11);
        FieldValue *valEx21 = curEx->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos21);

        FieldValue *valEy11 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos31);
        FieldValue *valEy21 = curEy->getFieldValueOrNullCurrentAfterShiftByAbsolutePos (pos41);

#ifdef PARALLEL_GRID
        if (valEx11 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos11)
            || valEx21 == NULLPTR || ((ParallelGrid*) curEx)->isBufferLeftPosition (pos21)
            || valEy11 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos31)
            || valEy21 == NULLPTR || ((ParallelGrid*) curEy)->isBufferLeftPosition (pos41))
        {
          continue;
        }
#endif

        ASSERT (valEx11 != NULLPTR && valEx21 != NULLPTR && valEy11 != NULLPTR && valEy21 != NULLPTR);

        Ex1 = (*valEx11);
        Ex2 = (*valEx21);
        Ey1 = (*valEy11);
        Ey2 = (*valEy21);
      }
      else
      {
        UNREACHABLE;
      }

      FPValue arg = (coordX - diffx0) * sin(angleTeta)*cos(anglePhi) + (coordY - diffy0) * sin(angleTeta)*sin(anglePhi) + (z0 - diffz0) * cos (angleTeta);
      arg *= gridStep;

      FPValue k = 2*PhysicsConst::Pi / sourceWaveLength;

      FieldValue exponent (cos(k*arg), sin(k*arg));

      sum_teta += ((Ey1 + Ey2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (cos (anglePhi))
                   - (Ex1 + Ex2)/FPValue(2.0) * FPValue (cos (angleTeta)) * FPValue (sin (anglePhi))) * exponent * SQR (gridStep) * (z0==rightNTFF.get3 ()?1:-1);

      sum_phi += ((Ey1 + Ey2)/FPValue(2.0) * FPValue (sin (anglePhi))
                  + (Ex1 + Ex2)/FPValue(2.0) * FPValue (cos (anglePhi))) * exponent * SQR (gridStep) * (-1) * (z0==rightNTFF.get3 ()?1:-1);
    }
  }

  return NPair (sum_teta, sum_phi);
#else
  ASSERT_MESSAGE ("Solver is not compiled with support of complex values. Recompile it with -DCOMPLEX_FIELD_VALUES=ON.");
#endif
}

template <LayoutType layout_type>
NPair
SchemeHelper::ntffN3D (FPValue angleTeta, FPValue anglePhi,
                       GridCoordinate3D leftNTFF,
                       GridCoordinate3D rightNTFF,
                       YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *yeeLayout,
                       FPValue gridStep,
                       FPValue sourceWaveLength, // TODO: check sourceWaveLengthNumerical
                       Grid<GridCoordinate3D> *curEx,
                       Grid<GridCoordinate3D> *curEy,
                       Grid<GridCoordinate3D> *curEz,
                       Grid<GridCoordinate3D> *curHx,
                       Grid<GridCoordinate3D> *curHy,
                       Grid<GridCoordinate3D> *curHz)
{
  NPair nx = ntffN3D_x (leftNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEy, curEz, curHx, curHy, curHz)
                     + ntffN3D_x (rightNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEy, curEz, curHx, curHy, curHz);
  NPair ny = ntffN3D_y (leftNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEy, curEz, curHx, curHy, curHz)
                     + ntffN3D_y (rightNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEy, curEz, curHx, curHy, curHz);
  NPair nz = ntffN3D_z (leftNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEy, curEz, curHx, curHy, curHz)
                     + ntffN3D_z (rightNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEy, curEz, curHx, curHy, curHz);

  return nx + ny + nz;
}

template <LayoutType layout_type>
NPair
SchemeHelper::ntffL3D (FPValue angleTeta, FPValue anglePhi,
                       GridCoordinate3D leftNTFF,
                       GridCoordinate3D rightNTFF,
                       YeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), GridCoordinate3DTemplate, layout_type> *yeeLayout,
                       FPValue gridStep,
                       FPValue sourceWaveLength, // TODO: check sourceWaveLengthNumerical
                       Grid<GridCoordinate3D> *curEx,
                       Grid<GridCoordinate3D> *curEy,
                       Grid<GridCoordinate3D> *curEz,
                       Grid<GridCoordinate3D> *curHx,
                       Grid<GridCoordinate3D> *curHy,
                       Grid<GridCoordinate3D> *curHz)
{
  NPair lx = ntffL3D_x (leftNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEy, curEz, curHx, curHy, curHz)
                     + ntffL3D_x (rightNTFF.get1 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEy, curEz, curHx, curHy, curHz);
  NPair ly = ntffL3D_y (leftNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEy, curEz, curHx, curHy, curHz)
                     + ntffL3D_y (rightNTFF.get2 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEy, curEz, curHx, curHy, curHz);
  NPair lz = ntffL3D_z (leftNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEy, curEz, curHx, curHy, curHz)
                     + ntffL3D_z (rightNTFF.get3 (), angleTeta, anglePhi, leftNTFF, rightNTFF, yeeLayout, gridStep, sourceWaveLength, curEx, curEy, curEz, curHx, curHy, curHz);

  return lx + ly + lz;
}

#endif /* !SCHEME_HELPER_H */
