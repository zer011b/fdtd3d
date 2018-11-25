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

    scheme->share ();
    scheme->rebalance ();

    if (SOLVER_SETTINGS.getDoSaveIntermediateRes ()
        && ((tStart) / SOLVER_SETTINGS.getIntermediateSaveStep () < (tStart + N) / SOLVER_SETTINGS.getIntermediateSaveStep ()))
    {
      //scheme->gatherFieldsTotal (SOLVER_SETTINGS.getDoSaveScatteredFieldIntermediate ());
      scheme->saveGrids (tStart + N);
    }

    if (SOLVER_SETTINGS.getDoUseNTFF ()
        && ((tStart) / SOLVER_SETTINGS.getIntermediateSaveStep () < (tStart + N) / SOLVER_SETTINGS.getIntermediateSaveStep ()))
    {
      //scheme->saveNTFF (SOLVER_SETTINGS.getDoCalcReverseNTFF (), tStart + N);
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

    scheme->share ();
    scheme->rebalance ();

    if (SOLVER_SETTINGS.getDoSaveIntermediateRes ()
        && ((tStart) / SOLVER_SETTINGS.getIntermediateSaveStep () < (tStart + N) / SOLVER_SETTINGS.getIntermediateSaveStep ()))
    {
      //scheme->gatherFieldsTotal (SOLVER_SETTINGS.getDoSaveScatteredFieldIntermediate ());
      scheme->saveGrids (tStart + N);
    }

    if (SOLVER_SETTINGS.getDoUseNTFF ()
        && ((tStart) / SOLVER_SETTINGS.getIntermediateSaveStep () < (tStart + N) / SOLVER_SETTINGS.getIntermediateSaveStep ()))
    {
      //scheme->saveNTFF (SOLVER_SETTINGS.getDoCalcReverseNTFF (), tStart + N);
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

    scheme->share ();
    scheme->rebalance ();

    if (SOLVER_SETTINGS.getDoSaveIntermediateRes ()
        && ((tStart) / SOLVER_SETTINGS.getIntermediateSaveStep () < (tStart + N) / SOLVER_SETTINGS.getIntermediateSaveStep ()))
    {
      //scheme->gatherFieldsTotal (SOLVER_SETTINGS.getDoSaveScatteredFieldIntermediate ());
      scheme->saveGrids (tStart + N);
    }

    if (SOLVER_SETTINGS.getDoUseNTFF ()
        && ((tStart) / SOLVER_SETTINGS.getIntermediateSaveStep () < (tStart + N) / SOLVER_SETTINGS.getIntermediateSaveStep ()))
    {
      //scheme->saveNTFF (SOLVER_SETTINGS.getDoCalcReverseNTFF (), tStart + N);
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

  static void initSigma (FieldValue *fieldValue, grid_coord dist, FPValue boundary, FPValue gridStep)
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

      TCoord<FPValue, true> size = layout->getEpsCoordFP (sigma->getTotalSize ());

      /*
       * TODO: add layout coordinates for material: sigma, eps, etc.
       */
      ASSERT (FPValue (grid_coord (posAbs.get1 () - FPValue (0.5))) == posAbs.get1 () - FPValue (0.5));
      if (posAbs.get1 () < PMLSizeX)
      {
        grid_coord dist = (grid_coord) (PMLSizeX - posAbs.get1 ());
        SchemeHelper::initSigma (&valSigma, dist, boundary, dx);
      }
      else if (posAbs.get1 () >= size.get1 () - PMLSizeX)
      {
        grid_coord dist = (grid_coord) (posAbs.get1 () - (size.get1 () - PMLSizeX));
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

  // static
  // NPair ntffN3D_x (grid_coord, FPValue, FPValue,
  //                  GridCoordinate3D, GridCoordinate3D,
  //                  YL3D_Dim3 *,
  //                  FPValue, FPValue,
  //                  Grid<GridCoordinate1D> *,
  //                  Grid<GridCoordinate3D> *,
  //                  Grid<GridCoordinate3D> *,
  //                  Grid<GridCoordinate3D> *);
  // static
  // NPair ntffN3D_y (grid_coord, FPValue, FPValue,
  //                  GridCoordinate3D, GridCoordinate3D,
  //                  YL3D_Dim3 *,
  //                  FPValue, FPValue,
  //                  Grid<GridCoordinate1D> *,
  //                  Grid<GridCoordinate3D> *,
  //                  Grid<GridCoordinate3D> *,
  //                  Grid<GridCoordinate3D> *);
  // static
  // NPair ntffN3D_z (grid_coord, FPValue, FPValue,
  //                  GridCoordinate3D, GridCoordinate3D,
  //                  YL3D_Dim3 *,
  //                  FPValue, FPValue,
  //                  Grid<GridCoordinate1D> *,
  //                  Grid<GridCoordinate3D> *,
  //                  Grid<GridCoordinate3D> *,
  //                  Grid<GridCoordinate3D> *);
  // static
  // NPair ntffN3D (FPValue, FPValue,
  //                GridCoordinate3D, GridCoordinate3D,
  //                YL3D_Dim3 *,
  //                FPValue, FPValue,
  //                Grid<GridCoordinate1D> *,
  //                Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
  //                Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  //
  // template <SchemeType_t Type, LayoutType layout_type>
  // static
  // NPair ntffN2D (FPValue, FPValue,
  //                GridCoordinate2D, GridCoordinate2D,
  //                YeeGridLayout<Type, GridCoordinate2DTemplate, layout_type> *,
  //                FPValue, FPValue,
  //                Grid<GridCoordinate1D> *,
  //                Grid<GridCoordinate2D> *, Grid<GridCoordinate2D> *,
  //                Grid<GridCoordinate2D> *, Grid<GridCoordinate2D> *)
  // {}
  //
  // template <SchemeType_t Type, LayoutType layout_type>
  // static
  // NPair ntffN1D (FPValue, FPValue,
  //                GridCoordinate1D, GridCoordinate1D,
  //                YeeGridLayout<Type, GridCoordinate1DTemplate, layout_type> *,
  //                FPValue, FPValue,
  //                Grid<GridCoordinate1D> *,
  //                Grid<GridCoordinate1D> *, Grid<GridCoordinate1D> *,
  //                Grid<GridCoordinate1D> *, Grid<GridCoordinate1D> *)
  // {}
  //
  // static
  // NPair ntffL3D_x (grid_coord, FPValue, FPValue,
  //                  GridCoordinate3D, GridCoordinate3D,
  //                  YL3D_Dim3 *,
  //                  FPValue, FPValue,
  //                  Grid<GridCoordinate1D> *,
  //                  Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  // static
  // NPair ntffL3D_y (grid_coord, FPValue, FPValue,
  //                  GridCoordinate3D, GridCoordinate3D,
  //                  YL3D_Dim3 *,
  //                  FPValue, FPValue,
  //                  Grid<GridCoordinate1D> *,
  //                  Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  // static
  // NPair ntffL3D_z (grid_coord, FPValue, FPValue,
  //                  GridCoordinate3D, GridCoordinate3D,
  //                  YL3D_Dim3 *,
  //                  FPValue, FPValue,
  //                  Grid<GridCoordinate1D> *,
  //                  Grid<GridCoordinate3D> *,
  //                  Grid<GridCoordinate3D> *,
  //                  Grid<GridCoordinate3D> *);
  // static
  // NPair ntffL3D (FPValue, FPValue,
  //                GridCoordinate3D, GridCoordinate3D,
  //                YL3D_Dim3 *,
  //                FPValue, FPValue,
  //                Grid<GridCoordinate1D> *,
  //                Grid<GridCoordinate3D> *,
  //                Grid<GridCoordinate3D> *,
  //                Grid<GridCoordinate3D> *);
  //
  // template <SchemeType_t Type, LayoutType layout_type>
  // static
  // NPair ntffL2D (FPValue, FPValue,
  //                GridCoordinate2D, GridCoordinate2D,
  //                YeeGridLayout<Type, GridCoordinate2DTemplate, layout_type> *,
  //                FPValue, FPValue,
  //                Grid<GridCoordinate1D> *,
  //                Grid<GridCoordinate2D> *,
  //                Grid<GridCoordinate2D> *,
  //                Grid<GridCoordinate2D> *)
  // {}
  //
  // template <SchemeType_t Type, LayoutType layout_type>
  // static
  // NPair ntffL1D (FPValue, FPValue,
  //                GridCoordinate1D, GridCoordinate1D,
  //                YeeGridLayout<Type, GridCoordinate1DTemplate, layout_type> *,
  //                FPValue, FPValue,
  //                Grid<GridCoordinate1D> *,
  //                Grid<GridCoordinate1D> *,
  //                Grid<GridCoordinate1D> *,
  //                Grid<GridCoordinate1D> *)
  // {}

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
  }
};

#endif /* !SCHEME_HELPER_H */
