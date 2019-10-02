#ifndef SCHEME_H
#define SCHEME_H

#include "GridInterface.h"
#include "PhysicsConst.h"
#include "YeeGridLayout.h"
#include "ParallelYeeGridLayout.h"
#include "CallBack.h"
#include "SchemeHelper.h"
#include "InternalScheme.h"

class SchemeHelper;

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
class Scheme
{
  friend class SchemeHelper;

  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

protected:

  time_step totalTimeSteps;
  time_step NTimeSteps;

  InternalScheme<Type, TCoord, layout_type> *intScheme;

#ifdef CUDA_ENABLED
  InternalSchemeGPU<Type, TCoord, layout_type> *gpuIntScheme;
  InternalSchemeGPU<Type, TCoord, layout_type> *gpuIntSchemeOnGPU;
  InternalSchemeGPU<Type, TCoord, layout_type> *d_gpuIntSchemeOnGPU;
#endif /* CUDA_ENABLED */

  TC blockCount;
  TC blockSize;

#ifdef PARALLEL_GRID
  ParallelGridGroup *eGroup;
  ParallelGridGroup *hGroup;
#endif /* PARALLEL_GRID */

private:

  bool useParallel;

  Grid<TC> *totalEx;
  Grid<TC> *totalEy;
  Grid<TC> *totalEz;
  Grid<TC> *totalHx;
  Grid<TC> *totalHy;
  Grid<TC> *totalHz;

  bool totalInitialized;

  Grid<TC> *totalEps;
  Grid<TC> *totalMu;
  Grid<TC> *totalOmegaPE;
  Grid<TC> *totalOmegaPM;
  Grid<TC> *totalGammaE;
  Grid<TC> *totalGammaM;

  time_step totalStep;

  int process;

  int numProcs;

  Dumper<TC> *dumper[FILE_TYPE_COUNT];
  Loader<TC> *loader[FILE_TYPE_COUNT];

  Dumper<GridCoordinate1D> *dumper1D[FILE_TYPE_COUNT];

  CoordinateType ct1;
  CoordinateType ct2;
  CoordinateType ct3;

  YeeGridLayout<Type, TCoord, layout_type> *yeeLayout;

private:

  void performNSteps (time_step tStart, time_step N);
  void performNStepsForBlock (time_step tStart, time_step N, TC blockIdx);

#ifdef PARALLEL_GRID
  void tryShareE ();
  void tryShareH ();
  void shareE ();
  void shareH ();
#endif /* PARALLEL_GRID */

  void rebalance ();

  void initCallBacks ();
  void initGrids ();
  void initBlocks (time_step t_total);

#ifdef PARALLEL_GRID
  void initParallelBlocks ();
#endif

  uint64_t estimateCurrentSize ();

#ifdef CUDA_ENABLED
  void setupBlocksForGPU (TC &blockCount, TC &blockSize);
#endif

  template <uint8_t grid_type>
  void performFieldSteps (time_step, TC, TC);

  template <uint8_t grid_type, bool usePML, bool useMetamaterials>
  void calculateFieldStep (time_step, TC, TC);

private:

  void makeGridScattered (Grid<TC> *, GridType);
  void gatherFieldsTotal (bool);
  void saveGrids (time_step);
  void saveNTFF (bool, time_step);

  TC getStartCoord (GridType, TC);
  TC getEndCoord (GridType, TC);

  void initMaterialFromFile (GridType, Grid<TC> *, Grid<TC> *);
  void initGridWithInitialVals (GridType, Grid<TC> *, FPValue);

  void initSigmas ();

  bool doSkipMakeScattered (TCFP);

  TC getStartCoordRes (OrthogonalAxis, TC, TC);
  TC getEndCoordRes (OrthogonalAxis, TC, TC);

  /*
   * 3D ntff
   * TODO: add 1D,2D modes
   */
  NPair ntffN (FPValue angleTeta, FPValue anglePhi, Grid<TC> *, Grid<TC> *, Grid<TC> *, Grid<TC> *, Grid<TC> *, Grid<TC> *, TC, TC);
  NPair ntffL (FPValue angleTeta, FPValue anglePhi, Grid<TC> *, Grid<TC> *, Grid<TC> *, Grid<TC> *, Grid<TC> *, Grid<TC> *, TC, TC);

  FPValue Pointing_scat (FPValue angleTeta, FPValue anglePhi, Grid<TC> *, Grid<TC> *, Grid<TC> *, Grid<TC> *,
                         Grid<TC> *, Grid<TC> *, TC, TC);
  FPValue Pointing_inc (FPValue angleTeta, FPValue anglePhi);

  void performCudaSteps ();

  void initFullMaterialGrids ();
  void initFullFieldGrids ();

public:

  /**
   * Perform all time steps for scheme
   */
  void performSteps ()
  {
    /*
     * Each NTimeSteps sharing will be performed for parallel builds.
     *
     * For non-Cuda solver (both sequential and parallel), NTimeSteps == 1
     * For Cuda solver, NTimeSteps == bufSize - 1
     */
    if (!SOLVER_SETTINGS.getDoUseCuda ())
    {
      ASSERT (NTimeSteps == 1);
    }

    for (time_step t = 0; t < totalTimeSteps; t += NTimeSteps)
    {
      performNSteps (t, NTimeSteps);
    }

    if (SOLVER_SETTINGS.getDoSaveRes ())
    {
      if (!SOLVER_SETTINGS.getDoSaveResPerProcess ())
      {
        gatherFieldsTotal (SOLVER_SETTINGS.getDoSaveScatteredFieldRes ());
      }
      else
      {
        ALWAYS_ASSERT (!SOLVER_SETTINGS.getDoSaveScatteredFieldRes ());
      }

      saveGrids (totalTimeSteps);
    }
  }

  void initScheme (FPValue, FPValue, time_step t_total);

  Scheme (YeeGridLayout<Type, TCoord, layout_type> *layout,
          bool parallelLayout,
          const TC& totSize,
          time_step tStep);

  ~Scheme ();
};

#include "Scheme.inc.h"

#endif /* SCHEME_H */
