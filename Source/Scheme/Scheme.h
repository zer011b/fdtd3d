#ifndef SCHEME_H
#define SCHEME_H

#include "GridInterface.h"
#include "PhysicsConst.h"
#include "YeeGridLayout.h"
#include "ParallelYeeGridLayout.h"
#include "CallBack.h"
#include "SchemeHelper.h"
#include "InternalScheme.h"

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
class Scheme
{
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

  TC blockCount;
  TC blockSize;
#endif /* CUDA_ENABLED */

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

public:

  /**
   * Perform all time steps for scheme
   */
  ICUDA_HOST
  void performSteps ()
  {
    for (time_step t = 0; t < totalTimeSteps; t += NTimeSteps)
    {
      /*
       * Each NTimeSteps sharing will be performed.
       *
       * For sequential solver, NTimeSteps == totalTimeSteps
       * For parallel/cuda solver, NTimeSteps == min (bufSize, cudaBufSize)
       */
      performNSteps (t, NTimeSteps);
    }
  }

private:

  void performNSteps (time_step tStart, time_step N);
  void performNStepsForBlock (time_step tStart, time_step N, TC blockIdx);
  void share ();
  void rebalance ();

  template <uint8_t grid_type>
  void performFieldSteps (time_step, TC, TC);

  template <uint8_t grid_type, bool usePML, bool useMetamaterials>
  void calculateFieldStep (time_step, TC, TC);

private:


  // TODO: unify
  void performNSteps (time_step, time_step);
  void performAmplitudeSteps (time_step);

  int updateAmplitude (FPValue, FieldPointValue *, FPValue *);

  void makeGridScattered (Grid<TC> *, GridType);
  void gatherFieldsTotal (bool);
  void saveGrids (time_step);
  void saveNTFF (bool, time_step);

  void additionalUpdateOfGrids (time_step, time_step &);

  TC getStartCoord (GridType, TC);
  TC getEndCoord (GridType, TC);

  void initMaterialFromFile (GridType, Grid<TC> *, Grid<TC> *);
  void initGridWithInitialVals (GridType, Grid<TC> *, FPValue);

  void initSigmas ();
  // {
  //   UNREACHABLE;
  // }

  void initSigma (FieldPointValue *, grid_coord, FPValue);

  bool doSkipMakeScattered (TCFP);

  TC getStartCoordRes (OrthogonalAxis, TC, TC);
  TC getEndCoordRes (OrthogonalAxis, TC, TC);

  /*
   * 3D ntff
   * TODO: add 1D,2D modes
   */
  NPair ntffN (FPValue angleTeta, FPValue anglePhi, Grid<TC> *, Grid<TC> *, Grid<TC> *, Grid<TC> *);
  NPair ntffL (FPValue angleTeta, FPValue anglePhi, Grid<TC> *, Grid<TC> *, Grid<TC> *);

  FPValue Pointing_scat (FPValue angleTeta, FPValue anglePhi, Grid<TC> *, Grid<TC> *, Grid<TC> *, Grid<TC> *,
                         Grid<TC> *, Grid<TC> *);
  FPValue Pointing_inc (FPValue angleTeta, FPValue anglePhi);

public:

  void performSteps ();
  void performCudaSteps ();

  void initScheme (FPValue, FPValue);
  void initCallBacks ();
  void initGrids ();

  void initFullMaterialGrids ();
  void initFullFieldGrids ();

  /**
   * Default constructor used for template instantiation
   */
  Scheme ()
  {
  }

  Scheme (YeeGridLayout<Type, TCoord, layout_type> *layout,
          bool parallelLayout,
          const TC& totSize,
          time_step tStep);

  ~Scheme ();
};

#include "Scheme.template.h"

#endif /* SCHEME_H */
