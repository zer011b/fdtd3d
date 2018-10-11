#ifndef SCHEME_H
#define SCHEME_H

#include "GridInterface.h"
#include "PhysicsConst.h"
#include "YeeGridLayout.h"
#include "ParallelYeeGridLayout.h"
#include "CallBack.h"
#include "SchemeHelper.h"

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
class Scheme
{
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

protected:

  InternalScheme<Type, TCoord, layout_type, Grid> internalScheme;

#ifdef CUDA_ENABLED
  InternalScheme<Type, TCoord, layout_type, CudaGrid> intGPUScheme;
  InternalScheme<Type, TCoord, layout_type, CudaGrid> *d_intGPUScheme;
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

#endif /* SCHEME_H */
