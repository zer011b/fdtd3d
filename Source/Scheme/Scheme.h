#ifndef SCHEME_H
#define SCHEME_H

#include "GridInterface.h"
#include "PhysicsConst.h"
#include "YeeGridLayout.h"
#include "ParallelYeeGridLayout.h"
#include "CallBack.h"

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

template <SchemeType_t Type, template <typename, bool> class TCoord, LayoutType layout_type>
class Scheme
{
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

protected:

  YeeGridLayout<Type, TCoord, layout_type> *yeeLayout;
  bool isParallelLayout;

private:

  CoordinateType ct1;
  CoordinateType ct2;
  CoordinateType ct3;

  Grid<TC> *Ex;
  Grid<TC> *Ey;
  Grid<TC> *Ez;
  Grid<TC> *Hx;
  Grid<TC> *Hy;
  Grid<TC> *Hz;

  Grid<TC> *Dx;
  Grid<TC> *Dy;
  Grid<TC> *Dz;
  Grid<TC> *Bx;
  Grid<TC> *By;
  Grid<TC> *Bz;

  Grid<TC> *D1x;
  Grid<TC> *D1y;
  Grid<TC> *D1z;
  Grid<TC> *B1x;
  Grid<TC> *B1y;
  Grid<TC> *B1z;

  Grid<TC> *ExAmplitude;
  Grid<TC> *EyAmplitude;
  Grid<TC> *EzAmplitude;
  Grid<TC> *HxAmplitude;
  Grid<TC> *HyAmplitude;
  Grid<TC> *HzAmplitude;

  Grid<TC> *Eps;
  Grid<TC> *Mu;

  Grid<TC> *SigmaX;
  Grid<TC> *SigmaY;
  Grid<TC> *SigmaZ;

  Grid<TC> *OmegaPE;
  Grid<TC> *GammaE;

  Grid<TC> *OmegaPM;
  Grid<TC> *GammaM;

  Grid<GridCoordinate1D> *EInc;
  Grid<GridCoordinate1D> *HInc;

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

  // Wave parameters
  FPValue sourceWaveLength;
  FPValue sourceWaveLengthNumerical;

  FPValue sourceFrequency;
  FPValue relPhaseVelocity;

  /** Courant number */
  FPValue courantNum;

  // dx
  FPValue gridStep;

  // dt
  FPValue gridTimeStep;

  time_step totalStep;

  int process;

  int numProcs;

  TC leftNTFF;
  TC rightNTFF;

  Dumper<TC> *dumper[FILE_TYPE_COUNT];
  Loader<TC> *loader[FILE_TYPE_COUNT];

  Dumper<GridCoordinate1D> *dumper1D[FILE_TYPE_COUNT];

  SourceCallBack ExBorder;
  SourceCallBack ExInitial;

  SourceCallBack EyBorder;
  SourceCallBack EyInitial;

  SourceCallBack EzBorder;
  SourceCallBack EzInitial;

  SourceCallBack HxBorder;
  SourceCallBack HxInitial;

  SourceCallBack HyBorder;
  SourceCallBack HyInitial;

  SourceCallBack HzBorder;
  SourceCallBack HzInitial;

  SourceCallBack Jx;
  SourceCallBack Jy;
  SourceCallBack Jz;
  SourceCallBack Mx;
  SourceCallBack My;
  SourceCallBack Mz;

  SourceCallBack ExExact;
  SourceCallBack EyExact;
  SourceCallBack EzExact;
  SourceCallBack HxExact;
  SourceCallBack HyExact;
  SourceCallBack HzExact;

  bool useParallel;

  /*
   * TODO: maybe add separate for Dx, etc.
   */
  static const bool doNeedEx;
  static const bool doNeedEy;
  static const bool doNeedEz;
  static const bool doNeedHx;
  static const bool doNeedHy;
  static const bool doNeedHz;

  static const bool doNeedSigmaX;
  static const bool doNeedSigmaY;
  static const bool doNeedSigmaZ;

private:

  void initCoordTypes ();

  template <uint8_t grid_type>
  void calculateTFSF (TC, FieldValue &, FieldValue &, FieldValue &, FieldValue &,
                      TC, TC, TC, TC);

  template <uint8_t grid_type, bool usePML, bool useMetamaterials>
  void calculateFieldStep (time_step, TC, TC);

  template <uint8_t grid_type, bool usePML, bool useMetamaterials>
  void calculateFieldStepInit (Grid<TC> **, GridType *, Grid<TC> **, GridType *, Grid<TC> **, GridType *, Grid<TC> **, GridType *,
    Grid<TC> **, GridType *, Grid<TC> **, GridType *, Grid<TC> **, GridType *, Grid<TC> **, Grid<TC> **,
    Grid<TC> **, GridType *, Grid<TC> **, GridType *, SourceCallBack *, SourceCallBack *, SourceCallBack *, FPValue *);

  template <uint8_t grid_type, bool usePML>
  void calculateFieldStepIteration (time_step, TC, Grid<TC> *, GridType, Grid<TC> *, GridType, Grid<TC> *, Grid<TC> *, SourceCallBack, FPValue);
  void calculateFieldStepIterationPMLMetamaterials (time_step, TC, Grid<TC> *, Grid<TC> *, GridType,
       Grid<TC> *, GridType,  Grid<TC> *, GridType,  Grid<TC> *, GridType, FPValue);
  template <bool useMetamaterials>
  void calculateFieldStepIterationPML (time_step, TC, Grid<TC> *, Grid<TC> *, Grid<TC> *, GridType, GridType,
       Grid<TC> *, GridType,  Grid<TC> *, GridType,  Grid<TC> *, GridType, FPValue);

  template <uint8_t grid_type>
  void calculateFieldStepIterationBorder (time_step, TC, Grid<TC> *, SourceCallBack);
  template <uint8_t grid_type>
  void calculateFieldStepIterationExact (time_step, TC, Grid<TC> *, SourceCallBack,
    FPValue &, FPValue &, FPValue &, FPValue &, FPValue &, FPValue &);

  template<uint8_t EnumVal> void performPointSourceCalc (time_step);

  // TODO: unify
  void performNSteps (time_step, time_step);
  void performAmplitudeSteps (time_step);

  int updateAmplitude (FPValue, FieldPointValue *, FPValue *);

  void performPlaneWaveESteps (time_step);
  void performPlaneWaveHSteps (time_step);

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

  bool doSkipBorderFunc (TC, Grid<TC> *);
  bool doSkipMakeScattered (TCFP);

  TC getStartCoordRes (OrthogonalAxis, TC, TC);
  TC getEndCoordRes (OrthogonalAxis, TC, TC);

  FieldValue approximateIncidentWaveE (TCFP pos);
  FieldValue approximateIncidentWaveH (TCFP pos);

  void calculateTFSFExAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { UNREACHABLE; }
  void calculateTFSFEyAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { UNREACHABLE; }
  void calculateTFSFEzAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { UNREACHABLE; }
  void calculateTFSFHxAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { UNREACHABLE; }
  void calculateTFSFHyAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { UNREACHABLE; }
  void calculateTFSFHzAsserts (TC pos11, TC pos12, TC pos21, TC pos22) { UNREACHABLE; }

  /*
   * 3D ntff
   * TODO: add 1D,2D modes
   */
  NPair ntffN (FPValue angleTeta, FPValue anglePhi, Grid<TC> *, Grid<TC> *, Grid<TC> *, Grid<TC> *);
  NPair ntffL (FPValue angleTeta, FPValue anglePhi, Grid<TC> *, Grid<TC> *, Grid<TC> *);

  FPValue Pointing_scat (FPValue angleTeta, FPValue anglePhi, Grid<TC> *, Grid<TC> *, Grid<TC> *, Grid<TC> *,
                         Grid<TC> *, Grid<TC> *);
  FPValue Pointing_inc (FPValue angleTeta, FPValue anglePhi);

  FieldValue calcField (FieldValue prev, FieldValue oppositeField12, FieldValue oppositeField11,
                        FieldValue oppositeField22, FieldValue oppositeField21, FieldValue prevRightSide,
                        FPValue Ca, FPValue Cb, FPValue delta)
  {
    FieldValue tmp = oppositeField12 - oppositeField11 - oppositeField22 + oppositeField21 + prevRightSide * delta;
    return Ca * prev + Cb * tmp;
  }

  FieldValue calcFieldDrude (FieldValue curDOrB, FieldValue prevDOrB, FieldValue prevPrevDOrB,
                             FieldValue prevEOrH, FieldValue prevPrevEOrH,
                             FPValue b0, FPValue b1, FPValue b2, FPValue a1, FPValue a2)
  {
    return b0 * curDOrB + b1 * prevDOrB + b2 * prevPrevDOrB - a1 * prevEOrH - a2 * prevPrevEOrH;
  }

  FieldValue calcFieldFromDOrB (FieldValue prevEOrH, FieldValue curDOrB, FieldValue prevDOrB,
                                FPValue Ca, FPValue Cb, FPValue Cc)
  {
    return Ca * prevEOrH + Cb * curDOrB - Cc * prevDOrB;
  }

  template <uint8_t grid_type>
  void performFieldSteps (time_step t, TC Start, TC End)
  {
    /*
     * TODO: remove check performed on each iteration
     */
    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        calculateFieldStep<grid_type, true, true> (t, Start, End);
      }
      else
      {
        calculateFieldStep<grid_type, true, false> (t, Start, End);
      }
    }
    else
    {
      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        calculateFieldStep<grid_type, false, true> (t, Start, End);
      }
      else
      {
        calculateFieldStep<grid_type, false, false> (t, Start, End);
      }
    }

    bool doUsePointSource;
    switch (grid_type)
    {
      case (static_cast<uint8_t> (GridType::EX)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceEx ();
        break;
      }
      case (static_cast<uint8_t> (GridType::EY)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceEy ();
        break;
      }
      case (static_cast<uint8_t> (GridType::EZ)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceEz ();
        break;
      }
      case (static_cast<uint8_t> (GridType::HX)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceHx ();
        break;
      }
      case (static_cast<uint8_t> (GridType::HY)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceHy ();
        break;
      }
      case (static_cast<uint8_t> (GridType::HZ)):
      {
        doUsePointSource = SOLVER_SETTINGS.getDoUsePointSourceHz ();
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    if (doUsePointSource)
    {
      performPointSourceCalc<grid_type> (t);
    }
  }

public:

  void performSteps ();
  void performCudaSteps ();

  void initScheme (FPValue, FPValue);
  void initCallBacks ();
  void initGrids ();

  void initFullMaterialGrids ();
  void initFullFieldGrids ();

#ifdef PARALLEL_GRID
  void allocateParallelGrids ();
#endif

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

class SchemeHelper
{
public:

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
                             bool doNeedEx, Grid<GridCoordinate1D> **Ex, Grid<GridCoordinate1D> **totalEx,
                             bool doNeedEy, Grid<GridCoordinate1D> **Ey, Grid<GridCoordinate1D> **totalEy,
                             bool doNeedEz, Grid<GridCoordinate1D> **Ez, Grid<GridCoordinate1D> **totalEz,
                             bool doNeedHx, Grid<GridCoordinate1D> **Hx, Grid<GridCoordinate1D> **totalHx,
                             bool doNeedHy, Grid<GridCoordinate1D> **Hy, Grid<GridCoordinate1D> **totalHy,
                             bool doNeedHz, Grid<GridCoordinate1D> **Hz, Grid<GridCoordinate1D> **totalHz)
  {
#ifdef PARALLEL_GRID
#ifdef GRID_1D
    if (*totalInitialized)
    {
      if (doNeedEx)
      {
        *totalEx = ((ParallelGrid *) *Ex)->gatherFullGridPlacement (*totalEx);
      }
      if (doNeedEy)
      {
        *totalEy = ((ParallelGrid *) *Ey)->gatherFullGridPlacement (*totalEy);
      }
      if (doNeedEz)
      {
        *totalEz = ((ParallelGrid *) *Ez)->gatherFullGridPlacement (*totalEz);
      }

      if (doNeedHx)
      {
        *totalHx = ((ParallelGrid *) *Hx)->gatherFullGridPlacement (*totalHx);
      }
      if (doNeedHy)
      {
        *totalHy = ((ParallelGrid *) *Hy)->gatherFullGridPlacement (*totalHy);
      }
      if (doNeedHz)
      {
        *totalHz = ((ParallelGrid *) *Hz)->gatherFullGridPlacement (*totalHz);
      }
    }
    else
    {
      if (doNeedEx)
      {
        *totalEx = ((ParallelGrid *) *Ex)->gatherFullGrid ();
      }
      if (doNeedEy)
      {
        *totalEy = ((ParallelGrid *) *Ey)->gatherFullGrid ();
      }
      if (doNeedEz)
      {
        *totalEz = ((ParallelGrid *) *Ez)->gatherFullGrid ();
      }

      if (doNeedHx)
      {
        *totalHx = ((ParallelGrid *) *Hx)->gatherFullGrid ();
      }
      if (doNeedHy)
      {
        *totalHy = ((ParallelGrid *) *Hy)->gatherFullGrid ();
      }
      if (doNeedHz)
      {
        *totalHz = ((ParallelGrid *) *Hz)->gatherFullGrid ();
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
                             bool doNeedEx, Grid<GridCoordinate2D> **Ex, Grid<GridCoordinate2D> **totalEx,
                             bool doNeedEy, Grid<GridCoordinate2D> **Ey, Grid<GridCoordinate2D> **totalEy,
                             bool doNeedEz, Grid<GridCoordinate2D> **Ez, Grid<GridCoordinate2D> **totalEz,
                             bool doNeedHx, Grid<GridCoordinate2D> **Hx, Grid<GridCoordinate2D> **totalHx,
                             bool doNeedHy, Grid<GridCoordinate2D> **Hy, Grid<GridCoordinate2D> **totalHy,
                             bool doNeedHz, Grid<GridCoordinate2D> **Hz, Grid<GridCoordinate2D> **totalHz)
  {
#ifdef PARALLEL_GRID
#ifdef GRID_2D
    if (*totalInitialized)
    {
      if (doNeedEx)
      {
        *totalEx = ((ParallelGrid *) *Ex)->gatherFullGridPlacement (*totalEx);
      }
      if (doNeedEy)
      {
        *totalEy = ((ParallelGrid *) *Ey)->gatherFullGridPlacement (*totalEy);
      }
      if (doNeedEz)
      {
        *totalEz = ((ParallelGrid *) *Ez)->gatherFullGridPlacement (*totalEz);
      }

      if (doNeedHx)
      {
        *totalHx = ((ParallelGrid *) *Hx)->gatherFullGridPlacement (*totalHx);
      }
      if (doNeedHy)
      {
        *totalHy = ((ParallelGrid *) *Hy)->gatherFullGridPlacement (*totalHy);
      }
      if (doNeedHz)
      {
        *totalHz = ((ParallelGrid *) *Hz)->gatherFullGridPlacement (*totalHz);
      }
    }
    else
    {
      if (doNeedEx)
      {
        *totalEx = ((ParallelGrid *) *Ex)->gatherFullGrid ();
      }
      if (doNeedEy)
      {
        *totalEy = ((ParallelGrid *) *Ey)->gatherFullGrid ();
      }
      if (doNeedEz)
      {
        *totalEz = ((ParallelGrid *) *Ez)->gatherFullGrid ();
      }

      if (doNeedHx)
      {
        *totalHx = ((ParallelGrid *) *Hx)->gatherFullGrid ();
      }
      if (doNeedHy)
      {
        *totalHy = ((ParallelGrid *) *Hy)->gatherFullGrid ();
      }
      if (doNeedHz)
      {
        *totalHz = ((ParallelGrid *) *Hz)->gatherFullGrid ();
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
                             bool doNeedEx, Grid<GridCoordinate3D> **Ex, Grid<GridCoordinate3D> **totalEx,
                             bool doNeedEy, Grid<GridCoordinate3D> **Ey, Grid<GridCoordinate3D> **totalEy,
                             bool doNeedEz, Grid<GridCoordinate3D> **Ez, Grid<GridCoordinate3D> **totalEz,
                             bool doNeedHx, Grid<GridCoordinate3D> **Hx, Grid<GridCoordinate3D> **totalHx,
                             bool doNeedHy, Grid<GridCoordinate3D> **Hy, Grid<GridCoordinate3D> **totalHy,
                             bool doNeedHz, Grid<GridCoordinate3D> **Hz, Grid<GridCoordinate3D> **totalHz)
  {
#ifdef PARALLEL_GRID
#ifdef GRID_3D
    if (totalInitialized)
    {
      if (doNeedEx)
      {
        *totalEx = ((ParallelGrid *) *Ex)->gatherFullGridPlacement (*totalEx);
      }
      if (doNeedEy)
      {
        *totalEy = ((ParallelGrid *) *Ey)->gatherFullGridPlacement (*totalEy);
      }
      if (doNeedEz)
      {
        *totalEz = ((ParallelGrid *) *Ez)->gatherFullGridPlacement (*totalEz);
      }

      if (doNeedHx)
      {
        *totalHx = ((ParallelGrid *) *Hx)->gatherFullGridPlacement (*totalHx);
      }
      if (doNeedHy)
      {
        *totalHy = ((ParallelGrid *) *Hy)->gatherFullGridPlacement (*totalHy);
      }
      if (doNeedHz)
      {
        *totalHz = ((ParallelGrid *) *Hz)->gatherFullGridPlacement (*totalHz);
      }
    }
    else
    {
      if (doNeedEx)
      {
        *totalEx = ((ParallelGrid *) *Ex)->gatherFullGrid ();
      }
      if (doNeedEy)
      {
        *totalEy = ((ParallelGrid *) *Ey)->gatherFullGrid ();
      }
      if (doNeedEz)
      {
        *totalEz = ((ParallelGrid *) *Ez)->gatherFullGrid ();
      }

      if (doNeedHx)
      {
        *totalHx = ((ParallelGrid *) *Hx)->gatherFullGrid ();
      }
      if (doNeedHy)
      {
        *totalHy = ((ParallelGrid *) *Hy)->gatherFullGrid ();
      }
      if (doNeedHz)
      {
        *totalHz = ((ParallelGrid *) *Hz)->gatherFullGrid ();
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

  static void initSigma (FieldPointValue *fieldValue, grid_coord dist, FPValue boundary, FPValue gridStep)
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

    fieldValue->setCurValue (getFieldValueRealOnly (val));
  }

  static FieldValue approximateIncidentWaveHelper (FPValue d, Grid<GridCoordinate1D> *FieldInc)
  {
    FPValue coordD1 = (FPValue) ((grid_coord) d);
    FPValue coordD2 = coordD1 + 1;
    FPValue proportionD2 = d - coordD1;
    FPValue proportionD1 = 1 - proportionD2;

    GridCoordinate1D pos1 ((grid_coord) coordD1
#ifdef DEBUG_INFO
                              , FieldInc->getSize ().getType1 ()
#endif
                          );
    GridCoordinate1D pos2 ((grid_coord) coordD2
#ifdef DEBUG_INFO
                              , FieldInc->getSize ().getType1 ()
#endif
                          );

    FieldPointValue *val1 = FieldInc->getFieldPointValue (pos1);
    FieldPointValue *val2 = FieldInc->getFieldPointValue (pos2);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    return proportionD1 * val1->getPrevValue () + proportionD2 * val2->getPrevValue ();
#else
    ALWAYS_ASSERT (0);
#endif
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

      FieldPointValue* valSigma = new FieldPointValue ();
      TCoord<FPValue, true> posAbs = layout->getEpsCoordFP (sigma->getTotalPosition (pos));

      TCoord<FPValue, true> size = layout->getEpsCoordFP (sigma->getTotalSize ());

      /*
       * TODO: add layout coordinates for material: sigma, eps, etc.
       */
      ASSERT (FPValue (grid_coord (posAbs.get1 () - FPValue (0.5))) == posAbs.get1 () - FPValue (0.5));
      if (posAbs.get1 () < PMLSizeX)
      {
        grid_coord dist = (grid_coord) (PMLSizeX - posAbs.get1 ());
        SchemeHelper::initSigma (valSigma, dist, boundary, dx);
      }
      else if (posAbs.get1 () >= size.get1 () - PMLSizeX)
      {
        grid_coord dist = (grid_coord) (posAbs.get1 () - (size.get1 () - PMLSizeX));
        SchemeHelper::initSigma (valSigma, dist, boundary, dx);
      }
      else
      {
        valSigma->setCurValue (getFieldValueRealOnly (FPValue (0)));
      }

      sigma->setFieldPointValue (valSigma, pos);
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

      FieldPointValue* valSigma = new FieldPointValue ();
      TCoord<FPValue, true> posAbs = layout->getEpsCoordFP (sigma->getTotalPosition (pos));

      TCoord<FPValue, true> size = layout->getEpsCoordFP (sigma->getTotalSize ());

      /*
       * TODO: add layout coordinates for material: sigma, eps, etc.
       */
      ASSERT (FPValue (grid_coord (posAbs.get2 () - FPValue (0.5))) == posAbs.get2 () - FPValue (0.5));
      if (posAbs.get2 () < PMLSizeY)
      {
        grid_coord dist = (grid_coord) (PMLSizeY - posAbs.get2 ());
        SchemeHelper::initSigma (valSigma, dist, boundary, dx);
      }
      else if (posAbs.get2 () >= size.get2 () - PMLSizeY)
      {
        grid_coord dist = (grid_coord) (posAbs.get2 () - (size.get2 () - PMLSizeY));
        SchemeHelper::initSigma (valSigma, dist, boundary, dx);
      }
      else
      {
        valSigma->setCurValue (getFieldValueRealOnly (FPValue (0)));
      }

      sigma->setFieldPointValue (valSigma, pos);
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

      FieldPointValue* valSigma = new FieldPointValue ();
      TCoord<FPValue, true> posAbs = layout->getEpsCoordFP (sigma->getTotalPosition (pos));

      TCoord<FPValue, true> size = layout->getEpsCoordFP (sigma->getTotalSize ());

      /*
       * TODO: add layout coordinates for material: sigma, eps, etc.
       */
      ASSERT (FPValue (grid_coord (posAbs.get3 () - FPValue (0.5))) == posAbs.get3 () - FPValue (0.5));
      if (posAbs.get3 () < PMLSizeZ)
      {
        grid_coord dist = (grid_coord) (PMLSizeZ - posAbs.get3 ());
        SchemeHelper::initSigma (valSigma, dist, boundary, dx);
      }
      else if (posAbs.get3 () >= size.get3 () - PMLSizeZ)
      {
        grid_coord dist = (grid_coord) (posAbs.get3 () - (size.get3 () - PMLSizeZ));
        SchemeHelper::initSigma (valSigma, dist, boundary, dx);
      }
      else
      {
        valSigma->setCurValue (getFieldValueRealOnly (FPValue (0)));
      }

      sigma->setFieldPointValue (valSigma, pos);
    }
  }

  static
  NPair ntffN3D_x (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YL3D_Dim3 *,
                   FPValue, FPValue,
                   Grid<GridCoordinate1D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *);
  static
  NPair ntffN3D_y (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YL3D_Dim3 *,
                   FPValue, FPValue,
                   Grid<GridCoordinate1D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *);
  static
  NPair ntffN3D_z (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YL3D_Dim3 *,
                   FPValue, FPValue,
                   Grid<GridCoordinate1D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *);
  static
  NPair ntffN3D (FPValue, FPValue,
                 GridCoordinate3D, GridCoordinate3D,
                 YL3D_Dim3 *,
                 FPValue, FPValue,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
                 Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);

  template <SchemeType_t Type, LayoutType layout_type>
  static
  NPair ntffN2D (FPValue, FPValue,
                 GridCoordinate2D, GridCoordinate2D,
                 YeeGridLayout<Type, GridCoordinate2DTemplate, layout_type> *,
                 FPValue, FPValue,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate2D> *, Grid<GridCoordinate2D> *,
                 Grid<GridCoordinate2D> *, Grid<GridCoordinate2D> *)
  {}

  template <SchemeType_t Type, LayoutType layout_type>
  static
  NPair ntffN1D (FPValue, FPValue,
                 GridCoordinate1D, GridCoordinate1D,
                 YeeGridLayout<Type, GridCoordinate1DTemplate, layout_type> *,
                 FPValue, FPValue,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate1D> *, Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate1D> *, Grid<GridCoordinate1D> *)
  {}

  static
  NPair ntffL3D_x (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YL3D_Dim3 *,
                   FPValue, FPValue,
                   Grid<GridCoordinate1D> *,
                   Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  static
  NPair ntffL3D_y (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YL3D_Dim3 *,
                   FPValue, FPValue,
                   Grid<GridCoordinate1D> *,
                   Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  static
  NPair ntffL3D_z (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YL3D_Dim3 *,
                   FPValue, FPValue,
                   Grid<GridCoordinate1D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *);
  static
  NPair ntffL3D (FPValue, FPValue,
                 GridCoordinate3D, GridCoordinate3D,
                 YL3D_Dim3 *,
                 FPValue, FPValue,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate3D> *,
                 Grid<GridCoordinate3D> *,
                 Grid<GridCoordinate3D> *);

  template <SchemeType_t Type, LayoutType layout_type>
  static
  NPair ntffL2D (FPValue, FPValue,
                 GridCoordinate2D, GridCoordinate2D,
                 YeeGridLayout<Type, GridCoordinate2DTemplate, layout_type> *,
                 FPValue, FPValue,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate2D> *,
                 Grid<GridCoordinate2D> *,
                 Grid<GridCoordinate2D> *)
  {}

  template <SchemeType_t Type, LayoutType layout_type>
  static
  NPair ntffL1D (FPValue, FPValue,
                 GridCoordinate1D, GridCoordinate1D,
                 YeeGridLayout<Type, GridCoordinate1DTemplate, layout_type> *,
                 FPValue, FPValue,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate1D> *)
  {}

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

  static bool doSkipBorderFunc1D (GridCoordinate1D pos, Grid<GridCoordinate1D> *grid)
  {
    return pos.get1 () != 0 && pos.get1 () != grid->getTotalSize ().get1 () - 1;
  }
  static bool doSkipBorderFunc2D (GridCoordinate2D pos, Grid<GridCoordinate2D> *grid)
  {
    return pos.get1 () != 0 && pos.get1 () != grid->getTotalSize ().get1 () - 1
           && pos.get2 () != 0 && pos.get2 () != grid->getTotalSize ().get2 () - 1;
  }
  static bool doSkipBorderFunc3D (GridCoordinate3D pos, Grid<GridCoordinate3D> *grid)
  {
    return pos.get1 () != 0 && pos.get1 () != grid->getTotalSize ().get1 () - 1
           && pos.get2 () != 0 && pos.get2 () != grid->getTotalSize ().get2 () - 1
           && pos.get3 () != 0 && pos.get3 () != grid->getTotalSize ().get3 () - 1;
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

  template <SchemeType_t Type, template <typename, bool> class TCoord>
  static
  FieldValue approximateIncidentWave (TCoord<FPValue, true>, TCoord<FPValue, true>, FPValue, Grid<GridCoordinate1D> *, FPValue, FPValue);

  template <SchemeType_t Type, template <typename, bool> class TCoord>
  static
  FieldValue approximateIncidentWaveE (TCoord<FPValue, true>, TCoord<FPValue, true>, Grid<GridCoordinate1D> *, FPValue, FPValue);

  template <SchemeType_t Type, template <typename, bool> class TCoord>
  static
  FieldValue approximateIncidentWaveH (TCoord<FPValue, true>, TCoord<FPValue, true>, Grid<GridCoordinate1D> *, FPValue, FPValue);

#ifdef PARALLEL_GRID
  template <SchemeType_t Type, LayoutType layout_type>
  static
  void allocateParallelGrids (ParallelYeeGridLayout<Type, layout_type> *pLayout,
                              bool doNeedEx, bool doNeedEy, bool doNeedEz,
                              bool doNeedHx, bool doNeedHy, bool doNeedHz,
                              bool doNeedSigmaX, bool doNeedSigmaY, bool doNeedSigmaZ,
                              ParallelGridCoordinate bufSize, ParallelGrid **Eps, ParallelGrid **Mu,
                              ParallelGrid **Ex, ParallelGrid **Ey, ParallelGrid **Ez,
                              ParallelGrid **Hx, ParallelGrid **Hy, ParallelGrid **Hz,
                              ParallelGrid **Dx, ParallelGrid **Dy, ParallelGrid **Dz,
                              ParallelGrid **Bx, ParallelGrid **By, ParallelGrid **Bz,
                              ParallelGrid **D1x, ParallelGrid **D1y, ParallelGrid **D1z,
                              ParallelGrid **B1x, ParallelGrid **B1y, ParallelGrid **B1z,
                              ParallelGrid **SigmaX, ParallelGrid **SigmaY, ParallelGrid **SigmaZ,
                              ParallelGrid **ExAmplitude, ParallelGrid **EyAmplitude, ParallelGrid **EzAmplitude,
                              ParallelGrid **HxAmplitude, ParallelGrid **HyAmplitude, ParallelGrid **HzAmplitude,
                              ParallelGrid **OmegaPE, ParallelGrid **GammaE,
                              ParallelGrid **OmegaPM, ParallelGrid **GammaM)
  {
    *Eps = new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "Eps");
    *Mu = new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getMuSizeForCurNode (), "Mu");

    *Ex = doNeedEx ? new ParallelGrid (pLayout->getExSize (), bufSize, 0, pLayout->getExSizeForCurNode (), "Ex") : NULLPTR;
    *Ey = doNeedEy ? new ParallelGrid (pLayout->getEySize (), bufSize, 0, pLayout->getEySizeForCurNode (), "Ey") : NULLPTR;
    *Ez = doNeedEz ? new ParallelGrid (pLayout->getEzSize (), bufSize, 0, pLayout->getEzSizeForCurNode (), "Ez") : NULLPTR;
    *Hx = doNeedHx ? new ParallelGrid (pLayout->getHxSize (), bufSize, 0, pLayout->getHxSizeForCurNode (), "Hx") : NULLPTR;
    *Hy = doNeedHy ? new ParallelGrid (pLayout->getHySize (), bufSize, 0, pLayout->getHySizeForCurNode (), "Hy") : NULLPTR;
    *Hz = doNeedHz ? new ParallelGrid (pLayout->getHzSize (), bufSize, 0, pLayout->getHzSizeForCurNode (), "Hz") : NULLPTR;

    if (SOLVER_SETTINGS.getDoUsePML ())
    {
      *Dx = doNeedEx ? new ParallelGrid (pLayout->getExSize (), bufSize, 0, pLayout->getExSizeForCurNode (), "Dx") : NULLPTR;
      *Dy = doNeedEy ? new ParallelGrid (pLayout->getEySize (), bufSize, 0, pLayout->getEySizeForCurNode (), "Dy") : NULLPTR;
      *Dz = doNeedEz ? new ParallelGrid (pLayout->getEzSize (), bufSize, 0, pLayout->getEzSizeForCurNode (), "Dz") : NULLPTR;
      *Bx = doNeedHx ? new ParallelGrid (pLayout->getHxSize (), bufSize, 0, pLayout->getHxSizeForCurNode (), "Bx") : NULLPTR;
      *By = doNeedHy ? new ParallelGrid (pLayout->getHySize (), bufSize, 0, pLayout->getHySizeForCurNode (), "By") : NULLPTR;
      *Bz = doNeedHz ? new ParallelGrid (pLayout->getHzSize (), bufSize, 0, pLayout->getHzSizeForCurNode (), "Bz") : NULLPTR;

      if (SOLVER_SETTINGS.getDoUseMetamaterials ())
      {
        *D1x = doNeedEx ? new ParallelGrid (pLayout->getExSize (), bufSize, 0, pLayout->getExSizeForCurNode (), "D1x") : NULLPTR;
        *D1y = doNeedEy ? new ParallelGrid (pLayout->getEySize (), bufSize, 0, pLayout->getEySizeForCurNode (), "D1y") : NULLPTR;
        *D1z = doNeedEz ? new ParallelGrid (pLayout->getEzSize (), bufSize, 0, pLayout->getEzSizeForCurNode (), "D1z") : NULLPTR;
        *B1x = doNeedHx ? new ParallelGrid (pLayout->getHxSize (), bufSize, 0, pLayout->getHxSizeForCurNode (), "B1x") : NULLPTR;
        *B1y = doNeedHy ? new ParallelGrid (pLayout->getHySize (), bufSize, 0, pLayout->getHySizeForCurNode (), "B1y") : NULLPTR;
        *B1z = doNeedHz ? new ParallelGrid (pLayout->getHzSize (), bufSize, 0, pLayout->getHzSizeForCurNode (), "B1z") : NULLPTR;
      }

      *SigmaX = doNeedSigmaX ? new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "SigmaX") : NULLPTR;
      *SigmaY = doNeedSigmaY ? new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "SigmaY") : NULLPTR;
      *SigmaZ = doNeedSigmaZ ? new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "SigmaZ") : NULLPTR;
    }

    if (SOLVER_SETTINGS.getDoUseAmplitudeMode ())
    {
      *ExAmplitude = doNeedEx ? new ParallelGrid (pLayout->getExSize (), bufSize, 0, pLayout->getExSizeForCurNode (), "ExAmp") : NULLPTR;
      *EyAmplitude = doNeedEy ? new ParallelGrid (pLayout->getEySize (), bufSize, 0, pLayout->getEySizeForCurNode (), "EyAmp") : NULLPTR;
      *EzAmplitude = doNeedEz ? new ParallelGrid (pLayout->getEzSize (), bufSize, 0, pLayout->getEzSizeForCurNode (), "EzAmp") : NULLPTR;
      *HxAmplitude = doNeedHx ? new ParallelGrid (pLayout->getHxSize (), bufSize, 0, pLayout->getHxSizeForCurNode (), "HxAmp") : NULLPTR;
      *HyAmplitude = doNeedHy ? new ParallelGrid (pLayout->getHySize (), bufSize, 0, pLayout->getHySizeForCurNode (), "HyAmp") : NULLPTR;
      *HzAmplitude = doNeedHz ? new ParallelGrid (pLayout->getHzSize (), bufSize, 0, pLayout->getHzSizeForCurNode (), "HzAmp") : NULLPTR;
    }

    if (SOLVER_SETTINGS.getDoUseMetamaterials ())
    {
      *OmegaPE = new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "OmegaPE");
      *GammaE = new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "GammaE");
      *OmegaPM = new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "OmegaPM");
      *GammaM = new ParallelGrid (pLayout->getEpsSize (), bufSize, 0, pLayout->getEpsSizeForCurNode (), "GammaM");
    }
  }
#endif
};

#endif /* SCHEME_H */
