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

typedef YeeGridLayout<GridCoordinate1DTemplate, LayoutType::E_CENTERED> YL1D;
typedef YeeGridLayout<GridCoordinate2DTemplate, LayoutType::E_CENTERED> YL2D;
typedef YeeGridLayout<GridCoordinate3DTemplate, LayoutType::E_CENTERED> YL3D;
#ifdef PARALLEL_GRID
typedef ParallelYeeGridLayout<LayoutType::E_CENTERED> PYL;
#endif /* PARALLEL_GRID */

// TODO: remove TCoord, as Layout has it defined
template <SchemeType Type, template <typename, bool> class TCoord, typename Layout>
class Scheme
{
  typedef TCoord<grid_coord, true> TC;
  typedef TCoord<grid_coord, false> TCS;
  typedef TCoord<FPValue, true> TCFP;
  typedef TCoord<FPValue, false> TCSFP;

protected:

  Layout *yeeLayout;

private:

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
  void initGridWithInitialVals (GridType, Grid<TC> *);

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

  void calculateTFSFExAsserts (TC, TC, TC, TC);
  void calculateTFSFEyAsserts (TC, TC, TC, TC);
  void calculateTFSFEzAsserts (TC, TC, TC, TC);
  void calculateTFSFHxAsserts (TC, TC, TC, TC);
  void calculateTFSFHyAsserts (TC, TC, TC, TC);
  void calculateTFSFHzAsserts (TC, TC, TC, TC);

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
    if (solverSettings.getDoUsePML ())
    {
      if (solverSettings.getDoUseMetamaterials ())
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
      if (solverSettings.getDoUseMetamaterials ())
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
        doUsePointSource = solverSettings.getDoUsePointSourceEx ();
        break;
      }
      case (static_cast<uint8_t> (GridType::EY)):
      {
        doUsePointSource = solverSettings.getDoUsePointSourceEy ();
        break;
      }
      case (static_cast<uint8_t> (GridType::EZ)):
      {
        doUsePointSource = solverSettings.getDoUsePointSourceEz ();
        break;
      }
      case (static_cast<uint8_t> (GridType::HX)):
      {
        doUsePointSource = solverSettings.getDoUsePointSourceHx ();
        break;
      }
      case (static_cast<uint8_t> (GridType::HY)):
      {
        doUsePointSource = solverSettings.getDoUsePointSourceHy ();
        break;
      }
      case (static_cast<uint8_t> (GridType::HZ)):
      {
        doUsePointSource = solverSettings.getDoUsePointSourceHz ();
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

  void initScheme (FPValue, FPValue);
  void initCallBacks ();
  void initGrids ();

  /**
   * Default constructor used for template instantiation
   */
  Scheme ()
  {
  }

  Scheme (Layout *layout,
          const TC& totSize,
          time_step tStep);

  ~Scheme ();
};

class SchemeHelper
{
public:

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

    GridCoordinate1D pos1 (coordD1);
    GridCoordinate1D pos2 (coordD2);

    FieldPointValue *val1 = FieldInc->getFieldPointValue (pos1);
    FieldPointValue *val2 = FieldInc->getFieldPointValue (pos2);

    return proportionD1 * val1->getPrevValue () + proportionD2 * val2->getPrevValue ();
  }

  template <template <typename, bool> class TCoord, typename Layout>
  static
  void initSigmaX (Layout *layout, FPValue dx, Grid< TCoord<grid_coord, true> > *sigma)
  {
    TCoord<grid_coord, true> PMLSize = layout->getLeftBorderPML () * (layout->getIsDoubleMaterialPrecision () ? 2 : 1);
    FPValue boundary = PMLSize.getX () * dx;

    for (grid_coord i = 0; i < sigma->getSize ().calculateTotalCoord (); ++i)
    {
      TCoord<grid_coord, true> pos = sigma->calculatePositionFromIndex (i);

      FieldPointValue* valSigma = new FieldPointValue ();
      TCoord<FPValue, true> posAbs = layout->getEpsCoordFP (sigma->getTotalPosition (pos));

      TCoord<FPValue, true> size = layout->getEpsCoordFP (sigma->getTotalSize ());
      grid_coord dist;

      /*
       * TODO: add layout coordinates for material: sigma, eps, etc.
       */
      if (posAbs.getX () < PMLSize.getX ())
      {
        dist = PMLSize.getX () - posAbs.getX ();
      }
      else if (posAbs.getX () >= size.getX () - PMLSize.getX ())
      {
        dist = posAbs.getX () - (size.getX () - PMLSize.getX ());
      }

      SchemeHelper::initSigma (valSigma, dist, boundary, dx);
      sigma->setFieldPointValue (valSigma, pos);
    }
  }

  template <template <typename, bool> class TCoord, typename Layout>
  static
  void initSigmaY (Layout *layout, FPValue dx, Grid< TCoord<grid_coord, true> > *sigma)
  {
    TCoord<grid_coord, true> PMLSize = layout->getLeftBorderPML () * (layout->getIsDoubleMaterialPrecision () ? 2 : 1);
    FPValue boundary = PMLSize.getY () * dx;

    for (grid_coord i = 0; i < sigma->getSize ().calculateTotalCoord (); ++i)
    {
      TCoord<grid_coord, true> pos = sigma->calculatePositionFromIndex (i);

      FieldPointValue* valSigma = new FieldPointValue ();
      TCoord<FPValue, true> posAbs = layout->getEpsCoordFP (sigma->getTotalPosition (pos));

      TCoord<FPValue, true> size = layout->getEpsCoordFP (sigma->getTotalSize ());
      grid_coord dist;

      /*
       * TODO: add layout coordinates for material: sigma, eps, etc.
       */
      if (posAbs.getY () < PMLSize.getY ())
      {
        dist = PMLSize.getY () - posAbs.getY ();
      }
      else if (posAbs.getY () >= size.getY () - PMLSize.getY ())
      {
        dist = posAbs.getY () - (size.getY () - PMLSize.getY ());
      }

      SchemeHelper::initSigma (valSigma, dist, boundary, dx);
      sigma->setFieldPointValue (valSigma, pos);
    }
  }

  template <template <typename, bool> class TCoord, typename Layout>
  static
  void initSigmaZ (Layout *layout, FPValue dx, Grid< TCoord<grid_coord, true> > *sigma)
  {
    TCoord<grid_coord, true> PMLSize = layout->getLeftBorderPML () * (layout->getIsDoubleMaterialPrecision () ? 2 : 1);
    FPValue boundary = PMLSize.getZ () * dx;

    for (grid_coord i = 0; i < sigma->getSize ().calculateTotalCoord (); ++i)
    {
      TCoord<grid_coord, true> pos = sigma->calculatePositionFromIndex (i);

      FieldPointValue* valSigma = new FieldPointValue ();
      TCoord<FPValue, true> posAbs = layout->getEpsCoordFP (sigma->getTotalPosition (pos));

      TCoord<FPValue, true> size = layout->getEpsCoordFP (sigma->getTotalSize ());
      grid_coord dist;

      /*
       * TODO: add layout coordinates for material: sigma, eps, etc.
       */
      if (posAbs.getZ () < PMLSize.getZ ())
      {
        dist = PMLSize.getZ () - posAbs.getZ ();
      }
      else if (posAbs.getZ () >= size.getZ () - PMLSize.getZ ())
      {
        dist = posAbs.getZ () - (size.getZ () - PMLSize.getZ ());
      }

      SchemeHelper::initSigma (valSigma, dist, boundary, dx);
      sigma->setFieldPointValue (valSigma, pos);
    }
  }

  static
  NPair ntffN3D_x (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YL3D *,
                   FPValue, FPValue,
                   Grid<GridCoordinate1D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *);
  static
  NPair ntffN3D_y (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YL3D *,
                   FPValue, FPValue,
                   Grid<GridCoordinate1D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *);
  static
  NPair ntffN3D_z (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YL3D *,
                   FPValue, FPValue,
                   Grid<GridCoordinate1D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *);
  static
  NPair ntffN3D (FPValue, FPValue,
                 GridCoordinate3D, GridCoordinate3D,
                 YL3D *,
                 FPValue, FPValue,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
                 Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  static
  NPair ntffN2D (FPValue, FPValue,
                 GridCoordinate2D, GridCoordinate2D,
                 YL2D *,
                 FPValue, FPValue,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate2D> *, Grid<GridCoordinate2D> *,
                 Grid<GridCoordinate2D> *, Grid<GridCoordinate2D> *)
  {}
  static
  NPair ntffN1D (FPValue, FPValue,
                 GridCoordinate1D, GridCoordinate1D,
                 YL1D *,
                 FPValue, FPValue,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate1D> *, Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate1D> *, Grid<GridCoordinate1D> *)
  {}
  static
  NPair ntffL3D_x (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YL3D *,
                   FPValue, FPValue,
                   Grid<GridCoordinate1D> *,
                   Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  static
  NPair ntffL3D_y (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YL3D *,
                   FPValue, FPValue,
                   Grid<GridCoordinate1D> *,
                   Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  static
  NPair ntffL3D_z (grid_coord, FPValue, FPValue,
                   GridCoordinate3D, GridCoordinate3D,
                   YL3D *,
                   FPValue, FPValue,
                   Grid<GridCoordinate1D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *,
                   Grid<GridCoordinate3D> *);
  static
  NPair ntffL3D (FPValue, FPValue,
                 GridCoordinate3D, GridCoordinate3D,
                 YL3D *,
                 FPValue, FPValue,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate3D> *,
                 Grid<GridCoordinate3D> *,
                 Grid<GridCoordinate3D> *);
  static
  NPair ntffL2D (FPValue, FPValue,
                 GridCoordinate2D, GridCoordinate2D,
                 YL2D *,
                 FPValue, FPValue,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate2D> *,
                 Grid<GridCoordinate2D> *,
                 Grid<GridCoordinate2D> *)
  {}
  static
  NPair ntffL1D (FPValue, FPValue,
                 GridCoordinate1D, GridCoordinate1D,
                 YL1D *,
                 FPValue, FPValue,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate1D> *,
                 Grid<GridCoordinate1D> *)
  {}

  static grid_coord getStartCoordOrthX (GridCoordinate3D size)
  {
    return size.getX () / 2;
  }
  static grid_coord getStartCoordOrthY (GridCoordinate3D size)
  {
    return size.getY () / 2;
  }
  static grid_coord getStartCoordOrthZ (GridCoordinate3D size)
  {
    return size.getZ () / 2;
  }

  static grid_coord getEndCoordOrthX (GridCoordinate3D size)
  {
    return size.getX () / 2 + 1;
  }
  static grid_coord getEndCoordOrthY (GridCoordinate3D size)
  {
    return size.getY () / 2 + 1;
  }
  static grid_coord getEndCoordOrthZ (GridCoordinate3D size)
  {
    return size.getZ () / 2 + 1;
  }

  static bool doSkipBorderFunc1D (GridCoordinate1D pos, Grid<GridCoordinate1D> *grid)
  {
    return pos.getX () != 0 && pos.getX () != grid->getTotalSize ().getX () - 1;
  }
  static bool doSkipBorderFunc2D (GridCoordinate2D pos, Grid<GridCoordinate2D> *grid)
  {
    return pos.getX () != 0 && pos.getX () != grid->getTotalSize ().getX () - 1
           && pos.getY () != 0 && pos.getY () != grid->getTotalSize ().getY () - 1;
  }
  static bool doSkipBorderFunc3D (GridCoordinate3D pos, Grid<GridCoordinate3D> *grid)
  {
    return pos.getX () != 0 && pos.getX () != grid->getTotalSize ().getX () - 1
           && pos.getY () != 0 && pos.getY () != grid->getTotalSize ().getY () - 1
           && pos.getZ () != 0 && pos.getZ () != grid->getTotalSize ().getZ () - 1;
  }

  static bool doSkipMakeScattered1D (GridCoordinateFP1D pos, GridCoordinate1D left, GridCoordinate1D right)
  {
    GridCoordinateFP1D leftTFSF = convertCoord (left);
    GridCoordinateFP1D rightTFSF = convertCoord (right);
    return pos.getX () < leftTFSF.getX () || pos.getX () > rightTFSF.getX ();
  }
  static bool doSkipMakeScattered2D (GridCoordinateFP2D pos, GridCoordinate2D left, GridCoordinate2D right)
  {
    GridCoordinateFP2D leftTFSF = convertCoord (left);
    GridCoordinateFP2D rightTFSF = convertCoord (right);
    return pos.getX () < leftTFSF.getX () || pos.getX () > rightTFSF.getX ()
           || pos.getY () < leftTFSF.getY () || pos.getY () > rightTFSF.getY ();
  }
  static bool doSkipMakeScattered3D (GridCoordinateFP3D pos, GridCoordinate3D left, GridCoordinate3D right)
  {
    GridCoordinateFP3D leftTFSF = convertCoord (left);
    GridCoordinateFP3D rightTFSF = convertCoord (right);
    return pos.getX () < leftTFSF.getX () || pos.getX () > rightTFSF.getX ()
           || pos.getY () < leftTFSF.getY () || pos.getY () > rightTFSF.getY ()
           || pos.getZ () < leftTFSF.getZ () || pos.getZ () > rightTFSF.getZ ();
  }

  static GridCoordinate1D getStartCoordRes1D (OrthogonalAxis orthogonalAxis, GridCoordinate1D start, GridCoordinate1D size)
  {
    if (orthogonalAxis == OrthogonalAxis::Z)
    {
      return GridCoordinate1D (start.getX ());
    }
    else if (orthogonalAxis == OrthogonalAxis::Y)
    {
      return GridCoordinate1D (start.getX ());
    }
    else if (orthogonalAxis == OrthogonalAxis::X)
    {
      return GridCoordinate1D (SchemeHelper::getStartCoordOrthX (expandTo3D (size)));
    }
  }
  static GridCoordinate1D getEndCoordRes1D (OrthogonalAxis orthogonalAxis, GridCoordinate1D end, GridCoordinate1D size)
  {
    if (orthogonalAxis == OrthogonalAxis::Z)
    {
      return GridCoordinate1D (end.getX ());
    }
    else if (orthogonalAxis == OrthogonalAxis::Y)
    {
      return GridCoordinate1D (end.getX ());
    }
    else if (orthogonalAxis == OrthogonalAxis::X)
    {
      return GridCoordinate1D (SchemeHelper::getEndCoordOrthX (expandTo3D (size)));
    }
  }

  static GridCoordinate2D getStartCoordRes2D (OrthogonalAxis orthogonalAxis, GridCoordinate2D start, GridCoordinate2D size)
  {
    if (orthogonalAxis == OrthogonalAxis::Z)
    {
      return GridCoordinate2D (start.getX (), start.getY ());
    }
    else if (orthogonalAxis == OrthogonalAxis::Y)
    {
      return GridCoordinate2D (start.getX (), SchemeHelper::getStartCoordOrthY (expandTo3D (size)));
    }
    else if (orthogonalAxis == OrthogonalAxis::X)
    {
      return GridCoordinate2D (SchemeHelper::getStartCoordOrthX (expandTo3D (size)), start.getY ());
    }
  }
  static GridCoordinate2D getEndCoordRes2D (OrthogonalAxis orthogonalAxis, GridCoordinate2D end, GridCoordinate2D size)
  {
    if (orthogonalAxis == OrthogonalAxis::Z)
    {
      return GridCoordinate2D (end.getX (), end.getY ());
    }
    else if (orthogonalAxis == OrthogonalAxis::Y)
    {
      return GridCoordinate2D (end.getX (), SchemeHelper::getEndCoordOrthY (expandTo3D (size)));
    }
    else if (orthogonalAxis == OrthogonalAxis::X)
    {
      return GridCoordinate2D (SchemeHelper::getEndCoordOrthX (expandTo3D (size)), end.getY ());
    }
  }

  static GridCoordinate3D getStartCoordRes3D (OrthogonalAxis orthogonalAxis, GridCoordinate3D start, GridCoordinate3D size)
  {
    if (orthogonalAxis == OrthogonalAxis::Z)
    {
      return GridCoordinate3D (start.getX (), start.getY (), SchemeHelper::getStartCoordOrthZ (expandTo3D (size)));
    }
    else if (orthogonalAxis == OrthogonalAxis::Y)
    {
      return GridCoordinate3D (start.getX (), SchemeHelper::getStartCoordOrthY (expandTo3D (size)), start.getZ ());
    }
    else if (orthogonalAxis == OrthogonalAxis::X)
    {
      return GridCoordinate3D (SchemeHelper::getStartCoordOrthX (expandTo3D (size)), start.getY (), start.getZ ());
    }
  }
  static GridCoordinate3D getEndCoordRes3D (OrthogonalAxis orthogonalAxis, GridCoordinate3D end, GridCoordinate3D size)
  {
    if (orthogonalAxis == OrthogonalAxis::Z)
    {
      return GridCoordinate3D (end.getX (), end.getY (), SchemeHelper::getEndCoordOrthZ (expandTo3D (size)));
    }
    else if (orthogonalAxis == OrthogonalAxis::Y)
    {
      return GridCoordinate3D (end.getX (), SchemeHelper::getEndCoordOrthY (expandTo3D (size)), end.getZ ());
    }
    else if (orthogonalAxis == OrthogonalAxis::X)
    {
      return GridCoordinate3D (SchemeHelper::getEndCoordOrthX (expandTo3D (size)), end.getY (), end.getZ ());
    }
  }

  template <template <typename, bool> class TCoord>
  static
  FieldValue approximateIncidentWave (TCoord<FPValue, true>, TCoord<FPValue, true>, FPValue, Grid<GridCoordinate1D> *, FPValue, FPValue);

  template <template <typename, bool> class TCoord>
  static
  FieldValue approximateIncidentWaveE (TCoord<FPValue, true>, TCoord<FPValue, true>, Grid<GridCoordinate1D> *, FPValue, FPValue);

  template <template <typename, bool> class TCoord>
  static
  FieldValue approximateIncidentWaveH (TCoord<FPValue, true>, TCoord<FPValue, true>, Grid<GridCoordinate1D> *, FPValue, FPValue);

  static void calculateTFSFExAsserts1D (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
  {
    ASSERT (pos11.getX () == pos12.getX ());
    ASSERT (pos21.getX () == pos22.getX ());
  }
  static void calculateTFSFExAsserts2D (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
  {
    ASSERT (pos11.getX () == pos12.getX ());
    ASSERT (pos21.getX () == pos22.getX ());
    ASSERT (pos11.getY () < pos12.getY ());
    ASSERT (pos21.getY () == pos22.getY ());
  }
  static void calculateTFSFExAsserts3D (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
  {
    ASSERT (pos11.getX () == pos12.getX ());
    ASSERT (pos21.getX () == pos22.getX ());
    ASSERT (pos11.getY () < pos12.getY ());
    ASSERT (pos21.getY () == pos22.getY ());
    ASSERT (pos11.getZ () == pos12.getZ ());
    ASSERT (pos21.getZ () < pos22.getZ ());
  }

  static void calculateTFSFEyAsserts1D (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
  {
    ASSERT (pos11.getX () == pos12.getX ());
    ASSERT (pos21.getX () < pos22.getX ());
  }
  static void calculateTFSFEyAsserts2D (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
  {
    ASSERT (pos11.getX () == pos12.getX ());
    ASSERT (pos21.getX () < pos22.getX ());
    ASSERT (pos11.getY () == pos12.getY ());
    ASSERT (pos21.getY () == pos22.getY ());
  }
  static void calculateTFSFEyAsserts3D (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
  {
    ASSERT (pos11.getX () == pos12.getX ());
    ASSERT (pos21.getX () < pos22.getX ());
    ASSERT (pos11.getY () == pos12.getY ());
    ASSERT (pos21.getY () == pos22.getY ());
    ASSERT (pos11.getZ () < pos12.getZ ());
    ASSERT (pos21.getZ () == pos22.getZ ());
  }

  static void calculateTFSFEzAsserts1D (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
  {
    ASSERT (pos11.getX () < pos12.getX ());
    ASSERT (pos21.getX () == pos22.getX ());
  }
  static void calculateTFSFEzAsserts2D (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
  {
    ASSERT (pos11.getX () < pos12.getX ());
    ASSERT (pos21.getX () == pos22.getX ());
    ASSERT (pos11.getY () == pos12.getY ());
    ASSERT (pos21.getY () < pos22.getY ());
  }
  static void calculateTFSFEzAsserts3D (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
  {
    ASSERT (pos11.getX () < pos12.getX ());
    ASSERT (pos21.getX () == pos22.getX ());
    ASSERT (pos11.getY () == pos12.getY ());
    ASSERT (pos21.getY () < pos22.getY ());
    ASSERT (pos11.getZ () == pos12.getZ ());
    ASSERT (pos21.getZ () == pos22.getZ ());
  }

  static void calculateTFSFHxAsserts1D (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
  {
    ASSERT (pos11.getX () == pos12.getX ());
    ASSERT (pos21.getX () == pos22.getX ());
  }
  static void calculateTFSFHxAsserts2D (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
  {
    ASSERT (pos11.getX () == pos12.getX ());
    ASSERT (pos21.getX () == pos22.getX ());
    ASSERT (pos11.getY () == pos12.getY ());
    ASSERT (pos21.getY () < pos22.getY ());
  }
  static void calculateTFSFHxAsserts3D (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
  {
    ASSERT (pos11.getX () == pos12.getX ());
    ASSERT (pos21.getX () == pos22.getX ());
    ASSERT (pos11.getY () == pos12.getY ());
    ASSERT (pos21.getY () < pos22.getY ());
    ASSERT (pos11.getZ () < pos12.getZ ());
    ASSERT (pos21.getZ () == pos22.getZ ());
  }

  static void calculateTFSFHyAsserts1D (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
  {
    ASSERT (pos11.getX () < pos12.getX ());
    ASSERT (pos21.getX () == pos22.getX ());
  }
  static void calculateTFSFHyAsserts2D (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
  {
    ASSERT (pos11.getX () < pos12.getX ());
    ASSERT (pos21.getX () == pos22.getX ());
    ASSERT (pos11.getY () == pos12.getY ());
    ASSERT (pos21.getY () == pos22.getY ());
  }
  static void calculateTFSFHyAsserts3D (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
  {
    ASSERT (pos11.getX () < pos12.getX ());
    ASSERT (pos21.getX () == pos22.getX ());
    ASSERT (pos11.getY () == pos12.getY ());
    ASSERT (pos21.getY () == pos22.getY ());
    ASSERT (pos11.getZ () == pos12.getZ ());
    ASSERT (pos21.getZ () < pos22.getZ ());
  }

  static void calculateTFSFHzAsserts1D (GridCoordinate1D pos11, GridCoordinate1D pos12, GridCoordinate1D pos21, GridCoordinate1D pos22)
  {
    ASSERT (pos11.getX () == pos12.getX ());
    ASSERT (pos21.getX () < pos22.getX ());
  }
  static void calculateTFSFHzAsserts2D (GridCoordinate2D pos11, GridCoordinate2D pos12, GridCoordinate2D pos21, GridCoordinate2D pos22)
  {
    ASSERT (pos11.getX () == pos12.getX ());
    ASSERT (pos21.getX () < pos22.getX ());
    ASSERT (pos11.getY () < pos12.getY ());
    ASSERT (pos21.getY () == pos22.getY ());
  }
  static void calculateTFSFHzAsserts3D (GridCoordinate3D pos11, GridCoordinate3D pos12, GridCoordinate3D pos21, GridCoordinate3D pos22)
  {
    ASSERT (pos11.getX () == pos12.getX ());
    ASSERT (pos21.getX () < pos22.getX ());
    ASSERT (pos11.getY () < pos12.getY ());
    ASSERT (pos21.getY () == pos22.getY ());
    ASSERT (pos11.getZ () == pos12.getZ ());
    ASSERT (pos21.getZ () == pos22.getZ ());
  }
};

#endif /* SCHEME_H */
