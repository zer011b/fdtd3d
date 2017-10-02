#ifndef SCHEME_3D_H
#define SCHEME_3D_H

#include "GridInterface.h"
#include "PhysicsConst.h"
#include "Scheme.h"
#include "ParallelYeeGridLayout.h"

#ifdef GRID_3D

class Scheme3D: public Scheme
{
  YeeGridLayout *yeeLayout;

  Grid<GridCoordinate3D> *Ex;
  Grid<GridCoordinate3D> *Ey;
  Grid<GridCoordinate3D> *Ez;
  Grid<GridCoordinate3D> *Hx;
  Grid<GridCoordinate3D> *Hy;
  Grid<GridCoordinate3D> *Hz;

  Grid<GridCoordinate3D> *Dx;
  Grid<GridCoordinate3D> *Dy;
  Grid<GridCoordinate3D> *Dz;
  Grid<GridCoordinate3D> *Bx;
  Grid<GridCoordinate3D> *By;
  Grid<GridCoordinate3D> *Bz;

  Grid<GridCoordinate3D> *D1x;
  Grid<GridCoordinate3D> *D1y;
  Grid<GridCoordinate3D> *D1z;
  Grid<GridCoordinate3D> *B1x;
  Grid<GridCoordinate3D> *B1y;
  Grid<GridCoordinate3D> *B1z;

  Grid<GridCoordinate3D> *ExAmplitude;
  Grid<GridCoordinate3D> *EyAmplitude;
  Grid<GridCoordinate3D> *EzAmplitude;
  Grid<GridCoordinate3D> *HxAmplitude;
  Grid<GridCoordinate3D> *HyAmplitude;
  Grid<GridCoordinate3D> *HzAmplitude;

  Grid<GridCoordinate3D> *Eps;
  Grid<GridCoordinate3D> *Mu;

  Grid<GridCoordinate3D> *SigmaX;
  Grid<GridCoordinate3D> *SigmaY;
  Grid<GridCoordinate3D> *SigmaZ;

  Grid<GridCoordinate3D> *OmegaPE;
  Grid<GridCoordinate3D> *GammaE;

  Grid<GridCoordinate3D> *OmegaPM;
  Grid<GridCoordinate3D> *GammaM;

  Grid<GridCoordinate1D> *EInc;
  Grid<GridCoordinate1D> *HInc;

  Grid<GridCoordinate3D> *totalEx;
  Grid<GridCoordinate3D> *totalEy;
  Grid<GridCoordinate3D> *totalEz;
  Grid<GridCoordinate3D> *totalHx;
  Grid<GridCoordinate3D> *totalHy;
  Grid<GridCoordinate3D> *totalHz;

  bool totalInitialized;

  Grid<GridCoordinate3D> *totalEps;
  Grid<GridCoordinate3D> *totalMu;
  Grid<GridCoordinate3D> *totalOmegaPE;
  Grid<GridCoordinate3D> *totalOmegaPM;
  Grid<GridCoordinate3D> *totalGammaE;
  Grid<GridCoordinate3D> *totalGammaM;

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

  GridCoordinate3D leftNTFF;
  GridCoordinate3D rightNTFF;

  Dumper<GridCoordinate3D> *dumper[FILE_TYPE_COUNT];
  Loader<GridCoordinate3D> *loader[FILE_TYPE_COUNT];

  Dumper<GridCoordinate1D> *dumper1D[FILE_TYPE_COUNT];

private:

  void calculateExStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateEyStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateEzStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHxStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHyStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHzStep (time_step, GridCoordinate3D, GridCoordinate3D);

  void calculateExStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateEyStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateEzStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHxStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHyStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHzStepPML (time_step, GridCoordinate3D, GridCoordinate3D);

  FieldValue approximateIncidentWave (GridCoordinateFP3D, FPValue, Grid<GridCoordinate1D> &);
  FieldValue approximateIncidentWaveE (GridCoordinateFP3D);
  FieldValue approximateIncidentWaveH (GridCoordinateFP3D);

  void calculateExTFSF (GridCoordinate3D, FieldValue &, FieldValue &, FieldValue &, FieldValue &,
                        GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);
  void calculateEyTFSF (GridCoordinate3D, FieldValue &, FieldValue &, FieldValue &, FieldValue &,
                        GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);
  void calculateEzTFSF (GridCoordinate3D, FieldValue &, FieldValue &, FieldValue &, FieldValue &,
                        GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);
  void calculateHxTFSF (GridCoordinate3D, FieldValue &, FieldValue &, FieldValue &, FieldValue &,
                        GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);
  void calculateHyTFSF (GridCoordinate3D, FieldValue &, FieldValue &, FieldValue &, FieldValue &,
                        GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);
  void calculateHzTFSF (GridCoordinate3D, FieldValue &, FieldValue &, FieldValue &, FieldValue &,
                        GridCoordinate3D, GridCoordinate3D, GridCoordinate3D, GridCoordinate3D);

  void performExSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performEySteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performEzSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHxSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHySteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHzSteps (time_step, GridCoordinate3D, GridCoordinate3D);

  template<uint8_t EnumVal> void performPointSourceCalc (time_step);

  void performNSteps (time_step, time_step);
  void performAmplitudeSteps (time_step);

  int updateAmplitude (FPValue, FieldPointValue *, FPValue *);

  void performPlaneWaveESteps (time_step);
  void performPlaneWaveHSteps (time_step);

  void makeGridScattered (Grid<GridCoordinate3D> *, GridType);
  void gatherFieldsTotal (bool);
  void saveGrids (time_step);
  void saveNTFF (bool, time_step);

  void additionalUpdateOfGrids (time_step, time_step &);

  GridCoordinate3D getStartCoord (GridType, GridCoordinate3D);
  GridCoordinate3D getEndCoord (GridType, GridCoordinate3D);

public:

  virtual void performSteps () CXX11_OVERRIDE;

  void initScheme (FPValue, FPValue);

  void initGrids ();

  Scheme3D (YeeGridLayout *layout,
            const GridCoordinate3D& totSize,
            time_step tStep);

  ~Scheme3D ();

  struct NPair
  {
    FieldValue nTeta;
    FieldValue nPhi;

    NPair (FieldValue n_teta, FieldValue n_phi)
      : nTeta (n_teta)
    , nPhi (n_phi)
    {
    }

    NPair operator+ (const NPair &right)
    {
      return NPair (nTeta + right.nTeta, nPhi + right.nPhi);
    }
  };

  /*
   * 3D ntff
   */
  NPair ntffN_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  NPair ntffN_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  NPair ntffN_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);

  NPair ntffL_x (grid_coord x0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  NPair ntffL_y (grid_coord y0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  NPair ntffL_z (grid_coord z0, FPValue angleTeta, FPValue anglePhi, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);

  NPair ntffN (FPValue angleTeta, FPValue anglePhi,
               Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
               Grid<GridCoordinate3D> *);
  NPair ntffL (FPValue angleTeta, FPValue anglePhi,
               Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);

  FPValue Pointing_scat (FPValue angleTeta, FPValue anglePhi,
               Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *,
               Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *, Grid<GridCoordinate3D> *);
  FPValue Pointing_inc (FPValue angleTeta, FPValue anglePhi);
};

template<uint8_t EnumVal>
void
Scheme3D::performPointSourceCalc (time_step t)
{
  Grid<GridCoordinate3D> *grid = NULLPTR;

  switch (EnumVal)
  {
    case (static_cast<uint8_t> (GridType::EX)):
    {
      grid = Ex;
      break;
    }
    case (static_cast<uint8_t> (GridType::EY)):
    {
      grid = Ey;
      break;
    }
    case (static_cast<uint8_t> (GridType::EZ)):
    {
      grid = Ez;
      break;
    }
    case (static_cast<uint8_t> (GridType::HX)):
    {
      grid = Hx;
      break;
    }
    case (static_cast<uint8_t> (GridType::HY)):
    {
      grid = Hy;
      break;
    }
    case (static_cast<uint8_t> (GridType::HZ)):
    {
      grid = Hz;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  ASSERT (grid);

  GridCoordinate3D pos (solverSettings.getPointSourcePositionX (),
                        solverSettings.getPointSourcePositionY (),
                        solverSettings.getPointSourcePositionZ ());

  FieldPointValue* pointVal = grid->getFieldPointValueOrNullByAbsolutePos (pos);

  if (pointVal)
  {
#ifdef COMPLEX_FIELD_VALUES
    pointVal->setCurValue (FieldValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency),
                                       cos (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency)));
#else /* COMPLEX_FIELD_VALUES */
    pointVal->setCurValue (sin (gridTimeStep * t * 2 * PhysicsConst::Pi * sourceFrequency));
#endif /* !COMPLEX_FIELD_VALUES */
  }
}

#endif /* GRID_3D */

#endif /* SCHEME_3D_H */
