#ifndef SCHEME_TEZ_H
#define SCHEME_TEZ_H

#include "Scheme.h"
#include "GridInterface.h"
#include "ParallelYeeGridLayout.h"
#include "PhysicsConst.h"

#ifdef GRID_2D

class SchemeTEz: public Scheme
{
  YeeGridLayout *yeeLayout;

#if defined (PARALLEL_GRID)
  ParallelGrid Ex;
  ParallelGrid Ey;
  ParallelGrid Hz;

  ParallelGrid Dx;
  ParallelGrid Dy;
  ParallelGrid Bz;

  ParallelGrid ExAmplitude;
  ParallelGrid EyAmplitude;
  ParallelGrid HzAmplitude;

  ParallelGrid Eps;
  ParallelGrid Mu;

  ParallelGrid SigmaX;
  ParallelGrid SigmaY;
  ParallelGrid SigmaZ;
#else
  Grid<GridCoordinate2D> Ex;
  Grid<GridCoordinate2D> Ey;
  Grid<GridCoordinate2D> Hz;

  Grid<GridCoordinate2D> Dx;
  Grid<GridCoordinate2D> Dy;
  Grid<GridCoordinate2D> Bz;

  Grid<GridCoordinate2D> ExAmplitude;
  Grid<GridCoordinate2D> EyAmplitude;
  Grid<GridCoordinate2D> HzAmplitude;

  Grid<GridCoordinate2D> Eps;
  Grid<GridCoordinate2D> Mu;

  Grid<GridCoordinate2D> SigmaX;
  Grid<GridCoordinate2D> SigmaY;
  Grid<GridCoordinate2D> SigmaZ;
#endif

  // Wave parameters
  FPValue sourceWaveLength;
  FPValue sourceFrequency;

  /** Courant number */
  FPValue courantNum;

  // dx
  FPValue gridStep;

  // dt
  FPValue gridTimeStep;

  time_step totalStep;

  int process;

  bool calculateAmplitude;

  time_step amplitudeStepLimit;

  bool usePML;

  bool useTFSF;

  Grid<GridCoordinate1D> EInc;
  Grid<GridCoordinate1D> HInc;

  FPValue incidentWaveAngle;

  bool dumpRes;

private:

  void calculateExStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateEyStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHzStep (time_step, GridCoordinate3D, GridCoordinate3D);

  void calculateExStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateEyStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHzStepPML (time_step, GridCoordinate3D, GridCoordinate3D);

  void performExSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performEySteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHzSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performNSteps (time_step, time_step);
  void performAmplitudeSteps (time_step);

  int updateAmplitude (FPValue, FieldPointValue *, FPValue *);

  void performPlaneWaveESteps (time_step);
  void performPlaneWaveHSteps (time_step);

public:

  virtual void performSteps () CXX11_OVERRIDE;

  void initScheme (FPValue, FPValue);

  void initGrids ();

#if defined (PARALLEL_GRID)
  SchemeTEz (ParallelYeeGridLayout *layout,
             const GridCoordinate2D& totSize,
             const GridCoordinate2D& bufSize,
             time_step tStep,
             bool calcAmp = false,
             time_step ampStep = 0,
             bool doUsePML = false,
             bool doUseTFSF = false,
             FPValue angleIncWave = 0.0,
             bool doDumpRes = false) :
    yeeLayout (layout),
    Ex (layout->getExSize ().shrink (), bufSize, 0, layout->getExSizeForCurNode ()),
    Ey (layout->getEySize ().shrink (), bufSize, 0, layout->getEySizeForCurNode ()),
    Hz (layout->getHzSize ().shrink (), bufSize, 0, layout->getHzSizeForCurNode ()),
    Dx (layout->getExSize ().shrink (), bufSize, 0, layout->getExSizeForCurNode ()),
    Dy (layout->getEySize ().shrink (), bufSize, 0, layout->getEySizeForCurNode ()),
    Bz (layout->getHzSize ().shrink (), bufSize, 0, layout->getHzSizeForCurNode ()),
    ExAmplitude (layout->getExSize ().shrink (), bufSize, 0, layout->getExSizeForCurNode ()),
    EyAmplitude (layout->getEySize ().shrink (), bufSize, 0, layout->getEySizeForCurNode ()),
    HzAmplitude (layout->getHzSize ().shrink (), bufSize, 0, layout->getHzSizeForCurNode ()),
    Eps (layout->getEpsSize ().shrink (), bufSize, 0, layout->getEpsSizeForCurNode ()),
    Mu (layout->getMuSize ().shrink (), bufSize, 0, layout->getMuSizeForCurNode ()),
    SigmaX (layout->getEpsSize ().shrink (), bufSize, 0, layout->getEpsSizeForCurNode ()),
    SigmaY (layout->getEpsSize ().shrink (), bufSize, 0, layout->getEpsSizeForCurNode ()),
    SigmaZ (layout->getEpsSize ().shrink (), bufSize, 0, layout->getEpsSizeForCurNode ()),
    sourceWaveLength (0),
    sourceFrequency (0),
    courantNum (0),
    gridStep (0),
    gridTimeStep (0),
    totalStep (tStep),
    calculateAmplitude (calcAmp),
    amplitudeStepLimit (ampStep),
    usePML (doUsePML),
    useTFSF (doUseTFSF),
    EInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY ())), 0),
    HInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY ())), 0),
    incidentWaveAngle (angleIncWave),
    dumpRes (doDumpRes)
#else
  SchemeTEz (YeeGridLayout *layout,
             const GridCoordinate2D& totSize,
             time_step tStep,
             bool calcAmp = false,
             time_step ampStep = 0,
             bool doUsePML = false,
             bool doUseTFSF = false,
             FPValue angleIncWave = 0.0,
             bool doDumpRes = false) :
    yeeLayout (layout),
    Ex (layout->getExSize ().shrink (), 0),
    Ey (layout->getEySize ().shrink (), 0),
    Hz (layout->getHzSize ().shrink (), 0),
    Dx (layout->getExSize ().shrink (), 0),
    Dy (layout->getEySize ().shrink (), 0),
    Bz (layout->getHzSize ().shrink (), 0),
    ExAmplitude (layout->getExSize ().shrink (), 0),
    EyAmplitude (layout->getEySize ().shrink (), 0),
    HzAmplitude (layout->getHzSize ().shrink (), 0),
    Eps (layout->getEpsSize ().shrink (), 0),
    Mu (layout->getMuSize ().shrink (), 0),
    SigmaX (layout->getEpsSize ().shrink (), 0),
    SigmaY (layout->getEpsSize ().shrink (), 0),
    SigmaZ (layout->getEpsSize ().shrink (), 0),
    sourceWaveLength (0),
    sourceFrequency (0),
    courantNum (0),
    gridStep (0),
    gridTimeStep (0),
    totalStep (tStep),
    calculateAmplitude (calcAmp),
    amplitudeStepLimit (ampStep),
    usePML (doUsePML),
    useTFSF (doUseTFSF),
    EInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY ())), 0),
    HInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY ())), 0),
    incidentWaveAngle (angleIncWave),
    dumpRes (doDumpRes)
#endif
  {
    ASSERT (!doUseTFSF
            || (doUseTFSF
                && (yeeLayout->getIncidentWaveAngle1 () == PhysicsConst::Pi / 2
                    && (incidentWaveAngle == PhysicsConst::Pi / 4 || incidentWaveAngle == 0)
                    && yeeLayout->getIncidentWaveAngle3 () == 0)
                && yeeLayout->getSizeTFSF ().shrink () != GridCoordinate2D (0, 0)));

    ASSERT (!doUsePML || (doUsePML && (yeeLayout->getSizePML ().shrink () != GridCoordinate2D (0, 0))));

    ASSERT (!calculateAmplitude || calculateAmplitude && amplitudeStepLimit != 0);

#ifdef COMPLEX_FIELD_VALUES
    ASSERT (!calculateAmplitude);
#endif /* COMPLEX_FIELD_VALUES */
  }

  ~SchemeTEz ()
  {
  }
};

#endif /* GRID_2D */

#endif /* SCHEME_TEZ_H */
