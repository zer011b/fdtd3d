#ifndef SCHEME_TMZ_H
#define SCHEME_TMZ_H

#include "Scheme.h"
#include "GridInterface.h"
#include "ParallelYeeGridLayout.h"
#include "PhysicsConst.h"

#ifdef GRID_2D

class SchemeTMz: public Scheme
{
  YeeGridLayout *yeeLayout;

#if defined (PARALLEL_GRID)
  ParallelGrid Ez;
  ParallelGrid Hx;
  ParallelGrid Hy;

  ParallelGrid Dz;
  ParallelGrid Bx;
  ParallelGrid By;

  ParallelGrid D1z;
  ParallelGrid B1x;
  ParallelGrid B1y;

  ParallelGrid EzAmplitude;
  ParallelGrid HxAmplitude;
  ParallelGrid HyAmplitude;

  ParallelGrid Eps;
  ParallelGrid Mu;

  ParallelGrid SigmaX;
  ParallelGrid SigmaY;
  ParallelGrid SigmaZ;

  ParallelGrid OmegaPE;
  ParallelGrid GammaE;

  ParallelGrid OmegaPM;
  ParallelGrid GammaM;
#else
  Grid<GridCoordinate2D> Ez;
  Grid<GridCoordinate2D> Hx;
  Grid<GridCoordinate2D> Hy;

  Grid<GridCoordinate2D> Dz;
  Grid<GridCoordinate2D> Bx;
  Grid<GridCoordinate2D> By;

  Grid<GridCoordinate2D> D1z;
  Grid<GridCoordinate2D> B1x;
  Grid<GridCoordinate2D> B1y;

  Grid<GridCoordinate2D> EzAmplitude;
  Grid<GridCoordinate2D> HxAmplitude;
  Grid<GridCoordinate2D> HyAmplitude;

  Grid<GridCoordinate2D> Eps;
  Grid<GridCoordinate2D> Mu;

  Grid<GridCoordinate2D> SigmaX;
  Grid<GridCoordinate2D> SigmaY;
  Grid<GridCoordinate2D> SigmaZ;

  Grid<GridCoordinate2D> OmegaPE;
  Grid<GridCoordinate2D> GammaE;

  Grid<GridCoordinate2D> OmegaPM;
  Grid<GridCoordinate2D> GammaM;
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

  bool useMetamaterials;

  bool dumpRes;

private:

  void calculateEzStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHxStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHyStep (time_step, GridCoordinate3D, GridCoordinate3D);

  void calculateEzStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHxStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHyStepPML (time_step, GridCoordinate3D, GridCoordinate3D);

  FieldValue approximateIncidentWave (GridCoordinateFP2D, FPValue, Grid<GridCoordinate1D> &);
  FieldValue approximateIncidentWaveE (GridCoordinateFP2D);
  FieldValue approximateIncidentWaveH (GridCoordinateFP2D);

  void performEzSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHxSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHySteps (time_step, GridCoordinate3D, GridCoordinate3D);
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
  SchemeTMz (ParallelYeeGridLayout *layout,
             const GridCoordinate2D& totSize,
             const GridCoordinate2D& bufSize,
             time_step tStep,
             bool calcAmp = false,
             time_step ampStep = 0,
             bool doUsePML = false,
             bool doUseTFSF = false,
             FPValue angleIncWave = 0.0,
             bool doUseMetamaterials = false,
             bool doDumpRes = false) :
    yeeLayout (layout),
    Ez (layout->getEzSize ().shrink (), bufSize, 0, layout->getEzSizeForCurNode ()),
    Hx (layout->getHxSize ().shrink (), bufSize, 0, layout->getHxSizeForCurNode ()),
    Hy (layout->getHySize ().shrink (), bufSize, 0, layout->getHySizeForCurNode ()),
    Dz (layout->getEzSize ().shrink (), bufSize, 0, layout->getEzSizeForCurNode ()),
    Bx (layout->getHxSize ().shrink (), bufSize, 0, layout->getHxSizeForCurNode ()),
    By (layout->getHySize ().shrink (), bufSize, 0, layout->getHySizeForCurNode ()),
    D1z (layout->getEzSize ().shrink (), bufSize, 0, layout->getEzSizeForCurNode ()),
    B1x (layout->getHxSize ().shrink (), bufSize, 0, layout->getHxSizeForCurNode ()),
    B1y (layout->getHySize ().shrink (), bufSize, 0, layout->getHySizeForCurNode ()),
    EzAmplitude (layout->getEzSize ().shrink (), bufSize, 0, layout->getEzSizeForCurNode ()),
    HxAmplitude (layout->getHxSize ().shrink (), bufSize, 0, layout->getHxSizeForCurNode ()),
    HyAmplitude (layout->getHySize ().shrink (), bufSize, 0, layout->getHySizeForCurNode ()),
    Eps (layout->getEpsSize ().shrink (), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode ()),
    Mu (layout->getEpsSize ().shrink (), bufSize + GridCoordinate2D (1, 1), 0, layout->getMuSizeForCurNode ()),
    OmegaPE (layout->getEpsSize ().shrink (), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode ()),
    GammaE (layout->getEpsSize ().shrink (), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode ()),
    OmegaPM (layout->getEpsSize ().shrink (), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode ()),
    GammaM (layout->getEpsSize ().shrink (), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode ()),
    SigmaX (layout->getEpsSize ().shrink (), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode ()),
    SigmaY (layout->getEpsSize ().shrink (), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode ()),
    SigmaZ (layout->getEpsSize ().shrink (), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode ()),
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
    useMetamaterials (doUseMetamaterials),
    dumpRes (doDumpRes)
#else
  SchemeTMz (YeeGridLayout *layout,
             const GridCoordinate2D& totSize,
             time_step tStep,
             bool calcAmp = false,
             time_step ampStep = 0,
             bool doUsePML = false,
             bool doUseTFSF = false,
             FPValue angleIncWave = 0.0,
             bool doUseMetamaterials = false,
             bool doDumpRes = false) :
    yeeLayout (layout),
    Ez (layout->getEzSize ().shrink (), 0),
    Hx (layout->getHxSize ().shrink (), 0),
    Hy (layout->getHySize ().shrink (), 0),
    Dz (layout->getEzSize ().shrink (), 0),
    Bx (layout->getHxSize ().shrink (), 0),
    By (layout->getHySize ().shrink (), 0),
    D1z (layout->getEzSize ().shrink (), 0),
    B1x (layout->getHxSize ().shrink (), 0),
    B1y (layout->getHySize ().shrink (), 0),
    EzAmplitude (layout->getEzSize ().shrink (), 0),
    HxAmplitude (layout->getHxSize ().shrink (), 0),
    HyAmplitude (layout->getHySize ().shrink (), 0),
    Eps (layout->getEpsSize ().shrink (), 0),
    Mu (layout->getEpsSize ().shrink (), 0),
    OmegaPE (layout->getEpsSize ().shrink (), 0),
    GammaE (layout->getEpsSize ().shrink (), 0),
    OmegaPM (layout->getEpsSize ().shrink (), 0),
    GammaM (layout->getEpsSize ().shrink (), 0),
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
    useMetamaterials (doUseMetamaterials),
    dumpRes (doDumpRes)
#endif
  {
    ASSERT (!doUseTFSF
            || (doUseTFSF
                && (yeeLayout->getIncidentWaveAngle1 () == PhysicsConst::Pi / 2
                    && (incidentWaveAngle == PhysicsConst::Pi / 4 || incidentWaveAngle == 0)
                    && yeeLayout->getIncidentWaveAngle3 () == PhysicsConst::Pi / 2)
                && yeeLayout->getSizeTFSF ().shrink () != GridCoordinate2D (0, 0)));

    ASSERT (!doUsePML || (doUsePML && (yeeLayout->getSizePML ().shrink () != GridCoordinate2D (0, 0))));

    ASSERT (!calculateAmplitude || calculateAmplitude && amplitudeStepLimit != 0);

#ifdef COMPLEX_FIELD_VALUES
    ASSERT (!calculateAmplitude);
#endif /* COMPLEX_FIELD_VALUES */
  }

  ~SchemeTMz ()
  {
  }
};

#endif /* GRID_2D */

#endif /* SCHEME_TMZ_H */
