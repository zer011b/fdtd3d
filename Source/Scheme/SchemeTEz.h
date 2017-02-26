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
  void performNSteps (time_step, time_step, int);
  void performAmplitudeSteps (time_step, int);

  int updateAmplitude (FPValue, FieldPointValue *, FPValue *);

  void performPlaneWaveESteps (time_step);
  void performPlaneWaveHSteps (time_step);

public:

  virtual void performSteps (int) CXX11_OVERRIDE;

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
             FPValue angleIncWave = 0.0) :
    yeeLayout (layout),
    Ex (shrinkCoord (layout->getExSize ()), bufSize, 0, layout->getExSizeForCurNode (), layout->getExCoreSizePerNode ()),
    Ey (shrinkCoord (layout->getEySize ()), bufSize, 0, layout->getEySizeForCurNode (), layout->getEyCoreSizePerNode ()),
    Hz (shrinkCoord (layout->getHzSize ()), bufSize, 0, layout->getHzSizeForCurNode (), layout->getHzCoreSizePerNode ()),
    Dx (shrinkCoord (layout->getExSize ()), bufSize, 0, layout->getExSizeForCurNode (), layout->getExCoreSizePerNode ()),
    Dy (shrinkCoord (layout->getEySize ()), bufSize, 0, layout->getEySizeForCurNode (), layout->getEyCoreSizePerNode ()),
    Bz (shrinkCoord (layout->getHzSize ()), bufSize, 0, layout->getHzSizeForCurNode (), layout->getHzCoreSizePerNode ()),
    ExAmplitude (shrinkCoord (layout->getExSize ()), bufSize, 0, layout->getExSizeForCurNode (), layout->getExCoreSizePerNode ()),
    EyAmplitude (shrinkCoord (layout->getEySize ()), bufSize, 0, layout->getEySizeForCurNode (), layout->getEyCoreSizePerNode ()),
    HzAmplitude (shrinkCoord (layout->getHzSize ()), bufSize, 0, layout->getHzSizeForCurNode (), layout->getHzCoreSizePerNode ()),
    Eps (shrinkCoord (layout->getEpsSize ()), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    Mu (shrinkCoord (layout->getMuSize ()), bufSize + GridCoordinate2D (1, 1), 0, layout->getMuSizeForCurNode (), layout->getMuCoreSizePerNode ()),
    SigmaX (shrinkCoord (layout->getEpsSize ()), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    SigmaY (shrinkCoord (layout->getEpsSize ()), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
    SigmaZ (shrinkCoord (layout->getEpsSize ()), bufSize + GridCoordinate2D (1, 1), 0, layout->getEpsSizeForCurNode (), layout->getEpsCoreSizePerNode ()),
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
    incidentWaveAngle (angleIncWave)
#else
  SchemeTEz (YeeGridLayout *layout,
             const GridCoordinate2D& totSize,
             time_step tStep,
             bool calcAmp = false,
             time_step ampStep = 0,
             bool doUsePML = false,
             bool doUseTFSF = false,
             FPValue angleIncWave = 0.0) :
    yeeLayout (layout),
    Ex (shrinkCoord (layout->getExSize ()), 0),
    Ey (shrinkCoord (layout->getEySize ()), 0),
    Hz (shrinkCoord (layout->getHzSize ()), 0),
    Dx (shrinkCoord (layout->getExSize ()), 0),
    Dy (shrinkCoord (layout->getEySize ()), 0),
    Bz (shrinkCoord (layout->getHzSize ()), 0),
    ExAmplitude (shrinkCoord (layout->getExSize ()), 0),
    EyAmplitude (shrinkCoord (layout->getEySize ()), 0),
    HzAmplitude (shrinkCoord (layout->getHzSize ()), 0),
    Eps (shrinkCoord (layout->getEpsSize ()), 0),
    Mu (shrinkCoord (layout->getMuSize ()), 0),
    SigmaX (shrinkCoord (layout->getEpsSize ()), 0),
    SigmaY (shrinkCoord (layout->getEpsSize ()), 0),
    SigmaZ (shrinkCoord (layout->getEpsSize ()), 0),
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
    incidentWaveAngle (angleIncWave)
#endif
  {
    ASSERT (!doUseTFSF
            || (doUseTFSF
                && (incidentWaveAngle == PhysicsConst::Pi / 4 || incidentWaveAngle == 0)
                && shrinkCoord (yeeLayout->getSizeTFSF ()) != GridCoordinate2D (0, 0)));

    ASSERT (!doUsePML || (doUsePML && (shrinkCoord (yeeLayout->getSizePML ()) != GridCoordinate2D (0, 0))));

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
