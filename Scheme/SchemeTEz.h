#ifndef SCHEME_TEZ_H
#define SCHEME_TEZ_H

#include "Scheme.h"
#include "ParallelGrid.h"
#include "Grid.h"
#include "YeeGridLayout.h"
#include "PhysicsConst.h"

#ifdef GRID_2D

class SchemeTEz: public Scheme
{
  YeeGridLayout yeeLayout;

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
  FPValue waveLength;
  FPValue stepWaveLength;
  FPValue frequency;

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

#ifdef CXX11_ENABLED
  virtual void performSteps (int) override;
#else
  virtual void performSteps (int);
#endif

  void initScheme (FPValue, FPValue);

  void initGrids ();

#if defined (PARALLEL_GRID)
  void initProcess (int);
#endif

#if defined (PARALLEL_GRID)
  SchemeTEz (const GridCoordinate2D& totSize,
             const GridCoordinate2D& bufSizeL,
             const GridCoordinate2D& bufSizeR,
             const int curProcess,
             const int totalProc,
             time_step tStep,
             bool calcAmp = false,
             time_step ampStep = 0,
             bool doUsePML = false,
             GridCoordinate2D sizePML = GridCoordinate2D (0, 0),
             bool doUseTFSF = false,
             GridCoordinate2D sizeScatteredZone = GridCoordinate2D (0, 0),
             FPValue angleIncWave = 0.0) :
    yeeLayout (totSize, sizePML, sizeScatteredZone, PhysicsConst::Pi / 2, angleIncWave, 0),
    Ex (shrinkCoord (yeeLayout.getEzSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Ey (shrinkCoord (yeeLayout.getHxSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Hz (shrinkCoord (yeeLayout.getHySize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Dx (shrinkCoord (yeeLayout.getEzSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Dy (shrinkCoord (yeeLayout.getHxSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Bz (shrinkCoord (yeeLayout.getHySize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    ExAmplitude (shrinkCoord (yeeLayout.getEzSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    EyAmplitude (shrinkCoord (yeeLayout.getHxSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    HzAmplitude (shrinkCoord (yeeLayout.getHySize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Eps (shrinkCoord (yeeLayout.getEpsSize ()), bufSizeL + GridCoordinate2D (1, 1), bufSizeR + GridCoordinate2D (1, 1), curProcess, totalProc, 0),
    Mu (shrinkCoord (yeeLayout.getMuSize ()), bufSizeL + GridCoordinate2D (1, 1), bufSizeR + GridCoordinate2D (1, 1), curProcess, totalProc, 0),
    SigmaX (shrinkCoord (yeeLayout.getEpsSize ()), bufSizeL + GridCoordinate2D (1, 1), bufSizeR + GridCoordinate2D (1, 1), curProcess, totalProc, 0),
    SigmaY (shrinkCoord (yeeLayout.getEpsSize ()), bufSizeL + GridCoordinate2D (1, 1), bufSizeR + GridCoordinate2D (1, 1), curProcess, totalProc, 0),
    SigmaZ (shrinkCoord (yeeLayout.getEpsSize ()), bufSizeL + GridCoordinate2D (1, 1), bufSizeR + GridCoordinate2D (1, 1), curProcess, totalProc, 0),
    waveLength (0),
    stepWaveLength (0),
    frequency (0),
    gridStep (0),
    gridTimeStep (0),
    totalStep (tStep),
    process (curProcess),
    calculateAmplitude (calcAmp),
    amplitudeStepLimit (ampStep),
    usePML (doUsePML),
    useTFSF (doUseTFSF),
    EInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY ())), 0),
    HInc (GridCoordinate1D ((grid_coord) 100*(totSize.getX () + totSize.getY ())), 0),
    incidentWaveAngle (angleIncWave)
#else
  SchemeTEz (const GridCoordinate2D& totSize,
             time_step tStep,
             bool calcAmp = false,
             time_step ampStep = 0,
             bool doUsePML = false,
             GridCoordinate2D sizePML = GridCoordinate2D (0, 0),
             bool doUseTFSF = false,
             GridCoordinate2D sizeScatteredZone = GridCoordinate2D (0, 0),
             FPValue angleIncWave = 0.0) :
    yeeLayout (totSize, sizePML, sizeScatteredZone, PhysicsConst::Pi / 2, angleIncWave, 0),
    Ex (shrinkCoord (yeeLayout.getEzSize ()), 0),
    Ey (shrinkCoord (yeeLayout.getHxSize ()), 0),
    Hz (shrinkCoord (yeeLayout.getHySize ()), 0),
    Dx (shrinkCoord (yeeLayout.getEzSize ()), 0),
    Dy (shrinkCoord (yeeLayout.getHxSize ()), 0),
    Bz (shrinkCoord (yeeLayout.getHySize ()), 0),
    ExAmplitude (shrinkCoord (yeeLayout.getEzSize ()), 0),
    EyAmplitude (shrinkCoord (yeeLayout.getHxSize ()), 0),
    HzAmplitude (shrinkCoord (yeeLayout.getHySize ()), 0),
    Eps (shrinkCoord (yeeLayout.getEpsSize ()), 0),
    Mu (shrinkCoord (yeeLayout.getMuSize ()), 0),
    SigmaX (shrinkCoord (yeeLayout.getEpsSize ()), 0),
    SigmaY (shrinkCoord (yeeLayout.getEpsSize ()), 0),
    SigmaZ (shrinkCoord (yeeLayout.getEpsSize ()), 0),
    waveLength (0),
    stepWaveLength (0),
    frequency (0),
    gridStep (0),
    gridTimeStep (0),
    totalStep (tStep),
    process (0),
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
                && sizeScatteredZone != GridCoordinate2D (0, 0)));

    ASSERT (!doUsePML || (doUsePML && (sizePML != GridCoordinate2D (0, 0))));

    ASSERT (!calculateAmplitude || calculateAmplitude && amplitudeStepLimit != 0);
  }

  ~SchemeTEz ()
  {
  }
};

#endif /* GRID_2D */

#endif /* SCHEME_TEZ_H */
