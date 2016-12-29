#ifndef SCHEME_3D_H
#define SCHEME_3D_H

#include "Grid.h"
#include "ParallelGrid.h"
#include "PhysicsConst.h"
#include "Scheme.h"
#include "YeeGridLayout.h"

#ifdef GRID_3D

class Scheme3D: public Scheme
{
  YeeGridLayout yeeLayout;

#if defined (PARALLEL_GRID)
  ParallelGrid Ex;
  ParallelGrid Ey;
  ParallelGrid Ez;
  ParallelGrid Hx;
  ParallelGrid Hy;
  ParallelGrid Hz;

  ParallelGrid Dx;
  ParallelGrid Dy;
  ParallelGrid Dz;
  ParallelGrid Bx;
  ParallelGrid By;
  ParallelGrid Bz;

  ParallelGrid ExAmplitude;
  ParallelGrid EyAmplitude;
  ParallelGrid EzAmplitude;
  ParallelGrid HxAmplitude;
  ParallelGrid HyAmplitude;
  ParallelGrid HzAmplitude;

  ParallelGrid Eps;
  ParallelGrid Mu;

  ParallelGrid SigmaX;
  ParallelGrid SigmaY;
  ParallelGrid SigmaZ;
#else
  Grid<GridCoordinate3D> Ex;
  Grid<GridCoordinate3D> Ey;
  Grid<GridCoordinate3D> Ez;
  Grid<GridCoordinate3D> Hx;
  Grid<GridCoordinate3D> Hy;
  Grid<GridCoordinate3D> Hz;

  Grid<GridCoordinate3D> Dx;
  Grid<GridCoordinate3D> Dy;
  Grid<GridCoordinate3D> Dz;
  Grid<GridCoordinate3D> Bx;
  Grid<GridCoordinate3D> By;
  Grid<GridCoordinate3D> Bz;

  Grid<GridCoordinate3D> ExAmplitude;
  Grid<GridCoordinate3D> EyAmplitude;
  Grid<GridCoordinate3D> EzAmplitude;
  Grid<GridCoordinate3D> HxAmplitude;
  Grid<GridCoordinate3D> HyAmplitude;
  Grid<GridCoordinate3D> HzAmplitude;

  Grid<GridCoordinate3D> Eps;
  Grid<GridCoordinate3D> Mu;

  Grid<GridCoordinate3D> SigmaX;
  Grid<GridCoordinate3D> SigmaY;
  Grid<GridCoordinate3D> SigmaZ;
#endif

  // Wave parameters
  FieldValue waveLength;
  FieldValue stepWaveLength;
  FieldValue frequency;

  // dx
  FieldValue gridStep;

  // dt
  FieldValue gridTimeStep;

  time_step totalStep;

  int process;

  bool calculateAmplitude;

  time_step amplitudeStepLimit;

  bool usePML;

  bool useTFSF;

  Grid<GridCoordinate1D> EInc;
  Grid<GridCoordinate1D> HInc;

  FieldValue incidentWaveAngle1; // Teta
  FieldValue incidentWaveAngle2; // Phi
  FieldValue incidentWaveAngle3; // Psi

private:

  void performExSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performEySteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performEzSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHxSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHySteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHzSteps (time_step, GridCoordinate3D, GridCoordinate3D);

  void performNSteps (time_step, time_step, int);
  void performAmplitudeSteps (time_step, int);

  int updateAmplitude (FieldValue, FieldPointValue *, FieldValue *);

  void performPlaneWaveESteps (time_step);
  void performPlaneWaveHSteps (time_step);

public:

#ifdef CXX11_ENABLED
  virtual void performSteps (int) override;
#else
  virtual void performSteps (int);
#endif

  void initScheme (FieldValue, FieldValue);

  void initGrids ();

#if defined (PARALLEL_GRID)
  void initProcess (int);
#endif

#if defined (PARALLEL_GRID)
  Scheme3D (const GridCoordinate3D& totSize,
            const GridCoordinate3D& bufSizeL,
            const GridCoordinate3D& bufSizeR,
            const int curProcess,
            const int totalProc,
            time_step tStep,
            bool calcAmp = false,
            time_step ampStep = 0,
            bool doUsePML = false,
            GridCoordinate3D sizePML = GridCoordinate3D (0, 0, 0),
            bool doUseTFSF = false,
            GridCoordinate3D sizeScatteredZone = GridCoordinate3D (0, 0, 0),
            FieldValue angleIncWave1 = 0.0,
            FieldValue angleIncWave2 = 0.0,
            FieldValue angleIncWave3 = 0.0) :
    yeeLayout (totSize, sizePML, sizeScatteredZone, angleIncWave1, angleIncWave2, angleIncWave3),
    Ex (yeeLayout.getExSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Ey (yeeLayout.getEySize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Ez (yeeLayout.getEzSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Hx (yeeLayout.getHxSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Hy (yeeLayout.getHySize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Hz (yeeLayout.getHzSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Dx (yeeLayout.getExSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Dy (yeeLayout.getEySize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Dz (yeeLayout.getEzSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Bx (yeeLayout.getHxSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    By (yeeLayout.getHySize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Bz (yeeLayout.getHzSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    ExAmplitude (yeeLayout.getExSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    EyAmplitude (yeeLayout.getEySize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    EzAmplitude (yeeLayout.getEzSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    HxAmplitude (yeeLayout.getHxSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    HyAmplitude (yeeLayout.getHySize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    HzAmplitude (yeeLayout.getHzSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Eps (yeeLayout.getEpsSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Mu (yeeLayout.getEpsSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    SigmaX (totSize, bufSizeL, bufSizeR, curProcess, totalProc, 0),
    SigmaY (totSize, bufSizeL, bufSizeR, curProcess, totalProc, 0),
    SigmaZ (totSize, bufSizeL, bufSizeR, curProcess, totalProc, 0),
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
    EInc (GridCoordinate1D ((grid_coord) 10*(totSize.getX () + totSize.getY ())), 0),
    HInc (GridCoordinate1D ((grid_coord) 10*(totSize.getX () + totSize.getY ())), 0),
    incidentWaveAngle1 (angleIncWave1),
    incidentWaveAngle2 (angleIncWave2),
    incidentWaveAngle3 (angleIncWave3)
#else
  Scheme3D (const GridCoordinate3D& totSize,
            time_step tStep,
            bool calcAmp = false,
            time_step ampStep = 0,
            bool doUsePML = false,
            GridCoordinate3D sizePML = GridCoordinate3D (0, 0, 0),
            bool doUseTFSF = false,
            GridCoordinate3D sizeScatteredZone = GridCoordinate3D (0, 0, 0),
            FieldValue angleIncWave1 = 0.0,
            FieldValue angleIncWave2 = 0.0,
            FieldValue angleIncWave3 = 0.0) :
    yeeLayout (totSize, sizePML, sizeScatteredZone, angleIncWave1, angleIncWave2, angleIncWave3),
    Ex (yeeLayout.getExSize (), 0),
    Ey (yeeLayout.getEySize (), 0),
    Ez (yeeLayout.getEzSize (), 0),
    Hx (yeeLayout.getHxSize (), 0),
    Hy (yeeLayout.getHySize (), 0),
    Hz (yeeLayout.getHzSize (), 0),
    Dx (yeeLayout.getExSize (), 0),
    Dy (yeeLayout.getEySize (), 0),
    Dz (yeeLayout.getEzSize (), 0),
    Bx (yeeLayout.getHxSize (), 0),
    By (yeeLayout.getHySize (), 0),
    Bz (yeeLayout.getHzSize (), 0),
    ExAmplitude (yeeLayout.getExSize (), 0),
    EyAmplitude (yeeLayout.getEySize (), 0),
    EzAmplitude (yeeLayout.getEzSize (), 0),
    HxAmplitude (yeeLayout.getHxSize (), 0),
    HyAmplitude (yeeLayout.getHySize (), 0),
    HzAmplitude (yeeLayout.getHzSize (), 0),
    Eps (yeeLayout.getEpsSize (), 0),
    Mu (yeeLayout.getEpsSize (), 0),
    SigmaX (totSize, 0),
    SigmaY (totSize, 0),
    SigmaZ (totSize, 0),
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
    EInc (GridCoordinate1D ((grid_coord) 10*(totSize.getX () + totSize.getY ())), 0),
    HInc (GridCoordinate1D ((grid_coord) 10*(totSize.getX () + totSize.getY ())), 0),
    incidentWaveAngle1 (angleIncWave1),
    incidentWaveAngle2 (angleIncWave2),
    incidentWaveAngle3 (angleIncWave3)
#endif
  {
    ASSERT (!doUseTFSF
            || (doUseTFSF
                && incidentWaveAngle1 == PhysicsConst::Pi / 2
                && (incidentWaveAngle2 == PhysicsConst::Pi / 4 || incidentWaveAngle2 == 0)
                && sizeScatteredZone != GridCoordinate3D (0, 0, 0)));

    ASSERT (!doUsePML || (doUsePML && (sizePML != GridCoordinate3D (0, 0, 0))));

    ASSERT (!calculateAmplitude || calculateAmplitude && amplitudeStepLimit != 0);
  }

  ~Scheme3D ()
  {
  }
};

#endif /* GRID_3D */

#endif /* SCHEME_3D_H */
