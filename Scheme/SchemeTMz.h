#ifndef SCHEME_TEZ_H
#define SCHEME_TEZ_H

#include "Scheme.h"
#include "ParallelGrid.h"
#include "Grid.h"
#include "GridLayout.h"

#ifdef GRID_2D

class SchemeTMz: public Scheme
{
  YeeGridLayout yeeLayout;

#if defined (PARALLEL_GRID)
  ParallelGrid Ez;
  ParallelGrid Hx;
  ParallelGrid Hy;

  ParallelGrid EzAmplitude;
  ParallelGrid HxAmplitude;
  ParallelGrid HyAmplitude;

  ParallelGrid Eps;
  ParallelGrid Mu;
#else
  Grid<GridCoordinate2D> Ez;
  Grid<GridCoordinate2D> Hx;
  Grid<GridCoordinate2D> Hy;

  Grid<GridCoordinate2D> EzAmplitude;
  Grid<GridCoordinate2D> HxAmplitude;
  Grid<GridCoordinate2D> HyAmplitude;

  Grid<GridCoordinate2D> Eps;
  Grid<GridCoordinate2D> Mu;
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

private:

  void performEzSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHxSteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performHySteps (time_step, GridCoordinate3D, GridCoordinate3D);
  void performNSteps (time_step, time_step, int);
  void performAmplitudeSteps (time_step, int);

  int updateAmplitude (FieldPointValue *, FieldPointValue *, FieldValue *);

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
  SchemeTMz (const GridCoordinate2D& totSize,
             const GridCoordinate2D& bufSizeL, const GridCoordinate2D& bufSizeR,
             const int curProcess, const int totalProc, time_step tStep, bool calcAmp,
             time_step ampStep = 0) :
    yeeLayout (totSize),
    Ez (shrinkCoord (yeeLayout.getEzSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Hx (shrinkCoord (yeeLayout.getHxSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Hy (shrinkCoord (yeeLayout.getHySize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    EzAmplitude (shrinkCoord (yeeLayout.getEzSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    HxAmplitude (shrinkCoord (yeeLayout.getHxSize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    HyAmplitude (shrinkCoord (yeeLayout.getHySize ()), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Eps (totSize, bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Mu (totSize, bufSizeL, bufSizeR, curProcess, totalProc, 0),
    waveLength (0),
    stepWaveLength (0),
    frequency (0),
    gridStep (0),
    gridTimeStep (0),
    totalStep (tStep),
    process (curProcess),
    calculateAmplitude (calcAmp),
    amplitudeStepLimit (ampStep)
  {
    ASSERT (!calcAmp || calcAmp && ampStep != 0);
  }
#else
  SchemeTMz (const GridCoordinate2D& totSize, time_step tStep, bool calcAmp, time_step ampStep = 0) :
    yeeLayout (totSize),
    Ez (shrinkCoord (yeeLayout.getEzSize ()), 0),
    Hx (shrinkCoord (yeeLayout.getHxSize ()), 0),
    Hy (shrinkCoord (yeeLayout.getHySize ()), 0),
    EzAmplitude (shrinkCoord (yeeLayout.getEzSize ()), 0),
    HxAmplitude (shrinkCoord (yeeLayout.getHxSize ()), 0),
    HyAmplitude (shrinkCoord (yeeLayout.getHySize ()), 0),
    Eps (totSize, 0),
    Mu (totSize, 0),
    waveLength (0),
    stepWaveLength (0),
    frequency (0),
    gridStep (0),
    gridTimeStep (0),
    totalStep (tStep),
    process (0),
    calculateAmplitude (calcAmp),
    amplitudeStepLimit (ampStep)
  {
    ASSERT (!calculateAmplitude || calculateAmplitude && amplitudeStepLimit != 0);
  }
#endif

  ~SchemeTMz ()
  {
  }
};

#endif /* GRID_2D */

#endif /* SCHEME_TEZ_H */
