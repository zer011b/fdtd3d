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

  Grid<GridCoordinate2D> Dz;
  Grid<GridCoordinate2D> Bx;
  Grid<GridCoordinate2D> By;

  Grid<GridCoordinate2D> EzAmplitude;
  Grid<GridCoordinate2D> HxAmplitude;
  Grid<GridCoordinate2D> HyAmplitude;

  Grid<GridCoordinate2D> Eps;
  Grid<GridCoordinate2D> Mu;

  Grid<GridCoordinate2D> SigmaX;
  Grid<GridCoordinate2D> SigmaY;
  Grid<GridCoordinate2D> SigmaZ;
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

  Grid<GridCoordinate1D> EInc;
  Grid<GridCoordinate1D> HInc;

private:

  void calculateEzStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHxStep (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHyStep (time_step, GridCoordinate3D, GridCoordinate3D);

  void calculateEzStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHxStepPML (time_step, GridCoordinate3D, GridCoordinate3D);
  void calculateHyStepPML (time_step, GridCoordinate3D, GridCoordinate3D);

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
  SchemeTMz (const GridCoordinate2D& totSize,
             time_step tStep,
             bool calcAmp = false,
             time_step ampStep = 0,
             bool doUsePML = false,
             GridCoordinate2D sizePML = GridCoordinate2D (0, 0),
             bool doUseTFSF = false,
             GridCoordinate2D sizeScatteredZone = GridCoordinate2D (0, 0)) :
    yeeLayout (totSize, sizePML, sizeScatteredZone),
    Ez (shrinkCoord (yeeLayout.getEzSize ()), 0),
    Hx (shrinkCoord (yeeLayout.getHxSize ()), 0),
    Hy (shrinkCoord (yeeLayout.getHySize ()), 0),
    Dz (shrinkCoord (yeeLayout.getEzSize ()), 0),
    Bx (shrinkCoord (yeeLayout.getHxSize ()), 0),
    By (shrinkCoord (yeeLayout.getHySize ()), 0),
    EzAmplitude (shrinkCoord (yeeLayout.getEzSize ()), 0),
    HxAmplitude (shrinkCoord (yeeLayout.getHxSize ()), 0),
    HyAmplitude (shrinkCoord (yeeLayout.getHySize ()), 0),
    Eps (totSize, 0),
    Mu (totSize, 0),
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
    EInc (totSize.getX () + totSize.getY ()),
    HInc (totSize.getX () + totSize.getY ())
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
