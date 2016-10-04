#ifndef SCHEME_3D_H
#define SCHEME_3D_H

#include "Scheme.h"
#include "ParallelGrid.h"
#include "Grid.h"
#include "GridLayout.h"

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

  ParallelGrid Eps;
  ParallelGrid Mu;
#else
  Grid<GridCoordinate3D> Ex;
  Grid<GridCoordinate3D> Ey;
  Grid<GridCoordinate3D> Ez;
  Grid<GridCoordinate3D> Hx;
  Grid<GridCoordinate3D> Hy;
  Grid<GridCoordinate3D> Hz;

  Grid<GridCoordinate3D> Eps;
  Grid<GridCoordinate3D> Mu;
#endif

  // Wave parameters
  FieldValue waveLength;
  FieldValue stepWaveLength;
  FieldValue frequency;

  // dx
  FieldValue gridStep;

  // dt
  FieldValue gridTimeStep;

  uint32_t totalStep;

  int process;

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
            const GridCoordinate3D& bufSizeL, const GridCoordinate3D& bufSizeR,
            const int curProcess, const int totalProc, uint32_t tStep) :
    yeeLayout (totSize),
    Ex (yeeLayout.getExSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Ey (yeeLayout.getEySize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Ez (yeeLayout.getEzSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Hx (yeeLayout.getHxSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Hy (yeeLayout.getHySize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Hz (yeeLayout.getHzSize (), bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Eps (totSize, bufSizeL, bufSizeR, curProcess, totalProc, 0),
    Mu (totSize, bufSizeL, bufSizeR, curProcess, totalProc, 0),
    waveLength (0),
    stepWaveLength (0),
    frequency (0),
    gridStep (0),
    gridTimeStep (0),
    totalStep (tStep),
    process (curProcess)
  {
  }
#else
  Scheme3D (const GridCoordinate3D& totSize, uint32_t tStep) :
    yeeLayout (totSize),
    Ex (yeeLayout.getExSize (), 0),
    Ey (yeeLayout.getEySize (), 0),
    Ez (yeeLayout.getEzSize (), 0),
    Hx (yeeLayout.getHxSize (), 0),
    Hy (yeeLayout.getHySize (), 0),
    Hz (yeeLayout.getHzSize (), 0),
    Eps (totSize, 0),
    Mu (totSize, 0),
    waveLength (0),
    stepWaveLength (0),
    frequency (0),
    gridStep (0),
    gridTimeStep (0),
    totalStep (tStep),
    process (0)
  {
  }
#endif

  ~Scheme3D ()
  {
  }
};

#endif /* GRID_3D */

#endif /* SCHEME_3D_H */
