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

  ParallelGrid Eps;
  ParallelGrid Mu;
#else
  Grid<GridCoordinate2D> Ez;
  Grid<GridCoordinate2D> Hx;
  Grid<GridCoordinate2D> Hy;

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
  SchemeTMz (const GridCoordinate2D& totSize,
             const GridCoordinate2D& bufSizeL, const GridCoordinate2D& bufSizeR,
             const int process, const int totalProc, uint32_t tStep) :
    yeeLayout (totSize),
    Ez (shrinkCoord (yeeLayout.getEzSize ()), bufSizeL, bufSizeR, process, totalProc, 0),
    Hx (shrinkCoord (yeeLayout.getHxSize ()), bufSizeL, bufSizeR, process, totalProc, 0),
    Hy (shrinkCoord (yeeLayout.getHySize ()), bufSizeL, bufSizeR, process, totalProc, 0),
    Eps (totSize, bufSizeL, bufSizeR, process, totalProc, 0),
    Mu (totSize, bufSizeL, bufSizeR, process, totalProc, 0),
    waveLength (0),
    stepWaveLength (0),
    frequency (0),
    gridStep (0),
    gridTimeStep (0),
    totalStep (tStep)
  {
  }
#else
  SchemeTMz (const GridCoordinate2D& totSize, uint32_t tStep) :
    yeeLayout (totSize),
    Ez (shrinkCoord (yeeLayout.getEzSize ()), 0),
    Hx (shrinkCoord (yeeLayout.getHxSize ()), 0),
    Hy (shrinkCoord (yeeLayout.getHySize ()), 0),
    Eps (totSize, 0),
    Mu (totSize, 0),
    waveLength (0),
    stepWaveLength (0),
    frequency (0),
    gridStep (0),
    gridTimeStep (0),
    totalStep (tStep)
  {
  }
#endif

  ~SchemeTMz ()
  {
  }
};

#endif /* GRID_2D */

#endif /* SCHEME_TEZ_H */
