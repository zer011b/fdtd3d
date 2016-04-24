#ifndef SCHEME_TEZ_H
#define SCHEME_TEZ_H

#include "Scheme.h"
#include "ParallelGrid.h"

class SchemeTEz: public Scheme
{
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

public:
  void performStep () override;

  void initScheme (FieldValue wLength, FieldValue step);

  void initGrids ();

#if defined (PARALLEL_GRID)
  SchemeTEz (const GridCoordinate2D& totSize,
             const GridCoordinate2D& bufSizeL, const GridCoordinate2D& bufSizeR,
             const int process, const int totalProc, uint32_t step) :
    Ez (totSize, bufSizeL, bufSizeR, process, totalProc, step),
    Hx (totSize, bufSizeL, bufSizeR, process, totalProc, step),
    Hy (totSize, bufSizeL, bufSizeR, process, totalProc, step),
    Eps (totSize, bufSizeL, bufSizeR, process, totalProc, step),
    Mu (totSize, bufSizeL, bufSizeR, process, totalProc, step),
    waveLength (0),
    stepWaveLength (0),
    frequency (0),
    gridStep (0),
    gridTimeStep (0)
  {
  }
#else
  SchemeTEz (const GridCoordinate2D& totSize, uint32_t step) :
    Ez (totSize, step),
    Hx (totSize, step),
    Hy (totSize, step),
    Eps (totSize, step),
    Mu (totSize, step),
    waveLength (0),
    stepWaveLength (0),
    frequency (0),
    gridStep (0),
    gridTimeStep (0)
  {
  }
#endif

  ~SchemeTEz ()
  {
  }
};

#endif /* SCHEME_TEZ_H */
