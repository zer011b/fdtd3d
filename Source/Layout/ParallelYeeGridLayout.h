#ifndef PARALLEL_YEE_GRID_LAYOUT_H
#define PARALLEL_YEE_GRID_LAYOUT_H

#include "YeeGridLayout.h"
#include "ParallelGridCore.h"

#ifdef PARALLEL_GRID

/**
 * Parallel Yee grid layout with size of grid per nodes
 */
template <SchemeType Type, uint8_t layout_type>
class ParallelYeeGridLayout: public YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>
{
public:

  static const bool isParallel;

private:

  ParallelGridCoordinate sizeForCurNode; /**< size of grid for current node */
  const ParallelGridCore *parallelGridCore; /**< parallel grid core */

private:

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z)

  void CalculateGridSizeForNode (grid_coord &, grid_coord &, int, bool, grid_coord) const;

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_1D_Z */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)

  void CalculateGridSizeForNode (grid_coord &, grid_coord &, int, bool, grid_coord,
                                 grid_coord &, grid_coord &, int, bool, grid_coord) const;

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void CalculateGridSizeForNode (grid_coord &, grid_coord &, int, bool, grid_coord,
                                 grid_coord &, grid_coord &, int, bool, grid_coord,
                                 grid_coord &, grid_coord &, int, bool, grid_coord) const;

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

public:

  ParallelGridCoordinate getEpsSizeForCurNode () const
  {
    return YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::isDoubleMaterialPrecision ? getSizeForCurNode () * 2: getSizeForCurNode ();
  }
  ParallelGridCoordinate getMuSizeForCurNode () const
  {
    return YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::isDoubleMaterialPrecision ? getSizeForCurNode () * 2: getSizeForCurNode ();
  }

  ParallelGridCoordinate getExSizeForCurNode () const
  {
    return getSizeForCurNode ();
  }
  ParallelGridCoordinate getEySizeForCurNode () const
  {
    return getSizeForCurNode ();
  }
  ParallelGridCoordinate getEzSizeForCurNode () const
  {
    return getSizeForCurNode ();
  }
  ParallelGridCoordinate getHxSizeForCurNode () const
  {
    return getSizeForCurNode ();
  }
  ParallelGridCoordinate getHySizeForCurNode () const
  {
    return getSizeForCurNode ();
  }
  ParallelGridCoordinate getHzSizeForCurNode () const
  {
    return getSizeForCurNode ();
  }

  /**
   * Get size of grid for current node
   *
   * @return size of grid for current node
   */
  ParallelGridCoordinate getSizeForCurNode () const
  {
    return sizeForCurNode;
  } /* getSizeForCurNode */

  void Initialize (const ParallelGridCore *);

  /**
   * Constructor of Parallel Yee grid
   */
  ParallelYeeGridLayout<Type, layout_type> (ParallelGridCoordinate coordSize,
                                            ParallelGridCoordinate sizePML,
                                            ParallelGridCoordinate sizeScatteredZone,
                                            FPValue incidentWaveAngle1, /**< teta */
                                            FPValue incidentWaveAngle2, /**< phi */
                                            FPValue incidentWaveAngle3, /**< psi */
                                            bool isDoubleMaterialPrecision)
    : YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type> (coordSize,
                                                                        sizePML,
                                                                        sizeScatteredZone,
                                                                        incidentWaveAngle1,
                                                                        incidentWaveAngle2,
                                                                        incidentWaveAngle3,
                                                                        isDoubleMaterialPrecision)
  {
  } /* ParallelYeeGridLayout */

  ~ParallelYeeGridLayout ()
  {
  } /* ~ParallelYeeGridLayout */

#ifdef DYNAMIC_GRID
  bool Rebalance (time_step);
#endif /* DYNAMIC_GRID */
}; /* ParallelYeeGridLayout */

/**
 * Identify size of grid for current computational node
 */
template <SchemeType Type, uint8_t layout_type>
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
void
ParallelYeeGridLayout<Type, layout_type>::CalculateGridSizeForNode (grid_coord &c1, grid_coord &core1, int nodeGridSize1, bool has1, grid_coord size1) const
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_1D_Z) */
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
void
ParallelYeeGridLayout<Type, layout_type>::CalculateGridSizeForNode (grid_coord &c1, grid_coord &core1, int nodeGridSize1, bool has1, grid_coord size1,
                                                                    grid_coord &c2, grid_coord &core2, int nodeGridSize2, bool has2, grid_coord size2) const
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ */
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
ParallelYeeGridLayout<Type, layout_type>::CalculateGridSizeForNode (grid_coord &c1, grid_coord &core1, int nodeGridSize1, bool has1, grid_coord size1,
                                                                    grid_coord &c2, grid_coord &core2, int nodeGridSize2, bool has2, grid_coord size2,
                                                                    grid_coord &c3, grid_coord &core3, int nodeGridSize3, bool has3, grid_coord size3) const
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  core1 = size1 / nodeGridSize1;

  if (has1)
  {
    c1 = core1;
  }
  else
  {
    c1 = size1 - (nodeGridSize1 - 1) * (size1 / nodeGridSize1);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_1D_Z) ||
          PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ) ||
          PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  core2 = size2 / nodeGridSize2;

  if (has2)
  {
    c2 = core2;
  }
  else
  {
    c2 = size2 - (nodeGridSize2 - 1) * (size2 / nodeGridSize2);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ) ||
          PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  core3 = size3 / nodeGridSize3;

  if (has3)
  {
    c3 = core3;
  }
  else
  {
    c3 = size3 - (nodeGridSize3 - 1) * (size3 / nodeGridSize3);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
} /* ParallelYeeGridLayout::CalculateGridSizeForNode */

#ifdef GRID_1D

/**
 * Initialize size of grid per node
 */
template <SchemeType Type, uint8_t layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::Initialize (const ParallelGridCore *parallelCore) /**< initialized parallel grid core */
{
  ASSERT (parallelCore);
  parallelGridCore = parallelCore;

  grid_coord c1;
  grid_coord core1;

  CalculateGridSizeForNode (c1,
                            core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ());

  sizeForCurNode = GridCoordinate1D (core1
#ifdef DEBUG_INFO
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType1 ()
#endif /* DEBUG_INFO */
                                     );
} /* ParallelYeeGridLayout::Initialize */

#endif /* GRID_1D */

#ifdef GRID_2D

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y)

/**
 * Initialize size of grid per node
 */
template <SchemeType Type, uint8_t layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::Initialize (const ParallelGridCore *parallelCore) /**< initialized parallel grid core */
{
  ASSERT (parallelCore);
  parallelGridCore = parallelCore;

  grid_coord c1;
  grid_coord c2;

  grid_coord core1;
  grid_coord core2;

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  CalculateGridSizeForNode (c1,
                            core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ());
  core2 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ();
  c2 = core2;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  core1 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ();
  c1 = core1;
  CalculateGridSizeForNode (c2,
                            core2,
                            parallelGridCore->getNodeGridSizeY (),
                            parallelGridCore->getHasU (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ());
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

  sizeForCurNode = GridCoordinate2D (core1, core2
#ifdef DEBUG_INFO
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType1 ()
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType2 ()
#endif /* DEBUG_INFO */
                                     );
} /* ParallelYeeGridLayout::Initialize */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY

/**
 * Initialize size of grid per node
 */
template <SchemeType Type, uint8_t layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::Initialize (const ParallelGridCore *parallelCore) /**< initialized parallel grid core */
{
  ASSERT (parallelCore);
  parallelGridCore = parallelCore;

  grid_coord c1;
  grid_coord c2;

  grid_coord core1;
  grid_coord core2;

  CalculateGridSizeForNode (c1,
                            core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 (),
                            c2,
                            core2,
                            parallelGridCore->getNodeGridSizeY (),
                            parallelGridCore->getHasU (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ());

  sizeForCurNode = GridCoordinate2D (core1, core2
#ifdef DEBUG_INFO
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType1 ()
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType2 ()
#endif /* DEBUG_INFO */
                                     );
} /* ParallelYeeGridLayout::Initialize */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#endif /* GRID_2D */

#ifdef GRID_3D

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || \
    defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || \
    defined (PARALLEL_BUFFER_DIMENSION_1D_Z)

/**
 * Initialize size of grid per node
 */
template <SchemeType Type, uint8_t layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::Initialize (const ParallelGridCore *parallelCore) /**< initialized parallel grid core */
{
  ASSERT (parallelCore);
  parallelGridCore = parallelCore;

  grid_coord c1;
  grid_coord c2;
  grid_coord c3;

  grid_coord core1;
  grid_coord core2;
  grid_coord core3;

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  CalculateGridSizeForNode (c1,
                            core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ());
  core2 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ();
  c2 = core2;
  core3 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ();
  c3 = core3;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  core1 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ();
  c1 = core1;
  CalculateGridSizeForNode (c2,
                            core2,
                            parallelGridCore->getNodeGridSizeY (),
                            parallelGridCore->getHasU (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ());
  core3 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ();
  c3 = core3;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  core1 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ();
  c1 = core1;
  core2 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ();
  c2 = core2;
  CalculateGridSizeForNode (c3,
                            core3,
                            parallelGridCore->getNodeGridSizeZ (),
                            parallelGridCore->getHasF (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ());
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z */

  sizeForCurNode = GridCoordinate3D (core1, core2, core3
#ifdef DEBUG_INFO
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType1 ()
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType2 ()
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType3 ()
#endif /* DEBUG_INFO */
                                     );
} /* ParallelYeeGridLayout::Initialize */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_1D_Z */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)

/**
 * Initialize size of grid per node
 */
template <SchemeType Type, uint8_t layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::Initialize (const ParallelGridCore *parallelCore) /**< initialized parallel grid core */
{
  ASSERT (parallelCore);
  parallelGridCore = parallelCore;

  grid_coord c1;
  grid_coord c2;
  grid_coord c3;

  grid_coord core1;
  grid_coord core2;
  grid_coord core3;

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  CalculateGridSizeForNode (c1,
                            core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 (),
                            c2,
                            core2,
                            parallelGridCore->getNodeGridSizeY (),
                            parallelGridCore->getHasU (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ());
  core3 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ();
  c3 = core3;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  core1 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ();
  c1 = core1;
  CalculateGridSizeForNode (c2,
                            core2,
                            parallelGridCore->getNodeGridSizeY (),
                            parallelGridCore->getHasU (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 (),
                            c3,
                            core3,
                            parallelGridCore->getNodeGridSizeZ (),
                            parallelGridCore->getHasF (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ());
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  core2 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ();
  c2 = core2;
  CalculateGridSizeForNode (c1,
                            core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 (),
                            c3,
                            core3,
                            parallelGridCore->getNodeGridSizeZ (),
                            parallelGridCore->getHasF (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ());
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */

  sizeForCurNode = GridCoordinate3D (core1, core2, core3
#ifdef DEBUG_INFO
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType1 ()
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType2 ()
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType3 ()
#endif /* DEBUG_INFO */
                                     );
} /* ParallelYeeGridLayout::Initialize */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ */

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ

/**
 * Initialize size of grid per node
 */
template <SchemeType Type, uint8_t layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::Initialize (const ParallelGridCore *parallelCore) /**< initialized parallel grid core */
{
  ASSERT (parallelCore);
  parallelGridCore = parallelCore;

  grid_coord c1;
  grid_coord c2;
  grid_coord c3;

  grid_coord core1;
  grid_coord core2;
  grid_coord core3;

  CalculateGridSizeForNode (c1,
                            core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 (),
                            c2,
                            core2,
                            parallelGridCore->getNodeGridSizeY (),
                            parallelGridCore->getHasU (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 (),
                            c3,
                            core3,
                            parallelGridCore->getNodeGridSizeZ (),
                            parallelGridCore->getHasF (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ());

  sizeForCurNode = GridCoordinate3D (core1, core2, core3
#ifdef DEBUG_INFO
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType1 ()
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType2 ()
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType3 ()
#endif /* DEBUG_INFO */
                                     );
} /* ParallelYeeGridLayout::Initialize */

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#endif /* GRID_3D */

#ifdef DYNAMIC_GRID

/**
 * Rebalance total grid size between notes
 *
 * @return true if size was changed
 *         false, otherwise
 */
template <SchemeType Type, uint8_t layout_type>
bool ParallelYeeGridLayout<Type, layout_type>::Rebalance (time_step difft) /**< number of time steps elapsed since the last rebalance */
{
  ParallelGridCoordinate newSize = sizeForCurNode;
  ParallelGridCoordinate oldSize = sizeForCurNode;

  timespec calcClock = parallelGridCore->getCalcClock ();

  grid_coord minX = 4;
  grid_coord maxX = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 () - minX * (parallelGridCore->getTotalProcCount () - 1);

  FPValue timesec = (FPValue) calcClock.tv_sec + ((FPValue) calcClock.tv_nsec) / 1000000000;
  FPValue speedCur = (difft * oldSize.calculateTotalCoord ()) / timesec;

  std::vector<FPValue> speed (parallelGridCore->getTotalProcCount ());
  FPValue sumSpeed = 0;

  for (int process = 0; process < parallelGridCore->getTotalProcCount (); ++process)
  {
    FPValue speedP;

    if (process == parallelGridCore->getProcessId ())
    {
      speedP = speedCur;
    }

    MPI_Bcast (&speedP, 1, MPI_DOUBLE, process, parallelGridCore->getCommunicator ());

    speed[process] = speedP;
    sumSpeed += speed[process];

    MPI_Barrier (parallelGridCore->getCommunicator ());
  }

  grid_coord x = ((FPValue)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ()) * speedCur / (sumSpeed);

  if (x < minX)
  {
    x = minX;
  }
  else if (x > maxX)
  {
    x = maxX;
  }

  // printf ("#%d x=%d speed=%f time=%lu.%lu totalX=%f difft=%u sumSpeed=%f\n",
  //         parallelGridCore->getProcessId (),
  //         x,
  //         speed[parallelGridCore->getProcessId ()],
  //         calcClock.tv_sec,
  //         calcClock.tv_nsec,
  //         (FPValue)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 (),
  //         difft,
  //         sumSpeed);

  newSize.set1 (x);

  if (parallelGridCore->getProcessId () == 0)
  {
    /*
     * Add all the remaining to the 0 process
     */
    grid_coord sumX = 0;
    for (int process = 0; process < parallelGridCore->getTotalProcCount (); ++process)
    {
      grid_coord x_n = ((FPValue)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ()) * speed[process] / (sumSpeed);
      if (x_n < minX)
      {
        x_n = minX;
      }
      else if (x_n > maxX)
      {
        x_n = maxX;
      }
      sumX += x_n;
    }

    newSize.set1 (x + YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 () - sumX);

    // printf ("!!! %d %lu %u\n", sumX, newSize.get1 (), x);
  }

  sizeForCurNode = newSize;

  return sizeForCurNode != oldSize;
} /* ParallelYeeGridLayout::Rebalance */

#endif /* DYNAMIC_GRID */

#endif /* PARALLEL_GRID */

#endif /* PARALLEL_YEE_GRID_LAYOUT_H */
