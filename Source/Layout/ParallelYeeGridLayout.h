#ifndef PARALLEL_YEE_GRID_LAYOUT_H
#define PARALLEL_YEE_GRID_LAYOUT_H

#include "YeeGridLayout.h"
#include "ParallelGridCore.h"
#include <algorithm>

#ifdef PARALLEL_GRID

class Entry_t
{
public:
  int pid1;
  int pid2;
  FPValue val;

  Entry_t (int newPid1, int newPid2, FPValue newVal)
  : pid1 (newPid1), pid2 (newPid2), val (newVal) {}

  bool operator < (Entry_t entry)
  {
    return val < entry.val;
  }
};

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
  ParallelGridCore *parallelGridCore; /**< parallel grid core */

private:

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z)

  void CalculateGridSizeForNode (grid_coord &, int, bool, grid_coord) const;

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_1D_Z */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)

  void CalculateGridSizeForNode (grid_coord &, int, bool, grid_coord,
                                 grid_coord &, int, bool, grid_coord) const;

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void CalculateGridSizeForNode (grid_coord &, int, bool, grid_coord,
                                 grid_coord &, int, bool, grid_coord,
                                 grid_coord &, int, bool, grid_coord) const;

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

  void Initialize (ParallelGridCore *);

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
ParallelYeeGridLayout<Type, layout_type>::CalculateGridSizeForNode (grid_coord &core1, int nodeGridSize1, bool has1, grid_coord size1) const
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_1D_Z) */
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
void
ParallelYeeGridLayout<Type, layout_type>::CalculateGridSizeForNode (grid_coord &core1, int nodeGridSize1, bool has1, grid_coord size1,
                                                                    grid_coord &core2, int nodeGridSize2, bool has2, grid_coord size2) const
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ */
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
ParallelYeeGridLayout<Type, layout_type>::CalculateGridSizeForNode (grid_coord &core1, int nodeGridSize1, bool has1, grid_coord size1,
                                                                    grid_coord &core2, int nodeGridSize2, bool has2, grid_coord size2,
                                                                    grid_coord &core3, int nodeGridSize3, bool has3, grid_coord size3) const
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  core1 = size1 / nodeGridSize1;

  if (!has1)
  {
    core1 = size1 - (nodeGridSize1 - 1) * core1;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_1D_Z) ||
          PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ) ||
          PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  core2 = size2 / nodeGridSize2;

  if (!has2)
  {
    core2 = size2 - (nodeGridSize2 - 1) * core2;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ) ||
          PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  core3 = size3 / nodeGridSize3;

  if (!has3)
  {
    core3 = size3 - (nodeGridSize3 - 1) * core3;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
} /* ParallelYeeGridLayout::CalculateGridSizeForNode */

#ifdef GRID_1D

/**
 * Initialize size of grid per node
 */
template <SchemeType Type, uint8_t layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::Initialize (ParallelGridCore *parallelCore) /**< initialized parallel grid core */
{
  ASSERT (parallelCore);
  parallelGridCore = parallelCore;

  grid_coord core1;

  CalculateGridSizeForNode (core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ());

  sizeForCurNode = GridCoordinate1D (core1
#ifdef DEBUG_INFO
                                     , YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::getSize ().getType1 ()
#endif /* DEBUG_INFO */
                                     );

#ifdef DYNAMIC_GRID
  parallelGridCore->setCalcClockCount (parallelGridCore->getProcessId (), sizeForCurNode.calculateTotalCoord ());

  int dirL = parallelGridCore->getNodeForDirection (LEFT);
  if (dirL != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirL, 1);
  }
  int dirR = parallelGridCore->getNodeForDirection (RIGHT);
  if (dirR != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirR, 1);
  }
#endif
} /* ParallelYeeGridLayout::Initialize */

#endif /* GRID_1D */

#ifdef GRID_2D

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y)

/**
 * Initialize size of grid per node
 */
template <SchemeType Type, uint8_t layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::Initialize (ParallelGridCore *parallelCore) /**< initialized parallel grid core */
{
  ASSERT (parallelCore);
  parallelGridCore = parallelCore;

  grid_coord core1;
  grid_coord core2;

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  CalculateGridSizeForNode (core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ());
  core2 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  core1 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ();
  CalculateGridSizeForNode (core2,
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

#ifdef DYNAMIC_GRID
  parallelGridCore->setCalcClockCount (parallelGridCore->getProcessId (), sizeForCurNode.calculateTotalCoord ());

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  int dirL = parallelGridCore->getNodeForDirection (LEFT);
  if (dirL != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirL, sizeForCurNode.get2 ());
  }
  int dirR = parallelGridCore->getNodeForDirection (RIGHT);
  if (dirR != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirR, sizeForCurNode.get2 ());
  }
#endif

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  int dirD = parallelGridCore->getNodeForDirection (DOWN);
  if (dirD != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirD, sizeForCurNode.get1 ());
  }
  int dirU = parallelGridCore->getNodeForDirection (UP);
  if (dirU != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirU, sizeForCurNode.get1 ());
  }
#endif
#endif
} /* ParallelYeeGridLayout::Initialize */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY

/**
 * Initialize size of grid per node
 */
template <SchemeType Type, uint8_t layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::Initialize (ParallelGridCore *parallelCore) /**< initialized parallel grid core */
{
  ASSERT (parallelCore);
  parallelGridCore = parallelCore;

  grid_coord core1;
  grid_coord core2;

  CalculateGridSizeForNode (core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 (),
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

#ifdef DYNAMIC_GRID
  parallelGridCore->setCalcClockCount (parallelGridCore->getProcessId (), sizeForCurNode.calculateTotalCoord ());

  int dirL = parallelGridCore->getNodeForDirection (LEFT);
  if (dirL != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirL, sizeForCurNode.get2 ());
  }
  int dirR = parallelGridCore->getNodeForDirection (RIGHT);
  if (dirR != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirR, sizeForCurNode.get2 ());
  }

  int dirD = parallelGridCore->getNodeForDirection (DOWN);
  if (dirD != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirD, sizeForCurNode.get1 ());
  }
  int dirU = parallelGridCore->getNodeForDirection (UP);
  if (dirU != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirU, sizeForCurNode.get1 ());
  }


  int dirLD = parallelGridCore->getNodeForDirection (LEFT_DOWN);
  if (dirLD != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLD, 1);
  }
  int dirRD = parallelGridCore->getNodeForDirection (RIGHT_DOWN);
  if (dirRD != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRD, 1);
  }

  int dirLU = parallelGridCore->getNodeForDirection (LEFT_UP);
  if (dirLU != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLU, 1);
  }
  int dirRU = parallelGridCore->getNodeForDirection (RIGHT_UP);
  if (dirRU != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRU, 1);
  }
#endif
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
ParallelYeeGridLayout<Type, layout_type>::Initialize (ParallelGridCore *parallelCore) /**< initialized parallel grid core */
{
  ASSERT (parallelCore);
  parallelGridCore = parallelCore;

  grid_coord core1;
  grid_coord core2;
  grid_coord core3;

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  CalculateGridSizeForNode (core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ());
  core2 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ();
  core3 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  core1 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ();
  CalculateGridSizeForNode (core2,
                            parallelGridCore->getNodeGridSizeY (),
                            parallelGridCore->getHasU (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ());
  core3 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  core1 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ();
  core2 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ();
  CalculateGridSizeForNode (core3,
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

#ifdef DYNAMIC_GRID
  parallelGridCore->setCalcClockCount (parallelGridCore->getProcessId (), sizeForCurNode.calculateTotalCoord ());

  /* TODO: add this */
  UNREACHABLE;
#endif
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
ParallelYeeGridLayout<Type, layout_type>::Initialize (ParallelGridCore *parallelCore) /**< initialized parallel grid core */
{
  ASSERT (parallelCore);
  parallelGridCore = parallelCore;

  grid_coord core1;
  grid_coord core2;
  grid_coord core3;

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  CalculateGridSizeForNode (core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 (),
                            core2,
                            parallelGridCore->getNodeGridSizeY (),
                            parallelGridCore->getHasU (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ());
  core3 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ();
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  core1 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ();
  CalculateGridSizeForNode (core2,
                            parallelGridCore->getNodeGridSizeY (),
                            parallelGridCore->getHasU (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 (),
                            core3,
                            parallelGridCore->getNodeGridSizeZ (),
                            parallelGridCore->getHasF (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ());
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  core2 = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ();
  CalculateGridSizeForNode (core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 (),
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

#ifdef DYNAMIC_GRID
  parallelGridCore->setCalcClockCount (parallelGridCore->getProcessId (), sizeForCurNode.calculateTotalCoord ());

  /* TODO: add this */
  UNREACHABLE;
#endif
} /* ParallelYeeGridLayout::Initialize */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ */

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ

/**
 * Initialize size of grid per node
 */
template <SchemeType Type, uint8_t layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::Initialize (ParallelGridCore *parallelCore) /**< initialized parallel grid core */
{
  ASSERT (parallelCore);
  parallelGridCore = parallelCore;

  grid_coord core1;
  grid_coord core2;
  grid_coord core3;

  CalculateGridSizeForNode (core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 (),
                            core2,
                            parallelGridCore->getNodeGridSizeY (),
                            parallelGridCore->getHasU (),
                            YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 (),
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

#ifdef DYNAMIC_GRID
  parallelGridCore->setCalcClockCount (parallelGridCore->getProcessId (), sizeForCurNode.calculateTotalCoord ());

  /* TODO: add this */
  UNREACHABLE;
#endif
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

  uint32_t latency_measure_count = 10;
  uint32_t latency_buf_size = 10;

  if (parallelGridCore->getNodeState ()[parallelGridCore->getProcessId ()])
  {
#ifdef GRID_1D
    int dirL = parallelGridCore->getNodeForDirection (LEFT);
    if (dirL != PID_NONE)
    {
      parallelGridCore->setShareClockIterCur (dirL, parallelGridCore->getShareClockCountCur (dirL), difft);
      parallelGridCore->setShareClockIterCur (dirL, latency_buf_size, latency_measure_count);
    }
    int dirR = parallelGridCore->getNodeForDirection (RIGHT);
    if (dirR != PID_NONE)
    {
      parallelGridCore->setShareClockIterCur (dirR, parallelGridCore->getShareClockCountCur (dirR), difft);
      parallelGridCore->setShareClockIterCur (dirR, latency_buf_size, latency_measure_count);
    }
#endif
  }

#ifdef GRID_2D
  UNREACHABLE;
  // TODO: add setShareClockCount
#endif

#ifdef GRID_3D
  UNREACHABLE;
  // TODO: add setShareClockCount
#endif

  // Measure latency
  // if (...)
  {
    MPI_Datatype datatype;

    #ifdef FLOAT_VALUES
    #ifdef COMPLEX_FIELD_VALUES
      datatype = MPI_COMPLEX;
    #else /* COMPLEX_FIELD_VALUES */
      datatype = MPI_FLOAT;
    #endif /* !COMPLEX_FIELD_VALUES */
    #endif /* FLOAT_VALUES */

    #ifdef DOUBLE_VALUES
    #ifdef COMPLEX_FIELD_VALUES
      datatype = MPI_DOUBLE_COMPLEX;
    #else /* COMPLEX_FIELD_VALUES */
      datatype = MPI_DOUBLE;
    #endif /* !COMPLEX_FIELD_VALUES */
    #endif /* DOUBLE_VALUES */

    #ifdef LONG_DOUBLE_VALUES
    #ifdef COMPLEX_FIELD_VALUES
      datatype = MPI_LONG_DOUBLE_COMPLEX;
    #else /* COMPLEX_FIELD_VALUES */
      datatype = MPI_LONG_DOUBLE;
    #endif /* !COMPLEX_FIELD_VALUES */
    #endif /* LONG_DOUBLE_VALUES */

    std::vector<FieldValue> tmp_buffer (latency_buf_size);

    for (uint32_t count = 0; count < latency_measure_count; ++count)
    for (int buf = 0; buf < BUFFER_COUNT; ++buf)
    {
      BufferPosition bufferDirection = (BufferPosition) buf;

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
      if (parallelGridCore->getProcessId () >= parallelGridCore->getNodeGridSizeXYZ ())
      {
        break;
      }
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
      if (parallelGridCore->getProcessId () >= parallelGridCore->getNodeGridSizeXY ())
      {
        break;
      }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
      if (parallelGridCore->getProcessId () >= parallelGridCore->getNodeGridSizeYZ ())
      {
        break;
      }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
      if (parallelGridCore->getProcessId () >= parallelGridCore->getNodeGridSizeXZ ())
      {
        break;
      }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */

      int state = parallelGridCore->getNodeState ()[parallelGridCore->getProcessId ()];
      if (state == 0)
      {
        break;
      }

      BufferPosition opposite = parallelGridCore->getOppositeDirections ()[bufferDirection];

      int processTo = parallelGridCore->getNodeForDirection (bufferDirection);
      int processFrom = parallelGridCore->getNodeForDirection (opposite);

      if (processTo != PID_NONE
          && processFrom == PID_NONE)
      {
        parallelGridCore->StartShareClock (processTo, latency_buf_size);

        int retCode = MPI_Send (tmp_buffer.data(), latency_buf_size, datatype, processTo, parallelGridCore->getProcessId (), parallelGridCore->getCommunicator ());
        ASSERT (retCode == MPI_SUCCESS);

        parallelGridCore->StopShareClock (processTo, latency_buf_size);
      }
      else if (processTo == PID_NONE
               && processFrom != PID_NONE)
      {
        parallelGridCore->StartShareClock (processFrom, latency_buf_size);

        MPI_Status status;
        int retCode = MPI_Recv (tmp_buffer.data(), latency_buf_size, datatype, processFrom, processFrom, parallelGridCore->getCommunicator (), &status);
        ASSERT (retCode == MPI_SUCCESS);

        parallelGridCore->StopShareClock (processFrom, latency_buf_size);
      }
      else if (processTo != PID_NONE
               && processFrom != PID_NONE)
      {
#ifdef COMBINED_SENDRECV
        UNREACHABLE;
#else
        // Even send first, then receive. Non-even receive first, then send
        if (parallelGridCore->getIsEvenForDirection()[bufferDirection])
        {
          parallelGridCore->StartShareClock (processTo, latency_buf_size);

          int retCode = MPI_Send (tmp_buffer.data(), latency_buf_size, datatype, processTo, parallelGridCore->getProcessId (), parallelGridCore->getCommunicator ());
          ASSERT (retCode == MPI_SUCCESS);

          parallelGridCore->StopShareClock (processTo, latency_buf_size);
          parallelGridCore->StartShareClock (processFrom, latency_buf_size);

          MPI_Status status;
          retCode = MPI_Recv (tmp_buffer.data(), latency_buf_size, datatype, processFrom, processFrom, parallelGridCore->getCommunicator (), &status);
          ASSERT (retCode == MPI_SUCCESS);

          parallelGridCore->StopShareClock (processFrom, latency_buf_size);
        }
        else
        {
          parallelGridCore->StartShareClock (processFrom, latency_buf_size);

          MPI_Status status;
          int retCode = MPI_Recv (tmp_buffer.data(), latency_buf_size, datatype, processFrom, processFrom, parallelGridCore->getCommunicator (), &status);
          ASSERT (retCode == MPI_SUCCESS);

          parallelGridCore->StopShareClock (processFrom, latency_buf_size);
          parallelGridCore->StartShareClock (processTo, latency_buf_size);

          retCode = MPI_Send (tmp_buffer.data(), latency_buf_size, datatype, processTo, parallelGridCore->getProcessId (), parallelGridCore->getCommunicator ());
          ASSERT (retCode == MPI_SUCCESS);

          parallelGridCore->StopShareClock (processTo, latency_buf_size);
        }
#endif
      }
      else
      {
        /*
         * Do nothing
         */
      }
    }
  }

  parallelGridCore->ShareClocks ();

  /*
   * ==== Get current values of performance ====
   */

  std::vector<FPValue> curPoints (parallelGridCore->getTotalProcCount ());
  std::vector<FPValue> curTimes (parallelGridCore->getTotalProcCount ());

  for (int i = 0; i < parallelGridCore->getTotalProcCount (); ++i)
  {
    curPoints[i] = difft * parallelGridCore->getCalcClockCount (i);

    timespec calcClockCur = parallelGridCore->getCalcClock (i);
    FPValue timesec = (FPValue) calcClockCur.tv_sec + ((FPValue) calcClockCur.tv_nsec) / 1000000000;
    curTimes[i] = timesec;

#ifdef ENABLE_ASSERTS
    if (parallelGridCore->getNodeState ()[i] == 0)
    {
      ASSERT (curPoints[i] == 0 && curTimes[i] == 0);
    }
    else
    {
      ASSERT (curPoints[i] != 0 && curTimes[i] != 0);
    }
#endif
  }

  /*
   * ==== Get current values of latency and bw ====
   */

  /*
   * In case accuracy is not good, skip this measurement
   */
  std::vector< std::vector<int> > skipCurShareMeasurement (parallelGridCore->getTotalProcCount ());

  std::vector< std::vector<FPValue> > curShareLatency (parallelGridCore->getTotalProcCount ());
  std::vector< std::vector<FPValue> > curShareBandwidth (parallelGridCore->getTotalProcCount ());

  for (int i = 0; i < parallelGridCore->getTotalProcCount (); ++i)
  {
    skipCurShareMeasurement[i].resize (parallelGridCore->getTotalProcCount ());

    curShareLatency[i].resize (parallelGridCore->getTotalProcCount ());
    curShareBandwidth[i].resize (parallelGridCore->getTotalProcCount ());
  }

  for (int i = 0; i < parallelGridCore->getTotalProcCount (); ++i)
  {
    for (int j = 0; j < parallelGridCore->getTotalProcCount (); ++j)
    {
      if (i == j)
      {
        continue;
      }

      skipCurShareMeasurement[i][j] = 0;

      ShareClock_t map = parallelGridCore->getShareClock (i, j);

      FPValue latency = 0;
      FPValue bandwidth = 0;

      int index = 0;
      for (ShareClock_t::iterator it = map.begin (); it != map.end (); ++it)
      {
        FPValue bufSize = it->first;

#ifndef MPI_DYNAMIC_CLOCK
        FPValue clocks = (FPValue) it->second.tv_sec + ((FPValue) it->second.tv_nsec) / 1000000000;
#else
        FPValue clocks = it->second;
#endif

        if (bufSize == 1)
        {
          latency = clocks / parallelGridCore->getShareClockIter (i, j, bufSize);
        }
      }

      ASSERT (map.size () == CLOCK_BUF_SIZE
              || map.empty ());

      index = 0;
      for (ShareClock_t::iterator it = map.begin (); it != map.end (); ++it)
      {
        FPValue bufSize = it->first;

#ifndef MPI_DYNAMIC_CLOCK
        FPValue clocks = (FPValue) it->second.tv_sec + ((FPValue) it->second.tv_nsec) / 1000000000;
#else
        FPValue clocks = it->second;
#endif

        if (bufSize != 1)
        {
          FPValue t_avg = clocks / parallelGridCore->getShareClockIter (i, j, bufSize);
          bandwidth = bufSize / (t_avg - latency);
        }
      }

      curShareLatency[i][j] = latency;
      curShareBandwidth[i][j] = bandwidth;

      if (curShareBandwidth[i][j] < 0)
      {
        skipCurShareMeasurement[i][j] = 1;
      }

#ifdef ENABLE_ASSERTS
      if (parallelGridCore->getNodeState ()[i] == 0
          || parallelGridCore->getNodeState ()[j] == 0)
      {
        ASSERT (curShareLatency[i][j] == 0 && curShareBandwidth[i][j] == 0);
      }
      else
      {
        ASSERT (curShareLatency[i][j] != 0 && curShareBandwidth[i][j] != 0);
      }
#endif
    }
  }

  /*
   * ==== Calculate total perf ====
   */

  std::vector<FPValue> speed (parallelGridCore->getTotalProcCount ());
  FPValue sumSpeedEnabled = 0;
  for (int process = 0; process < parallelGridCore->getTotalProcCount (); ++process)
  {
    if (parallelGridCore->getNodeState ()[process] == 1)
    {
      parallelGridCore->perfPointsValues[process] += curPoints[process];
      parallelGridCore->perfTimeValues[process] += curTimes[process];
    }

    if (parallelGridCore->perfTimeValues[process] == 0)
    {
      speed[process] = 0;
    }
    else
    {
      speed[process] = parallelGridCore->perfPointsValues[process] / parallelGridCore->perfTimeValues[process];
    }

    if (parallelGridCore->getNodeState ()[process] == 1)
    {
      sumSpeedEnabled += speed[process];
    }
  }

  /*
   * ==== Calculate total latency and bandwidth ====
   */

  std::vector< std::vector<FPValue> > latency (parallelGridCore->getTotalProcCount ());
  std::vector< std::vector<FPValue> > bandwidth (parallelGridCore->getTotalProcCount ());
  for (int process = 0; process < parallelGridCore->getTotalProcCount (); ++process)
  {
    latency[process].resize (parallelGridCore->getTotalProcCount ());
    bandwidth[process].resize (parallelGridCore->getTotalProcCount ());

    for (int i = 0; i < parallelGridCore->getTotalProcCount (); ++i)
    {
      if (process == i)
      {
        continue;
      }

      if (parallelGridCore->getNodeState ()[process] == 1
          && parallelGridCore->getNodeState ()[i] == 1)
      {
        parallelGridCore->latencySumValues[process][i] += curShareLatency[process][i];
        parallelGridCore->latencyCountValues[process][i] += 1;

        if (skipCurShareMeasurement[process][i] == 0)
        {
          parallelGridCore->bandwidthSumValues[process][i] += curShareBandwidth[process][i];
          parallelGridCore->bandwidthCountValues[process][i] += 1;
        }
      }

      if (parallelGridCore->latencyCountValues[process][i] != 0)
      {
        latency[process][i] = parallelGridCore->latencySumValues[process][i] / parallelGridCore->latencyCountValues[process][i];
      }
      else
      {
        latency[process][i] = 0;
      }
      if (parallelGridCore->bandwidthCountValues[process][i] != 0)
      {
        bandwidth[process][i] = parallelGridCore->bandwidthSumValues[process][i] / parallelGridCore->bandwidthCountValues[process][i];
      }
      else
      {
        bandwidth[process][i] = 0;
      }
    }
  }

  /*
   * ==== Spread ====
   */

  /* TODO: this is for 1D only */

  std::vector<grid_coord> spread (parallelGridCore->getTotalProcCount ());
  grid_coord sum_spread = 0;
  for (int process = 0; process < parallelGridCore->getTotalProcCount (); ++process)
  {
    if (parallelGridCore->getNodeState ()[process] == 1)
    {
      spread[process] = ((FPValue)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ()) * speed[process] / (sumSpeedEnabled);
      if (spread[process] < 1)
      {
        spread[process] = 0;
        parallelGridCore->getNodeState ()[process] = 0;
      }
      else
      {
        parallelGridCore->getNodeState ()[process] = 1;
      }
      sum_spread += spread[process];
    }
    else
    {
      spread[process] = 0;
    }
  }

  grid_coord diff = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 () - sum_spread;
  int i = 0;
  while (diff > 0)
  {
    if (parallelGridCore->getNodeState ()[i] == 1)
    {
      spread[i]++;
      diff--;
    }

    ++i;
    if (i == parallelGridCore->getTotalProcCount ())
    {
      i = 0;
    }
  }

  /*
   * Now check if smth should be disabled
   */
  bool flag = true;
  while (flag)
  {
    std::vector<Entry_t> borders;

    for (int i = 0; i < parallelGridCore->getTotalProcCount (); ++i)
    {
      for (int j = i + 1; j < parallelGridCore->getTotalProcCount (); ++j)
      {
        if (latency[i][j] > 0 && bandwidth[i][j] > 0
            && parallelGridCore->getNodeState ()[i] == 1
            && parallelGridCore->getNodeState ()[j] == 1)
        {
          borders.push_back (Entry_t (i, j, latency[i][j] + 1 / bandwidth[i][j]));
        }
      }
    }

    if (borders.empty ())
    {
      printf ("Empty borders\n");
      break;
    }

    std::sort (borders.begin (), borders.end ());

    for (int index = borders.size () - 1; index >= 0 && flag; --index)
    {
      // try to remove this connection
      Entry_t entry = borders[index];

      FPValue perf_left = 0;
      uint32_t count_left = 0;
      for (int i = 0; i <= entry.pid1; ++i)
      {
        if (parallelGridCore->getNodeState ()[i] == 1)
        {
          perf_left += speed[i];
          ++count_left;
        }
      }

      FPValue perf_right = 0;
      uint32_t count_right = 0;
      for (int i = entry.pid2; i < parallelGridCore->getTotalProcCount (); ++i)
      {
        if (parallelGridCore->getNodeState ()[i] == 1)
        {
          perf_right += speed[i];
          ++count_right;
        }
      }

      FPValue perf_all = perf_left + perf_right;
      uint32_t count_all = count_left + count_right;

      printf ("# %d =========== %f %f =======\n", parallelGridCore->getProcessId (), perf_left, perf_right);

      FPValue overallSize = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ();

      // find the next maximum in the left
      int i;
      for (i = index + 1; i >= 0; --i)
      {
        if (borders[i].pid1 < borders[index].pid1
            && borders[i].pid2 < borders[index].pid1)
        {
          break;
        }
      }

      // find the next maximum in the right
      int j;
      for (j = index + 1; j >= 0; --j)
      {
        if (borders[j].pid1 > borders[index].pid2
            && borders[j].pid2 > borders[index].pid2)
        {
          break;
        }
      }

      bool noSecondMaxLeft = false;
      bool noSecondMaxRight = false;

      if (i < 0)
      {
        // no second max found
        noSecondMaxLeft = true;
      }
      if (j < 0)
      {
        // no second max found
        noSecondMaxRight = true;
      }

      FPValue valueLeft;
      if (noSecondMaxLeft)
      {
        // either the single node to the left, or no info there
        ASSERT (count_left == 1);
        valueLeft = overallSize / perf_left;
      }
      else
      {
        valueLeft = overallSize / perf_left + 2 * borders[i].val;
      }

      FPValue valueRight;
      if (noSecondMaxRight)
      {
        // either the single node to the left, or no info there
        ASSERT (count_right == 1);
        valueRight = overallSize / perf_right;
      }
      else
      {
        valueRight = overallSize / perf_right + 2 * borders[j].val;
      }

      FPValue valueAll = overallSize / perf_all + 2 * borders[index].val;

      printf ("%f %f %f\n", valueAll, valueLeft, valueRight);

      if (valueLeft < valueAll && valueLeft < valueRight)
      {
        // disable nodes to the right
        printf ("DISABLE RIGHT\n");
        for (int k = entry.pid2; k < parallelGridCore->getTotalProcCount (); ++k)
        {
          parallelGridCore->getNodeState ()[k] = 0;
        }
      }
      else if (valueRight < valueAll && valueRight < valueLeft)
      {
        // disable nodes to the left
        printf ("DISABLE LEFT\n");
        for (int k = 0; k <= entry.pid1; ++k)
        {
          parallelGridCore->getNodeState ()[k] = 0;
        }
      }
      else
      {
        printf ("DECIDED TO BREAK 2\n");
        flag = false;
        break;
        // check for K nodes to the left/right
      }

      {
        sumSpeedEnabled = 0;
        for (int process = 0; process < parallelGridCore->getTotalProcCount (); ++process)
        {
          if (parallelGridCore->getNodeState ()[process] == 1)
          {
            sumSpeedEnabled += speed[process];
          }
        }

        sum_spread = 0;
        for (int process = 0; process < parallelGridCore->getTotalProcCount (); ++process)
        {
          if (parallelGridCore->getNodeState ()[process] == 1)
          {
            spread[process] = ((FPValue)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ()) * speed[process] / (sumSpeedEnabled);
            if (spread[process] < 1)
            {
              spread[process] = 0;
              parallelGridCore->getNodeState ()[process] = 0;
            }
            else
            {
              parallelGridCore->getNodeState ()[process] = 1;
            }
            sum_spread += spread[process];
          }
          else
          {
            spread[process] = 0;
          }
        }

        grid_coord diff = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 () - sum_spread;
        int i = 0;
        while (diff > 0)
        {
          if (parallelGridCore->getNodeState ()[i] == 1)
          {
            spread[i]++;
            diff--;
          }

          ++i;
          if (i == parallelGridCore->getTotalProcCount ())
          {
            i = 0;
          }
        }
      }
    }
  }

  grid_coord x = spread[parallelGridCore->getProcessId ()];

  printf ("#%d state=%d x=%d speed=%f (perfpoints=%f, perftimes=%f) totalX=%f difft=%u sumSpeedEnabled=%f\n",
          parallelGridCore->getProcessId (),
          parallelGridCore->getNodeState ()[parallelGridCore->getProcessId ()],
          x,
          speed[parallelGridCore->getProcessId ()],
          parallelGridCore->perfPointsValues[parallelGridCore->getProcessId ()],
          parallelGridCore->perfTimeValues[parallelGridCore->getProcessId ()],
          (FPValue)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 (),
          difft,
          sumSpeedEnabled);
  if (parallelGridCore->getProcessId () == 0)
  {
    for (int i = 0; i < parallelGridCore->getTotalProcCount (); ++i)
    for (int j = 0; j < parallelGridCore->getTotalProcCount (); ++j)
    {
      printf ("Share: %d--%d (latency=%.15f (%f / %f), bw=%.15f (%f / %f))\n",
        i, j,
        latency[i][j],
        parallelGridCore->latencySumValues[i][j],
        parallelGridCore->latencyCountValues[i][j],
        bandwidth[i][j],
        parallelGridCore->bandwidthSumValues[i][j],
        parallelGridCore->bandwidthCountValues[i][j]);
    }
  }

  newSize.set1 (x);

  sizeForCurNode = newSize;

  /*
   * ================== Clear current clocks and counter with number of points =========================================
   */

  parallelGridCore->ClearCalcClocks ();
  parallelGridCore->ClearShareClocks ();

  /*
   * ================== Set new counters for number of points for the next between rebalance episode ===================
   */
  if (parallelGridCore->getNodeState ()[parallelGridCore->getProcessId ()] == 1)
  {
    parallelGridCore->setCalcClockCount (parallelGridCore->getProcessId (), sizeForCurNode.calculateTotalCoord ());

#ifdef GRID_1D
    int dirL = parallelGridCore->getNodeForDirection (LEFT);
    if (dirL != PID_NONE)
    {
      parallelGridCore->setShareClockCountCur (dirL, 1);
    }
    int dirR = parallelGridCore->getNodeForDirection (RIGHT);
    if (dirR != PID_NONE)
    {
      parallelGridCore->setShareClockCountCur (dirR, 1);
    }
#endif
  }

#ifdef GRID_2D
  UNREACHABLE;
  // TODO: add setShareClockCount
#endif

#ifdef GRID_3D
  UNREACHABLE;
  // TODO: add setShareClockCount
#endif

  return true;
} /* ParallelYeeGridLayout::Rebalance */

#endif /* DYNAMIC_GRID */

#endif /* PARALLEL_GRID */

#endif /* PARALLEL_YEE_GRID_LAYOUT_H */
