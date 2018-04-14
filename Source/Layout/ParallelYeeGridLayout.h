#ifndef PARALLEL_YEE_GRID_LAYOUT_H
#define PARALLEL_YEE_GRID_LAYOUT_H

#include "YeeGridLayout.h"
#include "ParallelGridCore.h"
#include <algorithm>

#ifdef PARALLEL_GRID

class Entry_t
{
public:
  int coord;
  int axis; // 0 - Ox, 1 - Oy, 2 - Oz
  FPValue val;

  Entry_t (int newcoord, int newaxis, FPValue newVal)
  : coord (newcoord), axis (newaxis), val (newVal) {}

  bool operator < (Entry_t entry) const
  {
    return val < entry.val;
  }
};

enum Axis_t
{
  OX,
  OY,
  OZ
};

/**
 * Parallel Yee grid layout with size of grid per nodes
 */
template <SchemeType_t Type, LayoutType layout_type>
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

#ifdef DYNAMIC_GRID
  void InitializeCounters ();
#endif

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

private:
  void doAdditionalShareMeasurements (uint32_t, uint32_t);
  void InitializeIterationCounters (time_step);

  void getCurrentPerfValues (time_step, std::vector<FPValue> &, std::vector<FPValue> &);
  void getCurrentShareValues (std::vector< std::vector<int> > &, std::vector< std::vector<FPValue> > &, std::vector< std::vector<FPValue> > &);

  void calcTotalPerf (std::vector<FPValue> &, FPValue &, const std::vector<FPValue> &, const std::vector<FPValue> &);
  void calcTotalLatencyAndBandwidth (std::vector< std::vector<FPValue> > &, std::vector< std::vector<FPValue> > &,
                                     const std::vector< std::vector<FPValue> > &, const std::vector< std::vector<FPValue> > &,
                                     std::vector< std::vector<int> >);

  void spreadGridPointsPerAxis (std::vector<grid_coord> &, grid_coord &, FPValue, int, int, Axis_t, int, int, Axis_t, int, int, Axis_t,
                                const std::vector<FPValue> &, bool);

  void findMaxTimes (FPValue &max_share_LR,
                                                               FPValue &max_share_DU,
                                                               FPValue &max_share_BF,
                                                               FPValue &max_share_LD_RU,
                                                               FPValue &max_share_LU_RD,
                                                               FPValue &max_share_LB_RF,
                                                               FPValue &max_share_LF_RB,
                                                               FPValue &max_share_DB_UF,
                                                               FPValue &max_share_DF_UB,
                                                               FPValue &max_share_LDB_RUF,
                                                               FPValue &max_share_RDB_LUF,
                                                               FPValue &max_share_RUB_LDF,
                                                               FPValue &max_share_LUB_RDF,
                                                               const std::vector<grid_coord> &spreadX,
                                                               const std::vector<grid_coord> &spreadY,
                                                               const std::vector<grid_coord> &spreadZ,
                                                               const std::vector< std::vector<FPValue> > &latency,
                                                               const std::vector< std::vector<FPValue> > &bandwidth,
                                                               int axisStart1,
                                                               int axisSize1,
                                                               int axisStart2,
                                                               int axisSize2,
                                                               int axisStart3,
                                                               int axisSize3);
public:
#endif /* DYNAMIC_GRID */
}; /* ParallelYeeGridLayout */

/**
 * Identify size of grid for current computational node
 */
template <SchemeType_t Type, LayoutType layout_type>
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

#ifdef DYNAMIC_GRID

template <SchemeType_t Type, LayoutType layout_type>
void
ParallelYeeGridLayout<Type, layout_type>::InitializeCounters ()
{
  parallelGridCore->setCalcClockCount (parallelGridCore->getProcessId (), sizeForCurNode.calculateTotalCoord ());

#ifdef GRID_1D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
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
#endif

#ifdef GRID_2D
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
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

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
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

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
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
#endif

#ifdef GRID_3D
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int dirL = parallelGridCore->getNodeForDirection (LEFT);
  if (dirL != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirL, sizeForCurNode.get2 () * sizeForCurNode.get3 ());
  }
  int dirR = parallelGridCore->getNodeForDirection (RIGHT);
  if (dirR != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirR, sizeForCurNode.get2 () * sizeForCurNode.get3 ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int dirD = parallelGridCore->getNodeForDirection (DOWN);
  if (dirD != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirD, sizeForCurNode.get1 () * sizeForCurNode.get3 ());
  }
  int dirU = parallelGridCore->getNodeForDirection (UP);
  if (dirU != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirU, sizeForCurNode.get1 () * sizeForCurNode.get3 ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int dirB = parallelGridCore->getNodeForDirection (BACK);
  if (dirB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirB, sizeForCurNode.get1 () * sizeForCurNode.get2 ());
  }
  int dirF = parallelGridCore->getNodeForDirection (FRONT);
  if (dirF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirF, sizeForCurNode.get1 () * sizeForCurNode.get2 ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int dirLD = parallelGridCore->getNodeForDirection (LEFT_DOWN);
  if (dirLD != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLD, sizeForCurNode.get3 ());
  }
  int dirRD = parallelGridCore->getNodeForDirection (RIGHT_DOWN);
  if (dirRD != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRD, sizeForCurNode.get3 ());
  }

  int dirLU = parallelGridCore->getNodeForDirection (LEFT_UP);
  if (dirLU != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLU, sizeForCurNode.get3 ());
  }
  int dirRU = parallelGridCore->getNodeForDirection (RIGHT_UP);
  if (dirRU != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRU, sizeForCurNode.get3 ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int dirDB = parallelGridCore->getNodeForDirection (DOWN_BACK);
  if (dirDB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirDB, sizeForCurNode.get1 ());
  }
  int dirDF = parallelGridCore->getNodeForDirection (DOWN_FRONT);
  if (dirDF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirDF, sizeForCurNode.get1 ());
  }

  int dirUB = parallelGridCore->getNodeForDirection (UP_BACK);
  if (dirUB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirUB, sizeForCurNode.get1 ());
  }
  int dirUF = parallelGridCore->getNodeForDirection (UP_FRONT);
  if (dirUF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirUF, sizeForCurNode.get1 ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int dirLB = parallelGridCore->getNodeForDirection (LEFT_BACK);
  if (dirLB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLB, sizeForCurNode.get2 ());
  }
  int dirLF = parallelGridCore->getNodeForDirection (LEFT_FRONT);
  if (dirLF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLF, sizeForCurNode.get2 ());
  }

  int dirRB = parallelGridCore->getNodeForDirection (RIGHT_BACK);
  if (dirRB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRB, sizeForCurNode.get2 ());
  }
  int dirRF = parallelGridCore->getNodeForDirection (RIGHT_FRONT);
  if (dirRF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRF, sizeForCurNode.get2 ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int dirLDB = parallelGridCore->getNodeForDirection (LEFT_DOWN_BACK);
  if (dirLDB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLDB, 1);
  }
  int dirLDF = parallelGridCore->getNodeForDirection (LEFT_DOWN_FRONT);
  if (dirLDF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLDF, 1);
  }

  int dirLUB = parallelGridCore->getNodeForDirection (LEFT_UP_BACK);
  if (dirLUB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLUB, 1);
  }
  int dirLUF = parallelGridCore->getNodeForDirection (LEFT_UP_FRONT);
  if (dirLUF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirLUF, 1);
  }

  int dirRDB = parallelGridCore->getNodeForDirection (RIGHT_DOWN_BACK);
  if (dirRDB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRDB, 1);
  }
  int dirRDF = parallelGridCore->getNodeForDirection (RIGHT_DOWN_FRONT);
  if (dirRDF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRDF, 1);
  }

  int dirRUB = parallelGridCore->getNodeForDirection (RIGHT_UP_BACK);
  if (dirRUB != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRUB, 1);
  }
  int dirRUF = parallelGridCore->getNodeForDirection (RIGHT_UP_FRONT);
  if (dirRUF != PID_NONE)
  {
    parallelGridCore->setShareClockCountCur (dirRUF, 1);
  }
#endif
#endif
}

#endif

#ifdef GRID_1D

/**
 * Initialize size of grid per node
 */
template <SchemeType_t Type, LayoutType layout_type>
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
  InitializeCounters ();
#endif
} /* ParallelYeeGridLayout::Initialize */

#endif /* GRID_1D */

#ifdef GRID_2D

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y)

/**
 * Initialize size of grid per node
 */
template <SchemeType_t Type, LayoutType layout_type>
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
  InitializeCounters ();
#endif
} /* ParallelYeeGridLayout::Initialize */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY

/**
 * Initialize size of grid per node
 */
template <SchemeType_t Type, LayoutType layout_type>
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
  InitializeCounters ();
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
template <SchemeType_t Type, LayoutType layout_type>
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
  InitializeCounters ();
#endif
} /* ParallelYeeGridLayout::Initialize */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_1D_Z */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)

/**
 * Initialize size of grid per node
 */
template <SchemeType_t Type, LayoutType layout_type>
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
  InitializeCounters ();
#endif
} /* ParallelYeeGridLayout::Initialize */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ */

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ

/**
 * Initialize size of grid per node
 */
template <SchemeType_t Type, LayoutType layout_type>
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
  InitializeCounters ();
#endif
} /* ParallelYeeGridLayout::Initialize */

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#endif /* GRID_3D */

#ifdef DYNAMIC_GRID

template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::doAdditionalShareMeasurements (uint32_t latency_measure_count,
                                                                              uint32_t latency_buf_size)
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

  for (uint32_t index = latency_buf_size / CLOCK_BUF_SIZE; index < latency_buf_size; index += latency_buf_size / CLOCK_BUF_SIZE)
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

    uint32_t ssize;
    if (processTo != PID_NONE)
    {
      if (index != parallelGridCore->getShareClockCountCur (processTo))
      {
        ssize = index;
      }
      else
      {
        ssize = latency_buf_size;
      }
      parallelGridCore->setShareClockIterCur (processTo, ssize, latency_measure_count);
    }

    uint32_t rsize;
    if (processFrom != PID_NONE)
    {
      if (index != parallelGridCore->getShareClockCountCur (processFrom))
      {
        rsize = index;
      }
      else
      {
        rsize = latency_buf_size;
      }
      parallelGridCore->setShareClockIterCur (processFrom, rsize, latency_measure_count);
    }

    if (processTo != PID_NONE
        && processFrom == PID_NONE)
    {
      parallelGridCore->StartShareClock (processTo, ssize);

      int retCode = MPI_Send (tmp_buffer.data(), ssize, datatype, processTo, parallelGridCore->getProcessId (), parallelGridCore->getCommunicator ());
      ASSERT (retCode == MPI_SUCCESS);

      parallelGridCore->StopShareClock (processTo, ssize);
    }
    else if (processTo == PID_NONE
             && processFrom != PID_NONE)
    {
      parallelGridCore->StartShareClock (processFrom, rsize);

      MPI_Status status;
      int retCode = MPI_Recv (tmp_buffer.data(), rsize, datatype, processFrom, processFrom, parallelGridCore->getCommunicator (), &status);
      ASSERT (retCode == MPI_SUCCESS);

      parallelGridCore->StopShareClock (processFrom, rsize);
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
        parallelGridCore->StartShareClock (processTo, ssize);

        int retCode = MPI_Send (tmp_buffer.data(), ssize, datatype, processTo, parallelGridCore->getProcessId (), parallelGridCore->getCommunicator ());
        ASSERT (retCode == MPI_SUCCESS);

        parallelGridCore->StopShareClock (processTo, ssize);
        parallelGridCore->StartShareClock (processFrom, rsize);

        MPI_Status status;
        retCode = MPI_Recv (tmp_buffer.data(), rsize, datatype, processFrom, processFrom, parallelGridCore->getCommunicator (), &status);
        ASSERT (retCode == MPI_SUCCESS);

        parallelGridCore->StopShareClock (processFrom, rsize);
      }
      else
      {
        parallelGridCore->StartShareClock (processFrom, rsize);

        MPI_Status status;
        int retCode = MPI_Recv (tmp_buffer.data(), rsize, datatype, processFrom, processFrom, parallelGridCore->getCommunicator (), &status);
        ASSERT (retCode == MPI_SUCCESS);

        parallelGridCore->StopShareClock (processFrom, rsize);
        parallelGridCore->StartShareClock (processTo, ssize);

        retCode = MPI_Send (tmp_buffer.data(), ssize, datatype, processTo, parallelGridCore->getProcessId (), parallelGridCore->getCommunicator ());
        ASSERT (retCode == MPI_SUCCESS);

        parallelGridCore->StopShareClock (processTo, ssize);
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

template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::InitializeIterationCounters (time_step difft)
{
  if (parallelGridCore->getNodeState ()[parallelGridCore->getProcessId ()])
  {
    for (int dir = 0; dir < BUFFER_COUNT; ++dir)
    {
      int pid = parallelGridCore->getNodeForDirection ((BufferPosition) dir);
      if (pid == PID_NONE)
      {
        continue;
      }

      parallelGridCore->setShareClockIterCur (pid, parallelGridCore->getShareClockCountCur (pid), difft);
    }
  }
}

template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::getCurrentPerfValues (time_step difft,
                                                                     std::vector<FPValue> &curPoints,
                                                                     std::vector<FPValue> &curTimes)
{
  for (int i = 0; i < parallelGridCore->getTotalProcCount (); ++i)
  {
    curPoints[i] = difft * parallelGridCore->getCalcClockCount (i);

    CalcClock_t calcClockCur = parallelGridCore->getCalcClock (i);
#ifdef MPI_DYNAMIC_CLOCK
    FPValue timesec = calcClockCur;
#else
    FPValue timesec = (FPValue) calcClockCur.tv_sec + ((FPValue) calcClockCur.tv_nsec) / 1000000000;
#endif
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
}

template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::getCurrentShareValues (std::vector< std::vector<int> > &skipCurShareMeasurement,
                                                                      std::vector< std::vector<FPValue> > &curShareLatency,
                                                                      std::vector< std::vector<FPValue> > &curShareBandwidth)
{
  for (int i = 0; i < parallelGridCore->getTotalProcCount (); ++i)
  {
    for (int j = 0; j < parallelGridCore->getTotalProcCount (); ++j)
    {
      if (i == j)
      {
        continue;
      }

      if (parallelGridCore->getNodeState ()[i] == 0
          || parallelGridCore->getNodeState ()[j] == 0)
      {
        curShareLatency[i][j] = 0;
        curShareBandwidth[i][j] = 0;
        continue;
      }

      skipCurShareMeasurement[i][j] = 0;

      ShareClock_t map = parallelGridCore->getShareClock (i, j);

      FPValue latency = 0;
      FPValue bandwidth = 0;

      ASSERT (map.size () == CLOCK_BUF_SIZE
              || map.empty ());

  //     ASSERT (map.size () == 2);
  //
  //     FPValue val_size1 = 0;
  //     FPValue val_time1 = 0;
  //     FPValue val_size2 = 0;
  //     FPValue val_time2 = 0;
  //
  //     int index = 0;
  //     for (ShareClock_t::iterator it = map.begin (); it != map.end (); ++it)
  //     {
  //       FPValue bufSize = it->first;
  //
  // #ifndef MPI_DYNAMIC_CLOCK
  //       FPValue clocks = (FPValue) it->second.tv_sec + ((FPValue) it->second.tv_nsec) / 1000000000;
  // #else
  //       FPValue clocks = it->second;
  // #endif
  //
  //       // if (bufSize == 1)
  //       // {
  //       //   latency = clocks / parallelGridCore->getShareClockIter (i, j, bufSize);
  //       // }
  //       if (index == 0)
  //       {
  //         val_size1 = bufSize;
  //         val_time1 = clocks;
  //       }
  //       else if (index == 1)
  //       {
  //         val_size2 = bufSize;
  //         val_time2 = clocks;
  //       }
  //
  //       ++index;
  //     }

      FPValue avg_sum_x = 0;
      FPValue avg_sum_y = 0;
      FPValue avg_sum_x2 = 0;
      FPValue avg_sum_xy = 0;
      int index = 0;
      for (ShareClock_t::iterator it = map.begin (); it != map.end (); ++it)
      {
        FPValue bufSize = it->first;
#ifndef MPI_DYNAMIC_CLOCK
        FPValue clocks = (FPValue) it->second.tv_sec + ((FPValue) it->second.tv_nsec) / 1000000000;
#else
        FPValue clocks = it->second;
#endif

        avg_sum_x += bufSize;
        avg_sum_y += clocks;
        avg_sum_x2 += SQR(bufSize);
        avg_sum_xy += bufSize * clocks;

        ++index;
      }

      avg_sum_x /= index;
      avg_sum_y /= index;
      avg_sum_x2 /= index;
      avg_sum_xy /= index;

      bandwidth = (avg_sum_x2 - SQR(avg_sum_x)) / (avg_sum_xy - avg_sum_x * avg_sum_y);
      latency = avg_sum_y - avg_sum_x / bandwidth;

      if (latency < 0)
      {
        latency = 0;
      }

      if (bandwidth < 0)
      {
        bandwidth = 1000000000;
      }

      // bandwidth = (val_size1 - val_size2) / (val_time1 / parallelGridCore->getShareClockIter (i, j, val_size1) - val_time2 / parallelGridCore->getShareClockIter (i, j, val_size2));
      // latency = val_time1 / parallelGridCore->getShareClockIter (i, j, val_size1) - val_size1 / bandwidth;

  //       index = 0;
  //       for (ShareClock_t::iterator it = map.begin (); it != map.end (); ++it)
  //       {
  //         FPValue bufSize = it->first;
  //
  // #ifndef MPI_DYNAMIC_CLOCK
  //         FPValue clocks = (FPValue) it->second.tv_sec + ((FPValue) it->second.tv_nsec) / 1000000000;
  // #else
  //         FPValue clocks = it->second;
  // #endif
  //
  //         if (bufSize != 1)
  //         {
  //           FPValue t_avg = clocks / parallelGridCore->getShareClockIter (i, j, bufSize);
  //           bandwidth = bufSize / (t_avg - latency);
  //         }
  //       }

      curShareLatency[i][j] = latency;
      curShareBandwidth[i][j] = bandwidth;

      // if (curShareLatency[i][j] < 0)
      // {
      //   curShareLatency[i][j] = 0;
      // }
      // if (curShareBandwidth[i][j] < 0)
      // {
      //   curShareBandwidth[i][j] = 1000000000000;
      // }

      if (curShareBandwidth[i][j] < 0 || curShareLatency[i][j] < 0)
      {
        // printf ("INCORRECT: %d %d -> %f %f ---- first (%f, %f, %d), second (%f, %f, %d)\n",
        //   i, j, latency, bandwidth,
        //   val_size1, val_time1, parallelGridCore->getShareClockIter (i, j, val_size1),
        //   val_size2, val_time2, parallelGridCore->getShareClockIter (i, j, val_size2));

        printf ("INCORRECT: %d %d -> %f %f\n", i, j, latency, bandwidth);
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
        ASSERT (/*curShareLatency[i][j] != 0 && */curShareBandwidth[i][j] != 0);
      }
  #endif
    }
  }
}

template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::calcTotalPerf (std::vector<FPValue> &speed,
                                                              FPValue &sumSpeedEnabled,
                                                              const std::vector<FPValue> &curPoints,
                                                              const std::vector<FPValue> &curTimes)
{
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
}

template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::calcTotalLatencyAndBandwidth (std::vector< std::vector<FPValue> > &latency,
                                                                             std::vector< std::vector<FPValue> > &bandwidth,
                                                                             const std::vector< std::vector<FPValue> > &curShareLatency,
                                                                             const std::vector< std::vector<FPValue> > &curShareBandwidth,
                                                                             std::vector< std::vector<int> > skipCurShareMeasurement)
{
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
        if (skipCurShareMeasurement[process][i] == 0)
        {
          parallelGridCore->latencySumValues[process][i] += curShareLatency[process][i];
          parallelGridCore->latencyCountValues[process][i] += 1;

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
}

// template <SchemeType_t Type, LayoutType layout_type>
// void ParallelYeeGridLayout<Type, layout_type>::spreadGridPoints ()
// {
// #ifdef GRID_1D
//   std::vector<grid_coord> spread (parallelGridCore->getTotalProcCount ());
//   grid_coord sum_spread = 0;
//   for (int process = 0; process < parallelGridCore->getTotalProcCount (); ++process)
//   {
//     if (parallelGridCore->getNodeState ()[process] == 1)
//     {
//       spread[process] = ((FPValue)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ()) * speed[process] / (sumSpeedEnabled);
//       if (spread[process] < 1)
//       {
//         spread[process] = 0;
//         parallelGridCore->getNodeState ()[process] = 0;
//       }
//       else
//       {
//         parallelGridCore->getNodeState ()[process] = 1;
//       }
//       sum_spread += spread[process];
//     }
//     else
//     {
//       spread[process] = 0;
//     }
//   }
//
//   grid_coord diff = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 () - sum_spread;
//   int i = 0;
//   while (diff > 0)
//   {
//     if (parallelGridCore->getNodeState ()[i] == 1)
//     {
//       spread[i]++;
//       diff--;
//     }
//
//     ++i;
//     if (i == parallelGridCore->getTotalProcCount ())
//     {
//       i = 0;
//     }
//   }
// #endif
// }

template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::spreadGridPointsPerAxis (std::vector<grid_coord> &spread,
                                                                        grid_coord &sum_spread,
                                                                        FPValue sumSpeedEnabled,
                                                                        int axisStart1,
                                                                        int axisSize1,
                                                                        Axis_t axis1,
                                                                        int axisStart2,
                                                                        int axisSize2,
                                                                        Axis_t axis2,
                                                                        int axisStart3,
                                                                        int axisSize3,
                                                                        Axis_t axis3,
                                                                        const std::vector<FPValue> &speed,
                                                                        bool justFillSpread)
{
  ASSERT (axis1 != axis2 && axis2 != axis3 && axis1 != axis3);

  grid_coord totalSize;
  if (axis1 == OX)
  {
    totalSize = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ();
  }
  if (axis1 == OY)
  {
    totalSize = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ();
  }
  if (axis1 == OZ)
  {
    totalSize = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ();
  }

  for (int processAxis1 = 0; processAxis1 < axisSize1; ++processAxis1)
  {
    bool enabled = false;
    FPValue sumSpeedEnabledAxis = 0;

    for (int processAxis2 = 0; processAxis2 < axisSize2; ++processAxis2)
    for (int processAxis3 = 0; processAxis3 < axisSize3; ++processAxis3)
    {
      int processX;
      int processY;
      int processZ;
      if (axis1 == OX)
      {
        processX = processAxis1;
      }
      else if (axis1 == OY)
      {
        processY = processAxis1;
      }
      else if (axis1 == OZ)
      {
        processZ = processAxis1;
      }

      if (axis2 == OX)
      {
        processX = processAxis2;
      }
      else if (axis2 == OY)
      {
        processY = processAxis2;
      }
      else if (axis2 == OZ)
      {
        processZ = processAxis2;
      }

      if (axis3 == OX)
      {
        processX = processAxis3;
      }
      else if (axis3 == OY)
      {
        processY = processAxis3;
      }
      else if (axis3 == OZ)
      {
        processZ = processAxis3;
      }

      int pid = parallelGridCore->getNodeGrid (processX, processY, processZ);

      if (parallelGridCore->getNodeState ()[pid] == 1)
      {
        enabled = true;
        sumSpeedEnabledAxis += speed[pid];
      }
    }

    if (enabled)
    {
      spread[processAxis1] = ((FPValue) totalSize) * sumSpeedEnabledAxis / (sumSpeedEnabled);
      if (spread[processAxis1] < 1)
      {
        spread[processAxis1] = 0;

        if (!justFillSpread)
        {
          for (int processAxis2 = 0; processAxis2 < axisSize2; ++processAxis2)
          for (int processAxis3 = 0; processAxis3 < axisSize3; ++processAxis3)
          {
            int processX;
            int processY;
            int processZ;
            if (axis1 == OX)
            {
              processX = processAxis1;
            }
            else if (axis1 == OY)
            {
              processY = processAxis1;
            }
            else if (axis1 == OZ)
            {
              processZ = processAxis1;
            }

            if (axis2 == OX)
            {
              processX = processAxis2;
            }
            else if (axis2 == OY)
            {
              processY = processAxis2;
            }
            else if (axis2 == OZ)
            {
              processZ = processAxis2;
            }

            if (axis3 == OX)
            {
              processX = processAxis3;
            }
            else if (axis3 == OY)
            {
              processY = processAxis3;
            }
            else if (axis3 == OZ)
            {
              processZ = processAxis3;
            }

            int pid = parallelGridCore->getNodeGrid (processX, processY, processZ);
            parallelGridCore->getNodeState ()[pid] = 0;
          }
        }
      }
      // else
      // {
      //   parallelGridCore->getNodeState ()[process] = 1;
      // }
      sum_spread += spread[processAxis1];
    }
    else
    {
      spread[processAxis1] = 0;
    }
  }

  grid_coord diff = totalSize - sum_spread;
  int i = 0;
  while (diff > 0)
  {
    if (spread[i] > 0)
    {
      spread[i]++;
      diff--;
    }

    ++i;
    if (i == axisSize1)
    {
      i = 0;
    }
  }
}

template <SchemeType_t Type, LayoutType layout_type>
void ParallelYeeGridLayout<Type, layout_type>::findMaxTimes (FPValue &max_share_LR,
                                                             FPValue &max_share_DU,
                                                             FPValue &max_share_BF,
                                                             FPValue &max_share_LD_RU,
                                                             FPValue &max_share_LU_RD,
                                                             FPValue &max_share_LB_RF,
                                                             FPValue &max_share_LF_RB,
                                                             FPValue &max_share_DB_UF,
                                                             FPValue &max_share_DF_UB,
                                                             FPValue &max_share_LDB_RUF,
                                                             FPValue &max_share_RDB_LUF,
                                                             FPValue &max_share_RUB_LDF,
                                                             FPValue &max_share_LUB_RDF,
                                                             const std::vector<grid_coord> &spreadX,
                                                             const std::vector<grid_coord> &spreadY,
                                                             const std::vector<grid_coord> &spreadZ,
                                                             const std::vector< std::vector<FPValue> > &latency,
                                                             const std::vector< std::vector<FPValue> > &bandwidth,
                                                             int axisStartX,
                                                             int axisSizeX,
                                                             int axisStartY,
                                                             int axisSizeY,
                                                             int axisStartZ,
                                                             int axisSizeZ)
{
  for (int i = axisStartX; i < axisSizeX - 1; ++i)
  {
    for (int j = axisStartY; j < axisSizeY; ++j)
    for (int k = axisStartZ; k < axisSizeZ; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j, k);

      FPValue time_LR = latency[pid1][pid2] + spreadY[j] * spreadZ[k] / bandwidth[pid1][pid2];
      if (time_LR > max_share_LR)
      {
        max_share_LR = time_LR;
      }
    }

    for (int j = axisStartY; j < axisSizeY - 1; ++j)
    for (int k = axisStartZ; k < axisSizeZ; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j + 1, k);

      FPValue time_LD_RU = latency[pid1][pid2] + spreadZ[k] / bandwidth[pid1][pid2];
      if (time_LD_RU > max_share_LD_RU)
      {
        max_share_LD_RU = time_LD_RU;
      }
    }

    for (int j = axisStartY + 1; j < axisSizeY; ++j)
    for (int k = axisStartZ; k < axisSizeZ; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j - 1, k);

      FPValue time_LU_RD = latency[pid1][pid2] + spreadZ[k] / bandwidth[pid1][pid2];
      if (time_LU_RD > max_share_LU_RD)
      {
        max_share_LU_RD = time_LU_RD;
      }
    }

    for (int j = axisStartY; j < axisSizeY; ++j)
    for (int k = axisStartZ; k < axisSizeZ - 1; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j, k + 1);

      FPValue time_LB_RF = latency[pid1][pid2] + spreadY[j] / bandwidth[pid1][pid2];
      if (time_LB_RF > max_share_LB_RF)
      {
        max_share_LB_RF = time_LB_RF;
      }
    }

    for (int j = axisStartY; j < axisSizeY; ++j)
    for (int k = axisStartZ + 1; k < axisSizeZ; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j, k - 1);

      FPValue time_LF_RB = latency[pid1][pid2] + spreadY[j] / bandwidth[pid1][pid2];
      if (time_LF_RB > max_share_LF_RB)
      {
        max_share_LF_RB = time_LF_RB;
      }
    }

    for (int j = axisStartY; j < axisSizeY - 1; ++j)
    for (int k = axisStartZ; k < axisSizeZ - 1; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j + 1, k + 1);

      FPValue time_LDB_RUF = latency[pid1][pid2] + 1.0 / bandwidth[pid1][pid2];
      if (time_LDB_RUF > max_share_LDB_RUF)
      {
        max_share_LDB_RUF = time_LDB_RUF;
      }
    }

    for (int j = axisStartY + 1; j < axisSizeY; ++j)
    for (int k = axisStartZ; k < axisSizeZ - 1; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i + 1, j - 1, k + 1);

      FPValue time_LUB_RDF = latency[pid1][pid2] + 1.0 / bandwidth[pid1][pid2];
      if (time_LUB_RDF > max_share_LUB_RDF)
      {
        max_share_LUB_RDF = time_LUB_RDF;
      }
    }
  }

  for (int i = axisStartX; i < axisSizeX; ++i)
  {
    for (int j = axisStartY; j < axisSizeY - 1; ++j)
    for (int k = axisStartZ; k < axisSizeZ; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i, j + 1, k);

      FPValue time_DU = latency[pid1][pid2] + spreadX[i] * spreadZ[k] / bandwidth[pid1][pid2];
      if (time_DU > max_share_DU)
      {
        max_share_DU = time_DU;
      }
    }

    for (int j = axisStartY; j < axisSizeY; ++j)
    for (int k = axisStartZ; k < axisSizeZ - 1; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i, j, k + 1);

      FPValue time_BF = latency[pid1][pid2] + spreadX[i] * spreadY[j] / bandwidth[pid1][pid2];
      if (time_BF > max_share_BF)
      {
        max_share_BF = time_BF;
      }
    }

    for (int j = axisStartY; j < axisSizeY - 1; ++j)
    for (int k = axisStartZ; k < axisSizeZ - 1; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i, j + 1, k + 1);

      FPValue time_DB_UF = latency[pid1][pid2] + spreadX[i] / bandwidth[pid1][pid2];
      if (time_DB_UF > max_share_DB_UF)
      {
        max_share_DB_UF = time_DB_UF;
      }
    }

    for (int j = axisStartY; j < axisSizeY - 1; ++j)
    for (int k = axisStartZ + 1; k < axisSizeZ; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i, j + 1, k - 1);

      FPValue time_DF_UB = latency[pid1][pid2] + spreadX[i] / bandwidth[pid1][pid2];
      if (time_DF_UB > max_share_DF_UB)
      {
        max_share_DF_UB = time_DF_UB;
      }
    }
  }

  for (int i = axisStartX + 1; i < axisSizeX; ++i)
  {
    for (int j = axisStartY; j < axisSizeY - 1; ++j)
    for (int k = axisStartZ; k < axisSizeZ - 1; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i - 1, j + 1, k + 1);

      FPValue time_RDB_LUF = latency[pid1][pid2] + 1.0 / bandwidth[pid1][pid2];
      if (time_RDB_LUF > max_share_RDB_LUF)
      {
        max_share_RDB_LUF = time_RDB_LUF;
      }
    }

    for (int j = axisStartY + 1; j < axisSizeY; ++j)
    for (int k = axisStartZ; k < axisSizeZ - 1; ++k)
    {
      int pid1 = parallelGridCore->getNodeGrid (i, j, k);
      int pid2 = parallelGridCore->getNodeGrid (i - 1, j - 1, k + 1);

      FPValue time_RUB_LDF = latency[pid1][pid2] + 1.0 / bandwidth[pid1][pid2];
      if (time_RUB_LDF > max_share_RUB_LDF)
      {
        max_share_RUB_LDF = time_RUB_LDF;
      }
    }
  }
}

/**
 * Rebalance total grid size between notes
 *
 * @return true if size was changed
 *         false, otherwise
 */
template <SchemeType_t Type, LayoutType layout_type>
bool ParallelYeeGridLayout<Type, layout_type>::Rebalance (time_step difft) /**< number of time steps elapsed since the last rebalance */
{
  ParallelGridCoordinate newSize = sizeForCurNode;
  ParallelGridCoordinate oldSize = sizeForCurNode;

  uint32_t latency_measure_count = 10;
  uint32_t latency_buf_size = 100000;

  InitializeIterationCounters (difft);

  // Measure latency
  doAdditionalShareMeasurements (latency_measure_count, latency_buf_size);

  parallelGridCore->ShareClocks ();

  /*
   * ==== Get current values of performance ====
   */

  std::vector<FPValue> curPoints (parallelGridCore->getTotalProcCount ());
  std::vector<FPValue> curTimes (parallelGridCore->getTotalProcCount ());
  getCurrentPerfValues (difft, curPoints, curTimes);

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

  getCurrentShareValues (skipCurShareMeasurement, curShareLatency, curShareBandwidth);

  /*
   * ==== Calculate total perf ====
   */

  std::vector<FPValue> speed (parallelGridCore->getTotalProcCount ());
  FPValue sumSpeedEnabled = 0;
  calcTotalPerf (speed, sumSpeedEnabled, curPoints, curTimes);

  /*
   * ==== Calculate total latency and bandwidth ====
   */

  std::vector< std::vector<FPValue> > latency (parallelGridCore->getTotalProcCount ());
  std::vector< std::vector<FPValue> > bandwidth (parallelGridCore->getTotalProcCount ());
  calcTotalLatencyAndBandwidth (latency, bandwidth, curShareLatency, curShareBandwidth, skipCurShareMeasurement);

  /*
   * ==== Spread ====
   */
  // spreadGridPoints ();

#ifdef GRID_3D
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  std::vector<grid_coord> spreadX (parallelGridCore->getNodeGridSizeX ());
  grid_coord sum_spreadX = 0;

  spreadGridPointsPerAxis (spreadX, sum_spreadX, sumSpeedEnabled,
                           0,
                           parallelGridCore->getNodeGridSizeX (),
                           OX,
                           0,
                           parallelGridCore->getNodeGridSizeY (),
                           OY,
                           0,
                           parallelGridCore->getNodeGridSizeZ (),
                           OZ,
                           speed,
                           false);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  std::vector<grid_coord> spreadY (parallelGridCore->getNodeGridSizeY ());
  grid_coord sum_spreadY = 0;

  spreadGridPointsPerAxis (spreadY, sum_spreadY, sumSpeedEnabled,
                           0,
                           parallelGridCore->getNodeGridSizeY (),
                           OY,
                           0,
                           parallelGridCore->getNodeGridSizeX (),
                           OX,
                           0,
                           parallelGridCore->getNodeGridSizeZ (),
                           OZ,
                           speed,
                           false);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  std::vector<grid_coord> spreadZ (parallelGridCore->getNodeGridSizeZ ());
  grid_coord sum_spreadZ = 0;

  spreadGridPointsPerAxis (spreadZ, sum_spreadZ, sumSpeedEnabled,
                           0,
                           parallelGridCore->getNodeGridSizeZ (),
                           OZ,
                           0,
                           parallelGridCore->getNodeGridSizeX (),
                           OX,
                           0,
                           parallelGridCore->getNodeGridSizeY (),
                           OY,
                           speed,
                           false);
#endif
#endif

  grid_coord x = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 ();
  grid_coord y = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 ();
  grid_coord z = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 ();

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
  || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  x = spreadX[parallelGridCore->getNodeGridX ()];
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
  || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  y = spreadY[parallelGridCore->getNodeGridY ()];
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) \
  || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  z = spreadZ[parallelGridCore->getNodeGridZ ()];
#endif

  std::vector<FPValue> latencyX (parallelGridCore->getNodeGridSizeX () - 1);

  std::vector<FPValue> latencyX_diag_up (parallelGridCore->getNodeGridSizeX () - 1);
  std::vector<FPValue> latencyX_diag_down (parallelGridCore->getNodeGridSizeX () - 1);
  std::vector<FPValue> latencyX_diag_back (parallelGridCore->getNodeGridSizeX () - 1);
  std::vector<FPValue> latencyX_diag_front (parallelGridCore->getNodeGridSizeX () - 1);

  std::vector<FPValue> latencyX_diag_up_back (parallelGridCore->getNodeGridSizeX () - 1);
  std::vector<FPValue> latencyX_diag_up_front (parallelGridCore->getNodeGridSizeX () - 1);
  std::vector<FPValue> latencyX_diag_down_back (parallelGridCore->getNodeGridSizeX () - 1);
  std::vector<FPValue> latencyX_diag_down_front (parallelGridCore->getNodeGridSizeX () - 1);

  std::vector<FPValue> bandwidthX (parallelGridCore->getNodeGridSizeX () - 1);

  std::vector<FPValue> bandwidthX_diag_up (parallelGridCore->getNodeGridSizeX () - 1);
  std::vector<FPValue> bandwidthX_diag_down (parallelGridCore->getNodeGridSizeX () - 1);
  std::vector<FPValue> bandwidthX_diag_back (parallelGridCore->getNodeGridSizeX () - 1);
  std::vector<FPValue> bandwidthX_diag_front (parallelGridCore->getNodeGridSizeX () - 1);

  std::vector<FPValue> bandwidthX_diag_up_back (parallelGridCore->getNodeGridSizeX () - 1);
  std::vector<FPValue> bandwidthX_diag_up_front (parallelGridCore->getNodeGridSizeX () - 1);
  std::vector<FPValue> bandwidthX_diag_down_back (parallelGridCore->getNodeGridSizeX () - 1);
  std::vector<FPValue> bandwidthX_diag_down_front (parallelGridCore->getNodeGridSizeX () - 1);

  {
    for (int i = 0; i < parallelGridCore->getNodeGridSizeX () - 1; ++i)
    {
      for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
      for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j, k);

        latencyX[i] += latency[pid1][pid2];
        bandwidthX[i] += bandwidth[pid1][pid2];
      }
      latencyX[i] /= parallelGridCore->getNodeGridSizeY () * parallelGridCore->getNodeGridSizeZ ();
      bandwidthX[i] /= parallelGridCore->getNodeGridSizeY () * parallelGridCore->getNodeGridSizeZ ();

      for (int j = 0; j < parallelGridCore->getNodeGridSizeY () - 1; ++j)
      for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j + 1, k);

        latencyX_diag_up[i] += latency[pid1][pid2];
        bandwidthX_diag_up[i] += bandwidth[pid1][pid2];
      }
      latencyX_diag_up[i] /= (parallelGridCore->getNodeGridSizeY () - 1) * parallelGridCore->getNodeGridSizeZ ();
      bandwidthX_diag_up[i] /= (parallelGridCore->getNodeGridSizeY () - 1) * parallelGridCore->getNodeGridSizeZ ();

      for (int j = 1; j < parallelGridCore->getNodeGridSizeY (); ++j)
      for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j - 1, k);

        latencyX_diag_down[i] += latency[pid1][pid2];
        bandwidthX_diag_down[i] += bandwidth[pid1][pid2];
      }
      latencyX_diag_down[i] /= (parallelGridCore->getNodeGridSizeY () - 1) * parallelGridCore->getNodeGridSizeZ ();
      bandwidthX_diag_down[i] /= (parallelGridCore->getNodeGridSizeY () - 1) * parallelGridCore->getNodeGridSizeZ ();

      for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
      for (int k = 0; k < parallelGridCore->getNodeGridSizeZ () - 1; ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j, k + 1);

        latencyX_diag_front[i] += latency[pid1][pid2];
        bandwidthX_diag_front[i] += bandwidth[pid1][pid2];
      }
      latencyX_diag_front[i] /= parallelGridCore->getNodeGridSizeY () * (parallelGridCore->getNodeGridSizeZ () - 1);
      bandwidthX_diag_front[i] /= parallelGridCore->getNodeGridSizeY () * (parallelGridCore->getNodeGridSizeZ () - 1);

      for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
      for (int k = 1; k < parallelGridCore->getNodeGridSizeZ (); ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j, k - 1);

        latencyX_diag_back[i] += latency[pid1][pid2];
        bandwidthX_diag_back[i] += bandwidth[pid1][pid2];
      }
      latencyX_diag_back[i] /= parallelGridCore->getNodeGridSizeY () * (parallelGridCore->getNodeGridSizeZ () - 1);
      bandwidthX_diag_back[i] /= parallelGridCore->getNodeGridSizeY () * (parallelGridCore->getNodeGridSizeZ () - 1);


      for (int j = 0; j < parallelGridCore->getNodeGridSizeY () - 1; ++j)
      for (int k = 0; k < parallelGridCore->getNodeGridSizeZ () - 1; ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j + 1, k + 1);

        latencyX_diag_up_front[i] += latency[pid1][pid2];
        bandwidthX_diag_up_front[i] += bandwidth[pid1][pid2];
      }
      latencyX_diag_up_front[i] /= (parallelGridCore->getNodeGridSizeY () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);
      bandwidthX_diag_up_front[i] /= (parallelGridCore->getNodeGridSizeY () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);

      for (int j = 0; j < parallelGridCore->getNodeGridSizeY () - 1; ++j)
      for (int k = 1; k < parallelGridCore->getNodeGridSizeZ (); ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j + 1, k - 1);

        latencyX_diag_up_back[i] += latency[pid1][pid2];
        bandwidthX_diag_up_back[i] += bandwidth[pid1][pid2];
      }
      latencyX_diag_up_back[i] /= (parallelGridCore->getNodeGridSizeY () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);
      bandwidthX_diag_up_back[i] /= (parallelGridCore->getNodeGridSizeY () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);

      for (int j = 1; j < parallelGridCore->getNodeGridSizeY (); ++j)
      for (int k = 0; k < parallelGridCore->getNodeGridSizeZ () - 1; ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j - 1, k + 1);

        latencyX_diag_down_front[i] += latency[pid1][pid2];
        bandwidthX_diag_down_front[i] += bandwidth[pid1][pid2];
      }
      latencyX_diag_down_front[i] /= (parallelGridCore->getNodeGridSizeY () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);
      bandwidthX_diag_down_front[i] /= (parallelGridCore->getNodeGridSizeY () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);

      for (int j = 1; j < parallelGridCore->getNodeGridSizeY (); ++j)
      for (int k = 1; k < parallelGridCore->getNodeGridSizeZ (); ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j - 1, k - 1);

        latencyX_diag_down_back[i] += latency[pid1][pid2];
        bandwidthX_diag_down_back[i] += bandwidth[pid1][pid2];
      }
      latencyX_diag_down_back[i] /= (parallelGridCore->getNodeGridSizeY () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);
      bandwidthX_diag_down_back[i] /= (parallelGridCore->getNodeGridSizeY () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);
    }
  }

  std::vector<FPValue> latencyY (parallelGridCore->getNodeGridSizeY () - 1);

  std::vector<FPValue> latencyY_diag_left (parallelGridCore->getNodeGridSizeY () - 1);
  std::vector<FPValue> latencyY_diag_right (parallelGridCore->getNodeGridSizeY () - 1);
  std::vector<FPValue> latencyY_diag_back (parallelGridCore->getNodeGridSizeY () - 1);
  std::vector<FPValue> latencyY_diag_front (parallelGridCore->getNodeGridSizeY () - 1);

  std::vector<FPValue> latencyY_diag_left_back (parallelGridCore->getNodeGridSizeY () - 1);
  std::vector<FPValue> latencyY_diag_left_front (parallelGridCore->getNodeGridSizeY () - 1);
  std::vector<FPValue> latencyY_diag_right_back (parallelGridCore->getNodeGridSizeY () - 1);
  std::vector<FPValue> latencyY_diag_right_front (parallelGridCore->getNodeGridSizeY () - 1);

  std::vector<FPValue> bandwidthY (parallelGridCore->getNodeGridSizeY () - 1);

  std::vector<FPValue> bandwidthY_diag_left (parallelGridCore->getNodeGridSizeY () - 1);
  std::vector<FPValue> bandwidthY_diag_right (parallelGridCore->getNodeGridSizeY () - 1);
  std::vector<FPValue> bandwidthY_diag_back (parallelGridCore->getNodeGridSizeY () - 1);
  std::vector<FPValue> bandwidthY_diag_front (parallelGridCore->getNodeGridSizeY () - 1);

  std::vector<FPValue> bandwidthY_diag_left_back (parallelGridCore->getNodeGridSizeY () - 1);
  std::vector<FPValue> bandwidthY_diag_left_front (parallelGridCore->getNodeGridSizeY () - 1);
  std::vector<FPValue> bandwidthY_diag_right_back (parallelGridCore->getNodeGridSizeY () - 1);
  std::vector<FPValue> bandwidthY_diag_right_front (parallelGridCore->getNodeGridSizeY () - 1);

  {
    for (int j = 0; j < parallelGridCore->getNodeGridSizeY () - 1; ++j)
    {
      for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
      for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i, j + 1, k);

        latencyY[j] += latency[pid1][pid2];
        bandwidthY[j] += bandwidth[pid1][pid2];
      }
      latencyY[j] /= parallelGridCore->getNodeGridSizeX () * parallelGridCore->getNodeGridSizeZ ();
      bandwidthY[j] /= parallelGridCore->getNodeGridSizeX () * parallelGridCore->getNodeGridSizeZ ();

      for (int i = 0; i < parallelGridCore->getNodeGridSizeX () - 1; ++i)
      for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j + 1, k);

        latencyY_diag_right[j] += latency[pid1][pid2];
        bandwidthY_diag_right[j] += bandwidth[pid1][pid2];
      }
      latencyY_diag_right[j] /= (parallelGridCore->getNodeGridSizeX () - 1) * parallelGridCore->getNodeGridSizeZ ();
      bandwidthY_diag_right[j] /= (parallelGridCore->getNodeGridSizeX () - 1) * parallelGridCore->getNodeGridSizeZ ();

      for (int i = 1; i < parallelGridCore->getNodeGridSizeX (); ++i)
      for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i - 1, j + 1, k);

        latencyY_diag_left[j] += latency[pid1][pid2];
        bandwidthY_diag_left[j] += bandwidth[pid1][pid2];
      }
      latencyY_diag_left[j] /= (parallelGridCore->getNodeGridSizeX () - 1) * parallelGridCore->getNodeGridSizeZ ();
      bandwidthY_diag_left[j] /= (parallelGridCore->getNodeGridSizeX () - 1) * parallelGridCore->getNodeGridSizeZ ();

      for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
      for (int k = 0; k < parallelGridCore->getNodeGridSizeZ () - 1; ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i, j + 1, k + 1);

        latencyY_diag_front[j] += latency[pid1][pid2];
        bandwidthY_diag_front[j] += bandwidth[pid1][pid2];
      }
      latencyY_diag_front[j] /= parallelGridCore->getNodeGridSizeX () * (parallelGridCore->getNodeGridSizeZ () - 1);
      bandwidthY_diag_front[j] /= parallelGridCore->getNodeGridSizeX () * (parallelGridCore->getNodeGridSizeZ () - 1);

      for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
      for (int k = 1; k < parallelGridCore->getNodeGridSizeZ (); ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i, j + 1, k - 1);

        latencyY_diag_back[j] += latency[pid1][pid2];
        bandwidthY_diag_back[j] += bandwidth[pid1][pid2];
      }
      latencyY_diag_back[j] /= parallelGridCore->getNodeGridSizeX () * (parallelGridCore->getNodeGridSizeZ () - 1);
      bandwidthY_diag_back[j] /= parallelGridCore->getNodeGridSizeX () * (parallelGridCore->getNodeGridSizeZ () - 1);


      for (int i = 0; i < parallelGridCore->getNodeGridSizeX () - 1; ++i)
      for (int k = 0; k < parallelGridCore->getNodeGridSizeZ () - 1; ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j + 1, k + 1);

        latencyY_diag_right_front[j] += latency[pid1][pid2];
        bandwidthY_diag_right_front[j] += bandwidth[pid1][pid2];
      }
      latencyY_diag_right_front[j] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);
      bandwidthY_diag_right_front[j] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);

      for (int i = 0; i < parallelGridCore->getNodeGridSizeX () - 1; ++i)
      for (int k = 1; k < parallelGridCore->getNodeGridSizeZ (); ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j + 1, k - 1);

        latencyY_diag_right_back[j] += latency[pid1][pid2];
        bandwidthY_diag_right_back[j] += bandwidth[pid1][pid2];
      }
      latencyY_diag_right_back[j] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);
      bandwidthY_diag_right_back[j] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);

      for (int i = 1; i < parallelGridCore->getNodeGridSizeX (); ++i)
      for (int k = 0; k < parallelGridCore->getNodeGridSizeZ () - 1; ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i - 1, j + 1, k + 1);

        latencyY_diag_left_front[j] += latency[pid1][pid2];
        bandwidthY_diag_left_front[j] += bandwidth[pid1][pid2];
      }
      latencyY_diag_left_front[j] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);
      bandwidthY_diag_left_front[j] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);

      for (int i = 1; i < parallelGridCore->getNodeGridSizeX (); ++i)
      for (int k = 1; k < parallelGridCore->getNodeGridSizeZ (); ++k)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i - 1, j + 1, k - 1);

        latencyY_diag_left_back[j] += latency[pid1][pid2];
        bandwidthY_diag_left_back[j] += bandwidth[pid1][pid2];
      }
      latencyY_diag_left_back[j] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);
      bandwidthY_diag_left_back[j] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeZ () - 1);
    }
  }

  std::vector<FPValue> latencyZ (parallelGridCore->getNodeGridSizeZ () - 1);

  std::vector<FPValue> latencyZ_diag_left (parallelGridCore->getNodeGridSizeZ () - 1);
  std::vector<FPValue> latencyZ_diag_right (parallelGridCore->getNodeGridSizeZ () - 1);
  std::vector<FPValue> latencyZ_diag_down (parallelGridCore->getNodeGridSizeZ () - 1);
  std::vector<FPValue> latencyZ_diag_up (parallelGridCore->getNodeGridSizeZ () - 1);

  std::vector<FPValue> latencyZ_diag_left_down (parallelGridCore->getNodeGridSizeZ () - 1);
  std::vector<FPValue> latencyZ_diag_left_up (parallelGridCore->getNodeGridSizeZ () - 1);
  std::vector<FPValue> latencyZ_diag_right_down (parallelGridCore->getNodeGridSizeZ () - 1);
  std::vector<FPValue> latencyZ_diag_right_up (parallelGridCore->getNodeGridSizeZ () - 1);

  std::vector<FPValue> bandwidthZ (parallelGridCore->getNodeGridSizeZ () - 1);

  std::vector<FPValue> bandwidthZ_diag_left (parallelGridCore->getNodeGridSizeZ () - 1);
  std::vector<FPValue> bandwidthZ_diag_right (parallelGridCore->getNodeGridSizeZ () - 1);
  std::vector<FPValue> bandwidthZ_diag_down (parallelGridCore->getNodeGridSizeZ () - 1);
  std::vector<FPValue> bandwidthZ_diag_up (parallelGridCore->getNodeGridSizeZ () - 1);

  std::vector<FPValue> bandwidthZ_diag_left_down (parallelGridCore->getNodeGridSizeZ () - 1);
  std::vector<FPValue> bandwidthZ_diag_left_up (parallelGridCore->getNodeGridSizeZ () - 1);
  std::vector<FPValue> bandwidthZ_diag_right_down (parallelGridCore->getNodeGridSizeZ () - 1);
  std::vector<FPValue> bandwidthZ_diag_right_up (parallelGridCore->getNodeGridSizeZ () - 1);

  {
    for (int k = 0; k < parallelGridCore->getNodeGridSizeZ () - 1; ++k)
    {
      for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
      for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i, j, k + 1);

        latencyZ[k] += latency[pid1][pid2];
        bandwidthZ[k] += bandwidth[pid1][pid2];
      }
      latencyZ[k] /= parallelGridCore->getNodeGridSizeX () * parallelGridCore->getNodeGridSizeY ();
      bandwidthZ[k] /= parallelGridCore->getNodeGridSizeX () * parallelGridCore->getNodeGridSizeY ();

      for (int i = 0; i < parallelGridCore->getNodeGridSizeX () - 1; ++i)
      for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j, k + 1);

        latencyZ_diag_right[k] += latency[pid1][pid2];
        bandwidthZ_diag_right[k] += bandwidth[pid1][pid2];
      }
      latencyZ_diag_right[k] /= (parallelGridCore->getNodeGridSizeX () - 1) * parallelGridCore->getNodeGridSizeY ();
      bandwidthZ_diag_right[k] /= (parallelGridCore->getNodeGridSizeX () - 1) * parallelGridCore->getNodeGridSizeY ();

      for (int i = 1; i < parallelGridCore->getNodeGridSizeX (); ++i)
      for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i - 1, j, k + 1);

        latencyZ_diag_left[k] += latency[pid1][pid2];
        bandwidthZ_diag_left[k] += bandwidth[pid1][pid2];
      }
      latencyZ_diag_left[k] /= (parallelGridCore->getNodeGridSizeX () - 1) * parallelGridCore->getNodeGridSizeY ();
      bandwidthZ_diag_left[k] /= (parallelGridCore->getNodeGridSizeX () - 1) * parallelGridCore->getNodeGridSizeY ();

      for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
      for (int j = 0; j < parallelGridCore->getNodeGridSizeY () - 1; ++j)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i, j + 1, k + 1);

        latencyZ_diag_up[k] += latency[pid1][pid2];
        bandwidthZ_diag_up[k] += bandwidth[pid1][pid2];
      }
      latencyZ_diag_up[k] /= parallelGridCore->getNodeGridSizeX () * (parallelGridCore->getNodeGridSizeY () - 1);
      bandwidthZ_diag_up[k] /= parallelGridCore->getNodeGridSizeX () * (parallelGridCore->getNodeGridSizeY () - 1);

      for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
      for (int j = 1; j < parallelGridCore->getNodeGridSizeY (); ++j)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i, j - 1, k + 1);

        latencyZ_diag_down[k] += latency[pid1][pid2];
        bandwidthZ_diag_down[k] += bandwidth[pid1][pid2];
      }
      latencyZ_diag_down[k] /= parallelGridCore->getNodeGridSizeX () * (parallelGridCore->getNodeGridSizeY () - 1);
      bandwidthZ_diag_down[k] /= parallelGridCore->getNodeGridSizeX () * (parallelGridCore->getNodeGridSizeY () - 1);


      for (int i = 0; i < parallelGridCore->getNodeGridSizeX () - 1; ++i)
      for (int j = 0; j < parallelGridCore->getNodeGridSizeY () - 1; ++j)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j + 1, k + 1);

        latencyZ_diag_right_up[k] += latency[pid1][pid2];
        bandwidthZ_diag_right_up[k] += bandwidth[pid1][pid2];
      }
      latencyZ_diag_right_up[k] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeY () - 1);
      bandwidthZ_diag_right_up[k] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeY () - 1);

      for (int i = 1; i < parallelGridCore->getNodeGridSizeX (); ++i)
      for (int j = 0; j < parallelGridCore->getNodeGridSizeY () - 1; ++j)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i - 1, j + 1, k + 1);

        latencyZ_diag_left_up[k] += latency[pid1][pid2];
        bandwidthZ_diag_left_up[k] += bandwidth[pid1][pid2];
      }
      latencyZ_diag_left_up[k] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeY () - 1);
      bandwidthZ_diag_left_up[k] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeY () - 1);

      for (int i = 0; i < parallelGridCore->getNodeGridSizeX () - 1; ++i)
      for (int j = 1; j < parallelGridCore->getNodeGridSizeY (); ++j)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i + 1, j - 1, k + 1);

        latencyZ_diag_right_down[k] += latency[pid1][pid2];
        bandwidthZ_diag_right_down[k] += bandwidth[pid1][pid2];
      }
      latencyZ_diag_right_down[k] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeY () - 1);
      bandwidthZ_diag_right_down[k] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeY () - 1);

      for (int i = 1; i < parallelGridCore->getNodeGridSizeX (); ++i)
      for (int j = 1; j < parallelGridCore->getNodeGridSizeY (); ++j)
      {
        int pid1 = parallelGridCore->getNodeGrid (i, j, k);
        int pid2 = parallelGridCore->getNodeGrid (i - 1, j - 1, k + 1);

        latencyZ_diag_left_down[k] += latency[pid1][pid2];
        bandwidthZ_diag_left_down[k] += bandwidth[pid1][pid2];
      }
      latencyZ_diag_left_down[k] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeY () - 1);
      bandwidthZ_diag_left_down[k] /= (parallelGridCore->getNodeGridSizeX () - 1) * (parallelGridCore->getNodeGridSizeY () - 1);
    }
  }

  /*
   * Now check if smth should be disabled
   */
  bool flag = true;
  while (flag)
  {
    std::vector<Entry_t> borders;

    for (int i = 0; i < parallelGridCore->getNodeGridSizeX () - 1; ++i)
    {
      bool enabledL = false;
      bool enabledR = false;
      for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
      for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
      {
        if (parallelGridCore->getNodeState ()[parallelGridCore->getNodeGrid (i, j, k)] == 1)
        {
          enabledL = true;
        }

        if (parallelGridCore->getNodeState ()[parallelGridCore->getNodeGrid (i + 1, j, k)] == 1)
        {
          enabledR = true;
        }
      }

      if (enabledL && enabledR)
      {
        if (bandwidthX[i] > 0 && bandwidthX_diag_down[i] > 0
            && bandwidthX_diag_up[i] > 0 && bandwidthX_diag_back[i] > 0
            && bandwidthX_diag_front[i] > 0 && bandwidthX_diag_down_back[i] > 0
            && bandwidthX_diag_down_front[i] > 0 && bandwidthX_diag_up_back[i] > 0
            && bandwidthX_diag_up_front[i] > 0)
        {
          borders.push_back (Entry_t (i, 0, latencyX[i] + (parallelGridCore->getNodeGridSizeY () * parallelGridCore->getNodeGridSizeZ ()) / bandwidthX[i]
                                            + latencyX_diag_down[i] + parallelGridCore->getNodeGridSizeZ () / bandwidthX_diag_down[i]
                                            + latencyX_diag_up[i] + parallelGridCore->getNodeGridSizeZ () / bandwidthX_diag_up[i]
                                            + latencyX_diag_back[i] + parallelGridCore->getNodeGridSizeY () / bandwidthX_diag_back[i]
                                            + latencyX_diag_front[i] + parallelGridCore->getNodeGridSizeY () / bandwidthX_diag_front[i]
                                            + latencyX_diag_down_back[i] + 1.0 / bandwidthX_diag_down_back[i]
                                            + latencyX_diag_down_front[i] + 1.0 / bandwidthX_diag_down_front[i]
                                            + latencyX_diag_up_back[i] + 1.0 / bandwidthX_diag_up_back[i]
                                            + latencyX_diag_up_front[i] + 1.0 / bandwidthX_diag_up_front[i]
                                      ));
        }
      }
    }

    for (int j = 0; j < parallelGridCore->getNodeGridSizeY () - 1; ++j)
    {
      bool enabledD = false;
      bool enabledU = false;
      for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
      for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
      {
        if (parallelGridCore->getNodeState ()[parallelGridCore->getNodeGrid (i, j, k)] == 1)
        {
          enabledD = true;
        }

        if (parallelGridCore->getNodeState ()[parallelGridCore->getNodeGrid (i, j + 1, k)] == 1)
        {
          enabledU = true;
        }
      }

      if (enabledD && enabledU)
      {
        if (bandwidthY[j] > 0 && bandwidthY_diag_left[j] > 0
            && bandwidthY_diag_right[j] > 0 && bandwidthY_diag_back[j] > 0
            && bandwidthY_diag_front[j] > 0 && bandwidthY_diag_left_back[j] > 0
            && bandwidthY_diag_left_front[j] > 0 && bandwidthY_diag_right_back[j] > 0
            && bandwidthY_diag_right_front[j] > 0)
        {
          borders.push_back (Entry_t (j, 1, latencyY[j] + (parallelGridCore->getNodeGridSizeX () * parallelGridCore->getNodeGridSizeZ ()) / bandwidthY[j]
                                            + latencyY_diag_left[j] + parallelGridCore->getNodeGridSizeZ () / bandwidthY_diag_left[j]
                                            + latencyY_diag_right[j] + parallelGridCore->getNodeGridSizeZ () / bandwidthY_diag_right[j]
                                            + latencyY_diag_back[j] + parallelGridCore->getNodeGridSizeX () / bandwidthY_diag_back[j]
                                            + latencyY_diag_front[j] + parallelGridCore->getNodeGridSizeX () / bandwidthY_diag_front[j]
                                            + latencyY_diag_left_back[j] + 1.0 / bandwidthY_diag_left_back[j]
                                            + latencyY_diag_left_front[j] + 1.0 / bandwidthY_diag_left_front[j]
                                            + latencyY_diag_right_back[j] + 1.0 / bandwidthY_diag_right_back[j]
                                            + latencyY_diag_right_front[j] + 1.0 / bandwidthY_diag_right_front[j]
                                      ));
        }
      }
    }

    for (int k = 0; k < parallelGridCore->getNodeGridSizeZ () - 1; ++k)
    {
      bool enabledB = false;
      bool enabledF = false;
      for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
      for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
      {
        if (parallelGridCore->getNodeState ()[parallelGridCore->getNodeGrid (i, j, k)] == 1)
        {
          enabledB = true;
        }

        if (parallelGridCore->getNodeState ()[parallelGridCore->getNodeGrid (i, j, k + 1)] == 1)
        {
          enabledF = true;
        }
      }

      if (enabledB && enabledF)
      {
        if (bandwidthZ[k] > 0 && bandwidthZ_diag_left[k] > 0
            && bandwidthZ_diag_right[k] > 0 && bandwidthZ_diag_down[k] > 0
            && bandwidthZ_diag_up[k] > 0 && bandwidthZ_diag_left_down[k] > 0
            && bandwidthZ_diag_left_up[k] > 0 && bandwidthZ_diag_right_down[k] > 0
            && bandwidthZ_diag_right_up[k] > 0)
        {
          borders.push_back (Entry_t (k, 2, latencyZ[k] + (parallelGridCore->getNodeGridSizeX () * parallelGridCore->getNodeGridSizeY ()) / bandwidthZ[k]
                                            + latencyZ_diag_left[k] + parallelGridCore->getNodeGridSizeY () / bandwidthZ_diag_left[k]
                                            + latencyZ_diag_right[k] + parallelGridCore->getNodeGridSizeY () / bandwidthZ_diag_right[k]
                                            + latencyZ_diag_down[k] + parallelGridCore->getNodeGridSizeX () / bandwidthZ_diag_down[k]
                                            + latencyZ_diag_up[k] + parallelGridCore->getNodeGridSizeX () / bandwidthZ_diag_up[k]
                                            + latencyZ_diag_left_down[k] + 1.0 / bandwidthZ_diag_left_down[k]
                                            + latencyZ_diag_left_up[k] + 1.0 / bandwidthZ_diag_left_up[k]
                                            + latencyZ_diag_right_down[k] + 1.0 / bandwidthZ_diag_right_down[k]
                                            + latencyZ_diag_right_up[k] + 1.0 / bandwidthZ_diag_right_up[k]
                                      ));
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
      Entry_t entry = borders[index];
      printf ("!! BORDER %d : axis %d, coord %d, val %f\n", index, entry.axis, entry.coord, entry.val);
    }

    for (int index = borders.size () - 1; index >= 0 && flag; --index)
    {
      // try to remove this connection
      Entry_t entry = borders[index];

      FPValue perf_left;
      FPValue perf_right;

      uint32_t count_left;
      uint32_t count_right;
      if (entry.axis == 0)
      {
        perf_left = 0;
        count_left = 0;
        for (int i = 0; i <= entry.coord; ++i)
        {
          for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
          for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
          {
            int pid = parallelGridCore->getNodeGrid (i, j, k);
            if (parallelGridCore->getNodeState ()[pid] == 1)
            {
              perf_left += speed[pid];
              ++count_left;
            }
          }
        }

        perf_right = 0;
        count_right = 0;
        for (int i = entry.coord + 1; i < parallelGridCore->getNodeGridSizeX (); ++i)
        {
          for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
          for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
          {
            int pid = parallelGridCore->getNodeGrid (i, j, k);
            if (parallelGridCore->getNodeState ()[pid] == 1)
            {
              perf_right += speed[pid];
              ++count_right;
            }
          }
        }
      }
      else if (entry.axis == 1)
      {
        perf_left = 0;
        count_left = 0;
        for (int j = 0; j <= entry.coord; ++j)
        {
          for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
          for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
          {
            int pid = parallelGridCore->getNodeGrid (i, j, k);
            if (parallelGridCore->getNodeState ()[pid] == 1)
            {
              perf_left += speed[pid];
              ++count_left;
            }
          }
        }

        perf_right = 0;
        count_right = 0;
        for (int j = entry.coord + 1; j < parallelGridCore->getNodeGridSizeY (); ++j)
        {
          for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
          for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
          {
            int pid = parallelGridCore->getNodeGrid (i, j, k);
            if (parallelGridCore->getNodeState ()[pid] == 1)
            {
              perf_right += speed[pid];
              ++count_right;
            }
          }
        }
      }
      else if (entry.axis == 2)
      {
        perf_left = 0;
        count_left = 0;
        for (int k = 0; k <= entry.coord; ++k)
        {
          for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
          for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
          {
            int pid = parallelGridCore->getNodeGrid (i, j, k);
            if (parallelGridCore->getNodeState ()[pid] == 1)
            {
              perf_left += speed[pid];
              ++count_left;
            }
          }
        }

        perf_right = 0;
        count_right = 0;
        for (int k = entry.coord + 1; k < parallelGridCore->getNodeGridSizeZ (); ++k)
        {
          for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
          for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
          {
            int pid = parallelGridCore->getNodeGrid (i, j, k);
            if (parallelGridCore->getNodeState ()[pid] == 1)
            {
              perf_right += speed[pid];
              ++count_right;
            }
          }
        }
      }
      else
      {
        UNREACHABLE;
      }

      FPValue perf_all = perf_left + perf_right;
      uint32_t count_all = count_left + count_right;

      sumSpeedEnabled = perf_all;

      //printf ("# %d =========== %f %f %f =======\n", parallelGridCore->getProcessId (), perf_left, perf_right, perf_all);

      FPValue overallSize = YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.calculateTotalCoord ();

      FPValue max_share_LR_left = 0;
      FPValue max_share_DU_left = 0;
      FPValue max_share_BF_left = 0;

      FPValue max_share_LD_RU_left = 0;
      FPValue max_share_LU_RD_left = 0;
      FPValue max_share_LB_RF_left = 0;
      FPValue max_share_LF_RB_left = 0;
      FPValue max_share_DB_UF_left = 0;
      FPValue max_share_DF_UB_left = 0;

      FPValue max_share_LDB_RUF_left = 0;
      FPValue max_share_RDB_LUF_left = 0;
      FPValue max_share_RUB_LDF_left = 0;
      FPValue max_share_LUB_RDF_left = 0;

      FPValue max_share_time_left = 0;
      FPValue valueLeft = 0;


      FPValue max_share_LR_right = 0;
      FPValue max_share_DU_right = 0;
      FPValue max_share_BF_right = 0;

      FPValue max_share_LD_RU_right = 0;
      FPValue max_share_LU_RD_right = 0;
      FPValue max_share_LB_RF_right = 0;
      FPValue max_share_LF_RB_right = 0;
      FPValue max_share_DB_UF_right = 0;
      FPValue max_share_DF_UB_right = 0;

      FPValue max_share_LDB_RUF_right = 0;
      FPValue max_share_RDB_LUF_right = 0;
      FPValue max_share_RUB_LDF_right = 0;
      FPValue max_share_LUB_RDF_right = 0;

      FPValue max_share_time_right = 0;
      FPValue valueRight = 0;


      FPValue max_share_LR_all = 0;
      FPValue max_share_DU_all = 0;
      FPValue max_share_BF_all = 0;

      FPValue max_share_LD_RU_all = 0;
      FPValue max_share_LU_RD_all = 0;
      FPValue max_share_LB_RF_all = 0;
      FPValue max_share_LF_RB_all = 0;
      FPValue max_share_DB_UF_all = 0;
      FPValue max_share_DF_UB_all = 0;

      FPValue max_share_LDB_RUF_all = 0;
      FPValue max_share_RDB_LUF_all = 0;
      FPValue max_share_RUB_LDF_all = 0;
      FPValue max_share_LUB_RDF_all = 0;

      FPValue max_share_time_all = 0;
      FPValue valueAll = 0;

      if (entry.axis == 0)
      {
        {
          std::vector<grid_coord> tmp_spreadX (parallelGridCore->getNodeGridSizeX ());
          grid_coord tmp_sum_spreadX = 0;
          spreadGridPointsPerAxis (tmp_spreadX, tmp_sum_spreadX, sumSpeedEnabled,
                                   0, entry.coord + 1, OX,
                                   0, parallelGridCore->getNodeGridSizeY (), OY,
                                   0, parallelGridCore->getNodeGridSizeZ (), OZ,
                                   speed,
                                   true);

          findMaxTimes (max_share_LR_left, max_share_DU_left, max_share_BF_left,
                        max_share_LD_RU_left, max_share_LU_RD_left, max_share_LB_RF_left, max_share_LF_RB_left,
                        max_share_DB_UF_left, max_share_DF_UB_left,
                        max_share_LDB_RUF_left, max_share_RDB_LUF_left, max_share_RUB_LDF_left, max_share_LUB_RDF_left,
                        tmp_spreadX, spreadY, spreadZ, latency, bandwidth,
                        0, entry.coord + 1,
                        0, parallelGridCore->getNodeGridSizeY (),
                        0, parallelGridCore->getNodeGridSizeZ ());
          max_share_time_left = 2 * (max_share_LR_left + max_share_DU_left + max_share_BF_left
                                     + max_share_LD_RU_left + max_share_LU_RD_left + max_share_LB_RF_left + max_share_LF_RB_left
                                     + max_share_DB_UF_left + max_share_DF_UB_left
                                     + max_share_LDB_RUF_left + max_share_RDB_LUF_left + max_share_RUB_LDF_left + max_share_LUB_RDF_left);

          valueLeft = overallSize / perf_left + max_share_time_left;
        }

        {
          std::vector<grid_coord> tmp_spreadX (parallelGridCore->getNodeGridSizeX ());
          grid_coord tmp_sum_spreadX = 0;
          spreadGridPointsPerAxis (tmp_spreadX, tmp_sum_spreadX, sumSpeedEnabled,
                                   entry.coord + 1, parallelGridCore->getNodeGridSizeX (), OX,
                                   0, parallelGridCore->getNodeGridSizeY (), OY,
                                   0, parallelGridCore->getNodeGridSizeZ (), OZ,
                                   speed,
                                   true);

          findMaxTimes (max_share_LR_right, max_share_DU_right, max_share_BF_right,
                        max_share_LD_RU_right, max_share_LU_RD_right, max_share_LB_RF_right, max_share_LF_RB_right,
                        max_share_DB_UF_right, max_share_DF_UB_right,
                        max_share_LDB_RUF_right, max_share_RDB_LUF_right, max_share_RUB_LDF_right, max_share_LUB_RDF_right,
                        tmp_spreadX, spreadY, spreadZ, latency, bandwidth,
                        0, entry.coord + 1,
                        0, parallelGridCore->getNodeGridSizeY (),
                        0, parallelGridCore->getNodeGridSizeZ ());
          max_share_time_right = 2 * (max_share_LR_right + max_share_DU_right + max_share_BF_right
                                     + max_share_LD_RU_right + max_share_LU_RD_right + max_share_LB_RF_right + max_share_LF_RB_right
                                     + max_share_DB_UF_right + max_share_DF_UB_right
                                     + max_share_LDB_RUF_right + max_share_RDB_LUF_right + max_share_RUB_LDF_right + max_share_LUB_RDF_right);

          valueRight = overallSize / perf_right + max_share_time_right;
        }

        findMaxTimes (max_share_LR_all, max_share_DU_all, max_share_BF_all,
                      max_share_LD_RU_all, max_share_LU_RD_all, max_share_LB_RF_all, max_share_LF_RB_all,
                      max_share_DB_UF_all, max_share_DF_UB_all,
                      max_share_LDB_RUF_all, max_share_RDB_LUF_all, max_share_RUB_LDF_all, max_share_LUB_RDF_all,
                      spreadX, spreadY, spreadZ, latency, bandwidth,
                      0, parallelGridCore->getNodeGridSizeX (),
                      0, parallelGridCore->getNodeGridSizeY (),
                      0, parallelGridCore->getNodeGridSizeZ ());
        max_share_time_all = 2 * (max_share_LR_all + max_share_DU_all + max_share_BF_all
                                  + max_share_LD_RU_all + max_share_LU_RD_all + max_share_LB_RF_all + max_share_LF_RB_all
                                  + max_share_DB_UF_all + max_share_DF_UB_all
                                  + max_share_LDB_RUF_all + max_share_RDB_LUF_all + max_share_RUB_LDF_all + max_share_LUB_RDF_all);
        valueAll = overallSize / perf_all + max_share_time_all;

        bool disabled = false;

        FPValue new_perf_all;

        if (parallelGridCore->getProcessId () == 0)
        {
          printf ("# %d %d =========== %f %f %f =======\n", entry.axis, entry.coord, valueLeft, valueRight, valueAll);
        }

        // Check if right should be disabled
        if (valueLeft < valueAll && valueLeft < valueRight)
        {
          // disable right
          printf ("DISABLE RIGHT\n");
          disabled = true;
          for (int i = entry.coord + 1; i < parallelGridCore->getNodeGridSizeX (); ++i)
          for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
          for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
          {
            int pid = parallelGridCore->getNodeGrid (i, j, k);
            parallelGridCore->getNodeState ()[pid] = 0;
          }

          new_perf_all = perf_left;
        }
        else if (valueRight < valueAll && valueRight < valueLeft)
        {
          // disable left
          printf ("DISABLE LEFT\n");
          disabled = true;
          for (int i = 0; i < entry.coord + 1; ++i)
          for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
          for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
          {
            int pid = parallelGridCore->getNodeGrid (i, j, k);
            parallelGridCore->getNodeState ()[pid] = 0;
          }

          new_perf_all = perf_right;
        }
        else
        {
          printf ("DECIDED TO BREAK\n");
          flag = false;
          break;
        }

        if (disabled)
        {
          spreadGridPointsPerAxis (spreadX, sum_spreadX, new_perf_all,
                                   0, parallelGridCore->getNodeGridSizeX (), OX,
                                   0, parallelGridCore->getNodeGridSizeY (), OY,
                                   0, parallelGridCore->getNodeGridSizeZ (), OZ,
                                   speed,
                                   false);
          printf ("DECIDED TO BREAK\n");
          flag = false;
          break;
        }
      }
      else if (entry.axis == 1)
      {
        {
          std::vector<grid_coord> tmp_spreadY (parallelGridCore->getNodeGridSizeY ());
          grid_coord tmp_sum_spreadY = 0;
          spreadGridPointsPerAxis (tmp_spreadY, tmp_sum_spreadY, sumSpeedEnabled,
                                   0, entry.coord + 1, OY,
                                   0, parallelGridCore->getNodeGridSizeX (), OX,
                                   0, parallelGridCore->getNodeGridSizeZ (), OZ,
                                   speed,
                                   true);

          findMaxTimes (max_share_LR_left, max_share_DU_left, max_share_BF_left,
                        max_share_LD_RU_left, max_share_LU_RD_left, max_share_LB_RF_left, max_share_LF_RB_left,
                        max_share_DB_UF_left, max_share_DF_UB_left,
                        max_share_LDB_RUF_left, max_share_RDB_LUF_left, max_share_RUB_LDF_left, max_share_LUB_RDF_left,
                        spreadX, tmp_spreadY, spreadZ, latency, bandwidth,
                        0, parallelGridCore->getNodeGridSizeX (),
                        0, entry.coord + 1,
                        0, parallelGridCore->getNodeGridSizeZ ());
          max_share_time_left = 2 * (max_share_LR_left + max_share_DU_left + max_share_BF_left
                                     + max_share_LD_RU_left + max_share_LU_RD_left + max_share_LB_RF_left + max_share_LF_RB_left
                                     + max_share_DB_UF_left + max_share_DF_UB_left
                                     + max_share_LDB_RUF_left + max_share_RDB_LUF_left + max_share_RUB_LDF_left + max_share_LUB_RDF_left);

          valueLeft = overallSize / perf_left + max_share_time_left;
        }

        {
          std::vector<grid_coord> tmp_spreadY (parallelGridCore->getNodeGridSizeY ());
          grid_coord tmp_sum_spreadY = 0;
          spreadGridPointsPerAxis (tmp_spreadY, tmp_sum_spreadY, sumSpeedEnabled,
                                   entry.coord + 1, parallelGridCore->getNodeGridSizeY (), OY,
                                   0, parallelGridCore->getNodeGridSizeX (), OX,
                                   0, parallelGridCore->getNodeGridSizeZ (), OZ,
                                   speed,
                                   true);

          findMaxTimes (max_share_LR_right, max_share_DU_right, max_share_BF_right,
                        max_share_LD_RU_right, max_share_LU_RD_right, max_share_LB_RF_right, max_share_LF_RB_right,
                        max_share_DB_UF_right, max_share_DF_UB_right,
                        max_share_LDB_RUF_right, max_share_RDB_LUF_right, max_share_RUB_LDF_right, max_share_LUB_RDF_right,
                        spreadX, tmp_spreadY, spreadZ, latency, bandwidth,
                        0, parallelGridCore->getNodeGridSizeX (),
                        0, entry.coord + 1,
                        0, parallelGridCore->getNodeGridSizeZ ());
          max_share_time_right = 2 * (max_share_LR_right + max_share_DU_right + max_share_BF_right
                                     + max_share_LD_RU_right + max_share_LU_RD_right + max_share_LB_RF_right + max_share_LF_RB_right
                                     + max_share_DB_UF_right + max_share_DF_UB_right
                                     + max_share_LDB_RUF_right + max_share_RDB_LUF_right + max_share_RUB_LDF_right + max_share_LUB_RDF_right);

          valueRight = overallSize / perf_right + max_share_time_right;
        }

        findMaxTimes (max_share_LR_all, max_share_DU_all, max_share_BF_all,
                      max_share_LD_RU_all, max_share_LU_RD_all, max_share_LB_RF_all, max_share_LF_RB_all,
                      max_share_DB_UF_all, max_share_DF_UB_all,
                      max_share_LDB_RUF_all, max_share_RDB_LUF_all, max_share_RUB_LDF_all, max_share_LUB_RDF_all,
                      spreadX, spreadY, spreadZ, latency, bandwidth,
                      0, parallelGridCore->getNodeGridSizeX (),
                      0, parallelGridCore->getNodeGridSizeY (),
                      0, parallelGridCore->getNodeGridSizeZ ());
        max_share_time_all = 2 * (max_share_LR_all + max_share_DU_all + max_share_BF_all
                                  + max_share_LD_RU_all + max_share_LU_RD_all + max_share_LB_RF_all + max_share_LF_RB_all
                                  + max_share_DB_UF_all + max_share_DF_UB_all
                                  + max_share_LDB_RUF_all + max_share_RDB_LUF_all + max_share_RUB_LDF_all + max_share_LUB_RDF_all);
        valueAll = overallSize / perf_all + max_share_time_all;

        bool disabled = false;

        FPValue new_perf_all;

        if (parallelGridCore->getProcessId () == 0)
        {
          printf ("# %d %d =========== %f %f %f =======\n", entry.axis, entry.coord, valueLeft, valueRight, valueAll);
        }

        // Check if right should be disabled
        if (valueLeft < valueAll && valueLeft < valueRight)
        {
          // disable right
          printf ("DISABLE UP\n");
          disabled = true;
          for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
          for (int j = entry.coord + 1; j < parallelGridCore->getNodeGridSizeY (); ++j)
          for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
          {
            int pid = parallelGridCore->getNodeGrid (i, j, k);
            parallelGridCore->getNodeState ()[pid] = 0;
          }

          new_perf_all = perf_left;
        }
        else if (valueRight < valueAll && valueRight < valueLeft)
        {
          // disable left
          printf ("DISABLE DOWN\n");
          disabled = true;
          for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
          for (int j = 0; j < entry.coord + 1; ++j)
          for (int k = 0; k < parallelGridCore->getNodeGridSizeZ (); ++k)
          {
            int pid = parallelGridCore->getNodeGrid (i, j, k);
            parallelGridCore->getNodeState ()[pid] = 0;
          }

          new_perf_all = perf_right;
        }
        else
        {
          printf ("DECIDED TO BREAK\n");
          flag = false;
          break;
        }

        if (disabled)
        {
          spreadGridPointsPerAxis (spreadY, sum_spreadY, new_perf_all,
                                   0, parallelGridCore->getNodeGridSizeY (), OY,
                                   0, parallelGridCore->getNodeGridSizeX (), OX,
                                   0, parallelGridCore->getNodeGridSizeZ (), OZ,
                                   speed,
                                   false);
          printf ("DECIDED TO BREAK\n");
          flag = false;
          break;
        }
      }
      else if (entry.axis == 2)
      {
        {
          std::vector<grid_coord> tmp_spreadZ (parallelGridCore->getNodeGridSizeZ ());
          grid_coord tmp_sum_spreadZ = 0;
          spreadGridPointsPerAxis (tmp_spreadZ, tmp_sum_spreadZ, sumSpeedEnabled,
                                   0, entry.coord + 1, OZ,
                                   0, parallelGridCore->getNodeGridSizeX (), OX,
                                   0, parallelGridCore->getNodeGridSizeY (), OY,
                                   speed,
                                   true);

          findMaxTimes (max_share_LR_left, max_share_DU_left, max_share_BF_left,
                        max_share_LD_RU_left, max_share_LU_RD_left, max_share_LB_RF_left, max_share_LF_RB_left,
                        max_share_DB_UF_left, max_share_DF_UB_left,
                        max_share_LDB_RUF_left, max_share_RDB_LUF_left, max_share_RUB_LDF_left, max_share_LUB_RDF_left,
                        spreadX, spreadY, tmp_spreadZ, latency, bandwidth,
                        0, parallelGridCore->getNodeGridSizeX (),
                        0, parallelGridCore->getNodeGridSizeY (),
                        0, entry.coord + 1);
          max_share_time_left = 2 * (max_share_LR_left + max_share_DU_left + max_share_BF_left
                                     + max_share_LD_RU_left + max_share_LU_RD_left + max_share_LB_RF_left + max_share_LF_RB_left
                                     + max_share_DB_UF_left + max_share_DF_UB_left
                                     + max_share_LDB_RUF_left + max_share_RDB_LUF_left + max_share_RUB_LDF_left + max_share_LUB_RDF_left);

          valueLeft = overallSize / perf_left + max_share_time_left;
        }

        {
          std::vector<grid_coord> tmp_spreadZ (parallelGridCore->getNodeGridSizeZ ());
          grid_coord tmp_sum_spreadZ = 0;
          spreadGridPointsPerAxis (tmp_spreadZ, tmp_sum_spreadZ, sumSpeedEnabled,
                                   entry.coord + 1, parallelGridCore->getNodeGridSizeZ (), OZ,
                                   0, parallelGridCore->getNodeGridSizeX (), OX,
                                   0, parallelGridCore->getNodeGridSizeY (), OY,
                                   speed,
                                   true);

          findMaxTimes (max_share_LR_right, max_share_DU_right, max_share_BF_right,
                        max_share_LD_RU_right, max_share_LU_RD_right, max_share_LB_RF_right, max_share_LF_RB_right,
                        max_share_DB_UF_right, max_share_DF_UB_right,
                        max_share_LDB_RUF_right, max_share_RDB_LUF_right, max_share_RUB_LDF_right, max_share_LUB_RDF_right,
                        spreadX, spreadY, tmp_spreadZ, latency, bandwidth,
                        0, parallelGridCore->getNodeGridSizeX (),
                        0, parallelGridCore->getNodeGridSizeY (),
                        0, entry.coord + 1);
          max_share_time_right = 2 * (max_share_LR_right + max_share_DU_right + max_share_BF_right
                                     + max_share_LD_RU_right + max_share_LU_RD_right + max_share_LB_RF_right + max_share_LF_RB_right
                                     + max_share_DB_UF_right + max_share_DF_UB_right
                                     + max_share_LDB_RUF_right + max_share_RDB_LUF_right + max_share_RUB_LDF_right + max_share_LUB_RDF_right);

          valueRight = overallSize / perf_right + max_share_time_right;
        }

        findMaxTimes (max_share_LR_all, max_share_DU_all, max_share_BF_all,
                      max_share_LD_RU_all, max_share_LU_RD_all, max_share_LB_RF_all, max_share_LF_RB_all,
                      max_share_DB_UF_all, max_share_DF_UB_all,
                      max_share_LDB_RUF_all, max_share_RDB_LUF_all, max_share_RUB_LDF_all, max_share_LUB_RDF_all,
                      spreadX, spreadY, spreadZ, latency, bandwidth,
                      0, parallelGridCore->getNodeGridSizeX (),
                      0, parallelGridCore->getNodeGridSizeY (),
                      0, parallelGridCore->getNodeGridSizeZ ());
        max_share_time_all = 2 * (max_share_LR_all + max_share_DU_all + max_share_BF_all
                                  + max_share_LD_RU_all + max_share_LU_RD_all + max_share_LB_RF_all + max_share_LF_RB_all
                                  + max_share_DB_UF_all + max_share_DF_UB_all
                                  + max_share_LDB_RUF_all + max_share_RDB_LUF_all + max_share_RUB_LDF_all + max_share_LUB_RDF_all);
        valueAll = overallSize / perf_all + max_share_time_all;

        bool disabled = false;

        FPValue new_perf_all;

        if (parallelGridCore->getProcessId () == 0)
        {
          printf ("# %d %d =========== %f %f %f =======\n", entry.axis, entry.coord, valueLeft, valueRight, valueAll);
        }

        // Check if right should be disabled
        if (valueLeft < valueAll && valueLeft < valueRight)
        {
          // disable right
          printf ("DISABLE FRONT\n");
          disabled = true;
          for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
          for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
          for (int k = entry.coord + 1; k < parallelGridCore->getNodeGridSizeZ (); ++k)
          {
            int pid = parallelGridCore->getNodeGrid (i, j, k);
            parallelGridCore->getNodeState ()[pid] = 0;
          }

          new_perf_all = perf_left;
        }
        else if (valueRight < valueAll && valueRight < valueLeft)
        {
          // disable left
          printf ("DISABLE BACK\n");
          disabled = true;
          for (int i = 0; i < parallelGridCore->getNodeGridSizeX (); ++i)
          for (int j = 0; j < parallelGridCore->getNodeGridSizeY (); ++j)
          for (int k = 0; k < entry.coord + 1; ++k)
          {
            int pid = parallelGridCore->getNodeGrid (i, j, k);
            parallelGridCore->getNodeState ()[pid] = 0;
          }

          new_perf_all = perf_right;
        }
        else
        {
          printf ("DECIDED TO BREAK\n");
          flag = false;
          break;
        }

        if (disabled)
        {
          spreadGridPointsPerAxis (spreadZ, sum_spreadZ, new_perf_all,
                                   0, parallelGridCore->getNodeGridSizeZ (), OZ,
                                   0, parallelGridCore->getNodeGridSizeX (), OX,
                                   0, parallelGridCore->getNodeGridSizeY (), OY,
                                   speed,
                                   false);
          printf ("DECIDED TO BREAK\n");
          flag = false;
          break;
        }
      }
    }
  }

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
  || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  x = spreadX[parallelGridCore->getNodeGridX ()];
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
  || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  y = spreadY[parallelGridCore->getNodeGridY ()];
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) \
  || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  z = spreadZ[parallelGridCore->getNodeGridZ ()];
#endif

  printf ("#%d state=%d x=%d y=%d z=%d speed=%f (perfpoints=%f, perftimes=%f) totalX=%f totalY=%f totalZ=%f difft=%u sumSpeedEnabled=%f\n",
          parallelGridCore->getProcessId (),
          parallelGridCore->getNodeState ()[parallelGridCore->getProcessId ()],
          x, y, z,
          speed[parallelGridCore->getProcessId ()],
          parallelGridCore->perfPointsValues[parallelGridCore->getProcessId ()],
          parallelGridCore->perfTimeValues[parallelGridCore->getProcessId ()],
          (FPValue)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get1 (),
          (FPValue)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get2 (),
          (FPValue)YeeGridLayout<Type, ParallelGridCoordinateTemplate, layout_type>::size.get3 (),
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
  newSize.set2 (y);
  newSize.set3 (z);

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
    InitializeCounters ();
  }

  return true;
} /* ParallelYeeGridLayout::Rebalance */

#endif /* DYNAMIC_GRID */

#ifdef GRID_1D
typedef ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHy)), E_CENTERED> PYL_Dim1_ExHy;
typedef ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_ExHz)), E_CENTERED> PYL_Dim1_ExHz;
typedef ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHx)), E_CENTERED> PYL_Dim1_EyHx;
typedef ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EyHz)), E_CENTERED> PYL_Dim1_EyHz;
typedef ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHx)), E_CENTERED> PYL_Dim1_EzHx;
typedef ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim1_EzHy)), E_CENTERED> PYL_Dim1_EzHy;
#endif

#ifdef GRID_2D
typedef ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEx)), E_CENTERED> PYL_Dim2_TEx;
typedef ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEy)), E_CENTERED> PYL_Dim2_TEy;
typedef ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TEz)), E_CENTERED> PYL_Dim2_TEz;
typedef ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMx)), E_CENTERED> PYL_Dim2_TMx;
typedef ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMy)), E_CENTERED> PYL_Dim2_TMy;
typedef ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim2_TMz)), E_CENTERED> PYL_Dim2_TMz;
#endif

#ifdef GRID_3D
typedef ParallelYeeGridLayout<(static_cast<SchemeType_t> (SchemeType::Dim3)), E_CENTERED> PYL_Dim3;
#endif

#endif /* PARALLEL_GRID */

#endif /* PARALLEL_YEE_GRID_LAYOUT_H */
