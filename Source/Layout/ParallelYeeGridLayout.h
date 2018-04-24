#ifndef PARALLEL_YEE_GRID_LAYOUT_H
#define PARALLEL_YEE_GRID_LAYOUT_H

#include "YeeGridLayout.h"
#include "ParallelGridCore.h"
#include <algorithm>

#ifdef PARALLEL_GRID

enum Axis_t
{
  OX,
  OY,
  OZ
};

class NodeBorder_t
{
public:
  int pid_coord1;
  int pid_coord2;
  Axis_t axis;
  FPValue val;

  NodeBorder_t (int newcoord1, int newcoord2, Axis_t newaxis, FPValue newVal)
  : pid_coord1 (newcoord1), pid_coord2 (newcoord2), axis (newaxis), val (newVal) {}

  bool operator < (NodeBorder_t entry) const
  {
    return val < entry.val;
  }
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
  void spreadGridPointsPerAxis (std::vector<grid_coord> &, FPValue, int, int, Axis_t
#if defined (GRID_2D) || defined (GRID_3D)
                                , int, int, Axis_t
#endif
#if defined (GRID_3D)
                                , int, int, Axis_t
#endif
                                );

  void disableNodesAfterSpread (const std::vector<grid_coord> &
#if defined (GRID_2D) || defined (GRID_3D)
                                , const std::vector<grid_coord> &
#endif
#if defined (GRID_3D)
                                , const std::vector<grid_coord> &
#endif
                                );

  void spreadGridPoints (std::vector<grid_coord> &
#if defined (GRID_2D) || defined (GRID_3D)
                         , std::vector<grid_coord> &
#endif
#if defined (GRID_3D)
                         , std::vector<grid_coord> &
#endif
                         , FPValue);

  void checkDisablingConditions (std::vector<grid_coord> &
#if defined (GRID_2D) || defined (GRID_3D)
                                 , std::vector<grid_coord> &
#endif
#if defined (GRID_3D)
                                 , std::vector<grid_coord> &
#endif
                                 );

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  FPValue estimateTimeAcrossAxisX (int pidX1, int pidX2) const;
  void estimateBorderX (std::vector<NodeBorder_t> &, const std::vector<grid_coord> &) const;
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  FPValue estimateTimeAcrossAxisY (int pidY1, int pidY2) const;
  void estimateBorderY (std::vector<NodeBorder_t> &, const std::vector<grid_coord> &) const;
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  FPValue estimateTimeAcrossAxisZ (int pidZ1, int pidZ2) const;
  void estimateBorderZ (std::vector<NodeBorder_t> &, const std::vector<grid_coord> &) const;
#endif

  void estimatePerfLR (NodeBorder_t, FPValue &, FPValue &);

  void estimatePerfLRAxis (int, int, int, int,
#if defined (GRID_2D) || defined (GRID_3D)
                           int, int, int, int,
#endif
#if defined (GRID_3D)
                           int, int, int, int,
#endif
                           FPValue &, FPValue &);

  bool disableLR (NodeBorder_t,
                  std::vector<grid_coord> &,
#if defined (GRID_2D) || defined (GRID_3D)
                  std::vector<grid_coord> &,
#endif
#if defined (GRID_3D)
                  std::vector<grid_coord> &,
#endif
                  FPValue, FPValue);

  bool disableAxisLR (NodeBorder_t,
                      std::vector<grid_coord> &, int, int, int, int, int,
#if defined (GRID_2D) || defined (GRID_3D)
                      std::vector<grid_coord> &, int, int, int, int, int,
#endif
#if defined (GRID_3D)
                      std::vector<grid_coord> &, int, int, int, int, int,
#endif
                      FPValue, FPValue, FPValue, FPValue);

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
                     int axisStart1,
                     int axisSize1
#if defined (GRID_2D) || defined (GRID_3D)
                     , const std::vector<grid_coord> &spreadY,
                     int axisStart2,
                     int axisSize2
#endif
#if defined (GRID_3D)
                     , const std::vector<grid_coord> &spreadZ,
                     int axisStart3,
                     int axisSize3
#endif
                     );
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

#include "ParallelYeeGridLayout.template.h"

#endif /* PARALLEL_GRID */

#endif /* PARALLEL_YEE_GRID_LAYOUT_H */
