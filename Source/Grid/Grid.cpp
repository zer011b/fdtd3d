#include "Grid.h"

/**
 * Get first coordinate from which to perform computations at current step
 *
 * @return first coordinate from which to perform computations at current step
 */
template <>
GridCoordinate1D
Grid<GridCoordinate1D>::getComputationStart (GridCoordinate1D diffPosStart) const
{
  CoordinateType ct1 = CoordinateType::NONE;
#ifdef DEBUG_INFO
  ct1 = getSize ().getType1 ();
#endif /* !DEBUG_INFO */
  return GridCoordinate1D (0, ct1) + diffPosStart;
} /* Grid<GridCoordinate1D>::getComputationStart */

/**
 * Get first coordinate from which to perform computations at current step
 *
 * @return first coordinate from which to perform computations at current step
 */
template <>
GridCoordinate2D
Grid<GridCoordinate2D>::getComputationStart (GridCoordinate2D diffPosStart) const
{
  CoordinateType ct1 = CoordinateType::NONE;
  CoordinateType ct2 = CoordinateType::NONE;
#ifdef DEBUG_INFO
  ct1 = getSize ().getType1 ();
  ct2 = getSize ().getType2 ();
#endif /* !DEBUG_INFO */
  return GridCoordinate2D (0, 0, ct1, ct2) + diffPosStart;
} /* Grid<GridCoordinate2D>::getComputationStart */

/**
 * Get first coordinate from which to perform computations at current step
 *
 * @return first coordinate from which to perform computations at current step
 */
template <>
GridCoordinate3D
Grid<GridCoordinate3D>::getComputationStart (GridCoordinate3D diffPosStart) const
{
  CoordinateType ct1 = CoordinateType::NONE;
  CoordinateType ct2 = CoordinateType::NONE;
  CoordinateType ct3 = CoordinateType::NONE;
#ifdef DEBUG_INFO
  ct1 = getSize ().getType1 ();
  ct2 = getSize ().getType2 ();
  ct3 = getSize ().getType3 ();
#endif /* !DEBUG_INFO */
  return GridCoordinate3D (0, 0, 0, ct1, ct2, ct3) + diffPosStart;
} /* Grid<GridCoordinate3D>::getComputationStart */

/**
 * Check whether position is appropriate to get/set value from
 *
 * @return flag whether position is appropriate to get/set value from
 */
template<>
bool
Grid<GridCoordinate1D>::isLegitIndex (const GridCoordinate1D &position, /**< coordinate in grid */
                                      const GridCoordinate1D &sizeCoord) /**< size of grid */
{
  const grid_coord& px = position.get1 ();
  const grid_coord& sx = sizeCoord.get1 ();

  if (px >= sx)
  {
    return false;
  }

  return true;
} /* Grid<GridCoordinate1D>::isLegitIndex */

/**
 * Calculate one-dimensional coordinate from one-dimensional position
 *
 * @return one-dimensional coordinate from one-dimensional position
 */
template<>
grid_coord
Grid<GridCoordinate1D>::calculateIndexFromPosition (const GridCoordinate1D &position, /**< coordinate in grid */
                                                    const GridCoordinate1D &sizeCoord) /**< size of grid */
{
  const grid_coord& px = position.get1 ();

  return px;
} /* Grid<GridCoordinate1D>::calculateIndexFromPosition */

/**
 * Check whether position is appropriate to get/set value from
 *
 * @return flag whether position is appropriate to get/set value from
 */
template<>
bool
Grid<GridCoordinate2D>::isLegitIndex (const GridCoordinate2D &position, /**< coordinate in grid */
                                      const GridCoordinate2D &sizeCoord) /**< size of grid */
{
  const grid_coord& px = position.get1 ();
  const grid_coord& sx = sizeCoord.get1 ();

  const grid_coord& py = position.get2 ();
  const grid_coord& sy = sizeCoord.get2 ();

  if (px >= sx)
  {
    return false;
  }
  else if (py >= sy)
  {
    return false;
  }

  return true;
} /* Grid<GridCoordinate2D>::isLegitIndex */

/**
 * Calculate one-dimensional coordinate from two-dimensional position
 *
 * @return one-dimensional coordinate from two-dimensional position
 */
template<>
grid_coord
Grid<GridCoordinate2D>::calculateIndexFromPosition (const GridCoordinate2D &position, /**< coordinate in grid */
                                                    const GridCoordinate2D &sizeCoord) /**< size of grid */
{
  const grid_coord& px = position.get1 ();

  const grid_coord& py = position.get2 ();
  const grid_coord& sy = sizeCoord.get2 ();

  return px * sy + py;
} /* Grid<GridCoordinate2D>::calculateIndexFromPosition */

/**
 * Check whether position is appropriate to get/set value from
 *
 * @return flag whether position is appropriate to get/set value from
 */
template<>
bool
Grid<GridCoordinate3D>::isLegitIndex (const GridCoordinate3D &position, /**< coordinate in grid */
                                      const GridCoordinate3D &sizeCoord) /**< size of grid */
{
  const grid_coord& px = position.get1 ();
  const grid_coord& sx = sizeCoord.get1 ();

  const grid_coord& py = position.get2 ();
  const grid_coord& sy = sizeCoord.get2 ();

  const grid_coord& pz = position.get3 ();
  const grid_coord& sz = sizeCoord.get3 ();

  if (px >= sx)
  {
    return false;
  }
  else if (py >= sy)
  {
    return false;
  }
  else if (pz >= sz)
  {
    return false;
  }

  return true;
} /* Grid<GridCoordinate3D>::isLegitIndex */

/**
 * Calculate one-dimensional coordinate from two-dimensional position
 *
 * @return one-dimensional coordinate from two-dimensional position
 */
template<>
grid_coord
Grid<GridCoordinate3D>::calculateIndexFromPosition (const GridCoordinate3D& position, /**< coordinate in grid */
                                                    const GridCoordinate3D& sizeCoord) /**< size of grid */
{
  const grid_coord& px = position.get1 ();

  const grid_coord& py = position.get2 ();
  const grid_coord& sy = sizeCoord.get2 ();

  const grid_coord& pz = position.get3 ();
  const grid_coord& sz = sizeCoord.get3 ();

  return px * sy * sz + py * sz + pz;
} /* Grid<GridCoordinate3D>::calculateIndexFromPosition */

/**
 * Calculate position coordinate from one-dimensional index
 */
template <>
GridCoordinate1D
Grid<GridCoordinate1D>::calculatePositionFromIndex (grid_coord index) const /**< index in grid */
{
  CoordinateType ct1 = CoordinateType::NONE;
#ifdef DEBUG_INFO
  ct1 = getSize ().getType1 ();
#endif /* !DEBUG_INFO */
  return GridCoordinate1D (index, ct1);
} /* Grid<GridCoordinate1D>::calculatePositionFromIndex */

/**
 * Calculate position coordinate from one-dimensional index
 */
template <>
GridCoordinate2D
Grid<GridCoordinate2D>::calculatePositionFromIndex (grid_coord index) const /**< index in grid */
{
  const grid_coord& sx = size.get1 ();
  const grid_coord& sy = size.get2 ();

  grid_coord x = index / sy;
  index %= sy;
  grid_coord y = index;

  CoordinateType ct1 = CoordinateType::NONE;
  CoordinateType ct2 = CoordinateType::NONE;
#ifdef DEBUG_INFO
  ct1 = getSize ().getType1 ();
  ct2 = getSize ().getType2 ();
#endif /* !DEBUG_INFO */

  return GridCoordinate2D (x, y, ct1, ct2);
} /* Grid<GridCoordinate2D>::calculatePositionFromIndex */

/**
 * Calculate position coordinate from one-dimensional index
 */
template <>
GridCoordinate3D
Grid<GridCoordinate3D>::calculatePositionFromIndex (grid_coord index) const /**< index in grid */
{
  const grid_coord& sy = size.get2 ();
  const grid_coord& sz = size.get3 ();

  grid_coord tmp = sy * sz;
  grid_coord x = index / tmp;
  index %= tmp;
  grid_coord y = index / sz;
  index %= sz;
  grid_coord z = index;

  CoordinateType ct1 = CoordinateType::NONE;
  CoordinateType ct2 = CoordinateType::NONE;
  CoordinateType ct3 = CoordinateType::NONE;
#ifdef DEBUG_INFO
  ct1 = getSize ().getType1 ();
  ct2 = getSize ().getType2 ();
  ct3 = getSize ().getType3 ();
#endif /* !DEBUG_INFO */

  return GridCoordinate3D (x, y, z, ct1, ct2, ct3);
} /* Grid<GridCoordinate3D>::calculatePositionFromIndex */
