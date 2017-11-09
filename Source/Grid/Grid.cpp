#include "Grid.h"

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
  const grid_coord& px = position.getX ();
  const grid_coord& sx = sizeCoord.getX ();

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
  const grid_coord& px = position.getX ();

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
  const grid_coord& px = position.getX ();
  const grid_coord& sx = sizeCoord.getX ();

  const grid_coord& py = position.getY ();
  const grid_coord& sy = sizeCoord.getY ();

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
  const grid_coord& px = position.getX ();

  const grid_coord& py = position.getY ();
  const grid_coord& sy = sizeCoord.getY ();

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
  const grid_coord& px = position.getX ();
  const grid_coord& sx = sizeCoord.getX ();

  const grid_coord& py = position.getY ();
  const grid_coord& sy = sizeCoord.getY ();

  const grid_coord& pz = position.getZ ();
  const grid_coord& sz = sizeCoord.getZ ();

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
  const grid_coord& px = position.getX ();

  const grid_coord& py = position.getY ();
  const grid_coord& sy = sizeCoord.getY ();

  const grid_coord& pz = position.getZ ();
  const grid_coord& sz = sizeCoord.getZ ();

  return px * sy * sz + py * sz + pz;
} /* Grid<GridCoordinate3D>::calculateIndexFromPosition */

/**
 * Calculate position coordinate from one-dimensional index
 */
template <>
GridCoordinate1D
Grid<GridCoordinate1D>::calculatePositionFromIndex (grid_coord index) const /**< index in grid */
{
  return GridCoordinate1D (index);
} /* Grid<GridCoordinate1D>::calculatePositionFromIndex */

/**
 * Calculate position coordinate from one-dimensional index
 */
template <>
GridCoordinate2D
Grid<GridCoordinate2D>::calculatePositionFromIndex (grid_coord index) const /**< index in grid */
{
  const grid_coord& sx = size.getX ();
  const grid_coord& sy = size.getY ();

  grid_coord x = index / sy;
  index %= sy;
  grid_coord y = index;

  return GridCoordinate2D (x, y);
} /* Grid<GridCoordinate2D>::calculatePositionFromIndex */

/**
 * Calculate position coordinate from one-dimensional index
 */
template <>
GridCoordinate3D
Grid<GridCoordinate3D>::calculatePositionFromIndex (grid_coord index) const /**< index in grid */
{
  const grid_coord& sy = size.getY ();
  const grid_coord& sz = size.getZ ();

  grid_coord tmp = sy * sz;
  grid_coord x = index / tmp;
  index %= tmp;
  grid_coord y = index / sz;
  index %= sz;
  grid_coord z = index;

  return GridCoordinate3D (x, y, z);
} /* Grid<GridCoordinate3D>::calculatePositionFromIndex */
