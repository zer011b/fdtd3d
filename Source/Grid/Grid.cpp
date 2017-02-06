#include "Grid.h"

/**
 * ======== Static methods ========
 */
template<>
bool
Grid<GridCoordinate1D>::isLegitIndex (const GridCoordinate1D& position,
                                      const GridCoordinate1D& sizeCoord)
{
  const grid_coord& px = position.getX ();
  const grid_coord& sx = sizeCoord.getX ();

  if (px < 0 || px >= sx)
  {
    return false;
  }

  return true;
}

template<>
grid_iter
Grid<GridCoordinate1D>::calculateIndexFromPosition (const GridCoordinate1D& position,
                                                    const GridCoordinate1D& sizeCoord)
{
  const grid_coord& px = position.getX ();

  return px;
}

template<>
bool
Grid<GridCoordinate2D>::isLegitIndex (const GridCoordinate2D& position,
                                      const GridCoordinate2D& sizeCoord)
{
  const grid_coord& px = position.getX ();
  const grid_coord& sx = sizeCoord.getX ();

  const grid_coord& py = position.getY ();
  const grid_coord& sy = sizeCoord.getY ();

  if (px < 0 || px >= sx)
  {
    return false;
  }
  else if (py < 0 || py >= sy)
  {
    return false;
  }

  return true;
}

template<>
grid_iter
Grid<GridCoordinate2D>::calculateIndexFromPosition (const GridCoordinate2D& position,
                                                    const GridCoordinate2D& sizeCoord)
{
  const grid_coord& px = position.getX ();

  const grid_coord& py = position.getY ();
  const grid_coord& sy = sizeCoord.getY ();

  return px * sy + py;
}

template<>
bool
Grid<GridCoordinate3D>::isLegitIndex (const GridCoordinate3D& position,
                                      const GridCoordinate3D& sizeCoord)
{
  const grid_coord& px = position.getX ();
  const grid_coord& sx = sizeCoord.getX ();

  const grid_coord& py = position.getY ();
  const grid_coord& sy = sizeCoord.getY ();

  const grid_coord& pz = position.getZ ();
  const grid_coord& sz = sizeCoord.getZ ();

  if (px < 0 || px >= sx)
  {
    return false;
  }
  else if (py < 0 || py >= sy)
  {
    return false;
  }
  else if (pz < 0 || pz >= sz)
  {
    return false;
  }

  return true;
}

template<>
grid_iter
Grid<GridCoordinate3D>::calculateIndexFromPosition (const GridCoordinate3D& position,
                                                    const GridCoordinate3D& sizeCoord)
{
  const grid_coord& px = position.getX ();

  const grid_coord& py = position.getY ();
  const grid_coord& sy = sizeCoord.getY ();

  const grid_coord& pz = position.getZ ();
  const grid_coord& sz = sizeCoord.getZ ();

  return px * sy * sz + py * sz + pz;
}

/**
 * ======== Public methods ========
 */

template <>
GridCoordinate1D
Grid<GridCoordinate1D>::calculatePositionFromIndex (grid_iter index) const
{
  return GridCoordinate1D (index);
}

template <>
GridCoordinate2D
Grid<GridCoordinate2D>::calculatePositionFromIndex (grid_iter index) const
{
  const grid_coord& sx = size.getX ();
  const grid_coord& sy = size.getY ();

  grid_coord x = index / sy;
  index %= sy;
  grid_coord y = index;
  return GridCoordinate2D (x, y);
}

template <>
GridCoordinate3D
Grid<GridCoordinate3D>::calculatePositionFromIndex (grid_iter index) const
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
}
