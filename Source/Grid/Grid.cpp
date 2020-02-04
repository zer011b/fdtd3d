#include "Grid.h"

template <>
GridCoordinate1D
Grid<GridCoordinate1D>::getChunkStartPosition () const
{
  return getSize ().getZero ();
}

template <>
GridCoordinate2D
Grid<GridCoordinate2D>::getChunkStartPosition () const
{
  return getSize ().getZero ();
}

template <>
GridCoordinate3D
Grid<GridCoordinate3D>::getChunkStartPosition () const
{
  return getSize ().getZero ();
}

/**
 * Get first coordinate from which to perform computations at current step
 *
 * @return first coordinate from which to perform computations at current step
 */
template <>
GridCoordinate1D
Grid<GridCoordinate1D>::getComputationStart (const GridCoordinate1D & diffPosStart) const /**< offset from the left border */
{
  return getSize ().getZero () + diffPosStart;
} /* Grid<GridCoordinate1D>::getComputationStart */

#if defined (MODE_DIM2) || defined (MODE_DIM3)

/**
 * Get first coordinate from which to perform computations at current step
 *
 * @return first coordinate from which to perform computations at current step
 */
template <>
GridCoordinate2D
Grid<GridCoordinate2D>::getComputationStart (const GridCoordinate2D & diffPosStart) const /**< offset from the left border */
{
  return getSize ().getZero () + diffPosStart;
} /* Grid<GridCoordinate2D>::getComputationStart */

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

/**
 * Get first coordinate from which to perform computations at current step
 *
 * @return first coordinate from which to perform computations at current step
 */
template <>
GridCoordinate3D
Grid<GridCoordinate3D>::getComputationStart (const GridCoordinate3D & diffPosStart) const /**< offset from the left border */
{
  return getSize ().getZero () + diffPosStart;
} /* Grid<GridCoordinate3D>::getComputationStart */

#endif /* MODE_DIM3 */

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

#if defined (MODE_DIM2) || defined (MODE_DIM3)

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

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

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

#endif /* MODE_DIM3 */
