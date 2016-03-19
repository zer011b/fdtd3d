#include "GridCoordinate.h"

GridCoordinate::GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& sx
#if defined (GRID_2D) || defined (GRID_3D)
  , const grid_coord& sy
#if defined (GRID_3D)
  , const grid_coord& sz
#endif  /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D*/
  ) :
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  x (sx)
#if defined (GRID_2D) || defined (GRID_3D)
  , y (sy)
#if defined (GRID_3D)
  , z (sz)
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D*/
{
}

#if defined (GRID_2D) || defined (GRID_3D)
GridCoordinate::GridCoordinate (const grid_coord& s) :
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  x (s)
#if defined (GRID_2D) || defined (GRID_3D)
  , y (s)
#if defined (GRID_3D)
  , z (s)
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D*/
{
}
#endif

GridCoordinate::GridCoordinate () :
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  x (0)
#if defined (GRID_2D) || defined (GRID_3D)
  , y (0)
#if defined (GRID_3D)
  , z (0)
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D*/
{
}

GridCoordinate::GridCoordinate (const GridCoordinate& pos) :
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  x (pos.getX ())
#if defined (GRID_2D) || defined (GRID_3D)
  , y (pos.getY ())
#if defined (GRID_3D)
  , z (pos.getZ ())
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D*/
#endif /* GRID_1D || GRID_2D || GRID_3D*/
{
}

GridCoordinate::~GridCoordinate ()
{
}

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
const grid_coord&
GridCoordinate::getX () const
{
  return x;
}
void
GridCoordinate::setX (const grid_coord& new_x)
{
  x = new_x;
}
#if defined (GRID_2D) || defined (GRID_3D)
const grid_coord&
GridCoordinate::getY () const
{
  return y;
}
void
GridCoordinate::setY (const grid_coord& new_y)
{
  y = new_y;
}
#if defined (GRID_3D)
const grid_coord&
GridCoordinate::getZ () const
{
  return z;
}
void
GridCoordinate::setZ (const grid_coord& new_z)
{
  z = new_z;
}
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D*/

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
grid_iter
GridCoordinate::calculateTotalCoord () const
{
#if defined (GRID_1D)
  return x;
#else /* GRID_1D */
#if defined (GRID_2D)
  return x * y;
#else /* GRID_2D */
#if defined (GRID_3D)
  return x * y * z;
#endif /* GRID_3D */
#endif /* !GRID_2D */
#endif /* !GRID_1D */
}
#endif /* GRID_1D || GRID_2D || GRID_3D*/
