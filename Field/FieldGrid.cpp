#include <iostream>

#include "FieldGrid.h"

// ================================ GridSize ================================
GridCoordinate::GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& sx
#if defined (GRID_2D) || defined (GRID_3D)
  , const grid_coord& sy
#if defined (GRID_3D)
  , const grid_coord& sz
#endif
#endif
#endif
  ) :
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  x (sx)
#if defined (GRID_2D) || defined (GRID_3D)
  , y (sy)
#if defined (GRID_3D)
  , z (sz)
#endif
#endif
#endif 
{
}

GridCoordinate::~GridCoordinate ()
{
}

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
grid_coord&
GridCoordinate::getX ()
{
  return x;
}
#if defined (GRID_2D) || defined (GRID_3D)
grid_coord&
GridCoordinate::getY ()
{
  return y;
}
#if defined (GRID_3D)
grid_coord&
GridCoordinate::getZ ()
{
  return z;
}
#endif
#endif
#endif

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
grid_coord
GridCoordinate::calculateTotalCoord ()
{
#if defined (GRID_1D)
  return x;
#else
#if defined (GRID_2D)
  return x * y;
#else
#if defined (GRID_3D)
  return x * y * z;
#endif
#endif
#endif
}
#endif

// ================================ Grid ================================
Grid::Grid(const GridCoordinate& s) :
  size (s)
{
  gridValues.resize (size.calculateTotalCoord ());
  std::cout << "New grid with size: " << gridValues.size () << std::endl;
}

Grid::~Grid ()
{
}

GridCoordinate& Grid::getSize ()
{
  return size;
}

VectorFieldPointValues& Grid::getValues ()
{
  return gridValues;
}

bool
Grid::isLegitIndex (GridCoordinate& position)
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_coord& px = position.getX ();
  grid_coord& sx = size.getX ();
#if defined (GRID_2D) || defined (GRID_3D)
  grid_coord& py = position.getY ();
  grid_coord& sy = size.getY ();
#if defined (GRID_3D)
  grid_coord& pz = position.getZ ();
  grid_coord& sz = size.getZ ();
#endif
#endif
#endif

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  if (px < 0 || px >= sx)
  {
    return false;
  }
#if defined (GRID_2D) || defined (GRID_3D)
  else if (py < 0 || py >= sy)
  {
    return false;
  }
#if defined (GRID_3D)
  else if (pz < 0 || pz >= sz)
  {
    return false;
  }
#endif
#endif
#endif

  return true;
}

grid_coord
Grid::calculateIndexFromPosition (GridCoordinate& position)
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_coord& px = position.getX ();
  grid_coord& sx = size.getX ();
#if defined (GRID_2D) || defined (GRID_3D)
  grid_coord& py = position.getY ();
  grid_coord& sy = size.getY ();
#if defined (GRID_3D)
  grid_coord& pz = position.getZ ();
  grid_coord& sz = size.getZ ();
#endif
#endif
#endif

  grid_coord coord = 0;

#if defined (GRID_1D)
  coord = px;
#else
#if defined (GRID_2D)
  coord = px * sy + py;
#else
#if defined (GRID_3D)
  coord = px * sy * sz + py * sz + pz;
#endif
#endif
#endif

  return coord;
}

GridCoordinate
Grid::calculatePositionFromIndex (grid_coord index)
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_coord& sx = size.getX ();
#if defined (GRID_2D) || defined (GRID_3D)
  grid_coord& sy = size.getY ();
#if defined (GRID_3D)
  grid_coord& sz = size.getZ ();
#endif
#endif
#endif

#if defined (GRID_1D)
  grid_coord x = index;
  return GridCoordinate (x);
#else
#if defined (GRID_2D)
  grid_coord x = index / sy;
  index %= sy;
  grid_coord y = index;
  return GridCoordinate (x, y);
#else
#if defined (GRID_3D)
  grid_coord tmp = sy * sz;
  grid_coord x = index / tmp;
  index %= tmp;
  grid_coord y = index / sz;
  index %= sz;
  grid_coord z = index;
  return GridCoordinate (x, y, z);
#endif
#endif
#endif
}

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
void
Grid::setFieldPointValue (FieldPointValue& value, GridCoordinate& position)
{
  if (isLegitIndex (position))
  {
    grid_coord coord = calculateIndexFromPosition (position);
    gridValues[coord] = value;
  }
}

FieldPointValue&
Grid::getFieldPointValue (GridCoordinate& position)
{
  if (isLegitIndex (position))
  {
    grid_coord coord = calculateIndexFromPosition (position);
    return gridValues[coord];
  }
}
#endif