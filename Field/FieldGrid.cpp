#include <iostream>

#include "FieldGrid.h"

// ================================ GridSize ================================
GridCoordinate::GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_coord sx
#if defined (GRID_2D) || defined (GRID_3D)
  , grid_coord sy
#if defined (GRID_3D)
  , grid_coord sz
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
grid_coord
GridCoordinate::getX ()
{
  return x;
}
#if defined (GRID_2D) || defined (GRID_3D)
grid_coord
GridCoordinate::getY ()
{
  return y;
}
#if defined (GRID_3D)
grid_coord
GridCoordinate::getZ ()
{
  return z;
}
#endif
#endif
#endif

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
grid_coord
GridCoordinate::getTotalCoord ()
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
Grid::Grid(GridCoordinate& s) :
  size (s)
{
  gridValues.resize (size.getTotalCoord ());
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

void
Grid::setFieldPointValue (FieldPointValue& value, GridCoordinate& position)
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_coord px = position.getX ();
  grid_coord sx = size.getX ();
#if defined (GRID_2D) || defined (GRID_3D)
  grid_coord py = position.getY ();
  grid_coord sy = position.getY ();
#if defined (GRID_3D)
  grid_coord pz = position.getZ ();
  grid_coord sz = position.getZ ();
#endif
#endif
#endif

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_coord coord;

#if defined (GRID_1D)
  if (px < 0 || px >= sx)
  {
    return;
  }
  else
  {
    coord = px;
  }
#else
#if defined (GRID_2D)
  if (px < 0 || px >= sx)
  {
    return;
  }
  else if (py < 0 || py >= sy)
  {
    return;
  }
  else
  {
    coord = px * sy + py;
  }
#else
#if defined (GRID_3D)
  if (px < 0 || px >= sx)
  {
    return;
  }
  else if (py < 0 || py >= sy)
  {
    return;
  }
  else if (pz < 0 || pz >= sz)
  {
    return;
  }
  else
  {
    coord = px * sy * sz + py * sz + pz;
  }
#endif
#endif
#endif

  gridValues[coord] = value;
#endif
}