#include "GridCoordinate3D.h"

GridCoordinate3D convertCoord (GridCoordinateFP3D coord)
{
  return GridCoordinate3D ((grid_iter) coord.getX (),
                           (grid_iter) coord.getY (),
                           (grid_iter) coord.getZ ());
}

GridCoordinateFP3D convertCoord (GridCoordinate3D coord)
{
  return GridCoordinateFP3D (coord.getX (), coord.getY (), coord.getZ ());
}
