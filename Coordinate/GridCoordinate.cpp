#include "GridCoordinate3D.h"
#include "Assert.h"

GridCoordinate3D CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinateFP3D coord)
{
  ASSERT (((grid_iter) coord.getX ()) == coord.getX ());
  ASSERT (((grid_iter) coord.getY ()) == coord.getY ());
  ASSERT (((grid_iter) coord.getZ ()) == coord.getZ ());

  return GridCoordinate3D ((grid_iter) coord.getX (),
                           (grid_iter) coord.getY (),
                           (grid_iter) coord.getZ ());
}

GridCoordinateFP3D CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate3D coord)
{
  return GridCoordinateFP3D (coord.getX (), coord.getY (), coord.getZ ());
}

GridCoordinate2D CUDA_DEVICE CUDA_HOST shrinkCoord (GridCoordinate3D coord)
{
  return GridCoordinate2D (coord.getX (), coord.getY ());
}

GridCoordinateFP2D CUDA_DEVICE CUDA_HOST shrinkCoord (GridCoordinateFP3D coord)
{
  return GridCoordinateFP2D (coord.getX (), coord.getY ());
}
