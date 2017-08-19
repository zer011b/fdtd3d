#include "GridCoordinate3D.h"
#include "Assert.h"

GridCoordinate1D CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinateFP1D coord)
{
  ASSERT (((grid_iter) coord.getX ()) == coord.getX ());

  return GridCoordinate1D ((grid_iter) coord.getX ());
}

GridCoordinateFP1D CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate1D coord)
{
  return GridCoordinateFP1D (coord.getX ());
}

GridCoordinate2D CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinateFP2D coord)
{
  ASSERT (((grid_iter) coord.getX ()) == coord.getX ());
  ASSERT (((grid_iter) coord.getY ()) == coord.getY ());

  return GridCoordinate2D ((grid_iter) coord.getX (),
                           (grid_iter) coord.getY ());
}

GridCoordinateFP2D CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate2D coord)
{
  return GridCoordinateFP2D (coord.getX (), coord.getY ());
}

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
