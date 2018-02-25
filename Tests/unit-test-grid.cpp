/*
 * Unit test for basic operations with Grid
 */

#include <iostream>

#include "Assert.h"
#include "GridCoordinate3D.h"
#include "Grid.h"

#ifndef CXX11_ENABLED
#include "cstdlib"
#endif /* !CXX11_ENABLED */

template <class TCoord>
void testFunc (TCoord overallSize)
{
  Grid<TCoord> grid (overallSize, 0);

  /*
   * Copy constructor
   */
  Grid<TCoord> grid_1 (grid);

  grid.initialize ();

  /*
   * Copy constructor with initialize grid
   */
  Grid<TCoord> grid_2 (grid);

  /*
   * Operator =
   */
  grid = grid_1;

  /*
   * Operator = with initialized grid
   */
  grid = grid_2;

  /*
   * Dynamic creation of grid
   */
  Grid<TCoord> *grid_3 = new Grid<TCoord> (overallSize, 0);
  Grid<TCoord> *grid_4 = new Grid<TCoord> (overallSize, 0);

  *grid_3 = *grid_4;

  grid_4->initialize ();

  *grid_3 = *grid_4;

  delete grid_3;
  delete grid_4;
}

int main (int argc, char** argv)
{
  int gridSizeX = 32;
  int gridSizeY = 32;
  int gridSizeZ = 32;

  testFunc<GridCoordinate1D> (GridCoordinate1D (gridSizeX, CoordinateType::X));
  testFunc<GridCoordinate1D> (GridCoordinate1D (gridSizeY, CoordinateType::Y));
  testFunc<GridCoordinate1D> (GridCoordinate1D (gridSizeZ, CoordinateType::Z));

  testFunc<GridCoordinate2D> (GridCoordinate2D (gridSizeX, gridSizeY, CoordinateType::X, CoordinateType::Y));
  testFunc<GridCoordinate2D> (GridCoordinate2D (gridSizeX, gridSizeZ, CoordinateType::X, CoordinateType::Z));
  testFunc<GridCoordinate2D> (GridCoordinate2D (gridSizeY, gridSizeZ, CoordinateType::Y, CoordinateType::Z));

  testFunc<GridCoordinate3D> (GridCoordinate3D (gridSizeX, gridSizeY, gridSizeZ, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));

  return 0;
} /* main */
