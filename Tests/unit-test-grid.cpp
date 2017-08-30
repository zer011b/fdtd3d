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

int main (int argc, char** argv)
{
  int gridSizeX = 32;
  int gridSizeY = 32;
  int gridSizeZ = 32;

  GridCoordinate1D overallSize1D (gridSizeX);
  GridCoordinate2D overallSize2D (gridSizeX, gridSizeY);
  GridCoordinate3D overallSize3D (gridSizeX, gridSizeY, gridSizeZ);

  {
    Grid<GridCoordinate1D> grid1D (overallSize1D, 0);
    Grid<GridCoordinate2D> grid2D (overallSize2D, 0);
    Grid<GridCoordinate3D> grid3D (overallSize3D, 0);

    /*
     * Copy constructor
     */
    Grid<GridCoordinate1D> grid1D_1 (grid1D);
    Grid<GridCoordinate2D> grid2D_1 (grid2D);
    Grid<GridCoordinate3D> grid3D_1 (grid3D);

    grid1D.initialize ();
    grid2D.initialize ();
    grid3D.initialize ();

    /*
     * Copy constructor with initialize grid
     */
    Grid<GridCoordinate1D> grid1D_2 (grid1D);
    Grid<GridCoordinate2D> grid2D_2 (grid2D);
    Grid<GridCoordinate3D> grid3D_2 (grid3D);

    /*
     * Operator =
     */
    grid1D = grid1D_1;
    grid2D = grid2D_1;
    grid3D = grid3D_1;

    /*
     * Operator = with initialized grid
     */
    grid1D = grid1D_2;
    grid2D = grid2D_2;
    grid3D = grid3D_2;

    /*
     * Dynamic creation of grid
     */
    Grid<GridCoordinate3D> *grid3D_3 = new Grid<GridCoordinate3D> (overallSize3D, 0);
    Grid<GridCoordinate2D> *grid2D_3 = new Grid<GridCoordinate2D> (overallSize2D, 0);
    Grid<GridCoordinate1D> *grid1D_3 = new Grid<GridCoordinate1D> (overallSize1D, 0);

    Grid<GridCoordinate3D> *grid3D_4 = new Grid<GridCoordinate3D> (overallSize3D, 0);
    Grid<GridCoordinate2D> *grid2D_4 = new Grid<GridCoordinate2D> (overallSize2D, 0);
    Grid<GridCoordinate1D> *grid1D_4 = new Grid<GridCoordinate1D> (overallSize1D, 0);

    *grid3D_3 = *grid3D_4;
    *grid2D_3 = *grid2D_4;
    *grid1D_3 = *grid1D_4;

    grid3D_4->initialize ();
    grid2D_4->initialize ();
    grid1D_4->initialize ();

    *grid3D_3 = *grid3D_4;
    *grid2D_3 = *grid2D_4;
    *grid1D_3 = *grid1D_4;

    delete grid1D_3;
    delete grid2D_3;
    delete grid3D_3;

    delete grid1D_4;
    delete grid2D_4;
    delete grid3D_4;
  }

  return 0;
} /* main */
