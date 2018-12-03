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

#ifndef DEBUG_INFO
#error Test requires debug info
#endif /* !DEBUG_INFO */

template <class TCoord>
void testFunc (TCoord overallSize, int storedSteps, CoordinateType ct1, CoordinateType ct2, CoordinateType ct3)
{
  TCoord test_coord = TCoord::initAxesCoordinate (16, 16, 16, ct1, ct2, ct3);
  TCoord zero = TCoord::initAxesCoordinate (0, 0, 0, ct1, ct2, ct3);
  
  Grid<TCoord> grid (overallSize, 0, storedSteps);

  ASSERT (grid.getSize () == overallSize);
  ASSERT (grid.getTotalSize () == overallSize);
  ASSERT (grid.getTimeStep () == 0);
  
  ASSERT (grid.getTotalPosition (test_coord) == test_coord);
  ASSERT (grid.getRelativePosition (test_coord) == test_coord);
  
  ASSERT (grid.getComputationStart (zero) == zero);
  ASSERT (grid.getComputationEnd (zero) == overallSize);

  for (int i = 0; i < storedSteps; ++i)
  {
    grid.setFieldValue (FIELDVALUE (1502 * i, 189 * i), test_coord, i);
  }
  for (int i = 0; i < storedSteps; ++i)
  {
    ASSERT (*grid.getFieldValue (test_coord, i) == FieldValue (1502 * i, 189 * i));
    ASSERT (*grid.getFieldValueByAbsolutePos (test_coord, i) == FieldValue (1502 * i, 189 * i));
    ASSERT (*grid.getFieldValueOrNullByAbsolutePos (test_coord, i) == FieldValue (1502 * i, 189 * i));
  }
  
  grid.shiftInTime ();
  for (int j = 1; j < storedSteps; ++j)
  {
    int i = j - 1;
    ASSERT (*grid.getFieldValue (test_coord, j) == FieldValue (1502 * i, 189 * i));
    ASSERT (*grid.getFieldValueByAbsolutePos (test_coord, j) == FieldValue (1502 * i, 189 * i));
    ASSERT (*grid.getFieldValueOrNullByAbsolutePos (test_coord, j) == FieldValue (1502 * i, 189 * i));
  }
  
  grid.nextTimeStep (false);

#ifdef DEBUG_INFO
  ASSERT (grid.getTimeStep () == 1);
#endif /* DEBUG_INFO */
  
  grid.initialize (FieldValue (127, 1982));
  for (grid_coord i = 0; i < grid.getSize ().calculateTotalCoord (); ++i)
  {
    ASSERT (*grid.getFieldValue (i, 0) == FieldValue (127, 1982));
    
    if (grid.calculatePositionFromIndex (i) == test_coord)
    {
      for (int j = 1; j < storedSteps; ++j)
      {
        int i = j - 1;
        ASSERT (*grid.getFieldValue (test_coord, j) == FieldValue (1502 * i, 189 * i));
        ASSERT (*grid.getFieldValueByAbsolutePos (test_coord, j) == FieldValue (1502 * i, 189 * i));
        ASSERT (*grid.getFieldValueOrNullByAbsolutePos (test_coord, j) == FieldValue (1502 * i, 189 * i));
      }
    }
  }
  
  ASSERT (grid.getRaw (0) == grid.getFieldValue (zero, 0));
}

int main (int argc, char** argv)
{
  int gridSizeX = 32;
  int gridSizeY = 32;
  int gridSizeZ = 32;
  
  for (int i = 1; i < 10; ++i)
  {
#if defined (MODE_DIM1)
    testFunc<GridCoordinate1D> (GridCoordinate1D (gridSizeX, CoordinateType::X), i,
      CoordinateType::X, CoordinateType::NONE, CoordinateType::NONE);
    testFunc<GridCoordinate1D> (GridCoordinate1D (gridSizeY, CoordinateType::Y), i,
      CoordinateType::Y, CoordinateType::NONE, CoordinateType::NONE);
    testFunc<GridCoordinate1D> (GridCoordinate1D (gridSizeZ, CoordinateType::Z), i,
      CoordinateType::Z, CoordinateType::NONE, CoordinateType::NONE);
#endif /* MODE_DIM1 */

#if defined (MODE_DIM2)
    testFunc<GridCoordinate2D> (GridCoordinate2D (gridSizeX, gridSizeY, CoordinateType::X, CoordinateType::Y), i,
      CoordinateType::X, CoordinateType::Y, CoordinateType::NONE);
    testFunc<GridCoordinate2D> (GridCoordinate2D (gridSizeX, gridSizeZ, CoordinateType::X, CoordinateType::Z), i,
      CoordinateType::X, CoordinateType::Z, CoordinateType::NONE);
    testFunc<GridCoordinate2D> (GridCoordinate2D (gridSizeY, gridSizeZ, CoordinateType::Y, CoordinateType::Z), i,
      CoordinateType::Y, CoordinateType::Z, CoordinateType::NONE);
#endif /* MODE_DIM2 */

#if defined (MODE_DIM3)
    testFunc<GridCoordinate3D> (GridCoordinate3D (gridSizeX, gridSizeY, gridSizeZ,
                                                  CoordinateType::X, CoordinateType::Y, CoordinateType::Z), i,
                                CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
#endif /* MODE_DIM3 */
  }

  return 0;
} /* main */
