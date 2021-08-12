/*
 * Copyright (C) 2017 Gleb Balykov
 *
 * This file is part of fdtd3d.
 *
 * fdtd3d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * fdtd3d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with fdtd3d; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

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

  Grid<TCoord> grid (overallSize, storedSteps);

  ASSERT (grid.getSize () == overallSize);
  ASSERT (grid.getTotalSize () == overallSize);

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
    ASSERT (*grid.getFieldValue (test_coord, i) == FIELDVALUE (1502 * i, 189 * i));
    ASSERT (*grid.getFieldValueByAbsolutePos (test_coord, i) == FIELDVALUE (1502 * i, 189 * i));
    ASSERT (*grid.getFieldValueOrNullByAbsolutePos (test_coord, i) == FIELDVALUE (1502 * i, 189 * i));
  }

  grid.shiftInTime ();
  for (int j = 1; j < storedSteps; ++j)
  {
    int i = j - 1;
    ASSERT (*grid.getFieldValue (test_coord, j) == FIELDVALUE (1502 * i, 189 * i));
    ASSERT (*grid.getFieldValueByAbsolutePos (test_coord, j) == FIELDVALUE (1502 * i, 189 * i));
    ASSERT (*grid.getFieldValueOrNullByAbsolutePos (test_coord, j) == FIELDVALUE (1502 * i, 189 * i));
  }

  grid.initialize (FIELDVALUE (127, 1982));

  typename VectorFieldValues<TCoord>::Iterator iter = grid.begin ();
  typename VectorFieldValues<TCoord>::Iterator iter_end = grid.end ();
  for (; iter != iter_end; ++iter)
  {
    TCoord pos = iter.getPos ();
    ASSERT (*grid.getFieldValue (pos, 0) == FIELDVALUE (127, 1982));

    if (pos == test_coord)
    {
      for (int j = 1; j < storedSteps; ++j)
      {
        int i = j - 1;
        ASSERT (*grid.getFieldValue (test_coord, j) == FIELDVALUE (1502 * i, 189 * i));
        ASSERT (*grid.getFieldValueByAbsolutePos (test_coord, j) == FIELDVALUE (1502 * i, 189 * i));
        ASSERT (*grid.getFieldValueOrNullByAbsolutePos (test_coord, j) == FIELDVALUE (1502 * i, 189 * i));
      }
    }
  }

  ASSERT (grid.getRaw (0)->get (zero) == grid.getFieldValue (zero, 0));
}

int main (int argc, char** argv)
{
  int gridSizeX = 32;
  int gridSizeY = 32;
  int gridSizeZ = 32;

  {
    VectorFieldValues<GridCoordinate1D>::Iterator iter (GridCoordinate1D (2, CoordinateType::X),
                                                        GridCoordinate1D (2, CoordinateType::X),
                                                        GridCoordinate1D (4, CoordinateType::X));
    VectorFieldValues<GridCoordinate1D>::Iterator iter_end =
      VectorFieldValues<GridCoordinate1D>::Iterator::getEndIterator (GridCoordinate1D (2, CoordinateType::X),
                                                                     GridCoordinate1D (4, CoordinateType::X));
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate1D (2, CoordinateType::X));
    ALWAYS_ASSERT (iter != iter_end);
    ++iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate1D (3, CoordinateType::X));
    ++iter;
    ALWAYS_ASSERT (iter == iter_end);
    --iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate1D (3, CoordinateType::X));
    --iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate1D (2, CoordinateType::X));
    --iter_end;
    ALWAYS_ASSERT (iter_end.getPos () == GridCoordinate1D (3, CoordinateType::X));
    ++iter_end;

    VectorFieldValues<GridCoordinate1D>::Iterator iter1 = iter;
    ALWAYS_ASSERT (iter1 == iter);

    VectorFieldValues<GridCoordinate1D> vector (GridCoordinate1D (4, CoordinateType::X));
    ALWAYS_ASSERT (vector.getSize () == GridCoordinate1D (4, CoordinateType::X));
    iter = vector.begin ();
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate1D (0, CoordinateType::X));
    for (; iter != vector.end (); ++iter)
    {
      GridCoordinate1D pos = iter.getPos ();
      grid_coord i = pos.get1 ();
      vector.set (pos, FIELDVALUE (102 * i, 18 * i));
    }

    iter = vector.begin ();
    for (; iter != vector.end (); ++iter)
    {
      GridCoordinate1D pos = iter.getPos ();
      grid_coord i = pos.get1 ();
      ALWAYS_ASSERT (*vector.get (pos) == FIELDVALUE (102 * i, 18 * i));
    }

    vector.resizeAndEmpty (GridCoordinate1D (3, CoordinateType::X));
    ALWAYS_ASSERT (vector.getSize () == GridCoordinate1D (3, CoordinateType::X));
    vector.initialize (FIELDVALUE (12, 18));

    iter = vector.begin ();
    for (; iter != vector.end (); ++iter)
    {
      GridCoordinate1D pos = iter.getPos ();
      ALWAYS_ASSERT (*vector.get (pos) == FIELDVALUE (12, 18));
    }

    VectorFieldValues<GridCoordinate1D> vector1 (GridCoordinate1D (3, CoordinateType::X));
    vector1.copy (&vector);
    ALWAYS_ASSERT (vector1.begin () == vector.begin ());
    ALWAYS_ASSERT (vector1.end () == vector.end ());

    iter = vector1.begin ();
    for (; iter != vector1.end (); ++iter)
    {
      GridCoordinate1D pos = iter.getPos ();
      ALWAYS_ASSERT (*vector1.get (pos) == *vector.get (pos));
    }
  }

  {
    VectorFieldValues<GridCoordinate2D>::Iterator iter (GridCoordinate2D (2, 2, CoordinateType::X, CoordinateType::Y),
                                                        GridCoordinate2D (2, 2, CoordinateType::X, CoordinateType::Y),
                                                        GridCoordinate2D (4, 4, CoordinateType::X, CoordinateType::Y));
    VectorFieldValues<GridCoordinate2D>::Iterator iter_end =
      VectorFieldValues<GridCoordinate2D>::Iterator::getEndIterator (GridCoordinate2D (2, 2, CoordinateType::X, CoordinateType::Y),
                                                                     GridCoordinate2D (4, 4, CoordinateType::X, CoordinateType::Y));
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate2D (2, 2, CoordinateType::X, CoordinateType::Y));
    ALWAYS_ASSERT (iter != iter_end);
    ++iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate2D (2, 3, CoordinateType::X, CoordinateType::Y));
    ++iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate2D (3, 2, CoordinateType::X, CoordinateType::Y));
    ++iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate2D (3, 3, CoordinateType::X, CoordinateType::Y));
    --iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate2D (3, 2, CoordinateType::X, CoordinateType::Y));
    --iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate2D (2, 3, CoordinateType::X, CoordinateType::Y));
    --iter_end;
    ALWAYS_ASSERT (iter_end.getPos () == GridCoordinate2D (3, 3, CoordinateType::X, CoordinateType::Y));

    VectorFieldValues<GridCoordinate2D>::Iterator iter1 = iter;
    ALWAYS_ASSERT (iter1 == iter);

    VectorFieldValues<GridCoordinate2D> vector (GridCoordinate2D (4, 4, CoordinateType::X, CoordinateType::Y));
    ALWAYS_ASSERT (vector.getSize () == GridCoordinate2D (4, 4, CoordinateType::X, CoordinateType::Y));
    iter = vector.begin ();
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate2D (0, 0, CoordinateType::X, CoordinateType::Y));
    for (; iter != vector.end (); ++iter)
    {
      GridCoordinate2D pos = iter.getPos ();
      grid_coord i = pos.get1 ();
      grid_coord j = pos.get2 ();
      vector.set (pos, FIELDVALUE (102 * i, 18 * j));
    }

    iter = vector.begin ();
    for (; iter != vector.end (); ++iter)
    {
      GridCoordinate2D pos = iter.getPos ();
      grid_coord i = pos.get1 ();
      grid_coord j = pos.get2 ();
      ALWAYS_ASSERT (*vector.get (pos) == FIELDVALUE (102 * i, 18 * j));
    }

    vector.resizeAndEmpty (GridCoordinate2D (3, 3, CoordinateType::X, CoordinateType::Y));
    ALWAYS_ASSERT (vector.getSize () == GridCoordinate2D (3, 3, CoordinateType::X, CoordinateType::Y));
    vector.initialize (FIELDVALUE (12, 18));

    iter = vector.begin ();
    for (; iter != vector.end (); ++iter)
    {
      GridCoordinate2D pos = iter.getPos ();
      ALWAYS_ASSERT (*vector.get (pos) == FIELDVALUE (12, 18));
    }

    VectorFieldValues<GridCoordinate2D> vector1 (GridCoordinate2D (3, 3, CoordinateType::X, CoordinateType::Y));
    vector1.copy (&vector);
    ALWAYS_ASSERT (vector1.begin () == vector.begin ());
    ALWAYS_ASSERT (vector1.end () == vector.end ());

    iter = vector1.begin ();
    for (; iter != vector1.end (); ++iter)
    {
      GridCoordinate2D pos = iter.getPos ();
      ALWAYS_ASSERT (*vector1.get (pos) == *vector.get (pos));
    }
  }

  {
    VectorFieldValues<GridCoordinate3D>::Iterator iter (GridCoordinate3D (2, 2, 2, CoordinateType::X, CoordinateType::Y, CoordinateType::Z),
                                                        GridCoordinate3D (2, 2, 2, CoordinateType::X, CoordinateType::Y, CoordinateType::Z),
                                                        GridCoordinate3D (4, 4, 4, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    VectorFieldValues<GridCoordinate3D>::Iterator iter_end =
      VectorFieldValues<GridCoordinate3D>::Iterator::getEndIterator (GridCoordinate3D (2, 2, 2, CoordinateType::X, CoordinateType::Y, CoordinateType::Z),
                                                                     GridCoordinate3D (4, 4, 4, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate3D (2, 2, 2, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    ALWAYS_ASSERT (iter != iter_end);
    ++iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate3D (2, 2, 3, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    ++iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate3D (2, 3, 2, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    ++iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate3D (2, 3, 3, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    ++iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate3D (3, 2, 2, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    --iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate3D (2, 3, 3, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    --iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate3D (2, 3, 2, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    --iter;
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate3D (2, 2, 3, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    --iter_end;
    ALWAYS_ASSERT (iter_end.getPos () == GridCoordinate3D (3, 3, 3, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));

    VectorFieldValues<GridCoordinate3D>::Iterator iter1 = iter;
    ALWAYS_ASSERT (iter1 == iter);

    VectorFieldValues<GridCoordinate3D> vector (GridCoordinate3D (4, 4, 4, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    ALWAYS_ASSERT (vector.getSize () == GridCoordinate3D (4, 4, 4, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    iter = vector.begin ();
    ALWAYS_ASSERT (iter.getPos () == GridCoordinate3D (0, 0, 0, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    for (; iter != vector.end (); ++iter)
    {
      GridCoordinate3D pos = iter.getPos ();
      grid_coord i = pos.get1 ();
      grid_coord j = pos.get2 ();
      grid_coord k = pos.get3 ();
      vector.set (pos, FIELDVALUE (102 * i + 7 * k, 18 * j + k));
    }

    iter = vector.begin ();
    for (; iter != vector.end (); ++iter)
    {
      GridCoordinate3D pos = iter.getPos ();
      grid_coord i = pos.get1 ();
      grid_coord j = pos.get2 ();
      grid_coord k = pos.get3 ();
      ALWAYS_ASSERT (*vector.get (pos) == FIELDVALUE (102 * i + 7 * k, 18 * j + k));
    }

    vector.resizeAndEmpty (GridCoordinate3D (3, 3, 3, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    ALWAYS_ASSERT (vector.getSize () == GridCoordinate3D (3, 3, 3, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    vector.initialize (FIELDVALUE (12, 18));

    iter = vector.begin ();
    for (; iter != vector.end (); ++iter)
    {
      GridCoordinate3D pos = iter.getPos ();
      ALWAYS_ASSERT (*vector.get (pos) == FIELDVALUE (12, 18));
    }

    VectorFieldValues<GridCoordinate3D> vector1 (GridCoordinate3D (3, 3, 3, CoordinateType::X, CoordinateType::Y, CoordinateType::Z));
    vector1.copy (&vector);
    ALWAYS_ASSERT (vector1.begin () == vector.begin ());
    ALWAYS_ASSERT (vector1.end () == vector.end ());

    iter = vector1.begin ();
    for (; iter != vector1.end (); ++iter)
    {
      GridCoordinate3D pos = iter.getPos ();
      ALWAYS_ASSERT (*vector1.get (pos) == *vector.get (pos));
    }
  }

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
