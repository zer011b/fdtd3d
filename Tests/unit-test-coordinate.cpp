/*
 * Unit test for basic operations with GridCoordinate
 */

#include <iostream>

#include "Assert.h"
#include "GridCoordinate3D.h"

#ifndef CXX11_ENABLED
#include "cstdlib"
#endif /* !CXX11_ENABLED */

template<class TcoordType, bool doSignChecks>
void testFunc (uint8_t combination)
{
  CoordinateType t1, t2, t3;

  switch (combination)
  {
    case 0:
    {
      t1 = CoordinateType::X;
      t2 = CoordinateType::Y;
      t3 = CoordinateType::Z;
      break;
    }
    case 1:
    {
      t1 = CoordinateType::Y;
      t2 = CoordinateType::Z;
      t3 = CoordinateType::X;
      break;
    }
    case 2:
    {
      t1 = CoordinateType::Z;
      t2 = CoordinateType::X;
      t3 = CoordinateType::Y;
      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  for (grid_coord i = doSignChecks ? 0 : -1; i < 2; ++i)
  {
    /*
     * constructors
     */
    ALWAYS_ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks> (i, t1) == GridCoordinate1DTemplate<TcoordType, doSignChecks> (i, i, i, t1, t2, t3)));
    ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks> (i, i, t1, t2) == GridCoordinate2DTemplate<TcoordType, doSignChecks> (i, i, i, t1, t2, t3)));

    GridCoordinate1DTemplate<TcoordType, doSignChecks> coord1D (i, t1);
    coord1D.print ();
    ALWAYS_ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks> (coord1D.get1 (), t1) == coord1D));
    ALWAYS_ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks> (coord1D) == coord1D));

    GridCoordinate2DTemplate<TcoordType, doSignChecks> coord2D (i, i, t1, t2);
    coord2D.print ();
    ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks> (coord2D.get1 (), coord2D.get2 (), t1, t2) == coord2D));
    ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks> (coord2D) == coord2D));

    GridCoordinate3DTemplate<TcoordType, doSignChecks> coord3D (i, i, i, t1, t2, t3);
    coord3D.print ();
    ALWAYS_ASSERT ((GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord3D.get1 (), coord3D.get2 (), coord3D.get3 (), t1, t2, t3) == coord3D));
    ALWAYS_ASSERT ((GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord3D) == coord3D));

    /*
     * modification
     */
    GridCoordinate1DTemplate<TcoordType, doSignChecks> coord1D_1 (i + 5, t1);
    coord1D_1.set1 (coord1D_1.get1 () - 5);
    ALWAYS_ASSERT (coord1D_1 == coord1D);

    GridCoordinate2DTemplate<TcoordType, doSignChecks> coord2D_1 (i + 5, i + 5, t1, t2);
    coord2D_1.set1 (coord2D_1.get1 () - 5);
    coord2D_1.set2 (coord2D_1.get2 () - 5);
    ALWAYS_ASSERT (coord2D_1 == coord2D);

    GridCoordinate3DTemplate<TcoordType, doSignChecks> coord3D_1 (i + 5, i + 5, i + 5, t1, t2, t3);
    coord3D_1.set1 (coord3D_1.get1 () - 5);
    coord3D_1.set2 (coord3D_1.get2 () - 5);
    coord3D_1.set3 (coord3D_1.get3 () - 5);
    ALWAYS_ASSERT (coord3D_1 == coord3D);

    /*
     * total coordinate
     */
    ALWAYS_ASSERT (coord1D.calculateTotalCoord () == coord1D.get1 ());
    ALWAYS_ASSERT (coord2D.calculateTotalCoord () == coord2D.get1 () * coord2D.get2 ());
    ALWAYS_ASSERT (coord3D.calculateTotalCoord () == coord3D.get1 () * coord3D.get2 () * coord3D.get3 ());

    /*
     * max coord
     */
    ALWAYS_ASSERT (coord1D.getMax () == coord1D.get1 ());
    ALWAYS_ASSERT (coord2D.getMax () == coord2D.get1 () && coord2D.get1 () == coord2D.get2 ());
    ALWAYS_ASSERT (coord3D.getMax () == coord3D.get1 () && coord3D.get1 () == coord3D.get2 () && coord3D.get2 () == coord3D.get3 ());

    /*
     * arithmetic operations and comparison
     */
    coord1D_1 = coord1D + coord1D_1;
    ALWAYS_ASSERT (coord1D_1.get1 () == TcoordType (2) * coord1D.get1 ());
    ALWAYS_ASSERT (coord1D_1 == TcoordType (2) * coord1D);
    if (i > 0)
    {
      ALWAYS_ASSERT (coord1D_1 != coord1D);
      ALWAYS_ASSERT (coord1D_1 > coord1D);
      ALWAYS_ASSERT (coord1D < coord1D_1);
      ALWAYS_ASSERT (coord1D_1 >= coord1D);
      ALWAYS_ASSERT (coord1D <= coord1D_1);
    }
    coord1D_1 = coord1D_1 - coord1D;
    ALWAYS_ASSERT (coord1D_1.get1 () == coord1D.get1 ());
    ALWAYS_ASSERT (coord1D_1 == coord1D);

    coord2D_1 = coord2D + coord2D_1;
    ALWAYS_ASSERT (coord2D_1.get1 () == TcoordType (2) * coord2D.get1 ());
    ALWAYS_ASSERT (coord2D_1.get2 () == TcoordType (2) * coord2D.get2 ());
    ALWAYS_ASSERT (coord2D_1 == TcoordType (2) * coord2D);
    if (i > 0)
    {
      ALWAYS_ASSERT (coord2D_1 != coord2D);
      ALWAYS_ASSERT (coord2D_1 > coord2D);
      ALWAYS_ASSERT (coord2D < coord2D_1);
      ALWAYS_ASSERT (coord2D_1 >= coord2D);
      ALWAYS_ASSERT (coord2D <= coord2D_1);
    }
    coord2D_1 = coord2D_1 - coord2D;
    ALWAYS_ASSERT (coord2D_1.get1 () == coord2D.get1 ());
    ALWAYS_ASSERT (coord2D_1.get2 () == coord2D.get2 ());
    ALWAYS_ASSERT (coord2D_1 == coord2D);

    coord3D_1 = coord3D + coord3D_1;
    ALWAYS_ASSERT (coord3D_1.get1 () == TcoordType (2) * coord3D.get1 ());
    ALWAYS_ASSERT (coord3D_1.get2 () == TcoordType (2) * coord3D.get2 ());
    ALWAYS_ASSERT (coord3D_1.get2 () == TcoordType (2) * coord3D.get2 ());
    ALWAYS_ASSERT (coord3D_1 == TcoordType (2) * coord3D);
    if (i > 0)
    {
      ALWAYS_ASSERT (coord3D_1 != coord3D);
      ALWAYS_ASSERT (coord3D_1 > coord3D);
      ALWAYS_ASSERT (coord3D < coord3D_1);
      ALWAYS_ASSERT (coord3D_1 >= coord3D);
      ALWAYS_ASSERT (coord3D <= coord3D_1);
    }
    coord3D_1 = coord3D_1 - coord3D;
    ALWAYS_ASSERT (coord3D_1.get1 () == coord3D.get1 ());
    ALWAYS_ASSERT (coord3D_1.get2 () == coord3D.get2 ());
    ALWAYS_ASSERT (coord3D_1.get2 () == coord3D.get2 ());
    ALWAYS_ASSERT (coord3D_1 == coord3D);

    /*
     * arithmetic operations with opposite sign checks
     */
    GridCoordinate1DTemplate<TcoordType, !doSignChecks> coord1D_2 (i + 5, t1);
    ALWAYS_ASSERT ((coord1D_1 + coord1D_2 == TcoordType (2) * coord1D + GridCoordinate1DTemplate<TcoordType, doSignChecks> (5, t1)));
    GridCoordinate2DTemplate<TcoordType, !doSignChecks> coord2D_2 (i + 5, i + 15, t1, t2);
    ALWAYS_ASSERT ((coord2D_1 + coord2D_2 == TcoordType (2) * coord2D + GridCoordinate2DTemplate<TcoordType, doSignChecks> (5, 15, t1, t2)));
    GridCoordinate3DTemplate<TcoordType, !doSignChecks> coord3D_2 (i + 5, i + 15, i + 115, t1, t2, t3);
    ALWAYS_ASSERT ((coord3D_1 + coord3D_2 == TcoordType (2) * coord3D + GridCoordinate3DTemplate<TcoordType, doSignChecks> (5, 15, 115, t1, t2, t3)));

    /*
     * conversions
     */
    coord2D = expand (coord1D, t2);
    ALWAYS_ASSERT (coord2D.get1 () == coord1D.get1 () && coord2D.get2 () == 0);

    coord3D = expand2 (coord1D, t2, t3);
    ALWAYS_ASSERT (coord3D.get1 () == coord1D.get1 () && coord3D.get2 () == 0 && coord3D.get3 () == 0);

    ALWAYS_ASSERT (expand (coord2D, t3) == coord3D);
    ALWAYS_ASSERT (coord2D.shrink () == coord1D);
    ALWAYS_ASSERT (coord3D.shrink () == coord2D);
  }
}

int main (int argc, char** argv)
{
#ifndef DEBUG_INFO
  ALWAYS_ASSERT_MESSAGE ("Test requires debug info");
#endif /* !DEBUG_INFO */

  for (uint8_t i = 0; i < static_cast<uint8_t> (CoordinateType::COUNT); ++i)
  {
    testFunc<grid_coord, true> (i);
    testFunc<grid_coord, false> (i);
    testFunc<FPValue, true> (i);
    testFunc<FPValue, false> (i);
  }

  /*
   * test FPValue and grid_coord operations
   */
  for (grid_coord i = 0; i < 2; ++i)
  {
    GridCoordinate1D coord1D (i);
    ALWAYS_ASSERT (convertCoord (convertCoord (coord1D)) == coord1D);

    GridCoordinate2D coord2D (i, i);
    ALWAYS_ASSERT (convertCoord (convertCoord (coord2D)) == coord2D);

    GridCoordinate3D coord3D (i, i, i);
    ALWAYS_ASSERT (convertCoord (convertCoord (coord3D)) == coord3D);
  }

  return 0;
} /* main */
