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
void testFunc (CoordinateType t1, CoordinateType t2, CoordinateType t3,
               bool correct1D, bool correct2D, bool correct3D)
{
  ALWAYS_ASSERT (correct1D);

  for (grid_coord i = doSignChecks ? 0 : -3; i <= 3; ++i)
  {
    /*
     * constructors
     */
    ALWAYS_ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks> (i, t1) == GridCoordinate1DTemplate<TcoordType, doSignChecks> (i, i, i, t1, t2, t3)));
    if (correct2D)
    {
      ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks> (i, 10 * i, t1, t2) == GridCoordinate2DTemplate<TcoordType, doSignChecks> (i, 10 * i, i, t1, t2, t3)));
    }

    GridCoordinate1DTemplate<TcoordType, doSignChecks> *coord1D = new GridCoordinate1DTemplate<TcoordType, doSignChecks> (i, t1);
    GridCoordinate1DTemplate<TcoordType, doSignChecks> *coord1D_1 = new GridCoordinate1DTemplate<TcoordType, doSignChecks> (i + 5, t1);
    GridCoordinate1DTemplate<TcoordType, !doSignChecks> *coord1D_2 = new GridCoordinate1DTemplate<TcoordType, !doSignChecks> (i + 44, t1);
    coord1D->print ();
    ALWAYS_ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks> (coord1D->get1 (), t1) == *coord1D));
    ALWAYS_ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks> (*coord1D) == *coord1D));

    GridCoordinate2DTemplate<TcoordType, doSignChecks> *coord2D = NULLPTR;
    GridCoordinate2DTemplate<TcoordType, doSignChecks> *coord2D_1 = NULLPTR;
    GridCoordinate2DTemplate<TcoordType, !doSignChecks> *coord2D_2 = NULLPTR;
    if (correct2D)
    {
      coord2D = new GridCoordinate2DTemplate<TcoordType, doSignChecks> (i, 10 * i, t1, t2);
      coord2D_1 = new GridCoordinate2DTemplate<TcoordType, doSignChecks> (i + 5, 10 * i + 5, t1, t2);
      coord2D_2 = new GridCoordinate2DTemplate<TcoordType, !doSignChecks> (i + 44, 10 * i + 333, t1, t2);
      coord2D->print ();
      ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks> (coord2D->get1 (), coord2D->get2 (), t1, t2) == *coord2D));
      ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks> (*coord2D) == *coord2D));
    }

    GridCoordinate3DTemplate<TcoordType, doSignChecks> *coord3D = NULLPTR;
    GridCoordinate3DTemplate<TcoordType, doSignChecks> *coord3D_1 = NULLPTR;
    GridCoordinate3DTemplate<TcoordType, !doSignChecks> *coord3D_2 = NULLPTR;
    if (correct3D)
    {
      coord3D = new GridCoordinate3DTemplate<TcoordType, doSignChecks> (i, 10 * i, 17 * i, t1, t2, t3);
      coord3D_1 = new GridCoordinate3DTemplate<TcoordType, doSignChecks> (i + 5, 10 * i + 5, 17 * i + 5, t1, t2, t3);
      coord3D_2 = new GridCoordinate3DTemplate<TcoordType, !doSignChecks> (i + 44, 10 * i + 333, 17 * i + 941, t1, t2, t3);
      coord3D->print ();
      ALWAYS_ASSERT ((GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord3D->get1 (), coord3D->get2 (), coord3D->get3 (), t1, t2, t3) == *coord3D));
      ALWAYS_ASSERT ((GridCoordinate3DTemplate<TcoordType, doSignChecks> (*coord3D) == *coord3D));
    }

    /*
     * modification
     */

    coord1D_1->set1 (coord1D_1->get1 () - 5);
    ALWAYS_ASSERT (*coord1D_1 == *coord1D);

    if (correct2D)
    {
      coord2D_1->set1 (coord2D_1->get1 () - 5);
      coord2D_1->set2 (coord2D_1->get2 () - 5);
      ALWAYS_ASSERT (*coord2D_1 == *coord2D);
    }

    if (correct3D)
    {
      coord3D_1->set1 (coord3D_1->get1 () - 5);
      coord3D_1->set2 (coord3D_1->get2 () - 5);
      coord3D_1->set3 (coord3D_1->get3 () - 5);
      ALWAYS_ASSERT (*coord3D_1 == *coord3D);
    }

    /*
     * total coordinate
     */
    ALWAYS_ASSERT (coord1D->calculateTotalCoord () == coord1D->get1 ());
    if (correct2D)
    {
      ALWAYS_ASSERT (coord2D->calculateTotalCoord () == coord2D->get1 () * coord2D->get2 ());
    }
    if (correct3D)
    {
      ALWAYS_ASSERT (coord3D->calculateTotalCoord () == coord3D->get1 () * coord3D->get2 () * coord3D->get3 ());
    }

    /*
     * max coord
     */
    ALWAYS_ASSERT (coord1D->getMax () == coord1D->get1 ());
    if (correct2D)
    {
      ALWAYS_ASSERT (coord2D->getMax () == (coord2D->get1 () > coord2D->get2 () ? coord2D->get1 () : coord2D->get2 ()));
    }
    if (correct3D)
    {
      grid_coord max = coord3D->get1 () > coord3D->get2 () ? coord3D->get1 () : coord3D->get2 ();
      max = max > coord3D->get3 () ? max : coord3D->get3 ();
      ALWAYS_ASSERT (coord3D->getMax () == max);
    }

    /*
     * arithmetic operations and comparison
     */
    *coord1D_1 = *coord1D + *coord1D_1;
    ALWAYS_ASSERT (coord1D_1->get1 () == TcoordType (2) * coord1D->get1 ());
    ALWAYS_ASSERT ((*coord1D_1) == TcoordType (2) * (*coord1D));
    ALWAYS_ASSERT (coord1D_1->get1 () / TcoordType (2) == coord1D->get1 ());
    ALWAYS_ASSERT ((*coord1D_1) / TcoordType (2) == (*coord1D));
    if (i > 0)
    {
      ALWAYS_ASSERT (*coord1D_1 != *coord1D);
      ALWAYS_ASSERT (*coord1D_1 > *coord1D);
      ALWAYS_ASSERT (*coord1D < *coord1D_1);
      ALWAYS_ASSERT (*coord1D_1 >= *coord1D);
      ALWAYS_ASSERT (*coord1D <= *coord1D_1);
    }
    *coord1D_1 = *coord1D_1 - *coord1D;
    ALWAYS_ASSERT (coord1D_1->get1 () == coord1D->get1 ());
    ALWAYS_ASSERT (*coord1D_1 == *coord1D);

    if (correct2D)
    {
      *coord2D_1 = *coord2D + *coord2D_1;
      ALWAYS_ASSERT (coord2D_1->get1 () == TcoordType (2) * coord2D->get1 ());
      ALWAYS_ASSERT (coord2D_1->get2 () == TcoordType (2) * coord2D->get2 ());
      ALWAYS_ASSERT (*coord2D_1 == TcoordType (2) * (*coord2D));
      ALWAYS_ASSERT (coord2D_1->get1 () / TcoordType (2) == coord2D->get1 ());
      ALWAYS_ASSERT (coord2D_1->get2 () / TcoordType (2) == coord2D->get2 ());
      ALWAYS_ASSERT ((*coord2D_1) / TcoordType (2) == *coord2D);
      if (i > 0)
      {
        ALWAYS_ASSERT (*coord2D_1 != *coord2D);
        ALWAYS_ASSERT (*coord2D_1 > *coord2D);
        ALWAYS_ASSERT (*coord2D < *coord2D_1);
        ALWAYS_ASSERT (*coord2D_1 >= *coord2D);
        ALWAYS_ASSERT (*coord2D <= *coord2D_1);
      }
      *coord2D_1 = *coord2D_1 - *coord2D;
      ALWAYS_ASSERT (coord2D_1->get1 () == coord2D->get1 ());
      ALWAYS_ASSERT (coord2D_1->get2 () == coord2D->get2 ());
      ALWAYS_ASSERT (*coord2D_1 == *coord2D);
    }

    if (correct3D)
    {
      *coord3D_1 = *coord3D + *coord3D_1;
      ALWAYS_ASSERT (coord3D_1->get1 () == TcoordType (2) * coord3D->get1 ());
      ALWAYS_ASSERT (coord3D_1->get2 () == TcoordType (2) * coord3D->get2 ());
      ALWAYS_ASSERT (coord3D_1->get3 () == TcoordType (2) * coord3D->get3 ());
      ALWAYS_ASSERT (*coord3D_1 == TcoordType (2) * (*coord3D));
      ALWAYS_ASSERT (coord3D_1->get1 () / TcoordType (2) == coord3D->get1 ());
      ALWAYS_ASSERT (coord3D_1->get2 () / TcoordType (2) == coord3D->get2 ());
      ALWAYS_ASSERT (coord3D_1->get3 () / TcoordType (2) == coord3D->get3 ());
      ALWAYS_ASSERT ((*coord3D_1) / TcoordType (2) == *coord3D);
      if (i > 0)
      {
        ALWAYS_ASSERT (*coord3D_1 != *coord3D);
        ALWAYS_ASSERT (*coord3D_1 > *coord3D);
        ALWAYS_ASSERT (*coord3D < *coord3D_1);
        ALWAYS_ASSERT (*coord3D_1 >= *coord3D);
        ALWAYS_ASSERT (*coord3D <= *coord3D_1);
      }
      *coord3D_1 = *coord3D_1 - *coord3D;
      ALWAYS_ASSERT (coord3D_1->get1 () == coord3D->get1 ());
      ALWAYS_ASSERT (coord3D_1->get2 () == coord3D->get2 ());
      ALWAYS_ASSERT (coord3D_1->get3 () == coord3D->get3 ());
      ALWAYS_ASSERT (*coord3D_1 == *coord3D);
    }

    /*
     * arithmetic operations with opposite sign checks
     */
    ALWAYS_ASSERT ((*coord1D_1 + *coord1D_2 == TcoordType (2) * (*coord1D) + GridCoordinate1DTemplate<TcoordType, doSignChecks> (44, t1)));
    if (correct2D)
    {
      ALWAYS_ASSERT ((*coord2D_1 + *coord2D_2 == TcoordType (2) * (*coord2D) + GridCoordinate2DTemplate<TcoordType, doSignChecks> (44, 333, t1, t2)));
    }
    if (correct3D)
    {
      ALWAYS_ASSERT ((*coord3D_1 + *coord3D_2 == TcoordType (2) * (*coord3D) + GridCoordinate3DTemplate<TcoordType, doSignChecks> (44, 333, 941, t1, t2, t3)));
    }

    /*
     * conversions
     */
    ALWAYS_ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::initAxesCoordinate (i, 10 * i, 17 * i, CoordinateType::X, CoordinateType::NONE, CoordinateType::NONE).get1 () == i));
    ALWAYS_ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::initAxesCoordinate (i, 10 * i, 17 * i, CoordinateType::Y, CoordinateType::NONE, CoordinateType::NONE).get1 () == 10 * i));
    ALWAYS_ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::initAxesCoordinate (i, 10 * i, 17 * i, CoordinateType::Z, CoordinateType::NONE, CoordinateType::NONE).get1 () == 17 * i));

    ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::initAxesCoordinate (i, 10 * i, 17 * i, CoordinateType::X, CoordinateType::Y, CoordinateType::NONE).get1 () == i));
    ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::initAxesCoordinate (i, 10 * i, 17 * i, CoordinateType::X, CoordinateType::Y, CoordinateType::NONE).get2 () == 10 * i));
    ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::initAxesCoordinate (i, 10 * i, 17 * i, CoordinateType::Y, CoordinateType::Z, CoordinateType::NONE).get1 () == 10 * i));
    ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::initAxesCoordinate (i, 10 * i, 17 * i, CoordinateType::Y, CoordinateType::Z, CoordinateType::NONE).get2 () == 17 * i));
    ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::initAxesCoordinate (i, 10 * i, 17 * i, CoordinateType::X, CoordinateType::Z, CoordinateType::NONE).get1 () == i));
    ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::initAxesCoordinate (i, 10 * i, 17 * i, CoordinateType::X, CoordinateType::Z, CoordinateType::NONE).get2 () == 17 * i));

    ALWAYS_ASSERT ((GridCoordinate3DTemplate<TcoordType, doSignChecks>::initAxesCoordinate (i, 10 * i, 17 * i, CoordinateType::X, CoordinateType::Y, CoordinateType::Z).get1 () == i));
    ALWAYS_ASSERT ((GridCoordinate3DTemplate<TcoordType, doSignChecks>::initAxesCoordinate (i, 10 * i, 17 * i, CoordinateType::X, CoordinateType::Y, CoordinateType::Z).get2 () == 10 * i));
    ALWAYS_ASSERT ((GridCoordinate3DTemplate<TcoordType, doSignChecks>::initAxesCoordinate (i, 10 * i, 17 * i, CoordinateType::X, CoordinateType::Y, CoordinateType::Z).get3 () == 17 * i));

    if (correct2D)
    {
      *coord2D = expand (*coord1D, t2);
      ALWAYS_ASSERT (coord2D->get1 () == coord1D->get1 () && coord2D->get2 () == 0);
      ALWAYS_ASSERT (coord2D->shrink () == *coord1D);
    }

    if (correct3D)
    {
      *coord3D = expand2 (*coord1D, t2, t3);
      ALWAYS_ASSERT (coord3D->get1 () == coord1D->get1 () && coord3D->get2 () == 0 && coord3D->get3 () == 0);

      ALWAYS_ASSERT ((expandTo3D (GridCoordinate1DTemplate<TcoordType, doSignChecks> (i)) == GridCoordinate3DTemplate<TcoordType, doSignChecks> (i, 0, 0)));
      ALWAYS_ASSERT ((expandTo3D (GridCoordinate2DTemplate<TcoordType, doSignChecks> (i, 10 * i)) == GridCoordinate3DTemplate<TcoordType, doSignChecks> (i, 10 * i, 0)));
      ALWAYS_ASSERT ((expandTo3D (GridCoordinate3DTemplate<TcoordType, doSignChecks> (i, 10 * i, 17 * i)) == GridCoordinate3DTemplate<TcoordType, doSignChecks> (i, 10 * i, 17 * i)));

      GridCoordinate3DTemplate<TcoordType, doSignChecks> coord3D_3;
      GridCoordinate3DTemplate<TcoordType, doSignChecks> coord3D_4;
      expandTo3DStartEnd (GridCoordinate1DTemplate<TcoordType, doSignChecks> (i), GridCoordinate1DTemplate<TcoordType, doSignChecks> (10 * i), coord3D_3, coord3D_4);
      ALWAYS_ASSERT (coord3D_3.get1 () == i && coord3D_3.get2 () == 0 && coord3D_3.get3 () == 0);
      ALWAYS_ASSERT (coord3D_4.get1 () == 10 * i && coord3D_4.get2 () == 1 && coord3D_4.get3 () == 1);

      expandTo3DStartEnd (GridCoordinate2DTemplate<TcoordType, doSignChecks> (i, 10 * i), GridCoordinate2DTemplate<TcoordType, doSignChecks> (10 * i, 17 * i), coord3D_3, coord3D_4);
      ALWAYS_ASSERT (coord3D_3.get1 () == i && coord3D_3.get2 () == 10 * i && coord3D_3.get3 () == 0);
      ALWAYS_ASSERT (coord3D_4.get1 () == 10 * i && coord3D_4.get2 () == 17 * i && coord3D_4.get3 () == 1);

      expandTo3DStartEnd (GridCoordinate3DTemplate<TcoordType, doSignChecks> (i, 10 * i, 17 * i), GridCoordinate3DTemplate<TcoordType, doSignChecks> (10 * i, 17 * i, 22 * i), coord3D_3, coord3D_4);
      ALWAYS_ASSERT (coord3D_3.get1 () == i && coord3D_3.get2 () == 10 * i && coord3D_3.get3 () == 17 * i);
      ALWAYS_ASSERT (coord3D_4.get1 () == 10 * i && coord3D_4.get2 () == 17 * i && coord3D_4.get3 () == 22 * i);
    }

    if (correct2D && correct3D)
    {
      ALWAYS_ASSERT (expand (*coord2D, t3) == *coord3D);
      ALWAYS_ASSERT (coord3D->shrink () == *coord2D);
    }

    delete coord1D;
    delete coord2D;
    delete coord3D;

    delete coord1D_1;
    delete coord2D_1;
    delete coord3D_1;

    delete coord1D_2;
    delete coord2D_2;
    delete coord3D_2;
  }
}

int main (int argc, char** argv)
{
#ifndef DEBUG_INFO
  ALWAYS_ASSERT_MESSAGE ("Test requires debug info");
#endif /* !DEBUG_INFO */

  for (uint8_t i = 0; i < 4; ++i)
  {
    bool correct1D = false;
    bool correct2D = false;
    bool correct3D = false;

    CoordinateType t1 = CoordinateType::NONE;
    CoordinateType t2 = CoordinateType::NONE;
    CoordinateType t3 = CoordinateType::NONE;

    switch (i)
    {
      case 0:
      {
        t1 = CoordinateType::X;
        t2 = CoordinateType::Y;
        t3 = CoordinateType::Z;

        correct1D = true;
        correct2D = true;
        correct3D = true;
        break;
      }
      case 1:
      {
        t1 = CoordinateType::X;
        t2 = CoordinateType::Z;

        correct1D = true;
        correct2D = true;
        break;
      }
      case 2:
      {
        t1 = CoordinateType::Y;
        t2 = CoordinateType::Z;

        correct1D = true;
        correct2D = true;
        break;
      }
      case 3:
      {
        t1 = CoordinateType::Z;

        correct1D = true;
        break;
      }
      default:
      {
        UNREACHABLE;
      }
    }

    testFunc<grid_coord, true> (t1, t2, t3, correct1D, correct2D, correct3D);
    testFunc<grid_coord, false> (t1, t2, t3, correct1D, correct2D, correct3D);
    testFunc<FPValue, true> (t1, t2, t3, correct1D, correct2D, correct3D);
    testFunc<FPValue, false> (t1, t2, t3, correct1D, correct2D, correct3D);
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
