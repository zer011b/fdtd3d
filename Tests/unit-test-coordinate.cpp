/*
 * Unit test for basic operations with GridCoordinate
 */

#include <iostream>

#include "Assert.h"
#include "GridCoordinate3D.h"

#ifndef CXX11_ENABLED
#include "cstdlib"
#endif /* !CXX11_ENABLED */

#ifndef DEBUG_INFO
#error Test requires debug info
#endif /* !DEBUG_INFO */

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

    GridCoordinate1DTemplate<TcoordType, doSignChecks> coord1D__ (*coord1D);
    ALWAYS_ASSERT (coord1D__ == *coord1D);

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

      GridCoordinate2DTemplate<TcoordType, doSignChecks> coord2D__ (*coord2D);
      ALWAYS_ASSERT (coord2D__ == *coord2D);

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

      GridCoordinate3DTemplate<TcoordType, doSignChecks> coord3D__ (*coord3D);
      ALWAYS_ASSERT (coord3D__ == *coord3D);

      ALWAYS_ASSERT ((GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord3D->get1 (), coord3D->get2 (), coord3D->get3 (), t1, t2, t3) == *coord3D));
      ALWAYS_ASSERT ((GridCoordinate3DTemplate<TcoordType, doSignChecks> (*coord3D) == *coord3D));
    }

    /*
     * Operators
     */
    GridCoordinate1DTemplate<TcoordType, doSignChecks> coord1D___ (0, 0, 0, t1, t2, t3);
    coord1D___ = *coord1D;
    ALWAYS_ASSERT (coord1D___ == *coord1D);

    if (correct2D)
    {
      GridCoordinate2DTemplate<TcoordType, doSignChecks> coord2D___ (0, 0, 0, t1, t2, t3);
      coord2D___ = *coord2D;
      ALWAYS_ASSERT (coord2D___ == *coord2D);
    }

    if (correct3D)
    {
      GridCoordinate3DTemplate<TcoordType, doSignChecks> coord3D___ (0, 0, 0, t1, t2, t3);
      coord3D___ = *coord3D;
      ALWAYS_ASSERT (coord3D___ == *coord3D);
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
    ALWAYS_ASSERT ((*coord1D_1) == (*coord1D) * TcoordType (2));
    ALWAYS_ASSERT (coord1D_1->get1 () / TcoordType (2) == coord1D->get1 ());
    ALWAYS_ASSERT ((*coord1D_1) / TcoordType (2) == (*coord1D));

    ALWAYS_ASSERT ((*coord1D_1 * *coord1D).get1 () == coord1D_1->get1 () * coord1D->get1 ());

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
      ALWAYS_ASSERT (*coord2D_1 == (*coord2D) * TcoordType (2));
      ALWAYS_ASSERT (coord2D_1->get1 () / TcoordType (2) == coord2D->get1 ());
      ALWAYS_ASSERT (coord2D_1->get2 () / TcoordType (2) == coord2D->get2 ());
      ALWAYS_ASSERT ((*coord2D_1) / TcoordType (2) == *coord2D);

      ALWAYS_ASSERT ((*coord2D_1 * *coord2D).get1 () == coord2D_1->get1 () * coord2D->get1 ());
      ALWAYS_ASSERT ((*coord2D_1 * *coord2D).get2 () == coord2D_1->get2 () * coord2D->get2 ());

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
      ALWAYS_ASSERT (*coord3D_1 == (*coord3D) * TcoordType (2));
      ALWAYS_ASSERT (coord3D_1->get1 () / TcoordType (2) == coord3D->get1 ());
      ALWAYS_ASSERT (coord3D_1->get2 () / TcoordType (2) == coord3D->get2 ());
      ALWAYS_ASSERT (coord3D_1->get3 () / TcoordType (2) == coord3D->get3 ());
      ALWAYS_ASSERT ((*coord3D_1) / TcoordType (2) == *coord3D);

      ALWAYS_ASSERT ((*coord3D_1 * *coord3D).get1 () == coord3D_1->get1 () * coord3D->get1 ());
      ALWAYS_ASSERT ((*coord3D_1 * *coord3D).get2 () == coord3D_1->get2 () * coord3D->get2 ());
      ALWAYS_ASSERT ((*coord3D_1 * *coord3D).get3 () == coord3D_1->get3 () * coord3D->get3 ());

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
    ALWAYS_ASSERT ((*coord1D_1 + *coord1D_2 == (*coord1D) * TcoordType (2) + GridCoordinate1DTemplate<TcoordType, doSignChecks> (44, t1)));
    if (correct2D)
    {
      ALWAYS_ASSERT ((*coord2D_1 + *coord2D_2 == (*coord2D) * TcoordType (2) + GridCoordinate2DTemplate<TcoordType, doSignChecks> (44, 333, t1, t2)));
    }
    if (correct3D)
    {
      ALWAYS_ASSERT ((*coord3D_1 + *coord3D_2 == (*coord3D) * TcoordType (2) + GridCoordinate3DTemplate<TcoordType, doSignChecks> (44, 333, 941, t1, t2, t3)));
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

    if (correct3D)
    {
      ALWAYS_ASSERT ((expandTo3D (GridCoordinate1DTemplate<TcoordType, doSignChecks> (i, CoordinateType::X),
                                  CoordinateType::X, CoordinateType::NONE, CoordinateType::NONE)
                      == GridCoordinate3DTemplate<TcoordType, doSignChecks> (i, 0, 0, CoordinateType::X, CoordinateType::Y, CoordinateType::Z)));
      ALWAYS_ASSERT ((expandTo3D (GridCoordinate1DTemplate<TcoordType, doSignChecks> (i, CoordinateType::Y),
                                  CoordinateType::Y, CoordinateType::NONE, CoordinateType::NONE)
                      == GridCoordinate3DTemplate<TcoordType, doSignChecks> (0, i, 0, CoordinateType::X, CoordinateType::Y, CoordinateType::Z)));
      ALWAYS_ASSERT ((expandTo3D (GridCoordinate1DTemplate<TcoordType, doSignChecks> (i, CoordinateType::Z),
                                  CoordinateType::Z, CoordinateType::NONE, CoordinateType::NONE)
                      == GridCoordinate3DTemplate<TcoordType, doSignChecks> (0, 0, i, CoordinateType::X, CoordinateType::Y, CoordinateType::Z)));
      ALWAYS_ASSERT ((expandTo3D (GridCoordinate2DTemplate<TcoordType, doSignChecks> (i, 10 * i, CoordinateType::X, CoordinateType::Y),
                                  CoordinateType::X, CoordinateType::Y, CoordinateType::NONE)
                      == GridCoordinate3DTemplate<TcoordType, doSignChecks> (i, 10 * i, 0, CoordinateType::X, CoordinateType::Y, CoordinateType::Z)));
      ALWAYS_ASSERT ((expandTo3D (GridCoordinate2DTemplate<TcoordType, doSignChecks> (i, 10 * i, CoordinateType::Y, CoordinateType::Z),
                                  CoordinateType::Y, CoordinateType::Z, CoordinateType::NONE)
                      == GridCoordinate3DTemplate<TcoordType, doSignChecks> (0, i, 10 * i, CoordinateType::X, CoordinateType::Y, CoordinateType::Z)));
      ALWAYS_ASSERT ((expandTo3D (GridCoordinate3DTemplate<TcoordType, doSignChecks> (i, 10 * i, 17 * i, CoordinateType::X, CoordinateType::Y, CoordinateType::Z),
                                  CoordinateType::X, CoordinateType::Y, CoordinateType::Z)
                      == GridCoordinate3DTemplate<TcoordType, doSignChecks> (i, 10 * i, 17 * i, CoordinateType::X, CoordinateType::Y, CoordinateType::Z)));

      GridCoordinate3DTemplate<TcoordType, doSignChecks> coord3D_3;
      GridCoordinate3DTemplate<TcoordType, doSignChecks> coord3D_4;
      expandTo3DStartEnd (GridCoordinate1DTemplate<TcoordType, doSignChecks> (i, CoordinateType::X),
                          GridCoordinate1DTemplate<TcoordType, doSignChecks> (10 * i, CoordinateType::X),
                          coord3D_3, coord3D_4, CoordinateType::X, CoordinateType::NONE, CoordinateType::NONE);
      ALWAYS_ASSERT (coord3D_3.get1 () == i && coord3D_3.get2 () == 0 && coord3D_3.get3 () == 0);
      ALWAYS_ASSERT (coord3D_4.get1 () == 10 * i && coord3D_4.get2 () == 1 && coord3D_4.get3 () == 1);

      expandTo3DStartEnd (GridCoordinate1DTemplate<TcoordType, doSignChecks> (i, CoordinateType::Y),
                          GridCoordinate1DTemplate<TcoordType, doSignChecks> (10 * i, CoordinateType::Y),
                          coord3D_3, coord3D_4, CoordinateType::Y, CoordinateType::NONE, CoordinateType::NONE);
      ALWAYS_ASSERT (coord3D_3.get1 () == 0 && coord3D_3.get2 () == i && coord3D_3.get3 () == 0);
      ALWAYS_ASSERT (coord3D_4.get1 () == 1 && coord3D_4.get2 () == 10 * i && coord3D_4.get3 () == 1);

      expandTo3DStartEnd (GridCoordinate1DTemplate<TcoordType, doSignChecks> (i, CoordinateType::Z),
                          GridCoordinate1DTemplate<TcoordType, doSignChecks> (10 * i, CoordinateType::Z),
                          coord3D_3, coord3D_4, CoordinateType::Z, CoordinateType::NONE, CoordinateType::NONE);
      ALWAYS_ASSERT (coord3D_3.get1 () == 0 && coord3D_3.get2 () == 0 && coord3D_3.get3 () == i);
      ALWAYS_ASSERT (coord3D_4.get1 () == 1 && coord3D_4.get2 () == 1 && coord3D_4.get3 () == 10 * i)

      expandTo3DStartEnd (GridCoordinate2DTemplate<TcoordType, doSignChecks> (i, 10 * i, CoordinateType::X, CoordinateType::Y),
                          GridCoordinate2DTemplate<TcoordType, doSignChecks> (10 * i, 17 * i, CoordinateType::X, CoordinateType::Y),
                          coord3D_3, coord3D_4, CoordinateType::X, CoordinateType::Y, CoordinateType::NONE);
      ALWAYS_ASSERT (coord3D_3.get1 () == i && coord3D_3.get2 () == 10 * i && coord3D_3.get3 () == 0);
      ALWAYS_ASSERT (coord3D_4.get1 () == 10 * i && coord3D_4.get2 () == 17 * i && coord3D_4.get3 () == 1);

      expandTo3DStartEnd (GridCoordinate2DTemplate<TcoordType, doSignChecks> (i, 10 * i, CoordinateType::Y, CoordinateType::Z),
                          GridCoordinate2DTemplate<TcoordType, doSignChecks> (10 * i, 17 * i, CoordinateType::Y, CoordinateType::Z),
                          coord3D_3, coord3D_4, CoordinateType::Y, CoordinateType::Z, CoordinateType::NONE);
      ALWAYS_ASSERT (coord3D_3.get1 () == 0 && coord3D_3.get2 () == i && coord3D_3.get3 () == 10 * i);
      ALWAYS_ASSERT (coord3D_4.get1 () == 1 && coord3D_4.get2 () == 10 * i && coord3D_4.get3 () == 17 * i);

      expandTo3DStartEnd (GridCoordinate3DTemplate<TcoordType, doSignChecks> (i, 10 * i, 17 * i, CoordinateType::X, CoordinateType::Y, CoordinateType::Z),
                          GridCoordinate3DTemplate<TcoordType, doSignChecks> (10 * i, 17 * i, 22 * i, CoordinateType::X, CoordinateType::Y, CoordinateType::Z),
                          coord3D_3, coord3D_4, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
      ALWAYS_ASSERT (coord3D_3.get1 () == i && coord3D_3.get2 () == 10 * i && coord3D_3.get3 () == 17 * i);
      ALWAYS_ASSERT (coord3D_4.get1 () == 10 * i && coord3D_4.get2 () == 17 * i && coord3D_4.get3 () == 22 * i);
    }

    if (i > 0)
    {
      GridCoordinate1DTemplate<TcoordType, doSignChecks> pos1D_1 (10 * i, t1);
      GridCoordinate1DTemplate<TcoordType, doSignChecks> pos1D_2 (4 * i, t1);
      GridCoordinate1DTemplate<TcoordType, doSignChecks> pos1D_3 (2 * i, t1);
      GridCoordinate1DTemplate<TcoordType, doSignChecks> pos1D_4 (7 * i, t1);

      ALWAYS_ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::subWithBorder (pos1D_1, pos1D_2, pos1D_4) == pos1D_4));
      ALWAYS_ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::subWithBorder (pos1D_1, pos1D_3, pos1D_4) == (pos1D_1 - pos1D_3)));

      ALWAYS_ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::addWithBorder (pos1D_2, pos1D_4, pos1D_1) == pos1D_1));
      ALWAYS_ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::addWithBorder (pos1D_3, pos1D_4, pos1D_1) == (pos1D_3 + pos1D_4)));

      if (correct2D)
      {
        GridCoordinate2DTemplate<TcoordType, doSignChecks> pos2D_1 (41 * i, 10 * i, t1, t2);
        GridCoordinate2DTemplate<TcoordType, doSignChecks> pos2D_2 (18 * i, 4 * i, t1, t2);
        GridCoordinate2DTemplate<TcoordType, doSignChecks> pos2D_3 (11 * i, 2 * i, t1, t2);
        GridCoordinate2DTemplate<TcoordType, doSignChecks> pos2D_4 (25 * i, 7 * i, t1, t2);

        ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::subWithBorder (pos2D_1, pos2D_2, pos2D_4) == pos2D_4));
        ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::subWithBorder (pos2D_1, pos2D_3, pos2D_4) == (pos2D_1 - pos2D_3)));

        ALWAYS_ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::addWithBorder (pos2D_2, pos2D_4, pos2D_1) == pos2D_1));
        ALWAYS_ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::addWithBorder (pos2D_3, pos2D_4, pos2D_1) == (pos2D_3 + pos2D_4)));
      }

      if (correct3D)
      {
        GridCoordinate3DTemplate<TcoordType, doSignChecks> pos3D_1 (123 * i, 41 * i, 10 * i, t1, t2, t3);
        GridCoordinate3DTemplate<TcoordType, doSignChecks> pos3D_2 (100 * i, 18 * i, 4 * i, t1, t2, t3);
        GridCoordinate3DTemplate<TcoordType, doSignChecks> pos3D_3 (50 * i, 11 * i, 2 * i, t1, t2, t3);
        GridCoordinate3DTemplate<TcoordType, doSignChecks> pos3D_4 (50 * i, 25 * i, 7 * i, t1, t2, t3);

        ALWAYS_ASSERT ((GridCoordinate3DTemplate<TcoordType, doSignChecks>::subWithBorder (pos3D_1, pos3D_2, pos3D_4) == pos3D_4));
        ALWAYS_ASSERT ((GridCoordinate3DTemplate<TcoordType, doSignChecks>::subWithBorder (pos3D_1, pos3D_3, pos3D_4) == (pos3D_1 - pos3D_3)));

        ALWAYS_ASSERT ((GridCoordinate3DTemplate<TcoordType, doSignChecks>::addWithBorder (pos3D_2, pos3D_4, pos3D_1) == pos3D_1));
        ALWAYS_ASSERT ((GridCoordinate3DTemplate<TcoordType, doSignChecks>::addWithBorder (pos3D_3, pos3D_4, pos3D_1) == (pos3D_3 + pos3D_4)));
      }
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
    GridCoordinate1D coord1D (i, CoordinateType::X);
    ALWAYS_ASSERT (convertCoord (convertCoord (coord1D)) == coord1D);

    GridCoordinate2D coord2D (i, i, CoordinateType::X, CoordinateType::Y);
    ALWAYS_ASSERT (convertCoord (convertCoord (coord2D)) == coord2D);

    GridCoordinate3D coord3D (i, i, i, CoordinateType::X, CoordinateType::Y, CoordinateType::Z);
    ALWAYS_ASSERT (convertCoord (convertCoord (coord3D)) == coord3D);
  }

  return 0;
} /* main */
