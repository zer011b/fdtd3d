#ifndef GRID_COORDINATE_3D_H
#define GRID_COORDINATE_3D_H

#include "GridCoordinate2D.h"

template<class TcoordType, bool doSignChecks>
class GridCoordinate3DTemplate;

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator*
(TcoordType lhs, const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+
(GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator-
(GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs);

/**
 * 3-dimensional coordinate in the grid.
 */
template<class TcoordType, bool doSignChecks>
class GridCoordinate3DTemplate: public GridCoordinate2DTemplate<TcoordType, doSignChecks>
{
  TcoordType coord3;

#ifdef DEBUG_INFO
  CoordinateType type3;
#endif /* DEBUG_INFO */

public:

  static const Dimension dimension;

  explicit CUDA_DEVICE CUDA_HOST GridCoordinate3DTemplate<TcoordType, doSignChecks> (const TcoordType& c1 = 0,
                                                                                     const TcoordType& c2 = 0,
                                                                                     const TcoordType& c3 = 0,
                                                                                     CoordinateType t1 = CoordinateType::X,
                                                                                     CoordinateType t2 = CoordinateType::Y,
                                                                                     CoordinateType t3 = CoordinateType::Z)
    : GridCoordinate2DTemplate<TcoordType, doSignChecks> (c1, c2, t1, t2)
    , coord3 (c3)
#ifdef DEBUG_INFO
    , type3 (t3)
#endif /* DEBUG_INFO */
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () < GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()));
    ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 () < getType3 ()));
#endif /* DEBUG_INFO */

    if (doSignChecks)
    {
      TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
      TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
      ASSERT (coord1 >= 0 && coord2 >= 0 && coord3 >= 0);
    }
  }

  CUDA_DEVICE CUDA_HOST GridCoordinate3DTemplate<TcoordType, doSignChecks> (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& pos)
    : GridCoordinate2DTemplate<TcoordType, doSignChecks> (pos.get1 (), pos.get2 ()
#ifdef DEBUG_INFO
    , pos.getType1 ()
    , pos.getType2 ()
#endif /* DEBUG_INFO */
      )
    , coord3 (pos.get3 ())
#ifdef DEBUG_INFO
    , type3 (pos.getType3 ())
#endif /* DEBUG_INFO */
  {
  }

  CUDA_DEVICE CUDA_HOST ~GridCoordinate3DTemplate<TcoordType, doSignChecks> () {}

  // Calculate three-dimensional coordinate.
  TcoordType CUDA_DEVICE CUDA_HOST calculateTotalCoord () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType res = coord1 * coord2;
    if (doSignChecks)
    {
      ASSERT (res >= 0);
    }
    res *= coord3;
    if (doSignChecks)
    {
      ASSERT (res >= 0);
    }
    return res;
  }

  // Get one-dimensional coordinates.
  const TcoordType& CUDA_DEVICE CUDA_HOST get3 () const
  {
    return coord3;
  }

  // Set one-dimensional coordinates.
  void CUDA_DEVICE CUDA_HOST set3 (const TcoordType& new_c3)
  {
    coord3 = new_c3;
    if (doSignChecks)
    {
      ASSERT (coord3 >= 0);
    }
  }

#ifdef DEBUG_INFO
  CoordinateType CUDA_DEVICE CUDA_HOST getType3 () const
  {
    return type3;
  }
#endif /* DEBUG_INFO */

  TcoordType CUDA_DEVICE CUDA_HOST getMax () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    if (coord3 > coord1 && coord3 > coord2)
    {
      return coord3;
    }
    else
    {
      return GridCoordinate2DTemplate<TcoordType, doSignChecks>::getMax ();
    }
  }

  GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+ (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 () == rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()));
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return GridCoordinate3DTemplate (coord1 + rhs_c1, coord2 + rhs_c2, get3 () + rhs.get3 ()
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
      , GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
      , getType3 ()
#endif /* DEBUG_INFO */
      );
  }

  GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator- (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 () == rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()));
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return GridCoordinate3DTemplate (coord1 - rhs_c1, coord2 - rhs_c2, get3 () - rhs.get3 ()
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
      , GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
      , getType3 ()
#endif /* DEBUG_INFO */
      );
  }

  bool CUDA_DEVICE CUDA_HOST operator== (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 () == rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()));
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return coord1 == rhs_c1 && coord2 == rhs_c2 && get3 () == rhs.get3 ();
  }

  bool CUDA_DEVICE CUDA_HOST operator!= (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 () == rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()));
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return coord1 != rhs_c1 || coord2 != rhs_c2 || get3 () == rhs.get3 ();
  }

  bool CUDA_DEVICE CUDA_HOST operator> (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 () == rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()));
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return coord1 > rhs_c1 && coord2 > rhs_c2 && get3 () > rhs.get3 ();
  }

  bool CUDA_DEVICE CUDA_HOST operator< (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 () == rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()));
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return coord1 < rhs_c1 && coord2 < rhs_c2 && get3 () < rhs.get3 ();
  }

  bool CUDA_DEVICE CUDA_HOST CUDA_DEVICE CUDA_HOST operator>= (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 () == rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()));
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return coord1 >= rhs_c1 && coord2 >= rhs_c2 && get3 () >= rhs.get3 ();
  }

  bool CUDA_DEVICE CUDA_HOST operator<= (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT ((GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 () == rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()));
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return coord1 <= rhs_c1 && coord2 <= rhs_c2 && get3 () <= rhs.get3 ();
  }

  GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator* (TcoordType rhs) const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    return GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord1 * rhs, coord2 * rhs, get3 () * rhs
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
      , GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
      , getType3 ()
#endif /* DEBUG_INFO */
      );
  }

  GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator/ (TcoordType rhs) const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    return GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord1 / rhs, coord2 / rhs, get3 () / rhs
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
      , GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
      , getType3 ()
#endif /* DEBUG_INFO */
      );
  }

  static GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST initAxesCoordinate (const TcoordType& c1,
                                                                                               const TcoordType& c2,
                                                                                               const TcoordType& c3,
                                                                                               CoordinateType ct1,
                                                                                               CoordinateType ct2,
                                                                                               CoordinateType ct3)
  {
    ASSERT (ct1 == CoordinateType::X && ct2 == CoordinateType::Y && ct3 == CoordinateType::Z);
    return GridCoordinate3DTemplate<TcoordType, doSignChecks> (c1, c2, c3, ct1, ct2, ct3);
  }

  friend GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator* <TcoordType, doSignChecks>)
    (TcoordType lhs, const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs);
  friend GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator+ <TcoordType, doSignChecks>)
    (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs);
  friend GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator- <TcoordType, doSignChecks>)
    (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs);

  GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST shrink () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    return GridCoordinate2DTemplate<TcoordType, doSignChecks> (coord1, coord2
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
      , GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
#endif /* DEBUG_INFO */
      );
  }

  void print () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    printf ("Coord ("
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      "%f, "
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      "%f, "
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      "%f).\n"
#ifdef DEBUG_INFO
      , coordinateTypeNames[static_cast<uint8_t> (GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ())]
#endif /* DEBUG_INFO */
      , (FPValue) coord1
#ifdef DEBUG_INFO
      , coordinateTypeNames[static_cast<uint8_t> (GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ())]
#endif /* DEBUG_INFO */
      , (FPValue) coord2
#ifdef DEBUG_INFO
      , coordinateTypeNames[static_cast<uint8_t> (getType3 ())]
#endif /* DEBUG_INFO */
      , (FPValue) get3 ());
  }
};

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator* (TcoordType lhs, const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs)
{
  TcoordType coord1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  TcoordType coord2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (lhs * coord1, lhs * coord2, lhs * rhs.get3 ()
#ifdef DEBUG_INFO
    , rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
    , rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
    , rhs.getType3 ()
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+ (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs)
{
#ifdef DEBUG_INFO
  ASSERT ((lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::getType1 ()));
  ASSERT ((lhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 () == rhs.GridCoordinate2DTemplate<TcoordType, !doSignChecks>::getType2 ()));
  ASSERT (lhs.getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
  TcoordType lcoord1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  TcoordType rcoord1 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::get1 ();
  TcoordType lcoord2 = lhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
  TcoordType rcoord2 = rhs.GridCoordinate2DTemplate<TcoordType, !doSignChecks>::get2 ();

  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (lcoord1 + rcoord1, lcoord2 + rcoord2, lhs.get3 () + rhs.get3 ()
#ifdef DEBUG_INFO
    , lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
    , lhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
    , lhs.getType3 ()
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator- (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs)
{
#ifdef DEBUG_INFO
  ASSERT ((lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::getType1 ()));
  ASSERT ((lhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 () == rhs.GridCoordinate2DTemplate<TcoordType, !doSignChecks>::getType2 ()));
  ASSERT (lhs.getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
  TcoordType lcoord1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  TcoordType rcoord1 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::get1 ();
  TcoordType lcoord2 = lhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
  TcoordType rcoord2 = rhs.GridCoordinate2DTemplate<TcoordType, !doSignChecks>::get2 ();

  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (lcoord1 - rcoord1, lcoord2 - rcoord2, lhs.get3 () - rhs.get3 ()
#ifdef DEBUG_INFO
    , lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
    , lhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
    , lhs.getType3 ()
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expand (const GridCoordinate2DTemplate<TcoordType, doSignChecks> &coord,
                                                                                 CoordinateType t3 = CoordinateType::Z)
{
  TcoordType coord1 = coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  TcoordType coord2 = coord.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord1, coord2, 0
#ifdef DEBUG_INFO
    , coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
    , coord.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
    , t3
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expand2 (const GridCoordinate1DTemplate<TcoordType, doSignChecks> &coord,
                                                                                  CoordinateType t2 = CoordinateType::Y,
                                                                                  CoordinateType t3 = CoordinateType::Z)
{
  TcoordType coord1 = coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord1, 0, 0
#ifdef DEBUG_INFO
    , coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
    , t2
    , t3
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expandTo3D (const GridCoordinate1DTemplate<TcoordType, doSignChecks> &coord,
                                                                                     CoordinateType t2 = CoordinateType::Y,
                                                                                     CoordinateType t3 = CoordinateType::Z)
{
  return expand2 (coord, t2, t3);
}

template<class TcoordType, bool doSignChecks>
void CUDA_DEVICE CUDA_HOST expandTo3DStartEnd (const GridCoordinate1DTemplate<TcoordType, doSignChecks> &start,
                                               const GridCoordinate1DTemplate<TcoordType, doSignChecks> &end,
                                               GridCoordinate3DTemplate<TcoordType, doSignChecks> &start3D,
                                               GridCoordinate3DTemplate<TcoordType, doSignChecks> &end3D,
                                               CoordinateType t2 = CoordinateType::Y,
                                               CoordinateType t3 = CoordinateType::Z)
{
#ifdef DEBUG_INFO
  ASSERT (start.getType1 () == end.getType1 ());
#endif /* DEBUG_INFO */

  start3D = expandTo3D (start, t2, t3);

  TcoordType end_1 = end.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();

  end3D = GridCoordinate3DTemplate<TcoordType, doSignChecks> (end_1, 1, 1
#ifdef DEBUG_INFO
    , end.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
    , t2
    , t3
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expandTo3D (const GridCoordinate2DTemplate<TcoordType, doSignChecks> &coord,
                                                                                     CoordinateType t3 = CoordinateType::Z)
{
  return expand (coord, t3);
}

template<class TcoordType, bool doSignChecks>
void CUDA_DEVICE CUDA_HOST expandTo3DStartEnd (const GridCoordinate2DTemplate<TcoordType, doSignChecks> &start,
                                               const GridCoordinate2DTemplate<TcoordType, doSignChecks> &end,
                                               GridCoordinate3DTemplate<TcoordType, doSignChecks> &start3D,
                                               GridCoordinate3DTemplate<TcoordType, doSignChecks> &end3D,
                                               CoordinateType t3 = CoordinateType::Z)
{
#ifdef DEBUG_INFO
  ASSERT (start.getType1 () == end.getType1 ());
  ASSERT (start.getType2 () == end.getType2 ());
#endif /* DEBUG_INFO */

  start3D = expandTo3D (start, t3);

  TcoordType end_1 = end.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  TcoordType end_2 = end.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

  end3D = GridCoordinate3DTemplate<TcoordType, doSignChecks> (end_1, end_2, 1
#ifdef DEBUG_INFO
    , end.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
    , end.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
    , t3
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expandTo3D (const GridCoordinate3DTemplate<TcoordType, doSignChecks> &coord)
{
  return coord;
}

template<class TcoordType, bool doSignChecks>
void CUDA_DEVICE CUDA_HOST expandTo3DStartEnd (const GridCoordinate3DTemplate<TcoordType, doSignChecks> &start,
                                               const GridCoordinate3DTemplate<TcoordType, doSignChecks> &end,
                                               GridCoordinate3DTemplate<TcoordType, doSignChecks> &start3D,
                                               GridCoordinate3DTemplate<TcoordType, doSignChecks> &end3D)
{
  start3D = start;
  end3D = end;
}

typedef GridCoordinate3DTemplate<grid_coord, true> GridCoordinate3D;
typedef GridCoordinate3DTemplate<grid_coord, false> GridCoordinateSigned3D;
typedef GridCoordinate3DTemplate<FPValue, true> GridCoordinateFP3D;
typedef GridCoordinate3DTemplate<FPValue, false> GridCoordinateSignedFP3D;

template<bool doSignChecks>
GridCoordinate3DTemplate<grid_coord, doSignChecks> CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate3DTemplate<FPValue, doSignChecks> coord)
{
  ASSERT (((grid_coord) coord.get1 ()) == coord.get1 ());
  ASSERT (((grid_coord) coord.get2 ()) == coord.get2 ());
  ASSERT (((grid_coord) coord.get3 ()) == coord.get3 ());

  return GridCoordinate3DTemplate<grid_coord, doSignChecks> ((grid_coord) coord.get1 (),
                                                             (grid_coord) coord.get2 (),
                                                             (grid_coord) coord.get3 ()
#ifdef DEBUG_INFO
   , coord.GridCoordinate1DTemplate<FPValue, doSignChecks>::getType1 ()
   , coord.GridCoordinate2DTemplate<FPValue, doSignChecks>::getType2 ()
   , coord.getType3 ()
#endif /* DEBUG_INFO */
   );
}

template<bool doSignChecks>
GridCoordinate3DTemplate<FPValue, doSignChecks> CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate3DTemplate<grid_coord, doSignChecks> coord)
{
  return GridCoordinate3DTemplate<FPValue, doSignChecks> (coord.get1 (),
                                                          coord.get2 (),
                                                          coord.get3 ()
#ifdef DEBUG_INFO
    , coord.GridCoordinate1DTemplate<grid_coord, doSignChecks>::getType1 ()
    , coord.GridCoordinate2DTemplate<grid_coord, doSignChecks>::getType2 ()
    , coord.getType3 ()
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
const Dimension GridCoordinate3DTemplate<TcoordType, doSignChecks>::dimension = Dimension::Dim3;

#endif /* GRID_COORDINATE_3D_H */
