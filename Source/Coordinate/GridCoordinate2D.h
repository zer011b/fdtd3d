#ifndef GRID_COORDINATE_2D_H
#define GRID_COORDINATE_2D_H

#include "GridCoordinate1D.h"

template<class TcoordType, bool doSignChecks>
class GridCoordinate2DTemplate;

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator*
(TcoordType lhs, const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+
(GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator-
(GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs);

/**
 * 2-dimensional coordinate in the grid.
 */
template<class TcoordType, bool doSignChecks>
class GridCoordinate2DTemplate: public GridCoordinate1DTemplate<TcoordType, doSignChecks>
{
protected:

  TcoordType coord2;

#ifdef DEBUG_INFO
  CoordinateType type2;
#endif /* DEBUG_INFO */

public:

  static const Dimension dimension;

  explicit CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate<TcoordType, doSignChecks> (const TcoordType& c1 = 0,
                                                                                     const TcoordType& c2 = 0,
                                                                                     CoordinateType t1 = CoordinateType::X,
                                                                                     CoordinateType t2 = CoordinateType::Y)
    : GridCoordinate1DTemplate<TcoordType, doSignChecks> (c1, t1)
    , coord2 (c2)
#ifdef DEBUG_INFO
    , type2 (t2)
#endif /* DEBUG_INFO */
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () < getType2 ()));
#endif /* DEBUG_INFO */

    if (doSignChecks)
    {
      TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
      ASSERT (coord1 >= 0 && coord2 >= 0);
    }
  }

  explicit CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate<TcoordType, doSignChecks> (const TcoordType& c1,
                                                                                     const TcoordType& c2,
                                                                                     const TcoordType& tmp,
                                                                                     CoordinateType t1 = CoordinateType::X,
                                                                                     CoordinateType t2 = CoordinateType::Y,
                                                                                     CoordinateType t3 = CoordinateType::Z)
    : GridCoordinate1DTemplate<TcoordType, doSignChecks> (c1, t1)
    , coord2 (c2)
#ifdef DEBUG_INFO
    , type2 (t2)
#endif /* DEBUG_INFO */
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () < getType2 ()));
#endif /* DEBUG_INFO */

    if (doSignChecks)
    {
      TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
      ASSERT (coord1 >= 0 && coord2 >= 0);
    }
  }

  CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate<TcoordType, doSignChecks> (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& pos)
    : GridCoordinate1DTemplate<TcoordType, doSignChecks> (pos.get1 ()
#ifdef DEBUG_INFO
      , pos.getType1 ()
#endif /* DEBUG_INFO */
      )
    , coord2 (pos.get2 ())
#ifdef DEBUG_INFO
    , type2 (pos.getType2 ())
#endif /* DEBUG_INFO */
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () < getType2 ()));
#endif /* DEBUG_INFO */

    if (doSignChecks)
    {
      TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
      ASSERT (coord1 >= 0 && coord2 >= 0);
    }
  }

  CUDA_DEVICE CUDA_HOST ~GridCoordinate2DTemplate<TcoordType, doSignChecks> () {}

  // Calculate three-dimensional coordinate.
  TcoordType CUDA_DEVICE CUDA_HOST calculateTotalCoord () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType res = coord1 * coord2;
    if (doSignChecks)
    {
      ASSERT (res >= 0);
    }
    return res;
  }

  // Get one-dimensional coordinates.
  const TcoordType& CUDA_DEVICE CUDA_HOST get2 () const
  {
    return coord2;
  }

  // Set one-dimensional coordinates.
  void CUDA_DEVICE CUDA_HOST set2 (const TcoordType& new_c2)
  {
    coord2 = new_c2;
    if (doSignChecks)
    {
      ASSERT (coord2 >= 0);
    }
  }

#ifdef DEBUG_INFO
  CoordinateType CUDA_DEVICE CUDA_HOST getType2 () const
  {
    return type2;
  }
#endif /* DEBUG_INFO */

  TcoordType CUDA_DEVICE CUDA_HOST getMax () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return coord1 > coord2 ? coord1 : coord2;
  }

  GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+ (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();

    return GridCoordinate2DTemplate (coord1 + rhs_c1, get2 () + rhs.get2 ()
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), getType2 ()
#endif /* DEBUG_INFO */
      );
  }

  GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator- (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();

    return GridCoordinate2DTemplate (coord1 - rhs_c1, get2 () - rhs.get2 ()
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), getType2 ()
#endif /* DEBUG_INFO */
      );
  }

  bool CUDA_DEVICE CUDA_HOST operator== (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return coord1 == rhs_c1 && get2 () == rhs.get2 ();
  }

  bool CUDA_DEVICE CUDA_HOST operator!= (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return coord1 != rhs_c1 || get2 () != rhs.get2 ();
  }

  bool CUDA_DEVICE CUDA_HOST operator> (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return coord1 > rhs_c1 && get2 () > rhs.get2 ();
  }

  bool CUDA_DEVICE CUDA_HOST operator< (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return coord1 < rhs_c1 && get2 () < rhs.get2 ();
  }

  bool CUDA_DEVICE CUDA_HOST operator>= (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return coord1 >= rhs_c1 && get2 () >= rhs.get2 ();
  }

  bool CUDA_DEVICE CUDA_HOST operator<= (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()));
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return coord1 <= rhs_c1 && get2 () <= rhs.get2 ();
  }

  GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator* (TcoordType rhs) const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return GridCoordinate2DTemplate<TcoordType, doSignChecks> (coord1 * rhs, get2 () * rhs
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), getType2 ()
#endif /* DEBUG_INFO */
      );
  }

  GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator/ (TcoordType rhs) const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return GridCoordinate2DTemplate<TcoordType, doSignChecks> (coord1 / rhs, get2 () / rhs
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), getType2 ()
#endif /* DEBUG_INFO */
      );
  }

  static GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST initAxesCoordinate (const TcoordType& c1,
                                                                                               const TcoordType& c2,
                                                                                               const TcoordType& c3,
                                                                                               CoordinateType ct1,
                                                                                               CoordinateType ct2,
                                                                                               CoordinateType ct3)
  {
    ASSERT (ct3 == CoordinateType::NONE);
    if (ct1 == CoordinateType::X && ct2 == CoordinateType::Y)
    {
      return GridCoordinate2DTemplate<TcoordType, doSignChecks> (c1, c2, ct1, ct2);
    }
    else if (ct1 == CoordinateType::X && ct2 == CoordinateType::Z)
    {
      return GridCoordinate2DTemplate<TcoordType, doSignChecks> (c1, c3, ct1, ct2);
    }
    else if (ct1 == CoordinateType::Y && ct2 == CoordinateType::Z)
    {
      return GridCoordinate2DTemplate<TcoordType, doSignChecks> (c2, c3, ct1, ct2);
    }
    else
    {
      UNREACHABLE;
    }
  }

  friend GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator* <TcoordType, doSignChecks>)
    (TcoordType lhs, const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs);
  friend GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator+ <TcoordType, doSignChecks>)
    (GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs);
  friend GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator- <TcoordType, doSignChecks>)
    (GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs);

  GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST shrink () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (coord1
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
#endif /* DEBUG_INFO */
      );
  }

  void print () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    printf ("Coord ("
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      "%f, "
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      "%f ).\n"
#ifdef DEBUG_INFO
      , coordinateTypeNames[static_cast<uint8_t> (GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ())]
#endif /* DEBUG_INFO */
      , (FPValue) coord1
#ifdef DEBUG_INFO
      , coordinateTypeNames[static_cast<uint8_t> (getType2 ())]
#endif /* DEBUG_INFO */
      , (FPValue) get2 ());
  }
};

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator* (TcoordType lhs, const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs)
{
  TcoordType coord1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  return GridCoordinate2DTemplate<TcoordType, doSignChecks> (lhs * coord1, lhs * rhs.get2 ()
#ifdef DEBUG_INFO
    , rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), rhs.getType2 ()
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+ (GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs)
{
#ifdef DEBUG_INFO
  ASSERT ((lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::getType1 ()));
  ASSERT (lhs.getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
  TcoordType lcoord1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  TcoordType rcoord1 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::get1 ();
  return GridCoordinate2DTemplate<TcoordType, doSignChecks> (lcoord1 + rcoord1, lhs.get2 () + rhs.get2 ()
#ifdef DEBUG_INFO
    , lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), lhs.getType2 ()
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator- (GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs)
{
#ifdef DEBUG_INFO
  ASSERT ((lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::getType1 ()));
  ASSERT (lhs.getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
  TcoordType lcoord1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  TcoordType rcoord1 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::get1 ();
  return GridCoordinate2DTemplate<TcoordType, doSignChecks> (lcoord1 - rcoord1, lhs.get2 () - rhs.get2 ()
#ifdef DEBUG_INFO
    , lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), lhs.getType2 ()
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expand (const GridCoordinate1DTemplate<TcoordType, doSignChecks> &coord, CoordinateType t2 = CoordinateType::Y)
{
  TcoordType coord1 = coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  return GridCoordinate2DTemplate<TcoordType, doSignChecks> (coord1, 0
#ifdef DEBUG_INFO
    , coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), t2
#endif /* DEBUG_INFO */
    );
}

typedef GridCoordinate2DTemplate<grid_coord, true> GridCoordinate2D;
typedef GridCoordinate2DTemplate<grid_coord, false> GridCoordinateSigned2D;
typedef GridCoordinate2DTemplate<FPValue, true> GridCoordinateFP2D;
typedef GridCoordinate2DTemplate<FPValue, false> GridCoordinateSignedFP2D;

template<bool doSignChecks>
GridCoordinate2DTemplate<grid_coord, doSignChecks> CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate2DTemplate<FPValue, doSignChecks> coord)
{
  ASSERT (((grid_coord) coord.get1 ()) == coord.get1 ());
  ASSERT (((grid_coord) coord.get2 ()) == coord.get2 ());

  return GridCoordinate2DTemplate<grid_coord, doSignChecks> ((grid_coord) coord.get1 (), (grid_coord) coord.get2 ()
#ifdef DEBUG_INFO
    , coord.GridCoordinate1DTemplate<FPValue, doSignChecks>::getType1 (), coord.getType2 ()
#endif /* DEBUG_INFO */
    );
}

template<bool doSignChecks>
GridCoordinate2DTemplate<FPValue, doSignChecks> CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate2DTemplate<grid_coord, doSignChecks> coord)
{
  return GridCoordinate2DTemplate<FPValue, doSignChecks> (coord.get1 (), coord.get2 ()
#ifdef DEBUG_INFO
    , coord.GridCoordinate1DTemplate<grid_coord, doSignChecks>::getType1 (), coord.getType2 ()
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
const Dimension GridCoordinate2DTemplate<TcoordType, doSignChecks>::dimension = Dimension::Dim2;

#endif /* GRID_COORDINATE_2D_H */
