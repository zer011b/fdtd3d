#ifndef GRID_COORDINATE_1D_H
#define GRID_COORDINATE_1D_H

#include "FieldValue.h"
#include "Settings.h"

ENUM_CLASS (Dimension, uint8_t,
  Dim1,
  Dim2,
  Dim3
);

ENUM_CLASS (CoordinateType, uint8_t,
  NONE,
  X,
  Y,
  Z,
  COUNT
);

extern const char * coordinateTypeNames[];

template<class TcoordType, bool doSignChecks>
class GridCoordinate1DTemplate;

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
operator* (TcoordType lhs, const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
operator+ (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
operator- (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs);

/**
 * 1-dimensional coordinate in the grid
 */
template<class TcoordType, bool doSignChecks>
class GridCoordinate1DTemplate
{
protected:

  TcoordType coord1;

#ifdef DEBUG_INFO
  CoordinateType type1;
#endif /* DEBUG_INFO */

public:

  static const Dimension dimension;

  // Constructor for all cases.
  explicit CUDA_DEVICE CUDA_HOST GridCoordinate1DTemplate<TcoordType, doSignChecks> (const TcoordType& c1 = 0,
                                                                                     CoordinateType t1 = CoordinateType::X)
    : coord1 (c1)
#ifdef DEBUG_INFO
    , type1 (t1)
#endif /* DEBUG_INFO */
  {
    if (doSignChecks)
    {
      ASSERT (coord1 >= 0);
    }
  }

  explicit CUDA_DEVICE CUDA_HOST GridCoordinate1DTemplate<TcoordType, doSignChecks> (const TcoordType& c1,
                                                                                     const TcoordType& tmp1,
                                                                                     const TcoordType& tmp2,
                                                                                     CoordinateType t1 = CoordinateType::X,
                                                                                     CoordinateType t2 = CoordinateType::Y,
                                                                                     CoordinateType t3 = CoordinateType::Z)
    : coord1 (c1)
#ifdef DEBUG_INFO
    , type1 (t1)
#endif /* DEBUG_INFO */
  {
    if (doSignChecks)
    {
      ASSERT (coord1 >= 0);
    }
  }

  CUDA_DEVICE CUDA_HOST GridCoordinate1DTemplate<TcoordType, doSignChecks> (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& pos)
    : coord1 (pos.get1 ())
#ifdef DEBUG_INFO
    , type1 (pos.getType1 ())
#endif /* DEBUG_INFO */
  {
    if (doSignChecks)
    {
      ASSERT (coord1 >= 0);
    }
  }

  CUDA_DEVICE CUDA_HOST ~GridCoordinate1DTemplate<TcoordType, doSignChecks> () {};

  // Calculate total-dimensional coordinate.
  TcoordType CUDA_DEVICE CUDA_HOST calculateTotalCoord () const
  {
    return coord1;
  }

  // Get one-dimensional coordinates.
  const TcoordType& CUDA_DEVICE CUDA_HOST get1 () const
  {
    return coord1;
  }

  // Set one-dimensional coordinates.
  void CUDA_DEVICE CUDA_HOST set1 (const TcoordType& new_c1)
  {
    coord1 = new_c1;
    if (doSignChecks)
    {
      ASSERT (coord1 >= 0);
    }
  }

#ifdef DEBUG_INFO
  CoordinateType CUDA_DEVICE CUDA_HOST getType1 () const
  {
    return type1;
  }
#endif /* DEBUG_INFO */

  TcoordType CUDA_DEVICE CUDA_HOST getMax () const
  {
    return coord1;
  }

  GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+ (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (get1 () + rhs.get1 ()
#ifdef DEBUG_INFO
      , getType1 ()
#endif /* DEBUG_INFO */
      );
  }

  GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator- (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (get1 () - rhs.get1 ()
#ifdef DEBUG_INFO
      , getType1 ()
#endif /* DEBUG_INFO */
      );
  }

  bool CUDA_DEVICE CUDA_HOST operator== (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return get1 () == rhs.get1 ();
  }

  bool CUDA_DEVICE CUDA_HOST operator!= (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return get1 () != rhs.get1 ();
  }

  bool CUDA_DEVICE CUDA_HOST operator> (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return get1 () > rhs.get1 ();
  }

  bool CUDA_DEVICE CUDA_HOST operator< (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return get1 () < rhs.get1 ();
  }

  bool CUDA_DEVICE CUDA_HOST operator>= (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return get1 () >= rhs.get1 ();
  }

  bool CUDA_DEVICE CUDA_HOST operator<= (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return get1 () <= rhs.get1 ();
  }

  GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator* (TcoordType rhs) const
  {
    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (get1 () * rhs
#ifdef DEBUG_INFO
      , getType1 ()
#endif /* DEBUG_INFO */
      );
  }

  GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator/ (TcoordType rhs) const
  {
    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (get1 () / rhs
#ifdef DEBUG_INFO
      , getType1 ()
#endif /* DEBUG_INFO */
      );
  }

  static GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST initAxesCoordinate (const TcoordType& c1,
                                                                                               const TcoordType& c2,
                                                                                               const TcoordType& c3,
                                                                                               CoordinateType ct1,
                                                                                               CoordinateType ct2,
                                                                                               CoordinateType ct3)
  {
    ASSERT (ct2 == CoordinateType::NONE && ct3 == CoordinateType::NONE);

    if (ct1 == CoordinateType::X)
    {
      return GridCoordinate1DTemplate<TcoordType, doSignChecks> (c1, ct1);
    }
    else if (ct1 == CoordinateType::Y)
    {
      return GridCoordinate1DTemplate<TcoordType, doSignChecks> (c2, ct1);
    }
    else if (ct1 == CoordinateType::Z)
    {
      return GridCoordinate1DTemplate<TcoordType, doSignChecks> (c3, ct1);
    }
    else
    {
      UNREACHABLE;
    }
  }

  friend GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator* <TcoordType, doSignChecks>)
    (TcoordType lhs, const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs);
  friend GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator+ <TcoordType, doSignChecks>)
    (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs);
  friend GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator- <TcoordType, doSignChecks>)
    (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs);

  void print () const
  {
    printf ("Coord ("
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      "%f ).\n"
#ifdef DEBUG_INFO
      , coordinateTypeNames[static_cast<uint8_t> (getType1 ())]
#endif /* DEBUG_INFO */
      , (FPValue) get1 ());
  }
};

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator*
(TcoordType lhs, const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs)
{
  return GridCoordinate1DTemplate<TcoordType, doSignChecks> (lhs * rhs.get1 ()
#ifdef DEBUG_INFO
    , rhs.getType1 ()
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+
(GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs)
{
#ifdef DEBUG_INFO
  ASSERT (lhs.getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
  return GridCoordinate1DTemplate<TcoordType, doSignChecks> (lhs.get1 () + rhs.get1 ()
#ifdef DEBUG_INFO
    , lhs.getType1 ()
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator-
(GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs)
{
#ifdef DEBUG_INFO
  ASSERT (lhs.getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
  return GridCoordinate1DTemplate<TcoordType, doSignChecks> (lhs.get1 () - rhs.get1 ()
#ifdef DEBUG_INFO
    , lhs.getType1 ()
#endif /* DEBUG_INFO */
    );
}

typedef GridCoordinate1DTemplate<grid_coord, true> GridCoordinate1D;
typedef GridCoordinate1DTemplate<grid_coord, false> GridCoordinateSigned1D;
typedef GridCoordinate1DTemplate<FPValue, true> GridCoordinateFP1D;
typedef GridCoordinate1DTemplate<FPValue, false> GridCoordinateSignedFP1D;

template<bool doSignChecks>
GridCoordinate1DTemplate<grid_coord, doSignChecks> CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate1DTemplate<FPValue, doSignChecks> coord)
{
  ASSERT (((grid_coord) coord.get1 ()) == coord.get1 ());
  return GridCoordinate1DTemplate<grid_coord, doSignChecks> ((grid_coord) coord.get1 ()
#ifdef DEBUG_INFO
    , coord.getType1 ()
#endif /* DEBUG_INFO */
    );
}

template<bool doSignChecks>
GridCoordinate1DTemplate<FPValue, doSignChecks> CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate1DTemplate<grid_coord, doSignChecks> coord)
{
  return GridCoordinate1DTemplate<FPValue, doSignChecks> (coord.get1 ()
#ifdef DEBUG_INFO
    , coord.getType1 ()
#endif /* DEBUG_INFO */
    );
}

template<class TcoordType, bool doSignChecks>
const Dimension GridCoordinate1DTemplate<TcoordType, doSignChecks>::dimension = Dimension::Dim1;

#endif /* GRID_COORDINATE_1D_H */
