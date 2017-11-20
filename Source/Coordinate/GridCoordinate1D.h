#ifndef GRID_COORDINATE_1D_H
#define GRID_COORDINATE_1D_H

#include "FieldValue.h"
#include "Settings.h"

ENUM_CLASS (Dimension, uint8_t,
  Dim1,
  Dim2,
  Dim3
);

template<class TcoordType, bool doSignChecks>
class GridCoordinate1DTemplate;

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator* (TcoordType lhs, const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+ (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator- (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs);

/**
 * 1-dimensional coordinate in the grid
 */
template<class TcoordType, bool doSignChecks>
class GridCoordinate1DTemplate
{
protected:

  TcoordType x;

public:

  static const Dimension dimension;

  // Constructor for all cases.
  explicit CUDA_DEVICE CUDA_HOST GridCoordinate1DTemplate (const TcoordType& cx = 0)
    : x (cx)
  {
    if (doSignChecks)
    {
      ASSERT (x >= 0);
    }
  }

  explicit CUDA_DEVICE CUDA_HOST GridCoordinate1DTemplate (const TcoordType& cx,
                                                           const TcoordType& tmp1,
                                                           const TcoordType& tmp2)
    : GridCoordinate1DTemplate (cx)
  {
  }

  CUDA_DEVICE CUDA_HOST GridCoordinate1DTemplate (const GridCoordinate1DTemplate& coord)
    : x (coord.getX ())
  {
    if (doSignChecks)
    {
      ASSERT (x >= 0);
    }
  }

  CUDA_DEVICE CUDA_HOST ~GridCoordinate1DTemplate () {};

  // Calculate total-dimensional coordinate.
  TcoordType CUDA_DEVICE CUDA_HOST calculateTotalCoord () const
  {
    return x;
  }

  // Get one-dimensional coordinates.
  const TcoordType& CUDA_DEVICE CUDA_HOST getX () const
  {
    return x;
  }

  // Set one-dimensional coordinates.
  void CUDA_DEVICE CUDA_HOST setX (const TcoordType& new_x)
  {
    x = new_x;
    if (doSignChecks)
    {
      ASSERT (x >= 0);
    }
  }

  TcoordType CUDA_DEVICE CUDA_HOST getMax () const
  {
    return x;
  }

  GridCoordinate1DTemplate CUDA_DEVICE CUDA_HOST operator+ (const GridCoordinate1DTemplate& rhs) const
  {
    return GridCoordinate1DTemplate (getX () + rhs.getX ());
  }

  GridCoordinate1DTemplate CUDA_DEVICE CUDA_HOST operator- (const GridCoordinate1DTemplate& rhs) const
  {
    return GridCoordinate1DTemplate (getX () - rhs.getX ());
  }

  bool CUDA_DEVICE CUDA_HOST operator== (const GridCoordinate1DTemplate& rhs) const
  {
    return getX () == rhs.getX ();
  }

  bool CUDA_DEVICE CUDA_HOST operator!= (const GridCoordinate1DTemplate& rhs) const
  {
    return getX () != rhs.getX ();
  }

  bool CUDA_DEVICE CUDA_HOST operator> (const GridCoordinate1DTemplate& rhs) const
  {
    return getX () > rhs.getX ();
  }

  bool CUDA_DEVICE CUDA_HOST operator< (const GridCoordinate1DTemplate& rhs) const
  {
    return getX () < rhs.getX ();
  }

  bool CUDA_DEVICE CUDA_HOST operator>= (const GridCoordinate1DTemplate& rhs) const
  {
    return getX () >= rhs.getX ();
  }

  bool CUDA_DEVICE CUDA_HOST operator<= (const GridCoordinate1DTemplate& rhs) const
  {
    return getX () <= rhs.getX ();
  }

  GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator* (TcoordType rhs) const
  {
    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (getX () * rhs);
  }

  friend GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator* <TcoordType, doSignChecks>) (TcoordType lhs, const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs);
  friend GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator+ <TcoordType, doSignChecks>) (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs);
  friend GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator- <TcoordType, doSignChecks>) (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs);

  void print () const
  {
    printf ("Coord (" COORD_MOD ").\n", getX ());
  }
};

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator* (TcoordType lhs, const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs)
{
  return GridCoordinate1DTemplate<TcoordType, doSignChecks> (lhs * rhs.getX ());
}

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+ (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs)
{
  return GridCoordinate1DTemplate<TcoordType, doSignChecks> (lhs.getX () + rhs.getX ());
}

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator- (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs)
{
  return GridCoordinate1DTemplate<TcoordType, doSignChecks> (lhs.getX () - rhs.getX ());
}

typedef GridCoordinate1DTemplate<grid_coord, true> GridCoordinate1D;
typedef GridCoordinate1DTemplate<grid_coord, false> GridCoordinateSigned1D;
typedef GridCoordinate1DTemplate<FPValue, true> GridCoordinateFP1D;
typedef GridCoordinate1DTemplate<FPValue, false> GridCoordinateSignedFP1D;

template<bool doSignChecks>
GridCoordinate1DTemplate<grid_coord, doSignChecks> CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate1DTemplate<FPValue, doSignChecks> coord)
{
  ASSERT (((grid_coord) coord.getX ()) == coord.getX ());

  return GridCoordinate1DTemplate<grid_coord, doSignChecks> ((grid_coord) coord.getX ());
}

template<bool doSignChecks>
GridCoordinate1DTemplate<FPValue, doSignChecks> CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate1DTemplate<grid_coord, doSignChecks> coord)
{
  return GridCoordinate1DTemplate<FPValue, doSignChecks> (coord.getX ());
}

#endif /* GRID_COORDINATE_1D_H */
