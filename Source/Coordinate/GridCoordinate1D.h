#ifndef GRID_COORDINATE_1D_H
#define GRID_COORDINATE_1D_H

#include "FieldValue.h"
#include "Settings.h"

template<class TcoordType>
class GridCoordinate1DTemplate;

template<class TcoordType>
GridCoordinate1DTemplate<TcoordType> CUDA_DEVICE CUDA_HOST operator* (TcoordType lhs, const GridCoordinate1DTemplate<TcoordType>& rhs);

/**
 * 1-dimensional coordinate in the grid
 */
template<class TcoordType>
class GridCoordinate1DTemplate
{
protected:

  TcoordType x;

public:

  // Constructor for all cases.
  explicit CUDA_DEVICE CUDA_HOST GridCoordinate1DTemplate (const TcoordType& cx = 0)
    : x (cx) {}

  CUDA_DEVICE CUDA_HOST GridCoordinate1DTemplate (const GridCoordinate1DTemplate& coord)
    : x (coord.getX ()) {}

  CUDA_DEVICE CUDA_HOST ~GridCoordinate1DTemplate () {};

  // Calculate total-dimensional coordinate.
  grid_iter CUDA_DEVICE CUDA_HOST calculateTotalCoord () const
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

  GridCoordinate1DTemplate<TcoordType> CUDA_DEVICE CUDA_HOST operator- () const
  {
    return GridCoordinate1DTemplate<TcoordType> (- getX ());
  }

  GridCoordinate1DTemplate<TcoordType> CUDA_DEVICE CUDA_HOST operator* (TcoordType rhs) const
  {
    return GridCoordinate1DTemplate<TcoordType> (getX () * rhs);
  }

  friend GridCoordinate1DTemplate<TcoordType> CUDA_DEVICE CUDA_HOST (::operator* <TcoordType>) (TcoordType lhs, const GridCoordinate1DTemplate<TcoordType>& rhs);

  void print () const
  {
    printf ("Coord (%lu).\n", getX ());
  }
};

template<class TcoordType>
GridCoordinate1DTemplate<TcoordType> CUDA_DEVICE CUDA_HOST operator* (TcoordType lhs, const GridCoordinate1DTemplate<TcoordType>& rhs)
{
  return GridCoordinate1DTemplate<TcoordType> (lhs * rhs.getX ());
}

typedef GridCoordinate1DTemplate<grid_iter> GridCoordinate1D;
typedef GridCoordinate1DTemplate<FPValue> GridCoordinateFP1D;

GridCoordinate1D CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinateFP1D coord);
GridCoordinateFP1D CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate1D coord);

#endif /* GRID_COORDINATE_1D_H */
