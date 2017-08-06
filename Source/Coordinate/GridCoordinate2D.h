#ifndef GRID_COORDINATE_2D_H
#define GRID_COORDINATE_2D_H

#include "GridCoordinate1D.h"

template<class TcoordType>
class GridCoordinate2DTemplate;

template<class TcoordType>
GridCoordinate2DTemplate<TcoordType> CUDA_DEVICE CUDA_HOST operator* (TcoordType lhs, const GridCoordinate2DTemplate<TcoordType>& rhs);

/**
 * 2-dimensional coordinate in the grid.
 */
template<class TcoordType>
class GridCoordinate2DTemplate: public GridCoordinate1DTemplate<TcoordType>
{
protected:

  TcoordType y;

public:

  CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate ()
    : GridCoordinate1DTemplate<TcoordType> (), y (0) {}

  explicit CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate (const TcoordType& cx, const TcoordType& cy)
    : GridCoordinate1DTemplate<TcoordType> (cx), y (cy) {}

  CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate (const GridCoordinate2DTemplate& pos)
    : GridCoordinate1DTemplate<TcoordType> (pos.getX ()), y (pos.getY ()) {}

  CUDA_DEVICE CUDA_HOST ~GridCoordinate2DTemplate () {}

  // Calculate three-dimensional coordinate.
  grid_iter CUDA_DEVICE CUDA_HOST calculateTotalCoord () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    return x * y;
  }

  // Get one-dimensional coordinates.
  const TcoordType& CUDA_DEVICE CUDA_HOST getY () const
  {
    return y;
  }

  // Set one-dimensional coordinates.
  void CUDA_DEVICE CUDA_HOST setY (const TcoordType& new_y)
  {
    y = new_y;
  }

  TcoordType CUDA_DEVICE CUDA_HOST getMax () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    return x > y ? x : y;
  }

  GridCoordinate2DTemplate CUDA_DEVICE CUDA_HOST operator+ (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return GridCoordinate2DTemplate (x + rhs_x, getY () + rhs.getY ());
  }

  GridCoordinate2DTemplate CUDA_DEVICE CUDA_HOST operator- (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return GridCoordinate2DTemplate (x - rhs_x, getY () - rhs.getY ());
  }

  bool CUDA_DEVICE CUDA_HOST operator== (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return x == rhs_x && getY () == rhs.getY ();
  }

  bool CUDA_DEVICE CUDA_HOST operator!= (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return x != rhs_x || getY () != rhs.getY ();
  }

  bool CUDA_DEVICE CUDA_HOST operator> (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return x > rhs_x && getY () > rhs.getY ();
  }

  bool CUDA_DEVICE CUDA_HOST operator< (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return x < rhs_x && getY () < rhs.getY ();
  }

  bool CUDA_DEVICE CUDA_HOST operator>= (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return x >= rhs_x && getY () >= rhs.getY ();
  }

  bool CUDA_DEVICE CUDA_HOST operator<= (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return x <= rhs_x && getY () <= rhs.getY ();
  }

  GridCoordinate2DTemplate<TcoordType> CUDA_DEVICE CUDA_HOST operator- () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    return GridCoordinate2DTemplate<TcoordType> (- x, - getY ());
  }

  GridCoordinate2DTemplate<TcoordType> CUDA_DEVICE CUDA_HOST operator* (TcoordType rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    return GridCoordinate2DTemplate<TcoordType> (x * rhs, getY () * rhs);
  }

  friend GridCoordinate2DTemplate<TcoordType> CUDA_DEVICE CUDA_HOST (::operator* <TcoordType>) (TcoordType lhs, const GridCoordinate2DTemplate<TcoordType>& rhs);

  GridCoordinate1DTemplate<TcoordType> CUDA_DEVICE CUDA_HOST shrink () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    return GridCoordinate1DTemplate<TcoordType> (x);
  }

  void print () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    printf ("Coord (%lu,%lu).\n", x, getY ());
  }
};

template<class TcoordType>
GridCoordinate2DTemplate<TcoordType> CUDA_DEVICE CUDA_HOST operator* (TcoordType lhs, const GridCoordinate2DTemplate<TcoordType>& rhs)
{
  TcoordType x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
  return GridCoordinate2DTemplate<TcoordType> (lhs * x, lhs * rhs.getY ());
}

template<class TcoordType>
GridCoordinate2DTemplate<TcoordType> CUDA_DEVICE CUDA_HOST expand (const GridCoordinate1DTemplate<TcoordType> &coord)
{
  TcoordType x = coord.GridCoordinate1DTemplate<TcoordType>::getX ();
  return GridCoordinate2DTemplate<TcoordType> (x, 0);
}

typedef GridCoordinate2DTemplate<grid_iter> GridCoordinate2D;
typedef GridCoordinate2DTemplate<FPValue> GridCoordinateFP2D;

GridCoordinate2D CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinateFP2D coord);
GridCoordinateFP2D CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate2D coord);

#endif /* GRID_COORDINATE_2D_H */
