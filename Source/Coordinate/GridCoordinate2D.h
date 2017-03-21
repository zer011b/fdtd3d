#ifndef GRID_COORDINATE_2D_H
#define GRID_COORDINATE_2D_H

#include "GridCoordinate1D.h"

// Coordinate in the grid.
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

  // Constructor for case when grid has the same dimension for all axes.
  explicit CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate (const TcoordType& cxy)
    : GridCoordinate1DTemplate<TcoordType> (cxy), y (cxy) {}

  CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate (const GridCoordinate2DTemplate& pos)
    : GridCoordinate1DTemplate<TcoordType> (pos.getX ()), y (pos.getY ()) {}

  CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate (const GridCoordinate1DTemplate<TcoordType>& pos)
    : GridCoordinate1DTemplate<TcoordType> (pos), y (0) {}

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

  GridCoordinate2DTemplate CUDA_DEVICE CUDA_HOST operator- () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    return GridCoordinate2DTemplate (- x, - getY ());
  }
};

typedef GridCoordinate2DTemplate<grid_iter> GridCoordinate2D;
typedef GridCoordinate2DTemplate<FPValue> GridCoordinateFP2D;

#endif /* GRID_COORDINATE_2D_H */
