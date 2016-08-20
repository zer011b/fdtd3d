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

  GridCoordinate2DTemplate ()
    : GridCoordinate1DTemplate<TcoordType> (), y (0) {}

  explicit GridCoordinate2DTemplate (const TcoordType& cx, const TcoordType& cy)
    : GridCoordinate1DTemplate<TcoordType> (cx), y (cy) {}

  // Constructor for case when grid has the same dimension for all axes.
  explicit GridCoordinate2DTemplate (const TcoordType& cxy)
    : GridCoordinate1DTemplate<TcoordType> (cxy), y (cxy) {}

  GridCoordinate2DTemplate (const GridCoordinate2DTemplate& pos)
    : GridCoordinate1DTemplate<TcoordType> (pos.getX ()), y (pos.getY ()) {}

  ~GridCoordinate2DTemplate () {}

  // Calculate three-dimensional coordinate.
  grid_iter calculateTotalCoord () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    return x * y;
  }

  // Get one-dimensional coordinates.
  const TcoordType& getY () const
  {
    return y;
  }

  // Set one-dimensional coordinates.
  void setY (const TcoordType& new_y)
  {
    y = new_y;
  }

  TcoordType getMax () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    return x > y ? x : y;
  }

  GridCoordinate2DTemplate operator+ (const GridCoordinate2DTemplate& rhs)
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return GridCoordinate2DTemplate (x + rhs_x, getY () + rhs.getY ());
  }

  GridCoordinate2DTemplate operator- (const GridCoordinate2DTemplate& rhs)
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return GridCoordinate2DTemplate (x - rhs_x, getY () - rhs.getY ());
  }

  bool operator== (const GridCoordinate2DTemplate& rhs)
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return x == rhs_x && getY () == rhs.getY ();
  }

  bool operator!= (const GridCoordinate2DTemplate& rhs)
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return x != rhs_x || getY () != rhs.getY ();
  }

  bool operator> (const GridCoordinate2DTemplate& rhs)
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return x > rhs_x && getY () > rhs.getY ();
  }

  bool operator< (const GridCoordinate2DTemplate& rhs)
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return x < rhs_x && getY () < rhs.getY ();
  }

  bool operator>= (const GridCoordinate2DTemplate& rhs)
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return x >= rhs_x && getY () >= rhs.getY ();
  }

  bool operator<= (const GridCoordinate2DTemplate& rhs)
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    return x <= rhs_x && getY () <= rhs.getY ();
  }
};

typedef GridCoordinate2DTemplate<grid_iter> GridCoordinate2D;
typedef GridCoordinate2DTemplate<FieldValue> GridCoordinateFP2D;

#endif /* GRID_COORDINATE_2D_H */
