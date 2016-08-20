#ifndef GRID_COORDINATE_1D_H
#define GRID_COORDINATE_1D_H

#include "FieldValue.h"

// Coordinate in the grid.
template<class TcoordType>
class GridCoordinate1DTemplate
{
protected:

  TcoordType x;

public:

  // Constructor for all cases.
  explicit GridCoordinate1DTemplate (const TcoordType& cx = 0)
    : x (cx) {}
  GridCoordinate1DTemplate (const GridCoordinate1DTemplate& coord)
    : x (coord.getX ()) {}

  ~GridCoordinate1DTemplate () {};

  // Calculate total-dimensional coordinate.
  grid_iter calculateTotalCoord () const
  {
    return x;
  }

  // Get one-dimensional coordinates.
  const TcoordType& getX () const
  {
    return x;
  }

  // Set one-dimensional coordinates.
  void setX (const TcoordType& new_x)
  {
    x = new_x;
  }

  TcoordType getMax () const
  {
    return x;
  }

  GridCoordinate1DTemplate operator+ (const GridCoordinate1DTemplate& rhs)
  {
    return GridCoordinate1DTemplate (getX () + rhs.getX ());
  }

  GridCoordinate1DTemplate operator- (const GridCoordinate1DTemplate& rhs)
  {
    return GridCoordinate1DTemplate (getX () - rhs.getX ());
  }

  bool operator== (const GridCoordinate1DTemplate& rhs)
  {
    return getX () == rhs.getX ();
  }

  bool operator!= (const GridCoordinate1DTemplate& rhs)
  {
    return getX () != rhs.getX ();
  }

  bool operator> (const GridCoordinate1DTemplate& rhs)
  {
    return getX () > rhs.getX ();
  }

  bool operator< (const GridCoordinate1DTemplate& rhs)
  {
    return getX () < rhs.getX ();
  }

  bool operator>= (const GridCoordinate1DTemplate& rhs)
  {
    return getX () >= rhs.getX ();
  }

  bool operator<= (const GridCoordinate1DTemplate& rhs)
  {
    return getX () <= rhs.getX ();
  }
};

typedef GridCoordinate1DTemplate<grid_iter> GridCoordinate1D;
typedef GridCoordinate1DTemplate<FieldValue> GridCoordinateFP1D;

#endif /* GRID_COORDINATE_1D_H */
