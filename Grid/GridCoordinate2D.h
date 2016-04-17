#ifndef GRID_COORDINATE_2D_H
#define GRID_COORDINATE_2D_H

#include "GridCoordinate1D.h"

// Coordinate in the grid.
class GridCoordinate2D: public GridCoordinate1D
{
protected:

  grid_coord y;

public:

  GridCoordinate2D ()
    : GridCoordinate1D (), y (0) {}

  explicit GridCoordinate2D (const grid_coord& cx, const grid_coord& cy)
    : GridCoordinate1D (cx), y (cy) {}

  // Constructor for case when grid has the same dimension for all axes.
  explicit GridCoordinate2D (const grid_coord& cxy)
    : GridCoordinate1D (cxy), y (cxy) {}

  GridCoordinate2D (const GridCoordinate2D& pos)
    : GridCoordinate1D (pos.getX ()), y (pos.getY ()) {}

  ~GridCoordinate2D () {}

  // Calculate three-dimensional coordinate.
  grid_iter calculateTotalCoord () const
  {
    return x * y;
  }

  // Get one-dimensional coordinates.
  const grid_coord& getY () const
  {
    return y;
  }

  // Set one-dimensional coordinates.
  void setY (const grid_coord& new_y)
  {
    y = new_y;
  }

  grid_coord getMax () const
  {
    return x > y ? x : y;
  }

  GridCoordinate2D operator+ (const GridCoordinate2D& rhs)
  {
    return GridCoordinate2D (getX () + rhs.getX (), getY () + rhs.getY ());
  }

  GridCoordinate2D operator- (const GridCoordinate2D& rhs)
  {
    return GridCoordinate2D (getX () - rhs.getX (), getY () - rhs.getY ());
  }

  bool operator== (const GridCoordinate2D& rhs)
  {
    return getX () == rhs.getX () && getY () == rhs.getY ();
  }

  bool operator!= (const GridCoordinate2D& rhs)
  {
    return getX () != rhs.getX () || getY () != rhs.getY ();
  }

  bool operator> (const GridCoordinate2D& rhs)
  {
    return getX () > rhs.getX () && getY () > rhs.getY ();
  }

  bool operator< (const GridCoordinate2D& rhs)
  {
    return getX () < rhs.getX () && getY () < rhs.getY ();
  }

  bool operator>= (const GridCoordinate2D& rhs)
  {
    return getX () >= rhs.getX () && getY () >= rhs.getY ();
  }

  bool operator<= (const GridCoordinate2D& rhs)
  {
    return getX () <= rhs.getX () && getY () <= rhs.getY ();
  }
};

#endif /* GRID_COORDINATE_2D_H */
