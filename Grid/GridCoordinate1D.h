#ifndef GRID_COORDINATE_1D_H
#define GRID_COORDINATE_1D_H

#ifdef CXX11_ENABLED
#include <cstdint>
#else
#include <stdint.h>
#endif

// Type of one-dimensional coordinate.
typedef uint32_t grid_coord;
// Type of three-dimensional coordinate.
typedef uint64_t grid_iter;

// Coordinate in the grid.
class GridCoordinate1D
{
protected:

  grid_coord x;

public:

  // Constructor for all cases.
  explicit GridCoordinate1D (const grid_coord& cx = 0)
    : x (cx) {}
  GridCoordinate1D (const GridCoordinate1D& coord)
    : x (coord.getX ()) {}

  ~GridCoordinate1D () {};

  // Calculate total-dimensional coordinate.
  grid_iter calculateTotalCoord () const
  {
    return x;
  }

  // Get one-dimensional coordinates.
  const grid_coord& getX () const
  {
    return x;
  }

  // Set one-dimensional coordinates.
  void setX (const grid_coord& new_x)
  {
    x = new_x;
  }

  grid_coord getMax () const
  {
    return x;
  }

  GridCoordinate1D operator+ (const GridCoordinate1D& rhs)
  {
    return GridCoordinate1D (getX () + rhs.getX ());
  }

  GridCoordinate1D operator- (const GridCoordinate1D& rhs)
  {
    return GridCoordinate1D (getX () - rhs.getX ());
  }

  bool operator== (const GridCoordinate1D& rhs)
  {
    return getX () == rhs.getX ();
  }

  bool operator!= (const GridCoordinate1D& rhs)
  {
    return getX () != rhs.getX ();
  }

  bool operator> (const GridCoordinate1D& rhs)
  {
    return getX () > rhs.getX ();
  }

  bool operator< (const GridCoordinate1D& rhs)
  {
    return getX () < rhs.getX ();
  }

  bool operator>= (const GridCoordinate1D& rhs)
  {
    return getX () >= rhs.getX ();
  }

  bool operator<= (const GridCoordinate1D& rhs)
  {
    return getX () <= rhs.getX ();
  }
};

#endif /* GRID_COORDINATE_1D_H */
