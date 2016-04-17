#ifndef GRID_COORDINATE_3D_H
#define GRID_COORDINATE_3D_H

#include "GridCoordinate2D.h"

// Coordinate in the grid.
class GridCoordinate3D: public GridCoordinate2D
{
  grid_coord z;

public:

  GridCoordinate3D ()
    : GridCoordinate2D (), z (0) {}

  explicit GridCoordinate3D (const grid_coord& cx, const grid_coord& cy, const grid_coord& cz)
    : GridCoordinate2D (cx, cy), z (cz) {}

  // Constructor for case when grid has the same dimension for all axes.
  explicit GridCoordinate3D (const grid_coord& cxyz)
    : GridCoordinate2D (cxyz), z (cxyz) {}

  GridCoordinate3D (const GridCoordinate3D& pos)
    : GridCoordinate2D (pos.getX (), pos.getY ()), z (pos.getZ ()) {}

  ~GridCoordinate3D () {}

  // Calculate three-dimensional coordinate.
  grid_iter calculateTotalCoord () const
  {
    return x * y * z;
  }

  // Get one-dimensional coordinates.
  const grid_coord& getZ () const
  {
    return z;
  }

  // Set one-dimensional coordinates.
  void setZ (const grid_coord& new_z)
  {
    z = new_z;
  }

  grid_coord getMax () const
  {
    if (x > y && x > z)
    {
      return x;
    }
    else if (y > z)
    {
      return y;
    }
    else
    {
      return z;
    }
  }

  GridCoordinate3D operator+ (const GridCoordinate3D& rhs)
  {
    return GridCoordinate3D (getX () + rhs.getX (), getY () + rhs.getY (), getZ () + rhs.getZ ());
  }

  GridCoordinate3D operator- (const GridCoordinate3D& rhs)
  {
    return GridCoordinate3D (getX () - rhs.getX (), getY () - rhs.getY (), getZ () - rhs.getZ ());
  }

  bool operator== (const GridCoordinate3D& rhs)
  {
    return getX () == rhs.getX () && getY () == rhs.getY () && getZ () == rhs.getZ ();
  }

  bool operator!= (const GridCoordinate3D& rhs)
  {
    return getX () != rhs.getX () || getY () != rhs.getY () || getZ () == rhs.getZ ();
  }

  bool operator> (const GridCoordinate3D& rhs)
  {
    return getX () > rhs.getX () && getY () > rhs.getY () && getZ () > rhs.getZ ();
  }

  bool operator< (const GridCoordinate3D& rhs)
  {
    return getX () < rhs.getX () && getY () < rhs.getY () && getZ () < rhs.getZ ();
  }

  bool operator>= (const GridCoordinate3D& rhs)
  {
    return getX () >= rhs.getX () && getY () >= rhs.getY () && getZ () >= rhs.getZ ();
  }

  bool operator<= (const GridCoordinate3D& rhs)
  {
    return getX () <= rhs.getX () && getY () <= rhs.getY () && getZ () <= rhs.getZ ();
  }
};

#endif /* GRID_COORDINATE_3D_H */
