#ifndef GRID_COORDINATE_3D_H
#define GRID_COORDINATE_3D_H

#include "GridCoordinate2D.h"

// Coordinate in the grid.
template<class TcoordType>
class GridCoordinate3DTemplate: public GridCoordinate2DTemplate<TcoordType>
{
  TcoordType z;

public:

  GridCoordinate3DTemplate ()
    : GridCoordinate2DTemplate<TcoordType> (), z (0) {}

  explicit GridCoordinate3DTemplate (const TcoordType& cx, const TcoordType& cy, const TcoordType& cz)
    : GridCoordinate2DTemplate<TcoordType> (cx, cy), z (cz) {}

  // Constructor for case when grid has the same dimension for all axes.
  explicit GridCoordinate3DTemplate (const TcoordType& cxyz)
    : GridCoordinate2DTemplate<TcoordType> (cxyz), z (cxyz) {}

  GridCoordinate3DTemplate (const GridCoordinate3DTemplate& pos)
    : GridCoordinate2DTemplate<TcoordType> (pos.getX (), pos.getY ()), z (pos.getZ ()) {}

  GridCoordinate3DTemplate (const GridCoordinate1DTemplate<TcoordType>& pos)
    : GridCoordinate2DTemplate<TcoordType> (pos), z (0) {}

  GridCoordinate3DTemplate (const GridCoordinate2DTemplate<TcoordType>& pos)
    : GridCoordinate2DTemplate<TcoordType> (pos), z (0) {}

  ~GridCoordinate3DTemplate () {}

  // Calculate three-dimensional coordinate.
  grid_iter calculateTotalCoord () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType>::getY ();
    return x * y * z;
  }

  // Get one-dimensional coordinates.
  const TcoordType& getZ () const
  {
    return z;
  }

  // Set one-dimensional coordinates.
  void setZ (const TcoordType& new_z)
  {
    z = new_z;
  }

  TcoordType getMax () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType>::getY ();

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

  GridCoordinate3DTemplate operator+ (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType>::getY ();

    return GridCoordinate3DTemplate (x + rhs_x, y + rhs_y, getZ () + rhs.getZ ());
  }

  GridCoordinate3DTemplate operator- (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType>::getY ();

    return GridCoordinate3DTemplate (x - rhs_x, y - rhs_y, getZ () - rhs.getZ ());
  }

  bool operator== (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType>::getY ();

    return x == rhs_x && y == rhs_y && getZ () == rhs.getZ ();
  }

  bool operator!= (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType>::getY ();

    return x != rhs_x || y != rhs_y || getZ () == rhs.getZ ();
  }

  bool operator> (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType>::getY ();

    return x > rhs_x && y > rhs_y && getZ () > rhs.getZ ();
  }

  bool operator< (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType>::getY ();

    return x < rhs_x && y < rhs_y && getZ () < rhs.getZ ();
  }

  bool operator>= (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType>::getY ();

    return x >= rhs_x && y >= rhs_y && getZ () >= rhs.getZ ();
  }

  bool operator<= (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType>::getY ();

    return x <= rhs_x && y <= rhs_y && getZ () <= rhs.getZ ();
  }
};

typedef GridCoordinate3DTemplate<grid_iter> GridCoordinate3D;
typedef GridCoordinate3DTemplate<FieldValue> GridCoordinateFP3D;

GridCoordinate3D convertCoord (GridCoordinateFP3D coord);
GridCoordinateFP3D convertCoord (GridCoordinate3D coord);

GridCoordinate2D shrinkCoord (GridCoordinate3D coord);
GridCoordinateFP2D shrinkCoord (GridCoordinateFP3D coord);

#endif /* GRID_COORDINATE_3D_H */
