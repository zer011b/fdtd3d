#ifndef GRID_COORDINATE_3D_H
#define GRID_COORDINATE_3D_H

#include "GridCoordinate2D.h"

template<class TcoordType, bool doSignChecks>
class GridCoordinate3DTemplate;

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator* (TcoordType lhs, const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+ (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator- (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs);

/**
 * 3-dimensional coordinate in the grid.
 */
template<class TcoordType, bool doSignChecks>
class GridCoordinate3DTemplate: public GridCoordinate2DTemplate<TcoordType, doSignChecks>
{
  TcoordType z;

public:

  static const Dimension dimension;

  CUDA_DEVICE CUDA_HOST GridCoordinate3DTemplate ()
    : GridCoordinate2DTemplate<TcoordType, doSignChecks> (), z (0)
  {
    if (doSignChecks)
    {
      TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
      TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
      ASSERT (x >= 0 && y >= 0 && z >= 0);
    }
  }

  explicit CUDA_DEVICE CUDA_HOST GridCoordinate3DTemplate (const TcoordType& cx, const TcoordType& cy, const TcoordType& cz)
    : GridCoordinate2DTemplate<TcoordType, doSignChecks> (cx, cy), z (cz)
  {
    if (doSignChecks)
    {
      TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
      TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
      ASSERT (x >= 0 && y >= 0 && z >= 0);
    }
  }

  CUDA_DEVICE CUDA_HOST GridCoordinate3DTemplate (const GridCoordinate3DTemplate& pos)
    : GridCoordinate2DTemplate<TcoordType, doSignChecks> (pos.getX (), pos.getY ()), z (pos.getZ ())
  {
    if (doSignChecks)
    {
      TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
      TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
      ASSERT (x >= 0 && y >= 0 && z >= 0);
    }
  }

  CUDA_DEVICE CUDA_HOST ~GridCoordinate3DTemplate () {}

  // Calculate three-dimensional coordinate.
  TcoordType CUDA_DEVICE CUDA_HOST calculateTotalCoord () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
    TcoordType res = x * y;
    if (doSignChecks)
    {
      ASSERT (res >= 0);
    }
    res *= z;
    if (doSignChecks)
    {
      ASSERT (res >= 0);
    }
    return res;
  }

  // Get one-dimensional coordinates.
  const TcoordType& CUDA_DEVICE CUDA_HOST getZ () const
  {
    return z;
  }

  // Set one-dimensional coordinates.
  void CUDA_DEVICE CUDA_HOST setZ (const TcoordType& new_z)
  {
    z = new_z;
    if (doSignChecks)
    {
      ASSERT (z >= 0);
    }
  }

  TcoordType CUDA_DEVICE CUDA_HOST getMax () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

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

  GridCoordinate3DTemplate CUDA_DEVICE CUDA_HOST operator+ (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

    return GridCoordinate3DTemplate (x + rhs_x, y + rhs_y, getZ () + rhs.getZ ());
  }

  GridCoordinate3DTemplate CUDA_DEVICE CUDA_HOST operator- (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

    return GridCoordinate3DTemplate (x - rhs_x, y - rhs_y, getZ () - rhs.getZ ());
  }

  bool CUDA_DEVICE CUDA_HOST operator== (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

    return x == rhs_x && y == rhs_y && getZ () == rhs.getZ ();
  }

  bool CUDA_DEVICE CUDA_HOST operator!= (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

    return x != rhs_x || y != rhs_y || getZ () == rhs.getZ ();
  }

  bool CUDA_DEVICE CUDA_HOST operator> (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

    return x > rhs_x && y > rhs_y && getZ () > rhs.getZ ();
  }

  bool CUDA_DEVICE CUDA_HOST operator< (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

    return x < rhs_x && y < rhs_y && getZ () < rhs.getZ ();
  }

  bool CUDA_DEVICE CUDA_HOST CUDA_DEVICE CUDA_HOST operator>= (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

    return x >= rhs_x && y >= rhs_y && getZ () >= rhs.getZ ();
  }

  bool CUDA_DEVICE CUDA_HOST operator<= (const GridCoordinate3DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
    TcoordType rhs_y = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

    return x <= rhs_x && y <= rhs_y && getZ () <= rhs.getZ ();
  }

  GridCoordinate3DTemplate CUDA_DEVICE CUDA_HOST operator* (TcoordType rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

    return GridCoordinate3DTemplate (x * rhs, y * rhs, getZ () * rhs);
  }

  friend GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator* <TcoordType, doSignChecks>) (TcoordType lhs, const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs);
  friend GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator+ <TcoordType, doSignChecks>) (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs);
  friend GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator- <TcoordType, doSignChecks>) (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs);

  GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST shrink () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
    return GridCoordinate2DTemplate<TcoordType, doSignChecks> (x, y);
  }

  void print () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType y = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
    printf ("Coord (" COORD_MOD "," COORD_MOD "," COORD_MOD ").\n", x, y, getZ ());
  }
};

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator* (TcoordType lhs, const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs)
{
  TcoordType x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
  TcoordType y = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (lhs * x, lhs * y, lhs * rhs.getZ ());
}

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+ (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs)
{
  TcoordType x1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
  TcoordType y1 = lhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

  TcoordType x2 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::getX ();
  TcoordType y2 = rhs.GridCoordinate2DTemplate<TcoordType, !doSignChecks>::getY ();

  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (x1 + x2, y1 + y2, lhs.getZ () + rhs.getZ ());
}

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator- (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs)
{
  TcoordType x1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
  TcoordType y1 = lhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

  TcoordType x2 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::getX ();
  TcoordType y2 = rhs.GridCoordinate2DTemplate<TcoordType, !doSignChecks>::getY ();

  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (x1 - x2, y1 - y2, lhs.getZ () - rhs.getZ ());
}

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expand (const GridCoordinate2DTemplate<TcoordType, doSignChecks> &coord)
{
  TcoordType x = coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
  TcoordType y = coord.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (x, y, 0);
}

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expand2 (const GridCoordinate1DTemplate<TcoordType, doSignChecks> &coord)
{
  TcoordType x = coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (x, 0, 0);
}

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expandTo3D (const GridCoordinate1DTemplate<TcoordType, doSignChecks> &coord)
{
  TcoordType x = coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (x, 0, 0);
}

template<class TcoordType, bool doSignChecks>
void CUDA_DEVICE CUDA_HOST expandTo3DStartEnd (const GridCoordinate1DTemplate<TcoordType, doSignChecks> &start,
                                               const GridCoordinate1DTemplate<TcoordType, doSignChecks> &end,
                                               GridCoordinate3DTemplate<TcoordType, doSignChecks> &start3D,
                                               GridCoordinate3DTemplate<TcoordType, doSignChecks> &end3D)
{
  TcoordType start_x = start.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
  TcoordType end_x = end.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();

  start3D = GridCoordinate3DTemplate<TcoordType, doSignChecks> (start_x, 0, 0);
  end3D = GridCoordinate3DTemplate<TcoordType, doSignChecks> (end_x, 1, 1);
}

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expandTo3D (const GridCoordinate2DTemplate<TcoordType, doSignChecks> &coord)
{
  TcoordType x = coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
  TcoordType y = coord.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (x, y, 0);
}

template<class TcoordType, bool doSignChecks>
void CUDA_DEVICE CUDA_HOST expandTo3DStartEnd (const GridCoordinate2DTemplate<TcoordType, doSignChecks> &start,
                                               const GridCoordinate2DTemplate<TcoordType, doSignChecks> &end,
                                               GridCoordinate3DTemplate<TcoordType, doSignChecks> &start3D,
                                               GridCoordinate3DTemplate<TcoordType, doSignChecks> &end3D)
{
  TcoordType start_x = start.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
  TcoordType start_y = start.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

  TcoordType end_x = end.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
  TcoordType end_y = end.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();

  start3D = GridCoordinate3DTemplate<TcoordType, doSignChecks> (start_x, start_y, 0);
  end3D = GridCoordinate3DTemplate<TcoordType, doSignChecks> (end_x, end_y, 1);
}

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expandTo3D (const GridCoordinate3DTemplate<TcoordType, doSignChecks> &coord)
{
  TcoordType x = coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
  TcoordType y = coord.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
  TcoordType z = coord.GridCoordinate3DTemplate<TcoordType, doSignChecks>::getZ ();
  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (x, y, z);
}

template<class TcoordType, bool doSignChecks>
void CUDA_DEVICE CUDA_HOST expandTo3DStartEnd (const GridCoordinate3DTemplate<TcoordType, doSignChecks> &start,
                                               const GridCoordinate3DTemplate<TcoordType, doSignChecks> &end,
                                               GridCoordinate3DTemplate<TcoordType, doSignChecks> &start3D,
                                               GridCoordinate3DTemplate<TcoordType, doSignChecks> &end3D)
{
  TcoordType start_x = start.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
  TcoordType start_y = start.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
  TcoordType start_z = start.GridCoordinate3DTemplate<TcoordType, doSignChecks>::getZ ();

  TcoordType end_x = end.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
  TcoordType end_y = end.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getY ();
  TcoordType end_z = end.GridCoordinate3DTemplate<TcoordType, doSignChecks>::getZ ();

  start3D = GridCoordinate3DTemplate<TcoordType, doSignChecks> (start_x, start_y, start_z);
  end3D = GridCoordinate3DTemplate<TcoordType, doSignChecks> (end_x, end_y, end_z);
}

typedef GridCoordinate3DTemplate<grid_coord, true> GridCoordinate3D;
typedef GridCoordinate3DTemplate<grid_coord, false> GridCoordinateSigned3D;
typedef GridCoordinate3DTemplate<FPValue, true> GridCoordinateFP3D;
typedef GridCoordinate3DTemplate<FPValue, false> GridCoordinateSignedFP3D;

template<bool doSignChecks>
GridCoordinate3DTemplate<grid_coord, doSignChecks> CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate3DTemplate<FPValue, doSignChecks> coord)
{
  ASSERT (((grid_coord) coord.getX ()) == coord.getX ());
  ASSERT (((grid_coord) coord.getY ()) == coord.getY ());
  ASSERT (((grid_coord) coord.getZ ()) == coord.getZ ());

  return GridCoordinate3DTemplate<grid_coord, doSignChecks> ((grid_coord) coord.getX (),
                                                             (grid_coord) coord.getY (),
                                                             (grid_coord) coord.getZ ());
}

template<bool doSignChecks>
GridCoordinate3DTemplate<FPValue, doSignChecks> CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate3DTemplate<grid_coord, doSignChecks> coord)
{
  return GridCoordinate3DTemplate<FPValue, doSignChecks> (coord.getX (), coord.getY (), coord.getZ ());
}

#endif /* GRID_COORDINATE_3D_H */
