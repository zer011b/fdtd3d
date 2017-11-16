#ifndef GRID_COORDINATE_2D_H
#define GRID_COORDINATE_2D_H

#include "GridCoordinate1D.h"

template<class TcoordType, bool doSignChecks>
class GridCoordinate2DTemplate;

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator* (TcoordType lhs, const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+ (GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator- (GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs);

/**
 * 2-dimensional coordinate in the grid.
 */
template<class TcoordType, bool doSignChecks>
class GridCoordinate2DTemplate: public GridCoordinate1DTemplate<TcoordType, doSignChecks>
{
protected:

  TcoordType y;

public:

  CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate ()
    : GridCoordinate1DTemplate<TcoordType, doSignChecks> (), y (0)
  {
    if (doSignChecks)
    {
      TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
      ASSERT (x >= 0 && y >= 0);
    }
  }

  explicit CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate (const TcoordType& cx, const TcoordType& cy)
    : GridCoordinate1DTemplate<TcoordType, doSignChecks> (cx), y (cy)
  {
    if (doSignChecks)
    {
      TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
      ASSERT (x >= 0 && y >= 0);
    }
  }

  CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate (const GridCoordinate2DTemplate& pos)
    : GridCoordinate1DTemplate<TcoordType, doSignChecks> (pos.getX ()), y (pos.getY ())
  {
    if (doSignChecks)
    {
      TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
      ASSERT (x >= 0 && y >= 0);
    }
  }

  CUDA_DEVICE CUDA_HOST ~GridCoordinate2DTemplate () {}

  // Calculate three-dimensional coordinate.
  TcoordType CUDA_DEVICE CUDA_HOST calculateTotalCoord () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType res = x * y;
    if (doSignChecks)
    {
      ASSERT (res >= 0);
    }
    return res;
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
    if (doSignChecks)
    {
      ASSERT (y >= 0);
    }
  }

  TcoordType CUDA_DEVICE CUDA_HOST getMax () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    return x > y ? x : y;
  }

  GridCoordinate2DTemplate CUDA_DEVICE CUDA_HOST operator+ (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();

    return GridCoordinate2DTemplate (x + rhs_x, getY () + rhs.getY ());
  }

  GridCoordinate2DTemplate CUDA_DEVICE CUDA_HOST operator- (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();

    return GridCoordinate2DTemplate (x - rhs_x, getY () - rhs.getY ());
  }

  bool CUDA_DEVICE CUDA_HOST operator== (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    return x == rhs_x && getY () == rhs.getY ();
  }

  bool CUDA_DEVICE CUDA_HOST operator!= (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    return x != rhs_x || getY () != rhs.getY ();
  }

  bool CUDA_DEVICE CUDA_HOST operator> (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    return x > rhs_x && getY () > rhs.getY ();
  }

  bool CUDA_DEVICE CUDA_HOST operator< (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    return x < rhs_x && getY () < rhs.getY ();
  }

  bool CUDA_DEVICE CUDA_HOST operator>= (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    return x >= rhs_x && getY () >= rhs.getY ();
  }

  bool CUDA_DEVICE CUDA_HOST operator<= (const GridCoordinate2DTemplate& rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    TcoordType rhs_x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    return x <= rhs_x && getY () <= rhs.getY ();
  }

  GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator* (TcoordType rhs) const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();

    return GridCoordinate2DTemplate<TcoordType, doSignChecks> (x * rhs, getY () * rhs);
  }

  friend GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator* <TcoordType, doSignChecks>) (TcoordType lhs, const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs);
  friend GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator+ <TcoordType, doSignChecks>) (GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs);
  friend GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST (::operator- <TcoordType, doSignChecks>) (GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs);

  GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST shrink () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (x);
  }

  void print () const
  {
    TcoordType x = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
    printf ("Coord (" COORD_MOD "," COORD_MOD ").\n", x, getY ());
  }
};

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator* (TcoordType lhs, const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs)
{
  TcoordType x = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();

  return GridCoordinate2DTemplate<TcoordType, doSignChecks> (lhs * x, lhs * rhs.getY ());
}

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+ (GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs)
{
  TcoordType x1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();

  TcoordType x2 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::getX ();

  return GridCoordinate2DTemplate<TcoordType, doSignChecks> (x1 + x2, lhs.getY () + rhs.getY ());
}

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator- (GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs)
{
  TcoordType x1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();

  TcoordType x2 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::getX ();

  return GridCoordinate2DTemplate<TcoordType, doSignChecks> (x1 - x2, lhs.getY () - rhs.getY ());
}

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expand (const GridCoordinate1DTemplate<TcoordType, doSignChecks> &coord)
{
  TcoordType x = coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getX ();
  return GridCoordinate2DTemplate<TcoordType, doSignChecks> (x, 0);
}

typedef GridCoordinate2DTemplate<grid_coord, true> GridCoordinate2D;
typedef GridCoordinate2DTemplate<grid_coord, false> GridCoordinateSigned2D;
typedef GridCoordinate2DTemplate<FPValue, true> GridCoordinateFP2D;
typedef GridCoordinate2DTemplate<FPValue, false> GridCoordinateSignedFP2D;

template<bool doSignChecks>
GridCoordinate2DTemplate<grid_coord, doSignChecks> CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate2DTemplate<FPValue, doSignChecks> coord)
{
  ASSERT (((grid_coord) coord.getX ()) == coord.getX ());
  ASSERT (((grid_coord) coord.getY ()) == coord.getY ());

  return GridCoordinate2DTemplate<grid_coord, doSignChecks> ((grid_coord) coord.getX (),
                                                             (grid_coord) coord.getY ());
}

template<bool doSignChecks>
GridCoordinate2DTemplate<FPValue, doSignChecks> CUDA_DEVICE CUDA_HOST convertCoord (GridCoordinate2DTemplate<grid_coord, doSignChecks> coord)
{
  return GridCoordinate2DTemplate<FPValue, doSignChecks> (coord.getX (), coord.getY ());
}

#endif /* GRID_COORDINATE_2D_H */
