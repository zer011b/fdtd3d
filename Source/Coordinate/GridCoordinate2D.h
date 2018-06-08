#ifndef GRID_COORDINATE_2D_H
#define GRID_COORDINATE_2D_H

#include "GridCoordinate1D.h"

template<class TcoordType, bool doSignChecks>
class GridCoordinate2DTemplate;

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator*
(TcoordType lhs, const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+
(GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator-
(GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs);

/**
 * 2-dimensional coordinate in the grid.
 */
template<class TcoordType, bool doSignChecks>
class GridCoordinate2DTemplate: public GridCoordinate1DTemplate<TcoordType, doSignChecks>
{
protected:

  /**
   * Second coordinate
   */
  TcoordType coord2;

#ifdef DEBUG_INFO
  /**
   * Type of second coordinate
   */
  CoordinateType type2;
#endif /* DEBUG_INFO */

public:

  /**
   * Dimension of this grid coordinate
   */
  static const Dimension dimension;

  /**
   * Default constructor
   */
  CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate<TcoordType, doSignChecks> ()
    : GridCoordinate1DTemplate<TcoordType, doSignChecks> ()
    , coord2 (0)
#ifdef DEBUG_INFO
    , type2 (CoordinateType::NONE)
#endif /* DEBUG_INFO */
  {
  } /* GridCoordinate2DTemplate */

  /**
   * Constructor with specified coordinate and its type
   */
  explicit CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate<TcoordType, doSignChecks>
    (const TcoordType& c1, /**< first coordinate */
     const TcoordType& c2 /**< second coordinate */
#ifdef DEBUG_INFO
     , CoordinateType t1 /**< first coordinate type */
     , CoordinateType t2 /**< second coordinate type */
#endif /* DEBUG_INFO */
     )
    : GridCoordinate1DTemplate<TcoordType, doSignChecks> (c1
#ifdef DEBUG_INFO
                                                          , t1
#endif /* DEBUG_INFO */
                                                          )
    , coord2 (c2)
#ifdef DEBUG_INFO
    , type2 (t2)
#endif /* DEBUG_INFO */
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () < getType2 ())
            || (GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == getType2 ()
                && getType2 () == CoordinateType::NONE));
#endif /* DEBUG_INFO */

    if (doSignChecks)
    {
      TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
      ASSERT (coord1 >= 0 && coord2 >= 0);
    }
  } /* GridCoordinate2DTemplate */

  /**
   * Constructor with interface similar for all coordinate dimensions
   */
  explicit CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate<TcoordType, doSignChecks>
    (const TcoordType& c1, /**< first coordinate */
     const TcoordType& c2, /**< second coordinate */
     const TcoordType& tmp /**< unused coordinate */
#ifdef DEBUG_INFO
     , CoordinateType t1 /**< first coodinate type */
     , CoordinateType t2 /**< second coodinate type */
     , CoordinateType t3 /**< unused coordinate type */
#endif /* DEBUG_INFO */
     )
    : GridCoordinate1DTemplate<TcoordType, doSignChecks> (c1
#ifdef DEBUG_INFO
                                                          , t1
#endif /* DEBUG_INFO */
                                                          )
    , coord2 (c2)
#ifdef DEBUG_INFO
    , type2 (t2)
#endif /* DEBUG_INFO */
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () < getType2 ())
            || (GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == getType2 ()
                && getType2 () == CoordinateType::NONE));
#endif /* DEBUG_INFO */

    if (doSignChecks)
    {
      TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
      ASSERT (coord1 >= 0 && coord2 >= 0);
    }
  } /* GridCoordinate2DTemplate */

  /**
   * Copy constructor
   */
  CUDA_DEVICE CUDA_HOST GridCoordinate2DTemplate<TcoordType, doSignChecks>
    (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& pos) /**< new coordinate */
    : GridCoordinate1DTemplate<TcoordType, doSignChecks> (pos.get1 ()
#ifdef DEBUG_INFO
      , pos.getType1 ()
#endif /* DEBUG_INFO */
      )
    , coord2 (pos.get2 ())
#ifdef DEBUG_INFO
    , type2 (pos.getType2 ())
#endif /* DEBUG_INFO */
  {
#ifdef DEBUG_INFO
    ASSERT ((GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () < getType2 ())
            || (GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 () == getType2 ()
                && getType2 () == CoordinateType::NONE));
#endif /* DEBUG_INFO */

    if (doSignChecks)
    {
      TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
      ASSERT (coord1 >= 0 && coord2 >= 0);
    }
  } /* GridCoordinate2DTemplate */

  /**
   * Destructor
   */
  CUDA_DEVICE CUDA_HOST ~GridCoordinate2DTemplate<TcoordType, doSignChecks> () {}

  /**
   * Calculate total-dimensional coordinate
   *
   * @return total-dimensional coordinate
   */
  TcoordType CUDA_DEVICE CUDA_HOST calculateTotalCoord () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType res = coord1 * coord2;
    if (doSignChecks)
    {
      ASSERT (res >= 0);
    }
    return res;
  } /* GridCoordinate2DTemplate::calculateTotalCoord */

  /**
   * Get second coordinate
   *
   * @return second coordinate
   */
  const TcoordType& CUDA_DEVICE CUDA_HOST get2 () const
  {
    return coord2;
  } /* GridCoordinate2DTemplate::get2 */

  /**
   * Set second coordinate
   */
  void CUDA_DEVICE CUDA_HOST set2 (const TcoordType& new_c2) /**< new second coordinate */
  {
    coord2 = new_c2;
    if (doSignChecks)
    {
      ASSERT (coord2 >= 0);
    }
  } /* GridCoordinate2DTemplate::set2 */

#ifdef DEBUG_INFO
  /**
   * Get type of second coordinate
   *
   * @return type of second coordinate
   */
  CoordinateType CUDA_DEVICE CUDA_HOST getType2 () const
  {
    return type2;
  } /* GridCoordinate2DTemplate::getType2 */
#endif /* DEBUG_INFO */

  /**
   * Get maximum coordinate
   *
   * @return maximum coordinate
   */
  TcoordType CUDA_DEVICE CUDA_HOST getMax () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return coord1 > coord2 ? coord1 : coord2;
  } /* GridCoordinate2DTemplate::getMax */

  /**
   * Operator + for grid coordinates
   *
   * @return sum of two grid coordinates
   */
  GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  operator+ (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();

    return GridCoordinate2DTemplate (coord1 + rhs_c1, get2 () + rhs.get2 ()
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), getType2 ()
#endif /* DEBUG_INFO */
      );
  } /* GridCoordinate2DTemplate::operator+ */

  /**
   * Operator - for grid coordinates
   *
   * @return sum of two grid coordinates
   */
  GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  operator- (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();

    return GridCoordinate2DTemplate (coord1 - rhs_c1, get2 () - rhs.get2 ()
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), getType2 ()
#endif /* DEBUG_INFO */
      );
  } /* GridCoordinate2DTemplate::operator- */

  /**
   * Operator == for grid coordinates
   *
   * @return true, if grid coordinates are equal
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator== (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return coord1 == rhs_c1 && get2 () == rhs.get2 ();
  } /* GridCoordinate2DTemplate::operator== */

  /**
   * Operator != for grid coordinates
   *
   * @return true, if grid coordinates are not equal
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator!= (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return coord1 != rhs_c1 || get2 () != rhs.get2 ();
  } /* GridCoordinate2DTemplate::operator!= */

  /**
   * Operator > for grid coordinates
   *
   * @return true, if first grid coordinates is greater than second
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator> (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return coord1 > rhs_c1 && get2 () > rhs.get2 ();
  } /* GridCoordinate2DTemplate::operator> */

  /**
   * Operator < for grid coordinates
   *
   * @return true, if first grid coordinates is less than second
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator< (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const  /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return coord1 < rhs_c1 && get2 () < rhs.get2 ();
  } /* GridCoordinate2DTemplate::operator< */

  /**
   * Operator >= for grid coordinates
   *
   * @return true, if first grid coordinates is greater or equal than second
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator>= (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return coord1 >= rhs_c1 && get2 () >= rhs.get2 ();
  } /* GridCoordinate2DTemplate::operator>= */

  /**
   * Operator <= for grid coordinates
   *
   * @return true, if first grid coordinates is less or equal than second
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator<= (const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return coord1 <= rhs_c1 && get2 () <= rhs.get2 ();
  } /* GridCoordinate2DTemplate::operator<= */

  /**
   * Operator * for grid coordinate and coordinate
   *
   * @return grid coordinate wuth each coordinate multiplied by coordinate
   */
  GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  operator* (TcoordType rhs) const /**< operand */
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return GridCoordinate2DTemplate<TcoordType, doSignChecks> (coord1 * rhs, get2 () * rhs
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), getType2 ()
#endif /* DEBUG_INFO */
      );
  } /* GridCoordinate2DTemplate::operator* */

  /**
   * Operator / for grid coordinate and coordinate
   *
   * @return grid coordinate wuth each coordinate divided by coordinate
   */
  GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  operator/ (TcoordType rhs) const /**< operand */
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    return GridCoordinate2DTemplate<TcoordType, doSignChecks> (coord1 / rhs, get2 () / rhs
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), getType2 ()
#endif /* DEBUG_INFO */
      );
  } /* GridCoordinate2DTemplate::operator/ */

  /**
   * Initialize grid coordinate according to coordinate types. This function has same interfaces for all grid coordinate
   * dimensions and allows to create coordinate based on values of three coordinate types.
   *
   * For example, to initialize 2D-XZ coordinate, ct1 should be CoordinateType::X, ct2 - CoordinateType::Z,
   * c1 be the X coordinate, and c3 the Z coordinate.
   *
   * @return initialized grid coordinate
   */
  static GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  initAxesCoordinate (const TcoordType& c1, /**< first coordinate */
                      const TcoordType& c2, /**< second coordinate */
                      const TcoordType& c3, /**< third coordinate */
                      CoordinateType ct1, /**< first coordinate type */
                      CoordinateType ct2, /**< second coordinate type */
                      CoordinateType ct3) /**< third coordinate type */
  {
    ASSERT (ct3 == CoordinateType::NONE);
    if (ct1 == CoordinateType::X && ct2 == CoordinateType::Y)
    {
      return GridCoordinate2DTemplate<TcoordType, doSignChecks> (c1, c2
#ifdef DEBUG_INFO
                                                                 , ct1, ct2
#endif /* DEBUG_INFO */
                                                                 );
    }
    else if (ct1 == CoordinateType::X && ct2 == CoordinateType::Z)
    {
      return GridCoordinate2DTemplate<TcoordType, doSignChecks> (c1, c3
#ifdef DEBUG_INFO
                                                                 , ct1, ct2
#endif /* DEBUG_INFO */
                                                                 );
    }
    else if (ct1 == CoordinateType::Y && ct2 == CoordinateType::Z)
    {
      return GridCoordinate2DTemplate<TcoordType, doSignChecks> (c2, c3
#ifdef DEBUG_INFO
                                                                 , ct1, ct2
#endif /* DEBUG_INFO */
                                                                 );
    }
    else
    {
      UNREACHABLE;
    }
  } /* GridCoordinate2DTemplate::initAxesCoordinate */

  /*
   * Friend operators *,+,- declaration
   */
  friend GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  (::operator* <TcoordType, doSignChecks>)
    (TcoordType lhs, const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs);
  friend GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  (::operator+ <TcoordType, doSignChecks>)
    (GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs,
     const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs);
  friend GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  (::operator- <TcoordType, doSignChecks>)
    (GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs,
     const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs);

  /**
   * Print coordinate
   */
  void print () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    printf ("Coord ("
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      "%f, "
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      "%f ).\n"
#ifdef DEBUG_INFO
      , coordinateTypeNames[static_cast<uint8_t> (GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ())]
#endif /* DEBUG_INFO */
      , (FPValue) coord1
#ifdef DEBUG_INFO
      , coordinateTypeNames[static_cast<uint8_t> (getType2 ())]
#endif /* DEBUG_INFO */
      , (FPValue) get2 ());
  } /* GridCoordinate2DTemplate::print */
}; /* GridCoordinate2DTemplate */

/**
 * Coordinate operator * for number and grid coordinate
 *
 * @return grid coordinate, which is the result of multiplication of number and grid coordinate
 */
template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator*
  (TcoordType lhs, /**< first operand (number) */
   const GridCoordinate2DTemplate<TcoordType, doSignChecks>& rhs) /**< second operand */
{
  TcoordType coord1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  return GridCoordinate2DTemplate<TcoordType, doSignChecks> (lhs * coord1, lhs * rhs.get2 ()
#ifdef DEBUG_INFO
    , rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), rhs.getType2 ()
#endif /* DEBUG_INFO */
    );
} /* GridCoordinate2DTemplate::operator* */

/**
 * Coodinate operator + for two grid coordinates, only for one of which sign checks are enabled
 *
 * @return result of addition of two grid coordinates
 */
template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+
  (GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, /**< first operand */
   const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs) /**< second operand */
{
#ifdef DEBUG_INFO
  CoordinateType cct1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
  CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::getType1 ();
  ASSERT (cct1 == cct2);
  ASSERT (lhs.getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
  TcoordType lcoord1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  TcoordType rcoord1 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::get1 ();
  return GridCoordinate2DTemplate<TcoordType, doSignChecks> (lcoord1 + rcoord1, lhs.get2 () + rhs.get2 ()
#ifdef DEBUG_INFO
    , lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), lhs.getType2 ()
#endif /* DEBUG_INFO */
    );
} /* GridCoordinate2DTemplate::operator+ */

/**
 * Coodinate operator - for two grid coordinates, only for one of which sign checks are enabled
 *
 * @return result of substraction of two grid coordinates
 */
template<class TcoordType, bool doSignChecks>
GridCoordinate2DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator-
  (GridCoordinate2DTemplate<TcoordType, doSignChecks> &lhs, /**< first operand */
   const GridCoordinate2DTemplate<TcoordType, !doSignChecks>& rhs) /**< second operand */
{
#ifdef DEBUG_INFO
  CoordinateType cct1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
  CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::getType1 ();
  ASSERT (cct1 == cct2);
  ASSERT (lhs.getType2 () == rhs.getType2 ());
#endif /* DEBUG_INFO */
  TcoordType lcoord1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  TcoordType rcoord1 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::get1 ();
  return GridCoordinate2DTemplate<TcoordType, doSignChecks> (lcoord1 - rcoord1, lhs.get2 () - rhs.get2 ()
#ifdef DEBUG_INFO
    , lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 (), lhs.getType2 ()
#endif /* DEBUG_INFO */
    );
} /* GridCoordinate2DTemplate::operator- */

/**
 * 2D grid coordinate with sign checks and integer type for each individual coordinate
 */
typedef GridCoordinate2DTemplate<grid_coord, true> GridCoordinate2D;

/**
 * 2D grid coordinate without sign checks and integer type for each individual coordinate
 */
typedef GridCoordinate2DTemplate<grid_coord, false> GridCoordinateSigned2D;

/**
 * 2D grid coordinate with sign checks and floating point type for each individual coordinate
 */
typedef GridCoordinate2DTemplate<FPValue, true> GridCoordinateFP2D;

/**
 * 2D grid coordinate without sign checks and floating point type for each individual coordinate
 */
typedef GridCoordinate2DTemplate<FPValue, false> GridCoordinateSignedFP2D;

/**
 * Convert floating point coordinate to integer coordinate
 *
 * @return integer coodinate, corresponding to floating point coordinate
 */
template<bool doSignChecks>
GridCoordinate2DTemplate<grid_coord, doSignChecks> CUDA_DEVICE CUDA_HOST
convertCoord (GridCoordinate2DTemplate<FPValue, doSignChecks> coord) /**< floating point coordinate */
{
  ASSERT (((grid_coord) coord.get1 ()) == coord.get1 ());
  ASSERT (((grid_coord) coord.get2 ()) == coord.get2 ());

  return GridCoordinate2DTemplate<grid_coord, doSignChecks> ((grid_coord) coord.get1 (), (grid_coord) coord.get2 ()
#ifdef DEBUG_INFO
    , coord.GridCoordinate1DTemplate<FPValue, doSignChecks>::getType1 (), coord.getType2 ()
#endif /* DEBUG_INFO */
    );
} /* convertCoord */

/**
 * Convert integer coordinate to floating point coordinate
 *
 * @return floating point coodinate, corresponding to integer coordinate
 */
template<bool doSignChecks>
GridCoordinate2DTemplate<FPValue, doSignChecks> CUDA_DEVICE CUDA_HOST
convertCoord (GridCoordinate2DTemplate<grid_coord, doSignChecks> coord) /**< integer coordinate */
{
  return GridCoordinate2DTemplate<FPValue, doSignChecks> (coord.get1 (), coord.get2 ()
#ifdef DEBUG_INFO
    , coord.GridCoordinate1DTemplate<grid_coord, doSignChecks>::getType1 (), coord.getType2 ()
#endif /* DEBUG_INFO */
    );
} /* convertCoord */

/**
 * Dimension of 2D grid coordinate
 */
template<class TcoordType, bool doSignChecks>
const Dimension GridCoordinate2DTemplate<TcoordType, doSignChecks>::dimension = Dimension::Dim2;

#endif /* GRID_COORDINATE_2D_H */
