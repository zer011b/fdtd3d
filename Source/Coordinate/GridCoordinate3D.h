#ifndef GRID_COORDINATE_3D_H
#define GRID_COORDINATE_3D_H

#include "GridCoordinate2D.h"

template<class TcoordType, bool doSignChecks>
class GridCoordinate3DTemplate;

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator*
(TcoordType lhs, const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+
  (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs,
   const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator-
  (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs,
   const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs);

/**
 * 3-dimensional coordinate in the grid.
 */
template<class TcoordType, bool doSignChecks>
class GridCoordinate3DTemplate: public GridCoordinate2DTemplate<TcoordType, doSignChecks>
{
protected:

  /**
   * Third coordinate
   */
  TcoordType coord3;

#ifdef DEBUG_INFO
  /**
   * Type of third coordinate
   */
  CoordinateType type3;
#endif /* DEBUG_INFO */

public:

  /**
   * Dimension of this grid coordinate
   */
  static const Dimension dimension;

  /**
   * Default constructor
   */
  CUDA_DEVICE CUDA_HOST GridCoordinate3DTemplate<TcoordType, doSignChecks> ()
    : GridCoordinate2DTemplate<TcoordType, doSignChecks> ()
    , coord3 (0)
#ifdef DEBUG_INFO
    , type3 (CoordinateType::NONE)
#endif /* DEBUG_INFO */
  {
  } /* GridCoordinate3DTemplate */

  /**
   * Constructor with interface similar for all coordinate dimensions
   */
  explicit CUDA_DEVICE CUDA_HOST GridCoordinate3DTemplate<TcoordType, doSignChecks>
    (const TcoordType& c1, /**< first coordinate */
     const TcoordType& c2, /**< second coordinate */
     const TcoordType& c3 /**< unused coordinate */
#ifdef DEBUG_INFO
     , CoordinateType t1 /**< first coodinate type */
     , CoordinateType t2 /**< second coodinate type */
     , CoordinateType t3 /**< unused coordinate type */
#endif /* DEBUG_INFO */
     )
    : GridCoordinate2DTemplate<TcoordType, doSignChecks> (c1, c2
#ifdef DEBUG_INFO
                                                          , t1, t2
#endif /* DEBUG_INFO */
                                                          )
    , coord3 (c3)
#ifdef DEBUG_INFO
    , type3 (t3)
#endif /* DEBUG_INFO */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    ASSERT ((cct1 < cct2) || (cct1 == cct2 && cct2 == CoordinateType::NONE));
    ASSERT ((cct2 < getType3 ()) || (cct2 == getType3 () && getType3 () == CoordinateType::NONE));
#endif /* DEBUG_INFO */

    if (doSignChecks)
    {
      TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
      TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
      ASSERT (coord1 >= 0 && coord2 >= 0 && coord3 >= 0);
    }
  } /* GridCoordinate3DTemplate */

  /**
   * Copy constructor
   */
  CUDA_DEVICE CUDA_HOST GridCoordinate3DTemplate<TcoordType, doSignChecks>
    (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& pos) /**< new coordinate */
    : GridCoordinate2DTemplate<TcoordType, doSignChecks> (pos.get1 (), pos.get2 ()
#ifdef DEBUG_INFO
    , pos.getType1 ()
    , pos.getType2 ()
#endif /* DEBUG_INFO */
      )
    , coord3 (pos.get3 ())
#ifdef DEBUG_INFO
    , type3 (pos.getType3 ())
#endif /* DEBUG_INFO */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    ASSERT ((cct1 < cct2) || (cct1 == cct2 && cct2 == CoordinateType::NONE));
    ASSERT ((cct2 < getType3 ()) || (cct2 == getType3 () && getType3 () == CoordinateType::NONE));
#endif /* DEBUG_INFO */

    if (doSignChecks)
    {
      TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
      TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
      ASSERT (coord1 >= 0 && coord2 >= 0 && coord3 >= 0);
    }
  } /* GridCoordinate3DTemplate */

  /**
   * Destructor
   */
  CUDA_DEVICE CUDA_HOST ~GridCoordinate3DTemplate<TcoordType, doSignChecks> () {}

  /**
   * Calculate total-dimensional coordinate
   *
   * @return total-dimensional coordinate
   */
  TcoordType CUDA_DEVICE CUDA_HOST calculateTotalCoord () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType res = coord1 * coord2;
    if (doSignChecks)
    {
      ASSERT (res >= 0);
    }
    res *= coord3;
    if (doSignChecks)
    {
      ASSERT (res >= 0);
    }
    return res;
  } /* GridCoordinate3DTemplate::calculateTotalCoord */

  /**
   * Get third coordinate
   *
   * @return third coordinate
   */
  const TcoordType& CUDA_DEVICE CUDA_HOST get3 () const
  {
    return coord3;
  } /* GridCoordinate3DTemplate::get3 */

  /**
   * Set third coordinate
   */
  void CUDA_DEVICE CUDA_HOST set3 (const TcoordType& new_c3) /**< new third coordinate */
  {
    coord3 = new_c3;
    if (doSignChecks)
    {
      ASSERT (coord3 >= 0);
    }
  } /* GridCoordinate3DTemplate::set3 */

#ifdef DEBUG_INFO
  /**
   * Get type of third coordinate
   *
   * @return type of third coordinate
   */
  CoordinateType CUDA_DEVICE CUDA_HOST getType3 () const
  {
    return type3;
  } /* GridCoordinate3DTemplate::getType3 */
#endif /* DEBUG_INFO */

  /**
   * Get maximum coordinate
   *
   * @return maximum coordinate
   */
  TcoordType CUDA_DEVICE CUDA_HOST getMax () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    if (coord3 > coord1 && coord3 > coord2)
    {
      return coord3;
    }
    else
    {
      return GridCoordinate2DTemplate<TcoordType, doSignChecks>::getMax ();
    }
  } /* GridCoordinate3DTemplate::getMax */

  /**
   * Operator + for grid coordinates
   *
   * @return sum of two grid coordinates
   */
  GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  operator+ (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    cct1 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    cct2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return GridCoordinate3DTemplate (coord1 + rhs_c1, coord2 + rhs_c2, get3 () + rhs.get3 ()
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
      , GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
      , getType3 ()
#endif /* DEBUG_INFO */
      );
  } /* GridCoordinate3DTemplate::operator+ */

  /**
   * Operator - for grid coordinates
   *
   * @return sum of two grid coordinates
   */
  GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  operator- (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    cct1 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    cct2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return GridCoordinate3DTemplate (coord1 - rhs_c1, coord2 - rhs_c2, get3 () - rhs.get3 ()
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
      , GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
      , getType3 ()
#endif /* DEBUG_INFO */
      );
  } /* GridCoordinate3DTemplate::operator- */

  /**
   * Operator == for grid coordinates
   *
   * @return true, if grid coordinates are equal
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator== (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    cct1 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    cct2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return coord1 == rhs_c1 && coord2 == rhs_c2 && get3 () == rhs.get3 ();
  } /* GridCoordinate3DTemplate::operator== */

  /**
   * Operator != for grid coordinates
   *
   * @return true, if grid coordinates are not equal
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator!= (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    cct1 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    cct2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return coord1 != rhs_c1 || coord2 != rhs_c2 || get3 () == rhs.get3 ();
  } /* GridCoordinate3DTemplate::operator!= */

  /**
   * Operator > for grid coordinates
   *
   * @return true, if first grid coordinates is greater than second
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator> (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    cct1 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    cct2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return coord1 > rhs_c1 && coord2 > rhs_c2 && get3 () > rhs.get3 ();
  } /* GridCoordinate3DTemplate::operator> */

  /**
   * Operator < for grid coordinates
   *
   * @return true, if first grid coordinates is less than second
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator< (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    cct1 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    cct2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return coord1 < rhs_c1 && coord2 < rhs_c2 && get3 () < rhs.get3 ();
  } /* GridCoordinate3DTemplate::operator< */

  /**
   * Operator >= for grid coordinates
   *
   * @return true, if first grid coordinates is greater or equal than second
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST CUDA_DEVICE CUDA_HOST
  operator>= (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    cct1 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    cct2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return coord1 >= rhs_c1 && coord2 >= rhs_c2 && get3 () >= rhs.get3 ();
  } /* GridCoordinate3DTemplate::operator>= */

  /**
   * Operator <= for grid coordinates
   *
   * @return true, if first grid coordinates is less or equal than second
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator<= (const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    cct1 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    cct2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();

    return coord1 <= rhs_c1 && coord2 <= rhs_c2 && get3 () <= rhs.get3 ();
  } /* GridCoordinate3DTemplate::operator<= */

  /**
   * Operator * for grid coordinate and coordinate
   *
   * @return grid coordinate wuth each coordinate multiplied by coordinate
   */
  GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  operator* (TcoordType rhs) const /**< operand */
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    return GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord1 * rhs, coord2 * rhs, get3 () * rhs
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
      , GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
      , getType3 ()
#endif /* DEBUG_INFO */
      );
  } /* GridCoordinate3DTemplate::operator* */

  /**
   * Operator / for grid coordinate and coordinate
   *
   * @return grid coordinate wuth each coordinate divided by coordinate
   */
  GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  operator/ (TcoordType rhs) const /**< operand */
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    return GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord1 / rhs, coord2 / rhs, get3 () / rhs
#ifdef DEBUG_INFO
      , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
      , GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
      , getType3 ()
#endif /* DEBUG_INFO */
      );
  } /* GridCoordinate3DTemplate::operator/ */

  /**
   * Initialize grid coordinate according to coordinate types. This function has same interfaces for all grid coordinate
   * dimensions and allows to create coordinate based on values of three coordinate types.
   *
   * @return initialized grid coordinate
   */
  static GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  initAxesCoordinate (const TcoordType& c1, /**< first coordinate */
                      const TcoordType& c2, /**< second coordinate */
                      const TcoordType& c3, /**< third coordinate */
                      CoordinateType ct1, /**< first coordinate type */
                      CoordinateType ct2, /**< second coordinate type */
                      CoordinateType ct3) /**< third coordinate type */
  {
    ASSERT (ct1 == CoordinateType::X && ct2 == CoordinateType::Y && ct3 == CoordinateType::Z);
    return GridCoordinate3DTemplate<TcoordType, doSignChecks> (c1, c2, c3
#ifdef DEBUG_INFO
                                                               , ct1, ct2, ct3
#endif /* DEBUG_INFO */
                                                               );
  } /* GridCoordinate3DTemplate::initAxesCoordinate */

  /*
   * Friend operators *,+,- declaration
   */
  friend GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
    (::operator* <TcoordType, doSignChecks>)
    (TcoordType lhs, const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs);
  friend GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
    (::operator+ <TcoordType, doSignChecks>)
    (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs,
     const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs);
  friend GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
    (::operator- <TcoordType, doSignChecks>)
    (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs,
     const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs);

  /**
   * Print coordinate
   */
  void print () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    printf ("Coord ("
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      "%f, "
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      "%f, "
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      "%f).\n"
#ifdef DEBUG_INFO
      , coordinateTypeNames[static_cast<uint8_t> (GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ())]
#endif /* DEBUG_INFO */
      , (FPValue) coord1
#ifdef DEBUG_INFO
      , coordinateTypeNames[static_cast<uint8_t> (GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ())]
#endif /* DEBUG_INFO */
      , (FPValue) coord2
#ifdef DEBUG_INFO
      , coordinateTypeNames[static_cast<uint8_t> (getType3 ())]
#endif /* DEBUG_INFO */
      , (FPValue) get3 ());
  } /* GridCoordinate3DTemplate::print */
}; /* GridCoordinate3DTemplate */

/**
 * Coordinate operator * for number and grid coordinate
 *
 * @return grid coordinate, which is the result of multiplication of number and grid coordinate
 */
template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator*
  (TcoordType lhs, /**< first operand (number) */
   const GridCoordinate3DTemplate<TcoordType, doSignChecks>& rhs) /**< second operand */
{
  TcoordType coord1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  TcoordType coord2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (lhs * coord1, lhs * coord2, lhs * rhs.get3 ()
#ifdef DEBUG_INFO
    , rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
    , rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
    , rhs.getType3 ()
#endif /* DEBUG_INFO */
    );
} /* GridCoordinate3DTemplate::operator* */

/**
 * Coodinate operator + for two grid coordinates, only for one of which sign checks are enabled
 *
 * @return result of addition of two grid coordinates
 */
template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+
  (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs, /**< first operand */
   const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs) /**< second operand */
{
#ifdef DEBUG_INFO
  CoordinateType cct1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
  CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::getType1 ();
  ASSERT (cct1 == cct2);
  cct1 = lhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
  cct2 = rhs.GridCoordinate2DTemplate<TcoordType, !doSignChecks>::getType2 ();
  ASSERT (cct1 == cct2);
  ASSERT (lhs.getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
  TcoordType lcoord1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  TcoordType rcoord1 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::get1 ();
  TcoordType lcoord2 = lhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
  TcoordType rcoord2 = rhs.GridCoordinate2DTemplate<TcoordType, !doSignChecks>::get2 ();

  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (lcoord1 + rcoord1,
                                                             lcoord2 + rcoord2,
                                                             lhs.get3 () + rhs.get3 ()
#ifdef DEBUG_INFO
    , lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
    , lhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
    , lhs.getType3 ()
#endif /* DEBUG_INFO */
    );
} /* GridCoordinate3DTemplate::operator+ */

/**
 * Coodinate operator - for two grid coordinates, only for one of which sign checks are enabled
 *
 * @return result of substraction of two grid coordinates
 */
template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator-
  (GridCoordinate3DTemplate<TcoordType, doSignChecks> &lhs, /**< first operand */
   const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs) /**< second operand */
{
#ifdef DEBUG_INFO
  CoordinateType cct1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
  CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::getType1 ();
  ASSERT (cct1 == cct2);
  cct1 = lhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
  cct2 = rhs.GridCoordinate2DTemplate<TcoordType, !doSignChecks>::getType2 ();
  ASSERT (cct1 == cct2);
  ASSERT (lhs.getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
  TcoordType lcoord1 = lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  TcoordType rcoord1 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::get1 ();
  TcoordType lcoord2 = lhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
  TcoordType rcoord2 = rhs.GridCoordinate2DTemplate<TcoordType, !doSignChecks>::get2 ();

  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (lcoord1 - rcoord1,
                                                             lcoord2 - rcoord2,
                                                             lhs.get3 () - rhs.get3 ()
#ifdef DEBUG_INFO
    , lhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
    , lhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
    , lhs.getType3 ()
#endif /* DEBUG_INFO */
    );
} /* GridCoordinate3DTemplate::operator- */

/**
 * Expand 1D coordinate to 3D according to coordinate types. Basically, this is a reverse to initAxesCoordinate
 *
 * @return 3D coordinate corresponding to 1D coordinate from argument
 */
template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expandTo3D
  (const GridCoordinate1DTemplate<TcoordType, doSignChecks> &coord, /**< 1D coordinate to convert */
   CoordinateType t1, /**< first coordinate type of resulting 3D coordinate */
   CoordinateType t2, /**< second coordinate type of resulting 3D coordinate */
   CoordinateType t3) /**< third coordinate type of resulting 3D coordinate */
{
#ifdef DEBUG_INFO
  ASSERT (coord.getType1 () == t1);
  ASSERT (t2 == CoordinateType::NONE);
  ASSERT (t3 == CoordinateType::NONE);
#endif /* DEBUG_INFO */

  TcoordType coord1 = 0;
  TcoordType coord2 = 0;
  TcoordType coord3 = 0;

  if (t1 == CoordinateType::X)
  {
    coord1 = coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  }
  else if (t1 == CoordinateType::Y)
  {
    coord2 = coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  }
  else if (t1 == CoordinateType::Z)
  {
    coord3 = coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  }
  else
  {
    UNREACHABLE;
  }

  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord1, coord2, coord3
#ifdef DEBUG_INFO
    , CoordinateType::X
    , CoordinateType::Y
    , CoordinateType::Z
#endif /* DEBUG_INFO */
    );
} /* expandTo3D */

/**
 * Expand 1D start and end coordinates to 3D according to coordinate types.
 * Start and end coordinates must differ by all axes in order to perform loop iterations over them.
 *
 * @return 3D coordinate corresponding to 1D coordinate from argument
 */
template<class TcoordType, bool doSignChecks>
void CUDA_DEVICE CUDA_HOST expandTo3DStartEnd
  (const GridCoordinate1DTemplate<TcoordType, doSignChecks> &start, /**< 1D start coordinate */
   const GridCoordinate1DTemplate<TcoordType, doSignChecks> &end, /**< 1D end coordinate */
   GridCoordinate3DTemplate<TcoordType, doSignChecks> &start3D, /**< out: 3D start coordinate */
   GridCoordinate3DTemplate<TcoordType, doSignChecks> &end3D, /**< out: 3D end coordinate */
   CoordinateType t1, /**< first coordinate type */
   CoordinateType t2, /**< second coordinate type */
   CoordinateType t3) /**< third coordinate type */
{
#ifdef DEBUG_INFO
  ASSERT (start.getType1 () == end.getType1 ());
  ASSERT (start.getType1 () == t1);
  ASSERT (t2 == CoordinateType::NONE);
  ASSERT (t3 == CoordinateType::NONE);
#endif /* DEBUG_INFO */

  start3D = expandTo3D (start, t1, t2, t3);

  TcoordType coord1 = 1;
  TcoordType coord2 = 1;
  TcoordType coord3 = 1;

  if (t1 == CoordinateType::X)
  {
    coord1 = end.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  }
  else if (t1 == CoordinateType::Y)
  {
    coord2 = end.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  }
  else if (t1 == CoordinateType::Z)
  {
    coord3 = end.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  }
  else
  {
    UNREACHABLE;
  }

  end3D = GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord1, coord2, coord3
#ifdef DEBUG_INFO
    , CoordinateType::X
    , CoordinateType::Y
    , CoordinateType::Z
#endif /* DEBUG_INFO */
    );
} /* expandTo3DStartEnd */

/**
 * Expand 2D coordinate to 3D according to coordinate types. Basically, this is a reverse to initAxesCoordinate
 *
 * @return 3D coordinate corresponding to 2D coordinate from argument
 */
template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expandTo3D
  (const GridCoordinate2DTemplate<TcoordType, doSignChecks> &coord, /**< 2D coordinate to convert */
   CoordinateType t1, /**< first coordinate type of resulting 3D coordinate */
   CoordinateType t2, /**< second coordinate type of resulting 3D coordinate */
   CoordinateType t3) /**< third coordinate type of resulting 3D coordinate */
{
#ifdef DEBUG_INFO
  ASSERT (coord.getType1 () == t1);
  ASSERT (coord.getType2 () == t2);
  ASSERT (t3 == CoordinateType::NONE);
#endif /* DEBUG_INFO */

  TcoordType coord1 = 0;
  TcoordType coord2 = 0;
  TcoordType coord3 = 0;

  if (t1 == CoordinateType::X)
  {
    coord1 = coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  }
  else if (t1 == CoordinateType::Y)
  {
    coord2 = coord.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  }
  else
  {
    UNREACHABLE;
  }

  if (t2 == CoordinateType::Y)
  {
    coord2 = coord.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
  }
  else if (t2 == CoordinateType::Z)
  {
    coord3 = coord.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
  }
  else
  {
    UNREACHABLE;
  }

  return GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord1, coord2, coord3
#ifdef DEBUG_INFO
    , CoordinateType::X
    , CoordinateType::Y
    , CoordinateType::Z
#endif /* DEBUG_INFO */
    );
} /* expandTo3D */

/**
 * Expand 2D start and end coordinates to 3D according to coordinate types.
 * Start and end coordinates must differ by all axes in order to perform loop iterations over them.
 *
 * @return 3D coordinate corresponding to 2D coordinate from argument
 */
template<class TcoordType, bool doSignChecks>
void CUDA_DEVICE CUDA_HOST expandTo3DStartEnd
  (const GridCoordinate2DTemplate<TcoordType, doSignChecks> &start, /**< 2D start coordinate */
   const GridCoordinate2DTemplate<TcoordType, doSignChecks> &end, /**< 2D end coordinate */
   GridCoordinate3DTemplate<TcoordType, doSignChecks> &start3D, /**< out: 3D start coordinate */
   GridCoordinate3DTemplate<TcoordType, doSignChecks> &end3D, /**< out: 3D end coordinate */
   CoordinateType t1, /**< first coordinate type */
   CoordinateType t2, /**< second coordinate type */
   CoordinateType t3) /**< third coordinate type */
{
#ifdef DEBUG_INFO
  ASSERT (start.getType1 () == end.getType1 ());
  ASSERT (start.getType2 () == end.getType2 ());
  ASSERT (start.getType1 () == t1);
  ASSERT (start.getType2 () == t2);
  ASSERT (t3 == CoordinateType::NONE);
#endif /* DEBUG_INFO */

  start3D = expandTo3D (start, t1, t2, t3);

  TcoordType coord1 = 1;
  TcoordType coord2 = 1;
  TcoordType coord3 = 1;

  if (t1 == CoordinateType::X)
  {
    coord1 = end.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  }
  else if (t1 == CoordinateType::Y)
  {
    coord2 = end.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
  }
  else
  {
    UNREACHABLE;
  }

  if (t2 == CoordinateType::Y)
  {
    coord2 = end.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
  }
  else if (t2 == CoordinateType::Z)
  {
    coord3 = end.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
  }
  else
  {
    UNREACHABLE;
  }

  end3D = GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord1, coord2, coord3
#ifdef DEBUG_INFO
    , CoordinateType::X
    , CoordinateType::Y
    , CoordinateType::Z
#endif /* DEBUG_INFO */
    );
} /* expandTo3DStartEnd */

/**
 * Expand 3D coordinate to 3D according to coordinate types. Basically, this is a reverse to initAxesCoordinate
 *
 * @return 3D coordinate corresponding to 3D coordinate from argument
 */
template<class TcoordType, bool doSignChecks>
GridCoordinate3DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST expandTo3D
  (const GridCoordinate3DTemplate<TcoordType, doSignChecks> &coord, /**< 3D coordinate to convert */
   CoordinateType t1, /**< first coordinate type of resulting 3D coordinate */
   CoordinateType t2, /**< second coordinate type of resulting 3D coordinate */
   CoordinateType t3) /**< third coordinate type of resulting 3D coordinate */
{
#ifdef DEBUG_INFO
  ASSERT (coord.getType1 () == t1);
  ASSERT (coord.getType2 () == t2);
  ASSERT (coord.getType3 () == t3);
#endif /* DEBUG_INFO */

  return coord;
} /* expandTo3D */

/**
 * Expand 3D start and end coordinates to 3D according to coordinate types.
 * Start and end coordinates must differ by all axes in order to perform loop iterations over them.
 *
 * @return 3D coordinate corresponding to 3D coordinate from argument
 */
template<class TcoordType, bool doSignChecks>
void CUDA_DEVICE CUDA_HOST expandTo3DStartEnd
  (const GridCoordinate3DTemplate<TcoordType, doSignChecks> &start, /**< 3D start coordinate */
   const GridCoordinate3DTemplate<TcoordType, doSignChecks> &end, /**< 3D end coordinate */
   GridCoordinate3DTemplate<TcoordType, doSignChecks> &start3D, /**< out: 3D start coordinate */
   GridCoordinate3DTemplate<TcoordType, doSignChecks> &end3D, /**< out: 3D end coordinate */
   CoordinateType t1, /**< first coordinate type */
   CoordinateType t2, /**< second coordinate type */
   CoordinateType t3) /**< third coordinate type */
{
#ifdef DEBUG_INFO
  ASSERT (start.getType1 () == end.getType1 ());
  ASSERT (start.getType2 () == end.getType2 ());
  ASSERT (start.getType3 () == end.getType3 ());
  ASSERT (start.getType1 () == t1);
  ASSERT (start.getType2 () == t2);
  ASSERT (start.getType3 () == t3);
#endif /* DEBUG_INFO */

  start3D = start;
  end3D = end;
}

/**
 * 3D grid coordinate with sign checks and integer type for each individual coordinate
 */
typedef GridCoordinate3DTemplate<grid_coord, true> GridCoordinate3D;

/**
 * 3D grid coordinate without sign checks and integer type for each individual coordinate
 */
typedef GridCoordinate3DTemplate<grid_coord, false> GridCoordinateSigned3D;

/**
 * 3D grid coordinate with sign checks and floating point type for each individual coordinate
 */
typedef GridCoordinate3DTemplate<FPValue, true> GridCoordinateFP3D;

/**
 * 3D grid coordinate without sign checks and floating point type for each individual coordinate
 */
typedef GridCoordinate3DTemplate<FPValue, false> GridCoordinateSignedFP3D;

/**
 * Convert floating point coordinate to integer coordinate
 *
 * @return integer coodinate, corresponding to floating point coordinate
 */
template<bool doSignChecks>
GridCoordinate3DTemplate<grid_coord, doSignChecks> CUDA_DEVICE CUDA_HOST
convertCoord (GridCoordinate3DTemplate<FPValue, doSignChecks> coord) /**< floating point coordinate */
{
  ASSERT (((grid_coord) coord.get1 ()) == coord.get1 ());
  ASSERT (((grid_coord) coord.get2 ()) == coord.get2 ());
  ASSERT (((grid_coord) coord.get3 ()) == coord.get3 ());

  return GridCoordinate3DTemplate<grid_coord, doSignChecks> ((grid_coord) coord.get1 (),
                                                             (grid_coord) coord.get2 (),
                                                             (grid_coord) coord.get3 ()
#ifdef DEBUG_INFO
   , coord.GridCoordinate1DTemplate<FPValue, doSignChecks>::getType1 ()
   , coord.GridCoordinate2DTemplate<FPValue, doSignChecks>::getType2 ()
   , coord.getType3 ()
#endif /* DEBUG_INFO */
   );
} /* convertCoord */

/**
 * Convert integer coordinate to floating point coordinate
 *
 * @return floating point coodinate, corresponding to integer coordinate
 */
template<bool doSignChecks>
GridCoordinate3DTemplate<FPValue, doSignChecks> CUDA_DEVICE CUDA_HOST
convertCoord (GridCoordinate3DTemplate<grid_coord, doSignChecks> coord) /**< integer coordinate */
{
  return GridCoordinate3DTemplate<FPValue, doSignChecks> (coord.get1 (),
                                                          coord.get2 (),
                                                          coord.get3 ()
#ifdef DEBUG_INFO
    , coord.GridCoordinate1DTemplate<grid_coord, doSignChecks>::getType1 ()
    , coord.GridCoordinate2DTemplate<grid_coord, doSignChecks>::getType2 ()
    , coord.getType3 ()
#endif /* DEBUG_INFO */
    );
} /* convertCoord */

/**
 * Dimension of 3D grid coordinate
 */
template<class TcoordType, bool doSignChecks>
const Dimension GridCoordinate3DTemplate<TcoordType, doSignChecks>::dimension = Dimension::Dim3;

#endif /* GRID_COORDINATE_3D_H */
