#ifndef GRID_COORDINATE_1D_H
#define GRID_COORDINATE_1D_H

#include "FieldValue.h"
#include "Settings.h"

/**
 * Dimension of coordinate
 */
ENUM_CLASS (Dimension, uint8_t,
  Dim1, /**< 1D */
  Dim2, /**< 2D */
  Dim3 /**< 3D */
); /* Dimension */

/**
 * Type of each individual coordinate, for example, for 2D-XY coordinates have types X and Y
 */
ENUM_CLASS (CoordinateType, uint8_t,
  NONE,
  X,
  Y,
  Z,
  COUNT
); /* CoordinateType */

extern const char * coordinateTypeNames[];

template<class TcoordType, bool doSignChecks>
class GridCoordinate1DTemplate;

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
operator* (TcoordType lhs, const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
operator+ (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs,
           const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs);

template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
operator- (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs,
           const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs);

/**
 * 1-dimensional coordinate in the grid
 */
template<class TcoordType, bool doSignChecks>
class GridCoordinate1DTemplate
{
protected:

  /**
   * First coordinate
   */
  TcoordType coord1;

#ifdef DEBUG_INFO
  /**
   * Type of first coordinate
   */
  CoordinateType type1;
#endif /* DEBUG_INFO */

public:

  /**
   * Dimension of this grid coordinate
   */
  static const Dimension dimension;

  /**
   * Default constructor
   */
  CUDA_DEVICE CUDA_HOST GridCoordinate1DTemplate<TcoordType, doSignChecks> ()
    : coord1 (0)
#ifdef DEBUG_INFO
    , type1 (CoordinateType::NONE)
#endif /* DEBUG_INFO */
  {
  } /* GridCoordinate1DTemplate */

  /**
   * Constructor with specified coordinate and its type
   */
  explicit CUDA_DEVICE CUDA_HOST GridCoordinate1DTemplate<TcoordType, doSignChecks>
    (const TcoordType& c1 /**< first coordinate */
#ifdef DEBUG_INFO
     , CoordinateType t1 /**< coordinate type */
#endif /* DEBUG_INFO */
     )
    : coord1 (c1)
#ifdef DEBUG_INFO
    , type1 (t1)
#endif /* DEBUG_INFO */
  {
    if (doSignChecks)
    {
      ASSERT (coord1 >= 0);
    }
  } /* GridCoordinate1DTemplate */

  /**
   * Constructor with interface similar for all coordinate dimensions
   */
  explicit CUDA_DEVICE CUDA_HOST GridCoordinate1DTemplate<TcoordType, doSignChecks>
    (const TcoordType& c1, /**< coordinate */
     const TcoordType& tmp1, /**< unused coordinate */
     const TcoordType& tmp2 /**< unused coordinate */
#ifdef DEBUG_INFO
     , CoordinateType t1 /**< coodinate type */
     , CoordinateType t2 /**< unused coordinate type */
     , CoordinateType t3 /**< unused coordinate type */
#endif /* DEBUG_INFO */
     )
    : coord1 (c1)
#ifdef DEBUG_INFO
    , type1 (t1)
#endif /* DEBUG_INFO */
  {
    if (doSignChecks)
    {
      ASSERT (coord1 >= 0);
    }
  } /* GridCoordinate1DTemplate */

  /**
   * Copy constructor
   */
  CUDA_DEVICE CUDA_HOST GridCoordinate1DTemplate<TcoordType, doSignChecks>
    (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& coordinate) /**< new coordinate */
    : coord1 (coordinate.get1 ())
#ifdef DEBUG_INFO
    , type1 (coordinate.getType1 ())
#endif /* DEBUG_INFO */
  {
    if (doSignChecks)
    {
      ASSERT (coord1 >= 0);
    }
  } /* GridCoordinate1DTemplate */

  /**
   * Destructor
   */
  CUDA_DEVICE CUDA_HOST ~GridCoordinate1DTemplate<TcoordType, doSignChecks> () {};

  /**
   * Calculate total-dimensional coordinate
   *
   * @return total-dimensional coordinate
   */
  TcoordType CUDA_DEVICE CUDA_HOST calculateTotalCoord () const
  {
    return coord1;
  } /* GridCoordinate1DTemplate::calculateTotalCoord */

  /**
   * Get first coordinate
   *
   * @return first coordinate
   */
  const TcoordType& CUDA_DEVICE CUDA_HOST get1 () const
  {
    return coord1;
  } /* GridCoordinate1DTemplate::get1 */

  /**
   * Set first coordinate
   */
  void CUDA_DEVICE CUDA_HOST set1 (const TcoordType& new_c1) /**< new first coordinate */
  {
    coord1 = new_c1;
    if (doSignChecks)
    {
      ASSERT (coord1 >= 0);
    }
  } /* GridCoordinate1DTemplate::set1 */

#ifdef DEBUG_INFO
  /**
   * Get type of first coordinate
   *
   * @return type of first coordinate
   */
  CoordinateType CUDA_DEVICE CUDA_HOST getType1 () const
  {
    return type1;
  } /* GridCoordinate1DTemplate::getType1 */
#endif /* DEBUG_INFO */

  /**
   * Get maximum coordinate
   *
   * @return maximum coordinate
   */
  TcoordType CUDA_DEVICE CUDA_HOST getMax () const
  {
    return coord1;
  } /* GridCoordinate1DTemplate::getMax */

  /**
   * Operator + for grid coordinates
   *
   * @return sum of two grid coordinates
   */
  GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  operator+ (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (get1 () + rhs.get1 ()
#ifdef DEBUG_INFO
      , getType1 ()
#endif /* DEBUG_INFO */
      );
  } /* GridCoordinate1DTemplate::operator+ */

  /**
   * Operator - for grid coordinates
   *
   * @return sum of two grid coordinates
   */
  GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  operator- (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (get1 () - rhs.get1 ()
#ifdef DEBUG_INFO
      , getType1 ()
#endif /* DEBUG_INFO */
      );
  } /* GridCoordinate1DTemplate::operator- */

  /**
   * Operator == for grid coordinates
   *
   * @return true, if grid coordinates are equal
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator== (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return get1 () == rhs.get1 ();
  } /* GridCoordinate1DTemplate::operator== */

  /**
   * Operator != for grid coordinates
   *
   * @return true, if grid coordinates are not equal
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator!= (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return get1 () != rhs.get1 ();
  } /* GridCoordinate1DTemplate::operator!= */

  /**
   * Operator > for grid coordinates
   *
   * @return true, if first grid coordinates is greater than second
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator> (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return get1 () > rhs.get1 ();
  } /* GridCoordinate1DTemplate::operator> */

  /**
   * Operator < for grid coordinates
   *
   * @return true, if first grid coordinates is less than second
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator< (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return get1 () < rhs.get1 ();
  } /* GridCoordinate1DTemplate::operator< */

  /**
   * Operator >= for grid coordinates
   *
   * @return true, if first grid coordinates is greater or equal than second
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator>= (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return get1 () >= rhs.get1 ();
  } /* GridCoordinate1DTemplate::operator>= */

  /**
   * Operator <= for grid coordinates
   *
   * @return true, if first grid coordinates is less or equal than second
   *         false, otherwise
   */
  bool CUDA_DEVICE CUDA_HOST
  operator<= (const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    ASSERT (getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
    return get1 () <= rhs.get1 ();
  } /* GridCoordinate1DTemplate::operator<= */

  /**
   * Operator * for grid coordinate and coordinate
   *
   * @return grid coordinate wuth each coordinate multiplied by coordinate
   */
  GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  operator* (TcoordType rhs) const /**< operand */
  {
    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (get1 () * rhs
#ifdef DEBUG_INFO
      , getType1 ()
#endif /* DEBUG_INFO */
      );
  } /* GridCoordinate1DTemplate::operator* */

  /**
   * Operator / for grid coordinate and coordinate
   *
   * @return grid coordinate wuth each coordinate divided by coordinate
   */
  GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  operator/ (TcoordType rhs) const /**< operand */
  {
    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (get1 () / rhs
#ifdef DEBUG_INFO
      , getType1 ()
#endif /* DEBUG_INFO */
      );
  } /* GridCoordinate1DTemplate::operator/ */

  /**
   * Initialize grid coordinate according to coordinate types. This function has same interfaces for all grid coordinate
   * dimensions and allows to create coordinate based on values of three coordinate types.
   *
   * For example, to initialize 1D-Z coordinate, ct1 should be CoordinateType::Z and c3 be the coordinate itself.
   *
   * @return initialized grid coordinate
   */
  static GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  initAxesCoordinate (const TcoordType& c1, /**< first coordinate */
                      const TcoordType& c2, /**< second coordinate */
                      const TcoordType& c3, /**< third coordinate */
                      CoordinateType ct1, /**< first coordinate type */
                      CoordinateType ct2, /**< second coordinate type */
                      CoordinateType ct3) /**< third coordinate type */
  {
    ASSERT (ct2 == CoordinateType::NONE && ct3 == CoordinateType::NONE);

    if (ct1 == CoordinateType::X)
    {
      return GridCoordinate1DTemplate<TcoordType, doSignChecks> (c1
#ifdef DEBUG_INFO
                                                                 , ct1
#endif /* DEBUG_INFO */
                                                                 );
    }
    else if (ct1 == CoordinateType::Y)
    {
      return GridCoordinate1DTemplate<TcoordType, doSignChecks> (c2
#ifdef DEBUG_INFO
                                                                 , ct1
#endif /* DEBUG_INFO */
                                                                 );
    }
    else if (ct1 == CoordinateType::Z)
    {
      return GridCoordinate1DTemplate<TcoordType, doSignChecks> (c3
#ifdef DEBUG_INFO
                                                                 , ct1
#endif /* DEBUG_INFO */
                                                                 );
    }
    else
    {
      UNREACHABLE;
    }
  } /* GridCoordinate1DTemplate::initAxesCoordinate */

  /*
   * Friend operators *,+,- declaration
   */
  friend GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  (::operator* <TcoordType, doSignChecks>)
    (TcoordType lhs, const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs);
  friend GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  (::operator+ <TcoordType, doSignChecks>)
    (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs,
     const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs);
  friend GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST
  (::operator- <TcoordType, doSignChecks>)
    (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs,
     const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs);

  /**
   * Print coordinate
   */
  void print () const
  {
    printf ("Coord ("
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      "%f ).\n"
#ifdef DEBUG_INFO
      , coordinateTypeNames[static_cast<uint8_t> (getType1 ())]
#endif /* DEBUG_INFO */
      , (FPValue) get1 ());
  } /* GridCoordinate1DTemplate::print */
}; /* GridCoordinate1DTemplate */

/**
 * Coordinate operator * for number and grid coordinate
 *
 * @return grid coordinate, which is the result of multiplication of number and grid coordinate
 */
template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator*
  (TcoordType lhs, /**< first operand (number) */
   const GridCoordinate1DTemplate<TcoordType, doSignChecks>& rhs) /**< second operand */
{
  return GridCoordinate1DTemplate<TcoordType, doSignChecks> (lhs * rhs.get1 ()
#ifdef DEBUG_INFO
    , rhs.getType1 ()
#endif /* DEBUG_INFO */
    );
} /* GridCoordinate1DTemplate::operator* */

/**
 * Coodinate operator + for two grid coordinates, only for one of which sign checks are enabled
 *
 * @return result of addition of two grid coordinates
 */
template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator+
  (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs, /**< first operand */
   const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs) /**< second operand */
{
#ifdef DEBUG_INFO
  ASSERT (lhs.getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
  return GridCoordinate1DTemplate<TcoordType, doSignChecks> (lhs.get1 () + rhs.get1 ()
#ifdef DEBUG_INFO
    , lhs.getType1 ()
#endif /* DEBUG_INFO */
    );
} /* GridCoordinate1DTemplate::operator+ */

/**
 * Coodinate operator - for two grid coordinates, only for one of which sign checks are enabled
 *
 * @return result of substraction of two grid coordinates
 */
template<class TcoordType, bool doSignChecks>
GridCoordinate1DTemplate<TcoordType, doSignChecks> CUDA_DEVICE CUDA_HOST operator-
  (GridCoordinate1DTemplate<TcoordType, doSignChecks> &lhs, /**< first operand */
   const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs) /**< second operand */
{
#ifdef DEBUG_INFO
  ASSERT (lhs.getType1 () == rhs.getType1 ());
#endif /* DEBUG_INFO */
  return GridCoordinate1DTemplate<TcoordType, doSignChecks> (lhs.get1 () - rhs.get1 ()
#ifdef DEBUG_INFO
    , lhs.getType1 ()
#endif /* DEBUG_INFO */
    );
} /* GridCoordinate1DTemplate::operator- */

/**
 * 1D grid coordinate with sign checks and integer type for each individual coordinate
 */
typedef GridCoordinate1DTemplate<grid_coord, true> GridCoordinate1D;

/**
 * 1D grid coordinate without sign checks and integer type for each individual coordinate
 */
typedef GridCoordinate1DTemplate<grid_coord, false> GridCoordinateSigned1D;

/**
 * 1D grid coordinate with sign checks and floating point type for each individual coordinate
 */
typedef GridCoordinate1DTemplate<FPValue, true> GridCoordinateFP1D;

/**
 * 1D grid coordinate without sign checks and floating point type for each individual coordinate
 */
typedef GridCoordinate1DTemplate<FPValue, false> GridCoordinateSignedFP1D;

/**
 * Convert floating point coordinate to integer coordinate
 *
 * @return integer coodinate, corresponding to floating point coordinate
 */
template<bool doSignChecks>
GridCoordinate1DTemplate<grid_coord, doSignChecks> CUDA_DEVICE CUDA_HOST
convertCoord (GridCoordinate1DTemplate<FPValue, doSignChecks> coord) /**< floating point coordinate */
{
  ASSERT (((grid_coord) coord.get1 ()) == coord.get1 ());
  return GridCoordinate1DTemplate<grid_coord, doSignChecks> ((grid_coord) coord.get1 ()
#ifdef DEBUG_INFO
    , coord.getType1 ()
#endif /* DEBUG_INFO */
    );
} /* convertCoord */

/**
 * Convert integer coordinate to floating point coordinate
 *
 * @return floating point coodinate, corresponding to integer coordinate
 */
template<bool doSignChecks>
GridCoordinate1DTemplate<FPValue, doSignChecks> CUDA_DEVICE CUDA_HOST
convertCoord (GridCoordinate1DTemplate<grid_coord, doSignChecks> coord) /**< integer coordinate */
{
  return GridCoordinate1DTemplate<FPValue, doSignChecks> (coord.get1 ()
#ifdef DEBUG_INFO
    , coord.getType1 ()
#endif /* DEBUG_INFO */
    );
} /* convertCoord */

/**
 * Dimension of 1D grid coordinate
 */
template<class TcoordType, bool doSignChecks>
const Dimension GridCoordinate1DTemplate<TcoordType, doSignChecks>::dimension = Dimension::Dim1;

#endif /* GRID_COORDINATE_1D_H */
