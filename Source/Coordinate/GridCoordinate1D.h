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
   * Copy constructor
   */
  CUDA_DEVICE CUDA_HOST GridCoordinate1DTemplate<TcoordType, doSignChecks>
    (const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& coordinate) /**< new coordinate */
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
   * Operator =
   *
   * @return updated coordinate
   */
  GridCoordinate1DTemplate<TcoordType, doSignChecks> & operator= (const GridCoordinate1DTemplate<TcoordType, !doSignChecks> & rhs) /**< operand */
  {
    coord1 = rhs.get1 ();

#ifdef DEBUG_INFO
    type1 = rhs.getType1 ();
#endif /* DEBUG_INFO */

    if (doSignChecks)
    {
      ASSERT (coord1 >= 0);
    }

    return *this;
  } /* GridCoordinate1DTemplate::operator= */

  /**
   * Calculate total-dimensional coordinate
   *
   * @return total-dimensional coordinate
   */
  CUDA_DEVICE CUDA_HOST
  TcoordType calculateTotalCoord () const
  {
    return coord1;
  } /* GridCoordinate1DTemplate::calculateTotalCoord */

  /**
   * Get first coordinate
   *
   * @return first coordinate
   */
  CUDA_DEVICE CUDA_HOST
  const TcoordType & get1 () const
  {
    return coord1;
  } /* GridCoordinate1DTemplate::get1 */

  /**
   * Set first coordinate
   */
  CUDA_DEVICE CUDA_HOST
  void set1 (const TcoordType& new_c1) /**< new first coordinate */
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
  CUDA_DEVICE CUDA_HOST
  CoordinateType getType1 () const
  {
    return type1;
  } /* GridCoordinate1DTemplate::getType1 */
#endif /* DEBUG_INFO */

  /**
   * Get maximum coordinate
   *
   * @return maximum coordinate
   */
  CUDA_DEVICE CUDA_HOST
  TcoordType getMax () const
  {
    return coord1;
  } /* GridCoordinate1DTemplate::getMax */

  /**
   * Operator + for grid coordinates
   *
   * @return sum of two grid coordinates
   */
  CUDA_DEVICE CUDA_HOST
  GridCoordinate1DTemplate<TcoordType, doSignChecks>
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
   * Operator + for grid coordinates
   *
   * @return sum of two grid coordinates
   */
  CUDA_DEVICE CUDA_HOST
  GridCoordinate1DTemplate<TcoordType, doSignChecks>
  operator+ (const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs) const /**< operand */
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
  CUDA_DEVICE CUDA_HOST
  GridCoordinate1DTemplate<TcoordType, doSignChecks>
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
   * Operator - for grid coordinates
   *
   * @return sum of two grid coordinates
   */
  CUDA_DEVICE CUDA_HOST
  GridCoordinate1DTemplate<TcoordType, doSignChecks>
  operator- (const GridCoordinate1DTemplate<TcoordType, !doSignChecks>& rhs) const /**< operand */
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
  CUDA_DEVICE CUDA_HOST
  bool
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
  CUDA_DEVICE CUDA_HOST
  bool
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
  CUDA_DEVICE CUDA_HOST
  bool
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
  CUDA_DEVICE CUDA_HOST
  bool
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
  CUDA_DEVICE CUDA_HOST
  bool
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
  CUDA_DEVICE CUDA_HOST
  bool
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
   * @return grid coordinate with each coordinate multiplied by coordinate
   */
  CUDA_DEVICE CUDA_HOST
  GridCoordinate1DTemplate<TcoordType, doSignChecks>
  operator* (TcoordType rhs) const /**< operand */
  {
    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (get1 () * rhs
#ifdef DEBUG_INFO
      , getType1 ()
#endif /* DEBUG_INFO */
      );
  } /* GridCoordinate1DTemplate::operator* */

  /**
   * Operator * for grid coordinates
   *
   * @return grid coordinate with each coordinate multiplied by rhs coordinate
   */
  CUDA_DEVICE CUDA_HOST
  GridCoordinate1DTemplate<TcoordType, doSignChecks>
  operator* (GridCoordinate1DTemplate<TcoordType, doSignChecks> rhs) const /**< operand */
  {
    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (get1 () * rhs.get1 ()
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
  CUDA_DEVICE CUDA_HOST
  GridCoordinate1DTemplate<TcoordType, doSignChecks>
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
  static
  CUDA_DEVICE CUDA_HOST
  GridCoordinate1DTemplate<TcoordType, doSignChecks>
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

    ASSERT (ct1 == CoordinateType::Z);
    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (c3
#ifdef DEBUG_INFO
                                                               , ct1
#endif /* DEBUG_INFO */
                                                               );
  } /* GridCoordinate1DTemplate::initAxesCoordinate */

  /**
   * Substract and check for border (if gets below border, then cut off by border)
   *
   * @return substraction result cut off by border
   */
  static
  CUDA_DEVICE CUDA_HOST
  GridCoordinate1DTemplate<TcoordType, doSignChecks>
  subWithBorder (GridCoordinate1DTemplate<TcoordType, doSignChecks> coord1, /**< first operand of substraction */
                 GridCoordinate1DTemplate<TcoordType, doSignChecks> coord2, /**< second operand of substraction */
                 GridCoordinate1DTemplate<TcoordType, doSignChecks> border) /**< border to cut off */
  {
#ifdef DEBUG_INFO
    ASSERT (coord1.getType1 () == coord2.getType1 ()
            && coord1.getType1 () == border.getType1 ());
#endif /* DEBUG_INFO */

    TcoordType c1 = coord1.get1 () - coord2.get1 ();
    if (c1 < border.get1 ())
    {
      c1 = border.get1 ();
    }

    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (c1
#ifdef DEBUG_INFO
                                                               , coord1.getType1 ()
#endif /* DEBUG_INFO */
                                                               );
  } /* GridCoordinate1DTemplate::subWithBorder */

  /**
   * Add and check for border (if gets above border, then cut off by border)
   *
   * @return addition result cut off by border
   */
  static
  CUDA_DEVICE CUDA_HOST
  GridCoordinate1DTemplate<TcoordType, doSignChecks>
  addWithBorder (GridCoordinate1DTemplate<TcoordType, doSignChecks> coord1, /**< first operand of addition */
                 GridCoordinate1DTemplate<TcoordType, doSignChecks> coord2, /**< second operand of addition */
                 GridCoordinate1DTemplate<TcoordType, doSignChecks> border) /**< border to cut off */
  {
#ifdef DEBUG_INFO
    ASSERT (coord1.getType1 () == coord2.getType1 ()
            && coord1.getType1 () == border.getType1 ());
#endif /* DEBUG_INFO */

    TcoordType c1 = coord1.get1 () + coord2.get1 ();
    if (c1 > border.get1 ())
    {
      c1 = border.get1 ();
    }

    return GridCoordinate1DTemplate<TcoordType, doSignChecks> (c1
#ifdef DEBUG_INFO
                                                               , coord1.getType1 ()
#endif /* DEBUG_INFO */
                                                               );
  } /* GridCoordinate1DTemplate::addWithBorder */

  /**
   * Print coordinate to console
   */
  CUDA_DEVICE CUDA_HOST
  void
  print () const
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
CUDA_DEVICE CUDA_HOST
GridCoordinate1DTemplate<grid_coord, doSignChecks>
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
CUDA_DEVICE CUDA_HOST
GridCoordinate1DTemplate<FPValue, doSignChecks>
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
