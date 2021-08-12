/*
 * Copyright (C) 2016 Gleb Balykov
 *
 * This file is part of fdtd3d.
 *
 * fdtd3d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * fdtd3d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with fdtd3d; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef GRID_COORDINATE_3D_H
#define GRID_COORDINATE_3D_H

#include "GridCoordinate2D.h"

template<class TcoordType, bool doSignChecks>
class GridCoordinate3DTemplate;

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
   * Copy constructor
   */
  CUDA_DEVICE CUDA_HOST GridCoordinate3DTemplate<TcoordType, doSignChecks>
    (const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& pos) /**< new coordinate */
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
   * Operator =
   *
   * @return updated coordinate
   */
  GridCoordinate3DTemplate<TcoordType, doSignChecks> & operator= (const GridCoordinate3DTemplate<TcoordType, !doSignChecks> & rhs) /**< operand */
  {
    GridCoordinate1DTemplate<TcoordType, doSignChecks>::coord1 = rhs.get1 ();
    GridCoordinate2DTemplate<TcoordType, doSignChecks>::coord2 = rhs.get2 ();
    coord3 = rhs.get3 ();

#ifdef DEBUG_INFO
    GridCoordinate1DTemplate<TcoordType, doSignChecks>::type1 = rhs.getType1 ();
    GridCoordinate2DTemplate<TcoordType, doSignChecks>::type2 = rhs.getType2 ();
    type3 = rhs.getType3 ();

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
  CUDA_DEVICE CUDA_HOST
  const TcoordType & get3 () const
  {
    return coord3;
  } /* GridCoordinate3DTemplate::get3 */

  /**
   * Set third coordinate
   */
  CUDA_DEVICE CUDA_HOST
  void set3 (const TcoordType& new_c3) /**< new third coordinate */
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
  CUDA_DEVICE CUDA_HOST
  CoordinateType getType3 () const
  {
    return type3;
  } /* GridCoordinate3DTemplate::getType3 */
#endif /* DEBUG_INFO */

  /**
   * Get maximum coordinate
   *
   * @return maximum coordinate
   */
  CUDA_DEVICE CUDA_HOST
  TcoordType getMax () const
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
  CUDA_DEVICE CUDA_HOST
  GridCoordinate3DTemplate<TcoordType, doSignChecks>
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
   * Operator + for grid coordinates
   *
   * @return sum of two grid coordinates
   */
  CUDA_DEVICE CUDA_HOST
  GridCoordinate3DTemplate<TcoordType, doSignChecks>
  operator+ (const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    cct1 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    cct2 = rhs.GridCoordinate2DTemplate<TcoordType, !doSignChecks>::getType2 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, !doSignChecks>::get2 ();

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
  CUDA_DEVICE CUDA_HOST
  GridCoordinate3DTemplate<TcoordType, doSignChecks>
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
   * Operator - for grid coordinates
   *
   * @return sum of two grid coordinates
   */
  CUDA_DEVICE CUDA_HOST
  GridCoordinate3DTemplate<TcoordType, doSignChecks>
  operator- (const GridCoordinate3DTemplate<TcoordType, !doSignChecks>& rhs) const /**< operand */
  {
#ifdef DEBUG_INFO
    CoordinateType cct1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ();
    CoordinateType cct2 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::getType1 ();
    ASSERT (cct1 == cct2);
    cct1 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ();
    cct2 = rhs.GridCoordinate2DTemplate<TcoordType, !doSignChecks>::getType2 ();
    ASSERT (cct1 == cct2);
    ASSERT (getType3 () == rhs.getType3 ());
#endif /* DEBUG_INFO */
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType rhs_c1 = rhs.GridCoordinate1DTemplate<TcoordType, !doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs_c2 = rhs.GridCoordinate2DTemplate<TcoordType, !doSignChecks>::get2 ();

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
  CUDA_DEVICE CUDA_HOST
  bool
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
  CUDA_DEVICE CUDA_HOST
  bool
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

    return coord1 != rhs_c1 || coord2 != rhs_c2 || get3 () != rhs.get3 ();
  } /* GridCoordinate3DTemplate::operator!= */

  /**
   * Operator > for grid coordinates
   *
   * @return true, if first grid coordinates is greater than second
   *         false, otherwise
   */
  CUDA_DEVICE CUDA_HOST
  bool
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
  CUDA_DEVICE CUDA_HOST
  bool
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
  CUDA_DEVICE CUDA_HOST
  bool
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
  CUDA_DEVICE CUDA_HOST
  bool
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
  CUDA_DEVICE CUDA_HOST
  GridCoordinate3DTemplate<TcoordType, doSignChecks>
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
   * Operator * for two grid coordinates
   *
   * @return grid coordinate with each coordinate multiplied by rhs coordinate
   */
  CUDA_DEVICE CUDA_HOST
  GridCoordinate3DTemplate<TcoordType, doSignChecks>
  operator* (GridCoordinate3DTemplate<TcoordType, doSignChecks> rhs) const /**< operand */
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
    TcoordType rhs1 = rhs.GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    TcoordType rhs2 = rhs.GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    return GridCoordinate3DTemplate<TcoordType, doSignChecks> (coord1 * rhs1, coord2 * rhs2, get3 () * rhs.get3 ()
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
  CUDA_DEVICE CUDA_HOST
  GridCoordinate3DTemplate<TcoordType, doSignChecks>
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
  static
  CUDA_DEVICE CUDA_HOST
  GridCoordinate3DTemplate<TcoordType, doSignChecks>
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

  /**
   * Substract and check for border (if gets below border, then cut off by border)
   *
   * @return substraction result cut off by border
   */
  static
  CUDA_DEVICE CUDA_HOST
  GridCoordinate3DTemplate<TcoordType, doSignChecks>
  subWithBorder (GridCoordinate3DTemplate<TcoordType, doSignChecks> coord1, /**< first operand of substraction */
                 GridCoordinate3DTemplate<TcoordType, doSignChecks> coord2, /**< second operand of substraction */
                 GridCoordinate3DTemplate<TcoordType, doSignChecks> border) /**< border to cut off */
  {
#ifdef DEBUG_INFO
    ASSERT (coord1.getType1 () == coord2.getType1 ()
            && coord1.getType1 () == border.getType1 ());
    ASSERT (coord1.getType2 () == coord2.getType2 ()
            && coord1.getType2 () == border.getType2 ());
    ASSERT (coord1.getType3 () == coord2.getType3 ()
            && coord1.getType3 () == border.getType3 ());
#endif /* DEBUG_INFO */

    TcoordType c1 = coord1.get1 () - coord2.get1 ();
    if (c1 < border.get1 ())
    {
      c1 = border.get1 ();
    }

    TcoordType c2 = coord1.get2 () - coord2.get2 ();
    if (c2 < border.get2 ())
    {
      c2 = border.get2 ();
    }

    TcoordType c3 = coord1.get3 () - coord2.get3 ();
    if (c3 < border.get3 ())
    {
      c3 = border.get3 ();
    }

    return GridCoordinate3DTemplate<TcoordType, doSignChecks> (c1, c2, c3
#ifdef DEBUG_INFO
                                                               , coord1.getType1 ()
                                                               , coord1.getType2 ()
                                                               , coord1.getType3 ()
#endif /* DEBUG_INFO */
                                                               );
  } /* GridCoordinate3DTemplate::subWithBorder */

  /**
   * Add and check for border (if gets above border, then cut off by border)
   *
   * @return addition result cut off by border
   */
  static
  CUDA_DEVICE CUDA_HOST
  GridCoordinate3DTemplate<TcoordType, doSignChecks>
  addWithBorder (GridCoordinate3DTemplate<TcoordType, doSignChecks> coord1, /**< first operand of addition */
                 GridCoordinate3DTemplate<TcoordType, doSignChecks> coord2, /**< second operand of addition */
                 GridCoordinate3DTemplate<TcoordType, doSignChecks> border) /**< border to cut off */
  {
#ifdef DEBUG_INFO
    ASSERT (coord1.getType1 () == coord2.getType1 ()
            && coord1.getType1 () == border.getType1 ());
    ASSERT (coord1.getType2 () == coord2.getType2 ()
            && coord1.getType2 () == border.getType2 ());
    ASSERT (coord1.getType3 () == coord2.getType3 ()
            && coord1.getType3 () == border.getType3 ());
#endif /* DEBUG_INFO */

    TcoordType c1 = coord1.get1 () + coord2.get1 ();
    if (c1 > border.get1 ())
    {
      c1 = border.get1 ();
    }

    TcoordType c2 = coord1.get2 () + coord2.get2 ();
    if (c2 > border.get2 ())
    {
      c2 = border.get2 ();
    }

    TcoordType c3 = coord1.get3 () + coord2.get3 ();
    if (c3 > border.get3 ())
    {
      c3 = border.get3 ();
    }

    return GridCoordinate3DTemplate<TcoordType, doSignChecks> (c1, c2, c3
#ifdef DEBUG_INFO
                                                               , coord1.getType1 ()
                                                               , coord1.getType2 ()
                                                               , coord1.getType3 ()
#endif /* DEBUG_INFO */
                                                               );
  } /* GridCoordinate3DTemplate::addWithBorder */

  /**
   * Print coordinate to console
   */
  CUDA_HOST
  void
  print () const
  {
    TcoordType coord1 = GridCoordinate1DTemplate<TcoordType, doSignChecks>::get1 ();
    TcoordType coord2 = GridCoordinate2DTemplate<TcoordType, doSignChecks>::get2 ();
    printf ("Coord ("
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      FP_MOD ", "
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      FP_MOD ", "
#ifdef DEBUG_INFO
      "%s : "
#endif /* DEBUG_INFO */
      FP_MOD " ).\n"
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

  /**
   * Get zero coordinate with the same types
   *
   * @return zero coordinate with the same types
   */
  CUDA_DEVICE CUDA_HOST
  GridCoordinate3DTemplate<TcoordType, doSignChecks> getZero () const
  {
    return GridCoordinate3DTemplate<TcoordType, doSignChecks> (0,
                                                               0,
                                                               0
#ifdef DEBUG_INFO
                                                               , GridCoordinate1DTemplate<TcoordType, doSignChecks>::getType1 ()
                                                               , GridCoordinate2DTemplate<TcoordType, doSignChecks>::getType2 ()
                                                               , getType3 ()
#endif /* DEBUG_INFO */
                                                               );
  } /* getZero */
}; /* GridCoordinate3DTemplate */

/**
 * Expand 1D coordinate to 3D according to coordinate types. Basically, this is a reverse to initAxesCoordinate
 *
 * @return 3D coordinate corresponding to 1D coordinate from argument
 */
template<class TcoordType, bool doSignChecks>
CUDA_DEVICE CUDA_HOST
GridCoordinate3DTemplate<TcoordType, doSignChecks> expandTo3D
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
CUDA_DEVICE CUDA_HOST
void
expandTo3DStartEnd
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
CUDA_DEVICE CUDA_HOST
GridCoordinate3DTemplate<TcoordType, doSignChecks>
expandTo3D
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
CUDA_DEVICE CUDA_HOST
void
expandTo3DStartEnd
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
CUDA_DEVICE CUDA_HOST
GridCoordinate3DTemplate<TcoordType, doSignChecks>
expandTo3D
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
CUDA_DEVICE CUDA_HOST
void
expandTo3DStartEnd
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
CUDA_DEVICE CUDA_HOST
GridCoordinate3DTemplate<grid_coord, doSignChecks>
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
CUDA_DEVICE CUDA_HOST
GridCoordinate3DTemplate<FPValue, doSignChecks>
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
