/*
 * Copyright (C) 2017 Gleb Balykov
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

#include <iomanip>
#include <limits>

#include "TXTDumper.h"

/**
 * Save one line of txt file
 */
template <>
void
TXTDumper<GridCoordinate1D>::printLine (std::ofstream & file, /**< file to save to */
                                        const GridCoordinate1D & pos) /**< coordinate */
{
  file << pos.get1 () << " ";
} /* TXTDumper::printLine */

#if defined (MODE_DIM2) || defined (MODE_DIM3)

/**
 * Save one line of txt file
 */
template <>
void
TXTDumper<GridCoordinate2D>::printLine (std::ofstream & file, /**< file to save to */
                                        const GridCoordinate2D & pos) /**< coordinate */
{
  file << pos.get1 () << " " << pos.get2 () << " ";
} /* TXTDumper::printLine */

#endif /* MODE_DIM2) || MODE_DIM3 */

#if defined (MODE_DIM3)

/**
 * Save one line of txt file
 */
template <>
void
TXTDumper<GridCoordinate3D>::printLine (std::ofstream & file, /**< file to save to */
                                        const GridCoordinate3D & pos) /**< coordinate */
{
  file << pos.get1 () << " " << pos.get2 () << " " << pos.get3 () << " ";
} /* TXTDumper::printLine */

#endif /* MODE_DIM3 */
