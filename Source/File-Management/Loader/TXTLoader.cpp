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

#include <iostream>

#include "TXTLoader.h"

/**
 * Skip words with indexes loaded from file
 *
 * @return index of field values
 */
template<>
uint32_t
TXTLoader<GridCoordinate1D>::skipIndexes (GridCoordinate1D pos, /**< position */
                                          const std::vector<std::string> &tokens) /**< words from line */
{
  uint32_t word_index = 0;
  ASSERT (pos.get1 () == STOI (tokens[word_index].c_str ()));
  ++word_index;
  return word_index;
} /* TXTLoader::skipIndexes */

#if defined (MODE_DIM2) || defined (MODE_DIM3)

/**
 * Skip words with indexes loaded from file
 *
 * @return index of field values
 */
template<>
uint32_t
TXTLoader<GridCoordinate2D>::skipIndexes (GridCoordinate2D pos, /**< position */
                                          const std::vector<std::string> &tokens) /**< words from line */
{
  uint32_t word_index = 0;
  ASSERT (pos.get1 () == STOI (tokens[word_index].c_str ()));
  ++word_index;
  ASSERT (pos.get2 () == STOI (tokens[word_index].c_str ()));
  ++word_index;
  return word_index;
} /* TXTLoader::skipIndexes */

#endif /* MODE_DIM2 || MODE_DIM3 */

#if defined (MODE_DIM3)

/**
 * Skip words with indexes loaded from file
 *
 * @return index of field values
 */
template<>
uint32_t
TXTLoader<GridCoordinate3D>::skipIndexes (GridCoordinate3D pos, /**< position */
                                          const std::vector<std::string> &tokens) /**< words from line */
{
  uint32_t word_index = 0;
  ASSERT (pos.get1 () == STOI (tokens[word_index].c_str ()));
  ++word_index;
  ASSERT (pos.get2 () == STOI (tokens[word_index].c_str ()));
  ++word_index;
  ASSERT (pos.get3 () == STOI (tokens[word_index].c_str ()));
  ++word_index;
  return word_index;
} /* TXTLoader::skipIndexes */

#endif /* MODE_DIM3 */
