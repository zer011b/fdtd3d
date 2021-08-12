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

#include "ParallelGrid.h"

#include <cmath>

#ifdef PARALLEL_GRID

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)

/**
 * Helper struct for pairs
 */
struct Pair
{
  grid_coord n; /**< first axis nodes grid size */
  grid_coord m; /**< second axis nodes grid size */

  /**
   * Constructor with nodes grid sizes
   */
  Pair (grid_coord newN = 0, /**< first axis nodes grid size */
        grid_coord newM = 0) /**< second axis nodes grid size */
    : n (newN)
  , m (newM)
  {
  } /* Pair */
}; /* Pair */

/**
 * Initialize parallel grid virtual topology as optimal for current number of processes and specified grid sizes
 */
void
ParallelGridCore::initOptimal (grid_coord size1, /**< grid size by first axis */
                               grid_coord size2, /**< grid size by second axis */
                               int &nodeGridSize1, /**< out: first axis nodes grid size */
                               int &nodeGridSize2) /**< out: second axis nodes grid size */
{
  /*
   * Find allowed pairs of node grid sizes
   */
  std::vector<Pair> allowedPairs;

  grid_coord gcd1 = ParallelGridCore::greatestCommonDivider (size1, (grid_coord) totalProcCount);

  for (grid_coord n = 1; n <= gcd1; ++n)
  {
    if (gcd1 % n != 0)
    {
      continue;
    }

    DOUBLE m_fp = ((DOUBLE) totalProcCount) / ((DOUBLE) n);
    grid_coord m = (grid_coord) m_fp;

    DOUBLE b1_fp = ((DOUBLE) size2) / ((DOUBLE)m);
    grid_coord b1 = (grid_coord) b1_fp;

    if (m == m_fp && b1 == b1_fp)
    {
      allowedPairs.push_back (Pair (n, m));
    }
  }

  ASSERT (allowedPairs.size () > 0);

  DOUBLE optimal_n = sqrt (((DOUBLE) size1 * totalProcCount) / ((DOUBLE) size2));

  Pair pair_left = allowedPairs[0];
  Pair pair_right = allowedPairs[0];

  for (std::vector<Pair>::iterator it = allowedPairs.begin ();
       it != allowedPairs.end ();
       ++it)
  {
    if (it->n < optimal_n)
    {
      pair_left = *it;
    }
    else
    {
      pair_right = *it;
      break;
    }
  }

  /*
   * This heavily depends on the parallel grid sharing scheme
   */
#define func(pair) \
  ((size1) / (pair.n) + (size2) / (pair.m))

  if (func (pair_left) < func (pair_right))
  {
    nodeGridSize1 = pair_left.n;
    nodeGridSize2 = pair_left.m;
  }
  else
  {
    nodeGridSize1 = pair_right.n;
    nodeGridSize2 = pair_right.m;
  }

#undef func
} /* ParallelGridCore::initOptimal */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ) ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ */

#endif /* PARALLEL_GRID */
