#include "ParallelGrid.h"

#include <cmath>

#ifdef PARALLEL_GRID

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)

/**
 * Find proportion for computational nodes grid
 */
void
ParallelGridCore::FindProportionForNodeGrid (int &nodeGridSize1, /**< out: first axis nodes grid size */
                                             int &nodeGridSize2, /**< out: second axis nodes grid size */
                                             int &left, /**< out: number of left unused computational nodes */
                                             FPValue alpha) /**< preferred proportion between second and first axis */
{
  /*
   * Bad case, too many nodes left unused. Let's change proportion.
   */

  int min_left = left;
  int min_size1 = nodeGridSize1;
  int min_size2 = nodeGridSize2;
  FPValue min_alpha = ((FPValue) min_size2) / ((FPValue) min_size1);

  for (int size1 = 2; size1 <= totalProcCount / 2; ++size1)
  {
    int size2 = totalProcCount / size1;
    int left_new = totalProcCount - (size1 * size2);

    if (left_new < min_left)
    {
      min_left = left_new;
      min_size1 = size1;
      min_size2 = size2;
      min_alpha = ((FPValue) size2) / ((FPValue) size1);
    }
    else if (left_new == min_left)
    {
      FPValue new_alpha = ((FPValue) size2) / ((FPValue) size1);

      FPValue diff_alpha = fabs (new_alpha - alpha);
      FPValue diff_alpha_min = fabs (min_alpha - alpha);

      if (diff_alpha < diff_alpha_min)
      {
        min_left = left_new;
        min_size1 = size1;
        min_size2 = size2;
        min_alpha = ((FPValue) size2) / ((FPValue) size1);
      }
    }
  }

  nodeGridSize1 = min_size1;
  nodeGridSize2 = min_size2;
  left = min_left;
} /* ParallelGridCore::FindProportionForNodeGrid */

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
                               int &nodeGridSize2, /**< out: second axis nodes grid size */
                               int &left) /**< out: number of left unused computational nodes */
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

    FPValue m_fp = ((FPValue) totalProcCount) / ((FPValue) n);
    grid_coord m = (grid_coord) m_fp;

    FPValue b1_fp = ((FPValue) size2) / ((FPValue)m);
    grid_coord b1 = (grid_coord) b1_fp;

    if (m == m_fp && b1 == b1_fp)
    {
      allowedPairs.push_back (Pair (n, m));
    }
  }

  ASSERT (allowedPairs.size () > 0);

  FPValue optimal_n = sqrt (((FPValue) size1 * totalProcCount) / ((FPValue) size2));

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

  left = 0;
} /* ParallelGridCore::initOptimal */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ) ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ */

#endif /* PARALLEL_GRID */
