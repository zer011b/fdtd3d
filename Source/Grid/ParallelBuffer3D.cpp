#include "ParallelGrid.h"

#include <cmath>
#include <set>

#ifdef PARALLEL_GRID

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ

/**
 * Find proportion for computational nodes grid
 */
void
ParallelGridCore::FindProportionForNodeGrid (int &nodeGridSize1, /**< out: first axis nodes grid size */
                                             int &nodeGridSize2, /**< out: second axis nodes grid size */
                                             int &nodeGridSize3, /**< out: third axis nodes grid size */
                                             int &left, /**< out: number of left unused computational nodes */
                                             FPValue alpha, /**< preferred proportion between second and first axis */
                                             FPValue betta) /**< preferred proportion between third and first axis */
{
  /*
   * Bad case, too many nodes left unused. Let's change proportion.
   */

  int min_left = left;
  int min_size1 = nodeGridSize1;
  int min_size2 = nodeGridSize2;
  int min_size3 = nodeGridSize3;
  FPValue min_alpha = ((FPValue) min_size2) / ((FPValue) min_size1);
  FPValue min_betta = ((FPValue) min_size3) / ((FPValue) min_size1);

  for (int size1 = 2; size1 <= totalProcCount / 4; ++size1)
  {
    for (int size2 = 2; size2 <= totalProcCount / 4; ++size2)
    {
      int size3 = totalProcCount / (size1 * size2);
      int left_new = totalProcCount - (size1 * size2 * size3);

      if (left_new < min_left)
      {
        min_left = left_new;
        min_size1 = size1;
        min_size2 = size2;
        min_size3 = size3;
        min_alpha = ((FPValue) size2) / ((FPValue) size1);
        min_betta = ((FPValue) size3) / ((FPValue) size1);
      }
      else if (left_new == min_left)
      {
        FPValue new_alpha = ((FPValue) size2) / ((FPValue) size1);
        FPValue new_betta = ((FPValue) size3) / ((FPValue) size1);

        FPValue diff_alpha = fabs (new_alpha - alpha);
        FPValue diff_betta = fabs (new_betta - betta);

        FPValue diff_alpha_min = fabs (min_alpha - alpha);
        FPValue diff_betta_min = fabs (min_betta - betta);

        FPValue norm = sqrt(diff_alpha * diff_alpha + diff_betta * diff_betta);
        FPValue norm_min = sqrt(diff_alpha_min * diff_alpha_min + diff_betta_min * diff_betta_min);

        if (norm < norm_min)
        {
          min_left = left_new;
          min_size1 = size1;
          min_size2 = size2;
          min_size3 = size3;
          min_alpha = ((FPValue) size2) / ((FPValue) size1);
          min_betta = ((FPValue) size3) / ((FPValue) size1);
        }
      }
    }
  }

  nodeGridSize1 = min_size1;
  nodeGridSize2 = min_size2;
  nodeGridSize3 = min_size3;
  left = min_left;
} /* ParallelGridCore::FindProportionForNodeGrid */

/**
 * Helper struct for triples
 */
struct Triple
{
  grid_coord n; /**< first axis nodes grid size */
  grid_coord m; /**< second axis nodes grid size */
  grid_coord k; /**< third axis nodes grid size */

  /**
   * Constructor with nodes grid sizes
   */
  Triple (grid_coord newN = 0, /**< first axis nodes grid size */
          grid_coord newM = 0, /**< second axis nodes grid size */
          grid_coord newK = 0) /**< third axis nodes grid size */
    : n (newN)
  , m (newM)
  , k (newK)
  {
  } /* Triple */
}; /* Triple */

/**
 * Initialize parallel grid virtual topology as optimal for current number of processes and specified grid sizes
 */
void
ParallelGridCore::initOptimal (grid_coord size1, /**< grid size by first axis */
                               grid_coord size2, /**< grid size by second axis */
                               grid_coord size3, /**< grid size by third axis */
                               int &nodeGridSize1, /**< out: first axis nodes grid size */
                               int &nodeGridSize2, /**< out: second axis nodes grid size */
                               int &nodeGridSize3, /**< out: third axis nodes grid size */
                               int &left) /**< out: number of left unused computational nodes */
{
  /*
   * Find allowed triples of node grid sizes
   */

  std::vector<Triple> allowedTriples;

  grid_coord gcd1 = ParallelGridCore::greatestCommonDivider (size1, (grid_coord) totalProcCount);
  grid_coord gcd2 = ParallelGridCore::greatestCommonDivider (size2, (grid_coord) totalProcCount);

  std::set<grid_coord> allowedM;

  for (grid_coord n = 1; n <= gcd1; ++n)
  {
    if (gcd1 % n != 0)
    {
      continue;
    }

    for (grid_coord m = 1; m <= gcd2; ++m)
    {
      if (gcd2 % m != 0)
      {
        continue;
      }

      FPValue k_fp = ((FPValue) totalProcCount) / ((FPValue) n * m);
      grid_coord k = (grid_coord) k_fp;

      FPValue c1_fp = ((FPValue) size3) / k_fp;
      grid_coord c1 = (grid_coord) c1_fp;

      if (k == k_fp && c1 == c1_fp)
      {
        allowedTriples.push_back (Triple (n, m, k));

        allowedM.insert (m);
      }
    }
  }

  /*
   * allowedTriples are sorted in ascending order for n, m, k
   */

  ASSERT (allowedTriples.size () > 0);

  Triple min = allowedTriples[0];

  for (std::set<grid_coord>::iterator it = allowedM.begin ();
       it != allowedM.end ();
       ++it)
  {
    grid_coord m = *it;

    FPValue optimal_n = sqrt (((FPValue) size1 * totalProcCount) / ((FPValue) m * size3));

    Triple left = allowedTriples[0];
    Triple right = allowedTriples[0];

    for (std::vector<Triple>::iterator iter = allowedTriples.begin ();
         iter != allowedTriples.end ();
         ++iter)
    {
      grid_coord n_iter = iter->n;
      grid_coord m_iter = iter->m;

      if (m != m_iter)
      {
        continue;
      }

      if (n_iter < optimal_n)
      {
        left = *iter;
      }
      else
      {
        right = *iter;
        break;
      }
    }

  /*
   * This heavily depends on the parallel grid sharing scheme
   */
#define func(triple) \
  ((size1)*(size2) / (triple.n * triple.m) + \
   (size2)*(size3) / (triple.m * triple.k) + \
   (size1)*(size3) / (triple.n * triple.k) + \
   4*((size1) / triple.n + (size2) / triple.n + (size3) / triple.k))

    Triple min_cur;

    if (func (left) < func (right))
    {
      min_cur = left;
    }
    else
    {
      min_cur = right;
    }

    if (func (min_cur) < func (min))
    {
      min = min_cur;
    }

#undef func
  }

  nodeGridSize1 = min.n;
  nodeGridSize2 = min.m;
  nodeGridSize3 = min.k;

  left = 0;
} /* ParallelGridCore::initOptimal */

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#endif /* PARALLEL_GRID */
