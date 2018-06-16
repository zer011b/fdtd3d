#include "ParallelGrid.h"

#include <cmath>
#include <set>

#ifdef PARALLEL_GRID

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ

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
                               int &nodeGridSize3) /**< out: third axis nodes grid size */
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

      DOUBLE k_fp = ((DOUBLE) totalProcCount) / ((DOUBLE) n * m);
      grid_coord k = (grid_coord) k_fp;

      DOUBLE c1_fp = ((DOUBLE) size3) / k_fp;
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

    DOUBLE optimal_n = sqrt (((DOUBLE) size1 * totalProcCount) / ((DOUBLE) m * size3));

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
} /* ParallelGridCore::initOptimal */

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#endif /* PARALLEL_GRID */
