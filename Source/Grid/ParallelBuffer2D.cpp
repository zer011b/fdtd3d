#include "ParallelGrid.h"

#include <cmath>

#ifdef PARALLEL_GRID

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
    || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)

void
ParallelGridCore::FindProportionForNodeGrid (int& nodeGridSize1, int& nodeGridSize2, int& left, FPValue alpha)
{
  int min_left = left;
  int min_size1 = nodeGridSize1;
  int min_size2 = nodeGridSize2;
  FPValue min_alpha = ((FPValue) min_size2) / ((FPValue) min_size1);

  // Bad case, too many nodes left unused. Let's change proportion.
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
}

void
ParallelGridCore::NodeGridInitInner (FPValue& overall1, FPValue& overall2,
                                       int& nodeGridSize1, int& nodeGridSize2, int& left)
{
  FPValue alpha = overall2 / overall1;
  FPValue sqrtVal = ((FPValue) (totalProcCount)) / alpha;
  sqrtVal = sqrt (sqrtVal);

  if (sqrtVal <= 1.0 || alpha * sqrtVal <= 1.0)
  {
    // Unproportional calculational/node grids

    sqrtVal = 2;
  }
  else
  {
    sqrtVal = round (sqrtVal);
  }

  nodeGridSize1 = (int) sqrtVal;
  nodeGridSize2 = totalProcCount / nodeGridSize1;

  left = totalProcCount - nodeGridSize1 * nodeGridSize2;

  if (left > 0)
  {
    // Bad case, too many nodes left unused. Let's change proportion.
    FindProportionForNodeGrid (nodeGridSize1, nodeGridSize2, left, alpha);
  }

  ASSERT (nodeGridSize1 > 1 && nodeGridSize2 > 1);
}

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ) ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ */

#endif /* PARALLEL_GRID */
