#include "ParallelGrid.h"

#include <cmath>

#ifdef PARALLEL_GRID

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ

void
ParallelGrid::FindProportionForNodeGrid (int& nodeGridSize1, int& nodeGridSize2, int& nodeGridSize3, int& left,
                                         FPValue alpha, FPValue betta)
{
  int min_left = left;
  int min_size1 = nodeGridSize1;
  int min_size2 = nodeGridSize2;
  int min_size3 = nodeGridSize3;
  FPValue min_alpha = ((FPValue) min_size2) / ((FPValue) min_size1);
  FPValue min_betta = ((FPValue) min_size3) / ((FPValue) min_size1);

  // Bad case, too many nodes left unused. Let's change proportion.
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
}

void
ParallelGrid::NodeGridInitInner (FPValue& overall1, FPValue& overall2, FPValue& overall3,
                                 int& nodeGridSize1, int& nodeGridSize2, int& nodeGridSize3, int& left)
{
  FPValue alpha = overall2 / overall1;
  FPValue betta = overall3 / overall1;
  FPValue cbrtVal = ((FPValue) (totalProcCount)) / (alpha * betta);
  cbrtVal = cbrt (cbrtVal);

  if (cbrtVal <= 1.0 || alpha*cbrtVal <= 1.0 || betta*cbrtVal <= 1.0)
  {
    cbrtVal = 2;
  }
  else
  {
    cbrtVal = round (cbrtVal);
  }

  nodeGridSize1 = (int) cbrtVal;
  nodeGridSize2 = alpha * nodeGridSize1;
  nodeGridSize3 = totalProcCount / (nodeGridSize1 * nodeGridSize2);

  left = totalProcCount - nodeGridSize1 * nodeGridSize2 * nodeGridSize3;

  if (left > 0)
  {
    // Bad case, too many nodes left unused. Let's change proportion.
    FindProportionForNodeGrid (nodeGridSize1, nodeGridSize2, nodeGridSize3, left, alpha, betta);
  }

  ASSERT (nodeGridSize1 > 1 && nodeGridSize2 > 1 && nodeGridSize3 > 1);
}

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#endif /* PARALLEL_GRID */
