#include "ParallelGrid.h"

#include <cmath>

#if defined (PARALLEL_GRID)

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ

void
ParallelGrid::FindProportionForNodeGrid (int& nodeGridSize1, int& nodeGridSize2, int& nodeGridSize3, int& left,
                                 FieldValue alpha, FieldValue betta)
{
  int min_left = left;
  int min_size1 = nodeGridSize1;
  int min_size2 = nodeGridSize2;
  int min_size3 = nodeGridSize3;
  FieldValue min_alpha = ((FieldValue) min_size2) / ((FieldValue) min_size1);
  FieldValue min_betta = ((FieldValue) min_size3) / ((FieldValue) min_size1);

  // Bad case, too many nodes left unused. Let's change proportion.
  for (int size1 = 2; size1 < totalProcCount / 4; ++size1)
  {
    for (int size2 = 2; size2 < totalProcCount / 4; ++size2)
    {
      int size3 = totalProcCount / (size1 * size2);
      int left_new = totalProcCount - (size1 * size2 * size3);

      if (left_new < min_left)
      {
        min_left = left_new;
        min_size1 = size1;
        min_size2 = size2;
        min_size3 = size3;
        min_alpha = ((FieldValue) size2) / ((FieldValue) size1);
        min_betta = ((FieldValue) size3) / ((FieldValue) size1);
      }
      else if (left_new == min_left)
      {
        FieldValue new_alpha = ((FieldValue) size2) / ((FieldValue) size1);
        FieldValue new_betta = ((FieldValue) size3) / ((FieldValue) size1);

        FieldValue diff_alpha = fabs (new_alpha - alpha);
        FieldValue diff_betta = fabs (new_betta - betta);

        FieldValue diff_alpha_min = fabs (min_alpha - alpha);
        FieldValue diff_betta_min = fabs (min_betta - betta);

        FieldValue norm = sqrt(diff_alpha * diff_alpha + diff_betta * diff_betta);
        FieldValue norm_min = sqrt(diff_alpha_min * diff_alpha_min + diff_betta_min * diff_betta_min);

        if (norm < norm_min)
        {
          min_left = left_new;
          min_size1 = size1;
          min_size2 = size2;
          min_size3 = size3;
          min_alpha = ((FieldValue) size2) / ((FieldValue) size1);
          min_betta = ((FieldValue) size3) / ((FieldValue) size1);
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
ParallelGrid::NodeGridInitInner (FieldValue& overall1, FieldValue& overall2, FieldValue& overall3,
                         int& nodeGridSize1, int& nodeGridSize2, int& nodeGridSize3, int& left)
{
  FieldValue alpha = overall2 / overall1;
  FieldValue betta = overall3 / overall1;
  FieldValue cbrtVal = ((FieldValue) (totalProcCount)) / (alpha * betta);
  cbrtVal = cbrt (cbrtVal);

  if (cbrtVal <= 1.0 || alpha*cbrtVal <= 1.0 || betta*cbrtVal <= 1.0)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 3D parallel buffers. Use 2D or 1D ones.");
  }

  cbrtVal = round (cbrtVal);

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

#endif
#endif
