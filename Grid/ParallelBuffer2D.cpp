#include "Grid.h"

#include <cmath>

#if defined (PARALLEL_GRID)

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)

void
Grid::FindProportionForNodeGrid (int& nodeGridSize1, int& nodeGridSize2, int& left, FieldValue alpha)
{
  int min_left = left;
  int min_size1 = nodeGridSize1;
  int min_size2 = nodeGridSize2;
  FieldValue min_alpha = ((FieldValue) min_size2) / ((FieldValue) min_size1);

  // Bad case, too many nodes left unused. Let's change proportion.
  for (int size1 = 2; size1 < totalProcCount / 2; ++size1)
  {
    int size2 = totalProcCount / size1;
    int left_new = totalProcCount - (size1 * size2);

    if (left_new < min_left)
    {
      min_left = left_new;
      min_size1 = size1;
      min_size2 = size2;
      min_alpha = ((FieldValue) size2) / ((FieldValue) size1);
    }
    else if (left_new == min_left)
    {
      FieldValue new_alpha = ((FieldValue) size2) / ((FieldValue) size1);

      FieldValue diff_alpha = fabs (new_alpha - alpha);
      FieldValue diff_alpha_min = fabs (min_alpha - alpha);

      if (diff_alpha < diff_alpha_min)
      {
        min_left = left_new;
        min_size1 = size1;
        min_size2 = size2;
        min_alpha = ((FieldValue) size2) / ((FieldValue) size1);
      }
    }
  }

  nodeGridSize1 = min_size1;
  nodeGridSize2 = min_size2;
  left = min_left;
}

void
Grid::NodeGridInitInner (FieldValue& overall1, FieldValue& overall2,
                         int& nodeGridSize1, int& nodeGridSize2, int& left)
{
  FieldValue alpha = overall2 / overall1;
  FieldValue sqrtVal = ((FieldValue) (totalProcCount)) / alpha;
  sqrtVal = sqrt (sqrtVal);

  if (sqrtVal <= 1.0 || alpha*sqrtVal <= 1.0)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 2D parallel buffers. Use 1D ones.");
  }

  sqrtVal = round (sqrtVal);

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

#endif
#endif
