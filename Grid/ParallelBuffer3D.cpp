#include "Grid.h"

#include <cmath>

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ

void
Grid::FindProportionForNodeGrid (int& nodeGridSize1, int& nodeGridSize2, int& nodeGridSize3, int& left)
{
  int original_left = left;
  int size1 = nodeGridSize1;
  int size2 = nodeGridSize2;

  // Bad case, too many nodes left unused. Let's change proportion.
  bool find = true;
  bool direction = nodeGridSize1 > nodeGridSize2 ? true : false;
  while (find)
  {
    find = false;
    if (direction && nodeGridSize1 > 2)
    {
      find = true;
      --nodeGridSize1;
      nodeGridSize2 = totalProcCount / nodeGridSize1;
    }
    else if (!direction && nodeGridSize2 > 2)
    {
      find = true;
      --nodeGridSize2;
      nodeGridSize1 = totalProcCount / nodeGridSize2;
    }

    left = totalProcCount - nodeGridSize1 * nodeGridSize2;

    if (find && left == 0)
    {
      find = false;
    }
  }

  if (left >= original_left)
  {
    nodeGridSize1 = size1;
    nodeGridSize2 = size2;
  }
}

#endif
