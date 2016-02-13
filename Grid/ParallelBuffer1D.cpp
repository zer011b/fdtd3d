#include "Grid.h"

extern const char* BufferPositionNames[];

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z)

void
Grid::CalculateGridSizeForNode (grid_coord& c1, int nodeGridSize1, grid_coord size1)
{
  if ((processId + 1) % nodeGridSize1 != 0)
    c1 = size1 / nodeGridSize1;
  else
    c1 = size1 - (nodeGridSize1 - 1) * (size1 / nodeGridSize1);
}

#endif
