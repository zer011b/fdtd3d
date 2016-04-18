#include "ParallelGrid.h"

#ifdef PRINT_MESSAGE
extern const char* BufferPositionNames[];
#endif

#ifdef GRID_1D
#ifdef PARALLEL_GRID

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
void
ParallelGrid::NodeGridInit ()
{
  nodeGridSizeX = totalProcCount;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %d.\n", processId,
    nodeGridSizeX);
#endif
}

GridCoordinate1D
ParallelGrid::GridInit ()
{
  grid_coord c1;

  CalculateGridSizeForNode (c1, nodeGridSizeX, hasR, totalSize.getX ());

  return GridCoordinate1D (c1);
}
#endif

#endif /* PARALLEL_GRID */
#endif /* GRID_1D */
