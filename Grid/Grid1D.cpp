#include "Grid.h"

extern const char* BufferPositionNames[];

#ifdef GRID_1D
#ifdef PARALLEL_GRID

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
void
Grid::NodeGridInit ()
{
  nodeGridSizeX = totalProcCount;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %d.\n", processId,
    nodeGridSizeX);
#endif
}

GridCoordinate
Grid::GridInit ()
{
  grid_coord c1;

  CalculateGridSizeForNode (c1, nodeGridSizeX, hasR, totalSize.getX ());

  return GridCoordinate (c1);
}
#endif

#endif /* PARALLEL_GRID */
#endif /* GRID_1D */
