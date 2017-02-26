#include "ParallelGrid.h"

#ifdef GRID_1D

#ifdef PARALLEL_GRID

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X

/**
 * Initialize 1D grid of computational nodes
 */
void
ParallelGridCore::NodeGridInit (ParallelGridCoordinateFP desiredProportion) /**< desired relation values */
{
  nodeGridSizeX = totalProcCount;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %d.\n",
          processId,
          nodeGridSizeX);
#endif /* PRINT_MESSAGE */
} /* ParallelGridCore::NodeGridInit */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#endif /* PARALLEL_GRID */

#endif /* GRID_1D */
