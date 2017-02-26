#include "ParallelGrid.h"

#ifdef GRID_2D

#ifdef PARALLEL_GRID

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y)

/**
 * Initialize 2D grid of computational nodes
 */
void
ParallelGridCore::NodeGridInit (ParallelGridCoordinateFP desiredProportion) /**< desired relation values */
{
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  nodeGridSizeX = totalProcCount;
  nodeGridSizeY = 1;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  nodeGridSizeX = 1;
  nodeGridSizeY = totalProcCount;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%d.\n",
          processId,
          nodeGridSizeX,
          nodeGridSizeY);
#endif /* PRINT_MESSAGE */
} /* ParallelGridCore::NodeGridInit */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY

/**
 * Initialize 2D grid of computational nodes
 */
void
ParallelGridCore::NodeGridInit (ParallelGridCoordinateFP desiredProportion) /**< desired relation values */
{
  if (totalProcCount < 4)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 2D parallel buffers. Use 1D ones.");
  }

  int left;
  NodeGridInitInner (desiredProportion.getX (), nodeGridSizeX, nodeGridSizeY, left);

  nodeGridSizeXY = nodeGridSizeX * nodeGridSizeY;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%d. %d node(s) unused.\n",
          processId,
          nodeGridSizeX,
          nodeGridSizeY,
          left);
#endif /* PRINT_MESSAGE */
} /* ParallelGridCore::NodeGridInit */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#endif /* PARALLEL_GRID */

#endif /* GRID_2D */
