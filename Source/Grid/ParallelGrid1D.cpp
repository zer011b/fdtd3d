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

/**
 * Initialize size of grid per node
 *
 * TODO: move to layout
 *
 * @return size of grid for current node
 */
GridCoordinate1D
ParallelGrid::GridInit (GridCoordinate1D &coreSize) /**< out: size of grid for node, except the node at the right
                                                     *        border which is assigned all the data which is left after
                                                     *        equal spread for all nodes. Thus, for all nodes except
                                                     *        node at the right border core and returned sizes are
                                                     *        the same */
{
  grid_coord c1;
  grid_coord core1;

  CalculateGridSizeForNode (c1,
                            core1,
                            parallelGridCore->getNodeGridSizeX (),
                            parallelGridCore->getHasR (),
                            totalSize.getX ());

  coreSize = GridCoordinate1D (core1);

  return GridCoordinate1D (c1);
} /* ParallelGrid::GridInit */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#endif /* PARALLEL_GRID */

#endif /* GRID_1D */
