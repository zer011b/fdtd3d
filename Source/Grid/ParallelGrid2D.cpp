#include "ParallelGrid.h"

#ifdef GRID_2D

#ifdef PARALLEL_GRID

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
void
ParallelGridCore::NodeGridInit ()
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
  printf ("Nodes' grid process #%d: %dx%d.\n", processId,
    nodeGridSizeX, nodeGridSizeY);
#endif /* PRINT_MESSAGE */
}

/**
 * Initialize size of grid per node
 *
 * TODO: move to layout
 *
 * @return size of grid for current node
 */
GridCoordinate2D
ParallelGrid::GridInit (GridCoordinate2D &coreSize) /**< out: size of grid for node, except the node at the right
                                                     *        border which is assigned all the data which is left after
                                                     *        equal spread for all nodes. Thus, for all nodes except
                                                     *        node at the right border core and returned sizes are
                                                     *        the same */
{
  grid_coord c1;
  grid_coord c2;

  grid_coord core1;
  grid_coord core2;

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  CalculateGridSizeForNode (c1, core1, parallelGridCore->getNodeGridSizeX (), parallelGridCore->getHasR (), totalSize.getX ());
  core2 = totalSize.getY ();
  c2 = core2;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  core1 = totalSize.getX ();
  c1 = core1;
  CalculateGridSizeForNode (c2, core2, parallelGridCore->getNodeGridSizeY (), parallelGridCore->getHasU (), totalSize.getY ());
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

  coreSize = GridCoordinate2D (core1, core2);

  return GridCoordinate2D (c1, c2);
} /* ParallelGrid::GridInit */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
void
ParallelGridCore::NodeGridInit ()
{
  if (totalProcCount < 4)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 2D parallel buffers. Use 1D ones.");
  }

  FPValue overall1 = (FPValue) totalSize.getX ();
  FPValue overall2 = (FPValue) totalSize.getY ();

  int left;
  NodeGridInitInner (overall1, overall2, nodeGridSizeX, nodeGridSizeY, left);

  nodeGridSizeXY = nodeGridSizeX * nodeGridSizeY;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%d. %d node(s) unused.\n", processId,
    nodeGridSizeX, nodeGridSizeY, left);
#endif /* PRINT_MESSAGE */
}

/**
 * Initialize size of grid per node
 *
 * @return size of grid for current node
 */
GridCoordinate2D
ParallelGrid::GridInit (GridCoordinate2D &coreSize) /**< out: size of grid for node, except the node at the right
                                                     *        border which is assigned all the data which is left after
                                                     *        equal spread for all nodes. Thus, for all nodes except
                                                     *        node at the right border core and returned sizes are
                                                     *        the same */
{
  grid_coord c1;
  grid_coord c2;

  grid_coord core1;
  grid_coord core2;

  CalculateGridSizeForNode (c1, core1, parallelGridCore->getNodeGridSizeX (), hasR, totalSize.getX (),
                            c2, core2, parallelGridCore->getNodeGridSizeY (), hasU, totalSize.getY ());

  coreSize = GridCoordinate2D (core1, core2);

  return GridCoordinate2D (c1, c2);
} /* ParallelGrid::GridInit */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#endif /* PARALLEL_GRID */

#endif /* GRID_2D */
