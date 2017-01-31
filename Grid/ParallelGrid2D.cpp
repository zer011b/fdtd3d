#include "ParallelGrid.h"

#ifdef GRID_2D
#ifdef PARALLEL_GRID

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
void
ParallelGrid::NodeGridInit ()
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

GridCoordinate2D
ParallelGrid::GridInit (GridCoordinate2D &core)
{
  grid_coord c1;
  grid_coord c2;

  grid_coord core1;
  grid_coord core2;

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  CalculateGridSizeForNode (c1, core1, nodeGridSizeX, hasR, totalSize.getX ());
  core2 = totalSize.getY ();
  c2 = c2;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  core1 = totalSize.getX ();
  c1 = core1;
  CalculateGridSizeForNode (c2, core2, nodeGridSizeY, hasU, totalSize.getY ());
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

  core = GridCoordinate2D (core1, core2);

  return GridCoordinate2D (c1, c2);
}
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
void
ParallelGrid::NodeGridInit ()
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

GridCoordinate2D
ParallelGrid::GridInit (GridCoordinate2D &core)
{
  grid_coord c1;
  grid_coord c2;

  grid_coord core1;
  grid_coord core2;

  CalculateGridSizeForNode (c1, core1, nodeGridSizeX, hasR, totalSize.getX (),
                            c2, core2, nodeGridSizeY, hasU, totalSize.getY ());

  core = GridCoordinate2D (core1, core2);

  return GridCoordinate2D (c1, c2);
}
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#endif /* PARALLEL_GRID */
#endif /* GRID_2D */
