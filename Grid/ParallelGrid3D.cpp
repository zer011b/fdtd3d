#include "ParallelGrid.h"

#include <cmath>

#ifdef PRINT_MESSAGE
extern const char* BufferPositionNames[];
#endif

#ifdef PARALLEL_GRID
#ifdef GRID_3D

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
void
ParallelGrid::NodeGridInit ()
{
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  nodeGridSizeX = totalProcCount;
  nodeGridSizeY = 1;
  nodeGridSizeZ = 1;
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  nodeGridSizeX = 1;
  nodeGridSizeY = totalProcCount;
  nodeGridSizeZ = 1;
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  nodeGridSizeX = 1;
  nodeGridSizeY = 1;
  nodeGridSizeZ = totalProcCount;
#endif

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%dx%d.\n", processId,
    nodeGridSizeX, nodeGridSizeY, nodeGridSizeZ);
#endif
}

GridCoordinate3D
ParallelGrid::GridInit ()
{
  grid_coord c1;
  grid_coord c2;
  grid_coord c3;

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  CalculateGridSizeForNode (c1, nodeGridSizeX, hasR, totalSize.getX ());
  c2 = totalSize.getY ();
  c3 = totalSize.getZ ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  c1 = totalSize.getX ();
  CalculateGridSizeForNode (c2, nodeGridSizeY, hasU, totalSize.getY ());
  c3 = totalSize.getZ ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  c1 = totalSize.getX ();
  c2 = totalSize.getY ();
  CalculateGridSizeForNode (c3, nodeGridSizeZ, hasF, totalSize.getZ ());
#endif

  return GridCoordinate3D (c1, c2, c3);
}
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
void
ParallelGrid::NodeGridInit ()
{
  if (totalProcCount < 4)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 2D parallel buffers. Use 1D ones.");
  }

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  FieldValue overall1 = (FieldValue) totalSize.getX ();
  FieldValue overall2 = (FieldValue) totalSize.getY ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  FieldValue overall1 = (FieldValue) totalSize.getY ();
  FieldValue overall2 = (FieldValue) totalSize.getZ ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  FieldValue overall1 = (FieldValue) totalSize.getX ();
  FieldValue overall2 = (FieldValue) totalSize.getZ ();
#endif

  int left;
  int nodeGridSizeTmp1;
  int nodeGridSizeTmp2;
  NodeGridInitInner (overall1, overall2, nodeGridSizeTmp1, nodeGridSizeTmp2, left);

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  nodeGridSizeX = nodeGridSizeTmp1;
  nodeGridSizeY = nodeGridSizeTmp2;
  nodeGridSizeZ = 1;

  nodeGridSizeXY = nodeGridSizeX * nodeGridSizeY;
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  nodeGridSizeX = 1;
  nodeGridSizeY = nodeGridSizeTmp1;
  nodeGridSizeZ = nodeGridSizeTmp2;

  nodeGridSizeYZ = nodeGridSizeY * nodeGridSizeZ;
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  nodeGridSizeX = nodeGridSizeTmp1;
  nodeGridSizeY = 1;
  nodeGridSizeZ = nodeGridSizeTmp2;

  nodeGridSizeXZ = nodeGridSizeX * nodeGridSizeZ;
#endif

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%dx%d. %d node(s) unused.\n", processId,
    nodeGridSizeX, nodeGridSizeY, nodeGridSizeZ, left);
#endif
}

GridCoordinate3D
ParallelGrid::GridInit ()
{
  grid_coord c1;
  grid_coord c2;
  grid_coord c3;

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  CalculateGridSizeForNode (c1, nodeGridSizeX, hasR, totalSize.getX (),
                            c2, nodeGridSizeY, hasU, totalSize.getY ());
  c3 = totalSize.getZ ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  c1 = totalSize.getX ();
  CalculateGridSizeForNode (c2, nodeGridSizeY, hasU, totalSize.getY (),
                            c3, nodeGridSizeZ, hasF, totalSize.getZ ());
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  c2 = totalSize.getY ();
  CalculateGridSizeForNode (c1, nodeGridSizeX, hasR, totalSize.getX (),
                            c3, nodeGridSizeZ, hasF, totalSize.getZ ());
#endif

  return GridCoordinate3D (c1, c2, c3);
}
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
ParallelGrid::NodeGridInit ()
{
  if (totalProcCount < 8)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 3D parallel buffers. Use 2D or 1D ones.");
  }

  FieldValue overall1 = (FieldValue) totalSize.getX ();
  FieldValue overall2 = (FieldValue) totalSize.getY ();
  FieldValue overall3 = (FieldValue) totalSize.getZ ();

  int left;
  int nodeGridSizeTmp1;
  int nodeGridSizeTmp2;
  int nodeGridSizeTmp3;
  NodeGridInitInner (overall1, overall2, overall3, nodeGridSizeTmp1, nodeGridSizeTmp2, nodeGridSizeTmp3, left);

  nodeGridSizeX = nodeGridSizeTmp1;
  nodeGridSizeY = nodeGridSizeTmp2;
  nodeGridSizeZ = nodeGridSizeTmp3;

  nodeGridSizeXYZ = nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ;
  nodeGridSizeXY = nodeGridSizeX * nodeGridSizeY;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%dx%d. %d node(s) unused.\n", processId,
    nodeGridSizeX, nodeGridSizeY, nodeGridSizeZ, left);
  //printf ("Grid size for #%d process: %dx%dx%d.\n", processId, c1, c2, c3);
#endif
}

GridCoordinate3D
ParallelGrid::GridInit ()
{
  grid_coord c1;
  grid_coord c2;
  grid_coord c3;

  CalculateGridSizeForNode (c1, nodeGridSizeX, hasR, totalSize.getX (),
                            c2, nodeGridSizeY, hasU, totalSize.getY (),
                            c3, nodeGridSizeZ, hasF, totalSize.getZ ());

  return GridCoordinate3D (c1, c2, c3);
}
#endif

#endif /* PARALLEL_GRID */
#endif /* GRID_3D */
