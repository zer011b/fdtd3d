#include "Grid.h"

#include <cmath>

extern const char* BufferPositionNames[];

#ifdef GRID_3D

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
void
Grid::NodeGridInit ()
{
  grid_coord c1;
  grid_coord c2;
  grid_coord c3;

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  nodeGridSizeX = totalProcCount;
  nodeGridSizeY = 1;
  nodeGridSizeZ = 1;

  CalculateGridSizeForNode (c1, nodeGridSizeX, totalSize.getX ());
  c2 = totalSize.getY ();
  c3 = totalSize.getZ ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  nodeGridSizeX = 1;
  nodeGridSizeY = totalProcCount;
  nodeGridSizeZ = 1;

  c1 = totalSize.getX ();
  CalculateGridSizeForNode (c2, nodeGridSizeY, totalSize.getY ());
  c3 = totalSize.getZ ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  nodeGridSizeX = 1;
  nodeGridSizeY = 1;
  nodeGridSizeZ = totalProcCount;

  c1 = totalSize.getX ();
  c2 = totalSize.getY ();
  CalculateGridSizeForNode (c3, nodeGridSizeZ, totalSize.getZ ());
#endif

  currentSize = GridCoordinate (c1, c2, c3);
  size = currentSize + bufferSizeLeft + bufferSizeRight;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%dx%d.\n", processId,
    nodeGridSizeX, nodeGridSizeY, nodeGridSizeZ);
#endif
}
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
void
Grid::NodeGridInit ()
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
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  FieldValue overall1 = (FieldValue) totalSize.getX ();
  FieldValue overall2 = (FieldValue) totalSize.getZ ();
#endif

  int left;
  int nodeGridSizeTmp1;
  int nodeGridSizeTmp2;
  NodeGridInitInner (overall1, overall2, nodeGridSizeTmp1, nodeGridSizeTmp2, left);

  grid_coord c1;
  grid_coord c2;
  grid_coord c3;

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  nodeGridSizeX = nodeGridSizeTmp1;
  nodeGridSizeY = nodeGridSizeTmp2;
  nodeGridSizeZ = 1;

  CalculateGridSizeForNode (c1, nodeGridSizeX, totalSize.getX (), c2, nodeGridSizeY, totalSize.getY ());
  c3 = totalSize.getZ ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  nodeGridSizeX = 1;
  nodeGridSizeY = nodeGridSizeTmp1;
  nodeGridSizeZ = nodeGridSizeTmp2;

  c1 = totalSize.getX ();
  CalculateGridSizeForNode (c2, nodeGridSizeY, totalSize.getY (), c3, nodeGridSizeZ, totalSize.getZ ());
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  nodeGridSizeX = nodeGridSizeTmp1;
  nodeGridSizeY = 1;
  nodeGridSizeZ = nodeGridSizeTmp2;

  c2 = totalSize.getY ();
  CalculateGridSizeForNode (c1, nodeGridSizeX, totalSize.getX (), c3, nodeGridSizeZ, totalSize.getZ ());
#endif

  currentSize = GridCoordinate (c1, c2, c3);
  size = currentSize + bufferSizeLeft + bufferSizeRight;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%dx%d. %d node(s) unused.\n", processId,
    nodeGridSizeX, nodeGridSizeY, nodeGridSizeZ, left);
#endif
}
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
Grid::NodeGridInit ()
{
  if (totalProcCount < 8)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 3D parallel buffers. Use 2D or 1D ones.");
  }

  grid_coord overall1 = totalSize.getX ();
  grid_coord overall2 = totalSize.getY ();
  grid_coord overall3 = totalSize.getZ ();

  int left;
  int nodeGridSizeTmp1;
  int nodeGridSizeTmp2;
  int nodeGridSizeTmp3;
  NodeGridInitInner (overall1, overall2, overall3, nodeGridSizeTmp1, nodeGridSizeTmp2, nodeGridSizeTmp3, left);

  grid_coord c1;
  grid_coord c2;
  grid_coord c3;

  nodeGridSizeX = nodeGridSizeTmp1;
  nodeGridSizeY = nodeGridSizeTmp2;
  nodeGridSizeZ = nodeGridSizeTmp3;

  CalculateGridSizeForNode (c1, nodeGridSizeX, totalSize.getX (),
                            c2, nodeGridSizeY, totalSize.getY (),
                            c3, nodeGridSizeZ, totalSize.getZ ());

  currentSize = GridCoordinate (c1, c2, c3);
  size = currentSize + bufferSizeLeft + bufferSizeRight;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%dx%d. %d node(s) unused.\n", processId,
    nodeGridSizeX, nodeGridSizeY, nodeGridSizeZ, left);
#endif
}
#endif




#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
Grid::SendReceiveBuffer (BufferPosition bufferDirection)
{

}
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X ||
          PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_3D_XYZ */


#endif /* GRID_3D */
