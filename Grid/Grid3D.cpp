#include "Grid.h"

#include <cmath>

extern const char* BufferPositionNames[];

#ifdef PARALLEL_GRID
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

void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{
  NodeGridInit ();

  // Return if node not used.
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  if (processId >= nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ)
  {
    return;
  }
#endif

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  if (processId >= nodeGridSizeX * nodeGridSizeY)
  {
    return;
  }
#endif

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  if (processId >= nodeGridSizeY * nodeGridSizeZ)
  {
    return;
  }
#endif

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  if (processId >= nodeGridSizeX * nodeGridSizeZ)
  {
    return;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  bool hasL = false;
  bool hasR = false;

  if (processId % nodeGridSizeX != 0)
  {
    hasL = true;
  }

  if ((processId + 1) % nodeGridSizeX != 0)
  {
    hasR = true;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  bool hasU = false;
  bool hasD = false;

  if (processId >= nodeGridSizeX)
  {
    hasD = true;
  }

  if (processId < nodeGridSizeX * nodeGridSizeY - nodeGridSizeX)
  {
    hasU = true;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  bool hasF = false;
  bool hasB = false;

  if (processId >= nodeGridSizeX * nodeGridSizeY)
  {
    hasB = true;
  }

  if (processId < nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ - nodeGridSizeX * nodeGridSizeY)
  {
    hasF = true;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (hasL)
  {
    buffersSend[LEFT].resize (bufferSizeLeft.getX () * currentSize.getY () * currentSize.getZ () * numTimeStepsInBuild);
    buffersReceive[LEFT].resize (bufferSizeLeft.getX () * currentSize.getY () * currentSize.getZ () * numTimeStepsInBuild);
  }
  if (hasR)
  {
    buffersSend[RIGHT].resize (bufferSizeRight.getX () * currentSize.getY () * currentSize.getZ () * numTimeStepsInBuild);
    buffersReceive[RIGHT].resize (bufferSizeRight.getX () * currentSize.getY () * currentSize.getZ () * numTimeStepsInBuild);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (hasD)
  {
    buffersSend[DOWN].resize (bufferSizeLeft.getY () * currentSize.getX () * currentSize.getZ () * numTimeStepsInBuild);
    buffersReceive[DOWN].resize (bufferSizeLeft.getY () * currentSize.getX () * currentSize.getZ () * numTimeStepsInBuild);
  }
  if (hasU)
  {
    buffersSend[UP].resize (bufferSizeRight.getY () * currentSize.getX () * currentSize.getZ () * numTimeStepsInBuild);
    buffersReceive[UP].resize (bufferSizeRight.getY () * currentSize.getX () * currentSize.getZ () * numTimeStepsInBuild);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (hasB)
  {
    buffersSend[BACK].resize (bufferSizeLeft.getZ () * currentSize.getX () * currentSize.getY () * numTimeStepsInBuild);
    buffersReceive[BACK].resize (bufferSizeLeft.getZ () * currentSize.getX () * currentSize.getY () * numTimeStepsInBuild);
  }
  if (hasF)
  {
    buffersSend[FRONT].resize (bufferSizeRight.getZ () * currentSize.getX () * currentSize.getY () * numTimeStepsInBuild);
    buffersReceive[FRONT].resize (bufferSizeRight.getZ () * currentSize.getX () * currentSize.getY () * numTimeStepsInBuild);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (hasL && hasD)
  {
    buffersSend[LEFT_DOWN].resize (bufferSizeLeft.getX () * bufferSizeLeft.getY () * currentSize.getZ () * numTimeStepsInBuild);
    buffersReceive[LEFT_DOWN].resize (bufferSizeLeft.getX () * bufferSizeLeft.getY () * currentSize.getZ () * numTimeStepsInBuild);
  }
  if (hasL && hasU)
  {
    buffersSend[LEFT_UP].resize (bufferSizeLeft.getX () * bufferSizeRight.getY () * currentSize.getZ () * numTimeStepsInBuild);
    buffersReceive[LEFT_UP].resize (bufferSizeLeft.getX () * bufferSizeRight.getY () * currentSize.getZ () * numTimeStepsInBuild);
  }
  if (hasR && hasD)
  {
    buffersSend[RIGHT_DOWN].resize (bufferSizeRight.getX () * bufferSizeLeft.getY () * currentSize.getZ () * numTimeStepsInBuild);
    buffersReceive[RIGHT_DOWN].resize (bufferSizeRight.getX () * bufferSizeLeft.getY () * currentSize.getZ () * numTimeStepsInBuild);
  }
  if (hasR && hasU)
  {
    buffersSend[RIGHT_UP].resize (bufferSizeRight.getX () * bufferSizeRight.getY () * currentSize.getZ () * numTimeStepsInBuild);
    buffersReceive[RIGHT_UP].resize (bufferSizeRight.getX () * bufferSizeRight.getY () * currentSize.getZ () * numTimeStepsInBuild);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (hasD && hasB)
  {
    buffersSend[DOWN_BACK].resize (bufferSizeLeft.getY () * bufferSizeLeft.getZ () * currentSize.getX () * numTimeStepsInBuild);
    buffersReceive[DOWN_BACK].resize (bufferSizeLeft.getY () * bufferSizeLeft.getZ () * currentSize.getX () * numTimeStepsInBuild);
  }
  if (hasD && hasF)
  {
    buffersSend[DOWN_FRONT].resize (bufferSizeLeft.getY () * bufferSizeRight.getZ () * currentSize.getX () * numTimeStepsInBuild);
    buffersReceive[DOWN_FRONT].resize (bufferSizeLeft.getY () * bufferSizeRight.getZ () * currentSize.getX () * numTimeStepsInBuild);
  }
  if (hasU && hasB)
  {
    buffersSend[UP_BACK].resize (bufferSizeRight.getY () * bufferSizeLeft.getZ () * currentSize.getX () * numTimeStepsInBuild);
    buffersReceive[UP_BACK].resize (bufferSizeRight.getY () * bufferSizeLeft.getZ () * currentSize.getX () * numTimeStepsInBuild);
  }
  if (hasU && hasF)
  {
    buffersSend[UP_FRONT].resize (bufferSizeRight.getY () * bufferSizeRight.getZ () * currentSize.getX () * numTimeStepsInBuild);
    buffersReceive[UP_FRONT].resize (bufferSizeRight.getY () * bufferSizeRight.getZ () * currentSize.getX () * numTimeStepsInBuild);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (hasL && hasB)
  {
    buffersSend[LEFT_BACK].resize (bufferSizeLeft.getX () * bufferSizeLeft.getZ () * currentSize.getY () * numTimeStepsInBuild);
    buffersReceive[LEFT_BACK].resize (bufferSizeLeft.getX () * bufferSizeLeft.getZ () * currentSize.getY () * numTimeStepsInBuild);
  }
  if (hasL && hasF)
  {
    buffersSend[LEFT_FRONT].resize (bufferSizeLeft.getX () * bufferSizeRight.getZ () * currentSize.getY () * numTimeStepsInBuild);
    buffersReceive[LEFT_FRONT].resize (bufferSizeLeft.getX () * bufferSizeRight.getZ () * currentSize.getY () * numTimeStepsInBuild);
  }
  if (hasR && hasB)
  {
    buffersSend[RIGHT_BACK].resize (bufferSizeRight.getX () * bufferSizeLeft.getZ () * currentSize.getY () * numTimeStepsInBuild);
    buffersReceive[RIGHT_BACK].resize (bufferSizeRight.getX () * bufferSizeLeft.getZ () * currentSize.getY () * numTimeStepsInBuild);
  }
  if (hasR && hasF)
  {
    buffersSend[RIGHT_FRONT].resize (bufferSizeRight.getX () * bufferSizeRight.getZ () * currentSize.getY () * numTimeStepsInBuild);
    buffersReceive[RIGHT_FRONT].resize (bufferSizeRight.getX () * bufferSizeRight.getZ () * currentSize.getY () * numTimeStepsInBuild);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (hasL && hasD && hasB)
  {
    buffersSend[LEFT_DOWN_BACK].resize (bufferSizeLeft.getX () * bufferSizeLeft.getY () * bufferSizeLeft.getZ () * numTimeStepsInBuild);
    buffersReceive[LEFT_DOWN_BACK].resize (bufferSizeLeft.getX () * bufferSizeLeft.getY () * bufferSizeLeft.getZ () * numTimeStepsInBuild);
  }
  if (hasL && hasD && hasF)
  {
    buffersSend[LEFT_DOWN_FRONT].resize (bufferSizeLeft.getX () * bufferSizeLeft.getY () * bufferSizeRight.getZ () * numTimeStepsInBuild);
    buffersReceive[LEFT_DOWN_FRONT].resize (bufferSizeLeft.getX () * bufferSizeLeft.getY () * bufferSizeRight.getZ () * numTimeStepsInBuild);
  }
  if (hasL && hasU && hasB)
  {
    buffersSend[LEFT_UP_BACK].resize (bufferSizeLeft.getX () * bufferSizeRight.getY () * bufferSizeLeft.getZ () * numTimeStepsInBuild);
    buffersReceive[LEFT_UP_BACK].resize (bufferSizeLeft.getX () * bufferSizeRight.getY () * bufferSizeLeft.getZ () * numTimeStepsInBuild);
  }
  if (hasL && hasU && hasF)
  {
    buffersSend[LEFT_UP_FRONT].resize (bufferSizeLeft.getX () * bufferSizeRight.getY () * bufferSizeRight.getZ () * numTimeStepsInBuild);
    buffersReceive[LEFT_UP_FRONT].resize (bufferSizeLeft.getX () * bufferSizeRight.getY () * bufferSizeRight.getZ () * numTimeStepsInBuild);
  }

  if (hasR && hasD && hasB)
  {
    buffersSend[RIGHT_DOWN_BACK].resize (bufferSizeRight.getX () * bufferSizeLeft.getY () * bufferSizeLeft.getZ () * numTimeStepsInBuild);
    buffersReceive[RIGHT_DOWN_BACK].resize (bufferSizeRight.getX () * bufferSizeLeft.getY () * bufferSizeLeft.getZ () * numTimeStepsInBuild);
  }
  if (hasR && hasD && hasF)
  {
    buffersSend[RIGHT_DOWN_FRONT].resize (bufferSizeRight.getX () * bufferSizeLeft.getY () * bufferSizeRight.getZ () * numTimeStepsInBuild);
    buffersReceive[RIGHT_DOWN_FRONT].resize (bufferSizeRight.getX () * bufferSizeLeft.getY () * bufferSizeRight.getZ () * numTimeStepsInBuild);
  }
  if (hasR && hasU && hasB)
  {
    buffersSend[RIGHT_UP_BACK].resize (bufferSizeRight.getX () * bufferSizeRight.getY () * bufferSizeLeft.getZ () * numTimeStepsInBuild);
    buffersReceive[RIGHT_UP_BACK].resize (bufferSizeRight.getX () * bufferSizeRight.getY () * bufferSizeLeft.getZ () * numTimeStepsInBuild);
  }
  if (hasR && hasU && hasF)
  {
    buffersSend[RIGHT_UP_FRONT].resize (bufferSizeRight.getX () * bufferSizeRight.getY () * bufferSizeRight.getZ () * numTimeStepsInBuild);
    buffersReceive[RIGHT_UP_FRONT].resize (bufferSizeRight.getX () * bufferSizeRight.getY () * bufferSizeRight.getZ () * numTimeStepsInBuild);
  }
#endif
}





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

#endif /* PARALLEL_GRID */
#endif /* GRID_3D */
