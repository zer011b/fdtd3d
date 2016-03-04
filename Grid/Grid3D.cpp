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

  directions[LEFT] = processId - 1;
  directions[RIGHT] = processId + 1;

  CalculateGridSizeForNode (c1, nodeGridSizeX, totalSize.getX ());
  c2 = totalSize.getY ();
  c3 = totalSize.getZ ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  nodeGridSizeX = 1;
  nodeGridSizeY = totalProcCount;
  nodeGridSizeZ = 1;

  directions[DOWN] = processId - 1;
  directions[UP] = processId + 1;

  c1 = totalSize.getX ();
  CalculateGridSizeForNode (c2, nodeGridSizeY, totalSize.getY ());
  c3 = totalSize.getZ ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  nodeGridSizeX = 1;
  nodeGridSizeY = 1;
  nodeGridSizeZ = totalProcCount;

  directions[BACK] = processId - 1;
  directions[FRONT] = processId + 1;

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
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
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

  nodeGridSizeXY = nodeGridSizeX * nodeGridSizeY;

  directions[LEFT] = processId - 1;
  directions[RIGHT] = processId + 1;
  directions[DOWN] = processId - nodeGridSizeX;
  directions[UP] = processId + nodeGridSizeX;
  directions[LEFT_DOWN] = processId - nodeGridSizeX - 1;
  directions[LEFT_UP] = processId + nodeGridSizeX - 1;
  directions[RIGHT_DOWN] = processId - nodeGridSizeX + 1;
  directions[RIGHT_UP] = processId + nodeGridSizeX + 1;

  CalculateGridSizeForNode (c1, nodeGridSizeX, totalSize.getX (), c2, nodeGridSizeY, totalSize.getY ());
  c3 = totalSize.getZ ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  nodeGridSizeX = 1;
  nodeGridSizeY = nodeGridSizeTmp1;
  nodeGridSizeZ = nodeGridSizeTmp2;

  nodeGridSizeYZ = nodeGridSizeY * nodeGridSizeZ;

  directions[DOWN] = processId - 1;
  directions[UP] = processId + 1;
  directions[BACK] = processId - nodeGridSizeY;
  directions[FRONT] = processId + nodeGridSizeY;
  directions[DOWN_BACK] = processId - nodeGridSizeY - 1;
  directions[DOWN_FRONT] = processId + nodeGridSizeY - 1;
  directions[UP_BACK] = processId - nodeGridSizeY + 1;
  directions[UP_FRONT] = processId + nodeGridSizeY + 1;

  c1 = totalSize.getX ();
  CalculateGridSizeForNode (c2, nodeGridSizeY, totalSize.getY (), c3, nodeGridSizeZ, totalSize.getZ ());
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  nodeGridSizeX = nodeGridSizeTmp1;
  nodeGridSizeY = 1;
  nodeGridSizeZ = nodeGridSizeTmp2;

  nodeGridSizeXZ = nodeGridSizeX * nodeGridSizeZ;

  directions[LEFT] = processId - 1;
  directions[RIGHT] = processId + 1;
  directions[BACK] = processId - nodeGridSizeX;
  directions[FRONT] = processId + nodeGridSizeX;
  directions[LEFT_BACK] = processId - nodeGridSizeX - 1;
  directions[LEFT_FRONT] = processId + nodeGridSizeX - 1;
  directions[RIGHT_BACK] = processId - nodeGridSizeX + 1;
  directions[RIGHT_FRONT] = processId + nodeGridSizeX + 1;

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

  FieldValue overall1 = (FieldValue) totalSize.getX ();
  FieldValue overall2 = (FieldValue) totalSize.getY ();
  FieldValue overall3 = (FieldValue) totalSize.getZ ();

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

  nodeGridSizeXYZ = nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ;
  nodeGridSizeXY = nodeGridSizeX * nodeGridSizeY;

  directions[LEFT] = processId - 1;
  directions[RIGHT] = processId + 1;
  directions[DOWN] = processId - nodeGridSizeX;
  directions[UP] = processId + nodeGridSizeX;
  directions[BACK] = processId - nodeGridSizeXY;
  directions[FRONT] = processId + nodeGridSizeXY;
  directions[LEFT_DOWN] = processId - nodeGridSizeX - 1;
  directions[LEFT_UP] = processId + nodeGridSizeX - 1;
  directions[RIGHT_DOWN] = processId - nodeGridSizeX + 1;
  directions[RIGHT_UP] = processId + nodeGridSizeX + 1;
  directions[LEFT_BACK] = processId - nodeGridSizeXY - 1;
  directions[LEFT_FRONT] = processId + nodeGridSizeXY - 1;
  directions[RIGHT_BACK] = processId - nodeGridSizeXY + 1;
  directions[RIGHT_FRONT] = processId + nodeGridSizeXY + 1;
  directions[DOWN_BACK] = processId - nodeGridSizeXY - nodeGridSizeX;
  directions[DOWN_FRONT] = processId + nodeGridSizeXY - nodeGridSizeX;
  directions[UP_BACK] = processId - nodeGridSizeXY + nodeGridSizeX;
  directions[UP_FRONT] = processId + nodeGridSizeXY + nodeGridSizeX;
  directions[LEFT_DOWN_BACK] = processId - nodeGridSizeXY - nodeGridSizeX - 1;
  directions[LEFT_DOWN_FRONT] = processId + nodeGridSizeXY - nodeGridSizeX - 1;
  directions[LEFT_UP_BACK] = processId - nodeGridSizeXY + nodeGridSizeX - 1;
  directions[LEFT_UP_FRONT] = processId + nodeGridSizeXY + nodeGridSizeX - 1;
  directions[RIGHT_DOWN_BACK] = processId - nodeGridSizeXY - nodeGridSizeX + 1;
  directions[RIGHT_DOWN_FRONT] = processId + nodeGridSizeXY - nodeGridSizeX + 1;
  directions[RIGHT_UP_BACK] = processId - nodeGridSizeXY + nodeGridSizeX + 1;
  directions[RIGHT_UP_FRONT] = processId + nodeGridSizeXY + nodeGridSizeX + 1;

  CalculateGridSizeForNode (c1, nodeGridSizeX, totalSize.getX (),
                            c2, nodeGridSizeY, totalSize.getY (),
                            c3, nodeGridSizeZ, totalSize.getZ ());

  currentSize = GridCoordinate (c1, c2, c3);
  size = currentSize + bufferSizeLeft + bufferSizeRight;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%dx%d. %d node(s) unused.\n", processId,
    nodeGridSizeX, nodeGridSizeY, nodeGridSizeZ, left);
  printf ("Grid size for #%d process: %dx%dx%d.\n", processId, c1, c2, c3);
#endif
}
#endif

void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{
  NodeGridInit ();

  // Return if node not used.
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  if (processId >= nodeGridSizeXYZ)
  {
    return;
  }
#endif

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  if (processId >= nodeGridSizeXY)
  {
    return;
  }
#endif

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  if (processId >= nodeGridSizeYZ)
  {
    return;
  }
#endif

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  if (processId >= nodeGridSizeXZ)
  {
    return;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  hasL = false;
  hasR = false;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)
  if (processId > 0)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (processId % nodeGridSizeX > 0)
#endif
  {
    hasL = true;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)
  if (processId < nodeGridSizeX - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (processId % nodeGridSizeX < nodeGridSizeX - 1)
#endif
  {
    hasR = true;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  hasU = false;
  hasD = false;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
  if (processId > 0)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (processId >= nodeGridSizeX)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (processId % nodeGridSizeY > 0)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if ((processId % (nodeGridSizeXY)) >= nodeGridSizeX)
#endif
  {
    hasD = true;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
  if (processId < nodeGridSizeY - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (processId < nodeGridSizeXY - nodeGridSizeX)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (processId % nodeGridSizeY < nodeGridSizeY - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if ((processId % (nodeGridSizeXY)) < nodeGridSizeXY - nodeGridSizeX)
#endif
  {
    hasU = true;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  hasF = false;
  hasB = false;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  if (processId > 0)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (processId >= nodeGridSizeY)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  if (processId >= nodeGridSizeX)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (processId >= nodeGridSizeXY)
#endif
  {
    hasB = true;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  if (processId < nodeGridSizeZ - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (processId < nodeGridSizeYZ - nodeGridSizeY)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  if (processId < nodeGridSizeXZ - nodeGridSizeX)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (processId < nodeGridSizeXYZ - nodeGridSizeXY)
#endif
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

void
Grid::SendReceiveBuffer (BufferPosition bufferDirection)
{
  // Return if node not used.
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  if (processId >= nodeGridSizeXYZ)
  {
    return;
  }
#endif

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  if (processId >= nodeGridSizeXY)
  {
    return;
  }
#endif

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  if (processId >= nodeGridSizeYZ)
  {
    return;
  }
#endif

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  if (processId >= nodeGridSizeXZ)
  {
    return;
  }
#endif

  bool doSend = doShare[bufferDirection].first;
  bool doReceive = doShare[bufferDirection].second;

  // Copy to send buffer
  if (doSend)
  {
    for (grid_iter index = 0, i = sendStart[bufferDirection].getX ();
         i < sendEnd[bufferDirection].getX (); ++i)
    {
      for (grid_coord j = sendStart[bufferDirection].getY ();
           j < sendEnd[bufferDirection].getY (); ++j)
      {
        for (grid_coord k = sendStart[bufferDirection].getZ ();
             k < sendEnd[bufferDirection].getZ (); ++k)
        {
          GridCoordinate pos (i, j, k);
          FieldPointValue* val = getFieldPointValue (pos);
          buffersSend[bufferDirection][index++] = val->getCurValue ();
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
          buffersSend[bufferDirection][index++] = val->getPrevValue ();
#if defined (TWO_TIME_STEPS)
          buffersSend[bufferDirection][index++] = val->getPrevPrevValue ();
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
        }
      }
    }
  }

  BufferPosition opposite = oppositeDirections[bufferDirection];
  int processTo = directions[bufferDirection];
  int processFrom = directions[opposite];

#if PRINT_MESSAGE
  printf ("===Raw #%d directions %s %s %d %d [%d %d].\n", processId, BufferPositionNames[bufferDirection],
          BufferPositionNames[opposite], doSend, doReceive, processTo, processFrom);
#endif

  if (doSend && !doReceive)
  {
    SendRawBuffer (bufferDirection, processTo);
  }
  else if (!doSend && doReceive)
  {
    ReceiveRawBuffer (opposite, processFrom);
  }
  else if (doSend && doReceive)
  {
    SendReceiveRawBuffer (bufferDirection, processTo, opposite, processFrom);
  }
  else
  {
    // Do nothing
  }

  // Copy from receive buffer
  if (doReceive)
  {
    for (grid_iter index = 0, i = recvStart[bufferDirection].getX ();
         i < recvEnd[bufferDirection].getX (); ++i)
    {
      for (grid_coord j = recvStart[bufferDirection].getY ();
           j < recvEnd[bufferDirection].getY (); ++j)
      {
        for (grid_coord k = recvStart[bufferDirection].getZ ();
             k < recvEnd[bufferDirection].getZ (); ++k)
        {
#if defined (TWO_TIME_STEPS)
          FieldPointValue* val = new FieldPointValue (buffersReceive[opposite][index++],
                                                      buffersReceive[opposite][index++],
                                                      buffersReceive[opposite][index++]);
#else /* TWO_TIME_STEPS */
#if defined (ONE_TIME_STEP)
          FieldPointValue* val = new FieldPointValue (buffersReceive[opposite][index++],
                                                      buffersReceive[opposite][index++]);
#else /* ONE_TIME_STEP */
          FieldPointValue* val = new FieldPointValue (buffersReceive[opposite][index++]);
#endif /* !ONE_TIME_STEP */
#endif /* !TWO_TIME_STEPS */

          GridCoordinate pos (i, j, k);
          setFieldPointValue (val, GridCoordinate (pos));
        }
      }
    }
  }
}

#endif /* PARALLEL_GRID */
#endif /* GRID_3D */
