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
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
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
  if ((processId % (nodeGridSizeX * nodeGridSizeY)) >= nodeGridSizeX)
#endif
  {
    hasD = true;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
  if (processId < nodeGridSizeY - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (processId < nodeGridSizeX * nodeGridSizeY - nodeGridSizeX)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (processId % nodeGridSizeY < nodeGridSizeY - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if ((processId % (nodeGridSizeX * nodeGridSizeY)) < nodeGridSizeX * nodeGridSizeY - nodeGridSizeX)
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
  if (processId >= nodeGridSizeX * nodeGridSizeY)
#endif
  {
    hasB = true;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  if (processId < nodeGridSizeZ - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (processId < nodeGridSizeY * nodeGridSizeZ - nodeGridSizeY)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  if (processId < nodeGridSizeX * nodeGridSizeZ - nodeGridSizeX)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (processId < nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ - nodeGridSizeX * nodeGridSizeY)
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

  grid_iter pos1 = 0;
  grid_iter pos2 = 0;
  grid_iter pos3 = 0;
  grid_iter pos4 = 0;
  grid_iter pos5 = 0;
  grid_iter pos6 = 0;

  grid_iter pos7 = 0;
  grid_iter pos8 = 0;
  grid_iter pos9 = 0;
  grid_iter pos10 = 0;
  grid_iter pos11 = 0;
  grid_iter pos12 = 0;

  int processTo;
  int processFrom;

  BufferPosition opposite;

  bool doSend = true;
  bool doReceive = true;

  //printf ("Buffer direction %s\n", BufferPositionNames[bufferDirection]);

  switch (bufferDirection)
  {
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = 2 * bufferSizeLeft.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = size.getX () - bufferSizeRight.getX ();
      pos8 = size.getX ();
      pos9 = bufferSizeLeft.getY ();
      pos10 = size.getY () - bufferSizeRight.getY ();
      pos11 = bufferSizeLeft.getZ ();
      pos12 = size.getZ () - bufferSizeRight.getZ ();

      opposite = RIGHT;
      processTo = processId - 1;
      processFrom = processId + 1;

      if (!hasL)
      {
        doSend = false;
      }
      else if (!hasR)
      {
        doReceive = false;
      }

      break;
    }
    case RIGHT:
    {
      // Send coordinates
      pos1 = size.getX () - 2 * bufferSizeRight.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = 0;
      pos8 = bufferSizeLeft.getX ();
      pos9 = bufferSizeLeft.getY ();
      pos10 = size.getY () - bufferSizeRight.getY ();
      pos11 = bufferSizeLeft.getZ ();
      pos12 = size.getZ () - bufferSizeRight.getZ ();

      opposite = LEFT;
      processTo = processId + 1;
      processFrom = processId - 1;

      if (!hasL)
      {
        doReceive = false;
      }
      else if (!hasR)
      {
        doSend = false;
      }

      break;
    }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case UP:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = size.getY () - 2 * bufferSizeRight.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = bufferSizeLeft.getX ();
      pos8 = size.getX () - bufferSizeRight.getX ();
      pos9 = 0;
      pos10 = bufferSizeLeft.getY ();
      pos11 = bufferSizeLeft.getZ ();
      pos12 = size.getZ () - bufferSizeRight.getZ ();

      opposite = DOWN;
      processTo = processId + nodeGridSizeX;
      processFrom = processId - nodeGridSizeX;

      if (!hasD)
      {
        doReceive = false;
      }
      else if (!hasU)
      {
        doSend = false;
      }

      break;
    }
    case DOWN:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = 2 * bufferSizeLeft.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = bufferSizeLeft.getX ();
      pos8 = size.getX () - bufferSizeRight.getX ();
      pos9 = size.getY () - bufferSizeRight.getY ();
      pos10 = size.getY ();
      pos11 = bufferSizeLeft.getZ ();
      pos12 = size.getZ () - bufferSizeRight.getZ ();

      opposite = UP;
      processTo = processId - nodeGridSizeX;
      processFrom = processId + nodeGridSizeX;

      if (!hasD)
      {
        doSend = false;
      }
      else if (!hasU)
      {
        doReceive = false;
      }

      break;
    }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case FRONT:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = size.getZ () - 2 * bufferSizeRight.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = bufferSizeLeft.getX ();
      pos8 = size.getX () - bufferSizeRight.getX ();
      pos9 = bufferSizeLeft.getY ();
      pos10 = size.getY () - bufferSizeRight.getY ();
      pos11 = 0;
      pos12 = bufferSizeLeft.getZ ();

      opposite = BACK;
      processTo = processId + nodeGridSizeX * nodeGridSizeY;
      processFrom = processId - nodeGridSizeX * nodeGridSizeY;

      if (!hasB)
      {
        doReceive = false;
      }
      else if (!hasF)
      {
        doSend = false;
      }

      break;
    }
    case BACK:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = 2 * bufferSizeLeft.getZ ();

      // Opposite receive coordinates
      pos7 = bufferSizeLeft.getX ();
      pos8 = size.getX () - bufferSizeRight.getX ();
      pos9 = bufferSizeLeft.getY ();
      pos10 = size.getY () - bufferSizeRight.getY ();
      pos11 = size.getZ () - bufferSizeRight.getZ ();
      pos12 = size.getZ ();

      opposite = FRONT;
      processTo = processId - nodeGridSizeX * nodeGridSizeY;
      processFrom = processId + nodeGridSizeX * nodeGridSizeY;

      if (!hasB)
      {
        doSend = false;
      }
      else if (!hasF)
      {
        doReceive = false;
      }

      break;
    }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT_UP:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = 2 * bufferSizeLeft.getX ();
      pos3 = size.getY () - 2 * bufferSizeRight.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = size.getX () - bufferSizeRight.getX ();
      pos8 = size.getX ();
      pos9 = 0;
      pos10 = bufferSizeLeft.getY ();
      pos11 = bufferSizeLeft.getZ ();
      pos12 = size.getZ () - bufferSizeRight.getZ ();

      opposite = RIGHT_DOWN;
      processTo = processId + nodeGridSizeX - 1;
      processFrom = processId - nodeGridSizeX + 1;

      if (!hasR || !hasD)
      {
        doReceive = false;
      }
      if (!hasL || !hasU)
      {
        doSend = false;
      }

      break;
    }
    case LEFT_DOWN:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = 2 * bufferSizeLeft.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = 2 * bufferSizeLeft.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = size.getX () - bufferSizeRight.getX ();
      pos8 = size.getX ();
      pos9 = size.getY () - bufferSizeRight.getY ();
      pos10 = size.getY ();
      pos11 = bufferSizeLeft.getZ ();
      pos12 = size.getZ () - bufferSizeRight.getZ ();

      opposite = RIGHT_UP;
      processTo = processId - nodeGridSizeX - 1;
      processFrom = processId + nodeGridSizeX + 1;

      if (!hasL || !hasD)
      {
        doSend = false;
      }
      if (!hasR || !hasU)
      {
        doReceive = false;
      }

      break;
    }
    case RIGHT_UP:
    {
      // Send coordinates
      pos1 = size.getX () - 2 * bufferSizeRight.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = size.getY () - 2 * bufferSizeRight.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = 0;
      pos8 = bufferSizeLeft.getX ();
      pos9 = 0;
      pos10 = bufferSizeLeft.getY ();
      pos11 = bufferSizeLeft.getZ ();
      pos12 = size.getZ () - bufferSizeRight.getZ ();

      opposite = LEFT_DOWN;
      processTo = processId + nodeGridSizeX + 1;
      processFrom = processId - nodeGridSizeX - 1;

      if (!hasL || !hasD)
      {
        doReceive = false;
      }
      if (!hasR || !hasU)
      {
        doSend = false;
      }

      break;
    }
    case RIGHT_DOWN:
    {
      // Send coordinates
      pos1 = size.getX () - 2 * bufferSizeRight.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = 2 * bufferSizeLeft.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = 0;
      pos8 = bufferSizeLeft.getX ();
      pos9 = size.getY () - bufferSizeRight.getY ();
      pos10 = size.getY ();
      pos11 = bufferSizeLeft.getZ ();
      pos12 = size.getZ () - bufferSizeRight.getZ ();

      opposite = LEFT_UP;
      processTo = processId - nodeGridSizeX + 1;
      processFrom = processId + nodeGridSizeX - 1;

      if (!hasR || !hasD)
      {
        doSend = false;
      }
      if (!hasL || !hasU)
      {
        doReceive = false;
      }

      break;
    }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case UP_FRONT:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = size.getY () - 2 * bufferSizeRight.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = size.getZ () - 2 * bufferSizeRight.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = bufferSizeLeft.getX ();
      pos8 = size.getX () - bufferSizeRight.getX ();
      pos9 = 0;
      pos10 = bufferSizeLeft.getY ();
      pos11 = 0;
      pos12 = bufferSizeLeft.getZ ();

      opposite = DOWN_BACK;
      processTo = processId + nodeGridSizeX + nodeGridSizeX * nodeGridSizeY;
      processFrom = processId - nodeGridSizeX - nodeGridSizeX * nodeGridSizeY;

      if (!hasD || !hasB)
      {
        doReceive = false;
      }
      if (!hasU || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case UP_BACK:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos5 = size.getY () - 2 * bufferSizeRight.getY ();
      pos6 = size.getY () - bufferSizeRight.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = 2 * bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = bufferSizeLeft.getX ();
      pos8 = size.getX () - bufferSizeRight.getX ();
      pos9 = 0;
      pos10 = bufferSizeLeft.getY ();
      pos11 = size.getZ () - bufferSizeRight.getZ ();
      pos12 = size.getZ ();

      opposite = DOWN_FRONT;
      processTo = processId + nodeGridSizeX - nodeGridSizeX * nodeGridSizeY;
      processFrom = processId - nodeGridSizeX + nodeGridSizeX * nodeGridSizeY;

      if (!hasU || !hasB)
      {
        doSend = false;
      }
      if (!hasD || !hasF)
      {
        doReceive = false;
      }

      break;
    }
    case DOWN_FRONT:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = bufferSizeRight.getY ();
      pos4 = 2 * bufferSizeRight.getY ();
      pos5 = size.getZ () - 2 * bufferSizeRight.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = bufferSizeLeft.getX ();
      pos8 = size.getX () - bufferSizeRight.getX ();
      pos9 = size.getY () - bufferSizeRight.getY ();
      pos10 = size.getY ();
      pos11 = 0;
      pos12 = bufferSizeLeft.getZ ();

      opposite = UP_BACK;
      processTo = processId - nodeGridSizeX + nodeGridSizeX * nodeGridSizeY;
      processFrom = processId + nodeGridSizeX - nodeGridSizeX * nodeGridSizeY;

      if (!hasU || !hasB)
      {
        doReceive = false;
      }
      if (!hasD || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case DOWN_BACK:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = 2 * bufferSizeLeft.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = 2 * bufferSizeLeft.getZ ();

      // Opposite receive coordinates
      pos7 = bufferSizeLeft.getX ();
      pos8 = size.getX () - bufferSizeRight.getX ();
      pos9 = size.getY () - bufferSizeRight.getY ();
      pos10 = size.getY ();
      pos11 = size.getZ () - bufferSizeRight.getZ ();
      pos12 = size.getZ ();

      opposite = UP_FRONT;
      processTo = processId - nodeGridSizeX - nodeGridSizeX * nodeGridSizeY;
      processFrom = processId + nodeGridSizeX + nodeGridSizeX * nodeGridSizeY;

      if (!hasD || !hasB)
      {
        doSend = false;
      }
      if (!hasU || !hasF)
      {
        doReceive = false;
      }

      break;
    }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT_FRONT:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = 2 * bufferSizeLeft.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = size.getZ () - 2 * bufferSizeRight.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = size.getX () - bufferSizeRight.getX ();
      pos8 = size.getX ();
      pos9 = bufferSizeLeft.getY ();
      pos10 = size.getY () - bufferSizeRight.getY ();
      pos11 = 0;
      pos12 = bufferSizeLeft.getZ ();

      opposite = RIGHT_BACK;
      processTo = processId + nodeGridSizeX * nodeGridSizeY - 1;
      processFrom = processId - nodeGridSizeX * nodeGridSizeY + 1;

      if (!hasR || !hasB)
      {
        doReceive = false;
      }
      if (!hasL || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case LEFT_BACK:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = 2 * bufferSizeLeft.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = 2 * bufferSizeLeft.getZ ();

      // Opposite receive coordinates
      pos7 = size.getX () - bufferSizeRight.getX ();
      pos8 = size.getX ();
      pos9 = bufferSizeLeft.getY ();
      pos10 = size.getY () - bufferSizeRight.getY ();
      pos11 = size.getZ () - bufferSizeRight.getZ ();
      pos12 = size.getZ ();

      opposite = RIGHT_FRONT;
      processTo = processId - nodeGridSizeX * nodeGridSizeY - 1;
      processFrom = processId + nodeGridSizeX * nodeGridSizeY + 1;

      if (!hasL || !hasB)
      {
        doSend = false;
      }
      if (!hasR || !hasF)
      {
        doReceive = false;
      }

      break;
    }
    case RIGHT_FRONT:
    {
      // Send coordinates
      pos1 = size.getX () - 2 * bufferSizeRight.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = size.getZ () - 2 * bufferSizeRight.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = 0;
      pos8 = bufferSizeLeft.getX ();
      pos9 = bufferSizeLeft.getY ();
      pos10 = size.getY () - bufferSizeRight.getY ();
      pos11 = 0;
      pos12 = bufferSizeLeft.getZ ();

      opposite = LEFT_BACK;
      processTo = processId + nodeGridSizeX * nodeGridSizeY + 1;
      processFrom = processId - nodeGridSizeX * nodeGridSizeY - 1;

      if (!hasL || !hasB)
      {
        doReceive = false;
      }
      if (!hasR || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case RIGHT_BACK:
    {
      // Send coordinates
      pos1 = size.getX () - 2 * bufferSizeRight.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = 2 * bufferSizeLeft.getZ ();

      // Opposite receive coordinates
      pos7 = 0;
      pos8 = bufferSizeLeft.getX ();
      pos9 = bufferSizeLeft.getY ();
      pos10 = size.getY () - bufferSizeRight.getY ();
      pos11 = size.getZ () - bufferSizeRight.getZ ();
      pos12 = size.getZ ();

      opposite = LEFT_FRONT;
      processTo = processId - nodeGridSizeX * nodeGridSizeY + 1;
      processFrom = processId + nodeGridSizeX * nodeGridSizeY - 1;

      if (!hasR || !hasB)
      {
        doSend = false;
      }
      if (!hasL || !hasF)
      {
        doReceive = false;
      }

      break;
    }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT_UP_FRONT:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = 2 * bufferSizeLeft.getX ();
      pos3 = size.getY () - 2 * bufferSizeRight.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = size.getZ () - 2 * bufferSizeRight.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = size.getX () - bufferSizeRight.getX ();
      pos8 = size.getX ();
      pos9 = 0;
      pos10 = bufferSizeLeft.getY ();
      pos11 = 0;
      pos12 = bufferSizeLeft.getZ ();

      opposite = RIGHT_DOWN_BACK;
      processTo = processId + nodeGridSizeX + nodeGridSizeX * nodeGridSizeY - 1;
      processFrom = processId - nodeGridSizeX - nodeGridSizeX * nodeGridSizeY + 1;

      if (!hasR || !hasD || !hasB)
      {
        doReceive = false;
      }
      if (!hasL || !hasU || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case LEFT_UP_BACK:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = 2 * bufferSizeLeft.getX ();
      pos3 = size.getY () - 2 * bufferSizeRight.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = 2 * bufferSizeLeft.getZ ();

      // Opposite receive coordinates
      pos7 = size.getX () - bufferSizeRight.getX ();
      pos8 = size.getX ();
      pos9 = 0;
      pos10 = bufferSizeLeft.getY ();
      pos11 = size.getZ () - bufferSizeRight.getZ ();
      pos12 = size.getZ ();

      opposite = RIGHT_DOWN_FRONT;
      processTo = processId + nodeGridSizeX - nodeGridSizeX * nodeGridSizeY - 1;
      processFrom = processId - nodeGridSizeX + nodeGridSizeX * nodeGridSizeY + 1;

      if (!hasR || !hasD || !hasF)
      {
        doReceive = false;
      }
      if (!hasL || !hasU || !hasB)
      {
        doSend = false;
      }

      break;
    }
    case LEFT_DOWN_FRONT:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = 2 * bufferSizeLeft.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = 2 * bufferSizeLeft.getY ();
      pos5 = size.getZ () - 2 * bufferSizeRight.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = size.getX () - bufferSizeRight.getX ();
      pos8 = size.getX ();
      pos9 = size.getY () - bufferSizeRight.getY ();
      pos10 = size.getY ();
      pos11 = 0;
      pos12 = bufferSizeLeft.getZ ();

      opposite = RIGHT_UP_BACK;
      processTo = processId - nodeGridSizeX + nodeGridSizeX * nodeGridSizeY - 1;
      processFrom = processId + nodeGridSizeX - nodeGridSizeX * nodeGridSizeY + 1;

      if (!hasR || !hasU || !hasB)
      {
        doReceive = false;
      }
      if (!hasL || !hasD || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case LEFT_DOWN_BACK:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = 2 * bufferSizeLeft.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = 2 * bufferSizeLeft.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = 2 * bufferSizeLeft.getZ ();

      // Opposite receive coordinates
      pos7 = size.getX () - bufferSizeRight.getX ();
      pos8 = size.getX ();
      pos9 = size.getY () - bufferSizeRight.getY ();
      pos10 = size.getY ();
      pos11 = size.getZ () - bufferSizeRight.getZ ();
      pos12 = size.getZ ();

      opposite = RIGHT_UP_FRONT;
      processTo = processId - nodeGridSizeX - nodeGridSizeX * nodeGridSizeY - 1;
      processFrom = processId + nodeGridSizeX + nodeGridSizeX * nodeGridSizeY + 1;

      if (!hasR || !hasU || !hasF)
      {
        doReceive = false;
      }
      if (!hasL || !hasD || !hasB)
      {
        doSend = false;
      }

      break;
    }
    case RIGHT_UP_FRONT:
    {
      // Send coordinates
      pos1 = size.getX () - 2 * bufferSizeRight.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = size.getY () - 2 * bufferSizeRight.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = size.getZ () - 2 * bufferSizeRight.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = 0;
      pos8 = bufferSizeLeft.getX ();
      pos9 = 0;
      pos10 = bufferSizeLeft.getY ();
      pos11 = 0;
      pos12 = bufferSizeLeft.getZ ();

      opposite = LEFT_DOWN_BACK;
      processTo = processId + nodeGridSizeX + nodeGridSizeX * nodeGridSizeY + 1;
      processFrom = processId - nodeGridSizeX - nodeGridSizeX * nodeGridSizeY - 1;

      if (!hasL || !hasD || !hasB)
      {
        doReceive = false;
      }
      if (!hasR || !hasU || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case RIGHT_UP_BACK:
    {
      // Send coordinates
      pos1 = size.getX () - 2 * bufferSizeRight.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = size.getY () - 2 * bufferSizeRight.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = 2 * bufferSizeLeft.getZ ();

      // Opposite receive coordinates
      pos7 = 0;
      pos8 = bufferSizeLeft.getX ();
      pos9 = 0;
      pos10 = bufferSizeLeft.getY ();
      pos11 = size.getZ () - bufferSizeRight.getZ ();
      pos12 = size.getZ ();

      opposite = LEFT_DOWN_FRONT;
      processTo = processId + nodeGridSizeX - nodeGridSizeX * nodeGridSizeY + 1;
      processFrom = processId - nodeGridSizeX + nodeGridSizeX * nodeGridSizeY - 1;

      if (!hasL || !hasD || !hasF)
      {
        doReceive = false;
      }
      if (!hasR || !hasU || !hasB)
      {
        doSend = false;
      }

      break;
    }
    case RIGHT_DOWN_FRONT:
    {
      // Send coordinates
      pos1 = size.getX () - 2 * bufferSizeRight.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = 2 * bufferSizeLeft.getY ();
      pos5 = size.getZ () - 2 * bufferSizeRight.getZ ();
      pos6 = size.getZ () - bufferSizeRight.getZ ();

      // Opposite receive coordinates
      pos7 = 0;
      pos8 = bufferSizeLeft.getX ();
      pos9 = size.getY () - bufferSizeRight.getY ();
      pos10 = size.getY ();
      pos11 = 0;
      pos12 = bufferSizeLeft.getZ ();

      opposite = LEFT_UP_BACK;
      processTo = processId - nodeGridSizeX + nodeGridSizeX * nodeGridSizeY + 1;
      processFrom = processId + nodeGridSizeX - nodeGridSizeX * nodeGridSizeY - 1;

      if (!hasL || !hasU || !hasB)
      {
        doReceive = false;
      }
      if (!hasR || !hasD || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case RIGHT_DOWN_BACK:
    {
      // Send coordinates
      pos1 = size.getX () - 2 * bufferSizeRight.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = 2 * bufferSizeLeft.getY ();
      pos5 = bufferSizeLeft.getZ ();
      pos6 = 2 * bufferSizeLeft.getZ ();

      // Opposite receive coordinates
      pos7 = 0;
      pos8 = bufferSizeLeft.getX ();
      pos9 = size.getY () - bufferSizeRight.getY ();
      pos10 = size.getY ();
      pos11 = size.getZ () - bufferSizeRight.getZ ();
      pos12 = size.getZ ();

      opposite = LEFT_UP_FRONT;
      processTo = processId - nodeGridSizeX - nodeGridSizeX * nodeGridSizeY + 1;
      processFrom = processId + nodeGridSizeX + nodeGridSizeX * nodeGridSizeY - 1;

      if (!hasL || !hasU || !hasF)
      {
        doReceive = false;
      }
      if (!hasR || !hasD || !hasB)
      {
        doSend = false;
      }

      break;
    }
#endif
    default:
    {
      UNREACHABLE;
    }
  }

  // Copy to send buffer
  if (doSend)
  {
    for (grid_iter index = 0, i = pos1; i < pos2; ++i)
    {
      for (grid_coord j = pos3; j < pos4; ++j)
      {
        for (grid_coord k = pos5; k < pos6; ++k)
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
    for (grid_iter index = 0, i = pos7; i < pos8; ++i)
    {
      for (grid_coord j = pos9; j < pos10; ++j)
      {
        for (grid_coord k = pos11; k < pos12; ++k)
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
