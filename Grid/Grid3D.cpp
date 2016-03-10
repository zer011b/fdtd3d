#include "Grid.h"

#include <cmath>

extern const char* BufferPositionNames[];

#ifdef PARALLEL_GRID
#ifdef GRID_3D

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
void
Grid::NodeGridInit ()
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

void
Grid::GridInit ()
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

  currentSize = GridCoordinate (c1, c2, c3);
  size = currentSize + bufferSizeLeft + bufferSizeRight;
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

void
Grid::GridInit ()
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

  currentSize = GridCoordinate (c1, c2, c3);
  size = currentSize + bufferSizeLeft + bufferSizeRight;
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

void
Grid::GridInit ()
{
  grid_coord c1;
  grid_coord c2;
  grid_coord c3;

  CalculateGridSizeForNode (c1, nodeGridSizeX, hasR, totalSize.getX (),
                            c2, nodeGridSizeY, hasU, totalSize.getY (),
                            c3, nodeGridSizeZ, hasF, totalSize.getZ ());

  currentSize = GridCoordinate (c1, c2, c3);
  size = currentSize + bufferSizeLeft + bufferSizeRight;
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

  GridInit ();

  InitBuffers (numTimeStepsInBuild);
  InitDirections ();

#if PRINT_MESSAGE
  printf ("Grid size for #%d process: %dx%dx%d.\n", processId,
    currentSize.getX (), currentSize.getY (), currentSize.getZ ());
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
  printf ("\tSHARE RAW. PID=#%d. Directions TO(%d)=%s=#%d, FROM(%d)=%s=#%d.\n",
    processId, doSend, BufferPositionNames[bufferDirection], processTo,
               doReceive, BufferPositionNames[opposite], processFrom);
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
