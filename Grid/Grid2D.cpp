#include "Grid.h"

extern const char* BufferPositionNames[];

#ifdef GRID_2D
#ifdef PARALLEL_GRID

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
void
Grid::NodeGridInit ()
{
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  nodeGridSizeX = totalProcCount;
  nodeGridSizeY = 1;
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  nodeGridSizeX = 1;
  nodeGridSizeY = totalProcCount;
#endif

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%d.\n", processId,
    nodeGridSizeX, nodeGridSizeY);
#endif
}

void
Grid::GridInit ()
{
  grid_coord c1;
  grid_coord c2;

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  CalculateGridSizeForNode (c1, nodeGridSizeX, hasR, totalSize.getX ());
  c2 = totalSize.getY ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  c1 = totalSize.getX ();
  CalculateGridSizeForNode (c2, nodeGridSizeY, hasU, totalSize.getY ());
#endif

  currentSize = GridCoordinate (c1, c2);
  size = currentSize + bufferSizeLeft + bufferSizeRight;
}
#endif

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
void
Grid::NodeGridInit ()
{
  if (totalProcCount < 4)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 2D parallel buffers. Use 1D ones.");
  }

  FieldValue overall1 = (FieldValue) totalSize.getX ();
  FieldValue overall2 = (FieldValue) totalSize.getY ();

  int left;
  NodeGridInitInner (overall1, overall2, nodeGridSizeX, nodeGridSizeY, left);

  nodeGridSizeXY = nodeGridSizeX * nodeGridSizeY;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%d. %d node(s) unused.\n", processId,
    nodeGridSizeX, nodeGridSizeY, left);
#endif
}

void
Grid::GridInit ()
{
  grid_coord c1;
  grid_coord c2;

  CalculateGridSizeForNode (c1, nodeGridSizeX, hasR, totalSize.getX (),
                            c2, nodeGridSizeY, hasU, totalSize.getY ());

  currentSize = GridCoordinate (c1, c2);
  size = currentSize + bufferSizeLeft + bufferSizeRight;
}
#endif

void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{
  NodeGridInit ();

  // Return if node not used.
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  if (processId >= nodeGridSizeXY)
  {
    return;
  }
#endif

  InitBuffers (numTimeStepsInBuild);
  InitDirections ();

  GridInit ();

#if PRINT_MESSAGE
  printf ("Grid size for #%d process: %dx%d.\n", processId,
    currentSize.getX (), currentSize.getY ());
#endif
}

void
Grid::SendReceiveBuffer (BufferPosition bufferDirection)
{
  // Return if node not used.
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (processId >= nodeGridSizeXY)
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
        GridCoordinate pos (i, j);
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

        GridCoordinate pos (i, j);
        setFieldPointValue (val, GridCoordinate (pos));
      }
    }
  }
}

#endif /* PARALLEL_GRID */
#endif /* GRID_2D */
