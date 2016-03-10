#include "Grid.h"

extern const char* BufferPositionNames[];

#ifdef GRID_1D
#ifdef PARALLEL_GRID

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
void
Grid::NodeGridInit ()
{
  nodeGridSizeX = totalProcCount;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %d.\n", processId,
    nodeGridSizeX);
#endif
}

void
Grid::GridInit ()
{
  grid_coord c1;

  CalculateGridSizeForNode (c1, nodeGridSizeX, hasR, totalSize.getX ());

  currentSize = GridCoordinate (c1);
  size = currentSize + bufferSizeLeft + bufferSizeRight;
}
#endif

void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{
  NodeGridInit ();

  GridInit ();

  InitBuffers (numTimeStepsInBuild);
  InitDirections ();

#if PRINT_MESSAGE
  printf ("Grid size for #%d process: %d.\n", processId,
    currentSize.getX ());
#endif
}

void
Grid::SendReceiveBuffer (BufferPosition bufferDirection)
{
  bool doSend = doShare[bufferDirection].first;
  bool doReceive = doShare[bufferDirection].second;

  // Copy to send buffer
  if (doSend)
  {
    for (grid_iter index = 0, pos = sendStart[bufferDirection].getX ();
         pos < sendEnd[bufferDirection].getX (); ++pos)
    {
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

  BufferPosition opposite = oppositeDirections[bufferDirection];
  int processTo = directions[bufferDirection];
  int processFrom = directions[opposite];

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
    for (grid_iter index = 0, pos = recvStart[bufferDirection].getX ();
         pos < recvEnd[bufferDirection].getX (); ++pos)
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

      setFieldPointValue (val, GridCoordinate (pos));
    }
  }
}

#endif /* PARALLEL_GRID */
#endif /* GRID_1D */
