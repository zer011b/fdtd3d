#include "Grid.h"

extern const char* BufferPositionNames[];

#ifdef GRID_1D
#ifdef PARALLEL_GRID

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
void
Grid::NodeGridInit ()
{
  grid_coord c1;

  nodeGridSizeX = totalProcCount;

  CalculateGridSizeForNode (c1, nodeGridSizeX, totalSize.getX ());

  currentSize = GridCoordinate (c1);
  size = currentSize + bufferSizeLeft + bufferSizeRight;

  directions[LEFT] = processId - 1;
  directions[RIGHT] = processId + 1;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %d.\n", processId,
    nodeGridSizeX);
#endif
}
#endif

void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{
  NodeGridInit ();

  hasL = false;
  hasR = false;

  if (processId > 0)
  {
    hasL = true;
  }

  if (processId < totalProcCount - 1)
  {
    hasR = true;
  }

  if (hasL)
  {
    buffersSend[LEFT].resize (bufferSizeLeft.getX () * numTimeStepsInBuild);
    buffersReceive[LEFT].resize (bufferSizeLeft.getX () * numTimeStepsInBuild);
  }

  if (hasR)
  {
    buffersSend[RIGHT].resize (bufferSizeRight.getX () * numTimeStepsInBuild);
    buffersReceive[RIGHT].resize (bufferSizeRight.getX () * numTimeStepsInBuild);
  }
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
