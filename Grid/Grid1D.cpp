#include "Grid.h"

extern const char* BufferPositionNames[];

#ifdef GRID_1D
#ifdef PARALLEL_GRID

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
void
Grid::NodeGridInit ()
{
  nodeGridSizeX = totalProcCount;
  grid_coord c1;

  CalculateGridSizeForNode (c1, nodeGridSizeX, totalSize.getX ());

  currentSize = GridCoordinate (c1);
  size = currentSize + bufferSizeLeft + bufferSizeRight;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %d.\n", processId,
    nodeGridSizeX);
#endif
}
#endif

void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{
  if (processId != 0)
  {
    buffersSend[LEFT].resize (bufferSizeLeft.getX () * numTimeStepsInBuild);
    buffersReceive[LEFT].resize (bufferSizeLeft.getX () * numTimeStepsInBuild);
  }

  if (processId != totalProcCount - 1)
  {
    buffersSend[RIGHT].resize (bufferSizeRight.getX () * numTimeStepsInBuild);
    buffersReceive[RIGHT].resize (bufferSizeRight.getX () * numTimeStepsInBuild);
  }
}

void
Grid::SendReceiveBuffer (BufferPosition bufferDirection)
{
  grid_iter pos1 = 0;
  grid_iter pos2 = 0;

  grid_iter pos3 = 0;
  grid_iter pos4 = 0;

  int processTo;
  int processFrom;

  BufferPosition opposite;

  bool doSend = true;
  bool doReceive = true;

  switch (bufferDirection)
  {
    case LEFT:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = 2 * bufferSizeLeft.getX ();

      // Opposite receive coordinates
      pos3 = size.getX () - bufferSizeRight.getX ();
      pos4 = size.getX ();

      opposite = RIGHT;
      processTo = processId - 1;
      processFrom = processId + 1;

      if (processId == 0)
      {
        doSend = false;
      }
      else if (processId == totalProcCount - 1)
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

      // Opposite receive coordinates
      pos3 = 0;
      pos4 = bufferSizeLeft.getX ();

      opposite = LEFT;
      processTo = processId + 1;
      processFrom = processId - 1;

      if (processId == 0)
      {
        doReceive = false;
      }
      else if (processId == totalProcCount - 1)
      {
        doSend = false;
      }

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  // Copy to send buffer
  if (doSend)
  {
    for (grid_iter index = 0, pos = pos1; pos < pos2; ++pos)
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
    for (grid_iter index = 0, pos = pos3; pos < pos4; ++pos)
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
