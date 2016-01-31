#include <cmath>

#include "Grid.h"

extern const char* BufferPositionNames[];

#ifdef GRID_2D


#ifdef PARALLEL_BUFFER_DIMENSION_1D
void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{
  nodeGridSizeX = totalProcCount;
  nodeGridSizeY = 1;

  if (processId % nodeGridSizeX != 0)
  {
    buffersSend[LEFT].resize (bufferSizeLeft.getX () * currentSize.getY () * numTimeStepsInBuild);
    buffersReceive[LEFT].resize (bufferSizeLeft.getX () * currentSize.getY () * numTimeStepsInBuild);
  }

  if ((processId + 1) % nodeGridSizeX != 0)
  {
    buffersSend[RIGHT].resize (bufferSizeRight.getX () * currentSize.getY () * numTimeStepsInBuild);
    buffersReceive[RIGHT].resize (bufferSizeRight.getX () * currentSize.getY () * numTimeStepsInBuild);
  }
}
#endif /* PARALLEL_BUFFER_DIMENSION_1D */


#ifdef PARALLEL_BUFFER_DIMENSION_2D
void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{
  if (totalProcCount < 4)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 2D parallel buffers. Use 1D ones.");
  }

  grid_coord overall_x = totalSize.getX ();
  grid_coord overall_y = totalSize.getY ();

  FieldValue alpha = 0;
  if (overall_x > overall_y)
  {
    alpha = overall_x / overall_y;
  }
  else
  {
    alpha = overall_y / overall_x;
  }

  FieldValue sqrtVal = ((FieldValue) (totalProcCount)) / alpha;
  sqrtVal = sqrt (sqrtVal);

  if (sqrtVal <= 1)
  {
    ASSERT_MESSAGE ("Unsupported number of nodes for 2D parallel buffers. Use 1D ones.");
  }

  sqrtVal = round (sqrtVal);
  ASSERT (sqrtVal == floor (sqrtVal));

  if (overall_x > overall_y)
  {
    nodeGridSizeY = (int) sqrtVal;
    nodeGridSizeX = totalProcCount / nodeGridSizeX;
  }
  else
  {
    nodeGridSizeX = (int) sqrtVal;
    nodeGridSizeY = totalProcCount / nodeGridSizeX;
  }

  int left = totalProcCount - nodeGridSizeX * nodeGridSizeY;

  // Considerable. Could give up if only one left
  if (left > 0) /* left > 1 */
  {
    // Bad case, too many nodes left unused. Let's change proportion.
    bool find = true;
    bool directionX = nodeGridSizeX > nodeGridSizeY ? true : false;
    while (find)
    {
      find = false;
      if (directionX && nodeGridSizeX > 2)
      {
        find = true;
        --nodeGridSizeX;
        nodeGridSizeY = totalProcCount / nodeGridSizeX;
      }
      else if (!directionX && nodeGridSizeY > 2)
      {
        find = true;
        --nodeGridSizeY;
        nodeGridSizeX = totalProcCount / nodeGridSizeY;
      }

      left = totalProcCount - nodeGridSizeX * nodeGridSizeY;

      if (find && left == 0)
      {
        find = false;
      }
    }
  }

  ASSERT (nodeGridSizeX > 1 && nodeGridSizeY > 1);

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%d. %d node(s) unused.\n", processId,
    nodeGridSizeX, nodeGridSizeY, left);
#endif

  if (processId >= nodeGridSizeX * nodeGridSizeY)
  {
    return;
  }

  bool hasL = false;
  bool hasR = false;
  bool hasU = false;
  bool hasD = false;

  if (processId % nodeGridSizeX != 0)
  {
    hasL = true;
  }

  if ((processId + 1) % nodeGridSizeX != 0)
  {
    hasR = true;
  }

  if (processId >= nodeGridSizeX)
  {
    hasD = true;
  }

  if (processId < nodeGridSizeX * nodeGridSizeY - nodeGridSizeX)
  {
    hasU = true;
  }

  if (hasL)
  {
    buffersSend[LEFT].resize (bufferSizeLeft.getX () * currentSize.getY () * numTimeStepsInBuild);
    buffersReceive[LEFT].resize (bufferSizeLeft.getX () * currentSize.getY () * numTimeStepsInBuild);

    if (hasD)
    {
      buffersSend[LEFT_DOWN].resize (bufferSizeLeft.getX () * bufferSizeLeft.getY () * numTimeStepsInBuild);
      buffersReceive[LEFT_DOWN].resize (bufferSizeLeft.getX () * bufferSizeLeft.getY () * numTimeStepsInBuild);
    }
    if (hasU)
    {
      buffersSend[LEFT_UP].resize (bufferSizeLeft.getX () * bufferSizeRight.getY () * numTimeStepsInBuild);
      buffersReceive[LEFT_UP].resize (bufferSizeLeft.getX () * bufferSizeRight.getY () * numTimeStepsInBuild);
    }
  }

  if (hasR)
  {
    buffersSend[RIGHT].resize (bufferSizeRight.getX () * currentSize.getY () * numTimeStepsInBuild);
    buffersReceive[RIGHT].resize (bufferSizeRight.getX () * currentSize.getY () * numTimeStepsInBuild);

    if (hasD)
    {
      buffersSend[RIGHT_DOWN].resize (bufferSizeRight.getX () * bufferSizeLeft.getY () * numTimeStepsInBuild);
      buffersReceive[RIGHT_DOWN].resize (bufferSizeRight.getX () * bufferSizeLeft.getY () * numTimeStepsInBuild);
    }
    if (hasU)
    {
      buffersSend[RIGHT_UP].resize (bufferSizeRight.getX () * bufferSizeRight.getY () * numTimeStepsInBuild);
      buffersReceive[RIGHT_UP].resize (bufferSizeRight.getX () * bufferSizeRight.getY () * numTimeStepsInBuild);
    }
  }

  if (hasD)
  {
    buffersSend[DOWN].resize (bufferSizeLeft.getY () * currentSize.getX () * numTimeStepsInBuild);
    buffersReceive[DOWN].resize (bufferSizeLeft.getY () * currentSize.getX () * numTimeStepsInBuild);
  }

  if (hasU)
  {
    buffersSend[UP].resize (bufferSizeRight.getY () * currentSize.getX () * numTimeStepsInBuild);
    buffersReceive[UP].resize (bufferSizeRight.getY () * currentSize.getX () * numTimeStepsInBuild);
  }
}
#endif /* PARALLEL_BUFFER_DIMENSION_2D */

#if defined (PARALLEL_BUFFER_DIMENSION_1D) || defined (PARALLEL_BUFFER_DIMENSION_2D)
void
Grid::SendReceiveBuffer (BufferPosition bufferDirection)
{
  if (processId >= nodeGridSizeX * nodeGridSizeY)
  {
    return;
  }

  grid_iter pos1 = 0;
  grid_iter pos2 = 0;
  grid_iter pos3 = 0;
  grid_iter pos4 = 0;

  grid_iter pos5 = 0;
  grid_iter pos6 = 0;
  grid_iter pos7 = 0;
  grid_iter pos8 = 0;

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
      pos3 = bufferSizeLeft.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();

      // Opposite receive coordinates
      pos5 = size.getX () - bufferSizeRight.getX ();
      pos6 = size.getX ();
      pos7 = bufferSizeLeft.getY ();
      pos8 = size.getY () - bufferSizeRight.getY ();

      opposite = RIGHT;
      processTo = processId - 1;
      processFrom = processId + 1;

      if (processId % nodeGridSizeX == 0)
      {
        doSend = false;
      }
      else if ((processId + 1) % nodeGridSizeX == 0)
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

      // Opposite receive coordinates
      pos5 = 0;
      pos6 = bufferSizeLeft.getX ();
      pos7 = bufferSizeLeft.getY ();
      pos8 = size.getY () - bufferSizeRight.getY ();

      opposite = LEFT;
      processTo = processId + 1;
      processFrom = processId - 1;

      if (processId % nodeGridSizeX == 0)
      {
        doReceive = false;
      }
      else if ((processId + 1) % nodeGridSizeX == 0)
      {
        doSend = false;
      }

      break;
    }
#ifdef PARALLEL_BUFFER_DIMENSION_2D
    case UP:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = size.getY () - 2 * bufferSizeRight.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();

      // Opposite receive coordinates
      pos5 = bufferSizeLeft.getX ();
      pos6 = size.getX () - bufferSizeRight.getX ();
      pos7 = 0;
      pos8 = bufferSizeLeft.getY ();

      opposite = DOWN;
      processTo = processId + nodeGridSizeX;
      processFrom = processId - nodeGridSizeX;

      if (processId < nodeGridSizeX)
      {
        doReceive = false;
      }
      else if (processId >= nodeGridSizeX * nodeGridSizeY - nodeGridSizeX)
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

      // Opposite receive coordinates
      pos5 = bufferSizeLeft.getX ();
      pos6 = size.getX () - bufferSizeRight.getX ();
      pos7 = size.getY () - bufferSizeRight.getY ();
      pos8 = size.getY ();

      opposite = UP;
      processTo = processId - nodeGridSizeX;
      processFrom = processId + nodeGridSizeX;

      if (processId < nodeGridSizeX)
      {
        doSend = false;
      }
      else if (processId >= nodeGridSizeX * nodeGridSizeY - nodeGridSizeX)
      {
        doReceive = false;
      }

      break;
    }
    case LEFT_UP:
    {
      // Send coordinates
      pos1 = bufferSizeLeft.getX ();
      pos2 = 2 * bufferSizeLeft.getX ();
      pos3 = size.getY () - 2 * bufferSizeRight.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();

      // Opposite receive coordinates
      pos5 = size.getX () - bufferSizeRight.getX ();
      pos6 = size.getX ();
      pos7 = 0;
      pos8 = bufferSizeLeft.getY ();

      opposite = RIGHT_DOWN;
      processTo = processId + nodeGridSizeX - 1;
      processFrom = processId - nodeGridSizeX + 1;;

      if (processId < nodeGridSizeX || (processId + 1) % nodeGridSizeX == 0)
      {
        doReceive = false;
      }
      if (processId >= nodeGridSizeX * nodeGridSizeY - nodeGridSizeX || processId % nodeGridSizeX == 0)
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

      // Opposite receive coordinates
      pos5 = size.getX () - bufferSizeRight.getX ();
      pos6 = size.getX ();
      pos7 = size.getY () - bufferSizeRight.getY ();
      pos8 = size.getY ();

      opposite = RIGHT_UP;
      processTo = processId - nodeGridSizeX - 1;
      processFrom = processId + nodeGridSizeX + 1;

      if (processId < nodeGridSizeX || processId % nodeGridSizeX == 0)
      {
        doSend = false;
      }
      if (processId >= nodeGridSizeX * nodeGridSizeY - nodeGridSizeX || (processId + 1) % nodeGridSizeX == 0)
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

      // Opposite receive coordinates
      pos5 = 0;
      pos6 = bufferSizeLeft.getX ();
      pos7 = 0;
      pos8 = bufferSizeLeft.getY ();

      opposite = LEFT_DOWN;
      processTo = processId + nodeGridSizeX + 1;
      processFrom = processId - nodeGridSizeX - 1;

      if (processId < nodeGridSizeX || processId % nodeGridSizeX == 0)
      {
        doReceive = false;
      }
      if (processId >= nodeGridSizeX * nodeGridSizeY - nodeGridSizeX || (processId + 1) % nodeGridSizeX == 0)
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

      // Opposite receive coordinates
      pos5 = 0;
      pos6 = bufferSizeLeft.getX ();
      pos7 = size.getY () - bufferSizeRight.getY ();
      pos8 = size.getY ();

      opposite = LEFT_UP;
      processTo = processId - nodeGridSizeX + 1;
      processFrom = processId + nodeGridSizeX - 1;

      if (processId < nodeGridSizeX || (processId + 1) % nodeGridSizeX == 0)
      {
        doSend = false;
      }
      if (processId >= nodeGridSizeX * nodeGridSizeY - nodeGridSizeX || processId % nodeGridSizeX == 0)
      {
        doReceive = false;
      }

      break;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_2D */
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
    for (grid_iter index = 0, i = pos5; i < pos6; ++i)
    {
      for (grid_coord j = pos7; j < pos8; ++j)
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
#endif /* PARALLEL_BUFFER_DIMENSION_1D || PARALLEL_BUFFER_DIMENSION_2D */


#endif /* GRID_2D */
