#include "Grid.h"

extern const char* BufferPositionNames[];

#ifdef GRID_2D
#ifdef PARALLEL_GRID

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
void
Grid::NodeGridInit ()
{
  grid_coord c1;
  grid_coord c2;

#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  nodeGridSizeX = totalProcCount;
  nodeGridSizeY = 1;

  CalculateGridSizeForNode (c1, nodeGridSizeX, totalSize.getX ());
  c2 = totalSize.getY ();
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  nodeGridSizeX = 1;
  nodeGridSizeY = totalProcCount;

  c1 = totalSize.getX ();
  CalculateGridSizeForNode (c2, nodeGridSizeY, totalSize.getY ());
#endif

  currentSize = GridCoordinate (c1, c2);
  size = currentSize + bufferSizeLeft + bufferSizeRight;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%d.\n", processId,
    nodeGridSizeX, nodeGridSizeY);
#endif
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

  grid_coord c1;
  grid_coord c2;

  CalculateGridSizeForNode (c1, nodeGridSizeX, totalSize.getX (), c2, nodeGridSizeY, totalSize.getY ());

  currentSize = GridCoordinate (c1, c2);
  size = currentSize + bufferSizeLeft + bufferSizeRight;

#if PRINT_MESSAGE
  printf ("Nodes' grid process #%d: %dx%d. %d node(s) unused.\n", processId,
    nodeGridSizeX, nodeGridSizeY, left);
#endif
}
#endif

void
Grid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{
  NodeGridInit ();

  // Return if node not used.
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  if (processId >= nodeGridSizeX * nodeGridSizeY)
  {
    return;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  hasL = false;
  hasR = false;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)
  if (processId > 0)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (processId % nodeGridSizeX > 0)
#endif
  {
    hasL = true;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)
  if (processId < nodeGridSizeX - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (processId % nodeGridSizeX < nodeGridSizeX - 1)
#endif
  {
    hasR = true;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  hasU = false;
  hasD = false;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
  if (processId > 0)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (processId >= nodeGridSizeX)
#endif
  {
    hasD = true;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
  if (processId < nodeGridSizeY - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (processId < nodeGridSizeX * nodeGridSizeY - nodeGridSizeX)
#endif
  {
    hasU = true;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (hasL)
  {
    buffersSend[LEFT].resize (bufferSizeLeft.getX () * currentSize.getY () * numTimeStepsInBuild);
    buffersReceive[LEFT].resize (bufferSizeLeft.getX () * currentSize.getY () * numTimeStepsInBuild);
  }
  if (hasR)
  {
    buffersSend[RIGHT].resize (bufferSizeRight.getX () * currentSize.getY () * numTimeStepsInBuild);
    buffersReceive[RIGHT].resize (bufferSizeRight.getX () * currentSize.getY () * numTimeStepsInBuild);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
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
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (hasL && hasD)
  {
    buffersSend[LEFT_DOWN].resize (bufferSizeLeft.getX () * bufferSizeLeft.getY () * numTimeStepsInBuild);
    buffersReceive[LEFT_DOWN].resize (bufferSizeLeft.getX () * bufferSizeLeft.getY () * numTimeStepsInBuild);
  }
  if (hasL && hasU)
  {
    buffersSend[LEFT_UP].resize (bufferSizeLeft.getX () * bufferSizeRight.getY () * numTimeStepsInBuild);
    buffersReceive[LEFT_UP].resize (bufferSizeLeft.getX () * bufferSizeRight.getY () * numTimeStepsInBuild);
  }
  if (hasR && hasD)
  {
    buffersSend[RIGHT_DOWN].resize (bufferSizeRight.getX () * bufferSizeLeft.getY () * numTimeStepsInBuild);
    buffersReceive[RIGHT_DOWN].resize (bufferSizeRight.getX () * bufferSizeLeft.getY () * numTimeStepsInBuild);
  }
  if (hasR && hasU)
  {
    buffersSend[RIGHT_UP].resize (bufferSizeRight.getX () * bufferSizeRight.getY () * numTimeStepsInBuild);
    buffersReceive[RIGHT_UP].resize (bufferSizeRight.getX () * bufferSizeRight.getY () * numTimeStepsInBuild);
  }
#endif
}

void
Grid::SendReceiveBuffer (BufferPosition bufferDirection)
{
  // Return if node not used.
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (processId >= nodeGridSizeX * nodeGridSizeY)
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

  int processTo;
  int processFrom;

  BufferPosition opposite;

  bool doSend = true;
  bool doReceive = true;

  //printf ("Buffer direction %s\n", BufferPositionNames[bufferDirection]);

  switch (bufferDirection)
  {
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
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

      // Opposite receive coordinates
      pos5 = 0;
      pos6 = bufferSizeLeft.getX ();
      pos7 = bufferSizeLeft.getY ();
      pos8 = size.getY () - bufferSizeRight.getY ();

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
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
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

      if (!hasD)
      {
        doReceive = false;
      }
      if (!hasU)
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
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
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

      // Opposite receive coordinates
      pos5 = size.getX () - bufferSizeRight.getX ();
      pos6 = size.getX ();
      pos7 = size.getY () - bufferSizeRight.getY ();
      pos8 = size.getY ();

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

      // Opposite receive coordinates
      pos5 = 0;
      pos6 = bufferSizeLeft.getX ();
      pos7 = 0;
      pos8 = bufferSizeLeft.getY ();

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

      // Opposite receive coordinates
      pos5 = 0;
      pos6 = bufferSizeLeft.getX ();
      pos7 = size.getY () - bufferSizeRight.getY ();
      pos8 = size.getY ();

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

#endif /* PARALLEL_GRID */
#endif /* GRID_2D */
