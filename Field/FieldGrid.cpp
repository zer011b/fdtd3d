#include <iostream>
#include <cmath>

#include "FieldGrid.h"
#include "Assert.h"

// ================================ GridSize ================================
GridCoordinate::GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& sx
#if defined (GRID_2D) || defined (GRID_3D)
  , const grid_coord& sy
#if defined (GRID_3D)
  , const grid_coord& sz
#endif  /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D*/
  ) :
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  x (sx)
#if defined (GRID_2D) || defined (GRID_3D)
  , y (sy)
#if defined (GRID_3D)
  , z (sz)
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D*/
{
}

GridCoordinate::~GridCoordinate ()
{
}

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
const grid_coord&
GridCoordinate::getX () const
{
  return x;
}
#if defined (GRID_2D) || defined (GRID_3D)
const grid_coord&
GridCoordinate::getY () const
{
  return y;
}
#if defined (GRID_3D)
const grid_coord&
GridCoordinate::getZ () const
{
  return z;
}
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D*/

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
grid_iter
GridCoordinate::calculateTotalCoord () const
{
#if defined (GRID_1D)
  return x;
#else /* GRID_1D */
#if defined (GRID_2D)
  return x * y;
#else /* GRID_2D */
#if defined (GRID_3D)
  return x * y * z;
#endif /* GRID_3D */
#endif /* !GRID_2D */
#endif /* !GRID_1D */
}
#endif /* GRID_1D || GRID_2D || GRID_3D*/

// ================================ Grid ================================

#if defined (PRINT_MESSAGE)
const char* BufferPositionNames[] =
{
#if defined (GRID_1D)
  "LEFT",
  "RIGHT"
#endif
#if defined (GRID_2D)
  "LEFT",
  "RIGHT",
  "UP",
  "DOWN",
  "LEFT_UP",
  "LEFT_DOWN",
  "RIGHT_UP",
  "RIGHT_DOWN",
#endif
#if defined (GRID_3D)
  "LEFT",
  "RIGHT",
  "UP",
  "DOWN",
  "FRONT",
  "BACK",
  "LEFT_FRONT",
  "LEFT_BACK",
  "LEFT_UP",
  "LEFT_DOWN",
  "RIGHT_FRONT",
  "RIGHT_BACK",
  "RIGHT_UP",
  "RIGHT_DOWN",
  "UP_FRONT",
  "UP_BACK",
  "DOWN_FRONT",
  "DOWN_BACK",
  "LEFT_UP_FRONT",
  "LEFT_UP_BACK",
  "LEFT_DOWN_FRONT",
  "LEFT_DOWN_BACK",
  "RIGHT_UP_FRONT",
  "RIGHT_UP_BACK",
  "RIGHT_DOWN_FRONT",
  "RIGHT_DOWN_BACK",
#endif /* GRID_3D */
};
#endif

#if defined (PARALLEL_GRID)
Grid::Grid (const GridCoordinate& totSize, const GridCoordinate& curSize,
            const GridCoordinate& bufSizeL, const GridCoordinate& bufSizeR,
            const int process, const int totalProc) :
  size (curSize + bufSizeL + bufSizeR),
  currentSize (curSize),
  bufferSizeLeft (bufSizeL),
  bufferSizeRight (bufSizeR),
  totalSize (totSize),
  processId (process),
  totalProcCount (totalProc)
{
  gridValues.resize (size.calculateTotalCoord ());

#if PRINT_MESSAGE
  printf ("New grid for proc: %d (of %d) with raw size: %lu.\n", process, totalProcCount, gridValues.size ());
#endif

#if defined (ONE_TIME_STEP)
  grid_iter numTimeStepsInBuild = 2;
#endif
#if defined (TWO_TIME_STEPS)
  grid_iter numTimeStepsInBuild = 3;
#endif

  buffersSend.resize (BUFFER_COUNT);
  buffersReceive.resize (BUFFER_COUNT);

#if defined (GRID_1D)
  buffersSend[LEFT].resize (bufferSizeLeft.getX () * numTimeStepsInBuild);
  buffersSend[RIGHT].resize (bufferSizeRight.getX () * numTimeStepsInBuild);

  buffersReceive[LEFT].resize (bufferSizeLeft.getX () * numTimeStepsInBuild);
  buffersReceive[RIGHT].resize (bufferSizeRight.getX () * numTimeStepsInBuild);
#endif

#if defined (GRID_2D)
  FieldValue sqrtVal = sqrt ((FieldValue) totalProcCount);
  ASSERT (sqrtVal == floor (sqrtVal));
  sqrtProc = (int) sqrtVal;

  buffersSend[LEFT].resize (bufferSizeLeft.getX () * currentSize.getY () * numTimeStepsInBuild);
  buffersSend[RIGHT].resize (bufferSizeRight.getX () * currentSize.getY () * numTimeStepsInBuild);
  buffersSend[DOWN].resize (bufferSizeLeft.getY () * currentSize.getX () * numTimeStepsInBuild);
  buffersSend[UP].resize (bufferSizeRight.getY () * currentSize.getX () * numTimeStepsInBuild);
  buffersSend[LEFT_DOWN].resize (bufferSizeLeft.getX () * bufferSizeLeft.getY () * numTimeStepsInBuild);
  buffersSend[LEFT_UP].resize (bufferSizeLeft.getX () * bufferSizeRight.getY () * numTimeStepsInBuild);
  buffersSend[RIGHT_DOWN].resize (bufferSizeRight.getX () * bufferSizeLeft.getY () * numTimeStepsInBuild);
  buffersSend[RIGHT_UP].resize (bufferSizeRight.getX () * bufferSizeRight.getY () * numTimeStepsInBuild);

  buffersReceive[LEFT].resize (bufferSizeLeft.getX () * currentSize.getY () * numTimeStepsInBuild);
  buffersReceive[RIGHT].resize (bufferSizeRight.getX () * currentSize.getY () * numTimeStepsInBuild);
  buffersReceive[DOWN].resize (bufferSizeLeft.getY () * currentSize.getX () * numTimeStepsInBuild);
  buffersReceive[UP].resize (bufferSizeRight.getY () * currentSize.getX () * numTimeStepsInBuild);
  buffersReceive[LEFT_DOWN].resize (bufferSizeLeft.getX () * bufferSizeLeft.getY () * numTimeStepsInBuild);
  buffersReceive[LEFT_UP].resize (bufferSizeLeft.getX () * bufferSizeRight.getY () * numTimeStepsInBuild);
  buffersReceive[RIGHT_DOWN].resize (bufferSizeRight.getX () * bufferSizeLeft.getY () * numTimeStepsInBuild);
  buffersReceive[RIGHT_UP].resize (bufferSizeRight.getX () * bufferSizeRight.getY () * numTimeStepsInBuild);
#endif

#if defined (GRID_3D)

#endif
}
#else /* PARALLEL_GRID */
Grid::Grid(const GridCoordinate& s) :
  size (s)
{
  gridValues.resize (size.calculateTotalCoord ());

#if PRINT_MESSAGE
  printf ("New grid with raw size: %lu.\n", gridValues.size ());
#endif
}
#endif /* !PARALLEL_GRID */

Grid::~Grid ()
{
  for (FieldPointValue* current : gridValues)
  {
    if (current)
    {
      delete current;
    }
  }
}

const GridCoordinate& Grid::getSize () const
{
  return size;
}

VectorFieldPointValues& Grid::getValues ()
{
  return gridValues;
}

bool
Grid::isLegitIndexWithSize (const GridCoordinate& position, const GridCoordinate& sizeCoord) const
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& px = position.getX ();
  const grid_coord& sx = sizeCoord.getX ();
#if defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& py = position.getY ();
  const grid_coord& sy = sizeCoord.getY ();
#if defined (GRID_3D)
  const grid_coord& pz = position.getZ ();
  const grid_coord& sz = sizeCoord.getZ ();
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D*/

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  if (px < 0 || px >= sx)
  {
    return false;
  }
#if defined (GRID_2D) || defined (GRID_3D)
  else if (py < 0 || py >= sy)
  {
    return false;
  }
#if defined (GRID_3D)
  else if (pz < 0 || pz >= sz)
  {
    return false;
  }
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D*/

  return true;
}

bool
Grid::isLegitIndex (const GridCoordinate& position) const
{
  return isLegitIndexWithSize (position, size);
}

grid_iter
Grid::calculateIndexFromPositionWithSize (const GridCoordinate& position,
                                          const GridCoordinate& sizeCoord) const
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& px = position.getX ();
  const grid_coord& sx = sizeCoord.getX ();
#if defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& py = position.getY ();
  const grid_coord& sy = sizeCoord.getY ();
#if defined (GRID_3D)
  const grid_coord& pz = position.getZ ();
  const grid_coord& sz = sizeCoord.getZ ();
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D*/

  grid_coord coord = 0;

#if defined (GRID_1D)
  coord = px;
#else /* GRID_1D */
#if defined (GRID_2D)
  coord = px * sy + py;
#else /* GRID_2D */
#if defined (GRID_3D)
  coord = px * sy * sz + py * sz + pz;
#endif /* GRID_3D */
#endif /* !GRID_2D */
#endif /* !GRID_1D */

  return coord;
}

grid_iter
Grid::calculateIndexFromPosition (const GridCoordinate& position) const
{
  return calculateIndexFromPositionWithSize (position, size);
}

GridCoordinate
Grid::calculatePositionFromIndex (grid_iter index) const
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& sx = size.getX ();
#if defined (GRID_2D) || defined (GRID_3D)
  const grid_coord& sy = size.getY ();
#if defined (GRID_3D)
  const grid_coord& sz = size.getZ ();
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */

#if defined (GRID_1D)
  grid_coord x = index;
  return GridCoordinate (x);
#else /* GRID_1D */
#if defined (GRID_2D)
  grid_coord x = index / sy;
  index %= sy;
  grid_coord y = index;
  return GridCoordinate (x, y);
#else /* GRID_2D */
#if defined (GRID_3D)
  grid_coord tmp = sy * sz;
  grid_coord x = index / tmp;
  index %= tmp;
  grid_coord y = index / sz;
  index %= sz;
  grid_coord z = index;
  return GridCoordinate (x, y, z);
#endif /* GRID_3D */
#endif /* !GRID_2D */
#endif /* !GRID_1D */
}

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
void
Grid::setFieldPointValue (FieldPointValue* value, const GridCoordinate& position)
{
  ASSERT (isLegitIndex (position));
  ASSERT (value);

  grid_iter coord = calculateIndexFromPosition (position);

  if (gridValues[coord])
  {
    delete gridValues[coord];
  }

  gridValues[coord] = value;
}

void
Grid::setFieldPointValueCurrent (const FieldValue& value,
                                 const GridCoordinate& position)
{
  ASSERT (isLegitIndex (position));

  grid_iter coord = calculateIndexFromPosition (position);

  gridValues[coord]->setCurValue (value);
}

FieldPointValue*
Grid::getFieldPointValue (const GridCoordinate& position)
{
  ASSERT (isLegitIndex (position));

  grid_iter coord = calculateIndexFromPosition (position);
  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
}

FieldPointValue*
Grid::getFieldPointValue (grid_iter coord)
{
  ASSERT (coord >= 0 && coord < size.calculateTotalCoord ());

  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
}

/*#if defined (PARALLEL_GRID)
FieldPointValue*
Grid::getFieldPointValueGlobal (const GridCoordinate& position)
{
  return NULL;
}

FieldPointValue*
Grid::getFieldPointValueGlobal (grid_iter coord)
{
  return NULL;
}
#endif*/

#endif /* GRID_1D || GRID_2D || GRID_3D */

// =================================================================================================
// Parallel features of the grid
#if defined (PARALLEL_GRID)

void
Grid::SendRawBuffer (BufferPosition buffer, int processTo)
{
#if PRINT_MESSAGE
  printf ("Send raw #%d direction %s.\n", processId, BufferPositionNames[buffer]);
#endif
  MPI_Status status;

  FieldValue* rawBuffer = buffersSend[buffer].data ();

#if FULL_VALUES
  int retCode = MPI_Send (rawBuffer, buffersSend[buffer].size (), MPI_DOUBLE,
                          processTo, processId, MPI_COMM_WORLD);
#else /* FULL_VALUES */
  int retCode = MPI_Send (rawBuffer, buffersSend[buffer].size (), MPI_FLOAT,
                          processTo, processId, MPI_COMM_WORLD);
#endif

  ASSERT (retCode == MPI_SUCCESS);
}

void
Grid::ReceiveRawBuffer (BufferPosition buffer, int processFrom)
{
#if PRINT_MESSAGE
  printf ("Receive raw #%d direction %s.\n", processId, BufferPositionNames[buffer]);
#endif
  MPI_Status status;

  FieldValue* rawBuffer = buffersReceive[buffer].data ();

#if FULL_VALUES
  int retCode = MPI_Recv (rawBuffer, buffersReceive[buffer].size (), MPI_DOUBLE,
                          processFrom, processFrom, MPI_COMM_WORLD, &status);
#else /* FULL_VALUES */
  int retCode = MPI_Recv (rawBuffer, buffersReceive[buffer].size (), MPI_FLOAT,
                          processFrom, processFrom, MPI_COMM_WORLD, &status);
#endif

  ASSERT (retCode == MPI_SUCCESS);
}

void
Grid::SendReceiveRawBuffer (BufferPosition bufferSend, int processTo,
                            BufferPosition bufferReceive, int processFrom)
{
#if PRINT_MESSAGE
  printf ("Send/Receive raw #%d directions %s %s.\n", processId, BufferPositionNames[bufferSend],
          BufferPositionNames[bufferReceive]);
#endif
  MPI_Status status;

  FieldValue* rawBufferSend = buffersSend[bufferSend].data ();
  FieldValue* rawBufferReceive = buffersReceive[bufferReceive].data ();

#if FULL_VALUES
  int retCode = MPI_Sendrecv (rawBufferSend, buffersSend[bufferSend].size (), MPI_DOUBLE,
                              processTo, processId,
                              rawBufferReceive, buffersReceive[bufferReceive].size (), MPI_DOUBLE,
                              processFrom, processFrom,
                              MPI_COMM_WORLD, &status);
#else /* FULL_VALUES */
  int retCode = MPI_Sendrecv (rawBufferSend, buffersSend[bufferSend].size (), MPI_FLOAT,
                              processTo, processId,
                              rawBufferReceive, buffersReceive[bufferReceive].size (), MPI_FLOAT,
                              processFrom, processFrom,
                              MPI_COMM_WORLD, &status);
#endif

  ASSERT (retCode == MPI_SUCCESS);
}

#if defined (GRID_1D)
void
Grid::SendReceiveBuffer1D (BufferPosition bufferDirection)
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
#endif

#if defined (GRID_2D)
void
Grid::SendReceiveBuffer2D (BufferPosition bufferDirection)
{
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

      if (processId % sqrtProc == 0)
      {
        doSend = false;
      }
      else if ((processId + 1) % sqrtProc == 0)
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

      if (processId % sqrtProc == 0)
      {
        doReceive = false;
      }
      else if ((processId + 1) % sqrtProc == 0)
      {
        doSend = false;
      }

      break;
    }
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
      processTo = processId + sqrtProc;
      processFrom = processId - sqrtProc;

      if (processId < sqrtProc)
      {
        doReceive = false;
      }
      else if (processId >= totalProcCount - sqrtProc)
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
      processTo = processId - sqrtProc;
      processFrom = processId + sqrtProc;

      if (processId < sqrtProc)
      {
        doSend = false;
      }
      else if (processId >= totalProcCount - sqrtProc)
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
      processTo = processId + sqrtProc - 1;
      processFrom = processId - sqrtProc + 1;;

      if (processId < sqrtProc || (processId + 1) % sqrtProc == 0)
      {
        doReceive = false;
      }
      if (processId >= totalProcCount - sqrtProc || processId % sqrtProc == 0)
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
      processTo = processId - sqrtProc - 1;
      processFrom = processId + sqrtProc + 1;

      if (processId < sqrtProc || processId % sqrtProc == 0)
      {
        doSend = false;
      }
      if (processId >= totalProcCount - sqrtProc || (processId + 1) % sqrtProc == 0)
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
      processTo = processId + sqrtProc + 1;
      processFrom = processId - sqrtProc - 1;

      if (processId < sqrtProc || processId % sqrtProc == 0)
      {
        doReceive = false;
      }
      if (processId >= totalProcCount - sqrtProc || (processId + 1) % sqrtProc == 0)
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
      processTo = processId - sqrtProc + 1;
      processFrom = processId + sqrtProc - 1;

      if (processId < sqrtProc || (processId + 1) % sqrtProc == 0)
      {
        doSend = false;
      }
      if (processId >= totalProcCount - sqrtProc || processId % sqrtProc == 0)
      {
        doReceive = false;
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
#endif

#if defined (GRID_3D)
void
Grid::SendReceiveBuffer3D (BufferPosition bufferDirection)
{

}
#endif /* GRID_3D */

void
Grid::SendReceiveBuffer (BufferPosition bufferDirection)
{
// #if PRINT_MESSAGE
//   printf ("Send/Receive #%d direction %s.\n", processId, BufferPositionNames[bufferDirection]);
// #endif

#if defined (GRID_1D)
  SendReceiveBuffer1D (bufferDirection);
#endif /* GRID_1D */
#if defined (GRID_2D)
  SendReceiveBuffer2D (bufferDirection);
#endif /* GRID_2D */
#if defined (GRID_3D)
  SendReceiveBuffer3D (bufferDirection);
#endif /* GRID_3D */
}

void
Grid::SendReceive ()
{
// #if PRINT_MESSAGE
//   printf ("Send/Receive %d\n", processId);
// #endif

  for (int buf = 0; buf < BUFFER_COUNT; ++buf)
  {
    SendReceiveBuffer ((BufferPosition) buf);
  }
}

void
Grid::Share ()
{
  SendReceive ();

  MPI_Barrier (MPI_COMM_WORLD);
}

#endif /* PARALLEL_GRID */

void
Grid::shiftInTime ()
{
  for (FieldPointValue* current : getValues ())
  {
    current->shiftInTime ();
  }
}
