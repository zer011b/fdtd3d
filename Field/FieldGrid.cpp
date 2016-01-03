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

  requestsSend.resize (BUFFER_COUNT);
  requestsReceive.resize (BUFFER_COUNT);

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
Grid::SendRawBuffer (FieldValue* rawBuffer, int processTo, grid_iter size, MPI_Request* request)
{
#if FULL_VALUES
  int retCode = MPI_Isend (rawBuffer, (int) size , MPI_DOUBLE, processTo, processId, MPI_COMM_WORLD, request);
#else /* FULL_VALUES */
  int retCode = MPI_Isend (rawBuffer, (int) size, MPI_FLOAT, processTo, processId, MPI_COMM_WORLD, request);
#endif

  ASSERT (retCode == MPI_SUCCESS);
}

void
Grid::ReceiveRawBuffer (FieldValue* rawBuffer, int processFrom, grid_iter size, MPI_Request* request)
{
#if FULL_VALUES
  int retCode = MPI_Irecv (rawBuffer, (int) size, MPI_DOUBLE, processFrom, processFrom, MPI_COMM_WORLD, request);
#else /* FULL_VALUES */
  int retCode = MPI_Irecv (rawBuffer, (int) size, MPI_FLOAT, processFrom, processFrom, MPI_COMM_WORLD, request);
#endif

  ASSERT (retCode == MPI_SUCCESS);
}

#if defined (GRID_1D)
void
Grid::SendBuffer1D (BufferPosition buffer, int processTo)
{
  grid_iter pos1 = 0;
  grid_iter pos2 = 0;

  grid_iter index = 0;
  grid_iter total = 0;

  switch (buffer)
  {
    case LEFT:
    {
      total = bufferSizeLeft.calculateTotalCoord ();

      pos1 = total;
      pos2 = 2 * total;

      break;
    }
    case RIGHT:
    {
      total = bufferSizeRight.calculateTotalCoord ();

      pos1 = size.calculateTotalCoord() - 2 * total;
      pos2 = size.calculateTotalCoord () - total;

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  for (grid_iter pos = pos1; pos < pos2; ++pos)
  {
    FieldPointValue* val = getFieldPointValue (pos);
    buffersSend[buffer][index++] = val->getCurValue ();
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    buffersSend[buffer][index++] = val->getPrevValue ();
#if defined (TWO_TIME_STEPS)
    buffersSend[buffer][index++] = val->getPrevPrevValue ();
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
  }

  FieldValue* rawValues = buffersSend[buffer].data ();
  SendRawBuffer (rawValues, processTo, buffersSend[buffer].size (), &requestsSend[buffer]);
}
#endif /* GRID_1D */
#if defined (GRID_2D)
void
Grid::SendBuffer2D (BufferPosition buffer, int processTo)
{
  grid_iter pos1 = 0;
  grid_iter pos2 = 0;
  grid_iter pos3 = 0;
  grid_iter pos4 = 0;

  grid_iter index = 0;

  switch (buffer)
  {
    case LEFT:
    {
      pos1 = bufferSizeLeft.getX ();
      pos2 = 2 * bufferSizeLeft.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();

      break;
    }
    case RIGHT:
    {
      pos1 = size.getX () - 2 * bufferSizeRight.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();

      break;
    }
    case UP:
    {
      pos1 = bufferSizeLeft.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = size.getY () - 2 * bufferSizeRight.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();

      break;
    }
    case DOWN:
    {
      pos1 = bufferSizeLeft.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = 2 * bufferSizeLeft.getY ();

      break;
    }
    case LEFT_UP:
    {
      pos1 = bufferSizeLeft.getX ();
      pos2 = 2 * bufferSizeLeft.getX ();
      pos3 = size.getY () - 2 * bufferSizeRight.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();

      break;
    }
    case LEFT_DOWN:
    {
      pos1 = bufferSizeLeft.getX ();
      pos2 = 2 * bufferSizeLeft.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = 2 * bufferSizeLeft.getY ();

      break;
    }
    case RIGHT_UP:
    {
      pos1 = size.getX () - 2 * bufferSizeRight.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = size.getY () - 2 * bufferSizeRight.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();

      break;
    }
    case RIGHT_DOWN:
    {
      pos1 = size.getX () - 2 * bufferSizeRight.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = 2 * bufferSizeLeft.getY ();

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  for (grid_coord i = pos1; i < pos2; ++i)
  {
    for (grid_coord j = pos3; j < pos4; ++j)
    {
      GridCoordinate pos (i, j);
      FieldPointValue* val = getFieldPointValue (pos);
      buffersSend[buffer][index++] = val->getCurValue ();
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
      buffersSend[buffer][index++] = val->getPrevValue ();
#if defined (TWO_TIME_STEPS)
      buffersSend[buffer][index++] = val->getPrevPrevValue ();
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
    }
  }

  FieldValue* rawValues = buffersSend[buffer].data ();
  SendRawBuffer (rawValues, processTo, buffersSend[buffer].size (), &requestsSend[buffer]);
}
#endif /* GRID_2D */
#if defined (GRID_3D)
void
Grid::SendBuffer3D (BufferPosition buffer, int processTo)
{

}
#endif /* GRID_3D */

void
Grid::SendBuffer (BufferPosition buffer, int processTo)
{
#if PRINT_MESSAGE
  printf ("Send #%d %s.\n", processId, BufferPositionNames[buffer]);
#endif

#if defined (GRID_1D)
  SendBuffer1D (buffer, processTo);
#endif /* GRID_1D */
#if defined (GRID_2D)
  SendBuffer2D (buffer, processTo);
#endif /* GRID_2D */
#if defined (GRID_3D)
  SendBuffer3D (buffer, processTo);
#endif /* GRID_3D */
}

#if defined (GRID_1D)
void
Grid::CopyReceiveBuffer1D (BufferPosition buffer)
{
  grid_iter pos1 = 0;
  grid_iter pos2 = 0;

  grid_iter index = 0;
  grid_iter total = 0;

  switch (buffer)
  {
    case LEFT:
    {
      total = bufferSizeLeft.calculateTotalCoord ();

      pos1 = 0;
      pos2 = total;

      break;
    }
    case RIGHT:
    {
      total = bufferSizeRight.calculateTotalCoord ();

      pos1 = size.calculateTotalCoord () - total;
      pos2 = size.calculateTotalCoord ();

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  for (grid_iter pos = pos1; pos < pos2; ++pos)
  {
#if defined (TWO_TIME_STEPS)
    FieldPointValue* val = new FieldPointValue (buffersReceive[buffer][index++],
                                                buffersReceive[buffer][index++],
                                                buffersReceive[buffer][index++]);
#else /* TWO_TIME_STEPS */
#if defined (ONE_TIME_STEP)
    FieldPointValue* val = new FieldPointValue (buffersReceive[buffer][index++],
                                                buffersReceive[buffer][index++]);
#else /* ONE_TIME_STEP */
    FieldPointValue* val = new FieldPointValue (buffersReceive[buffer][index++]);
#endif /* !ONE_TIME_STEP */
#endif /* !TWO_TIME_STEPS */

    setFieldPointValue (val, GridCoordinate (pos));
  }
}
#endif /* GRID_1D */
#if defined (GRID_2D)
void
Grid::CopyReceiveBuffer2D (BufferPosition buffer)
{
  grid_iter pos1 = 0;
  grid_iter pos2 = 0;
  grid_iter pos3 = 0;
  grid_iter pos4 = 0;

  grid_iter index = 0;

  switch (buffer)
  {
    case LEFT:
    {
      pos1 = 0;
      pos2 = bufferSizeLeft.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();

      break;
    }
    case RIGHT:
    {
      pos1 = size.getX () - bufferSizeRight.getX ();
      pos2 = size.getX ();
      pos3 = bufferSizeLeft.getY ();
      pos4 = size.getY () - bufferSizeRight.getY ();

      break;
    }
    case UP:
    {
      pos1 = bufferSizeLeft.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = size.getY () - bufferSizeRight.getY ();
      pos4 = size.getY ();

      break;
    }
    case DOWN:
    {
      pos1 = bufferSizeLeft.getX ();
      pos2 = size.getX () - bufferSizeRight.getX ();
      pos3 = 0;
      pos4 = bufferSizeLeft.getY ();

      break;
    }
    case LEFT_UP:
    {
      pos1 = 0;
      pos2 = bufferSizeLeft.getX ();
      pos3 = size.getY () - bufferSizeRight.getY ();
      pos4 = size.getY ();

      break;
    }
    case LEFT_DOWN:
    {
      pos1 = 0;
      pos2 = bufferSizeLeft.getX ();
      pos3 = 0;
      pos4 = bufferSizeLeft.getY ();

      break;
    }
    case RIGHT_UP:
    {
      pos1 = size.getX () - bufferSizeRight.getX ();
      pos2 = size.getX ();
      pos3 = size.getY () - bufferSizeRight.getY ();
      pos4 = size.getY ();

      break;
    }
    case RIGHT_DOWN:
    {
      pos1 = size.getX () - bufferSizeRight.getX ();
      pos2 = size.getX ();
      pos3 = 0;
      pos4 = bufferSizeLeft.getY ();

      break;
    }
    default:
    {
      UNREACHABLE;
    }
  }

  for (grid_coord i = pos1; i < pos2; ++i)
  {
    for (grid_coord j = pos3; j < pos4; ++j)
    {
#if defined (TWO_TIME_STEPS)
      FieldPointValue* val = new FieldPointValue (buffersReceive[buffer][index++],
                                                  buffersReceive[buffer][index++],
                                                  buffersReceive[buffer][index++]);
#else /* TWO_TIME_STEPS */
#if defined (ONE_TIME_STEP)
      FieldPointValue* val = new FieldPointValue (buffersReceive[buffer][index++],
                                                  buffersReceive[buffer][index++]);
#else /* ONE_TIME_STEP */
      FieldPointValue* val = new FieldPointValue (buffersReceive[buffer][index++]);
#endif /* !ONE_TIME_STEP */
#endif /* !TWO_TIME_STEPS */

      GridCoordinate pos (i, j);
      setFieldPointValue (val, GridCoordinate (pos));
    }
  }
}
#endif /* GRID_2D */
#if defined (GRID_3D)
void
Grid::CopyReceiveBuffer3D (BufferPosition buffer)
{

}
#endif /* GRID_3D */

void
Grid::ReceiveBuffer (BufferPosition buffer, int processFrom)
{
  FieldValue* rawValues = buffersReceive[buffer].data ();

  ReceiveRawBuffer (rawValues, processFrom, buffersReceive[buffer].size (), &requestsReceive[buffer]);
}

void
Grid::CopyReceiveBuffer (BufferPosition buffer)
{
#if PRINT_MESSAGE
  printf ("Receive #%d %s.\n", processId, BufferPositionNames[buffer]);
#endif

#if defined (GRID_1D)
  CopyReceiveBuffer1D (buffer);
#endif /* GRID_1D */
#if defined (GRID_2D)
  CopyReceiveBuffer2D (buffer);
#endif /* GRID_2D */
#if defined (GRID_3D)
  CopyReceiveBuffer3D (buffer);
#endif /* GRID_3D */
}

void
Grid::Send ()
{
#if defined (GRID_1D)
  if (processId != 0)
  {
    SendBuffer (LEFT, processId - 1);
  }
  if (processId != totalProcCount - 1)
  {
    SendBuffer (RIGHT, processId + 1);
  }
#endif

#if defined (GRID_2D)
  if (processId >= sqrtProc)
  {
    SendBuffer (DOWN, processId - sqrtProc);

    if (processId % sqrtProc != 0)
    {
      SendBuffer (LEFT_DOWN, processId - sqrtProc - 1);
    }
    if ((processId + 1) % sqrtProc != 0)
    {
      SendBuffer (RIGHT_DOWN, processId - sqrtProc + 1);
    }
  }
  if (processId % sqrtProc != 0)
  {
    SendBuffer (LEFT, processId - 1);
  }
  if ((processId + 1) % sqrtProc != 0)
  {
    SendBuffer (RIGHT, processId - 1);
  }
  if (processId < totalProcCount - sqrtProc)
  {
    SendBuffer (UP, processId + sqrtProc);

    if (processId % sqrtProc != 0)
    {
      SendBuffer (LEFT_UP, processId + sqrtProc - 1);
    }
    if ((processId + 1) % sqrtProc != 0)
    {
      SendBuffer (RIGHT_UP, processId + sqrtProc + 1);
    }
  }
#endif

#if defined (GRID_3D)

#endif
}

void
Grid::Receive ()
{
#if defined (GRID_1D)
  if (processId != 0)
  {
    ReceiveBuffer (LEFT, processId - 1);
  }
  if (processId != totalProcCount - 1)
  {
    ReceiveBuffer (RIGHT, processId + 1);
  }
#endif

#if defined (GRID_2D)
  if (processId >= sqrtProc)
  {
    ReceiveBuffer (DOWN, processId - sqrtProc);

    if (processId % sqrtProc != 0)
    {
      ReceiveBuffer (LEFT_DOWN, processId - sqrtProc - 1);
    }
    if ((processId + 1) % sqrtProc != 0)
    {
      ReceiveBuffer (RIGHT_DOWN, processId - sqrtProc + 1);
    }
  }
  if (processId % sqrtProc != 0)
  {
    ReceiveBuffer (LEFT, processId - 1);
  }
  if ((processId + 1) % sqrtProc != 0)
  {
    ReceiveBuffer (RIGHT, processId - 1);
  }
  if (processId < totalProcCount - sqrtProc)
  {
    ReceiveBuffer (UP, processId + sqrtProc);

    if (processId % sqrtProc != 0)
    {
      ReceiveBuffer (LEFT_UP, processId + sqrtProc - 1);
    }
    if ((processId + 1) % sqrtProc != 0)
    {
      ReceiveBuffer (RIGHT_UP, processId + sqrtProc + 1);
    }
  }
#endif

#if defined (GRID_3D)

#endif
}

void
Grid::CopyReceive ()
{
#if defined (GRID_1D)
  if (processId != 0)
  {
    CopyReceiveBuffer (LEFT);
  }
  if (processId != totalProcCount - 1)
  {
    CopyReceiveBuffer (RIGHT);
  }
#endif

#if defined (GRID_2D)
  if (processId >= sqrtProc)
  {
    CopyReceiveBuffer (DOWN);

    if (processId % sqrtProc != 0)
    {
      CopyReceiveBuffer (LEFT_DOWN);
    }
    if ((processId + 1) % sqrtProc != 0)
    {
      CopyReceiveBuffer (RIGHT_DOWN);
    }
  }
  if (processId % sqrtProc != 0)
  {
    CopyReceiveBuffer (LEFT);
  }
  if ((processId + 1) % sqrtProc != 0)
  {
    CopyReceiveBuffer (RIGHT);
  }
  if (processId < totalProcCount - sqrtProc)
  {
    CopyReceiveBuffer (UP);

    if (processId % sqrtProc != 0)
    {
      CopyReceiveBuffer (LEFT_UP);
    }
    if ((processId + 1) % sqrtProc != 0)
    {
      CopyReceiveBuffer (RIGHT_UP);
    }
  }
#endif

#if defined (GRID_3D)

#endif
}

void
Grid::AwaitSendReceive ()
{
  MPI_Status statusSend;
  MPI_Status statusReceive;

  int retCodeSend = 0;
  int retCodeReceive = 0;

#if defined (GRID_1D)
  if (processId != 0)
  {
#if PRINT_MESSAGE
    printf ("Wait for share send #%d %s.\n", processId, BufferPositionNames[LEFT]);
#endif
    retCodeSend = MPI_Wait (&requestsSend[LEFT], &statusSend);
    ASSERT (retCodeSend == MPI_SUCCESS);

#if PRINT_MESSAGE
    printf ("Wait for share receive #%d %s.\n", processId, BufferPositionNames[LEFT]);
#endif
    retCodeReceive = MPI_Wait (&requestsReceive[LEFT], &statusReceive);
    ASSERT (retCodeReceive == MPI_SUCCESS);

#if PRINT_MESSAGE
    printf ("Wait for share OK #%d %s.\n", processId, BufferPositionNames[LEFT]);
#endif
  }
  if (processId != totalProcCount - 1)
  {
#if PRINT_MESSAGE
    printf ("Wait for share send #%d %s.\n", processId, BufferPositionNames[RIGHT]);
#endif
    retCodeSend = MPI_Wait (&requestsSend[RIGHT], &statusSend);
    ASSERT (retCodeSend == MPI_SUCCESS);

#if PRINT_MESSAGE
    printf ("Wait for share receive #%d %s.\n", processId, BufferPositionNames[RIGHT]);
#endif
    retCodeReceive = MPI_Wait (&requestsReceive[RIGHT], &statusReceive);
    ASSERT (retCodeReceive == MPI_SUCCESS);

#if PRINT_MESSAGE
    printf ("Wait for share OK #%d %s.\n", processId, BufferPositionNames[RIGHT]);
#endif
  }
#endif

#if defined (GRID_2D)
  if (processId >= sqrtProc)
  {
#if PRINT_MESSAGE
    printf ("Wait for share send #%d %s.\n", processId, BufferPositionNames[DOWN]);
#endif
    retCodeSend = MPI_Wait (&requestsSend[DOWN], &statusSend);
    ASSERT (retCodeSend == MPI_SUCCESS);

#if PRINT_MESSAGE
    printf ("Wait for share receive #%d %s.\n", processId, BufferPositionNames[DOWN]);
#endif
    retCodeReceive = MPI_Wait (&requestsReceive[DOWN], &statusReceive);
    ASSERT (retCodeReceive == MPI_SUCCESS);

#if PRINT_MESSAGE
    printf ("Wait for share OK #%d %s.\n", processId, BufferPositionNames[DOWN]);
#endif

    if (processId % sqrtProc != 0)
    {
#if PRINT_MESSAGE
      printf ("Wait for share send #%d %s.\n", processId, BufferPositionNames[LEFT_DOWN]);
#endif
      retCodeSend = MPI_Wait (&requestsSend[LEFT_DOWN], &statusSend);
      ASSERT (retCodeSend == MPI_SUCCESS);

#if PRINT_MESSAGE
      printf ("Wait for share receive #%d %s.\n", processId, BufferPositionNames[LEFT_DOWN]);
#endif
      retCodeReceive = MPI_Wait (&requestsReceive[LEFT_DOWN], &statusReceive);
      ASSERT (retCodeReceive == MPI_SUCCESS);

#if PRINT_MESSAGE
      printf ("Wait for share OK #%d %s.\n", processId, BufferPositionNames[LEFT_DOWN]);
#endif
    }
    if ((processId + 1) % sqrtProc != 0)
    {
#if PRINT_MESSAGE
      printf ("Wait for share send #%d %s.\n", processId, BufferPositionNames[RIGHT_DOWN]);
#endif
      retCodeSend = MPI_Wait (&requestsSend[RIGHT_DOWN], &statusSend);
      ASSERT (retCodeSend == MPI_SUCCESS);

#if PRINT_MESSAGE
      printf ("Wait for share receive #%d %s.\n", processId, BufferPositionNames[RIGHT_DOWN]);
#endif
      retCodeReceive = MPI_Wait (&requestsReceive[RIGHT_DOWN], &statusReceive);
      ASSERT (retCodeReceive == MPI_SUCCESS);

#if PRINT_MESSAGE
      printf ("Wait for share OK #%d %s.\n", processId, BufferPositionNames[RIGHT_DOWN]);
#endif
    }
  }
  if (processId % sqrtProc != 0)
  {
#if PRINT_MESSAGE
    printf ("Wait for share send #%d %s.\n", processId, BufferPositionNames[LEFT]);
#endif
    retCodeSend = MPI_Wait (&requestsSend[LEFT], &statusSend);
    ASSERT (retCodeSend == MPI_SUCCESS);

#if PRINT_MESSAGE
    printf ("Wait for share receive #%d %s.\n", processId, BufferPositionNames[LEFT]);
#endif
    retCodeReceive = MPI_Wait (&requestsReceive[LEFT], &statusReceive);
    ASSERT (retCodeReceive == MPI_SUCCESS);

#if PRINT_MESSAGE
    printf ("Wait for share OK #%d %s.\n", processId, BufferPositionNames[LEFT]);
#endif
  }
  if ((processId + 1) % sqrtProc != 0)
  {
#if PRINT_MESSAGE
    printf ("Wait for share send #%d %s.\n", processId, BufferPositionNames[RIGHT]);
#endif
    retCodeSend = MPI_Wait (&requestsSend[RIGHT], &statusSend);
    ASSERT (retCodeSend == MPI_SUCCESS);

#if PRINT_MESSAGE
    printf ("Wait for share receive #%d %s.\n", processId, BufferPositionNames[RIGHT]);
#endif
    retCodeReceive = MPI_Wait (&requestsReceive[RIGHT], &statusReceive);
    ASSERT (retCodeReceive == MPI_SUCCESS);

#if PRINT_MESSAGE
    printf ("Wait for share OK #%d %s.\n", processId, BufferPositionNames[RIGHT]);
#endif
  }
  if (processId < totalProcCount - sqrtProc)
  {
#if PRINT_MESSAGE
    printf ("Wait for share send #%d %s.\n", processId, BufferPositionNames[UP]);
#endif
    retCodeSend = MPI_Wait (&requestsSend[UP], &statusSend);
    ASSERT (retCodeSend == MPI_SUCCESS);

#if PRINT_MESSAGE
    printf ("Wait for share receive #%d %s.\n", processId, BufferPositionNames[UP]);
#endif
    retCodeReceive = MPI_Wait (&requestsReceive[UP], &statusReceive);
    ASSERT (retCodeReceive == MPI_SUCCESS);

#if PRINT_MESSAGE
    printf ("Wait for share OK #%d %s.\n", processId, BufferPositionNames[UP]);
#endif

    if (processId % sqrtProc != 0)
    {
#if PRINT_MESSAGE
      printf ("Wait for share send #%d %s.\n", processId, BufferPositionNames[LEFT_UP]);
#endif
      retCodeSend = MPI_Wait (&requestsSend[LEFT_UP], &statusSend);
      ASSERT (retCodeSend == MPI_SUCCESS);

#if PRINT_MESSAGE
      printf ("Wait for share receive #%d %s.\n", processId, BufferPositionNames[LEFT_UP]);
#endif
      retCodeReceive = MPI_Wait (&requestsReceive[LEFT_UP], &statusReceive);
      ASSERT (retCodeReceive == MPI_SUCCESS);

#if PRINT_MESSAGE
      printf ("Wait for share OK #%d %s.\n", processId, BufferPositionNames[LEFT_UP]);
#endif
    }
    if ((processId + 1) % sqrtProc != 0)
    {
#if PRINT_MESSAGE
      printf ("Wait for share send #%d %s.\n", processId, BufferPositionNames[RIGHT_UP]);
#endif
      retCodeSend = MPI_Wait (&requestsSend[RIGHT_UP], &statusSend);
      ASSERT (retCodeSend == MPI_SUCCESS);

#if PRINT_MESSAGE
      printf ("Wait for share receive #%d %s.\n", processId, BufferPositionNames[RIGHT_UP]);
#endif
      retCodeReceive = MPI_Wait (&requestsReceive[RIGHT_UP], &statusReceive);
      ASSERT (retCodeReceive == MPI_SUCCESS);

#if PRINT_MESSAGE
      printf ("Wait for share OK #%d %s.\n", processId, BufferPositionNames[RIGHT_UP]);
#endif
    }
  }
#endif

#if defined (GRID_3D)

#endif
}

void
Grid::Share ()
{
  Send ();

  Receive ();

  //MPI_Barrier (MPI_COMM_WORLD);

  AwaitSendReceive ();

  CopyReceive ();
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
