#include <iostream>

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
  buffersSend[LEFT].resize (bufferSizeLeft.calculateTotalCoord () * numTimeStepsInBuild);
  buffersSend[RIGHT].resize (bufferSizeRight.calculateTotalCoord () * numTimeStepsInBuild);

  buffersReceive[LEFT].resize (bufferSizeLeft.calculateTotalCoord () * numTimeStepsInBuild);
  buffersReceive[RIGHT].resize (bufferSizeRight.calculateTotalCoord () * numTimeStepsInBuild);
#endif

#if defined (GRID_2D)

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

/*bool
Grid::isLegitIndexInBuffer (BufferPosition buffer, const GridCoordinate& position) const
{
  switch (buffer)
  {
#if defined (GRID_1D)
    case LEFT:
    {
      return isLegitIndexWithSize (position, bufferSizeLeft);
    }
    case RIGHT:
    {
      return isLegitIndexWithSize (position, bufferSizeRight);
    }
#endif
#if defined (GRID_2D)

#endif
#if defined (GRID_3D)

#endif
  }
}

grid_iter
Grid::calculateIndexFromPositionInBuffer (BufferPosition buffer,
                                          const GridCoordinate& position) const
{
  switch (buffer)
  {
#if defined (GRID_1D)
    case LEFT:
    {
      return calculateIndexFromPositionWithSize (position, bufferSizeLeft);
    }
    case RIGHT:
    {
      return calculateIndexFromPositionWithSize (position, bufferSizeRight);
    }
#endif
#if defined (GRID_2D)

#endif
#if defined (GRID_3D)

#endif
  }
}

void
Grid::setFieldPointValueInBuffer (BufferPosition buffer, FieldPointValue* value,
                                  const GridCoordinate& position)
{
  ASSERT (isLegitIndexInBuffer (buffer, position));
  ASSERT (value);

  grid_iter coord = calculateIndexFromPositionInBuffer (buffer, position);

  if (buffers[buffer][coord])
  {
    delete buffers[buffer][coord];
  }

  buffers[buffer][coord] = value;
}

// Get field point at coordinate in grid.
FieldPointValue* getFieldPointValueInBuffer (BufferPosition buffer, const GridCoordinate& position)
{
  return NULL;
}
FieldPointValue* getFieldPointValueInBuffer (BufferPosition buffer, grid_iter coord)
{
  return NULL;
}
*/
void
Grid::SendRawBuffer (FieldValue* rawBuffer, int processTo, grid_iter size, MPI_Request* request)
{
#if defined (TWO_TIME_STEPS)
  int timeSteps = 3;
#else /* TWO_TIME_STEPS */
#if defined (ONE_TIME_STEP)
  int timeSteps = 2;
#endif /* ONE_TIME_STEP */
#endif /* !TWO_TIME_STEPS */

#if FULL_VALUES
  int retCode = MPI_Isend (rawBuffer, (int) size * timeSteps, MPI_DOUBLE, processTo, processId, MPI_COMM_WORLD, request);
#else /* FULL_VALUES */
  int retCode = MPI_Isend (rawBuffer, (int) size * timeSteps, MPI_FLOAT, processTo, processId, MPI_COMM_WORLD, request);
#endif

  ASSERT (retCode == MPI_SUCCESS);
}

void
Grid::ReceiveRawBuffer (FieldValue* rawBuffer, int processFrom, grid_iter size, MPI_Request* request)
{
#if defined (TWO_TIME_STEPS)
  int timeSteps = 3;
#else /* TWO_TIME_STEPS */
#if defined (ONE_TIME_STEP)
  int timeSteps = 2;
#endif /* ONE_TIME_STEP */
#endif /* !TWO_TIME_STEPS */

#if FULL_VALUES
  int retCode = MPI_Irecv (rawBuffer, (int) size * timeSteps, MPI_DOUBLE, processFrom, processFrom, MPI_COMM_WORLD, request);
#else /* FULL_VALUES */
  int retCode = MPI_Irecv (rawBuffer, (int) size * timeSteps, MPI_FLOAT, processFrom, processFrom, MPI_COMM_WORLD, request);
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

  for (grid_iter pos = pos1;
       pos < pos2;
       ++pos)
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
  SendRawBuffer (rawValues, processTo, total, &requestsSend[buffer]);
}
#endif /* GRID_1D */
#if defined (GRID_2D)
void
Grid::SendBuffer2D (BufferPosition buffer, int processTo)
{

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
  printf ("Send #%d %d.\n", processId, buffer);
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
Grid::ReceiveBuffer1D (BufferPosition buffer, int processFrom)
{
  FieldValue* rawValues = buffersReceive[buffer].data ();

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

  ReceiveRawBuffer (rawValues, processFrom, total, &requestsReceive[buffer]);

  for (grid_iter pos = pos1; pos < pos2; ++pos)
  {
#if defined (TWO_TIME_STEPS)
    FieldPointValue* val = new FieldPointValue (rawValues[index++], rawValues[index++], rawValues[index++]);
#else /* TWO_TIME_STEPS */
#if defined (ONE_TIME_STEP)
    FieldPointValue* val = new FieldPointValue (rawValues[index++], rawValues[index++]);
#else /* ONE_TIME_STEP */
    FieldPointValue* val = new FieldPointValue (rawValues[index++]);
#endif /* !ONE_TIME_STEP */
#endif /* !TWO_TIME_STEPS */

    setFieldPointValue (val, GridCoordinate (pos));
  }
}
#endif /* GRID_1D */
#if defined (GRID_2D)
void
Grid::ReceiveBuffer2D (BufferPosition buffer, int processFrom)
{

}
#endif /* GRID_2D */
#if defined (GRID_3D)
void
Grid::ReceiveBuffer3D (BufferPosition buffer, int processFrom)
{

}
#endif /* GRID_3D */


void
Grid::ReceiveBuffer (BufferPosition buffer, int processFrom)
{
#if PRINT_MESSAGE
  printf ("Receive #%d %d.\n", processId, buffer);
#endif

#if defined (GRID_1D)
  ReceiveBuffer1D (buffer, processFrom);
#endif /* GRID_1D */
#if defined (GRID_2D)
  ReceiveBuffer2D (buffer, processFrom);
#endif /* GRID_2D */
#if defined (GRID_3D)
  ReceiveBuffer3D (buffer, processFrom);
#endif /* GRID_3D */
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
