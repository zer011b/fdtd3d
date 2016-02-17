#include <iostream>

#include "Grid.h"
#include "Assert.h"

extern const char* BufferPositionNames[];

#if defined (PARALLEL_GRID)
Grid::Grid (const GridCoordinate& totSize,
            const GridCoordinate& bufSizeL, const GridCoordinate& bufSizeR,
            const int process, const int totalProc, uint32_t step) :
  totalSize (totSize),
  bufferSizeLeft (bufSizeL),
  bufferSizeRight (bufSizeR),
  processId (process),
  totalProcCount (totalProc),
  timeStep (step)
{
#if defined (ONE_TIME_STEP)
  grid_iter numTimeStepsInBuild = 2;
#endif
#if defined (TWO_TIME_STEPS)
  grid_iter numTimeStepsInBuild = 3;
#endif

  buffersSend.resize (BUFFER_COUNT);
  buffersReceive.resize (BUFFER_COUNT);

  // Call specific constructor.
  ParallelGridConstructor (numTimeStepsInBuild);

  gridValues.resize (size.calculateTotalCoord ());

#if PRINT_MESSAGE
  printf ("New grid for proc: %d (of %d) with raw size: %lu.\n", process, totalProcCount, gridValues.size ());
#endif
}
#else /* PARALLEL_GRID */
Grid::Grid(const GridCoordinate& s, uint32_t step) :
  size (s),
  timeStep (step)
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

/**
 * Send/receive method to be called for all grid types.
 */
void
Grid::SendReceive ()
{
// #if PRINT_MESSAGE
//   printf ("Send/Receive %d\n", processId);
// #endif

  // Go through all directions and send/receive.
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

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
void
Grid::CalculateGridSizeForNode (grid_coord& c1, int nodeGridSize1, grid_coord size1)
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
void
Grid::CalculateGridSizeForNode (grid_coord& c1, int nodeGridSize1, grid_coord size1,
                                grid_coord& c2, int nodeGridSize2, grid_coord size2)
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
Grid::CalculateGridSizeForNode (grid_coord& c1, int nodeGridSize1, grid_coord size1,
                                grid_coord& c2, int nodeGridSize2, grid_coord size2,
                                grid_coord& c3, int nodeGridSize3, grid_coord size3)
#endif
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if ((processId + 1) % nodeGridSize1 != 0)
    c1 = size1 / nodeGridSize1;
  else
    c1 = size1 - (nodeGridSize1 - 1) * (size1 / nodeGridSize1);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (processId < nodeGridSize1 * nodeGridSize2 - nodeGridSize1)
    c2 = size2 / nodeGridSize2;
  else
    c2 = size2 - (nodeGridSize2 - 1) * (size2 / nodeGridSize2);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (processId < nodeGridSize1 * nodeGridSize2 * nodeGridSize3 - nodeGridSize1 * nodeGridSize2)
    c3 = size3 / nodeGridSize3;
  else
    c3 = size3 - (nodeGridSize3 - 1) * (size3 / nodeGridSize3);
#endif
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

void nextTimeStep ()
{

}
