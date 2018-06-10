#include "ParallelGrid.h"

#ifdef PARALLEL_GRID

#if PRINT_MESSAGE
/**
 * Names of buffers of parallel grid for debug purposes.
 */
const char* BufferPositionNames[] =
{
#define FUNCTION(X) #X,
#include "BufferPosition.inc.h"
}; /* BufferPositionNames */
#endif /* PRINT_MESSAGE */

ParallelGridCore *ParallelGrid::parallelGridCore = NULLPTR;

#ifdef DEBUG_INFO

#ifdef GRID_1D
#define COORD_TYPES ,ct1
#endif /* GRID_1D */

#ifdef GRID_2D
#define COORD_TYPES ,ct1,ct2
#endif /* GRID_2D */

#ifdef GRID_3D
#define COORD_TYPES ,ct1,ct2,ct3
#endif /* GRID_3D */

#else /* DEBUG_INFO */
#define COORD_TYPES
#endif /* !DEBUG_INFO */

/**
 * Parallel grid constructor
 */
ParallelGrid::ParallelGrid (const ParallelGridCoordinate &totSize, /**< total size of grid */
                            const ParallelGridCoordinate &bufSize, /**< buffer size */
                            time_step step, /**< start time step */
                            ParallelGridCoordinate curSize,  /**< size of grid for current node, received from layout */
                            const char * name) /**< name of grid */
  : ParallelGridBase (step, name)
#ifdef DEBUG_INFO
  , ct1 (CoordinateType::NONE)
  , ct2 (CoordinateType::NONE)
  , ct3 (CoordinateType::NONE)
#endif /* DEBUG_INFO */
  , totalSize (totSize)
  , shareStep (0)
  , bufferSize (ParallelGridCoordinate ())
  , currentSize (curSize)
{
#ifdef DEBUG_INFO
  /*
   * Initialize coordinate types
   */
  ct1 = totSize.getType1 ();

#if defined (GRID_2D) || defined (GRID_3D)
  ct2 = totSize.getType2 ();
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
  ct3 = totSize.getType3 ();
#endif /* GRID_3D */
#endif /* DEBUG_INFO */

  /*
   * Check that buffer size is equal for all coordinate axes
   */
  ASSERT (bufSize.get1 () != 0);

#if defined (GRID_2D) || defined (GRID_3D)
  ASSERT (bufSize.get1 () == bufSize.get2 ());
#endif /* GRID_2D || GRID_3D */

#ifdef GRID_3D
  ASSERT (bufSize.get1 () == bufSize.get3 ());
#endif /* GRID_3D */

#ifdef DYNAMIC_GRID
  /*
   * TODO: support other buffer sizes
   */
  ASSERT (bufSize.get1 () == 1);
#endif /* DYNAMIC_GRID */

  /*
   * Set buffer size with virtual topology in mind
   */
#ifdef GRID_1D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  bufferSize = bufSize;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */
#endif /* GRID_1D */

#ifdef GRID_2D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  bufferSize = ParallelGridCoordinate (bufSize.get1 (), 0 COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  bufferSize = ParallelGridCoordinate (0, bufSize.get2 () COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  bufferSize = bufSize;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */
#endif /* GRID_2D */

#ifdef GRID_3D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  bufferSize = ParallelGridCoordinate (bufSize.get1 (), 0, 0 COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  bufferSize = ParallelGridCoordinate (0, bufSize.get2 (), 0 COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  bufferSize = ParallelGridCoordinate (0, 0, bufSize.get3 () COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  bufferSize = ParallelGridCoordinate (bufSize.get1 (), bufSize.get2 (), 0 COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  bufferSize = ParallelGridCoordinate (0, bufSize.get2 (), bufSize.get3 () COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  bufferSize = ParallelGridCoordinate (bufSize.get1 (), 0, bufSize.get3 () COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  bufferSize = bufSize;
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
#endif /* GRID_3D */

  /*
   * Construct parallel grid internals
   */
  ParallelGridConstructor ();

  gridValues.resize (size.calculateTotalCoord ());

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "New grid '%s' for proc: %d (of %d) with raw size: %lu.\n",
           gridName.data (),
           parallelGridCore->getProcessId (),
           parallelGridCore->getTotalProcCount (),
           gridValues.size ());

  gatherStartPosition ();
} /* ParallelGrid::ParallelGrid */

/**
 * Gather start position for all computational nodes
 */
void ParallelGrid::gatherStartPosition ()
{
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP,
           "Gather start position for '%s' for proc: %d (of %d).\n",
           gridName.data (),
           parallelGridCore->getProcessId (),
           parallelGridCore->getTotalProcCount ());

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_coord startx = 0;
  grid_coord endx = 0;
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
  grid_coord starty = 0;
  grid_coord endy = 0;
#endif /* GRID_2D || GRID_3D */
#ifdef GRID_3D
  grid_coord startz = 0;
  grid_coord endz = 0;
#endif /* GRID_3D */

  for (int process = 0; process < ParallelGrid::getParallelCore ()->getTotalProcCount (); ++process)
  {
    /*
     * Receive start position from previous nodes
     */

    bool hasReceivedX = false;
    bool hasReceivedY = false;
    bool hasReceivedZ = false;

    MPI_Status status;
    int retCode;

    int state = 1;
#ifdef DYNAMIC_GRID
    state = ParallelGrid::getParallelCore ()->getNodeState ()[ParallelGrid::getParallelCore ()->getProcessId ()];
#endif /* DYNAMIC_GRID */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
  defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

    if (process == ParallelGrid::getParallelCore ()->getProcessId ()
        && state
        && ParallelGrid::getParallelCore ()->getNodeForDirection (LEFT) != PID_NONE)
    {
      retCode = MPI_Recv (&startx,
                          1,
                          MPI_COORD,
                          ParallelGrid::getParallelCore ()->getNodeForDirection (LEFT),
                          0,
                          ParallelGrid::getParallelCore ()->getCommunicator (),
                          &status);
      ASSERT (retCode == MPI_SUCCESS);

      hasReceivedX = true;
    }
    else if (state
             && ParallelGrid::getParallelCore ()->getNodeForDirection (RIGHT) == process)
    {
      retCode = MPI_Send (&endx,
                          1,
                          MPI_COORD,
                          ParallelGrid::getParallelCore ()->getNodeForDirection (RIGHT),
                          0,
                          ParallelGrid::getParallelCore ()->getCommunicator ());
      ASSERT (retCode == MPI_SUCCESS);
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
  defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

    if (process == ParallelGrid::getParallelCore ()->getProcessId ()
        && state
        && ParallelGrid::getParallelCore ()->getNodeForDirection (DOWN) != PID_NONE)
    {
      retCode = MPI_Recv (&starty,
                          1,
                          MPI_COORD,
                          ParallelGrid::getParallelCore ()->getNodeForDirection (DOWN),
                          1,
                          ParallelGrid::getParallelCore ()->getCommunicator (),
                          &status);
      ASSERT (retCode == MPI_SUCCESS);

      hasReceivedY = true;
    }
    else if (state
             && ParallelGrid::getParallelCore ()->getNodeForDirection (UP) == process)
    {
      retCode = MPI_Send (&endy,
                          1,
                          MPI_COORD,
                          ParallelGrid::getParallelCore ()->getNodeForDirection (UP),
                          1,
                          ParallelGrid::getParallelCore ()->getCommunicator ());
      ASSERT (retCode == MPI_SUCCESS);
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
  defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

    if (process == ParallelGrid::getParallelCore ()->getProcessId ()
        && state
        && ParallelGrid::getParallelCore ()->getNodeForDirection (BACK) != PID_NONE)
    {
      retCode = MPI_Recv (&startz,
                          1,
                          MPI_COORD,
                          ParallelGrid::getParallelCore ()->getNodeForDirection (BACK),
                          2,
                          ParallelGrid::getParallelCore ()->getCommunicator (),
                          &status);
      ASSERT (retCode == MPI_SUCCESS);

      hasReceivedZ = true;
    }
    else if (state
             && ParallelGrid::getParallelCore ()->getNodeForDirection (FRONT) == process)
    {
      retCode = MPI_Send (&endz,
                          1,
                          MPI_COORD,
                          ParallelGrid::getParallelCore ()->getNodeForDirection (FRONT),
                          2,
                          ParallelGrid::getParallelCore ()->getCommunicator ());
      ASSERT (retCode == MPI_SUCCESS);
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

    if (process == ParallelGrid::getParallelCore ()->getProcessId ()
        && state)
    {
#ifndef DYNAMIC_GRID
      // For dynamic grid only one node can be left, so nothing is received
      // TODO: add check that a single node is left for dynamic grid
      ASSERT (hasReceivedX || hasReceivedY || hasReceivedZ || process == 0);
#endif /* !DYNAMIC_GRID */

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
      endx = startx + currentSize.get1 ();
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
      endy = starty + currentSize.get2 ();
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
      endz = startz + currentSize.get3 ();
#endif /* GRID_3D */

#ifdef GRID_1D
      ParallelGridCoordinate startPosition = GridCoordinate1D (startx COORD_TYPES);
#endif /* GRID_1D */
#ifdef GRID_2D
      ParallelGridCoordinate startPosition = GridCoordinate2D (startx, starty COORD_TYPES);
#endif /* GRID_2D */
#ifdef GRID_3D
      ParallelGridCoordinate startPosition = GridCoordinate3D (startx, starty, startz COORD_TYPES);
#endif /* GRID_3D */

      initializeStartPosition (startPosition);
    }

    if (process == ParallelGrid::getParallelCore ()->getProcessId ()
        && state)
    {
#ifdef GRID_1D
      DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Start pos (" COORD_MOD ") for grid '%s' for proc %d (of %d)\n",
               startx,
               gridName.data (),
               parallelGridCore->getProcessId (),
               parallelGridCore->getTotalProcCount ());
#endif /* GRID_1D */
#ifdef GRID_2D
      DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Start pos (" COORD_MOD ", " COORD_MOD ") for grid '%s' for proc %d (of %d)\n",
               startx,
               starty,
               gridName.data (),
               parallelGridCore->getProcessId (),
               parallelGridCore->getTotalProcCount ());
#endif /* GRID_2D */
#ifdef GRID_3D
      DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Start pos (" COORD_MOD ", " COORD_MOD ", " COORD_MOD ") for grid '%s' for proc %d (of %d)\n",
               startx,
               starty,
               startz,
               gridName.data (),
               parallelGridCore->getProcessId (),
               parallelGridCore->getTotalProcCount ());
#endif /* GRID_3D */
    }

    MPI_Barrier (ParallelGrid::getParallelCore ()->getCommunicator ());
  }
} /* ParallelGrid::gatherStartPosition */

/**
 * Initialize absolute start position of chunk for current node
 */
void ParallelGrid::initializeStartPosition (ParallelGridCoordinate chunkStartPos) /**< start position of chunk, that is
                                                                                   *   assigned to current process,
                                                                                   *   (except buffers) */
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
  defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  grid_coord posX;
  if (parallelGridCore->getNodeForDirection (LEFT) == PID_NONE)
  {
    posX = 0;
  }
  else
  {
    posX = chunkStartPos.get1 () - bufferSize.get1 ();
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
        PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
  defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  grid_coord posY;
  if (parallelGridCore->getNodeForDirection (DOWN) == PID_NONE)
  {
    posY = 0;
  }
  else
  {
    posY = chunkStartPos.get2 () - bufferSize.get2 ();
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
        PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
  defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  grid_coord posZ;
  if (parallelGridCore->getNodeForDirection (BACK) == PID_NONE)
  {
    posZ = 0;
  }
  else
  {
    posZ = chunkStartPos.get3 () - bufferSize.get3 ();
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
        PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#ifdef GRID_1D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */
#endif /* GRID_1D */

#ifdef GRID_2D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX, 0 COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  posStart = ParallelGridCoordinate (0, posY COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  posStart = ParallelGridCoordinate (posX, posY COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */
#endif /* GRID_2D */

#ifdef GRID_3D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX, 0, 0 COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  posStart = ParallelGridCoordinate (0, posY, 0 COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  posStart = ParallelGridCoordinate (0, 0, posZ COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  posStart = ParallelGridCoordinate (posX, posY, 0 COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  posStart = ParallelGridCoordinate (0, posY, posZ COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  posStart = ParallelGridCoordinate (posX, 0, posZ COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  posStart = ParallelGridCoordinate (posX, posY, posZ COORD_TYPES);
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
#endif /* GRID_3D */
} /* ParallelGrid::initializeStartPosition */

/**
 * Initialize start and end cooridnates for send/receive for all directions
 */
void
ParallelGrid::SendReceiveCoordinatesInit ()
{
  sendStart.resize (BUFFER_COUNT);
  sendEnd.resize (BUFFER_COUNT);
  recvStart.resize (BUFFER_COUNT);
  recvEnd.resize (BUFFER_COUNT);

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  SendReceiveCoordinatesInit1D_X ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  SendReceiveCoordinatesInit1D_Y ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  SendReceiveCoordinatesInit1D_Z ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  SendReceiveCoordinatesInit2D_XY ();
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  SendReceiveCoordinatesInit2D_YZ ();
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  SendReceiveCoordinatesInit2D_XZ ();
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  SendReceiveCoordinatesInit3D_XYZ ();
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
} /* ParallelGrid::SendReceiveCoordinatesInit */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

/**
 * Initialize start and end cooridnates for send/receive for all directions for 1D X mode
 */
void
ParallelGrid::SendReceiveCoordinatesInit1D_X ()
{
  grid_coord left_coord, right_coord;
  grid_coord down_coord, up_coord;
  grid_coord back_coord, front_coord;

  initBufferOffsets (left_coord, right_coord, down_coord, up_coord, back_coord, front_coord);

  sendStart[LEFT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[LEFT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    2 * left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[LEFT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[LEFT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.get1 ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendStart[RIGHT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - 2 * right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[RIGHT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[RIGHT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    0
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[RIGHT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );
} /* ParallelGrid::SendReceiveCoordinatesInit1D_X */
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

/**
 * Initialize start and end cooridnates for send/receive for all directions for 1D Y mode
 */
void
ParallelGrid::SendReceiveCoordinatesInit1D_Y ()
{
  grid_coord left_coord, right_coord;
  grid_coord down_coord, up_coord;
  grid_coord back_coord, front_coord;

  initBufferOffsets (left_coord, right_coord, down_coord, up_coord, back_coord, front_coord);

  sendStart[DOWN] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[DOWN] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , 2 * down_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[DOWN] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[DOWN] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.get2 ()
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendStart[UP] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.get2 () - 2 * up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[UP] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[UP] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , 0
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[UP] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );
} /* ParallelGrid::SendReceiveCoordinatesInit1D_Y */
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

/**
 * Initialize start and end cooridnates for send/receive for all directions for 1D Z mode
 */
void
ParallelGrid::SendReceiveCoordinatesInit1D_Z ()
{
  grid_coord left_coord, right_coord;
  grid_coord down_coord, up_coord;
  grid_coord back_coord, front_coord;

  initBufferOffsets (left_coord, right_coord, down_coord, up_coord, back_coord, front_coord);

  sendStart[BACK] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[BACK] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , 2 * back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[BACK] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[BACK] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendStart[FRONT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , size.get3 () - 2 * front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[FRONT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[FRONT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , 0
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[FRONT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
    COORD_TYPES
  );
} /* ParallelGrid::SendReceiveCoordinatesInit1D_Z */
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

/**
 * Initialize start and end cooridnates for send/receive for all directions for 2D XY mode
 */
void
ParallelGrid::SendReceiveCoordinatesInit2D_XY ()
{
  grid_coord left_coord, right_coord;
  grid_coord down_coord, up_coord;
  grid_coord back_coord, front_coord;

  initBufferOffsets (left_coord, right_coord, down_coord, up_coord, back_coord, front_coord);

  sendStart[LEFT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[LEFT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * left_coord
    , 2 * down_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[LEFT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[LEFT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 ()
    , size.get2 ()
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendStart[LEFT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.get2 () - 2 * up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[LEFT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * left_coord
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[LEFT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , 0
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[LEFT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 ()
    , down_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendStart[RIGHT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - 2 * right_coord
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[RIGHT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , 2 * down_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[RIGHT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[RIGHT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.get2 ()
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendStart[RIGHT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - 2 * right_coord
    , size.get2 () - 2 * up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[RIGHT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[RIGHT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , 0
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[RIGHT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , down_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );
} /* ParallelGrid::SendReceiveCoordinatesInit2D_XY */
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

/**
 * Initialize start and end cooridnates for send/receive for all directions for 2D YZ mode
 */
void
ParallelGrid::SendReceiveCoordinatesInit2D_YZ ()
{
  grid_coord left_coord, right_coord;
  grid_coord down_coord, up_coord;
  grid_coord back_coord, front_coord;

  initBufferOffsets (left_coord, right_coord, down_coord, up_coord, back_coord, front_coord);

  sendStart[DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , 2 * down_coord
#if defined (GRID_3D)
    , 2 * back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , size.get2 ()
#if defined (GRID_3D)
    , size.get3 ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendStart[DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , down_coord
#if defined (GRID_3D)
    , size.get3 () - 2 * front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , 2 * down_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , 0
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , size.get2 ()
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendStart[UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.get2 () - 2 * up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , 2 * back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , 0
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , down_coord
#if defined (GRID_3D)
    , size.get3 ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendStart[UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.get2 () - 2 * up_coord
#if defined (GRID_3D)
    , size.get3 () - 2 * front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , 0
#if defined (GRID_3D)
    , 0
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );
} /* ParallelGrid::SendReceiveCoordinatesInit2D_YZ */
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

/**
 * Initialize start and end cooridnates for send/receive for all directions for 2D XZ mode
 */
void
ParallelGrid::SendReceiveCoordinatesInit2D_XZ ()
{
  grid_coord left_coord, right_coord;
  grid_coord down_coord, up_coord;
  grid_coord back_coord, front_coord;

  initBufferOffsets (left_coord, right_coord, down_coord, up_coord, back_coord, front_coord);

  sendStart[LEFT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[LEFT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * left_coord
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , 2 * back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[LEFT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , down_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[LEFT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 ()
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendStart[LEFT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , down_coord
#if defined (GRID_3D)
    , size.get3 () - 2 * front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[LEFT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * left_coord
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[LEFT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , down_coord
#if defined (GRID_3D)
    , 0
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[LEFT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 ()
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendStart[RIGHT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - 2 * right_coord
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[RIGHT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , 2 * back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[RIGHT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , down_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[RIGHT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendStart[RIGHT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - 2 * right_coord
    , down_coord
#if defined (GRID_3D)
    , size.get3 () - 2 * front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  sendEnd[RIGHT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.get1 () - right_coord
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , size.get3 () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvStart[RIGHT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , down_coord
#if defined (GRID_3D)
    , 0
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );

  recvEnd[RIGHT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.get2 () - up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
    COORD_TYPES
  );
} /* ParallelGrid::SendReceiveCoordinatesInit2D_XZ */
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

/**
 * Initialize start and end cooridnates for send/receive for all directions for 3D XYZ mode
 */
void
ParallelGrid::SendReceiveCoordinatesInit3D_XYZ ()
{
  grid_coord left_coord, right_coord;
  grid_coord down_coord, up_coord;
  grid_coord back_coord, front_coord;

  initBufferOffsets (left_coord, right_coord, down_coord, up_coord, back_coord, front_coord);

  sendStart[LEFT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , down_coord
    , back_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  sendEnd[LEFT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    2 * left_coord
    , 2 * down_coord
    , 2 * back_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvStart[LEFT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 () - right_coord
    , size.get2 () - up_coord
    , size.get3 () - front_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvEnd[LEFT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 ()
    , size.get2 ()
    , size.get3 ()
#endif /* GRID_3D */
    COORD_TYPES
  );

  sendStart[LEFT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , down_coord
    , size.get3 () - 2 * front_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  sendEnd[LEFT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    2 * left_coord
    , 2 * down_coord
    , size.get3 () - front_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvStart[LEFT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 () - right_coord
    , size.get2 () - up_coord
    , 0
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvEnd[LEFT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 ()
    , size.get2 ()
    , back_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  sendStart[LEFT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , size.get2 () - 2 * up_coord
    , back_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  sendEnd[LEFT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    2 * left_coord
    , size.get2 () - up_coord
    , 2 * back_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvStart[LEFT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 () - right_coord
    , 0
    , size.get3 () - front_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvEnd[LEFT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 ()
    , down_coord
    , size.get3 ()
#endif /* GRID_3D */
    COORD_TYPES
  );

  sendStart[LEFT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , size.get2 () - 2 * up_coord
    , size.get3 () - 2 * front_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  sendEnd[LEFT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    2 * left_coord
    , size.get2 () - up_coord
    , size.get3 () - front_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvStart[LEFT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 () - right_coord
    , 0
    , 0
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvEnd[LEFT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 ()
    , down_coord
    , back_coord
#endif /* GRID_3D */
    COORD_TYPES
  );


  sendStart[RIGHT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 () - 2 * right_coord
    , down_coord
    , back_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  sendEnd[RIGHT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 () - right_coord
    , 2 * down_coord
    , 2 * back_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvStart[RIGHT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    0
    , size.get2 () - up_coord
    , size.get3 () - front_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvEnd[RIGHT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , size.get2 ()
    , size.get3 ()
#endif /* GRID_3D */
    COORD_TYPES
  );

  sendStart[RIGHT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 () - 2 * right_coord
    , down_coord
    , size.get3 () - 2 * front_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  sendEnd[RIGHT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 () - right_coord
    , 2 * down_coord
    , size.get3 () - front_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvStart[RIGHT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    0
    , size.get2 () - up_coord
    , 0
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvEnd[RIGHT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , size.get2 ()
    , back_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  sendStart[RIGHT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 () - 2 * right_coord
    , size.get2 () - 2 * up_coord
    , back_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  sendEnd[RIGHT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 () - right_coord
    , size.get2 () - up_coord
    , 2 * back_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvStart[RIGHT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    0
    , 0
    , size.get3 () - front_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvEnd[RIGHT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , down_coord
    , size.get3 ()
#endif /* GRID_3D */
    COORD_TYPES
  );

  sendStart[RIGHT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 () - 2 * right_coord
    , size.get2 () - 2 * up_coord
    , size.get3 () - 2 * front_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  sendEnd[RIGHT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.get1 () - right_coord
    , size.get2 () - up_coord
    , size.get3 () - front_coord
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvStart[RIGHT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    0
    , 0
    , 0
#endif /* GRID_3D */
    COORD_TYPES
  );

  recvEnd[RIGHT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , down_coord
    , back_coord
#endif /* GRID_3D */
    COORD_TYPES
  );
} /* ParallelGrid::SendReceiveCoordinatesInit3D_XYZ */
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

/**
 * Send raw buffer with data
 */
void
ParallelGrid::SendRawBuffer (BufferPosition buffer, /**< buffer's position to send (direction) */
                             int processTo) /**< id of computational node to send data to */
{
  DPRINTF (LOG_LEVEL_FULL, "\tSend RAW. PID=#%d. Direction TO=%s, size=%lu.\n",
           parallelGridCore->getProcessId (),
           BufferPositionNames[buffer],
           buffersSend[buffer].size ());

  FieldValue* rawBuffer = buffersSend[buffer].data ();

  MPI_Datatype datatype;

#ifdef FLOAT_VALUES
#ifdef COMPLEX_FIELD_VALUES
  datatype = MPI_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
  datatype = MPI_FLOAT;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* FLOAT_VALUES */

#ifdef DOUBLE_VALUES
#ifdef COMPLEX_FIELD_VALUES
  datatype = MPI_DOUBLE_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
  datatype = MPI_DOUBLE;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* DOUBLE_VALUES */

#ifdef LONG_DOUBLE_VALUES
#ifdef COMPLEX_FIELD_VALUES
  datatype = MPI_LONG_DOUBLE_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
  datatype = MPI_LONG_DOUBLE;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* LONG_DOUBLE_VALUES */

  int retCode = MPI_Send (rawBuffer,
                          buffersSend[buffer].size (),
                          datatype,
                          processTo,
                          parallelGridCore->getProcessId (),
                          ParallelGrid::getParallelCore ()->getCommunicator ());

  ASSERT (retCode == MPI_SUCCESS);
} /* ParallelGrid::SendRawBuffer */

/**
 * Receive raw buffer with data
 */
void
ParallelGrid::ReceiveRawBuffer (BufferPosition buffer, /**< buffer's position to receive (direction) */
                                int processFrom) /**< id of computational node to receive data from */
{
  DPRINTF (LOG_LEVEL_FULL, "\t\tReceive RAW. PID=#%d. Direction FROM=%s, size=%lu.\n",
           parallelGridCore->getProcessId (),
           BufferPositionNames[buffer],
           buffersReceive[buffer].size ());

  MPI_Status status;

  FieldValue* rawBuffer = buffersReceive[buffer].data ();

  MPI_Datatype datatype;

#ifdef FLOAT_VALUES
#ifdef COMPLEX_FIELD_VALUES
  datatype = MPI_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
  datatype = MPI_FLOAT;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* FLOAT_VALUES */

#ifdef DOUBLE_VALUES
#ifdef COMPLEX_FIELD_VALUES
  datatype = MPI_DOUBLE_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
  datatype = MPI_DOUBLE;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* DOUBLE_VALUES */

#ifdef LONG_DOUBLE_VALUES
#ifdef COMPLEX_FIELD_VALUES
  datatype = MPI_LONG_DOUBLE_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
  datatype = MPI_LONG_DOUBLE;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* LONG_DOUBLE_VALUES */

  int retCode = MPI_Recv (rawBuffer,
                          buffersReceive[buffer].size (),
                          datatype,
                          processFrom,
                          processFrom,
                          ParallelGrid::getParallelCore ()->getCommunicator (),
                          &status);

  ASSERT (retCode == MPI_SUCCESS);
} /* ParallelGrid::ReceiveRawBuffer */

/**
 * Send and receive raw buffers with data
 */
void
ParallelGrid::SendReceiveRawBuffer (BufferPosition bufferSend, /**< buffer's position to send (direction) */
                                    int processTo, /**< id of computational node to send data to */
                                    BufferPosition bufferReceive, /**< buffer's position to receive (direction) */
                                    int processFrom) /**< id of computational node to receive data from */
{
  DPRINTF (LOG_LEVEL_FULL, "\t\tSend/Receive RAW. PID=#%d. Directions TO=%s FROM=%s. Size TO=%lu FROM=%lu.\n",
           parallelGridCore->getProcessId (),
           BufferPositionNames[bufferSend],
           BufferPositionNames[bufferReceive],
           buffersSend[bufferSend].size (),
           buffersReceive[bufferReceive].size ());

  MPI_Status status;

  FieldValue* rawBufferSend = buffersSend[bufferSend].data ();
  FieldValue* rawBufferReceive = buffersReceive[bufferReceive].data ();

  MPI_Datatype datatype;

#ifdef FLOAT_VALUES
#ifdef COMPLEX_FIELD_VALUES
  datatype = MPI_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
  datatype = MPI_FLOAT;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* FLOAT_VALUES */

#ifdef DOUBLE_VALUES
#ifdef COMPLEX_FIELD_VALUES
  datatype = MPI_DOUBLE_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
  datatype = MPI_DOUBLE;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* DOUBLE_VALUES */

#ifdef LONG_DOUBLE_VALUES
#ifdef COMPLEX_FIELD_VALUES
  datatype = MPI_LONG_DOUBLE_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
  datatype = MPI_LONG_DOUBLE;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* LONG_DOUBLE_VALUES */

  int retCode = MPI_Sendrecv (rawBufferSend,
                              buffersSend[bufferSend].size (),
                              datatype,
                              processTo,
                              parallelGridCore->getProcessId (),
                              rawBufferReceive,
                              buffersReceive[bufferReceive].size (),
                              datatype,
                              processFrom,
                              processFrom,
                              ParallelGrid::getParallelCore ()->getCommunicator (),
                              &status);

  ASSERT (retCode == MPI_SUCCESS);
} /* ParallelGrid::SendReceiveRawBuffer */

/**
 * Send buffer in specified direction and receive buffer from the opposite direction
 */
void
ParallelGrid::SendReceiveBuffer (BufferPosition bufferDirection) /**< buffer direction to send data to and receive data
                                                                  *   from the opposite direction */
{
  /*
   * Return if node not used.
   */
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  if (parallelGridCore->getProcessId () >= parallelGridCore->getNodeGridSizeXYZ ())
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  if (parallelGridCore->getProcessId () >= parallelGridCore->getNodeGridSizeXY ())
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  if (parallelGridCore->getProcessId () >= parallelGridCore->getNodeGridSizeYZ ())
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  if (parallelGridCore->getProcessId () >= parallelGridCore->getNodeGridSizeXZ ())
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */

  BufferPosition opposite = parallelGridCore->getOppositeDirections ()[bufferDirection];

  int processTo = parallelGridCore->getNodeForDirection (bufferDirection);
  int processFrom = parallelGridCore->getNodeForDirection (opposite);

  /*
   * Copy to send buffer
   */
  if (processTo != PID_NONE)
  {
    grid_coord index = 0;

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    for (grid_coord i = sendStart[bufferDirection].get1 ();
         i < sendEnd[bufferDirection].get1 ();
         ++i)
    {
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
      for (grid_coord j = sendStart[bufferDirection].get2 ();
           j < sendEnd[bufferDirection].get2 ();
           ++j)
      {
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
        for (grid_coord k = sendStart[bufferDirection].get3 ();
             k < sendEnd[bufferDirection].get3 ();
             ++k)
        {
#endif /* GRID_3D */

#if defined (GRID_1D)
          ParallelGridCoordinate pos (i COORD_TYPES);
#endif /* GRID_1D */
#if defined (GRID_2D)
          ParallelGridCoordinate pos (i, j COORD_TYPES);
#endif /* GRID_2D */
#if defined (GRID_3D)
          ParallelGridCoordinate pos (i, j, k COORD_TYPES);
#endif /* GRID_3D */

          FieldPointValue* val = getFieldPointValue (pos);

          buffersSend[bufferDirection][index++] = val->getCurValue ();
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
          buffersSend[bufferDirection][index++] = val->getPrevValue ();
#if defined (TWO_TIME_STEPS)
          buffersSend[bufferDirection][index++] = val->getPrevPrevValue ();
#endif /* TWO_TIME_STEPS */
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

#if defined (GRID_3D)
        }
#endif /* GRID_3D */

#if defined (GRID_2D) || defined (GRID_3D)
      }
#endif /* GRID_2D || GRID_3D */

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    }
#endif /* GRID_1D || GRID_2D || GRID_3D */
  }

  DPRINTF (LOG_LEVEL_FULL, "\tSHARE RAW. PID=#%d. Directions TO(%s=#%d), FROM(%s=#%d).\n",
           parallelGridCore->getProcessId (),
           BufferPositionNames[bufferDirection],
           processTo,
           BufferPositionNames[opposite],
           processFrom);

  if (processTo != PID_NONE
      && processFrom == PID_NONE)
  {
#ifdef DYNAMIC_GRID
    parallelGridCore->StartShareClock (processTo, parallelGridCore->getShareClockCountCur (processTo));
#endif /* DYNAMIC_GRID */

    SendRawBuffer (bufferDirection, processTo);

#ifdef DYNAMIC_GRID
    parallelGridCore->StopShareClock (processTo, parallelGridCore->getShareClockCountCur (processTo));
#endif /* DYNAMIC_GRID */
  }
  else if (processTo == PID_NONE
           && processFrom != PID_NONE)
  {
#ifdef DYNAMIC_GRID
    parallelGridCore->StartShareClock (processFrom, parallelGridCore->getShareClockCountCur (processFrom));
#endif /* DYNAMIC_GRID */

    ReceiveRawBuffer (opposite, processFrom);

#ifdef DYNAMIC_GRID
    parallelGridCore->StopShareClock (processFrom, parallelGridCore->getShareClockCountCur (processFrom));
#endif /* DYNAMIC_GRID */
  }
  else if (processTo != PID_NONE
           && processFrom != PID_NONE)
  {
#ifdef COMBINED_SENDRECV
#ifdef DYNAMIC_GRID
    /*
     * TODO: support combined send/recv with dynamic grid
     */
    UNREACHABLE;
#endif /* DYNAMIC_GRID */
    SendReceiveRawBuffer (bufferDirection, processTo, opposite, processFrom);
#else /* COMBINED_SENDRECV */
    /*
     * Even nodes send first, then receive. Non-even receive first, then send
     */
    if (parallelGridCore->getIsEvenForDirection()[bufferDirection])
    {
#ifdef DYNAMIC_GRID
      parallelGridCore->StartShareClock (processTo, parallelGridCore->getShareClockCountCur (processTo));
#endif /* DYNAMIC_GRID */

      SendRawBuffer (bufferDirection, processTo);

#ifdef DYNAMIC_GRID
      parallelGridCore->StopShareClock (processTo, parallelGridCore->getShareClockCountCur (processTo));
      parallelGridCore->StartShareClock (processFrom, parallelGridCore->getShareClockCountCur (processFrom));
#endif /* DYNAMIC_GRID */

      ReceiveRawBuffer (opposite, processFrom);

#ifdef DYNAMIC_GRID
      parallelGridCore->StopShareClock (processFrom, parallelGridCore->getShareClockCountCur (processFrom));
#endif /* DYNAMIC_GRID */
    }
    else
    {
#ifdef DYNAMIC_GRID
      parallelGridCore->StartShareClock (processFrom, parallelGridCore->getShareClockCountCur (processFrom));
#endif /* DYNAMIC_GRID */

      ReceiveRawBuffer (opposite, processFrom);

#ifdef DYNAMIC_GRID
      parallelGridCore->StopShareClock (processFrom, parallelGridCore->getShareClockCountCur (processFrom));
      parallelGridCore->StartShareClock (processTo, parallelGridCore->getShareClockCountCur (processTo));
#endif /* DYNAMIC_GRID */

      SendRawBuffer (bufferDirection, processTo);

#ifdef DYNAMIC_GRID
      parallelGridCore->StopShareClock (processTo, parallelGridCore->getShareClockCountCur (processTo));
#endif /* DYNAMIC_GRID */
    }
#endif /* !COMBINED_SENDRECV */
  }
  else
  {
    /*
     * Do nothing (no neighbors in that direction)
     */
  }

  /*
   * Copy from receive buffer
   */
  if (processFrom != PID_NONE)
  {
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    for (grid_coord index = 0, i = recvStart[bufferDirection].get1 ();
         i < recvEnd[bufferDirection].get1 (); ++i)
    {
#endif /* GRID_1D || GRID_2D || GRID_3D */

#if defined (GRID_2D) || defined (GRID_3D)
      for (grid_coord j = recvStart[bufferDirection].get2 ();
           j < recvEnd[bufferDirection].get2 (); ++j)
      {
#endif /* GRID_2D || GRID_3D */

#if defined (GRID_3D)
        for (grid_coord k = recvStart[bufferDirection].get3 ();
             k < recvEnd[bufferDirection].get3 (); ++k)
        {
#endif /* GRID_3D */

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

#if defined (GRID_1D)
          ParallelGridCoordinate pos (i COORD_TYPES);
#endif /* GRID_1D */
#if defined (GRID_2D)
          ParallelGridCoordinate pos (i, j COORD_TYPES);
#endif /* GRID_2D */
#if defined (GRID_3D)
          ParallelGridCoordinate pos (i, j, k COORD_TYPES);
#endif /* GRID_3D */

          setFieldPointValue (val, pos);

#if defined (GRID_3D)
        }
#endif /* GRID_3D */

#if defined (GRID_2D) || defined (GRID_3D)
      }
#endif /* GRID_2D || GRID_3D */

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    }
#endif /* GRID_1D || GRID_2D || GRID_3D */
  }
} /* ParallelGrid::SendReceiveBuffer */

/**
 * Send/receive method to be called for all grid types.
 */
void
ParallelGrid::SendReceive ()
{
#ifdef DYNAMIC_GRID
  /*
   * No sharing for disabled nodes
   */
  if (parallelGridCore->getNodeState ()[parallelGridCore->getProcessId ()] == 0)
  {
    return;
  }
#endif /* DYNAMIC_GRID */

  DPRINTF (LOG_LEVEL_FULL, "Send/Receive PID=%d\n", parallelGridCore->getProcessId ());

  /*
   * Go through all directions and send/receive.
   */
  for (int buf = 0; buf < BUFFER_COUNT; ++buf)
  {
    SendReceiveBuffer ((BufferPosition) buf);
  }
} /* ParallelGrid::SendReceive */

/**
 * Perform share operations for grid
 */
void
ParallelGrid::share ()
{
  SendReceive ();
  MPI_Barrier (ParallelGrid::getParallelCore ()->getCommunicator ());
} /* ParallelGrid::share */

/**
 * Init parallel buffers
 */
void
ParallelGrid::InitBuffers ()
{
  /*
   * Return if node not used.
   */
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  if (parallelGridCore->getProcessId () >= parallelGridCore->getNodeGridSizeXYZ ())
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  if (parallelGridCore->getProcessId () >= parallelGridCore->getNodeGridSizeXY ())
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  if (parallelGridCore->getProcessId () >= parallelGridCore->getNodeGridSizeYZ ())
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  if (parallelGridCore->getProcessId () >= parallelGridCore->getNodeGridSizeXZ ())
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */

  /*
   * Number of time steps in build used to initialize parallel grid
   */
#if defined (ONE_TIME_STEP)
  const grid_coord numTimeStepsInBuild = 2;
#endif /* ONE_TIME_STEP */
#if defined (TWO_TIME_STEPS)
  const grid_coord numTimeStepsInBuild = 3;
#endif /* TWO_TIME_STEPS */

  buffersSend.resize (BUFFER_COUNT);
  buffersReceive.resize (BUFFER_COUNT);

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasL ())
  {
    int buf_size = bufferSize.get1 () * numTimeStepsInBuild;
#if defined (GRID_2D) || defined (GRID_3D)
    buf_size *= currentSize.get2 ();
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
    buf_size *= currentSize.get3 ();
#endif /* GRID_3D */
    buffersSend[LEFT].resize (buf_size);
    buffersReceive[LEFT].resize (buf_size);
  }
  if (parallelGridCore->getHasR ())
  {
    int buf_size = bufferSize.get1 () * numTimeStepsInBuild;
#if defined (GRID_2D) || defined (GRID_3D)
    buf_size *= currentSize.get2 ();
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
    buf_size *= currentSize.get3 ();
#endif /* GRID_3D */
    buffersSend[RIGHT].resize (buf_size);
    buffersReceive[RIGHT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasD ())
  {
    int buf_size = bufferSize.get2 () * currentSize.get1 () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.get3 ();
#endif /* GRID_3D */
    buffersSend[DOWN].resize (buf_size);
    buffersReceive[DOWN].resize (buf_size);
  }
  if (parallelGridCore->getHasU ())
  {
    int buf_size = bufferSize.get2 () * currentSize.get1 () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.get3 ();
#endif /* GRID_3D */
    buffersSend[UP].resize (buf_size);
    buffersReceive[UP].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasB ())
  {
    int buf_size = bufferSize.get3 () * currentSize.get2 () * currentSize.get1 () * numTimeStepsInBuild;
    buffersSend[BACK].resize (buf_size);
    buffersReceive[BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.get3 () * currentSize.get2 () * currentSize.get1 () * numTimeStepsInBuild;
    buffersSend[FRONT].resize (buf_size);
    buffersReceive[FRONT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasL () && parallelGridCore->getHasD ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get2 () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.get3 ();
#endif /* GRID_3D */
    buffersSend[LEFT_DOWN].resize (buf_size);
    buffersReceive[LEFT_DOWN].resize (buf_size);
  }
  if (parallelGridCore->getHasL () && parallelGridCore->getHasU ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get2 () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.get3 ();
#endif /* GRID_3D */
    buffersSend[LEFT_UP].resize (buf_size);
    buffersReceive[LEFT_UP].resize (buf_size);
  }
  if (parallelGridCore->getHasR () && parallelGridCore->getHasD ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get2 () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.get3 ();
#endif /* GRID_3D */
    buffersSend[RIGHT_DOWN].resize (buf_size);
    buffersReceive[RIGHT_DOWN].resize (buf_size);
  }
  if (parallelGridCore->getHasR () && parallelGridCore->getHasU ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get2 () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.get3 ();
#endif /* GRID_3D */
    buffersSend[RIGHT_UP].resize (buf_size);
    buffersReceive[RIGHT_UP].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasD () && parallelGridCore->getHasB ())
  {
    int buf_size = bufferSize.get2 () * bufferSize.get3 () * currentSize.get1 () * numTimeStepsInBuild;
    buffersSend[DOWN_BACK].resize (buf_size);
    buffersReceive[DOWN_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasD () && parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.get2 () * bufferSize.get3 () * currentSize.get1 () * numTimeStepsInBuild;
    buffersSend[DOWN_FRONT].resize (buf_size);
    buffersReceive[DOWN_FRONT].resize (buf_size);
  }
  if (parallelGridCore->getHasU () && parallelGridCore->getHasB ())
  {
    int buf_size = bufferSize.get2 () * bufferSize.get3 () * currentSize.get1 () * numTimeStepsInBuild;
    buffersSend[UP_BACK].resize (buf_size);
    buffersReceive[UP_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasU () && parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.get2 () * bufferSize.get3 () * currentSize.get1 () * numTimeStepsInBuild;
    buffersSend[UP_FRONT].resize (buf_size);
    buffersReceive[UP_FRONT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasL () && parallelGridCore->getHasB ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get3 () * currentSize.get2 () * numTimeStepsInBuild;
    buffersSend[LEFT_BACK].resize (buf_size);
    buffersReceive[LEFT_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasL () && parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get3 () * currentSize.get2 () * numTimeStepsInBuild;
    buffersSend[LEFT_FRONT].resize (buf_size);
    buffersReceive[LEFT_FRONT].resize (buf_size);
  }
  if (parallelGridCore->getHasR () && parallelGridCore->getHasB ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get3 () * currentSize.get2 () * numTimeStepsInBuild;
    buffersSend[RIGHT_BACK].resize (buf_size);
    buffersReceive[RIGHT_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasR () && parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get3 () * currentSize.get2 () * numTimeStepsInBuild;
    buffersSend[RIGHT_FRONT].resize (buf_size);
    buffersReceive[RIGHT_FRONT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int buf_size = bufferSize.get1 () * bufferSize.get2 () * bufferSize.get3 () * numTimeStepsInBuild;
  if (parallelGridCore->getHasL ()
      && parallelGridCore->getHasD ()
      && parallelGridCore->getHasB ())
  {
    buffersSend[LEFT_DOWN_BACK].resize (buf_size);
    buffersReceive[LEFT_DOWN_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasL ()
      && parallelGridCore->getHasD ()
      && parallelGridCore->getHasF ())
  {
    buffersSend[LEFT_DOWN_FRONT].resize (buf_size);
    buffersReceive[LEFT_DOWN_FRONT].resize (buf_size);
  }
  if (parallelGridCore->getHasL ()
      && parallelGridCore->getHasU ()
      && parallelGridCore->getHasB ())
  {
    buffersSend[LEFT_UP_BACK].resize (buf_size);
    buffersReceive[LEFT_UP_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasL ()
      && parallelGridCore->getHasU ()
      && parallelGridCore->getHasF ())
  {
    buffersSend[LEFT_UP_FRONT].resize (buf_size);
    buffersReceive[LEFT_UP_FRONT].resize (buf_size);
  }

  if (parallelGridCore->getHasR ()
      && parallelGridCore->getHasD ()
      && parallelGridCore->getHasB ())
  {
    buffersSend[RIGHT_DOWN_BACK].resize (buf_size);
    buffersReceive[RIGHT_DOWN_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasR ()
      && parallelGridCore->getHasD ()
      && parallelGridCore->getHasF ())
  {
    buffersSend[RIGHT_DOWN_FRONT].resize (buf_size);
    buffersReceive[RIGHT_DOWN_FRONT].resize (buf_size);
  }
  if (parallelGridCore->getHasR ()
      && parallelGridCore->getHasU ()
      && parallelGridCore->getHasB ())
  {
    buffersSend[RIGHT_UP_BACK].resize (buf_size);
    buffersReceive[RIGHT_UP_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasR ()
      && parallelGridCore->getHasU ()
      && parallelGridCore->getHasF ())
  {
    buffersSend[RIGHT_UP_FRONT].resize (buf_size);
    buffersReceive[RIGHT_UP_FRONT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
} /* ParallelGrid::InitBuffers */

/**
 * Initialize parallel grid parallel data: size and buffers
 */
void
ParallelGrid::ParallelGridConstructor ()
{
  grid_coord left_coord, right_coord;
  grid_coord down_coord, up_coord;
  grid_coord back_coord, front_coord;

  initBufferOffsets (left_coord, right_coord, down_coord, up_coord, back_coord, front_coord);

#ifdef GRID_1D
  ParallelGridCoordinate bufLeft (left_coord COORD_TYPES);
  ParallelGridCoordinate bufRight (right_coord COORD_TYPES);
#endif /* GRID_1D */
#ifdef GRID_2D
  ParallelGridCoordinate bufLeft (left_coord, down_coord COORD_TYPES);
  ParallelGridCoordinate bufRight (right_coord, up_coord COORD_TYPES);
#endif /* GRID_2D */
#ifdef GRID_3D
  ParallelGridCoordinate bufLeft (left_coord, down_coord, back_coord COORD_TYPES);
  ParallelGridCoordinate bufRight (right_coord, up_coord, front_coord COORD_TYPES);
#endif /* GRID_3D */

  size = currentSize + bufLeft + bufRight;

  /*
   * Init parallel buffers
   */
  InitBuffers ();

  SendReceiveCoordinatesInit ();

#ifdef GRID_1D
  DPRINTF (LOG_LEVEL_FULL, "Grid size for #%d process: " COORD_MOD ".\n",
          parallelGridCore->getProcessId (),
          currentSize.get1 ());
#endif /* GRID_1D */

#ifdef GRID_2D
  DPRINTF (LOG_LEVEL_FULL, "Grid size for #%d process: " COORD_MOD "x" COORD_MOD ".\n",
          parallelGridCore->getProcessId (),
          currentSize.get1 (),
          currentSize.get2 ());
#endif /* GRID_2D */

#ifdef GRID_3D
  DPRINTF (LOG_LEVEL_FULL, "Grid size for #%d process: " COORD_MOD "x" COORD_MOD "x" COORD_MOD ".\n",
          parallelGridCore->getProcessId (),
          currentSize.get1 (),
          currentSize.get2 (),
          currentSize.get3 ());
#endif /* GRID_3D */
} /* ParallelGrid::ParallelGridConstructor */

/**
 * Switch to next time step
 */
void
ParallelGrid::nextTimeStep ()
{
  ParallelGridBase::nextTimeStep ();

  nextShareStep ();

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  ASSERT (shareStep <= bufferSize.get1 ());

  bool is_share_time = shareStep == bufferSize.get1 ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  ASSERT (shareStep <= bufferSize.get2 ());

  bool is_share_time = shareStep == bufferSize.get2 ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  ASSERT (shareStep <= bufferSize.get3 ());

  bool is_share_time = shareStep == bufferSize.get3 ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  if (is_share_time)
  {
    share ();
    zeroShareStep ();
  }
} /* ParallelGrid::nextTimeStep */

/**
 * Allocate and gather full grid from all nodes to one non-parallel grid on each node
 *
 * @return full grid from all nodes as one non-parallel grid on each node
 *
 * Note: caller has to delete returned grid!
 */
ParallelGridBase *
ParallelGrid::gatherFullGrid () const
{
  ParallelGridBase *grid = new ParallelGridBase (totalSize, ParallelGridBase::timeStep, getName ().c_str ());

  /*
   * Fill new grid with values
   */
  for (grid_coord iter = 0; iter < grid->getSize ().calculateTotalCoord (); ++iter)
  {
    FieldPointValue *val = new FieldPointValue ();

    grid->setFieldPointValue (val, grid->calculatePositionFromIndex (iter));
  }

  return gatherFullGridPlacement (grid);
} /* ParallelGrid::gatherFullGrid */

/**
 * Gather full grid from all nodes to one non-parallel grid on each node
 *
 * @return full grid from all nodes as one non-parallel grid on each node
 *
 * Note: caller has to delete returned grid!
 */
ParallelGridBase *
ParallelGrid::gatherFullGridPlacement (ParallelGridBase *placementGrid) const
{
  ParallelGridBase *grid = placementGrid;

  /*
   * Each computational node broadcasts to all others its data
   */

  for (int process = 0; process < ParallelGrid::getParallelCore ()->getTotalProcCount (); ++process)
  {
    ParallelGridCoordinate chunkStart = getChunkStartPosition ();
    ParallelGridCoordinate chunkEnd = chunkStart + getCurrentSize ();

    /*
     * Send start coord, end coord
     */
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    grid_coord startX;
    grid_coord endX;

    if (process == ParallelGrid::getParallelCore ()->getProcessId ())
    {
      startX = chunkStart.get1 ();
      endX = chunkEnd.get1 ();
    }

    MPI_Bcast (&startX, 1, MPI_COORD, process, ParallelGrid::getParallelCore ()->getCommunicator ());
    MPI_Bcast (&endX, 1, MPI_COORD, process, ParallelGrid::getParallelCore ()->getCommunicator ());
#endif /* GRID_1D || GRID_2D || GRID_3D */

#if defined (GRID_2D) || defined (GRID_3D)
    grid_coord startY;
    grid_coord endY;

    if (process == ParallelGrid::getParallelCore ()->getProcessId ())
    {
      startY = chunkStart.get2 ();
      endY = chunkEnd.get2 ();
    }

    MPI_Bcast (&startY, 1, MPI_COORD, process, ParallelGrid::getParallelCore ()->getCommunicator ());
    MPI_Bcast (&endY, 1, MPI_COORD, process, ParallelGrid::getParallelCore ()->getCommunicator ());
#endif /* GRID_2D || GRID_3D */

#if defined (GRID_3D)
    grid_coord startZ;
    grid_coord endZ;

    if (process == ParallelGrid::getParallelCore ()->getProcessId ())
    {
      startZ = chunkStart.get3 ();
      endZ = chunkEnd.get3 ();
    }

    MPI_Bcast (&startZ, 1, MPI_COORD, process, ParallelGrid::getParallelCore ()->getCommunicator ());
    MPI_Bcast (&endZ, 1, MPI_COORD, process, ParallelGrid::getParallelCore ()->getCommunicator ());
#endif /* GRID_3D */

#ifdef GRID_1D
    ParallelGridCoordinate sizeCoord (endX - startX COORD_TYPES);
#endif /* GRID_1D */
#ifdef GRID_2D
    ParallelGridCoordinate sizeCoord (endX - startX, endY - startY COORD_TYPES);
#endif /* GRID_2D */
#ifdef GRID_3D
    ParallelGridCoordinate sizeCoord (endX - startX, endY - startY, endZ - startZ COORD_TYPES);
#endif /* GRID_3D */

    /*
     * Fill vectors with data for current computational node
     */
    grid_coord size = sizeCoord.calculateTotalCoord ();
    std::vector<FieldValue> current (size);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    std::vector<FieldValue> previous (size);
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

#if defined (TWO_TIME_STEPS)
    std::vector<FieldValue> previousPrev (size);
#endif /* TWO_TIME_STEPS */

    if (process == ParallelGrid::getParallelCore ()->getProcessId ())
    {
      grid_coord left_coord, right_coord;
      grid_coord down_coord, up_coord;
      grid_coord back_coord, front_coord;

      initBufferOffsets (left_coord, right_coord, down_coord, up_coord, back_coord, front_coord);

      grid_coord index = 0;

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
      for (grid_coord i = left_coord; i < left_coord + sizeCoord.get1 (); ++i)
      {
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
        for (grid_coord j = down_coord; j < down_coord + sizeCoord.get2 (); ++j)
        {
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
          for (grid_coord k = back_coord; k < back_coord + sizeCoord.get3 (); ++k)
          {
#endif /* GRID_3D */

#ifdef GRID_1D
            ParallelGridCoordinate pos (i COORD_TYPES);
#endif /* GRID_1D */
#ifdef GRID_2D
            ParallelGridCoordinate pos (i, j COORD_TYPES);
#endif /* GRID_2D */
#ifdef GRID_3D
            ParallelGridCoordinate pos (i, j, k COORD_TYPES);
#endif /* GRID_3D */

            grid_coord coord = calculateIndexFromPosition (pos);

            FieldPointValue *val = gridValues[coord];

            current[index] = val->getCurValue ();
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
            previous[index] = val->getPrevValue ();
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
#if defined (TWO_TIME_STEPS)
            previousPrev[index] = val->getPrevPrevValue ();
#endif /* TWO_TIME_STEPS */

            ++index;

#if defined (GRID_3D)
          }
#endif /* GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
        }
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
      }
#endif /* GRID_1D || GRID_2D || GRID_3D */
    }

    MPI_Datatype datatype;

#ifdef FLOAT_VALUES
#ifdef COMPLEX_FIELD_VALUES
    datatype = MPI_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
    datatype = MPI_FLOAT;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* FLOAT_VALUES */

#ifdef DOUBLE_VALUES
#ifdef COMPLEX_FIELD_VALUES
    datatype = MPI_DOUBLE_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
    datatype = MPI_DOUBLE;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* DOUBLE_VALUES */

#ifdef LONG_DOUBLE_VALUES
#ifdef COMPLEX_FIELD_VALUES
    datatype = MPI_LONG_DOUBLE_COMPLEX;
#else /* COMPLEX_FIELD_VALUES */
    datatype = MPI_LONG_DOUBLE;
#endif /* !COMPLEX_FIELD_VALUES */
#endif /* LONG_DOUBLE_VALUES */

    /*
     * Broadcast data
     */

    MPI_Bcast (current.data (), current.size (), datatype, process, ParallelGrid::getParallelCore ()->getCommunicator ());

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    MPI_Bcast (previous.data (), previous.size (), datatype, process, ParallelGrid::getParallelCore ()->getCommunicator ());
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

#if defined (TWO_TIME_STEPS)
    MPI_Bcast (previousPrev.data (), previousPrev.size (), datatype, process, ParallelGrid::getParallelCore ()->getCommunicator ());
#endif /* TWO_TIME_STEPS */

    grid_coord index = 0;

    /*
     * Store data to corresponding coordinates of the resulting grid
     */

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    for (grid_coord i = startX; i < endX; ++i)
    {
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
      for (grid_coord j = startY; j < endY; ++j)
      {
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
        for (grid_coord k = startZ; k < endZ; ++k)
        {
#endif /* GRID_3D */

#ifdef GRID_1D
          ParallelGridCoordinate pos (i COORD_TYPES);
#endif /* GRID_1D */
#ifdef GRID_2D
          ParallelGridCoordinate pos (i, j COORD_TYPES);
#endif /* GRID_2D */
#ifdef GRID_3D
          ParallelGridCoordinate pos (i, j, k COORD_TYPES);
#endif /* GRID_3D */

          FieldPointValue *val = grid->getFieldPointValue (pos);

          val->setCurValue (current[index]);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
          val->setPrevValue (previous[index]);
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */
#if defined (TWO_TIME_STEPS)
          val->setPrevPrevValue (previousPrev[index]);
#endif /* TWO_TIME_STEPS */

          ++index;

#if defined (GRID_3D)
        }
#endif /* GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
      }
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    }
#endif /* GRID_1D || GRID_2D || GRID_3D */

    MPI_Barrier (ParallelGrid::getParallelCore ()->getCommunicator ());
  }

  return grid;
} /* ParallelGrid::gatherFullGridPlacement */

/**
 * Identify buffer to which position corresponds to. In case coordinate is not in buffer, BUFFER_NONE is returned
 *
 * @return buffer, to which position corresponds to
 */
BufferPosition
ParallelGrid::getBufferForPosition (ParallelGridCoordinate pos) const
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)
  if (ParallelGrid::getParallelCore ()->getHasL ()
      && pos.get1 () < bufferSize.get1 ())
  {
    return LEFT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 ())
  {
    return RIGHT;
  }
  else
  {
    return BUFFER_NONE;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
  if (ParallelGrid::getParallelCore ()->getHasD ()
      && pos.get2 () < bufferSize.get2 ())
  {
    return DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get2 () >= getSize ().get2 () - bufferSize.get2 ())
  {
    return UP;
  }
  else
  {
    return BUFFER_NONE;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  if (ParallelGrid::getParallelCore ()->getHasB ()
      && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }
  else
  {
    return BUFFER_NONE;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (ParallelGrid::getParallelCore ()->getHasL ()
      && ParallelGrid::getParallelCore ()->getHasD ()
      && pos.get1 () < bufferSize.get1 () && pos.get2 () < bufferSize.get2 ())
  {
    return LEFT_DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 ())
  {
    return LEFT_UP;
  }

  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 ())
  {
    return LEFT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= bufferSize.get2 ())
  {
    return LEFT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 ())
  {
    return LEFT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 ())
  {
    return RIGHT_DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 ())
  {
    return RIGHT_UP;
  }

  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 ())
  {
    return RIGHT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 ())
  {
    return RIGHT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 ())
  {
    return RIGHT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 ())
  {
    return DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () < bufferSize.get2 ())
  {
    return DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 ())
  {
    return DOWN;
  }

  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 ())
  {
    return UP;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 ())
  {
    return UP;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 ())
  {
    return UP;
  }

  else
  {
    return BUFFER_NONE;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (ParallelGrid::getParallelCore ()->getHasD ()
      && ParallelGrid::getParallelCore ()->getHasB ()
      && pos.get2 () < bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return DOWN_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get2 () < bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return DOWN_FRONT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get2 () < bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get2 () < bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get2 () < bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return DOWN;
  }

  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return UP_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return UP_FRONT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return UP;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return UP;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return UP;
  }

  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get2 () >= bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }

  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get2 () >= bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }

  else
  {
    return BUFFER_NONE;
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  if (ParallelGrid::getParallelCore ()->getHasL ()
      && ParallelGrid::getParallelCore ()->getHasB ()
      && pos.get1 () < bufferSize.get1 () && pos.get3 () < bufferSize.get3 ())
  {
    return LEFT_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT_FRONT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get3 () >= bufferSize.get3 ())
  {
    return LEFT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get3 () < bufferSize.get3 ())
  {
    return RIGHT_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT_FRONT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get3 () >= bufferSize.get3 ())
  {
    return RIGHT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }

  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }

  else
  {
    return BUFFER_NONE;
  }
#endif

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  if (ParallelGrid::getParallelCore ()->getHasL ()
      && ParallelGrid::getParallelCore ()->getHasD ()
      && ParallelGrid::getParallelCore ()->getHasB ()
      && pos.get1 () < bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return LEFT_DOWN_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT_DOWN_FRONT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return LEFT_UP_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT_UP_FRONT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return RIGHT_DOWN_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT_DOWN_FRONT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return RIGHT_UP_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT_UP_FRONT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT_DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return LEFT_DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT_DOWN;
  }

  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT_DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return RIGHT_DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT_DOWN;
  }

  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT_UP;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return LEFT_UP;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT_UP;
  }

  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT_UP;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return RIGHT_UP;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT_UP;
  }

  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return DOWN_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return DOWN_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return DOWN_BACK;
  }

  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return DOWN_FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return DOWN_FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return DOWN_FRONT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return UP_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return UP_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return UP_BACK;
  }

  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return UP_FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return UP_FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return UP_FRONT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return LEFT_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return LEFT_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return LEFT_BACK;
  }

  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT_FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT_FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT_FRONT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return RIGHT_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return RIGHT_BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return RIGHT_BACK;
  }

  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT_FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT_FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT_FRONT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return LEFT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return LEFT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return LEFT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return LEFT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return RIGHT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return RIGHT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return RIGHT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return RIGHT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return DOWN;
  }

  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return DOWN;
  }

  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return DOWN;
  }
  else if (ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return DOWN;
  }

  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return UP;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return UP;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return UP;
  }

  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return UP;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return UP;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return UP;
  }

  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return UP;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 ())
  {
    return UP;
  }
  else if (ParallelGrid::getParallelCore ()->getHasU ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasF ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= bufferSize.get3 () && pos.get3 () < getSize ().get3 () - bufferSize.get3 ())
  {
    return UP;
  }

  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }

  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }

  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }
  else if (ParallelGrid::getParallelCore ()->getHasB ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () < bufferSize.get3 ())
  {
    return BACK;
  }

  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && !ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && !ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }

  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && !ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && !ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }
  else if (ParallelGrid::getParallelCore ()->getHasF ()
           && ParallelGrid::getParallelCore ()->getHasL ()
           && ParallelGrid::getParallelCore ()->getHasR ()
           && ParallelGrid::getParallelCore ()->getHasD ()
           && ParallelGrid::getParallelCore ()->getHasU ()
           && pos.get1 () >= bufferSize.get1 () && pos.get1 () < getSize ().get1 () - bufferSize.get1 () && pos.get2 () >= bufferSize.get2 () && pos.get2 () < getSize ().get2 () - bufferSize.get2 () && pos.get3 () >= getSize ().get3 () - bufferSize.get3 ())
  {
    return FRONT;
  }

  else
  {
    return BUFFER_NONE;
  }
#endif
} /* ParallelGrid::getBufferForPosition */

#undef COORD_TYPES

#endif /* PARALLEL_GRID */
