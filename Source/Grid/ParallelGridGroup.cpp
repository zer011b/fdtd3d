#include "ParallelGridGroup.h"

#ifdef PARALLEL_GRID

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

ParallelGridCore *ParallelGridGroup::parallelGridCore = NULLPTR;

ParallelGridGroup::ParallelGridGroup (ParallelGridCoordinate totSize,
                                      ParallelGridCoordinate bufSize,
                                      time_step stepLimit,
                                      ParallelGridCoordinate curSize,
                                      int storedTimeSteps,
                                      int tOffset,
                                      const char * name)
  : totalSize (totSize)
  , bufferSize (bufSize)
  , shareStep (0)
  , shareStepLimit (stepLimit)
  , currentSize (curSize)
  , storedSteps (storedTimeSteps)
  , timeOffset (tOffset)
  , groupName (name)
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
   * Construct parallel grid internals
   */
  ParallelGridGroupConstructor ();
}

bool
ParallelGridGroup::match (ParallelGridCoordinate totSize,
                          ParallelGridCoordinate bufSize,
                          time_step stepLimit,
                          ParallelGridCoordinate curSize,
                          int storedTimeSteps,
                          int tOffset) const
{
  return totalSize == totSize
         && bufferSize == bufSize
         && shareStepLimit == stepLimit
         && currentSize == curSize
         && storedSteps == storedTimeSteps
         && timeOffset == tOffset;
}

/**
 * Initialize parallel grid parallel data: size and buffers
 */
void
ParallelGridGroup::ParallelGridGroupConstructor ()
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

  gatherStartPosition ();

  DPRINTF (LOG_LEVEL_FULL, "New parallel group for #%d process: " C_MOD " x " C_MOD " x " C_MOD ".\n",
          parallelGridCore->getProcessId (),
          currentSize.get1 (),
          currentSize.get2 (),
          currentSize.get3 ());
} /* ParallelGridGroup::ParallelGridGroupConstructor */

/**
 * Gather start position for all computational nodes
 */
void ParallelGridGroup::gatherStartPosition ()
{
  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP,
           "Gather start position for '%s' for proc: %d (of %d).\n",
           groupName.data (),
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

  for (int process = 0; process < getParallelCore ()->getTotalProcCount (); ++process)
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
    state = getParallelCore ()->getNodeState ()[getParallelCore ()->getProcessId ()];
#endif /* DYNAMIC_GRID */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
  defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

    if (process == getParallelCore ()->getProcessId ()
        && state
        && getParallelCore ()->getNodeForDirection (LEFT) != PID_NONE)
    {
      retCode = MPI_Recv (&startx,
                          1,
                          MPI_COORD,
                          getParallelCore ()->getNodeForDirection (LEFT),
                          0,
                          getParallelCore ()->getCommunicator (),
                          &status);
      ASSERT (retCode == MPI_SUCCESS);

      hasReceivedX = true;
    }
    else if (state
             && getParallelCore ()->getNodeForDirection (RIGHT) == process)
    {
      retCode = MPI_Send (&endx,
                          1,
                          MPI_COORD,
                          getParallelCore ()->getNodeForDirection (RIGHT),
                          0,
                          getParallelCore ()->getCommunicator ());
      ASSERT (retCode == MPI_SUCCESS);
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
  defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

    if (process == getParallelCore ()->getProcessId ()
        && state
        && getParallelCore ()->getNodeForDirection (DOWN) != PID_NONE)
    {
      retCode = MPI_Recv (&starty,
                          1,
                          MPI_COORD,
                          getParallelCore ()->getNodeForDirection (DOWN),
                          1,
                          getParallelCore ()->getCommunicator (),
                          &status);
      ASSERT (retCode == MPI_SUCCESS);

      hasReceivedY = true;
    }
    else if (state
             && getParallelCore ()->getNodeForDirection (UP) == process)
    {
      retCode = MPI_Send (&endy,
                          1,
                          MPI_COORD,
                          getParallelCore ()->getNodeForDirection (UP),
                          1,
                          getParallelCore ()->getCommunicator ());
      ASSERT (retCode == MPI_SUCCESS);
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
  defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

    if (process == getParallelCore ()->getProcessId ()
        && state
        && getParallelCore ()->getNodeForDirection (BACK) != PID_NONE)
    {
      retCode = MPI_Recv (&startz,
                          1,
                          MPI_COORD,
                          getParallelCore ()->getNodeForDirection (BACK),
                          2,
                          getParallelCore ()->getCommunicator (),
                          &status);
      ASSERT (retCode == MPI_SUCCESS);

      hasReceivedZ = true;
    }
    else if (state
             && getParallelCore ()->getNodeForDirection (FRONT) == process)
    {
      retCode = MPI_Send (&endz,
                          1,
                          MPI_COORD,
                          getParallelCore ()->getNodeForDirection (FRONT),
                          2,
                          getParallelCore ()->getCommunicator ());
      ASSERT (retCode == MPI_SUCCESS);
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

    if (process == getParallelCore ()->getProcessId ()
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

    if (process == getParallelCore ()->getProcessId ()
        && state)
    {
#ifdef GRID_1D
      DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Start pos (" C_MOD ") for grid '%s' for proc %d (of %d)\n",
               startx,
               groupName.data (),
               parallelGridCore->getProcessId (),
               parallelGridCore->getTotalProcCount ());
#endif /* GRID_1D */
#ifdef GRID_2D
      DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Start pos (" C_MOD ", " C_MOD ") for grid '%s' for proc %d (of %d)\n",
               startx,
               starty,
               groupName.data (),
               parallelGridCore->getProcessId (),
               parallelGridCore->getTotalProcCount ());
#endif /* GRID_2D */
#ifdef GRID_3D
      DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "Start pos (" C_MOD ", " C_MOD ", " C_MOD ") for grid '%s' for proc %d (of %d)\n",
               startx,
               starty,
               startz,
               groupName.data (),
               parallelGridCore->getProcessId (),
               parallelGridCore->getTotalProcCount ());
#endif /* GRID_3D */
    }

    MPI_Barrier (getParallelCore ()->getCommunicator ());
  }
} /* ParallelGridGroup::gatherStartPosition */


/**
 * Initialize absolute start position of chunk for current node
 */
void
ParallelGridGroup::initializeStartPosition (ParallelGridCoordinate chunkStartPos) /**< start position of chunk, that is
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
} /* ParallelGridGroup::initializeStartPosition */

/**
 * Init parallel buffers
 */
void
ParallelGridGroup::InitBuffers ()
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

  buffersSend.resize (BUFFER_COUNT);
  buffersReceive.resize (BUFFER_COUNT);

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasL ())
  {
    int buf_size = bufferSize.get1 () * storedSteps;
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
    int buf_size = bufferSize.get1 () * storedSteps;
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
    int buf_size = bufferSize.get2 () * currentSize.get1 () * storedSteps;
#if defined (GRID_3D)
    buf_size *= currentSize.get3 ();
#endif /* GRID_3D */
    buffersSend[DOWN].resize (buf_size);
    buffersReceive[DOWN].resize (buf_size);
  }
  if (parallelGridCore->getHasU ())
  {
    int buf_size = bufferSize.get2 () * currentSize.get1 () * storedSteps;
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
    int buf_size = bufferSize.get3 () * currentSize.get2 () * currentSize.get1 () * storedSteps;
    buffersSend[BACK].resize (buf_size);
    buffersReceive[BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.get3 () * currentSize.get2 () * currentSize.get1 () * storedSteps;
    buffersSend[FRONT].resize (buf_size);
    buffersReceive[FRONT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasL () && parallelGridCore->getHasD ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get2 () * storedSteps;
#if defined (GRID_3D)
    buf_size *= currentSize.get3 ();
#endif /* GRID_3D */
    buffersSend[LEFT_DOWN].resize (buf_size);
    buffersReceive[LEFT_DOWN].resize (buf_size);
  }
  if (parallelGridCore->getHasL () && parallelGridCore->getHasU ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get2 () * storedSteps;
#if defined (GRID_3D)
    buf_size *= currentSize.get3 ();
#endif /* GRID_3D */
    buffersSend[LEFT_UP].resize (buf_size);
    buffersReceive[LEFT_UP].resize (buf_size);
  }
  if (parallelGridCore->getHasR () && parallelGridCore->getHasD ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get2 () * storedSteps;
#if defined (GRID_3D)
    buf_size *= currentSize.get3 ();
#endif /* GRID_3D */
    buffersSend[RIGHT_DOWN].resize (buf_size);
    buffersReceive[RIGHT_DOWN].resize (buf_size);
  }
  if (parallelGridCore->getHasR () && parallelGridCore->getHasU ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get2 () * storedSteps;
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
    int buf_size = bufferSize.get2 () * bufferSize.get3 () * currentSize.get1 () * storedSteps;
    buffersSend[DOWN_BACK].resize (buf_size);
    buffersReceive[DOWN_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasD () && parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.get2 () * bufferSize.get3 () * currentSize.get1 () * storedSteps;
    buffersSend[DOWN_FRONT].resize (buf_size);
    buffersReceive[DOWN_FRONT].resize (buf_size);
  }
  if (parallelGridCore->getHasU () && parallelGridCore->getHasB ())
  {
    int buf_size = bufferSize.get2 () * bufferSize.get3 () * currentSize.get1 () * storedSteps;
    buffersSend[UP_BACK].resize (buf_size);
    buffersReceive[UP_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasU () && parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.get2 () * bufferSize.get3 () * currentSize.get1 () * storedSteps;
    buffersSend[UP_FRONT].resize (buf_size);
    buffersReceive[UP_FRONT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasL () && parallelGridCore->getHasB ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get3 () * currentSize.get2 () * storedSteps;
    buffersSend[LEFT_BACK].resize (buf_size);
    buffersReceive[LEFT_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasL () && parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get3 () * currentSize.get2 () * storedSteps;
    buffersSend[LEFT_FRONT].resize (buf_size);
    buffersReceive[LEFT_FRONT].resize (buf_size);
  }
  if (parallelGridCore->getHasR () && parallelGridCore->getHasB ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get3 () * currentSize.get2 () * storedSteps;
    buffersSend[RIGHT_BACK].resize (buf_size);
    buffersReceive[RIGHT_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasR () && parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.get1 () * bufferSize.get3 () * currentSize.get2 () * storedSteps;
    buffersSend[RIGHT_FRONT].resize (buf_size);
    buffersReceive[RIGHT_FRONT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int buf_size = bufferSize.get1 () * bufferSize.get2 () * bufferSize.get3 () * storedSteps;
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
} /* ParallelGridGroup::InitBuffers */


/**
 * Initialize start and end cooridnates for send/receive for all directions
 */
void
ParallelGridGroup::SendReceiveCoordinatesInit ()
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
} /* ParallelGridGroup::SendReceiveCoordinatesInit */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

/**
 * Initialize start and end cooridnates for send/receive for all directions for 1D X mode
 */
void
ParallelGridGroup::SendReceiveCoordinatesInit1D_X ()
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
} /* ParallelGridGroup::SendReceiveCoordinatesInit1D_X */
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

/**
 * Initialize start and end cooridnates for send/receive for all directions for 1D Y mode
 */
void
ParallelGridGroup::SendReceiveCoordinatesInit1D_Y ()
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
} /* ParallelGridGroup::SendReceiveCoordinatesInit1D_Y */
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

/**
 * Initialize start and end cooridnates for send/receive for all directions for 1D Z mode
 */
void
ParallelGridGroup::SendReceiveCoordinatesInit1D_Z ()
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
} /* ParallelGridGroup::SendReceiveCoordinatesInit1D_Z */
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

/**
 * Initialize start and end cooridnates for send/receive for all directions for 2D XY mode
 */
void
ParallelGridGroup::SendReceiveCoordinatesInit2D_XY ()
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
} /* ParallelGridGroup::SendReceiveCoordinatesInit2D_XY */
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

/**
 * Initialize start and end cooridnates for send/receive for all directions for 2D YZ mode
 */
void
ParallelGridGroup::SendReceiveCoordinatesInit2D_YZ ()
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
} /* ParallelGridGroup::SendReceiveCoordinatesInit2D_YZ */
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

/**
 * Initialize start and end cooridnates for send/receive for all directions for 2D XZ mode
 */
void
ParallelGridGroup::SendReceiveCoordinatesInit2D_XZ ()
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
} /* ParallelGridGroup::SendReceiveCoordinatesInit2D_XZ */
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

/**
 * Initialize start and end cooridnates for send/receive for all directions for 3D XYZ mode
 */
void
ParallelGridGroup::SendReceiveCoordinatesInit3D_XYZ ()
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
} /* ParallelGridGroup::SendReceiveCoordinatesInit3D_XYZ */
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#endif /* PARALLEL_GRID */
