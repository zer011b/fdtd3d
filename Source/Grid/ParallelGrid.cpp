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

std::vector<ParallelGridGroup *> ParallelGrid::groups;

#ifdef DEBUG_INFO

#ifdef GRID_1D
#define COORD_TYPES , getGroupConst()->get_ct1 ()
#endif /* GRID_1D */

#ifdef GRID_2D
#define COORD_TYPES , getGroupConst()->get_ct1 (), getGroupConst()->get_ct2 ()
#endif /* GRID_2D */

#ifdef GRID_3D
#define COORD_TYPES , getGroupConst()->get_ct1 (), getGroupConst()->get_ct2 (), getGroupConst()->get_ct3 ()
#endif /* GRID_3D */

#else /* DEBUG_INFO */
#define COORD_TYPES
#endif /* !DEBUG_INFO */

/**
 * Parallel grid constructor
 */
ParallelGrid::ParallelGrid (const ParallelGridCoordinate &totSize, /**< total size of grid */
                            const ParallelGridCoordinate &bufSize, /**< buffer size */
                            time_step stepLimit, /**< step limit */
                            const ParallelGridCoordinate &curSize,  /**< size of grid for current node, received from layout */
                            int storedSteps, /**< number of steps in time for which to store grid values */
                            int timeOffset, /**< offset of time step in for t+timeOffset/2, at which grid should be shared */
                            const char * name) /**< name of grid */
  : ParallelGridBase (storedSteps, name)
{
  /*
   * These are required here to properly setup bufferSize, because ParallelGridGroup does not exist yet
   */
#ifdef DEBUG_INFO
#ifdef GRID_1D
#define COORD_TYPES_TMP , totSize.getType1 ()
#endif /* GRID_1D */

#ifdef GRID_2D
#define COORD_TYPES_TMP , totSize.getType1 (), totSize.getType2 ()
#endif /* GRID_2D */

#ifdef GRID_3D
#define COORD_TYPES_TMP , totSize.getType1 (), totSize.getType2 (), totSize.getType3 ()
#endif /* GRID_3D */

#else /* DEBUG_INFO */
#define COORD_TYPES_TMP
#endif /* !DEBUG_INFO */

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

  ParallelGridCoordinate bufferSize;

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
  bufferSize = ParallelGridCoordinate (bufSize.get1 (), 0 COORD_TYPES_TMP);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  bufferSize = ParallelGridCoordinate (0, bufSize.get2 () COORD_TYPES_TMP);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  bufferSize = bufSize;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */
#endif /* GRID_2D */

#ifdef GRID_3D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  bufferSize = ParallelGridCoordinate (bufSize.get1 (), 0, 0 COORD_TYPES_TMP);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  bufferSize = ParallelGridCoordinate (0, bufSize.get2 (), 0 COORD_TYPES_TMP);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  bufferSize = ParallelGridCoordinate (0, 0, bufSize.get3 () COORD_TYPES_TMP);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  bufferSize = ParallelGridCoordinate (bufSize.get1 (), bufSize.get2 (), 0 COORD_TYPES_TMP);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  bufferSize = ParallelGridCoordinate (0, bufSize.get2 (), bufSize.get3 () COORD_TYPES_TMP);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  bufferSize = ParallelGridCoordinate (bufSize.get1 (), 0, bufSize.get3 () COORD_TYPES_TMP);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  bufferSize = bufSize;
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
#endif /* GRID_3D */

#undef COORD_TYPES_TMP

  // check if group for this ParallelGrid already exists
  groupId = findGroup (totSize, bufferSize, stepLimit, curSize, storedSteps, timeOffset);

  if (groupId == INVALID_GROUP)
  {
    // group for this ParallelGrid doesn't exist, create new one
    ParallelGridGroup *group = new ParallelGridGroup (totSize, bufferSize, stepLimit, curSize, storedSteps, timeOffset, name);
    groupId = addGroup (group);
  }

  size = getGroup ()->getSize ();

  for (int i = 0; i < gridValues.size (); ++i)
  {
    gridValues[i] = new VectorFieldValues (size.calculateTotalCoord ());
  }

  DPRINTF (LOG_LEVEL_STAGES_AND_DUMP, "New grid '%s' for proc: %d (of %d) with %lu stored steps with raw size: %llu.\n",
           gridName.data (),
           parallelGridCore->getProcessId (),
           parallelGridCore->getTotalProcCount (),
           gridValues.size (),
           (unsigned long long)size.calculateTotalCoord ());
} /* ParallelGrid::ParallelGrid */

/**
 * Send raw buffer with data
 */
void
ParallelGrid::SendRawBuffer (BufferPosition buffer, /**< buffer's position to send (direction) */
                             int processTo) /**< id of computational node to send data to */
{
  VectorBuffers &buffersSend = getGroup ()->getBuffersSend ();

  DPRINTF (LOG_LEVEL_FULL, "\tSend RAW. PID=#%d. Direction TO=%s, size=%lu.\n",
           parallelGridCore->getProcessId (),
           BufferPositionNames[buffer],
           buffersSend[buffer].size ());

  FieldValue* rawBuffer = buffersSend[buffer].data ();

  int retCode = MPI_Send (rawBuffer,
                          buffersSend[buffer].size (),
                          MPI_FPVALUE,
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
  VectorBuffers &buffersReceive = getGroup ()->getBuffersReceive ();

  DPRINTF (LOG_LEVEL_FULL, "\t\tReceive RAW. PID=#%d. Direction FROM=%s, size=%lu.\n",
           parallelGridCore->getProcessId (),
           BufferPositionNames[buffer],
           buffersReceive[buffer].size ());

  MPI_Status status;

  FieldValue* rawBuffer = buffersReceive[buffer].data ();

  int retCode = MPI_Recv (rawBuffer,
                          buffersReceive[buffer].size (),
                          MPI_FPVALUE,
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
  VectorBuffers &buffersSend = getGroup ()->getBuffersSend ();
  VectorBuffers &buffersReceive = getGroup ()->getBuffersReceive ();

  DPRINTF (LOG_LEVEL_FULL, "\t\tSend/Receive RAW. PID=#%d. Directions TO=%s FROM=%s. Size TO=%lu FROM=%lu.\n",
           parallelGridCore->getProcessId (),
           BufferPositionNames[bufferSend],
           BufferPositionNames[bufferReceive],
           buffersSend[bufferSend].size (),
           buffersReceive[bufferReceive].size ());

  MPI_Status status;

  FieldValue* rawBufferSend = buffersSend[bufferSend].data ();
  FieldValue* rawBufferReceive = buffersReceive[bufferReceive].data ();

  int retCode = MPI_Sendrecv (rawBufferSend,
                              buffersSend[bufferSend].size (),
                              MPI_FPVALUE,
                              processTo,
                              parallelGridCore->getProcessId (),
                              rawBufferReceive,
                              buffersReceive[bufferReceive].size (),
                              MPI_FPVALUE,
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

  ParallelGridCoordinate sendStart = getSendStart (bufferDirection);
  ParallelGridCoordinate sendEnd = getSendEnd (bufferDirection);

  ParallelGridCoordinate recvStart = getRecvStart (bufferDirection);
  ParallelGridCoordinate recvEnd = getRecvEnd (bufferDirection);

  VectorBuffers &buffersSend = getGroup ()->getBuffersSend ();
  VectorBuffers &buffersReceive = getGroup ()->getBuffersReceive ();

  /*
   * Copy to send buffer
   */
  if (processTo != PID_NONE)
  {
    grid_coord index = 0;

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    for (grid_coord i = sendStart.get1 (); i < sendEnd.get1 (); ++i)
#endif /* GRID_1D || GRID_2D || GRID_3D */
    {
#if defined (GRID_2D) || defined (GRID_3D)
      for (grid_coord j = sendStart.get2 (); j < sendEnd.get2 (); ++j)
#endif /* GRID_2D || GRID_3D */
      {
#if defined (GRID_3D)
        for (grid_coord k = sendStart.get3 (); k < sendEnd.get3 (); ++k)
#endif /* GRID_3D */
        {

#if defined (GRID_1D)
          ParallelGridCoordinate pos (i COORD_TYPES);
#endif /* GRID_1D */
#if defined (GRID_2D)
          ParallelGridCoordinate pos (i, j COORD_TYPES);
#endif /* GRID_2D */
#if defined (GRID_3D)
          ParallelGridCoordinate pos (i, j, k COORD_TYPES);
#endif /* GRID_3D */

          grid_coord coord = calculateIndexFromPosition (pos);
          for (int t = 0; t < gridValues.size (); ++t)
          {
            buffersSend[bufferDirection][index++] = *getFieldValue (coord, t);
          }
        }
      }
    }
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
    for (grid_coord index = 0, i = recvStart.get1 (); i < recvEnd.get1 (); ++i)
#endif /* GRID_1D || GRID_2D || GRID_3D */
    {

#if defined (GRID_2D) || defined (GRID_3D)
      for (grid_coord j = recvStart.get2 (); j < recvEnd.get2 (); ++j)
#endif /* GRID_2D || GRID_3D */
      {

#if defined (GRID_3D)
        for (grid_coord k = recvStart.get3 (); k < recvEnd.get3 (); ++k)
#endif /* GRID_3D */
        {

#if defined (GRID_1D)
          ParallelGridCoordinate pos (i COORD_TYPES);
#endif /* GRID_1D */
#if defined (GRID_2D)
          ParallelGridCoordinate pos (i, j COORD_TYPES);
#endif /* GRID_2D */
#if defined (GRID_3D)
          ParallelGridCoordinate pos (i, j, k COORD_TYPES);
#endif /* GRID_3D */

          grid_coord coord = calculateIndexFromPosition (pos);
          for (int t = 0; t < gridValues.size (); ++t)
          {
            setFieldValue (buffersReceive[opposite][index++], coord, t);
          }
        }
      }
    }
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
 * Allocate and gather full grid from all nodes to one non-parallel grid on each node
 *
 * @return full grid from all nodes as one non-parallel grid on each node
 *
 * Note: caller has to delete returned grid!
 */
ParallelGridBase *
ParallelGrid::gatherFullGrid () const
{
  ParallelGridBase *grid = new ParallelGridBase (getTotalSize (), gridValues.size (), getName ());

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

  VectorFieldValues values (getTotalSize ().calculateTotalCoord ());

  /*
   * Each computational node broadcasts to all others its data
   */

  for (int process = 0; process < ParallelGrid::getParallelCore ()->getTotalProcCount (); ++process)
  {
    ParallelGridCoordinate chunkStart = getGroupConst ()->getChunkStartPosition ();
    ParallelGridCoordinate chunkEnd = chunkStart + getGroupConst ()->getCurrentSize ();

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

    for (int t = 0; t < gridValues.size (); ++t)
    {
      if (process == ParallelGrid::getParallelCore ()->getProcessId ())
      {
        grid_coord left_coord, right_coord;
        grid_coord down_coord, up_coord;
        grid_coord back_coord, front_coord;

        getGroupConst ()->initBufferOffsets (left_coord, right_coord, down_coord, up_coord, back_coord, front_coord);

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

              values[index] = (*gridValues[t])[coord];

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

        ASSERT (index == sizeCoord.calculateTotalCoord ());
      }

      /*
       * Broadcast data
       */

      MPI_Bcast (values.data (), sizeCoord.calculateTotalCoord (), MPI_FPVALUE, process, ParallelGrid::getParallelCore ()->getCommunicator ());

      grid_coord index = 0;

      /*
       * Store data to corresponding coordinates of the resulting grid
       */

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
      for (grid_coord i = startX; i < endX; ++i)
#endif /* GRID_1D || GRID_2D || GRID_3D */
      {
#if defined (GRID_2D) || defined (GRID_3D)
        for (grid_coord j = startY; j < endY; ++j)
#endif /* GRID_2D || GRID_3D */
        {
#if defined (GRID_3D)
          for (grid_coord k = startZ; k < endZ; ++k)
#endif /* GRID_3D */
          {

#ifdef GRID_1D
            ParallelGridCoordinate pos (i COORD_TYPES);
#endif /* GRID_1D */
#ifdef GRID_2D
            ParallelGridCoordinate pos (i, j COORD_TYPES);
#endif /* GRID_2D */
#ifdef GRID_3D
            ParallelGridCoordinate pos (i, j, k COORD_TYPES);
#endif /* GRID_3D */

            grid_coord coord = grid->calculateIndexFromPosition (pos);

            grid->setFieldValue (values[index], coord, t);

            ++index;
          }
        }
      }
      ASSERT (index == sizeCoord.calculateTotalCoord ());

      MPI_Barrier (ParallelGrid::getParallelCore ()->getCommunicator ());
    }
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
  ParallelGridCoordinate bufferSize = getGroupConst ()->getBufferSize ();

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
