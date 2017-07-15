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

/**
 * Initialize parallel grid core
 */
void
ParallelGrid::initializeParallelCore (ParallelGridCore *core) /**< new parallel grid core */
{
  ASSERT (parallelGridCore == NULLPTR);

  parallelGridCore = core;
} /* ParallelGrid::initializeParallelCore */

/**
 * Get parallel grid core
 */
ParallelGridCore *
ParallelGrid::getParallelCore ()
{
  return parallelGridCore;
} /* ParallelGrid::getParallelCore */

ParallelGridCore *ParallelGrid::parallelGridCore = NULLPTR;

/**
 * Parallel grid constructor
 */
ParallelGrid::ParallelGrid (const ParallelGridCoordinate &totSize, /**< total size of grid */
                            const ParallelGridCoordinate &bufSize, /**< buffer size */
                            time_step step, /**< start time step */
                            ParallelGridCoordinate curSize,  /**< size of grid for current node, received from layout */
                            ParallelGridCoordinate coreCurSize, /**< size of grid per node which is same for all nodes
                                                                 *   except the one at the right border
                                                                 *   (coreSizePerNode == sizeForCurNode for all nodes
                                                                 *   except theone at the right border) (is received
                                                                 *   from layout) */
                            const char * name) /**< name of grid */
  : ParallelGridBase (step, name)
  , totalSize (totSize)
  , shareStep (0)
  , bufferSize (ParallelGridCoordinate ())
  , currentSize (curSize)
  , coreCurrentSize (coreCurSize)
{
  /*
   * Check that buffer size is equal for all coordinate axes
   */
  ASSERT (bufSize.getX () != 0);

#if defined (GRID_2D) || defined (GRID_3D)
  ASSERT (bufSize.getX () == bufSize.getY ());
#endif /* GRID_2D || GRID_3D */

#ifdef GRID_3D
  ASSERT (bufSize.getX () == bufSize.getZ ());
#endif /* GRID_3D */

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
  bufferSize = ParallelGridCoordinate (bufSize.getX (), 0);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  bufferSize = ParallelGridCoordinate (0, bufSize.getY ());
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  bufferSize = bufSize;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */
#endif /* GRID_2D */

#ifdef GRID_3D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  bufferSize = ParallelGridCoordinate (bufSize.getX (), 0, 0);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  bufferSize = ParallelGridCoordinate (0, bufSize.getY (), 0);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */

#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  bufferSize = ParallelGridCoordinate (0, 0, bufSize.getZ ());
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  bufferSize = ParallelGridCoordinate (bufSize.getX (), bufSize.getY (), 0);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  bufferSize = ParallelGridCoordinate (0, bufSize.getY (), bufSize.getZ ());
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  bufferSize = ParallelGridCoordinate (bufSize.getX (), 0, bufSize.getZ ());
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

  initializeStartPosition ();
} /* ParallelGrid::ParallelGrid */

/**
 * Initialize absolute start position of chunk for current node
 */
void ParallelGrid::initializeStartPosition ()
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
  defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  grid_iter posX;
  if (parallelGridCore->getNodeGridX () == 0)
  {
    posX = 0;
  }
  else
  {
    posX = parallelGridCore->getNodeGridX () * coreCurrentSize.getX () - bufferSize.getX ();
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
        PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
  defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  grid_iter posY;
  if (parallelGridCore->getNodeGridY () == 0)
  {
    posY = 0;
  }
  else
  {
    posY = parallelGridCore->getNodeGridY () * coreCurrentSize.getY () - bufferSize.getY ();
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
        PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
  defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  grid_iter posZ;
  if (parallelGridCore->getNodeGridZ () == 0)
  {
    posZ = 0;
  }
  else
  {
    posZ = parallelGridCore->getNodeGridZ () * coreCurrentSize.getZ () - bufferSize.getZ ();
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
        PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#ifdef GRID_1D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */
#endif /* GRID_1D */

#ifdef GRID_2D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX, 0);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  posStart = ParallelGridCoordinate (0, posY);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  posStart = ParallelGridCoordinate (posX, posY);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */
#endif /* GRID_2D */

#ifdef GRID_3D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX, 0, 0);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  posStart = ParallelGridCoordinate (0, posY, 0);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  posStart = ParallelGridCoordinate (0, 0, posZ);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  posStart = ParallelGridCoordinate (posX, posY, 0);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  posStart = ParallelGridCoordinate (0, posY, posZ);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  posStart = ParallelGridCoordinate (posX, 0, posZ);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  posStart = ParallelGridCoordinate (posX, posY, posZ);
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
  );

  sendEnd[LEFT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    2 * left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvStart[LEFT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvEnd[LEFT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendStart[RIGHT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendEnd[RIGHT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
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
  );

  recvEnd[RIGHT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
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
  );

  sendEnd[DOWN] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , 2 * down_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvStart[DOWN] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvEnd[DOWN] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY ()
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendStart[UP] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - 2 * up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendEnd[UP] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
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
  );

  recvEnd[UP] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
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
  );

  sendEnd[BACK] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - up_coord
#if defined (GRID_3D)
    , 2 * back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvStart[BACK] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvEnd[BACK] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendStart[FRONT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    left_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , down_coord
#if defined (GRID_3D)
    , size.getZ () - 2 * front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendEnd[FRONT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
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
  );

  recvEnd[FRONT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
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
  );

  sendEnd[LEFT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * left_coord
    , 2 * down_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[LEFT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , size.getY () - up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[LEFT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , size.getY ()
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[LEFT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.getY () - 2 * up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[LEFT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * left_coord
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[LEFT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , 0
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[LEFT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , down_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[RIGHT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * right_coord
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[RIGHT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , 2 * down_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[RIGHT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , size.getY () - up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[RIGHT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.getY ()
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[RIGHT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * right_coord
    , size.getY () - 2 * up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[RIGHT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[RIGHT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , 0
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[RIGHT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , down_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
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
  );

  sendEnd[DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , 2 * down_coord
#if defined (GRID_3D)
    , 2 * back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , size.getY ()
#if defined (GRID_3D)
    , size.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , down_coord
#if defined (GRID_3D)
    , size.getZ () - 2 * front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , 2 * down_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.getY () - up_coord
#if defined (GRID_3D)
    , 0
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , size.getY ()
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.getY () - 2 * up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , size.getY () - up_coord
#if defined (GRID_3D)
    , 2 * back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , 0
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , down_coord
#if defined (GRID_3D)
    , size.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.getY () - 2 * up_coord
#if defined (GRID_3D)
    , size.getZ () - 2 * front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , 0
#if defined (GRID_3D)
    , 0
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
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
  );

  sendEnd[LEFT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * left_coord
    , size.getY () - up_coord
#if defined (GRID_3D)
    , 2 * back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[LEFT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , down_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[LEFT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[LEFT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , down_coord
#if defined (GRID_3D)
    , size.getZ () - 2 * front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[LEFT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * left_coord
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[LEFT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , down_coord
#if defined (GRID_3D)
    , 0
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[LEFT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , size.getY () - up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[RIGHT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * right_coord
    , down_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[RIGHT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , size.getY () - up_coord
#if defined (GRID_3D)
    , 2 * back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[RIGHT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , down_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[RIGHT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[RIGHT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * right_coord
    , down_coord
#if defined (GRID_3D)
    , size.getZ () - 2 * front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[RIGHT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - right_coord
    , size.getY () - up_coord
#if defined (GRID_3D)
    , size.getZ () - front_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[RIGHT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , down_coord
#if defined (GRID_3D)
    , 0
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[RIGHT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    left_coord
    , size.getY () - up_coord
#if defined (GRID_3D)
    , back_coord
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
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
  );

  sendEnd[LEFT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    2 * left_coord
    , 2 * down_coord
    , 2 * back_coord
#endif /* GRID_3D */
  );

  recvStart[LEFT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - right_coord
    , size.getY () - up_coord
    , size.getZ () - front_coord
#endif /* GRID_3D */
  );

  recvEnd[LEFT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX ()
    , size.getY ()
    , size.getZ ()
#endif /* GRID_3D */
  );

  sendStart[LEFT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , down_coord
    , size.getZ () - 2 * front_coord
#endif /* GRID_3D */
  );

  sendEnd[LEFT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    2 * left_coord
    , 2 * down_coord
    , size.getZ () - front_coord
#endif /* GRID_3D */
  );

  recvStart[LEFT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - right_coord
    , size.getY () - up_coord
    , 0
#endif /* GRID_3D */
  );

  recvEnd[LEFT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX ()
    , size.getY ()
    , back_coord
#endif /* GRID_3D */
  );

  sendStart[LEFT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , size.getY () - 2 * up_coord
    , back_coord
#endif /* GRID_3D */
  );

  sendEnd[LEFT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    2 * left_coord
    , size.getY () - up_coord
    , 2 * back_coord
#endif /* GRID_3D */
  );

  recvStart[LEFT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - right_coord
    , 0
    , size.getZ () - front_coord
#endif /* GRID_3D */
  );

  recvEnd[LEFT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX ()
    , down_coord
    , size.getZ ()
#endif /* GRID_3D */
  );

  sendStart[LEFT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , size.getY () - 2 * up_coord
    , size.getZ () - 2 * front_coord
#endif /* GRID_3D */
  );

  sendEnd[LEFT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    2 * left_coord
    , size.getY () - up_coord
    , size.getZ () - front_coord
#endif /* GRID_3D */
  );

  recvStart[LEFT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - right_coord
    , 0
    , 0
#endif /* GRID_3D */
  );

  recvEnd[LEFT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX ()
    , down_coord
    , back_coord
#endif /* GRID_3D */
  );


  sendStart[RIGHT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - 2 * right_coord
    , down_coord
    , back_coord
#endif /* GRID_3D */
  );

  sendEnd[RIGHT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - right_coord
    , 2 * down_coord
    , 2 * back_coord
#endif /* GRID_3D */
  );

  recvStart[RIGHT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    0
    , size.getY () - up_coord
    , size.getZ () - front_coord
#endif /* GRID_3D */
  );

  recvEnd[RIGHT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , size.getY ()
    , size.getZ ()
#endif /* GRID_3D */
  );

  sendStart[RIGHT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - 2 * right_coord
    , down_coord
    , size.getZ () - 2 * front_coord
#endif /* GRID_3D */
  );

  sendEnd[RIGHT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - right_coord
    , 2 * down_coord
    , size.getZ () - front_coord
#endif /* GRID_3D */
  );

  recvStart[RIGHT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    0
    , size.getY () - up_coord
    , 0
#endif /* GRID_3D */
  );

  recvEnd[RIGHT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , size.getY ()
    , back_coord
#endif /* GRID_3D */
  );

  sendStart[RIGHT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - 2 * right_coord
    , size.getY () - 2 * up_coord
    , back_coord
#endif /* GRID_3D */
  );

  sendEnd[RIGHT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - right_coord
    , size.getY () - up_coord
    , 2 * back_coord
#endif /* GRID_3D */
  );

  recvStart[RIGHT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    0
    , 0
    , size.getZ () - front_coord
#endif /* GRID_3D */
  );

  recvEnd[RIGHT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , down_coord
    , size.getZ ()
#endif /* GRID_3D */
  );

  sendStart[RIGHT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - 2 * right_coord
    , size.getY () - 2 * up_coord
    , size.getZ () - 2 * front_coord
#endif /* GRID_3D */
  );

  sendEnd[RIGHT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - right_coord
    , size.getY () - up_coord
    , size.getZ () - front_coord
#endif /* GRID_3D */
  );

  recvStart[RIGHT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    0
    , 0
    , 0
#endif /* GRID_3D */
  );

  recvEnd[RIGHT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    left_coord
    , down_coord
    , back_coord
#endif /* GRID_3D */
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
                          MPI_COMM_WORLD);

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
                          MPI_COMM_WORLD,
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
                              MPI_COMM_WORLD,
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

  bool doSend = parallelGridCore->getDoShare ()[bufferDirection].first;
  bool doReceive = parallelGridCore->getDoShare ()[bufferDirection].second;

  /*
   * Copy to send buffer
   */
  if (doSend)
  {
    grid_iter index = 0;

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    for (grid_coord i = sendStart[bufferDirection].getX ();
         i < sendEnd[bufferDirection].getX ();
         ++i)
    {
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
      for (grid_coord j = sendStart[bufferDirection].getY ();
           j < sendEnd[bufferDirection].getY ();
           ++j)
      {
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
        for (grid_coord k = sendStart[bufferDirection].getZ ();
             k < sendEnd[bufferDirection].getZ ();
             ++k)
        {
#endif /* GRID_3D */

#if defined (GRID_1D)
          ParallelGridCoordinate pos (i);
#endif /* GRID_1D */
#if defined (GRID_2D)
          ParallelGridCoordinate pos (i, j);
#endif /* GRID_2D */
#if defined (GRID_3D)
          ParallelGridCoordinate pos (i, j, k);
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

  BufferPosition opposite = parallelGridCore->getOppositeDirections ()[bufferDirection];

  int processTo = parallelGridCore->getDirections ()[bufferDirection];
  int processFrom = parallelGridCore->getDirections ()[opposite];

  DPRINTF (LOG_LEVEL_FULL, "\tSHARE RAW. PID=#%d. Directions TO(%d)=%s=#%d, FROM(%d)=%s=#%d.\n",
           parallelGridCore->getProcessId (),
           doSend,
           BufferPositionNames[bufferDirection],
           processTo,
           doReceive,
           BufferPositionNames[opposite],
           processFrom);

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
    /*
     * Do nothing
     */
  }

  /*
   * Copy from receive buffer
   */
  if (doReceive)
  {
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    for (grid_iter index = 0, i = recvStart[bufferDirection].getX ();
         i < recvEnd[bufferDirection].getX (); ++i)
    {
#endif /* GRID_1D || GRID_2D || GRID_3D */

#if defined (GRID_2D) || defined (GRID_3D)
      for (grid_coord j = recvStart[bufferDirection].getY ();
           j < recvEnd[bufferDirection].getY (); ++j)
      {
#endif /* GRID_2D || GRID_3D */

#if defined (GRID_3D)
        for (grid_coord k = recvStart[bufferDirection].getZ ();
             k < recvEnd[bufferDirection].getZ (); ++k)
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
          ParallelGridCoordinate pos (i);
#endif /* GRID_1D */

#if defined (GRID_2D)
          ParallelGridCoordinate pos (i, j);
#endif /* GRID_2D */

#if defined (GRID_3D)
          ParallelGridCoordinate pos (i, j, k);
#endif /* GRID_3D */

          setFieldPointValue (val, ParallelGridCoordinate (pos));

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

  MPI_Barrier (MPI_COMM_WORLD);
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
  const grid_iter numTimeStepsInBuild = 2;
#endif /* ONE_TIME_STEP */
#if defined (TWO_TIME_STEPS)
  const grid_iter numTimeStepsInBuild = 3;
#endif /* TWO_TIME_STEPS */

  buffersSend.resize (BUFFER_COUNT);
  buffersReceive.resize (BUFFER_COUNT);

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasL ())
  {
    int buf_size = bufferSize.getX () * numTimeStepsInBuild;
#if defined (GRID_2D) || defined (GRID_3D)
    buf_size *= currentSize.getY ();
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
    buf_size *= currentSize.getZ ();
#endif /* GRID_3D */
    buffersSend[LEFT].resize (buf_size);
    buffersReceive[LEFT].resize (buf_size);
  }
  if (parallelGridCore->getHasR ())
  {
    int buf_size = bufferSize.getX () * numTimeStepsInBuild;
#if defined (GRID_2D) || defined (GRID_3D)
    buf_size *= currentSize.getY ();
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
    buf_size *= currentSize.getZ ();
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
    int buf_size = bufferSize.getY () * currentSize.getX () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.getZ ();
#endif /* GRID_3D */
    buffersSend[DOWN].resize (buf_size);
    buffersReceive[DOWN].resize (buf_size);
  }
  if (parallelGridCore->getHasU ())
  {
    int buf_size = bufferSize.getY () * currentSize.getX () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.getZ ();
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
    int buf_size = bufferSize.getZ () * currentSize.getY () * currentSize.getX () * numTimeStepsInBuild;
    buffersSend[BACK].resize (buf_size);
    buffersReceive[BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.getZ () * currentSize.getY () * currentSize.getX () * numTimeStepsInBuild;
    buffersSend[FRONT].resize (buf_size);
    buffersReceive[FRONT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasL () && parallelGridCore->getHasD ())
  {
    int buf_size = bufferSize.getX () * bufferSize.getY () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.getZ ();
#endif /* GRID_3D */
    buffersSend[LEFT_DOWN].resize (buf_size);
    buffersReceive[LEFT_DOWN].resize (buf_size);
  }
  if (parallelGridCore->getHasL () && parallelGridCore->getHasU ())
  {
    int buf_size = bufferSize.getX () * bufferSize.getY () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.getZ ();
#endif /* GRID_3D */
    buffersSend[LEFT_UP].resize (buf_size);
    buffersReceive[LEFT_UP].resize (buf_size);
  }
  if (parallelGridCore->getHasR () && parallelGridCore->getHasD ())
  {
    int buf_size = bufferSize.getX () * bufferSize.getY () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.getZ ();
#endif /* GRID_3D */
    buffersSend[RIGHT_DOWN].resize (buf_size);
    buffersReceive[RIGHT_DOWN].resize (buf_size);
  }
  if (parallelGridCore->getHasR () && parallelGridCore->getHasU ())
  {
    int buf_size = bufferSize.getX () * bufferSize.getY () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.getZ ();
#endif /* GRID_3D */
    buffersSend[RIGHT_UP].resize (buf_size);
    buffersReceive[RIGHT_UP].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasD () && parallelGridCore->getHasB ())
  {
    int buf_size = bufferSize.getY () * bufferSize.getZ () * currentSize.getX () * numTimeStepsInBuild;
    buffersSend[DOWN_BACK].resize (buf_size);
    buffersReceive[DOWN_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasD () && parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.getY () * bufferSize.getZ () * currentSize.getX () * numTimeStepsInBuild;
    buffersSend[DOWN_FRONT].resize (buf_size);
    buffersReceive[DOWN_FRONT].resize (buf_size);
  }
  if (parallelGridCore->getHasU () && parallelGridCore->getHasB ())
  {
    int buf_size = bufferSize.getY () * bufferSize.getZ () * currentSize.getX () * numTimeStepsInBuild;
    buffersSend[UP_BACK].resize (buf_size);
    buffersReceive[UP_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasU () && parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.getY () * bufferSize.getZ () * currentSize.getX () * numTimeStepsInBuild;
    buffersSend[UP_FRONT].resize (buf_size);
    buffersReceive[UP_FRONT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasL () && parallelGridCore->getHasB ())
  {
    int buf_size = bufferSize.getX () * bufferSize.getZ () * currentSize.getY () * numTimeStepsInBuild;
    buffersSend[LEFT_BACK].resize (buf_size);
    buffersReceive[LEFT_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasL () && parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.getX () * bufferSize.getZ () * currentSize.getY () * numTimeStepsInBuild;
    buffersSend[LEFT_FRONT].resize (buf_size);
    buffersReceive[LEFT_FRONT].resize (buf_size);
  }
  if (parallelGridCore->getHasR () && parallelGridCore->getHasB ())
  {
    int buf_size = bufferSize.getX () * bufferSize.getZ () * currentSize.getY () * numTimeStepsInBuild;
    buffersSend[RIGHT_BACK].resize (buf_size);
    buffersReceive[RIGHT_BACK].resize (buf_size);
  }
  if (parallelGridCore->getHasR () && parallelGridCore->getHasF ())
  {
    int buf_size = bufferSize.getX () * bufferSize.getZ () * currentSize.getY () * numTimeStepsInBuild;
    buffersSend[RIGHT_FRONT].resize (buf_size);
    buffersReceive[RIGHT_FRONT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int buf_size = bufferSize.getX () * bufferSize.getY () * bufferSize.getZ () * numTimeStepsInBuild;
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
  ParallelGridCoordinate bufLeft (left_coord);
  ParallelGridCoordinate bufRight (right_coord);
#endif /* GRID_1D */
#ifdef GRID_2D
  ParallelGridCoordinate bufLeft (left_coord, down_coord);
  ParallelGridCoordinate bufRight (right_coord, up_coord);
#endif /* GRID_2D */
#ifdef GRID_3D
  ParallelGridCoordinate bufLeft (left_coord, down_coord, back_coord);
  ParallelGridCoordinate bufRight (right_coord, up_coord, front_coord);
#endif /* GRID_3D */

  size = currentSize + bufLeft + bufRight;

  /*
   * Init parallel buffers
   */
  InitBuffers ();

  SendReceiveCoordinatesInit ();

#ifdef GRID_1D
  DPRINTF (LOG_LEVEL_FULL, "Grid size for #%d process: %lu.\n",
          parallelGridCore->getProcessId (),
          currentSize.getX ());
#endif /* GRID_1D */

#ifdef GRID_2D
  DPRINTF (LOG_LEVEL_FULL, "Grid size for #%d process: %lux%lu.\n",
          parallelGridCore->getProcessId (),
          currentSize.getX (),
          currentSize.getY ());
#endif /* GRID_2D */

#ifdef GRID_3D
  DPRINTF (LOG_LEVEL_FULL, "Grid size for #%d process: %lux%lux%lu.\n",
          parallelGridCore->getProcessId (),
          currentSize.getX (),
          currentSize.getY (),
          currentSize.getZ ());
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
  ASSERT (shareStep <= bufferSize.getX ());

  bool is_share_time = shareStep == bufferSize.getX ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  ASSERT (shareStep <= bufferSize.getY ());

  bool is_share_time = shareStep == bufferSize.getY ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  ASSERT (shareStep <= bufferSize.getZ ());

  bool is_share_time = shareStep == bufferSize.getZ ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  if (is_share_time)
  {
    share ();
    zeroShareStep ();
  }
} /* ParallelGrid::nextTimeStep */

/**
 * Increase share step
 */
void
ParallelGrid::nextShareStep ()
{
  ++shareStep;
} /* ParallelGrid::nextShareStep */

/**
 * Set share step to zero
 */
void
ParallelGrid::zeroShareStep ()
{
  shareStep = 0;
} /* ParallelGrid::zeroShareStep */

/**
 * Get absolute position corresponding to first value in grid for current computational node (considering buffers)
 *
 * @return absolute position corresponding to first value in grid for current computational node (considering buffers)
 */
ParallelGridCoordinate
ParallelGrid::getStartPosition () const
{
  return posStart;
} /* ParallelGrid::getStartPosition */

/**
 * Get absolute position corresponding to first value in grid for current computational node (not considering buffers)
 *
 * @return absolute position corresponding to first value in grid for current computational node (not considering buffers)
 */
ParallelGridCoordinate
ParallelGrid::getChunkStartPosition () const
{
  ParallelGridCoordinate posStart;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  grid_iter posX = parallelGridCore->getNodeGridX () * coreCurrentSize.getX ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  grid_iter posY = parallelGridCore->getNodeGridY () * coreCurrentSize.getY ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  grid_iter posZ = parallelGridCore->getNodeGridZ () * coreCurrentSize.getZ ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#ifdef GRID_1D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */
#endif /* GRID_1D */

#ifdef GRID_2D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX, 0);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  posStart = ParallelGridCoordinate (0, posY);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  posStart = ParallelGridCoordinate (posX, posY);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */
#endif /* GRID_2D */

#ifdef GRID_3D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX, 0, 0);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X */
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  posStart = ParallelGridCoordinate (0, posY, 0);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y */
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  posStart = ParallelGridCoordinate (0, 0, posZ);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  posStart = ParallelGridCoordinate (posX, posY, 0);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  posStart = ParallelGridCoordinate (0, posY, posZ);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  posStart = ParallelGridCoordinate (posX, 0, posZ);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  posStart = ParallelGridCoordinate (posX, posY, posZ);
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
#endif /* GRID_3D */

  return posStart;
} /* ParallelGrid::getChunkStartPosition */

/**
 * Get total position in grid from relative position for current computational node
 *
 * @return total position in grid from relative position for current computational node
 */
ParallelGridCoordinate
ParallelGrid::getTotalPosition (ParallelGridCoordinate pos) /**< relative position for current computational node */
{
  ParallelGridCoordinate posStart = getStartPosition ();

  return posStart + pos;
} /* ParallelGrid::getTotalPosition */

/**
 * Get relative position for current computational node from total position
 *
 * @return relative position for current computational node from total position
 */
ParallelGridCoordinate
ParallelGrid::getRelativePosition (ParallelGridCoordinate pos) /**< total position in grid */
{
  ParallelGridCoordinate posStart = getStartPosition ();

  ASSERT (pos >= posStart);

  return pos - posStart;
} /* ParallelGrid::getRelativePosition */

/**
 * Get field point value at absolute coordinate in grid
 *
 * @return field point value
 */
FieldPointValue *
ParallelGrid::getFieldPointValueByAbsolutePos (const ParallelGridCoordinate &absPosition) /**< absolute coordinate in grid */
{
  return getFieldPointValue (getRelativePosition (absPosition));
} /* ParallelGrid::getFieldPointValueByAbsolutePos */

/**
 * Get field point value at absolute coordinate in grid. If current node does not contain this coordinate, return NULLPTR
 *
 * @return field point value or NULLPTR
 */
FieldPointValue *
ParallelGrid::getFieldPointValueOrNullByAbsolutePos (const ParallelGridCoordinate &absPosition) /**< absolute coordinate in grid */
{
  ParallelGridCoordinate posStart = getStartPosition ();
  ParallelGridCoordinate posEnd = posStart + getSize ();

  if (!(absPosition >= posStart)
      || !(absPosition < posEnd))
  {
    return NULLPTR;
  }

  ParallelGridCoordinate relPosition = getRelativePosition (absPosition);

  return getFieldPointValue (relPosition);
} /* ParallelGrid::getFieldPointValueOrNullByAbsolutePos */

/**
 * Get first coordinate from which to perform computations at current step
 *
 * @return first coordinate from which to perform computations at current step
 */
ParallelGridCoordinate
ParallelGrid::getComputationStart (ParallelGridCoordinate diffPosStart) const /**< layout coordinate modifier */
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_iter diffX = diffPosStart.getX ();
#endif
#if defined (GRID_2D) || defined (GRID_3D)
  grid_iter diffY = diffPosStart.getY ();
#endif
#if defined (GRID_3D)
  grid_iter diffZ = diffPosStart.getZ ();
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasL ())
  {
    diffX = shareStep + 1;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasD ())
  {
    diffY = shareStep + 1;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasB ())
  {
    diffZ = shareStep + 1;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_iter px = diffX;
#endif
#if defined (GRID_2D) || defined (GRID_3D)
  grid_iter py = diffY;
#endif
#if defined (GRID_3D)
  grid_iter pz = diffZ;
#endif

#ifdef GRID_1D
  return ParallelGridCoordinate (px);
#endif /* GRID_1D */
#ifdef GRID_2D
  return ParallelGridCoordinate (px, py);
#endif /* GRID_2D */
#ifdef GRID_3D
  return ParallelGridCoordinate (px, py, pz);
#endif /* GRID_3D */
} /* ParallelGrid::getComputationStart */

/**
 * Get last coordinate until which to perform computations at current step
 *
 * @return last coordinate until which to perform computations at current step
 */
ParallelGridCoordinate
ParallelGrid::getComputationEnd (ParallelGridCoordinate diffPosEnd) const /**< layout coordinate modifier */
{
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_iter diffX = diffPosEnd.getX ();
#endif
#if defined (GRID_2D) || defined (GRID_3D)
  grid_iter diffY = diffPosEnd.getY ();
#endif
#if defined (GRID_3D)
  grid_iter diffZ = diffPosEnd.getZ ();
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasR ())
  {
    diffX = shareStep + 1;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasU ())
  {
    diffY = shareStep + 1;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (parallelGridCore->getHasF ())
  {
    diffZ = shareStep + 1;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_iter px = ParallelGridBase::getSize ().getX () - diffX;
#endif
#if defined (GRID_2D) || defined (GRID_3D)
  grid_iter py = ParallelGridBase::getSize ().getY () - diffY;
#endif
#ifdef GRID_3D
  grid_iter pz = ParallelGridBase::getSize ().getZ () - diffZ;
#endif

#ifdef GRID_1D
  return ParallelGridCoordinate (px);
#endif
#ifdef GRID_2D
  return ParallelGridCoordinate (px, py);
#endif
#ifdef GRID_3D
  return ParallelGridCoordinate (px, py, pz);
#endif
} /* ParallelGrid::getComputationEnd */

/**
 * Initialize buffer offsets for computational node
 */
void
ParallelGrid::initBufferOffsets (grid_coord &left_coord, /**< out: left buffer size */
                                 grid_coord &right_coord, /**< out: right buffer size */
                                 grid_coord &down_coord, /**< out: down buffer size */
                                 grid_coord &up_coord, /**< out: up buffer size */
                                 grid_coord &back_coord, /**< out: back buffer size */
                                 grid_coord &front_coord) const /**< out: front buffer size */
{
  left_coord = 0;
  right_coord = 0;
  down_coord = 0;
  up_coord = 0;
  back_coord = 0;
  front_coord = 0;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  left_coord = 0;
  if (parallelGridCore->getHasL ())
  {
    left_coord = bufferSize.getX ();
  }
  else
  {
    ASSERT (left_coord == 0);
  }

  right_coord = 0;
  if (parallelGridCore->getHasR ())
  {
    right_coord = bufferSize.getX ();
  }
  else
  {
    ASSERT (right_coord == 0);
  }
#else /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
         PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  left_coord = 0;
  right_coord = 0;
#endif /* !PARALLEL_BUFFER_DIMENSION_1D_X && !PARALLEL_BUFFER_DIMENSION_2D_XY &&
          !PARALLEL_BUFFER_DIMENSION_2D_XZ && !PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  down_coord = 0;
  if (parallelGridCore->getHasD ())
  {
    down_coord = bufferSize.getY ();
  }
  else
  {
    ASSERT (down_coord == 0);
  }

  up_coord = 0;
  if (parallelGridCore->getHasU ())
  {
    up_coord = bufferSize.getY ();
  }
  else
  {
    ASSERT (up_coord == 0);
  }
#else /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
         PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  down_coord = 0;
  up_coord = 0;
#endif /* !PARALLEL_BUFFER_DIMENSION_1D_Y && !PARALLEL_BUFFER_DIMENSION_2D_XY &&
          !PARALLEL_BUFFER_DIMENSION_2D_YZ && !PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  back_coord = 0;
  if (parallelGridCore->getHasB ())
  {
    back_coord = bufferSize.getZ ();
  }
  else
  {
    ASSERT (back_coord == 0);
  }

  front_coord = 0;
  if (parallelGridCore->getHasF ())
  {
    front_coord = bufferSize.getZ ();
  }
  else
  {
    ASSERT (front_coord == 0);
  }
#else /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
         PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  back_coord = 0;
  front_coord = 0;
#endif /* !PARALLEL_BUFFER_DIMENSION_1D_Y && !PARALLEL_BUFFER_DIMENSION_2D_XY &&
          !PARALLEL_BUFFER_DIMENSION_2D_YZ && !PARALLEL_BUFFER_DIMENSION_3D_XYZ */
} /* ParallelGridCore::initBufferOffsets */

/**
 * Gather full grid from all nodes to one non-parallel grid on each node
 *
 * @full grid from all nodes as one non-parallel grid on each node
 */
ParallelGridBase
ParallelGrid::gatherFullGrid () const
{
  ParallelGridBase grid (totalSize, ParallelGridBase::timeStep);

  /*
   * Fill new grid with values
   */
  for (grid_iter iter = 0; iter < grid.getSize ().calculateTotalCoord (); ++iter)
  {
    FieldPointValue *val = new FieldPointValue ();

    grid.setFieldPointValue (val, grid.calculatePositionFromIndex (iter));
  }

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
      startX = chunkStart.getX ();
      endX = chunkEnd.getX ();
    }

    MPI_Bcast (&startX, 1, MPI_UNSIGNED, process, MPI_COMM_WORLD);
    MPI_Bcast (&endX, 1, MPI_UNSIGNED, process, MPI_COMM_WORLD);
#endif /* GRID_1D || GRID_2D || GRID_3D */

#if defined (GRID_2D) || defined (GRID_3D)
    grid_coord startY;
    grid_coord endY;

    if (process == ParallelGrid::getParallelCore ()->getProcessId ())
    {
      startY = chunkStart.getY ();
      endY = chunkEnd.getY ();
    }

    MPI_Bcast (&startY, 1, MPI_UNSIGNED, process, MPI_COMM_WORLD);
    MPI_Bcast (&endY, 1, MPI_UNSIGNED, process, MPI_COMM_WORLD);
#endif /* GRID_2D || GRID_3D */

#if defined (GRID_3D)
    grid_coord startZ;
    grid_coord endZ;

    if (process == ParallelGrid::getParallelCore ()->getProcessId ())
    {
      startZ = chunkStart.getZ ();
      endZ = chunkEnd.getZ ();
    }

    MPI_Bcast (&startZ, 1, MPI_UNSIGNED, process, MPI_COMM_WORLD);
    MPI_Bcast (&endZ, 1, MPI_UNSIGNED, process, MPI_COMM_WORLD);
#endif /* GRID_3D */

#ifdef GRID_1D
    ParallelGridCoordinate sizeCoord (endX - startX);
#endif /* GRID_1D */
#ifdef GRID_2D
    ParallelGridCoordinate sizeCoord (endX - startX, endY - startY);
#endif /* GRID_2D */
#ifdef GRID_3D
    ParallelGridCoordinate sizeCoord (endX - startX, endY - startY, endZ - startZ);
#endif /* GRID_3D */

    /*
     * Fill vectors with data for current computational node
     */
    grid_iter size = sizeCoord.calculateTotalCoord ();
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

      grid_iter index = 0;

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
      for (grid_coord i = left_coord; i < left_coord + sizeCoord.getX (); ++i)
      {
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
        for (grid_coord j = down_coord; j < down_coord + sizeCoord.getY (); ++j)
        {
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
          for (grid_coord k = back_coord; k < back_coord + sizeCoord.getZ (); ++k)
          {
#endif /* GRID_3D */

#ifdef GRID_1D
            ParallelGridCoordinate pos (i);
#endif /* GRID_1D */
#ifdef GRID_2D
            ParallelGridCoordinate pos (i, j);
#endif /* GRID_2D */
#ifdef GRID_3D
            ParallelGridCoordinate pos (i, j, k);
#endif /* GRID_3D */

            grid_iter coord = calculateIndexFromPosition (pos);

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

    MPI_Bcast (current.data (), current.size (), datatype, process, MPI_COMM_WORLD);

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    MPI_Bcast (previous.data (), previous.size (), datatype, process, MPI_COMM_WORLD);
#endif /* ONE_TIME_STEP || TWO_TIME_STEPS */

#if defined (TWO_TIME_STEPS)
    MPI_Bcast (previousPrev.data (), previousPrev.size (), datatype, process, MPI_COMM_WORLD);
#endif /* TWO_TIME_STEPS */

    grid_iter index = 0;

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
          ParallelGridCoordinate pos (i);
#endif /* GRID_1D */
#ifdef GRID_2D
          ParallelGridCoordinate pos (i, j);
#endif /* GRID_2D */
#ifdef GRID_3D
          ParallelGridCoordinate pos (i, j, k);
#endif /* GRID_3D */

          FieldPointValue *val = grid.getFieldPointValue (pos);

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

    MPI_Barrier (MPI_COMM_WORLD);
  }

  return grid;
} /* ParallelGrid::gatherFullGrid */

#endif /* PARALLEL_GRID */
