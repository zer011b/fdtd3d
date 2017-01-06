#include "ParallelGrid.h"

#if defined (PARALLEL_GRID)

#if PRINT_MESSAGE
// Names of buffers of parallel grid for debug purposes.
const char* BufferPositionNames[] =
{
#define FUNCTION(X) #X,
#include "BufferPosition.inc.h"
};
#endif /* PRINT_MESSAGE */

ParallelGrid::ParallelGrid (const ParallelGridCoordinate& totSize,
              const ParallelGridCoordinate& bufSizeL, const ParallelGridCoordinate& bufSizeR,
              const int process, const int totalProc, uint32_t step) :
  ParallelGridBase (step),
  totalSize (totSize),
#ifdef GRID_1D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  bufferSizeLeft (bufSizeL),
  bufferSizeRight (bufSizeR),
#endif
#endif
#ifdef GRID_2D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  bufferSizeLeft (bufSizeL.getX (), 0),
  bufferSizeRight (bufSizeR.getX (), 0),
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  bufferSizeLeft (0, bufSizeL.getY ()),
  bufferSizeRight (0, bufSizeR.getY ()),
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  bufferSizeLeft (bufSizeL),
  bufferSizeRight (bufSizeR),
#endif
#endif
#ifdef GRID_3D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  bufferSizeLeft (bufSizeL.getX (), 0, 0),
  bufferSizeRight (bufSizeR.getX (), 0, 0),
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  bufferSizeLeft (0, bufSizeL.getY (), 0),
  bufferSizeRight (0, bufSizeR.getY (), 0),
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  bufferSizeLeft (0, 0, bufSizeL.getZ ()),
  bufferSizeRight (0, 0, bufSizeR.getZ ()),
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  bufferSizeLeft (bufSizeL.getX (), bufSizeL.getY (), 0),
  bufferSizeRight (bufSizeR.getX (), bufSizeR.getY (), 0),
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  bufferSizeLeft (0, bufSizeL.getY (), bufSizeL.getZ ()),
  bufferSizeRight (0, bufSizeR.getY (), bufSizeR.getZ ()),
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  bufferSizeLeft (bufSizeL.getX (), 0, bufSizeL.getZ ()),
  bufferSizeRight (bufSizeR.getX (), 0, bufSizeR.getZ ()),
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  bufferSizeLeft (bufSizeL),
  bufferSizeRight (bufSizeR),
#endif
#endif
  processId (process),
  totalProcCount (totalProc)
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  , hasL (false),
  hasR (false)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  , hasD (false),
  hasU (false)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  , hasB (false),
  hasF (false)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  , shareStep (0)
{
  ASSERT (bufSizeL == bufSizeR);

#if defined (GRID_2D) || defined (GRID_3D)
  ASSERT (bufSizeL.getX () == bufSizeR.getY ());
#endif /* GRID_2D || GRID_3D */
#ifdef GRID_3D
  ASSERT (bufSizeL.getX () == bufSizeR.getZ ());
#endif /* GRID_3D */

#if defined (ONE_TIME_STEP)
  grid_iter numTimeStepsInBuild = 2;
#endif /* ONE_TIME_STEP */
#if defined (TWO_TIME_STEPS)
  grid_iter numTimeStepsInBuild = 3;
#endif /* TWO_TIME_STEPS */

  oppositeDirections.resize (BUFFER_COUNT);
  for (int i = 0; i < BUFFER_COUNT; ++i)
  {
    oppositeDirections[i] = getOpposite ((BufferPosition) i);
  }

  sendStart.resize (BUFFER_COUNT);
  sendEnd.resize (BUFFER_COUNT);
  recvStart.resize (BUFFER_COUNT);
  recvEnd.resize (BUFFER_COUNT);

  directions.resize (BUFFER_COUNT);

  buffersSend.resize (BUFFER_COUNT);
  buffersReceive.resize (BUFFER_COUNT);

  // Call specific constructor.
  ParallelGridConstructor (numTimeStepsInBuild);

  doShare.resize (BUFFER_COUNT);
  for (int i = 0; i < BUFFER_COUNT; ++i)
  {
    getShare ((BufferPosition) i, doShare[i]);
  }

  SendReceiveCoordinatesInit ();

  gridValues.resize (size.calculateTotalCoord ());

#if PRINT_MESSAGE
  printf ("New grid for proc: %d (of %d) with raw size: %lu.\n", process, totalProcCount, gridValues.size ());
#endif /* PRINT_MESSAGE */
}

void
ParallelGrid::SendReceiveCoordinatesInit ()
{
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
}


#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
ParallelGrid::SendReceiveCoordinatesInit1D_X ()
{
  sendStart[LEFT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendEnd[LEFT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvStart[LEFT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvEnd[LEFT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendStart[RIGHT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendEnd[RIGHT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvStart[RIGHT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    0
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvEnd[RIGHT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );
}
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */


#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
ParallelGrid::SendReceiveCoordinatesInit1D_Y ()
{
  sendStart[DOWN] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendEnd[DOWN] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , 2 * bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvStart[DOWN] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvEnd[DOWN] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendStart[UP] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - 2 * bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendEnd[UP] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvStart[UP] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , 0
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvEnd[UP] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );
}
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */


#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
ParallelGrid::SendReceiveCoordinatesInit1D_Z ()
{
  sendStart[BACK] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendEnd[BACK] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , 2 * bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvStart[BACK] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvEnd[BACK] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendStart[FRONT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  sendEnd[FRONT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvStart[FRONT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , 0
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );

  recvEnd[FRONT] = ParallelGridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
#endif /* GRID_1D || GRID_2D || GRID_3D */
  );
}
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */


#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
ParallelGrid::SendReceiveCoordinatesInit2D_XY ()
{
  sendStart[LEFT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[LEFT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , 2 * bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[LEFT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[LEFT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , size.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[LEFT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[LEFT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[LEFT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 0
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[LEFT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[RIGHT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[RIGHT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 2 * bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[RIGHT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[RIGHT_DOWN] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[RIGHT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[RIGHT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[RIGHT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , 0
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[RIGHT_UP] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );
}
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */


#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
ParallelGrid::SendReceiveCoordinatesInit2D_YZ ()
{
  sendStart[DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 2 * bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , 2 * bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY ()
#if defined (GRID_3D)
    , size.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 2 * bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , 0
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , 2 * bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , 0
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , 0
#if defined (GRID_3D)
    , 0
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );
}
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */


#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
ParallelGrid::SendReceiveCoordinatesInit2D_XZ ()
{
  sendStart[LEFT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[LEFT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , 2 * bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[LEFT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[LEFT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[LEFT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[LEFT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[LEFT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , 0
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[LEFT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[RIGHT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[RIGHT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , 2 * bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[RIGHT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[RIGHT_BACK] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendStart[RIGHT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  sendEnd[RIGHT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvStart[RIGHT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , 0
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );

  recvEnd[RIGHT_FRONT] = ParallelGridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
#endif /* GRID_2D || GRID_3D */
  );
}
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */


#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
ParallelGrid::SendReceiveCoordinatesInit3D_XYZ ()
{
  sendStart[LEFT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
  );

  sendEnd[LEFT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , 2 * bufferSizeLeft.getY ()
    , 2 * bufferSizeLeft.getZ ()
#endif /* GRID_3D */
  );

  recvStart[LEFT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
    , size.getZ () - bufferSizeRight.getZ ()
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
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif /* GRID_3D */
  );

  sendEnd[LEFT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , 2 * bufferSizeLeft.getY ()
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
  );

  recvStart[LEFT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
    , 0
#endif /* GRID_3D */
  );

  recvEnd[LEFT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX ()
    , size.getY ()
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
  );

  sendStart[LEFT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
  );

  sendEnd[LEFT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
    , 2 * bufferSizeLeft.getZ ()
#endif /* GRID_3D */
  );

  recvStart[LEFT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 0
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
  );

  recvEnd[LEFT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX ()
    , bufferSizeLeft.getY ()
    , size.getZ ()
#endif /* GRID_3D */
  );

  sendStart[LEFT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif /* GRID_3D */
  );

  sendEnd[LEFT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
  );

  recvStart[LEFT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 0
    , 0
#endif /* GRID_3D */
  );

  recvEnd[LEFT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX ()
    , bufferSizeLeft.getY ()
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
  );


  sendStart[RIGHT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
  );

  sendEnd[RIGHT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 2 * bufferSizeLeft.getY ()
    , 2 * bufferSizeLeft.getZ ()
#endif /* GRID_3D */
  );

  recvStart[RIGHT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    0
    , size.getY () - bufferSizeRight.getY ()
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
  );

  recvEnd[RIGHT_DOWN_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY ()
    , size.getZ ()
#endif /* GRID_3D */
  );

  sendStart[RIGHT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif /* GRID_3D */
  );

  sendEnd[RIGHT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 2 * bufferSizeLeft.getY ()
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
  );

  recvStart[RIGHT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    0
    , size.getY () - bufferSizeRight.getY ()
    , 0
#endif /* GRID_3D */
  );

  recvEnd[RIGHT_DOWN_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY ()
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
  );

  sendStart[RIGHT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
  );

  sendEnd[RIGHT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
    , 2 * bufferSizeLeft.getZ ()
#endif /* GRID_3D */
  );

  recvStart[RIGHT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    0
    , 0
    , size.getZ () - bufferSizeRight.getZ ()
#endif /* GRID_3D */
  );

  recvEnd[RIGHT_UP_BACK] = ParallelGridCoordinate (
#if defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
    , size.getZ ()
#endif /* GRID_3D */
  );

  sendStart[RIGHT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif /* GRID_3D */
  );

  sendEnd[RIGHT_UP_FRONT] = ParallelGridCoordinate (
#if defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
    , size.getZ () - bufferSizeRight.getZ ()
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
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
    , bufferSizeLeft.getZ ()
#endif /* GRID_3D */
  );
}
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

void
ParallelGrid::SendRawBuffer (BufferPosition buffer, int processTo)
{
#if PRINT_MESSAGE
  printf ("\t\tSend RAW. PID=#%d. Direction TO=%s, size=%lu.\n",
    processId, BufferPositionNames[buffer], buffersReceive[buffer].size ());
#endif /* PRINT_MESSAGE */
  MPI_Status status;

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

  int retCode = MPI_Send (rawBuffer, buffersSend[buffer].size (), datatype,
                          processTo, processId, MPI_COMM_WORLD);

  ASSERT (retCode == MPI_SUCCESS);
}

void
ParallelGrid::ReceiveRawBuffer (BufferPosition buffer, int processFrom)
{
#if PRINT_MESSAGE
  printf ("\t\tReceive RAW. PID=#%d. Direction FROM=%s, size=%lu.\n",
    processId, BufferPositionNames[buffer], buffersReceive[buffer].size ());
#endif /* PRINT_MESSAGE */
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

  int retCode = MPI_Recv (rawBuffer, buffersReceive[buffer].size (), datatype,
                          processFrom, processFrom, MPI_COMM_WORLD, &status);

  ASSERT (retCode == MPI_SUCCESS);
}

void
ParallelGrid::SendReceiveRawBuffer (BufferPosition bufferSend, int processTo,
                                    BufferPosition bufferReceive, int processFrom)
{
#if PRINT_MESSAGE
  printf ("\t\tSend/Receive RAW. PID=#%d. Directions TO=%s FROM=%s. Size TO=%lu FROM=%lu.\n",
    processId, BufferPositionNames[bufferSend], BufferPositionNames[bufferReceive],
    buffersReceive[bufferSend].size (), buffersReceive[bufferReceive].size ());
#endif /* PRINT_MESSAGE */
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

  int retCode = MPI_Sendrecv (rawBufferSend, buffersSend[bufferSend].size (), datatype,
                              processTo, processId,
                              rawBufferReceive, buffersReceive[bufferReceive].size (), datatype,
                              processFrom, processFrom,
                              MPI_COMM_WORLD, &status);

  ASSERT (retCode == MPI_SUCCESS);
}

/**
 * Send/receive method to be called for all grid types.
 */
void
ParallelGrid::SendReceive ()
{
// #if PRINT_MESSAGE
//   printf ("Send/Receive %d\n", processId);
// #endif /* PRINT_MESSAGE */

  // Go through all directions and send/receive.
  for (int buf = 0; buf < BUFFER_COUNT; ++buf)
  {
    SendReceiveBuffer ((BufferPosition) buf);
  }
}

void
ParallelGrid::share ()
{
  SendReceive ();

  MPI_Barrier (MPI_COMM_WORLD);
}

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
void
ParallelGrid::CalculateGridSizeForNode (grid_coord& c1, int nodeGridSize1, bool has1, grid_coord size1)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_1D_Z) */
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
void
ParallelGrid::CalculateGridSizeForNode (grid_coord& c1, int nodeGridSize1, bool has1, grid_coord size1,
                                        grid_coord& c2, int nodeGridSize2, bool has2, grid_coord size2)
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ */
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
ParallelGrid::CalculateGridSizeForNode (grid_coord& c1, int nodeGridSize1, bool has1, grid_coord size1,
                                        grid_coord& c2, int nodeGridSize2, bool has2, grid_coord size2,
                                        grid_coord& c3, int nodeGridSize3, bool has3, grid_coord size3)
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (has1)
    c1 = size1 / nodeGridSize1;
  else
    c1 = size1 - (nodeGridSize1 - 1) * (size1 / nodeGridSize1);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_1D_Z) ||
          PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ) ||
          PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (has2)
    c2 = size2 / nodeGridSize2;
  else
    c2 = size2 - (nodeGridSize2 - 1) * (size2 / nodeGridSize2);
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ) ||
          PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (has3)
    c3 = size3 / nodeGridSize3;
  else
    c3 = size3 - (nodeGridSize3 - 1) * (size3 / nodeGridSize3);
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
}

BufferPosition
ParallelGrid::getOpposite (BufferPosition direction)
{
  switch (direction)
  {
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT:
      return RIGHT;
    case RIGHT:
      return LEFT;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case UP:
      return DOWN;
    case DOWN:
      return UP;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case FRONT:
      return BACK;
    case BACK:
      return FRONT;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT_UP:
      return RIGHT_DOWN;
    case LEFT_DOWN:
      return RIGHT_UP;
    case RIGHT_UP:
      return LEFT_DOWN;
    case RIGHT_DOWN:
      return LEFT_UP;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT_FRONT:
      return RIGHT_BACK;
    case LEFT_BACK:
      return RIGHT_FRONT;
    case RIGHT_FRONT:
      return LEFT_BACK;
    case RIGHT_BACK:
      return LEFT_FRONT;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case UP_FRONT:
      return DOWN_BACK;
    case UP_BACK:
      return DOWN_FRONT;
    case DOWN_FRONT:
      return UP_BACK;
    case DOWN_BACK:
      return UP_FRONT;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT_UP_FRONT:
      return RIGHT_DOWN_BACK;
    case LEFT_UP_BACK:
      return RIGHT_DOWN_FRONT;
    case LEFT_DOWN_FRONT:
      return RIGHT_UP_BACK;
    case LEFT_DOWN_BACK:
      return RIGHT_UP_FRONT;
    case RIGHT_UP_FRONT:
      return LEFT_DOWN_BACK;
    case RIGHT_UP_BACK:
      return LEFT_DOWN_FRONT;
    case RIGHT_DOWN_FRONT:
      return LEFT_UP_BACK;
    case RIGHT_DOWN_BACK:
      return LEFT_UP_FRONT;
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
    default:
    {
      UNREACHABLE;
    }
  }

  UNREACHABLE;
  return BUFFER_COUNT;
}

void
ParallelGrid::getShare (BufferPosition direction, std::pair<bool, bool>& pair)
{
  bool doSend = true;
  bool doReceive = true;

  switch (direction)
  {
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT:
    {
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
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case UP:
    {
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
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case FRONT:
    {
      if (!hasB)
      {
        doReceive = false;
      }
      else if (!hasF)
      {
        doSend = false;
      }

      break;
    }
    case BACK:
    {
      if (!hasB)
      {
        doSend = false;
      }
      else if (!hasF)
      {
        doReceive = false;
      }

      break;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT_UP:
    {
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
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT_FRONT:
    {
      if (!hasR || !hasB)
      {
        doReceive = false;
      }
      if (!hasL || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case LEFT_BACK:
    {
      if (!hasL || !hasB)
      {
        doSend = false;
      }
      if (!hasR || !hasF)
      {
        doReceive = false;
      }

      break;
    }
    case RIGHT_FRONT:
    {
      if (!hasL || !hasB)
      {
        doReceive = false;
      }
      if (!hasR || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case RIGHT_BACK:
    {
      if (!hasR || !hasB)
      {
        doSend = false;
      }
      if (!hasL || !hasF)
      {
        doReceive = false;
      }

      break;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case UP_FRONT:
    {
      if (!hasD || !hasB)
      {
        doReceive = false;
      }
      if (!hasU || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case UP_BACK:
    {
      if (!hasU || !hasB)
      {
        doSend = false;
      }
      if (!hasD || !hasF)
      {
        doReceive = false;
      }

      break;
    }
    case DOWN_FRONT:
    {
      if (!hasU || !hasB)
      {
        doReceive = false;
      }
      if (!hasD || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case DOWN_BACK:
    {
      if (!hasD || !hasB)
      {
        doSend = false;
      }
      if (!hasU || !hasF)
      {
        doReceive = false;
      }

      break;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT_UP_FRONT:
    {
      if (!hasR || !hasD || !hasB)
      {
        doReceive = false;
      }
      if (!hasL || !hasU || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case LEFT_UP_BACK:
    {
      if (!hasR || !hasD || !hasF)
      {
        doReceive = false;
      }
      if (!hasL || !hasU || !hasB)
      {
        doSend = false;
      }

      break;
    }
    case LEFT_DOWN_FRONT:
    {
      if (!hasR || !hasU || !hasB)
      {
        doReceive = false;
      }
      if (!hasL || !hasD || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case LEFT_DOWN_BACK:
    {
      if (!hasR || !hasU || !hasF)
      {
        doReceive = false;
      }
      if (!hasL || !hasD || !hasB)
      {
        doSend = false;
      }

      break;
    }
    case RIGHT_UP_FRONT:
    {
      if (!hasL || !hasD || !hasB)
      {
        doReceive = false;
      }
      if (!hasR || !hasU || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case RIGHT_UP_BACK:
    {
      if (!hasL || !hasD || !hasF)
      {
        doReceive = false;
      }
      if (!hasR || !hasU || !hasB)
      {
        doSend = false;
      }

      break;
    }
    case RIGHT_DOWN_FRONT:
    {
      if (!hasL || !hasU || !hasB)
      {
        doReceive = false;
      }
      if (!hasR || !hasD || !hasF)
      {
        doSend = false;
      }

      break;
    }
    case RIGHT_DOWN_BACK:
    {
      if (!hasL || !hasU || !hasF)
      {
        doReceive = false;
      }
      if (!hasR || !hasD || !hasB)
      {
        doSend = false;
      }

      break;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
    default:
    {
      UNREACHABLE;
    }
  }

  pair.first = doSend;
  pair.second = doReceive;
}

void
ParallelGrid::InitDirections ()
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  directions[LEFT] = processId - 1;
  directions[RIGHT] = processId + 1;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  directions[DOWN] = processId - 1;
  directions[UP] = processId + 1;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_YZ */
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  directions[DOWN] = processId - nodeGridSizeX;
  directions[UP] = processId + nodeGridSizeX;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  directions[BACK] = processId - 1;
  directions[FRONT] = processId + 1;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z */
#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  directions[BACK] = processId - nodeGridSizeY;
  directions[FRONT] = processId + nodeGridSizeY;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  directions[BACK] = processId - nodeGridSizeX;
  directions[FRONT] = processId + nodeGridSizeX;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  directions[BACK] = processId - nodeGridSizeXY;
  directions[FRONT] = processId + nodeGridSizeXY;
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  directions[LEFT_DOWN] = processId - nodeGridSizeX - 1;
  directions[LEFT_UP] = processId + nodeGridSizeX - 1;
  directions[RIGHT_DOWN] = processId - nodeGridSizeX + 1;
  directions[RIGHT_UP] = processId + nodeGridSizeX + 1;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  directions[DOWN_BACK] = processId - nodeGridSizeY - 1;
  directions[DOWN_FRONT] = processId + nodeGridSizeY - 1;
  directions[UP_BACK] = processId - nodeGridSizeY + 1;
  directions[UP_FRONT] = processId + nodeGridSizeY + 1;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  directions[DOWN_BACK] = processId - nodeGridSizeXY - nodeGridSizeX;
  directions[DOWN_FRONT] = processId + nodeGridSizeXY - nodeGridSizeX;
  directions[UP_BACK] = processId - nodeGridSizeXY + nodeGridSizeX;
  directions[UP_FRONT] = processId + nodeGridSizeXY + nodeGridSizeX;
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  directions[LEFT_BACK] = processId - nodeGridSizeX - 1;
  directions[LEFT_FRONT] = processId + nodeGridSizeX - 1;
  directions[RIGHT_BACK] = processId - nodeGridSizeX + 1;
  directions[RIGHT_FRONT] = processId + nodeGridSizeX + 1;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  directions[LEFT_BACK] = processId - nodeGridSizeXY - 1;
  directions[LEFT_FRONT] = processId + nodeGridSizeXY - 1;
  directions[RIGHT_BACK] = processId - nodeGridSizeXY + 1;
  directions[RIGHT_FRONT] = processId + nodeGridSizeXY + 1;
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  directions[LEFT_DOWN_BACK] = processId - nodeGridSizeXY - nodeGridSizeX - 1;
  directions[LEFT_DOWN_FRONT] = processId + nodeGridSizeXY - nodeGridSizeX - 1;
  directions[LEFT_UP_BACK] = processId - nodeGridSizeXY + nodeGridSizeX - 1;
  directions[LEFT_UP_FRONT] = processId + nodeGridSizeXY + nodeGridSizeX - 1;
  directions[RIGHT_DOWN_BACK] = processId - nodeGridSizeXY - nodeGridSizeX + 1;
  directions[RIGHT_DOWN_FRONT] = processId + nodeGridSizeXY - nodeGridSizeX + 1;
  directions[RIGHT_UP_BACK] = processId - nodeGridSizeXY + nodeGridSizeX + 1;
  directions[RIGHT_UP_FRONT] = processId + nodeGridSizeXY + nodeGridSizeX + 1;
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
}

void
ParallelGrid::InitBufferFlags ()
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  hasL = false;
  hasR = false;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)
  if (processId > 0)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (processId % nodeGridSizeX > 0)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  {
    hasL = true;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)
  if (processId < nodeGridSizeX - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (processId % nodeGridSizeX < nodeGridSizeX - 1)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  {
    hasR = true;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  hasU = false;
  hasD = false;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
  if (processId > 0)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (processId >= nodeGridSizeX)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (processId % nodeGridSizeY > 0)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if ((processId % (nodeGridSizeXY)) >= nodeGridSizeX)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  {
    hasD = true;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
  if (processId < nodeGridSizeY - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (processId < nodeGridSizeXY - nodeGridSizeX)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (processId % nodeGridSizeY < nodeGridSizeY - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if ((processId % (nodeGridSizeXY)) < nodeGridSizeXY - nodeGridSizeX)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  {
    hasU = true;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  hasF = false;
  hasB = false;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  if (processId > 0)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (processId >= nodeGridSizeY)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  if (processId >= nodeGridSizeX)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (processId >= nodeGridSizeXY)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  {
    hasB = true;
  }

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  if (processId < nodeGridSizeZ - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (processId < nodeGridSizeYZ - nodeGridSizeY)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  if (processId < nodeGridSizeXZ - nodeGridSizeX)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (processId < nodeGridSizeXYZ - nodeGridSizeXY)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  {
    hasF = true;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
}

void
ParallelGrid::InitBuffers (grid_iter numTimeStepsInBuild)
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (hasL)
  {
    int buf_size = bufferSizeLeft.getX () * numTimeStepsInBuild;
#if defined (GRID_2D) || defined (GRID_3D)
    buf_size *= currentSize.getY ();
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
    buf_size *= currentSize.getZ ();
#endif /* GRID_3D */
    buffersSend[LEFT].resize (buf_size);
    buffersReceive[LEFT].resize (buf_size);
  }
  if (hasR)
  {
    int buf_size = bufferSizeRight.getX () * numTimeStepsInBuild;
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
  if (hasD)
  {
    int buf_size = bufferSizeLeft.getY () * currentSize.getX () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.getZ ();
#endif /* GRID_3D */
    buffersSend[DOWN].resize (buf_size);
    buffersReceive[DOWN].resize (buf_size);
  }
  if (hasU)
  {
    int buf_size = bufferSizeRight.getY () * currentSize.getX () * numTimeStepsInBuild;
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
  if (hasB)
  {
    int buf_size = bufferSizeLeft.getZ () * currentSize.getY () * currentSize.getX () * numTimeStepsInBuild;
    buffersSend[BACK].resize (buf_size);
    buffersReceive[BACK].resize (buf_size);
  }
  if (hasF)
  {
    int buf_size = bufferSizeRight.getZ () * currentSize.getY () * currentSize.getX () * numTimeStepsInBuild;
    buffersSend[FRONT].resize (buf_size);
    buffersReceive[FRONT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (hasL && hasD)
  {
    int buf_size = bufferSizeLeft.getX () * bufferSizeLeft.getY () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.getZ ();
#endif /* GRID_3D */
    buffersSend[LEFT_DOWN].resize (buf_size);
    buffersReceive[LEFT_DOWN].resize (buf_size);
  }
  if (hasL && hasU)
  {
    int buf_size = bufferSizeLeft.getX () * bufferSizeRight.getY () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.getZ ();
#endif /* GRID_3D */
    buffersSend[LEFT_UP].resize (buf_size);
    buffersReceive[LEFT_UP].resize (buf_size);
  }
  if (hasR && hasD)
  {
    int buf_size = bufferSizeRight.getX () * bufferSizeLeft.getY () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.getZ ();
#endif /* GRID_3D */
    buffersSend[RIGHT_DOWN].resize (buf_size);
    buffersReceive[RIGHT_DOWN].resize (buf_size);
  }
  if (hasR && hasU)
  {
    int buf_size = bufferSizeRight.getX () * bufferSizeRight.getY () * numTimeStepsInBuild;
#if defined (GRID_3D)
    buf_size *= currentSize.getZ ();
#endif /* GRID_3D */
    buffersSend[RIGHT_UP].resize (buf_size);
    buffersReceive[RIGHT_UP].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (hasD && hasB)
  {
    int buf_size = bufferSizeLeft.getY () * bufferSizeLeft.getZ () * currentSize.getX () * numTimeStepsInBuild;
    buffersSend[DOWN_BACK].resize (buf_size);
    buffersReceive[DOWN_BACK].resize (buf_size);
  }
  if (hasD && hasF)
  {
    int buf_size = bufferSizeLeft.getY () * bufferSizeRight.getZ () * currentSize.getX () * numTimeStepsInBuild;
    buffersSend[DOWN_FRONT].resize (buf_size);
    buffersReceive[DOWN_FRONT].resize (buf_size);
  }
  if (hasU && hasB)
  {
    int buf_size = bufferSizeRight.getY () * bufferSizeLeft.getZ () * currentSize.getX () * numTimeStepsInBuild;
    buffersSend[UP_BACK].resize (buf_size);
    buffersReceive[UP_BACK].resize (buf_size);
  }
  if (hasU && hasF)
  {
    int buf_size = bufferSizeRight.getY () * bufferSizeRight.getZ () * currentSize.getX () * numTimeStepsInBuild;
    buffersSend[UP_FRONT].resize (buf_size);
    buffersReceive[UP_FRONT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (hasL && hasB)
  {
    int buf_size = bufferSizeLeft.getX () * bufferSizeLeft.getY () * currentSize.getY () * numTimeStepsInBuild;
    buffersSend[LEFT_BACK].resize (buf_size);
    buffersReceive[LEFT_BACK].resize (buf_size);
  }
  if (hasL && hasF)
  {
    int buf_size = bufferSizeLeft.getX () * bufferSizeRight.getY () * currentSize.getY () * numTimeStepsInBuild;
    buffersSend[LEFT_FRONT].resize (buf_size);
    buffersReceive[LEFT_FRONT].resize (buf_size);
  }
  if (hasR && hasB)
  {
    int buf_size = bufferSizeRight.getX () * bufferSizeLeft.getY () * currentSize.getY () * numTimeStepsInBuild;
    buffersSend[RIGHT_BACK].resize (buf_size);
    buffersReceive[RIGHT_BACK].resize (buf_size);
  }
  if (hasR && hasF)
  {
    int buf_size = bufferSizeRight.getX () * bufferSizeRight.getY () * currentSize.getY () * numTimeStepsInBuild;
    buffersSend[RIGHT_FRONT].resize (buf_size);
    buffersReceive[RIGHT_FRONT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int buf_size = bufferSizeLeft.getX () * bufferSizeLeft.getY () * bufferSizeLeft.getZ () * numTimeStepsInBuild;
  if (hasL && hasD && hasB)
  {
    buffersSend[LEFT_DOWN_BACK].resize (buf_size);
    buffersReceive[LEFT_DOWN_BACK].resize (buf_size);
  }
  if (hasL && hasD && hasF)
  {
    buffersSend[LEFT_DOWN_FRONT].resize (buf_size);
    buffersReceive[LEFT_DOWN_FRONT].resize (buf_size);
  }
  if (hasL && hasU && hasB)
  {
    buffersSend[LEFT_UP_BACK].resize (buf_size);
    buffersReceive[LEFT_UP_BACK].resize (buf_size);
  }
  if (hasL && hasU && hasF)
  {
    buffersSend[LEFT_UP_FRONT].resize (buf_size);
    buffersReceive[LEFT_UP_FRONT].resize (buf_size);
  }

  if (hasR && hasD && hasB)
  {
    buffersSend[RIGHT_DOWN_BACK].resize (buf_size);
    buffersReceive[RIGHT_DOWN_BACK].resize (buf_size);
  }
  if (hasR && hasD && hasF)
  {
    buffersSend[RIGHT_DOWN_FRONT].resize (buf_size);
    buffersReceive[RIGHT_DOWN_FRONT].resize (buf_size);
  }
  if (hasR && hasU && hasB)
  {
    buffersSend[RIGHT_UP_BACK].resize (buf_size);
    buffersReceive[RIGHT_UP_BACK].resize (buf_size);
  }
  if (hasR && hasU && hasF)
  {
    buffersSend[RIGHT_UP_FRONT].resize (buf_size);
    buffersReceive[RIGHT_UP_FRONT].resize (buf_size);
  }
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
}

void
ParallelGrid::ParallelGridConstructor (grid_iter numTimeStepsInBuild)
{
  NodeGridInit ();

  // Return if node not used.
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  if (processId >= nodeGridSizeXYZ)
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  if (processId >= nodeGridSizeXY)
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  if (processId >= nodeGridSizeYZ)
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  if (processId >= nodeGridSizeXZ)
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */

  InitBufferFlags ();
  currentSize = GridInit ();

  ParallelGridCoordinate bufferSizeLeftCurrent (bufferSizeLeft);
  ParallelGridCoordinate bufferSizeRightCurrent (bufferSizeRight);

// #if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
//     defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
//   if (!hasL)
//   {
//     bufferSizeLeftCurrent.setX (0);
//   }
//   if (!hasR)
//   {
//     bufferSizeRightCurrent.setX (0);
//   }
// #endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
//           PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
//
// #if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
//     defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
//   if (!hasD)
//   {
//     bufferSizeLeftCurrent.setY (0);
//   }
//   if (!hasU)
//   {
//     bufferSizeRightCurrent.setY (0);
//   }
// #endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
//           PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
//
// #if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
//     defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
//   if (!hasB)
//   {
//     bufferSizeLeftCurrent.setZ (0);
//   }
//   if (!hasF)
//   {
//     bufferSizeRightCurrent.setZ (0);
//   }
// #endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
//           PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  size = currentSize + bufferSizeLeftCurrent + bufferSizeRightCurrent;

  InitBuffers (numTimeStepsInBuild);
  InitDirections ();

#if PRINT_MESSAGE
#ifdef GRID_1D
  printf ("Grid size for #%d process: %lu.\n", processId,
    currentSize.getX ());
#endif /* GRID_1D */
#ifdef GRID_2D
  printf ("Grid size for #%d process: %lux%lu.\n", processId,
    currentSize.getX (), currentSize.getY ());
#endif /* GRID_2D */
#ifdef GRID_3D
  printf ("Grid size for #%d process: %lux%lux%lu.\n", processId,
    currentSize.getX (), currentSize.getY (), currentSize.getZ ());
#endif /* GRID_3D */
#endif /* PRINT_MESSAGE */
}

void
ParallelGrid::SendReceiveBuffer (BufferPosition bufferDirection)
{
  // Return if node not used.
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  if (processId >= nodeGridSizeXYZ)
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  if (processId >= nodeGridSizeXY)
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  if (processId >= nodeGridSizeYZ)
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  if (processId >= nodeGridSizeXZ)
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */

  bool doSend = doShare[bufferDirection].first;
  bool doReceive = doShare[bufferDirection].second;

  // Copy to send buffer
  if (doSend)
  {
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    for (grid_iter index = 0, i = sendStart[bufferDirection].getX ();
         i < sendEnd[bufferDirection].getX (); ++i)
    {
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
      for (grid_coord j = sendStart[bufferDirection].getY ();
           j < sendEnd[bufferDirection].getY (); ++j)
      {
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
        for (grid_coord k = sendStart[bufferDirection].getZ ();
             k < sendEnd[bufferDirection].getZ (); ++k)
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

  BufferPosition opposite = oppositeDirections[bufferDirection];
  int processTo = directions[bufferDirection];
  int processFrom = directions[opposite];

#if PRINT_MESSAGE
  printf ("\tSHARE RAW. PID=#%d. Directions TO(%d)=%s=#%d, FROM(%d)=%s=#%d.\n",
    processId, doSend, BufferPositionNames[bufferDirection], processTo,
               doReceive, BufferPositionNames[opposite], processFrom);
#endif /* PRINT_MESSAGE */

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
}

void
ParallelGrid::nextTimeStep ()
{
  ParallelGridBase::nextTimeStep ();

  nextShareStep ();

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  ASSERT (shareStep <= bufferSizeLeft.getX ());

  bool is_share_time = shareStep == bufferSizeLeft.getX ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  ASSERT (shareStep <= bufferSizeLeft.getY ());

  bool is_share_time = shareStep == bufferSizeLeft.getY ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  ASSERT (shareStep <= bufferSizeLeft.getZ ());

  bool is_share_time = shareStep == bufferSizeLeft.getZ ();
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  if (is_share_time)
  {
    share ();

    shareStep = 0;
  }
}

void
ParallelGrid::nextShareStep ()
{
  ++shareStep;
}

void
ParallelGrid::zeroShareStep ()
{
  shareStep = 0;
}

ParallelGridCoordinate
ParallelGrid::getBufferSize () const
{
  return bufferSizeLeft;
}

FieldPointValue*
ParallelGrid::getFieldPointValueAbsoluteIndex (const ParallelGridCoordinate& position)
{
  ASSERT (isLegitIndex (position, totalSize));

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
  grid_iter posDiffX = position.getX ();
#endif
#if defined (GRID_2D) || defined (GRID_3D)
  grid_iter posDiffY = position.getY ();
#endif
#if defined (GRID_3D)
  grid_iter posDiffZ = position.getZ ();
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int pidX = getNodeGridX ();

  grid_iter posNodeStartX;
  grid_iter posNodeEndX;

  posNodeStartX = pidX * currentSize.getX ();

  if (pidX == nodeGridSizeX - 1)
  {
    posNodeEndX = totalSize.getX ();
  }
  else
  {
    posNodeEndX = (pidX + 1) * currentSize.getX ();
  }

  ASSERT (posNodeStartX <= position.getX () && position.getX () < posNodeEndX);

  posDiffX = position.getX () - posNodeStartX;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int pidY = getNodeGridY ();

  grid_iter posNodeStartY;
  grid_iter posNodeEndY;

  posNodeStartY = pidY * currentSize.getY ();

  if (pidY == nodeGridSizeY - 1)
  {
    posNodeEndY = totalSize.getY ();
  }
  else
  {
    posNodeEndY = (pidY + 1) * currentSize.getY ();
  }

  ASSERT (posNodeStartY <= position.getY () && position.getY () < posNodeEndY);

  posDiffY = position.getY () - posNodeStartY;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  int pidZ = getNodeGridZ ();

  grid_iter posNodeStartZ;
  grid_iter posNodeEndZ;

  posNodeStartZ = pidZ * currentSize.getZ ();

  if (pidZ == nodeGridSizeZ - 1)
  {
    posNodeEndZ = totalSize.getZ ();
  }
  else
  {
    posNodeEndZ = (pidZ + 1) * currentSize.getZ ();
  }

  ASSERT (posNodeStartZ <= position.getZ () && position.getZ () < posNodeEndZ);

  posDiffZ = position.getZ () - posNodeStartZ;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (GRID_1D)
  ParallelGridCoordinate pos (posDiffX);
#endif
#if defined (GRID_2D)
  ParallelGridCoordinate pos (posDiffX, posDiffY);
#endif
#if defined (GRID_3D)
  ParallelGridCoordinate pos (posDiffX, posDiffY, posDiffZ);
#endif

  grid_iter coord = calculateIndexFromPosition (pos, currentSize);

  FieldPointValue* value = gridValues[coord];

  ASSERT (value);

  return value;
}

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
int
ParallelGrid::getNodeGridX ()
{
  int pidX;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)

  pidX = processId;

#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  pidX = processId % nodeGridSizeX;

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  return pidX;
}
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
int
ParallelGrid::getNodeGridY ()
{
  int pidY;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y)

  pidY = processId;

#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY)

  pidY = processId / nodeGridSizeX;

#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)

  pidY = processId % nodeGridSizeY;

#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  int pidXY = processId % nodeGridSizeXY;
  pidY = pidXY / nodeGridSizeX;

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  return pidY;
}

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
        PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
int
ParallelGrid::getNodeGridZ ()
{
  int pidZ;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)

  pidZ = processId;

#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)

  pidZ = processId / nodeGridSizeY;

#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)

  pidZ = processId / nodeGridSizeX;

#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  pidZ = processId / nodeGridSizeXY;

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  return pidZ;
}

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

ParallelGridCoordinate
ParallelGrid::getTotalPosition (ParallelGridCoordinate pos)
{
  ParallelGridCoordinate posStart;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  grid_iter posX;
  if (getNodeGridX () == 0)
  {
    posX = getNodeGridX () * currentSize.getX ();
  }
  else
  {
    posX = getNodeGridX () * (currentSize.getX () - bufferSizeLeft.getX ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  grid_iter posY;
  if (getNodeGridY () == 0)
  {
    posY = getNodeGridY () * currentSize.getY ();
  }
  else
  {
    posY = getNodeGridY () * (currentSize.getY () - bufferSizeLeft.getY ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  grid_iter posZ;
  if (getNodeGridZ () == 0)
  {
    posZ = getNodeGridZ () * currentSize.getZ ();
  }
  else
  {
    posZ = getNodeGridZ () * (currentSize.getZ () - bufferSizeLeft.getZ ());
  }
#endif

#ifdef GRID_1D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX);
#endif
#endif

#ifdef GRID_2D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX, 0);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  posStart = ParallelGridCoordinate (0, posY);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  posStart = ParallelGridCoordinate (posX, posY);
#endif
#endif

#ifdef GRID_3D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX, 0, 0);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  posStart = ParallelGridCoordinate (0, posY, 0);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  posStart = ParallelGridCoordinate (0, 0, posZ);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  posStart = ParallelGridCoordinate (posX, posY, 0);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  posStart = ParallelGridCoordinate (0, posY, posZ);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  posStart = ParallelGridCoordinate (posX, 0, posZ);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  posStart = ParallelGridCoordinate (posX, posY, posZ);
#endif
#endif

  return posStart + pos;
}

ParallelGridCoordinate
ParallelGrid::getRelativePosition (ParallelGridCoordinate pos)
{
  ParallelGridCoordinate posStart;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  grid_iter posX;
  if (getNodeGridX () == 0)
  {
    posX = getNodeGridX () * currentSize.getX ();
  }
  else
  {
    posX = getNodeGridX () * (currentSize.getX () - bufferSizeLeft.getX ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  grid_iter posY;
  if (getNodeGridY () == 0)
  {
    posY = getNodeGridY () * currentSize.getY ();
  }
  else
  {
    posY = getNodeGridY () * (currentSize.getY () - bufferSizeLeft.getY ());
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  grid_iter posZ;
  if (getNodeGridZ () == 0)
  {
    posZ = getNodeGridZ () * currentSize.getZ ();
  }
  else
  {
    posZ = getNodeGridZ () * (currentSize.getZ () - bufferSizeLeft.getZ ());
  }
#endif

#ifdef GRID_1D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX);
#endif
#endif

#ifdef GRID_2D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX, 0);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  posStart = ParallelGridCoordinate (0, posY);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  posStart = ParallelGridCoordinate (posX, posY);
#endif
#endif

#ifdef GRID_3D
#ifdef PARALLEL_BUFFER_DIMENSION_1D_X
  posStart = ParallelGridCoordinate (posX, 0, 0);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Y
  posStart = ParallelGridCoordinate (0, posY, 0);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_1D_Z
  posStart = ParallelGridCoordinate (0, 0, posZ);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  posStart = ParallelGridCoordinate (posX, posY, 0);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  posStart = ParallelGridCoordinate (0, posY, posZ);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  posStart = ParallelGridCoordinate (posX, 0, posZ);
#endif
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  posStart = ParallelGridCoordinate (posX, posY, posZ);
#endif
#endif

  ASSERT (pos >= posStart);

  return pos - posStart;
}

#endif /* PARALLEL_GRID */
