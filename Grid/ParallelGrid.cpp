#include "Grid.h"

extern const char* BufferPositionNames[];

#if defined (PARALLEL_GRID)
void
Grid::SendReceiveCoordinatesInit ()
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  SendReceiveCoordinatesInit1D_X ();
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  SendReceiveCoordinatesInit1D_Y ();
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  SendReceiveCoordinatesInit1D_Z ();
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  SendReceiveCoordinatesInit2D_XY ();
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  SendReceiveCoordinatesInit2D_YZ ();
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  SendReceiveCoordinatesInit2D_XZ ();
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  SendReceiveCoordinatesInit3D_XYZ ();
#endif
}


#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
Grid::SendReceiveCoordinatesInit1D_X ()
{
  sendStart[LEFT] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
#endif
  );

  sendEnd[LEFT] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
#endif
  );

  recvStart[LEFT] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
#endif
  );

  recvEnd[LEFT] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
#endif
  );

  sendStart[RIGHT] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
#endif
  );

  sendEnd[RIGHT] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
#endif
  );

  recvStart[RIGHT] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    0
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
#endif
  );

  recvEnd[RIGHT] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
#endif
  );
}
#endif


#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
Grid::SendReceiveCoordinatesInit1D_Y ()
{
  sendStart[DOWN] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
#endif
  );

  sendEnd[DOWN] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , 2 * bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
#endif
  );

  recvStart[DOWN] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
#endif
  );

  recvEnd[DOWN] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
#endif
  );

  sendStart[UP] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - 2 * bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
#endif
  );

  sendEnd[UP] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
#endif
  );

  recvStart[UP] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , 0
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
#endif
  );

  recvEnd[UP] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
#endif
  );
}
#endif


#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
Grid::SendReceiveCoordinatesInit1D_Z ()
{
  sendStart[BACK] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
#endif
  );

  sendEnd[BACK] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , 2 * bufferSizeLeft.getZ ()
#endif
#endif
#endif
  );

  recvStart[BACK] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
#endif
  );

  recvEnd[BACK] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ ()
#endif
#endif
#endif
  );

  sendStart[FRONT] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif
#endif
#endif
  );

  sendEnd[FRONT] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
#endif
  );

  recvStart[FRONT] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , 0
#endif
#endif
#endif
  );

  recvEnd[FRONT] = GridCoordinate (
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
#if defined (GRID_2D) || defined (GRID_3D)
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
#endif
  );
}
#endif


#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
Grid::SendReceiveCoordinatesInit2D_XY ()
{
  sendStart[LEFT_DOWN] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );

  sendEnd[LEFT_DOWN] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , 2 * bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  recvStart[LEFT_DOWN] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );

  recvEnd[LEFT_DOWN] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , size.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  sendStart[LEFT_UP] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );

  sendEnd[LEFT_UP] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  recvStart[LEFT_UP] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 0
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );

  recvEnd[LEFT_UP] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  sendStart[RIGHT_DOWN] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );

  sendEnd[RIGHT_DOWN] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 2 * bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  recvStart[RIGHT_DOWN] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );

  recvEnd[RIGHT_DOWN] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  sendStart[RIGHT_UP] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );

  sendEnd[RIGHT_UP] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  recvStart[RIGHT_UP] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , 0
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );

  recvEnd[RIGHT_UP] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );
}
#endif


#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
Grid::SendReceiveCoordinatesInit2D_YZ ()
{
  sendStart[DOWN_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );

  sendEnd[DOWN_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 2 * bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , 2 * bufferSizeLeft.getZ ()
#endif
#endif
  );

  recvStart[DOWN_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  recvEnd[DOWN_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY ()
#if defined (GRID_3D)
    , size.getZ ()
#endif
#endif
  );

  sendStart[DOWN_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif
#endif
  );

  sendEnd[DOWN_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 2 * bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  recvStart[DOWN_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , 0
#endif
#endif
  );

  recvEnd[DOWN_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );

  sendStart[UP_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );

  sendEnd[UP_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , 2 * bufferSizeRight.getZ ()
#endif
#endif
  );

  recvStart[UP_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , 0
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  recvEnd[UP_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ ()
#endif
#endif
  );

  sendStart[UP_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif
#endif
  );

  sendEnd[UP_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  recvStart[UP_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , 0
#if defined (GRID_3D)
    , 0
#endif
#endif
  );

  recvEnd[UP_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );
}
#endif


#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
Grid::SendReceiveCoordinatesInit2D_XZ ()
{
  sendStart[LEFT_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );

  sendEnd[LEFT_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , 2 * bufferSizeLeft.getZ ()
#endif
#endif
  );

  recvStart[LEFT_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  recvEnd[LEFT_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ ()
#endif
#endif
  );

  sendStart[LEFT_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif
#endif
  );

  sendEnd[LEFT_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  recvStart[LEFT_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , 0
#endif
#endif
  );

  recvEnd[LEFT_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );

  sendStart[RIGHT_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );

  sendEnd[RIGHT_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , 2 * bufferSizeRight.getZ ()
#endif
#endif
  );

  recvStart[RIGHT_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  recvEnd[RIGHT_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ ()
#endif
#endif
  );

  sendStart[RIGHT_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif
#endif
  );

  sendEnd[RIGHT_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , size.getZ () - bufferSizeRight.getZ ()
#endif
#endif
  );

  recvStart[RIGHT_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , bufferSizeLeft.getY ()
#if defined (GRID_3D)
    , 0
#endif
#endif
  );

  recvEnd[RIGHT_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
#if defined (GRID_3D)
    , bufferSizeLeft.getZ ()
#endif
#endif
  );
}
#endif


#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
void
Grid::SendReceiveCoordinatesInit3D_XYZ ()
{
  sendStart[LEFT_DOWN_BACK] = GridCoordinate (
#if defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
    , bufferSizeLeft.getZ ()
#endif
  );

  sendEnd[LEFT_DOWN_BACK] = GridCoordinate (
#if defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , 2 * bufferSizeLeft.getY ()
    , 2 * bufferSizeLeft.getZ ()
#endif
  );

  recvStart[LEFT_DOWN_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
    , size.getZ () - bufferSizeRight.getZ ()
#endif
  );

  recvEnd[LEFT_DOWN_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , size.getY ()
    , size.getZ ()
#endif
  );

  sendStart[LEFT_DOWN_FRONT] = GridCoordinate (
#if defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif
  );

  sendEnd[LEFT_DOWN_FRONT] = GridCoordinate (
#if defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , 2 * bufferSizeLeft.getY ()
    , size.getZ () - bufferSizeRight.getZ ()
#endif
  );

  recvStart[LEFT_DOWN_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
    , 0
#endif
  );

  recvEnd[LEFT_DOWN_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , size.getY ()
    , bufferSizeLeft.getZ ()
#endif
  );

  sendStart[LEFT_UP_BACK] = GridCoordinate (
#if defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
    , bufferSizeLeft.getZ ()
#endif
  );

  sendEnd[LEFT_UP_BACK] = GridCoordinate (
#if defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
    , 2 * bufferSizeLeft.getZ ()
#endif
  );

  recvStart[LEFT_UP_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 0
    , size.getZ () - bufferSizeRight.getZ ()
#endif
  );

  recvEnd[LEFT_UP_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , bufferSizeLeft.getY ()
    , size.getZ ()
#endif
  );

  sendStart[LEFT_UP_FRONT] = GridCoordinate (
#if defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif
  );

  sendEnd[LEFT_UP_FRONT] = GridCoordinate (
#if defined (GRID_3D)
    2 * bufferSizeLeft.getX ()
    , size.getY () - bufferSizeRight.getY ()
    , size.getZ () - bufferSizeRight.getZ ()
#endif
  );

  recvStart[LEFT_UP_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 0
    , 0
#endif
  );

  recvEnd[LEFT_UP_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    size.getX ()
    , bufferSizeLeft.getY ()
    , bufferSizeLeft.getZ ()
#endif
  );


  sendStart[RIGHT_DOWN_BACK] = GridCoordinate (
#if defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
    , bufferSizeLeft.getZ ()
#endif
  );

  sendEnd[RIGHT_DOWN_BACK] = GridCoordinate (
#if defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 2 * bufferSizeLeft.getY ()
    , 2 * bufferSizeLeft.getZ ()
#endif
  );

  recvStart[RIGHT_DOWN_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , size.getY () - bufferSizeRight.getY ()
    , size.getZ () - bufferSizeRight.getZ ()
#endif
  );

  recvEnd[RIGHT_DOWN_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY ()
    , size.getZ ()
#endif
  );

  sendStart[RIGHT_DOWN_FRONT] = GridCoordinate (
#if defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , bufferSizeLeft.getY ()
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif
  );

  sendEnd[RIGHT_DOWN_FRONT] = GridCoordinate (
#if defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , 2 * bufferSizeLeft.getY ()
    , size.getZ () - bufferSizeRight.getZ ()
#endif
  );

  recvStart[RIGHT_DOWN_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , size.getY () - bufferSizeRight.getY ()
    , 0
#endif
  );

  recvEnd[RIGHT_DOWN_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , size.getY ()
    , bufferSizeLeft.getZ ()
#endif
  );

  sendStart[RIGHT_UP_BACK] = GridCoordinate (
#if defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
    , bufferSizeLeft.getZ ()
#endif
  );

  sendEnd[RIGHT_UP_BACK] = GridCoordinate (
#if defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
    , 2 * bufferSizeLeft.getZ ()
#endif
  );

  recvStart[RIGHT_UP_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , 0
    , size.getZ () - bufferSizeRight.getZ ()
#endif
  );

  recvEnd[RIGHT_UP_BACK] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
    , size.getZ ()
#endif
  );

  sendStart[RIGHT_UP_FRONT] = GridCoordinate (
#if defined (GRID_3D)
    size.getX () - 2 * bufferSizeRight.getX ()
    , size.getY () - 2 * bufferSizeRight.getY ()
    , size.getZ () - 2 * bufferSizeRight.getZ ()
#endif
  );

  sendEnd[RIGHT_UP_FRONT] = GridCoordinate (
#if defined (GRID_3D)
    size.getX () - bufferSizeRight.getX ()
    , size.getY () - bufferSizeRight.getY ()
    , size.getZ () - bufferSizeRight.getZ ()
#endif
  );

  recvStart[RIGHT_UP_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    0
    , 0
    , 0
#endif
  );

  recvEnd[RIGHT_UP_FRONT] = GridCoordinate (
#if defined (GRID_2D) || defined (GRID_3D)
    bufferSizeLeft.getX ()
    , bufferSizeLeft.getY ()
    , bufferSizeLeft.getZ ()
#endif
  );
}
#endif



void
Grid::SendRawBuffer (BufferPosition buffer, int processTo)
{
#if PRINT_MESSAGE
  printf ("Send raw #%d direction %s.\n", processId, BufferPositionNames[buffer]);
#endif
  MPI_Status status;

  FieldValue* rawBuffer = buffersSend[buffer].data ();

#ifdef FLOAT_VALUES
  int retCode = MPI_Send (rawBuffer, buffersSend[buffer].size (), MPI_FLOAT,
                          processTo, processId, MPI_COMM_WORLD);
#endif /* FLOAT_VALUES */
#ifdef DOUBLE_VALUES
  int retCode = MPI_Send (rawBuffer, buffersSend[buffer].size (), MPI_DOUBLE,
                          processTo, processId, MPI_COMM_WORLD);
#endif /* DOUBLE_VALUES */
#ifdef LONG_DOUBLE_VALUES
  int retCode = MPI_Send (rawBuffer, buffersSend[buffer].size (), MPI_LONG_DOUBLE,
                          processTo, processId, MPI_COMM_WORLD);
#endif /* LONG_DOUBLE_VALUES */

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

#ifdef FLOAT_VALUES
  int retCode = MPI_Recv (rawBuffer, buffersReceive[buffer].size (), MPI_FLOAT,
                          processFrom, processFrom, MPI_COMM_WORLD, &status);
#endif /* FLOAT_VALUES */
#ifdef DOUBLE_VALUES
  int retCode = MPI_Recv (rawBuffer, buffersReceive[buffer].size (), MPI_DOUBLE,
                          processFrom, processFrom, MPI_COMM_WORLD, &status);
#endif /* DOUBLE_VALUES */
#ifdef LONG_DOUBLE_VALUES
  int retCode = MPI_Recv (rawBuffer, buffersReceive[buffer].size (), MPI_LONG_DOUBLE,
                          processFrom, processFrom, MPI_COMM_WORLD, &status);
#endif /* LONG_DOUBLE_VALUES */

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

#ifdef FLOAT_VALUES
  int retCode = MPI_Sendrecv (rawBufferSend, buffersSend[bufferSend].size (), MPI_FLOAT,
                              processTo, processId,
                              rawBufferReceive, buffersReceive[bufferReceive].size (), MPI_FLOAT,
                              processFrom, processFrom,
                              MPI_COMM_WORLD, &status);
#endif /* FLOAT_VALUES */
#ifdef DOUBLE_VALUES
  int retCode = MPI_Sendrecv (rawBufferSend, buffersSend[bufferSend].size (), MPI_DOUBLE,
                              processTo, processId,
                              rawBufferReceive, buffersReceive[bufferReceive].size (), MPI_DOUBLE,
                              processFrom, processFrom,
                              MPI_COMM_WORLD, &status);
#endif /* DOUBLE_VALUES */
#ifdef LONG_DOUBLE_VALUES
  int retCode = MPI_Sendrecv (rawBufferSend, buffersSend[bufferSend].size (), MPI_LONG_DOUBLE,
                              processTo, processId,
                              rawBufferReceive, buffersReceive[bufferReceive].size (), MPI_LONG_DOUBLE,
                              processFrom, processFrom,
                              MPI_COMM_WORLD, &status);
#endif /* LONG_DOUBLE_VALUES */


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
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  if (processId < nodeGridSize1 - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (processId % nodeGridSize1 < nodeGridSize1 - 1)
#endif
    c1 = size1 / nodeGridSize1;
  else
    c1 = size1 - (nodeGridSize1 - 1) * (size1 / nodeGridSize1);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  if (processId < nodeGridSize1 * nodeGridSize2 - nodeGridSize1)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if ((processId % (nodeGridSize1 * nodeGridSize2)) < nodeGridSize1 * nodeGridSize2 - nodeGridSize1)
#endif
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

BufferPosition
Grid::getOpposite (BufferPosition direction)
{
  switch (direction)
  {
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT:
      return RIGHT;
    case RIGHT:
      return LEFT;
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case UP:
      return DOWN;
    case DOWN:
      return UP;
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case FRONT:
      return BACK;
    case BACK:
      return FRONT;
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT_UP:
      return RIGHT_DOWN;
    case LEFT_DOWN:
      return RIGHT_UP;
    case RIGHT_UP:
      return LEFT_DOWN;
    case RIGHT_DOWN:
      return LEFT_UP;
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case LEFT_FRONT:
      return RIGHT_BACK;
    case LEFT_BACK:
      return RIGHT_FRONT;
    case RIGHT_FRONT:
      return LEFT_BACK;
    case RIGHT_BACK:
      return LEFT_FRONT;
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    case UP_FRONT:
      return DOWN_BACK;
    case UP_BACK:
      return DOWN_FRONT;
    case DOWN_FRONT:
      return UP_BACK;
    case DOWN_BACK:
      return UP_FRONT;
#endif

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
#endif
    default:
    {
      UNREACHABLE;
    }
  }
}

void
Grid::getShare (BufferPosition direction, std::pair<bool, bool>& pair)
{
  bool doSend = true;
  bool doReceive = true;

  switch (direction)
  {
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
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
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
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
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
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
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
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
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
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
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
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
#endif

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
#endif
    default:
    {
      UNREACHABLE;
    }
  }

  pair.first = doSend;
  pair.second = doReceive;
}

#endif /* PARALLEL_GRID */
