#include "ParallelGridCore.h"

#ifdef PARALLEL_GRID

#if PRINT_MESSAGE
extern const char* BufferPositionNames[];
#endif /* PRINT_MESSAGE */

/**
 * Initialize vector with opposite directions
 *
 * @return vector with opposite directions
 */
void
ParallelGridCore::initOppositeDirections ()
{
  oppositeDirections.resize (BUFFER_COUNT);
  for (int i = 0; i < BUFFER_COUNT; ++i)
  {
    oppositeDirections[i] = getOpposite ((BufferPosition) i);
  }
} /* ParallelGridCore::initOppositeDirections */

/**
 * Get opposite direction for specified direction
 *
 * @return opposite direction for specified direction
 */
BufferPosition
ParallelGridCore::getOpposite (BufferPosition direction) /**< direction to get opposite for */
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
} /* ParallelGridCore::getOpposite */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
bool
ParallelGridCore::getHasL (int process) const
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)
  if (process > 0)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (process % nodeGridSizeX > 0)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  {
    return true;
  }

  return false;
}

bool
ParallelGridCore::getHasR (int process) const
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)
  if (process < nodeGridSizeX - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (process % nodeGridSizeX < nodeGridSizeX - 1)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  {
    return true;
  }

  return false;
}
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
bool
ParallelGridCore::getHasD (int process) const
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
  if (process > 0)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (process >= nodeGridSizeX)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (process % nodeGridSizeY > 0)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if ((process % (nodeGridSizeXY)) >= nodeGridSizeX)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  {
    return true;
  }

  return false;
}

bool
ParallelGridCore::getHasU (int process) const
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
  if (process < nodeGridSizeY - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
  if (process < nodeGridSizeXY - nodeGridSizeX)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (process % nodeGridSizeY < nodeGridSizeY - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if ((process % (nodeGridSizeXY)) < nodeGridSizeXY - nodeGridSizeX)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  {
    return true;
  }

  return false;
}
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
bool
ParallelGridCore::getHasB (int process) const
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  if (process > 0)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (process >= nodeGridSizeY)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  if (process >= nodeGridSizeX)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (process >= nodeGridSizeXY)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  {
    return true;
  }

  return false;
}

bool
ParallelGridCore::getHasF (int process) const
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  if (process < nodeGridSizeZ - 1)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (process < nodeGridSizeYZ - nodeGridSizeY)
#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  if (process < nodeGridSizeXZ - nodeGridSizeX)
#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (process < nodeGridSizeXYZ - nodeGridSizeXY)
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
  {
    return true;
  }

  return false;
}
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

/**
 * Initialize flags whether computational node has neighbors
 */
void
ParallelGridCore::InitBufferFlags ()
{
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  hasL = getHasL (processId);
  hasR = getHasR (processId);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  hasD = getHasD (processId);
  hasU = getHasU (processId);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  hasB = getHasB (processId);
  hasF = getHasF (processId);
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
} /* ParallelGridCore::InitBufferFlags */

/**
 * Initialize ids of neighbor computational nodes
 */
void
ParallelGridCore::InitDirections ()
{
#ifndef DYNAMIC_GRID
  directions.resize (BUFFER_COUNT);

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  directions[LEFT] = hasL ? processId - 1 : PID_NONE;
  directions[RIGHT] = hasR ? processId + 1 : PID_NONE;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  directions[DOWN] = hasD ? processId - 1 : PID_NONE;
  directions[UP] = hasU ? processId + 1 : PID_NONE;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_YZ */
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  directions[DOWN] = hasD ? processId - nodeGridSizeX : PID_NONE;
  directions[UP] = hasU ? processId + nodeGridSizeX : PID_NONE;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  directions[BACK] = hasB ? processId - 1 : PID_NONE;
  directions[FRONT] = hasF ? processId + 1 : PID_NONE;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z */
#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  directions[BACK] = hasB ? processId - nodeGridSizeY : PID_NONE;
  directions[FRONT] = hasF ? processId + nodeGridSizeY : PID_NONE;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  directions[BACK] = hasB ? processId - nodeGridSizeX : PID_NONE;
  directions[FRONT] = hasF ? processId + nodeGridSizeX : PID_NONE;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  directions[BACK] = hasB ? processId - nodeGridSizeXY : PID_NONE;
  directions[FRONT] = hasF ? processId + nodeGridSizeXY : PID_NONE;
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  directions[LEFT_DOWN] = hasL && hasD ? processId - nodeGridSizeX - 1 : PID_NONE;
  directions[LEFT_UP] = hasL && hasU ? processId + nodeGridSizeX - 1 : PID_NONE;
  directions[RIGHT_DOWN] = hasR && hasD ? processId - nodeGridSizeX + 1 : PID_NONE;
  directions[RIGHT_UP] = hasR && hasU ? processId + nodeGridSizeX + 1 : PID_NONE;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  directions[DOWN_BACK] = hasD && hasB ? processId - nodeGridSizeY - 1 : PID_NONE;
  directions[DOWN_FRONT] = hasD && hasF ? processId + nodeGridSizeY - 1 : PID_NONE;
  directions[UP_BACK] = hasU && hasB ? processId - nodeGridSizeY + 1 : PID_NONE;
  directions[UP_FRONT] = hasU && hasF ? processId + nodeGridSizeY + 1 : PID_NONE;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  directions[DOWN_BACK] = hasD && hasB ? processId - nodeGridSizeXY - nodeGridSizeX : PID_NONE;
  directions[DOWN_FRONT] = hasD && hasF ? processId + nodeGridSizeXY - nodeGridSizeX : PID_NONE;
  directions[UP_BACK] = hasU && hasB ? processId - nodeGridSizeXY + nodeGridSizeX : PID_NONE;
  directions[UP_FRONT] = hasU && hasF ? processId + nodeGridSizeXY + nodeGridSizeX : PID_NONE;
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  directions[LEFT_BACK] = hasL && hasB ? processId - nodeGridSizeX - 1 : PID_NONE;
  directions[LEFT_FRONT] = hasL && hasF ? processId + nodeGridSizeX - 1 : PID_NONE;
  directions[RIGHT_BACK] = hasR && hasB ? processId - nodeGridSizeX + 1 : PID_NONE;
  directions[RIGHT_FRONT] = hasR && hasF ? processId + nodeGridSizeX + 1 : PID_NONE;
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  directions[LEFT_BACK] = hasL && hasB ? processId - nodeGridSizeXY - 1 : PID_NONE;
  directions[LEFT_FRONT] = hasL && hasF ? processId + nodeGridSizeXY - 1 : PID_NONE;
  directions[RIGHT_BACK] = hasR && hasB ? processId - nodeGridSizeXY + 1 : PID_NONE;
  directions[RIGHT_FRONT] = hasR && hasF ? processId + nodeGridSizeXY + 1 : PID_NONE;
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  directions[LEFT_DOWN_BACK] = hasL && hasD && hasB ? processId - nodeGridSizeXY - nodeGridSizeX - 1 : PID_NONE;
  directions[LEFT_DOWN_FRONT] = hasL && hasD && hasF ? processId + nodeGridSizeXY - nodeGridSizeX - 1 : PID_NONE;
  directions[LEFT_UP_BACK] = hasL && hasU && hasB ? processId - nodeGridSizeXY + nodeGridSizeX - 1 : PID_NONE;
  directions[LEFT_UP_FRONT] = hasL && hasU && hasF ? processId + nodeGridSizeXY + nodeGridSizeX - 1 : PID_NONE;
  directions[RIGHT_DOWN_BACK] = hasR && hasD && hasB ? processId - nodeGridSizeXY - nodeGridSizeX + 1 : PID_NONE;
  directions[RIGHT_DOWN_FRONT] = hasR && hasD && hasF ? processId + nodeGridSizeXY - nodeGridSizeX + 1 : PID_NONE;
  directions[RIGHT_UP_BACK] = hasR && hasU && hasB ? processId - nodeGridSizeXY + nodeGridSizeX + 1 : PID_NONE;
  directions[RIGHT_UP_FRONT] = hasR && hasU && hasF ? processId + nodeGridSizeXY + nodeGridSizeX + 1 : PID_NONE;
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */
#endif

#ifdef DYNAMIC_GRID
  totalSumPerfPointsPerProcess.resize (totalProcCount);
  totalSumPerfTimePerProcess.resize (totalProcCount);

  totalSumLatencyPerConnection.resize (totalProcCount);
  totalSumLatencyCountPerConnection.resize (totalProcCount);

  totalSumBandwidthPerConnection.resize (totalProcCount);
  totalSumBandwidthCountPerConnection.resize (totalProcCount);

  curPoints.resize (totalProcCount);
  curTimes.resize (totalProcCount);
  skipCurShareMeasurement.resize (totalProcCount);
  curShareLatency.resize (totalProcCount);
  curShareBandwidth.resize (totalProcCount);

  speed.resize (totalProcCount);
  latency.resize (totalProcCount);
  bandwidth.resize (totalProcCount);

  for (int i = 0; i < totalProcCount; ++i)
  {
    totalSumLatencyPerConnection[i].resize (totalProcCount);
    totalSumLatencyCountPerConnection[i].resize (totalProcCount);

    totalSumBandwidthPerConnection[i].resize (totalProcCount);
    totalSumBandwidthCountPerConnection[i].resize (totalProcCount);

    skipCurShareMeasurement[i].resize (totalProcCount);
    curShareLatency[i].resize (totalProcCount);
    curShareBandwidth[i].resize (totalProcCount);

    latency[i].resize (totalProcCount);
    bandwidth[i].resize (totalProcCount);
  }

  for (int i = 0; i < totalProcCount; ++i)
  {
    SetNodesForDirections (i);
  }
#endif
} /* ParallelGridCore::InitDirections */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
/**
 * Get coordinate of process in the nodes' grid by Ox axis
 *
 * @return coordinate of process in the nodes' grid by Ox axis
 */
int
ParallelGridCore::getNodeGridX (int process) const /**< process id */
{
  int pidX;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)

  pidX = process;

#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || \
      defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  pidX = process % nodeGridSizeX;

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  return pidX;
} /* ParallelGridCore::getNodeGridX */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
/**
 * Get coordinate of process in the nodes' grid by Oy axis
 *
 * @return coordinate of process in the nodes' grid by Oy axis
 */
int
ParallelGridCore::getNodeGridY (int process) const /**< process id */
{
  int pidY;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y)

  pidY = process;

#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XY)

  pidY = process / nodeGridSizeX;

#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)

  pidY = process % nodeGridSizeY;

#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  int pidXY = process % nodeGridSizeXY;
  pidY = pidXY / nodeGridSizeX;

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  return pidY;
} /* ParallelGridCore::getNodeGridY */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
        PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
/**
 * Get coordinate of process in the nodes' grid by Oz axis
 *
 * @return coordinate of process in the nodes' grid by Oz axis
 */
int
ParallelGridCore::getNodeGridZ (int process) const /**< process id */
{
  int pidZ;

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)

  pidZ = process;

#elif defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)

  pidZ = process / nodeGridSizeY;

#elif defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)

  pidZ = process / nodeGridSizeX;

#elif defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  pidZ = process / nodeGridSizeXY;

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  return pidZ;
} /* ParallelGridCore::getNodeGridZ */

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

/**
 * Initialize parallel data common for all parallel grids on a single computational node
 */
void
ParallelGridCore::ParallelGridCoreConstructor (ParallelGridCoordinate size) /**< size of grid */
{
  NodeGridInit (size);

  /*
   * Return if node not used.
   */
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
  InitDirections ();

#ifdef DYNAMIC_GRID
  /*
   * All nodes are enables by default
   */
  nodeState.resize (totalProcCount);
  for (int i = 0; i < nodeState.size (); ++i)
  {
    nodeState[i] = 1;
  }
#endif
} /* ParallelGridCore::ParallelGridCoreConstructor */

/**
 * Constructor for code data of parallel grid shared between all parallel grid on a single computational node
 */
ParallelGridCore::ParallelGridCore (int process, /**< id of computational node */
                                    int totalProc, /**< total number of computational nodes */
                                    ParallelGridCoordinate size, /**< size of grid (not used
                                                                  *   for 1D buffer dimensions) */
                                    bool useManualTopology, /**< flag whether to use manual virtual topology */
                                    ParallelGridCoordinate topology) /**< topology size, specified manually */
  : processId (process)
  , totalProcCount (totalProc)
  , doUseManualTopology (useManualTopology)
  , topologySize (topology)
{
  /*
   * Set default values for flags whether computational node has neighbors
   */
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  hasL = false;
  hasR = false;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  hasD = false;
  hasU = false;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  hasB = false;
  hasF = false;
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  initOppositeDirections ();

  ParallelGridCoreConstructor (size);

#ifndef COMBINED_SENDRECV
  isEvenForDirection.resize (BUFFER_COUNT);
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  isEvenForDirection[LEFT] = isEvenForDirection[RIGHT] = getNodeGridX () % 2 == 0;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  isEvenForDirection[DOWN] = isEvenForDirection[UP] = getNodeGridY () % 2 == 0;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  isEvenForDirection[BACK] = isEvenForDirection[FRONT] = getNodeGridZ () % 2 == 0;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  isEvenForDirection[LEFT_DOWN] = isEvenForDirection[RIGHT_DOWN] = getNodeGridX () % 2 == 0;
  isEvenForDirection[LEFT_UP] = isEvenForDirection[RIGHT_UP] = getNodeGridX () % 2 == 0;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  isEvenForDirection[DOWN_BACK] = isEvenForDirection[UP_BACK] = getNodeGridY () % 2 == 0;
  isEvenForDirection[DOWN_FRONT] = isEvenForDirection[UP_FRONT] = getNodeGridY () % 2 == 0;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  isEvenForDirection[LEFT_BACK] = isEvenForDirection[RIGHT_BACK] = getNodeGridX () % 2 == 0;
  isEvenForDirection[LEFT_FRONT] = isEvenForDirection[RIGHT_FRONT] = getNodeGridX () % 2 == 0;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  isEvenForDirection[LEFT_DOWN_BACK] = isEvenForDirection[LEFT_DOWN_FRONT] = getNodeGridX () % 2 == 0;
  isEvenForDirection[LEFT_UP_BACK] = isEvenForDirection[LEFT_UP_FRONT] = getNodeGridX () % 2 == 0;
  isEvenForDirection[RIGHT_DOWN_BACK] = isEvenForDirection[RIGHT_DOWN_FRONT] = getNodeGridX () % 2 == 0;
  isEvenForDirection[RIGHT_UP_BACK] = isEvenForDirection[RIGHT_UP_FRONT] = getNodeGridX () % 2 == 0;
#endif
#endif /* !COMBINED_SENDRECV */

#ifdef GRID_1D
  totalProcCount = nodeGridSizeX;
#endif
#ifdef GRID_2D
  totalProcCount = nodeGridSizeX * nodeGridSizeY;
#endif
#ifdef GRID_3D
  totalProcCount = nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ;
#endif

  ASSERT (totalProcCount > 1);

  int retCode = MPI_Comm_split (MPI_COMM_WORLD, process < totalProcCount ? 0 : MPI_UNDEFINED, process, &communicator);
  ASSERT (retCode == MPI_SUCCESS);

#ifdef DYNAMIC_GRID
  calcClockSumBetweenRebalance.resize (totalProcCount);
  calcClockCountBetweenRebalance.resize (totalProcCount);

  shareClockCountBetweenRebalance.resize (totalProcCount);

  shareClockSumBetweenRebalance.resize (totalProcCount);
  shareClockIterBetweenRebalance.resize (totalProcCount);

  for (int i = 0; i < totalProcCount; ++i)
  {
    shareClockSumBetweenRebalance[i].resize (totalProcCount);
    shareClockIterBetweenRebalance[i].resize (totalProcCount);
  }

#ifdef MPI_DYNAMIC_CLOCK
  shareClockSec_buf = new FPValue [CLOCK_BUF_SIZE * totalProcCount];
#else
  shareClockSec_buf = new uint64_t [CLOCK_BUF_SIZE * totalProcCount];
  shareClockNSec_buf = new uint64_t [CLOCK_BUF_SIZE * totalProcCount];
#endif
  shareClockBufSize_buf = new uint32_t [CLOCK_BUF_SIZE * totalProcCount];

  shareClockBufSize2_buf = new uint32_t [CLOCK_BUF_SIZE * totalProcCount];
  shareClockIter_buf = new uint32_t [CLOCK_BUF_SIZE * totalProcCount];
#endif /* DYNAMIC_GRID */
} /* ParallelGridCore */

ParallelGridCore::~ParallelGridCore ()
{
#ifdef DYNAMIC_GRID
  delete[] shareClockSec_buf;

#ifndef MPI_DYNAMIC_CLOCK
  delete[] shareClockNSec_buf;
#endif
  delete[] shareClockBufSize_buf;

  delete[] shareClockBufSize2_buf;
  delete[] shareClockIter_buf;
#endif /* DYNAMIC_GRID */
}

int
ParallelGridCore::getNodeForDirection (BufferPosition dir) const
{
#ifdef DYNAMIC_GRID
  return getNodeForDirectionForProcess (getProcessId (), dir);
#else
  return directions[dir];
#endif
}

#endif /* PARALLEL_GRID */
