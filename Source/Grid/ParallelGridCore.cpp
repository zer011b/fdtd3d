#include "ParallelGridCore.h"

#ifdef PARALLEL_GRID

#ifdef DYNAMIC_GRID
#include <unistd.h>
#endif /* DYNAMIC_GRID */

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
  perfPointsValues.resize (totalProcCount);
  perfTimeValues.resize (totalProcCount);

  latencySumValues.resize (totalProcCount);
  latencyCountValues.resize (totalProcCount);

  bandwidthSumValues.resize (totalProcCount);
  bandwidthCountValues.resize (totalProcCount);

  for (int i = 0; i < totalProcCount; ++i)
  {
    latencySumValues[i].resize (totalProcCount);
    latencyCountValues[i].resize (totalProcCount);

    bandwidthSumValues[i].resize (totalProcCount);
    bandwidthCountValues[i].resize (totalProcCount);
  }

  SetNodesForDirections ();
#endif
} /* ParallelGridCore::InitDirections */

#ifdef DYNAMIC_GRID
void
ParallelGridCore::SetNodesForDirections ()
{
  nodesForDirections.resize (BUFFER_COUNT);

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  for (int i = processId - 1; i >= processId - getNodeGridX (); --i)
  {
    nodesForDirections[LEFT].push_back (i);
  }
  for (int i = processId + 1; i < processId - getNodeGridX () + getNodeGridSizeX (); ++i)
  {
    nodesForDirections[RIGHT].push_back (i);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  for (int i = processId - 1; i >= processId - getNodeGridY (); --i)
  {
    nodesForDirections[DOWN].push_back (i);
  }
  for (int i = processId + 1; i < processId - getNodeGridY () + getNodeGridSizeY (); ++i)
  {
    nodesForDirections[UP].push_back (i);
  }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  for (int i = processId - getNodeGridSizeX (); i >= processId - getNodeGridY () * getNodeGridSizeX (); i -= getNodeGridSizeX ())
  {
    nodesForDirections[DOWN].push_back (i);
  }
  for (int i = processId + getNodeGridSizeX (); i < processId - (getNodeGridY () - getNodeGridSizeY ()) * getNodeGridSizeX (); i += getNodeGridSizeX ())
  {
    nodesForDirections[UP].push_back (i);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  for (int i = processId - 1; i >= processId - getNodeGridZ (); --i)
  {
    nodesForDirections[BACK].push_back (i);
  }
  for (int i = processId + 1; i < processId - getNodeGridZ () + getNodeGridSizeZ (); ++i)
  {
    nodesForDirections[FRONT].push_back (i);
  }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  for (int i = processId - getNodeGridSizeY (); i >= processId - getNodeGridZ () * getNodeGridSizeY (); i -= getNodeGridSizeY ())
  {
    nodesForDirections[BACK].push_back (i);
  }
  for (int i = processId + getNodeGridSizeY (); i < processId - (getNodeGridZ () - getNodeGridSizeZ ()) * getNodeGridSizeY (); i += getNodeGridSizeY ())
  {
    nodesForDirections[FRONT].push_back (i);
  }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  for (int i = processId - getNodeGridSizeX (); i >= processId - getNodeGridZ () * getNodeGridSizeX (); i -= getNodeGridSizeX ())
  {
    nodesForDirections[BACK].push_back (i);
  }
  for (int i = processId + getNodeGridSizeX (); i < processId - (getNodeGridZ () - getNodeGridSizeZ ()) * getNodeGridSizeX (); i += getNodeGridSizeX ())
  {
    nodesForDirections[FRONT].push_back (i);
  }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  for (int i = processId - getNodeGridSizeXY (); i >= processId - getNodeGridZ () * getNodeGridSizeXY (); i -= getNodeGridSizeXY ())
  {
    nodesForDirections[BACK].push_back (i);
  }
  for (int i = processId + getNodeGridSizeXY (); i < processId - (getNodeGridZ () - getNodeGridSizeZ ()) * getNodeGridSizeXY (); i += getNodeGridSizeXY ())
  {
    nodesForDirections[FRONT].push_back (i);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (getHasL () && getHasD ())
  {
    int i = 0;
    for (i = processId - getNodeGridSizeX () - 1; getHasL (i) && getHasD (i); i -= getNodeGridSizeX () + 1)
    {
      nodesForDirections[LEFT_DOWN].push_back (i);
    }
    nodesForDirections[LEFT_DOWN].push_back (i);
  }
  if (getHasL () && getHasU ())
  {
    int i = 0;
    for (i = processId + getNodeGridSizeX () - 1; getHasL (i) && getHasU (i); i += getNodeGridSizeX () - 1)
    {
      nodesForDirections[LEFT_UP].push_back (i);
    }
    nodesForDirections[LEFT_UP].push_back (i);
  }
  if (getHasR () && getHasD ())
  {
    int i = 0;
    for (i = processId - getNodeGridSizeX () + 1; getHasR (i) && getHasD (i); i -= getNodeGridSizeX () - 1)
    {
      nodesForDirections[RIGHT_DOWN].push_back (i);
    }
    nodesForDirections[RIGHT_DOWN].push_back (i);
  }
  if (getHasR () && getHasU ())
  {
    int i = 0;
    for (i = processId + getNodeGridSizeX () + 1; getHasR (i) && getHasU (i); i += getNodeGridSizeX () + 1)
    {
      nodesForDirections[RIGHT_UP].push_back (i);
    }
    nodesForDirections[RIGHT_UP].push_back (i);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (getHasD () && getHasB ())
  {
    int i = 0;
    for (i = processId - getNodeGridSizeY () - 1; getHasD (i) && getHasB (i); i -= getNodeGridSizeY () + 1)
    {
      nodesForDirections[DOWN_BACK].push_back (i);
    }
    nodesForDirections[DOWN_BACK].push_back (i);
  }
  if (getHasD () && getHasF ())
  {
    int i = 0;
    for (i = processId + getNodeGridSizeY () - 1; getHasD (i) && getHasF (i); i += getNodeGridSizeY () - 1)
    {
      nodesForDirections[DOWN_FRONT].push_back (i);
    }
    nodesForDirections[DOWN_FRONT].push_back (i);
  }
  if (getHasU () && getHasB ())
  {
    int i = 0;
    for (i = processId - getNodeGridSizeY () + 1; getHasU (i) && getHasB (i); i -= getNodeGridSizeY () - 1)
    {
      nodesForDirections[UP_BACK].push_back (i);
    }
    nodesForDirections[UP_BACK].push_back (i);
  }
  if (getHasU () && getHasF ())
  {
    int i = 0;
    for (i = processId + getNodeGridSizeY () + 1; getHasU (i) && getHasF (i); i += getNodeGridSizeY () + 1)
    {
      nodesForDirections[UP_FRONT].push_back (i);
    }
    nodesForDirections[UP_FRONT].push_back (i);
  }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (getHasD () && getHasB ())
  {
    int i = 0;
    for (i = processId - getNodeGridSizeXY () - getNodeGridSizeX (); getHasD (i) && getHasB (i); i -= getNodeGridSizeXY () + getNodeGridSizeX ())
    {
      nodesForDirections[DOWN_BACK].push_back (i);
    }
    nodesForDirections[DOWN_BACK].push_back (i);
  }
  if (getHasD () && getHasF ())
  {
    int i = 0;
    for (i = processId + getNodeGridSizeXY () - getNodeGridSizeX (); getHasD (i) && getHasF (i); i += getNodeGridSizeXY () - getNodeGridSizeX ())
    {
      nodesForDirections[DOWN_FRONT].push_back (i);
    }
    nodesForDirections[DOWN_FRONT].push_back (i);
  }
  if (getHasU () && getHasB ())
  {
    int i = 0;
    for (i = processId - getNodeGridSizeXY () + getNodeGridSizeX (); getHasU (i) && getHasB (i); i -= getNodeGridSizeXY () - getNodeGridSizeX ())
    {
      nodesForDirections[UP_BACK].push_back (i);
    }
    nodesForDirections[UP_BACK].push_back (i);
  }
  if (getHasU () && getHasF ())
  {
    int i = 0;
    for (i = processId + getNodeGridSizeXY () + getNodeGridSizeX (); getHasU (i) && getHasF (i); i += getNodeGridSizeXY () + getNodeGridSizeX ())
    {
      nodesForDirections[UP_FRONT].push_back (i);
    }
    nodesForDirections[UP_FRONT].push_back (i);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  if (getHasL () && getHasB ())
  {
    int i = 0;
    for (i = processId - getNodeGridSizeX () - 1; getHasL (i) && getHasB (i); i -= getNodeGridSizeX () + 1)
    {
      nodesForDirections[LEFT_BACK].push_back (i);
    }
    nodesForDirections[LEFT_BACK].push_back (i);
  }
  if (getHasL () && getHasF ())
  {
    int i = 0;
    for (i = processId + getNodeGridSizeX () - 1; getHasL (i) && getHasF (i); i += getNodeGridSizeX () - 1)
    {
      nodesForDirections[LEFT_FRONT].push_back (i);
    }
    nodesForDirections[LEFT_FRONT].push_back (i);
  }
  if (getHasR () && getHasB ())
  {
    int i = 0;
    for (i = processId - getNodeGridSizeX () + 1; getHasR (i) && getHasB (i); i -= getNodeGridSizeX () - 1)
    {
      nodesForDirections[RIGHT_BACK].push_back (i);
    }
    nodesForDirections[RIGHT_BACK].push_back (i);
  }
  if (getHasR () && getHasF ())
  {
    int i = 0;
    for (i = processId + getNodeGridSizeX () + 1; getHasR (i) && getHasF (i); i += getNodeGridSizeX () + 1)
    {
      nodesForDirections[RIGHT_FRONT].push_back (i);
    }
    nodesForDirections[RIGHT_FRONT].push_back (i);
  }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (getHasL () && getHasB ())
  {
    int i = 0;
    for (i = processId - getNodeGridSizeXY () - 1; getHasL (i) && getHasB (i); i -= getNodeGridSizeXY () + 1)
    {
      nodesForDirections[LEFT_BACK].push_back (i);
    }
    nodesForDirections[LEFT_BACK].push_back (i);
  }
  if (getHasL () && getHasF ())
  {
    int i = 0;
    for (i = processId + getNodeGridSizeXY () - 1; getHasL (i) && getHasF (i); i += getNodeGridSizeXY () - 1)
    {
      nodesForDirections[LEFT_FRONT].push_back (i);
    }
    nodesForDirections[LEFT_FRONT].push_back (i);
  }
  if (getHasR () && getHasB ())
  {
    int i = 0;
    for (i = processId - getNodeGridSizeXY () + 1; getHasR (i) && getHasB (i); i -= getNodeGridSizeXY () - 1)
    {
      nodesForDirections[RIGHT_BACK].push_back (i);
    }
    nodesForDirections[RIGHT_BACK].push_back (i);
  }
  if (getHasR () && getHasF ())
  {
    int i = 0;
    for (i = processId + getNodeGridSizeXY () + 1; getHasR (i) && getHasF (i); i += getNodeGridSizeXY () + 1)
    {
      nodesForDirections[RIGHT_FRONT].push_back (i);
    }
    nodesForDirections[RIGHT_FRONT].push_back (i);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (getHasL () && getHasD () && getHasB ())
  {
    int i = 0;
    for (i = processId - getNodeGridSizeXY () - getNodeGridSizeX () - 1;
         getHasL (i) && getHasD (i) && getHasB (i);
         i -= getNodeGridSizeXY () + getNodeGridSizeX () + 1)
    {
      nodesForDirections[LEFT_DOWN_BACK].push_back (i);
    }
    nodesForDirections[LEFT_DOWN_BACK].push_back (i);
  }
  if (getHasL () && getHasD () && getHasF ())
  {
    int i = 0;
    for (i = processId + getNodeGridSizeXY () - getNodeGridSizeX () - 1;
         getHasL (i) && getHasD (i) && getHasF (i);
         i += getNodeGridSizeXY () - getNodeGridSizeX () - 1)
    {
      nodesForDirections[LEFT_DOWN_FRONT].push_back (i);
    }
    nodesForDirections[LEFT_DOWN_FRONT].push_back (i);
  }
  if (getHasL () && getHasU () && getHasB ())
  {
    int i = 0;
    for (i = processId - getNodeGridSizeXY () + getNodeGridSizeX () - 1;
         getHasL (i) && getHasU (i) && getHasB (i);
         i -= getNodeGridSizeXY () - getNodeGridSizeX () + 1)
    {
      nodesForDirections[LEFT_UP_BACK].push_back (i);
    }
    nodesForDirections[LEFT_UP_BACK].push_back (i);
  }
  if (getHasL () && getHasU () && getHasF ())
  {
    int i = 0;
    for (i = processId + getNodeGridSizeXY () + getNodeGridSizeX () - 1;
         getHasL (i) && getHasU (i) && getHasF (i);
         i += getNodeGridSizeXY () + getNodeGridSizeX () - 1)
    {
      nodesForDirections[LEFT_UP_FRONT].push_back (i);
    }
    nodesForDirections[LEFT_UP_FRONT].push_back (i);
  }

  if (getHasR () && getHasD () && getHasB ())
  {
    int i = 0;
    for (i = processId - getNodeGridSizeXY () - getNodeGridSizeX () + 1;
         getHasR (i) && getHasD (i) && getHasB (i);
         i -= getNodeGridSizeXY () + getNodeGridSizeX () - 1)
    {
      nodesForDirections[RIGHT_DOWN_BACK].push_back (i);
    }
    nodesForDirections[RIGHT_DOWN_BACK].push_back (i);
  }
  if (getHasR () && getHasD () && getHasF ())
  {
    int i = 0;
    for (i = processId + getNodeGridSizeXY () - getNodeGridSizeX () + 1;
         getHasR (i) && getHasD (i) && getHasF (i);
         i += getNodeGridSizeXY () - getNodeGridSizeX () + 1)
    {
      nodesForDirections[RIGHT_DOWN_FRONT].push_back (i);
    }
    nodesForDirections[RIGHT_DOWN_FRONT].push_back (i);
  }
  if (getHasR () && getHasU () && getHasB ())
  {
    int i = 0;
    for (i = processId - getNodeGridSizeXY () + getNodeGridSizeX () + 1;
         getHasR (i) && getHasU (i) && getHasB (i);
         i -= getNodeGridSizeXY () - getNodeGridSizeX () - 1)
    {
      nodesForDirections[RIGHT_UP_BACK].push_back (i);
    }
    nodesForDirections[RIGHT_UP_BACK].push_back (i);
  }
  if (getHasR () && getHasU () && getHasF ())
  {
    int i = 0;
    for (i = processId + getNodeGridSizeXY () + getNodeGridSizeX () + 1;
         getHasR (i) && getHasU (i) && getHasF (i);
         i += getNodeGridSizeXY () + getNodeGridSizeX () + 1)
    {
      nodesForDirections[RIGHT_UP_FRONT].push_back (i);
    }
    nodesForDirections[RIGHT_UP_FRONT].push_back (i);
  }
#endif

#if PRINT_MESSAGE
  MPI_Barrier (MPI_COMM_WORLD);
  if (processId == 0)
  {
    DPRINTF (LOG_LEVEL_NONE, "=== Processes map ===\n");
  }
  for (int pid = 0; pid < totalProcCount; ++pid)
  {
    if (processId == pid)
    {
      DPRINTF (LOG_LEVEL_NONE, "Process #%d:\n", processId);
      for (int dir = 0; dir < BUFFER_COUNT; ++dir)
      {
        DPRINTF (LOG_LEVEL_NONE, "  Processes to %s: ", BufferPositionNames[dir]);

        for (int i = 0; i < nodesForDirections[dir].size (); ++i)
        {
          DPRINTF (LOG_LEVEL_NONE, " %d, ", nodesForDirections[dir][i]);
        }

        DPRINTF (LOG_LEVEL_NONE, "\n");
      }
    }
    MPI_Barrier (MPI_COMM_WORLD);
  }
#endif
}
#endif

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

#ifdef DYNAMIC_GRID
/**
 * Start clock for calculations
 */
void
ParallelGridCore::StartCalcClock ()
{
  int status = clock_gettime (CLOCK_MONOTONIC, &calcStart);
  ASSERT (status == 0);
} /* ParallelGridCore::StartCalcClock */

/**
 * Stop clock for calculations
 */
void
ParallelGridCore::StopCalcClock ()
{
  int status = clock_gettime (CLOCK_MONOTONIC, &calcStop);
  ASSERT (status == 0);

  timespec diff;
  timespec_diff (&calcStart, &calcStop, &diff);

  timespec sum;
  timespec_sum (&calcClockSumBetweenRebalance[processId], &diff, &sum);

  calcClockSumBetweenRebalance[processId] = sum;
} /* ParallelGridCore::StopCalcClock */

/**
 * Start clock for share operations
 */
void
ParallelGridCore::StartShareClock (int pid, /**< pid of process, with which share operations' time is measured */
                                   uint32_t count)
{
  ASSERT (pid >= 0 && pid < totalProcCount);
  ASSERT (pid != processId);

#ifdef MPI_DYNAMIC_CLOCK
  ShareClock_t map = getShareClockCur (pid);
  FPValue val = -MPI_Wtime ();
  if (map.find (count) != map.end ())
  {
    val += map[count];
  }
  setShareClockCur (pid, count, val);
#else
  int status = clock_gettime (CLOCK_MONOTONIC, &shareStart);
  ASSERT (status == 0);
#endif
} /* ParallelGridCore::StartShareClock */

/**
 * Stop clock for share operations
 */
void
ParallelGridCore::StopShareClock (int pid, /**< pid of process, with which share operations' time is measured */
                                  uint32_t count)
{
  ASSERT (pid >= 0 && pid < totalProcCount);
  ASSERT (pid != processId);

  ShareClock_t map = getShareClockCur (pid);
  ASSERT (map.find (count) != map.end ());

#ifdef MPI_DYNAMIC_CLOCK
  FPValue val = MPI_Wtime ();
  val += map[count];
  setShareClockCur (pid, count, val);
#else
  int status = clock_gettime (CLOCK_MONOTONIC, &shareStop);
  ASSERT (status == 0);

  timespec diff;
  timespec_diff (&shareStart, &shareStop, &diff);

  timespec val_old = map[count];

  timespec sum;
  timespec_sum (&val_old, &diff, &sum);

  setShareClockCur (pid, count, sum);
#endif
} /* ParallelGridCore::StopShareClock */

/**
 * Calculate difference of two moments in time
 */
void
ParallelGridCore::timespec_diff (struct timespec *start, /**< start moment */
                                 struct timespec *stop, /**< end moment */
                                 struct timespec *result) /**< out: difference of two moments in time */
{
  if ((stop->tv_nsec - start->tv_nsec) < 0)
  {
    result->tv_sec = stop->tv_sec - start->tv_sec - 1;
    result->tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
  }
  else
  {
    result->tv_sec = stop->tv_sec - start->tv_sec;
    result->tv_nsec = stop->tv_nsec - start->tv_nsec;
  }
} /* ParallelGridCore::timespec_diff */

void
ParallelGridCore::timespec_sum (struct timespec *start, /**< start moment */
                                struct timespec *stop, /**< end moment */
                                struct timespec *result) /**< out: difference of two moments in time */
{
  result->tv_sec = start->tv_sec + stop->tv_sec;
  result->tv_nsec = start->tv_nsec + stop->tv_nsec;

  if (result->tv_nsec >= 1000000000)
  {
    result->tv_nsec -= 1000000000;
    result->tv_sec += 1;
  }
}

void
ParallelGridCore::timespec_avg (struct timespec *start, /**< start moment */
                                struct timespec *stop, /**< end moment */
                                struct timespec *result) /**< out: difference of two moments in time */
{
  result->tv_sec = (start->tv_sec + stop->tv_sec) / 2;
  result->tv_nsec = (start->tv_nsec + stop->tv_nsec) / 2;
}

void ParallelGridCore::ShareClocks ()
{
  ShareCalcClocks ();
  ShareShareClocks ();
}

void ParallelGridCore::ShareCalcClocks ()
{
  for (int process = 0; process < getTotalProcCount (); ++process)
  {
    uint64_t calcClockSec;
    uint64_t calcClockNSec;
    uint32_t calcClockCount;

    if (process == getProcessId ())
    {
      calcClockSec = (uint64_t) calcClockSumBetweenRebalance[process].tv_sec;
      calcClockNSec = (uint64_t) calcClockSumBetweenRebalance[process].tv_nsec;
      calcClockCount = calcClockCountBetweenRebalance[process];
    }

    MPI_Bcast (&calcClockSec, 1, MPI_LONG_LONG, process, communicator);
    MPI_Bcast (&calcClockNSec, 1, MPI_LONG_LONG, process, communicator);
    MPI_Bcast (&calcClockCount, 1, MPI_UNSIGNED, process, communicator);

    if (process != getProcessId ())
    {
      calcClockSumBetweenRebalance[process].tv_sec = calcClockSec;
      calcClockSumBetweenRebalance[process].tv_nsec = calcClockNSec;
      calcClockCountBetweenRebalance[process] = calcClockCount;
    }

    MPI_Barrier (communicator);
  }
}

/**
 * Share clock counters with other processes
 */
void ParallelGridCore::ShareShareClocks ()
{
  for (int process = 0; process < getTotalProcCount (); ++process)
  {
    if (process == getProcessId ())
    {
      int j = 0;
      int jj = 0;
      for (int i = 0; i < getTotalProcCount (); ++i)
      {
        ASSERT (shareClockSumBetweenRebalance[process][i].size () == CLOCK_BUF_SIZE
                || shareClockSumBetweenRebalance[process][i].empty ());

        // Map is temporary of size CLOCK_BUF_SIZE
        for (ShareClock_t::iterator it = shareClockSumBetweenRebalance[process][i].begin ();
             it != shareClockSumBetweenRebalance[process][i].end (); ++it)
        {
          shareClockBufSize_buf[j] = it->first;

#ifdef MPI_DYNAMIC_CLOCK
          shareClockSec_buf[j] = it->second;
#else
          shareClockSec_buf[j] = it->second.tv_sec;
          shareClockNSec_buf[j] = it->second.tv_sec;
#endif

          j++;
        }

        if (shareClockSumBetweenRebalance[process][i].empty ())
        {
          shareClockBufSize_buf[j] = 0;
          shareClockSec_buf[j] = 0;
#ifndef MPI_DYNAMIC_CLOCK
          shareClockNSec_buf[j] = 0;
#endif
          j++;

          shareClockBufSize_buf[j] = 0;
          shareClockSec_buf[j] = 0;
#ifndef MPI_DYNAMIC_CLOCK
          shareClockNSec_buf[j] = 0;
#endif
          j++;
        }


        for (IterCount_t::iterator it = shareClockIterBetweenRebalance[process][i].begin ();
             it != shareClockIterBetweenRebalance[process][i].end (); ++it)
        {
          shareClockBufSize2_buf[jj] = it->first;
          shareClockIter_buf[jj] = it->second;

          jj++;
        }

        if (shareClockIterBetweenRebalance[process][i].empty ())
        {
          shareClockBufSize2_buf[jj] = 0;
          shareClockIter_buf[jj] = 0;
          jj++;

          shareClockBufSize2_buf[jj] = 0;
          shareClockIter_buf[jj] = 0;
          jj++;
        }
      }
    }

    MPI_Bcast (shareClockBufSize_buf, CLOCK_BUF_SIZE * totalProcCount, MPI_UNSIGNED, process, communicator);
#ifdef MPI_DYNAMIC_CLOCK
    MPI_Bcast (shareClockSec_buf, CLOCK_BUF_SIZE * totalProcCount, MPI_FPVALUE, process, communicator);
#else
    MPI_Bcast (shareClockSec_buf, CLOCK_BUF_SIZE * totalProcCount, MPI_LONG_LONG, process, communicator);
    MPI_Bcast (shareClockNSec_buf, CLOCK_BUF_SIZE * totalProcCount, MPI_LONG_LONG, process, communicator);
#endif

    MPI_Bcast (shareClockBufSize2_buf, CLOCK_BUF_SIZE * totalProcCount, MPI_UNSIGNED, process, communicator);
    MPI_Bcast (shareClockIter_buf, CLOCK_BUF_SIZE * totalProcCount, MPI_UNSIGNED, process, communicator);

    if (process != getProcessId ())
    {
      for (int i = 0; i < totalProcCount; ++i)
      {
        for (int j = 0; j < CLOCK_BUF_SIZE; ++j)
        {
          int index = i * CLOCK_BUF_SIZE + j;

          uint32_t bufSize = shareClockBufSize_buf[index];

#ifdef MPI_DYNAMIC_CLOCK
          FPValue val = shareClockSec_buf[index];
          if (val == 0 && bufSize == 0)
          {
            continue;
          }
#else
          timespec val;
          val.tv_sec = shareClockSec_buf[index];
          val.tv_nsec = shareClockNSec_buf[index];
          if (val.tv_sec == 0 && val.tv_nsec == 0 && bufSize == 0)
          {
            continue;
          }
#endif

          shareClockSumBetweenRebalance[process][i][bufSize] = val;


          uint32_t bufSize2 = shareClockBufSize2_buf[index];
          uint32_t val2 = shareClockIter_buf[index];
          if (val2 == 0 && bufSize2 == 0)
          {
            continue;
          }
          shareClockIterBetweenRebalance[process][i][bufSize2] = val2;
        }
      }
    }

    MPI_Barrier (communicator);
  }

  /*
   * Balance share clocks
   */
  for (int i = 0; i < totalProcCount; ++i)
  {
    for (int j = i + 1; j < totalProcCount; ++j)
    {
      ASSERT (shareClockSumBetweenRebalance[i][j].size () == shareClockSumBetweenRebalance[j][i].size ());
      ASSERT (shareClockIterBetweenRebalance[i][j].size () == shareClockIterBetweenRebalance[j][i].size ());
      ASSERT (shareClockSumBetweenRebalance[i][j].size () == shareClockIterBetweenRebalance[i][j].size ());

      ASSERT (shareClockSumBetweenRebalance[i][j].size () == CLOCK_BUF_SIZE
              || shareClockSumBetweenRebalance[i][j].size () == 0);

      for (ShareClock_t::iterator it = shareClockSumBetweenRebalance[i][j].begin ();
           it != shareClockSumBetweenRebalance[i][j].end (); ++it)
      {
        uint32_t bufSize = it->first;

        ShareClock_t::iterator iter = shareClockSumBetweenRebalance[j][i].find (bufSize);
        ASSERT (iter != shareClockSumBetweenRebalance[j][i].end ());

#ifdef MPI_DYNAMIC_CLOCK
        FPValue tmp = (it->second + iter->second) / FPValue (2);
#else
        timespec tmp;
        timespec_avg (&it->second, &iter->second, &tmp);
#endif
        shareClockSumBetweenRebalance[i][j][bufSize] = tmp;
        shareClockSumBetweenRebalance[j][i][bufSize] = tmp;
      }
    }
  }
}

/**
 * Set clocks to zeros
 */
void
ParallelGridCore::ClearCalcClocks ()
{
  for (int i = 0; i < totalProcCount; ++i)
  {
    calcClockSumBetweenRebalance[i].tv_sec = 0;
    calcClockSumBetweenRebalance[i].tv_nsec = 0;

    calcClockCountBetweenRebalance[i] = 0;
  }
}

void
ParallelGridCore::ClearShareClocks ()
{
  for (int i = 0; i < totalProcCount; ++i)
  {
    shareClockCountBetweenRebalance[i] = 0;

    for (int j = 0; j < totalProcCount; ++j)
    {
      shareClockSumBetweenRebalance[i][j].clear ();
      shareClockIterBetweenRebalance[i][j].clear ();
    }
  }
}

#endif /* DYNAMIC_GRID */

int
ParallelGridCore::getNodeForDirection (BufferPosition dir) const
{
#ifdef DYNAMIC_GRID
  for (int i = 0; i < nodesForDirections[dir].size (); ++i)
  {
    int process = nodesForDirections[dir][i];

    /*
     * Choose the first enabled node
     */
    if (nodeState[process])
    {
      return process;
    }
  }

  return PID_NONE;
#else
  return directions[dir];
#endif
}

#endif /* PARALLEL_GRID */
