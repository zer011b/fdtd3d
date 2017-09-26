#include "ParallelGridCore.h"

#ifdef PARALLEL_GRID

#ifdef DYNAMIC_GRID
#include <unistd.h>
#endif /* DYNAMIC_GRID */

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

/**
 * Initialize flags whether computational node has neighbours
 */
void
ParallelGridCore::InitBufferFlags ()
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
} /* ParallelGridCore::InitBufferFlags */

/**
 * Initialize ids of neighbour computational nodes
 */
void
ParallelGridCore::InitDirections ()
{
  directions.resize (BUFFER_COUNT);

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
} /* ParallelGridCore::InitDirections */

/**
 * Identify if computational nodes needs to perform send/receive operations in case share is performed in
 * specified direction
 */
void
ParallelGridCore::getShare (BufferPosition direction, /**< direction of share operation (send, opposite for receive) */
                            std::pair<bool, bool>& pair) /**< out: pair of flags whether computational node needs to
                                                          *        perform send and receive operations */
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
} /* ParallelGridCore::getShare */

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
} /* ParallelGridCore::ParallelGridCoreConstructor */

/**
 * Constructor for code data of parallel grid shared between all parallel grid on a single computational node
 */
ParallelGridCore::ParallelGridCore (int process, /**< id of computational node */
                                    int totalProc, /**< total number of computational nodes */
                                    ParallelGridCoordinate size, /**< size of grid (not used
                                                                  *   for 1D buffer dimensions) */
                                    bool useManualTopology, /**< flag whether to use manual virtual topology */
                                    GridCoordinate3D topology) /**< topology size, specified manually */
  : processId (process)
  , totalProcCount (totalProc)
  , doUseManualTopology (useManualTopology)
  , topologySize (topology)
{
  /*
   * Set default values for flags whether computational node has neighbours
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

  doShare.resize (BUFFER_COUNT);
  for (int i = 0; i < BUFFER_COUNT; ++i)
  {
    getShare ((BufferPosition) i, doShare[i]);
  }

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
  calcClockAll.resize (totalProcCount);
  shareClockAll.resize (totalProcCount);
#endif /* DYNAMIC_GRID */
} /* ParallelGridCore */

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

  calcClock.tv_sec += diff.tv_sec;
  calcClock.tv_nsec += diff.tv_nsec;

  if (calcClock.tv_nsec >= 1000000000)
  {
    calcClock.tv_sec += 1;
    calcClock.tv_nsec -= 1000000000;
  }
} /* ParallelGridCore::StopCalcClock */

/**
 * Start clock for share operations
 */
void
ParallelGridCore::StartShareClock ()
{
  int status = clock_gettime (CLOCK_MONOTONIC, &shareStart);
  ASSERT (status == 0);
} /* ParallelGridCore::StartShareClock */

/**
 * Stop clock for share operations
 */
void
ParallelGridCore::StopShareClock ()
{
  int status = clock_gettime (CLOCK_MONOTONIC, &shareStop);
  ASSERT (status == 0);

  timespec diff;
  timespec_diff (&shareStart, &shareStop, &diff);

  shareClock.tv_sec += diff.tv_sec;
  shareClock.tv_nsec += diff.tv_nsec;
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

/**
 * Share clock counters with other processes
 */
void ParallelGridCore::ShareClocks ()
{
  for (int process = 0; process < getTotalProcCount (); ++process)
  {
    uint64_t calcClockSec;
    uint64_t calcClockNSec;

    uint64_t shareClockSec;
    uint64_t shareClockNSec;

    if (process == getProcessId ())
    {
      calcClockSec = (uint64_t) calcClock.tv_sec;
      calcClockNSec = (uint64_t) calcClock.tv_nsec;

      shareClockSec = (uint64_t) shareClock.tv_sec;
      shareClockNSec = (uint64_t) shareClock.tv_nsec;
    }

    MPI_Bcast (&calcClockSec, 1, MPI_LONG_LONG, process, communicator);
    MPI_Bcast (&calcClockNSec, 1, MPI_LONG_LONG, process, communicator);
    MPI_Bcast (&shareClockSec, 1, MPI_LONG_LONG, process, communicator);
    MPI_Bcast (&shareClockNSec, 1, MPI_LONG_LONG, process, communicator);

    timespec calc;
    timespec share;

    calc.tv_sec = calcClockSec;
    calc.tv_nsec = calcClockNSec;

    share.tv_sec = shareClockSec;
    share.tv_nsec = shareClockNSec;

    calcClockAll[process] = calc;
    shareClockAll[process] = share;

    MPI_Barrier (communicator);
  }
} /* ParallelGridCore::ShareClocks */

/**
 * Set clocks to zeros
 */
void
ParallelGridCore::ClearClocks ()
{
  calcClock.tv_sec = 0;
  calcClock.tv_nsec = 0;

  shareClock.tv_sec = 0;
  shareClock.tv_nsec = 0;
} /* ParallelGridCore::ClearClocks */

#endif /* DYNAMIC_GRID */

#endif /* PARALLEL_GRID */
