#ifndef PARALLEL_GRID_GROUP_H
#define PARALLEL_GRID_GROUP_H

#include "Grid.h"
#include "Parallel.h"
#include "ParallelGridCore.h"
#include "DynamicGrid.h"

#ifdef PARALLEL_GRID

/**
 * Type of buffer of values
 */
typedef std::vector<FieldValue> VectorBufferValues;

/**
 * Type of vector of buffers
 */
typedef std::vector<VectorBufferValues> VectorBuffers;

#define INVALID_GROUP (-1)

/**
 * Class with data shared between all members of one parallel group
 */
class ParallelGridGroup
{
private:

  static ParallelGridCore *parallelGridCore;

#ifdef DEBUG_INFO
  /**
   * Coordinate types for ParallelGridCoordinate, corresponding to this ParallelGridGroup
   */
  CoordinateType ct1;
  CoordinateType ct2;
  CoordinateType ct3;
#endif /* DEBUG_INFO */

  /**
   * ========================================
   * Parameters corresponding to parallelism
   * ========================================
   */

  /**
   * Current share step
   */
  time_step shareStep;

  /**
   * Step at which to perform share operations for grids for synchronization of computational nodes
   */
  time_step shareStepLimit;

  /**
   * Offset from timestep t (in form t+timeOffset/2), at which to perfrom share operations
   */
  int timeOffset;

  /**
   * Array of coordinate in grid from which to start send values corresponding to direction
   */
  std::vector<ParallelGridCoordinate> sendStart;

  /**
   * Array of coordinate in grid until which to send values corresponding to direction
   */
  std::vector<ParallelGridCoordinate> sendEnd;

  /**
   * Array of coordinate in grid from which to start saving received values corresponding to direction
   */
  std::vector<ParallelGridCoordinate> recvStart;

  /**
   * Array of coordinate in grid until which to save received values corresponding to direction
   */
  std::vector<ParallelGridCoordinate> recvEnd;

  /**
   * ========================================
   * Parameters corresponding to data in grid
   * ========================================
   */

  /**
   * Size of current node plus size of buffers
   */
  ParallelGridCoordinate size;

  /**
   * Total size of grid (size, which is specified at its declaration)
   */
  ParallelGridCoordinate totalSize;

  /**
   * Size of grid for current node without buffers (raw data which is assigned to current computational node)
   */
  ParallelGridCoordinate currentSize;

  /**
   * Absolute start position of chunk of current node, including buffers!
   */
  ParallelGridCoordinate posStart;

  /**
   * Size of buffer zone
   */
  ParallelGridCoordinate bufferSize;

  /**
   * Send buffers to send values from it
   *
   * TODO: remove copy to this buffer before send
   */
  VectorBuffers buffersSend;

  /**
   * Receive buffers to receive values into
   *
   * TODO: remove copy from this buffer after receive
   */
  VectorBuffers buffersReceive;

  int storedSteps;

  /**
   * Name of the parallel grid group.
   */
  std::string groupName;

public:

  /**
   * Initialize parallel grid core
   */
  static void initializeParallelCore (ParallelGridCore *core) /**< new parallel grid core */
  {
    ASSERT (parallelGridCore == NULLPTR);

    parallelGridCore = core;
  } /* initializeParallelCore */

public:

  ParallelGridGroup (ParallelGridCoordinate totSize,
                     ParallelGridCoordinate bufSize,
                     time_step stepLimit,
                     ParallelGridCoordinate curSize,
                     int storedTimeSteps,
                     int tOffset,
                     const char *name = "unnamed");
  ~ParallelGridGroup () {}

  bool match (ParallelGridCoordinate totSize,
              ParallelGridCoordinate bufSize,
              time_step stepLimit,
              ParallelGridCoordinate curSize,
              int storedTimeSteps,
              int tOffset) const;

  void ParallelGridGroupConstructor ();

  void InitBuffers ();

  void SendReceiveCoordinatesInit ();

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void SendReceiveCoordinatesInit1D_X ();

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void SendReceiveCoordinatesInit1D_Y ();

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void SendReceiveCoordinatesInit1D_Z ();

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void SendReceiveCoordinatesInit2D_XY ();

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void SendReceiveCoordinatesInit2D_YZ ();

#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void SendReceiveCoordinatesInit2D_XZ ();

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void SendReceiveCoordinatesInit3D_XYZ ();

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  void initializeStartPosition (ParallelGridCoordinate);
  void gatherStartPosition ();

  time_step getShareStepLimit () const
  {
    return shareStepLimit;
  }
  void setShareStepLimit (time_step step)
  {
    shareStepLimit = step;
  }

#ifdef DEBUG_INFO
  CoordinateType get_ct1 () const
  {
    return ct1;
  }
  CoordinateType get_ct2 () const
  {
    return ct2;
  }
  CoordinateType get_ct3 () const
  {
    return ct3;
  }
#endif /* DEBUG_INFO */

  /**
   * Get parallel grid core
   */
  static ParallelGridCore * getParallelCore ()
  {
    ASSERT (parallelGridCore != NULLPTR);
    return parallelGridCore;
  } /* getParallelCore */

  /**
   * Increase share step
   */
  void nextShareStep ()
  {
    ++shareStep;
  } /* nextShareStep */

  bool isShareTime () const
  {
    return shareStep == shareStepLimit;
  }

  /**
   * Get share step
   *
   * @return share step
   */
  time_step getShareStep () const
  {
    return shareStep;
  } /* getShareStep */

  void setShareStep (time_step step)
  {
    shareStep = step;
  }

  /**
   * Set share step to zero
   */
  void zeroShareStep ()
  {
    shareStep = 0;
  } /* zeroShareStepE */

  ParallelGridCoordinate getSendStart (int dir) const
  {
    return sendStart[dir];
  }
  ParallelGridCoordinate getSendEnd (int dir) const
  {
    return sendEnd[dir];
  }
  ParallelGridCoordinate getRecvStart (int dir) const
  {
    return recvStart[dir];
  }
  ParallelGridCoordinate getRecvEnd (int dir) const
  {
    return recvEnd[dir];
  }

  ParallelGridCoordinate getSize () const
  {
    return size;
  }

  ParallelGridCoordinate getTotalSize () const
  {
    return totalSize;
  }
  ParallelGridCoordinate getCurrentSize () const
  {
    return currentSize;
  }
  ParallelGridCoordinate getPosStart () const
  {
    return posStart;
  }
  ParallelGridCoordinate getBufferSize () const
  {
    return bufferSize;
  }

  VectorBuffers & getBuffersSend ()
  {
    return buffersSend;
  }
  VectorBuffers & getBuffersReceive ()
  {
    return buffersReceive;
  }

  /**
   * Get absolute position corresponding to first value in grid for current computational node (considering buffers)
   *
   * @return absolute position corresponding to first value in grid for current computational node (considering buffers)
   */
  ParallelGridCoordinate getStartPosition () const
  {
    return posStart;
  } /* getStartPosition */

  /**
   * Get absolute position corresponding to first value in grid for current computational node (not considering buffers)
   *
   * @return absolute position corresponding to first value in grid for current computational node (not considering buffers)
   */
  ParallelGridCoordinate getChunkStartPosition () const
  {
    grid_coord left_coord, right_coord;
    grid_coord down_coord, up_coord;
    grid_coord back_coord, front_coord;

    initBufferOffsets (left_coord, right_coord, down_coord, up_coord, back_coord, front_coord);

#ifdef GRID_1D
    return posStart + GridCoordinate1D (left_coord
#ifdef DEBUG_INFO
                                        , getSize ().getType1 ()
#endif /* DEBUG_INFO */
                                        );
#endif /* GRID_1D */

#ifdef GRID_2D
    return posStart + GridCoordinate2D (left_coord, down_coord
#ifdef DEBUG_INFO
                                        , getSize ().getType1 ()
                                        , getSize ().getType2 ()
#endif /* DEBUG_INFO */
                                        );
#endif /* GRID_2D */

#ifdef GRID_3D
    return posStart + GridCoordinate3D (left_coord, down_coord, back_coord
#ifdef DEBUG_INFO
                                        , getSize ().getType1 ()
                                        , getSize ().getType2 ()
                                        , getSize ().getType3 ()
#endif /* DEBUG_INFO */
                                        );
#endif /* GRID_3D */
  } /* getChunkStartPosition */

  /**
   * Get first coordinate from which to perform computations at current step
   *
   * @return first coordinate from which to perform computations at current step
   */
  ParallelGridCoordinate getComputationStart
    (const ParallelGridCoordinate & diffPosStart) const /**< layout coordinate modifier */
  {
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    grid_coord diffX = diffPosStart.get1 ();
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    grid_coord diffY = diffPosStart.get2 ();
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
    grid_coord diffZ = diffPosStart.get3 ();
#endif /* GRID_3D */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    if (parallelGridCore->getHasL ())
    {
      diffX = getShareStep () + 1;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    if (parallelGridCore->getHasD ())
    {
      diffY = getShareStep () + 1;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    if (parallelGridCore->getHasB ())
    {
      diffZ = getShareStep () + 1;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    grid_coord px = diffX;
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    grid_coord py = diffY;
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
    grid_coord pz = diffZ;
#endif /* GRID_3D */

#ifdef GRID_1D
    return ParallelGridCoordinate (px
#ifdef DEBUG_INFO
                                   , get_ct1 ()
#endif /* DEBUG_INFO */
                                   );
#endif /* GRID_1D */
#ifdef GRID_2D
    return ParallelGridCoordinate (px, py
#ifdef DEBUG_INFO
                                   , get_ct1 (), get_ct2 ()
#endif /* DEBUG_INFO */
                                   );
#endif /* GRID_2D */
#ifdef GRID_3D
    return ParallelGridCoordinate (px, py, pz
#ifdef DEBUG_INFO
                                   , get_ct1 (), get_ct2 (), get_ct3 ()
#endif /* DEBUG_INFO */
                                   );
#endif /* GRID_3D */
  } /* getComputationStart */

  /**
   * Get last coordinate until which to perform computations at current step
   *
   * @return last coordinate until which to perform computations at current step
   */
  ParallelGridCoordinate getComputationEnd
    (const ParallelGridCoordinate & diffPosEnd,
     const ParallelGridCoordinate & size) const /**< layout coordinate modifier */
  {
#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    grid_coord diffX = diffPosEnd.get1 ();
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    grid_coord diffY = diffPosEnd.get2 ();
#endif /* GRID_2D || GRID_3D */
#if defined (GRID_3D)
    grid_coord diffZ = diffPosEnd.get3 ();
#endif /* GRID_3D */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    if (parallelGridCore->getHasR ())
    {
      diffX = getShareStep () + 1;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    if (parallelGridCore->getHasU ())
    {
      diffY = getShareStep () + 1;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    if (parallelGridCore->getHasF ())
    {
      diffZ = getShareStep () + 1;
    }
#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)
    grid_coord px = size.get1 () - diffX;
#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)
    grid_coord py = size.get2 () - diffY;
#endif /* GRID_2D || GRID_3D */
#ifdef GRID_3D
    grid_coord pz = size.get3 () - diffZ;
#endif /* GRID_3D */

#ifdef GRID_1D
    return ParallelGridCoordinate (px
#ifdef DEBUG_INFO
                                   , get_ct1 ()
#endif /* DEBUG_INFO */
                                   );
#endif /* GRID_1D */
#ifdef GRID_2D
    return ParallelGridCoordinate (px, py
#ifdef DEBUG_INFO
                                   , get_ct1 (), get_ct2 ()
#endif /* DEBUG_INFO */
                                   );
#endif /* GRID_2D */
#ifdef GRID_3D
    return ParallelGridCoordinate (px, py, pz
#ifdef DEBUG_INFO
                                   , get_ct1 (), get_ct2 (), get_ct3 ()
#endif /* DEBUG_INFO */
                                   );
#endif /* GRID_3D */
  } /* ParallelGridGroup::getComputationEnd */

  /**
   * Initialize buffer offsets for computational node
   */
  void initBufferOffsets (grid_coord &left_coord, /**< out: left buffer size */
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
    if (parallelGridCore->getNodeForDirection (LEFT) != PID_NONE)
    {
      left_coord = bufferSize.get1 ();
    }
    else
    {
      ASSERT (left_coord == 0);
    }

    right_coord = 0;
    if (parallelGridCore->getNodeForDirection (RIGHT) != PID_NONE)
    {
      right_coord = bufferSize.get1 ();
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
    if (parallelGridCore->getNodeForDirection (DOWN) != PID_NONE)
    {
      down_coord = bufferSize.get2 ();
    }
    else
    {
      ASSERT (down_coord == 0);
    }

    up_coord = 0;
    if (parallelGridCore->getNodeForDirection (UP) != PID_NONE)
    {
      up_coord = bufferSize.get2 ();
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
    if (parallelGridCore->getNodeForDirection (BACK) != PID_NONE)
    {
      back_coord = bufferSize.get3 ();
    }
    else
    {
      ASSERT (back_coord == 0);
    }

    front_coord = 0;
    if (parallelGridCore->getNodeForDirection (FRONT) != PID_NONE)
    {
      front_coord = bufferSize.get3 ();
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
  } /* initBufferOffsets */
}; /* ParallelGridGroup */

#endif /* PARALLEL_GRID */

#endif /* !PARALLEL_GRID_GROUP_H */
