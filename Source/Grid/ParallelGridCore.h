#ifndef PARALLEL_GRID_CORE_H
#define PARALLEL_GRID_CORE_H

#include "Grid.h"

#ifdef PARALLEL_GRID

#include <mpi.h>

#ifdef DYNAMIC_GRID
#include <map>
#endif /* DYNAMIC_GRID */

/**
 * Base grid of parallel grid and parallel grid coordinate
 */
#ifdef GRID_1D
#define ParallelGridBase Grid<GridCoordinate1D>
#define ParallelGridCoordinateTemplate GridCoordinate1DTemplate
#define ParallelGridCoordinate GridCoordinate1D
#define ParallelGridCoordinateFP GridCoordinateFP1D
#endif /* GRID_1D */

#ifdef GRID_2D
#define ParallelGridBase Grid<GridCoordinate2D>
#define ParallelGridCoordinateTemplate GridCoordinate2DTemplate
#define ParallelGridCoordinate GridCoordinate2D
#define ParallelGridCoordinateFP GridCoordinateFP2D
#endif /* GRID_2D */

#ifdef GRID_3D
#define ParallelGridBase Grid<GridCoordinate3D>
#define ParallelGridCoordinateTemplate GridCoordinate3DTemplate
#define ParallelGridCoordinate GridCoordinate3D
#define ParallelGridCoordinateFP GridCoordinateFP3D
#endif /* GRID_3D */

/**
 * Process ID for non-existing processes
 */
#define PID_NONE (-1)


#ifdef DYNAMIC_GRID
/**
 * Size of buffer for additional measurements
 */
#define CLOCK_BUF_SIZE 1000

/**
 * Type for share clock for different buffer sizes.
 * Type of calc clock.
 */
#ifdef MPI_DYNAMIC_CLOCK
typedef std::map<uint32_t, FPValue> ShareClock_t;
typedef FPValue CalcClock_t;
#else /* MPI_DYNAMIC_CLOCK */
typedef std::map<uint32_t, timespec> ShareClock_t;
typedef timespec CalcClock_t;
#endif /* !MPI_DYNAMIC_CLOCK */

/**
 * Type for number of iterations for different buffer sizes.
 */
typedef std::map<uint32_t, uint32_t> IterCount_t;
#endif /* DYNAMIC_GRID */


/**
 * Parallel grid buffer types.
 */
enum BufferPosition
{
#define FUNCTION(X) X,
#include "BufferPosition.inc.h"
}; /* BufferPosition */

/**
 * Class with data shared between all parallel grids on a single computational node
 */
class ParallelGridCore
{
private:

  /**
   * Opposite buffer position corresponding to buffer position
   */
  std::vector<BufferPosition> oppositeDirections;

  /**
   * Current node (process) identificator.
   */
  int processId;

  /**
   * Overall count of nodes (processes).
   */
  int totalProcCount;

#ifndef DYNAMIC_GRID
  /**
   * Process ids corresponding to directions
   */
  std::vector<int> directions;
#endif

#ifdef DYNAMIC_GRID
  /**
   * States of all processes
   */
  std::vector<int> nodeState;

  /**
   * Lists of nodes for each direction (the closest come in the beggining of nodesForDirections[dir])
   */
  std::vector< std::vector<int> > nodesForDirections;
#endif

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)

  /**
   * Number of nodes in the grid by Ox axis
   */
  int nodeGridSizeX;

#endif /* GRID_1D || GRID_2D || GRID_3D */
#if defined (GRID_2D) || defined (GRID_3D)

  /**
   * Number of nodes in the grid by Oy axis
   */
  int nodeGridSizeY;

#endif /* GRID_2D || GRID_3D */
#ifdef GRID_3D

  /**
   * Number of nodes in the grid by Oz axis
   */
  int nodeGridSizeZ;

#endif /* GRID_3D */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  /**
   * Precalculated value of nodeGridSizeX * nodeGridSizeY
   */
  int nodeGridSizeXY;

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ

  /**
   * Precalculated value of nodeGridSizeY * nodeGridSizeZ
   */
  int nodeGridSizeYZ;

#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ

  /**
   * Precalculated value of nodeGridSizeX * nodeGridSizeZ
   */
  int nodeGridSizeXZ;

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ

  /**
   * Precalculated value of nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ
   */
  int nodeGridSizeXYZ;

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  /**
   * Flag whether computational node has left neighbor
   */
  bool hasL;

  /**
   * Flag whether computational node has right neighbor
   */
  bool hasR;

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  /**
   * Flag whether computational node has down neighbor
   */
  bool hasD;

  /**
   * Flag whether computational node has up neighbor
   */
  bool hasU;

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  /**
   * Flag whether computational node has back neighbor
   */
  bool hasB;

  /**
   * Flag whether computational node has front neighbor
   */
  bool hasF;

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#ifdef DYNAMIC_GRID
  /**
   * Latest clock counter for calculations of all processes
   */
  std::vector<CalcClock_t> calcClockSumBetweenRebalance;

  /**
   * Number of points, on which computations are performed
   */
  std::vector<grid_coord> calcClockCountBetweenRebalance;

  /**
   * Latest clock counter for share operations of all with all processes
   */
  std::vector< std::vector<ShareClock_t> > shareClockSumBetweenRebalance;

  /**
   * Number of points that are shared at a single time step
   */
  std::vector< uint32_t > shareClockCountBetweenRebalance;

  /**
   * Number of interations for different buffer sizes
   */
  std::vector< std::vector<IterCount_t> > shareClockIterBetweenRebalance;

  /**
   * Total values for performance: perf is totalSumPerfPointsPerProcess/totalSumPerfTimePerProcess then
   */
  std::vector<FPValue> totalSumPerfPointsPerProcess;
  std::vector<FPValue> totalSumPerfTimePerProcess;

  /**
   * Total values for latency: latency is totalSumLatencyPerConnection/totalSumLatencyCountPerConnection then
   */
  std::vector< std::vector<FPValue> > totalSumLatencyPerConnection;
  std::vector< std::vector<FPValue> > totalSumLatencyCountPerConnection;

  /**
   * Total values of bandwidth: latency is totalSumBandwidthPerConnection/totalSumBandwidthCountPerConnection then
   */
  std::vector< std::vector<FPValue> > totalSumBandwidthPerConnection;
  std::vector< std::vector<FPValue> > totalSumBandwidthCountPerConnection;

private:
  /*
   * TODO: Use this
   */
  time_step T_balance;
  time_step T_perf;

  /**
   * Helper buffers used for sharing
   */
#ifdef MPI_DYNAMIC_CLOCK
  FPValue *shareClockSec_buf;
#else
  uint64_t *shareClockSec_buf;
  uint64_t *shareClockNSec_buf;
#endif
  uint32_t *shareClockBufSize_buf;

  uint32_t *shareClockBufSize2_buf;
  uint32_t *shareClockIter_buf;

  /**
   * Helper clock counter for start of calculations of current process
   */
  timespec calcStart;

  /**
   * Helper clock counter for stop of calculations of current process
   */
  timespec calcStop;

#ifndef MPI_DYNAMIC_CLOCK
  /**
   * Helper clock counter for start of share operations of current process
   */
  timespec shareStart;

  /**
   * Helper clock counter for stop of share operations of current process
   */
  timespec shareStop;
#endif
#endif /* DYNAMIC_GRID */

  /**
   * Flag whether to use manual virtual topology or not
   */
  bool doUseManualTopology;

  /**
   * Virtual topology size, which was specified manually (not used in case of optimal virtual topology)
   */
  ParallelGridCoordinate topologySize;

  /**
   * Communicator for all processes, used in computations
   * (could differ from MPI_COMM_WORLD on the processes, which are not used in computations)
   */
  MPI_Comm communicator;

#ifndef COMBINED_SENDRECV
  std::vector<bool> isEvenForDirection;
#endif /* !COMBINED_SENDRECV */

private:

  void initOppositeDirections ();
  BufferPosition getOpposite (BufferPosition);

  /*
   * TODO: make names start with lower case
   */
  void NodeGridInit (ParallelGridCoordinate);
  void ParallelGridCoreConstructor (ParallelGridCoordinate);
  void InitBufferFlags ();
  void InitDirections ();

#ifdef DYNAMIC_GRID
  void timespec_diff (struct timespec *, struct timespec *, struct timespec *);
  void timespec_sum (struct timespec *, struct timespec *, struct timespec *);
  void timespec_avg (struct timespec *, struct timespec *, struct timespec *);

  void SetNodesForDirections ();
#endif /* DYNAMIC_GRID */

public:

  ParallelGridCore (int, int, ParallelGridCoordinate, bool, ParallelGridCoordinate);
  ~ParallelGridCore ();

  /**
   * Getter for communicator for all processes, used in computations
   *
   * @return communicator for all processes, used in computations
   */
  MPI_Comm getCommunicator () const
  {
    return communicator;
  } /* getCommunicator */

#ifndef COMBINED_SENDRECV
  const std::vector<bool> &getIsEvenForDirection () const
  {
    return isEvenForDirection;
  }
#endif /* !COMBINED_SENDRECV */

#if defined (GRID_1D) || defined (GRID_2D) || defined (GRID_3D)

  /**
   * Getter for number of nodes in the grid by Ox axis
   *
   * @return number of nodes in the grid by Ox axis
   */
  int getNodeGridSizeX () const
  {
    return nodeGridSizeX;
  } /* getNodeGridSizeX */

#endif /* GRID_1D || GRID_2D || GRID_3D */

#if defined (GRID_2D) || defined (GRID_3D)

  /**
   * Getter for number of nodes in the grid by Oy axis
   *
   * @return number of nodes in the grid by Oy axis
   */
  int getNodeGridSizeY () const
  {
    return nodeGridSizeY;
  } /* getNodeGridSizeY */

#endif /* GRID_2D || GRID_3D */

#ifdef GRID_3D

  /**
   * Getter for number of nodes in the grid by Oz axis
   *
   * @return number of nodes in the grid by Oz axis
   */
  int getNodeGridSizeZ () const
  {
    return nodeGridSizeZ;
  } /* getNodeGridSizeZ */

#endif /* GRID_3D */

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  /**
   * Getter for precalculated value of nodeGridSizeX * nodeGridSizeY
   *
   * @return precalculated value of nodeGridSizeX * nodeGridSizeY
   */
  int getNodeGridSizeXY () const
  {
    return nodeGridSizeXY;
  } /* getNodeGridSizeXY */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_3D_XYZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ

  /**
   * Getter for precalculated value of nodeGridSizeY * nodeGridSizeZ
   *
   * @return precalculated value of nodeGridSizeY * nodeGridSizeZ
   */
  int getNodeGridSizeYZ () const
  {
    return nodeGridSizeYZ;
  } /* getNodeGridSizeYZ */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */
#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ

  /**
   * Getter for precalculated value of nodeGridSizeX * nodeGridSizeZ
   *
   * @return precalculated value of nodeGridSizeX * nodeGridSizeZ
   */
  int getNodeGridSizeXZ () const
  {
    return nodeGridSizeXZ;
  } /* getNodeGridSizeXZ */

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */
#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ

  /**
   * Getter for precalculated value of nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ
   *
   * @return precalculated value of nodeGridSizeX * nodeGridSizeY * nodeGridSizeZ
   */
  int getNodeGridSizeXYZ () const
  {
    return nodeGridSizeXYZ;
  } /* getNodeGridSizeXYZ */

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#ifdef GRID_1D
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)
  int getNodeGrid (int coord) const
  {
    ASSERT (coord >= 0 && coord < getTotalProcCount ());
    return coord;
  }
#endif
#endif

#ifdef GRID_2D
  int getNodeGrid (int coord1, int coord2) const
  {
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
    ASSERT (coord1 >= 0 && coord1 < getNodeGridSizeX ());
#else
    ASSERT (coord1 == 0);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
    ASSERT (coord2 >= 0 && coord2 < getNodeGridSizeY ());
#else
    ASSERT (coord2 == 0);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)
    return coord1;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
    return coord2;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
    return coord2 * getNodeGridSizeX () + coord1;
#endif
  }
#endif

#ifdef GRID_3D
  int getNodeGrid (int coord1, int coord2, int coord3) const
  {
#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    ASSERT (coord1 >= 0 && coord1 < getNodeGridSizeX ());
#else
    ASSERT (coord1 == 0);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    ASSERT (coord2 >= 0 && coord2 < getNodeGridSizeY ());
#else
    ASSERT (coord2 == 0);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) \
    || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    ASSERT (coord3 >= 0 && coord3 < getNodeGridSizeZ ());
#else
    ASSERT (coord3 == 0);
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X)
    return coord1;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y)
    return coord2;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
    return coord3;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY)
    return coord2 * getNodeGridSizeX () + coord1;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
    return coord3 * getNodeGridSizeY () + coord2;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
    return coord3 * getNodeGridSizeX () + coord1;
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
    return coord3 * getNodeGridSizeXY () + coord2 * getNodeGridSizeX () + coord1;
#endif
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  /**
   * Get coordinate of process in the nodes' grid by Ox axis for current process
   *
   * @return coordinate of process in the nodes' grid by Ox axis for current process
   */
  int getNodeGridX () const
  {
    return getNodeGridX (processId);
  }
  int getNodeGridX (int) const;

  /**
   * Getter for flag whether computational node has left neighbor
   *
   * @return flag whether computational node has left neighbor
   */
  bool getHasL () const
  {
    return hasL;
  } /* getHasL */
  bool getHasL (int) const;

  /**
   * Getter for flag whether computational node has right neighbor
   *
   * @return flag whether computational node has right neighbor
   */
  bool getHasR () const
  {
    return hasR;
  } /* getHasR */
  bool getHasR (int) const;

#endif /* PARALLEL_BUFFER_DIMENSION_1D_X || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  /**
   * Get coordinate of process in the nodes' grid by Oy axis for current process
   *
   * @return coordinate of process in the nodes' grid by Oy axis for current process
   */
  int getNodeGridY () const
  {
    return getNodeGridY (processId);
  }
  int getNodeGridY (int) const;

  /**
   * Getter for flag whether computational node has down neighbor
   *
   * @return flag whether computational node has down neighbor
   */
  bool getHasD () const
  {
    return hasD;
  } /* getHasD */
  bool getHasD (int) const;

  /**
   * Getter for flag whether computational node has up neighbor
   *
   * @return flag whether computational node has up neighbor
   */
  bool getHasU () const
  {
    return hasU;
  } /* getHasU */
  bool getHasU (int) const;

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Y || PARALLEL_BUFFER_DIMENSION_2D_XY ||
          PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  /**
   * Get coordinate of process in the nodes' grid by Oz axis for current process
   *
   * @return coordinate of process in the nodes' grid by Oz axis for current process
   */
  int getNodeGridZ () const
  {
    return getNodeGridZ (processId);
  }
  int getNodeGridZ (int) const;

  /**
   * Getter for flag whether computational node has back neighbor
   *
   * @return flag whether computational node has back neighbor
   */
  bool getHasB () const
  {
    return hasB;
  } /* getHasB */
  bool getHasB (int) const;

  /**
   * Getter for flag whether computational node has front neighbor
   *
   * @return flag whether computational node has front neighbor
   */
  bool getHasF () const
  {
    return hasF;
  } /* getHasF */
  bool getHasF (int) const;

#endif /* PARALLEL_BUFFER_DIMENSION_1D_Z || PARALLEL_BUFFER_DIMENSION_2D_YZ ||
          PARALLEL_BUFFER_DIMENSION_2D_XZ || PARALLEL_BUFFER_DIMENSION_3D_XYZ */

  /**
   * Getter for id of process corresponding to current computational node
   *
   * @return id of process corresponding to current computational node
   */
  int getProcessId () const
  {
    return processId;
  } /* getProcessId */

  /**
   * Getter for overall count of nodes (processes)
   *
   * @return overall count of nodes (processes)
   */
  int getTotalProcCount () const
  {
    return totalProcCount;
  } /* getTotalProcCount */

  /**
   * Getter for opposite buffer position corresponding to buffer position
   *
   * @return opposite buffer position corresponding to buffer position
   */
  const std::vector<BufferPosition> &getOppositeDirections () const
  {
    return oppositeDirections;
  } /* getOppositeDirections */

  int getNodeForDirection (BufferPosition) const;

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ) || defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)

  void initOptimal (grid_coord, grid_coord, int &, int &);

#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY || PARALLEL_BUFFER_DIMENSION_2D_YZ || PARALLEL_BUFFER_DIMENSION_2D_XZ */

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)

  void initOptimal (grid_coord, grid_coord, grid_coord, int &, int &, int &);

#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#ifdef DYNAMIC_GRID

  std::vector<int> &getNodeState ()
  {
    return nodeState;
  }

  void StartCalcClock ();
  void StopCalcClock ();

  void StartShareClock (int, uint32_t);
  void StopShareClock (int, uint32_t);

  void ShareClocks ();
  void ShareCalcClocks ();
  void ShareShareClocks ();

  void ClearCalcClocks ();
  void ClearShareClocks ();

  FPValue getTotalSumPerfPointsPerProcess (int pid) const
  {
    ASSERT (totalProcCount == totalSumPerfPointsPerProcess.size ());
    ASSERT (pid >= 0 && pid < totalProcCount);
    return totalSumPerfPointsPerProcess[pid];
  }

  void setTotalSumPerfPointsPerProcess (int pid, FPValue val)
  {
    ASSERT (totalProcCount == totalSumPerfPointsPerProcess.size ());
    ASSERT (pid >= 0 && pid < totalProcCount);
    totalSumPerfPointsPerProcess[pid] = val;
  }

  void increaseTotalSumPerfPointsPerProcess (int pid, FPValue val)
  {
    setTotalSumPerfPointsPerProcess (pid, getTotalSumPerfPointsPerProcess (pid) + val);
  }

  FPValue getTotalSumPerfTimePerProcess (int pid) const
  {
    ASSERT (totalProcCount == totalSumPerfTimePerProcess.size ());
    ASSERT (pid >= 0 && pid < totalProcCount);
    return totalSumPerfTimePerProcess[pid];
  }

  void setTotalSumPerfTimePerProcess (int pid, FPValue val)
  {
    ASSERT (totalProcCount == totalSumPerfTimePerProcess.size ());
    ASSERT (pid >= 0 && pid < totalProcCount);
    totalSumPerfTimePerProcess[pid] = val;
  }

  void increaseTotalSumPerfTimePerProcess (int pid, FPValue val)
  {
    setTotalSumPerfTimePerProcess (pid, getTotalSumPerfTimePerProcess (pid) + val);
  }

  FPValue getTotalSumLatencyPerConnection (int pid1, int pid2)
  {
    ASSERT (totalProcCount == totalSumLatencyPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == totalSumLatencyPerConnection[pid1].size ());
    return totalSumLatencyPerConnection[pid1][pid2];
  }

  void setTotalSumLatencyPerConnection (int pid1, int pid2, FPValue val)
  {
    ASSERT (totalProcCount == totalSumLatencyPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == totalSumLatencyPerConnection[pid1].size ());
    totalSumLatencyPerConnection[pid1][pid2] = val;
  }

  void increaseTotalSumLatencyPerConnection (int pid1, int pid2, FPValue val)
  {
    setTotalSumLatencyPerConnection (pid1, pid2, getTotalSumLatencyPerConnection (pid1, pid2) + val);
  }

  FPValue getTotalSumLatencyCountPerConnection (int pid1, int pid2)
  {
    ASSERT (totalProcCount == totalSumLatencyCountPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == totalSumLatencyCountPerConnection[pid1].size ());
    return totalSumLatencyCountPerConnection[pid1][pid2];
  }

  void setTotalSumLatencyCountPerConnection (int pid1, int pid2, FPValue val)
  {
    ASSERT (totalProcCount == totalSumLatencyCountPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == totalSumLatencyCountPerConnection[pid1].size ());
    totalSumLatencyCountPerConnection[pid1][pid2] = val;
  }

  void increaseTotalSumLatencyCountPerConnection (int pid1, int pid2, FPValue val)
  {
    setTotalSumLatencyCountPerConnection (pid1, pid2, getTotalSumLatencyCountPerConnection (pid1, pid2) + val);
  }

  FPValue getTotalSumBandwidthPerConnection (int pid1, int pid2)
  {
    ASSERT (totalProcCount == totalSumBandwidthPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == totalSumBandwidthPerConnection[pid1].size ());
    return totalSumBandwidthPerConnection[pid1][pid2];
  }

  void setTotalSumBandwidthPerConnection (int pid1, int pid2, FPValue val)
  {
    ASSERT (totalProcCount == totalSumBandwidthPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == totalSumBandwidthPerConnection[pid1].size ());
    totalSumBandwidthPerConnection[pid1][pid2] = val;
  }

  void increaseTotalSumBandwidthPerConnection (int pid1, int pid2, FPValue val)
  {
    setTotalSumBandwidthPerConnection (pid1, pid2, getTotalSumBandwidthPerConnection (pid1, pid2) + val);
  }

  FPValue getTotalSumBandwidthCountPerConnection (int pid1, int pid2)
  {
    ASSERT (totalProcCount == totalSumBandwidthCountPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == totalSumBandwidthCountPerConnection[pid1].size ());
    return totalSumBandwidthCountPerConnection[pid1][pid2];
  }

  void setTotalSumBandwidthCountPerConnection (int pid1, int pid2, FPValue val)
  {
    ASSERT (totalProcCount == totalSumBandwidthCountPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == totalSumBandwidthCountPerConnection[pid1].size ());
    totalSumBandwidthCountPerConnection[pid1][pid2] = val;
  }

  void increaseTotalSumBandwidthCountPerConnection (int pid1, int pid2, FPValue val)
  {
    setTotalSumBandwidthCountPerConnection (pid1, pid2, getTotalSumBandwidthCountPerConnection (pid1, pid2) + val);
  }

  FPValue getLatencyForConnection (int pid1, int pid2)
  {
    FPValue count = getTotalSumLatencyCountPerConnection (pid1, pid2);
    if (count == 0)
    {
      return 0;
    }

    return getTotalSumLatencyPerConnection (pid1, pid2) / count;
  }

  FPValue getBandwidthForConnection (int pid1, int pid2)
  {
    FPValue count = getTotalSumBandwidthCountPerConnection (pid1, pid2);
    if (count == 0)
    {
      return 0;
    }

    return getTotalSumBandwidthPerConnection (pid1, pid2) / count;
  }

  /**
   * Getter for calculations clock
   *
   * @return calculations clock
   */
  CalcClock_t getCalcClock (int pid) const
  {
    ASSERT (pid >= 0 && pid < totalProcCount);
#ifdef MPI_DYNAMIC_CLOCK
    ASSERT (nodeState[pid] == 1 && calcClockSumBetweenRebalance[pid] > 0
            || nodeState[pid] == 0 && calcClockSumBetweenRebalance[pid] == 0);
#else
    ASSERT (nodeState[pid] == 1 && (calcClockSumBetweenRebalance[pid].tv_sec > 0 || calcClockSumBetweenRebalance[pid].tv_nsec > 0)
            || nodeState[pid] == 0 && calcClockSumBetweenRebalance[pid].tv_sec == 0 && calcClockSumBetweenRebalance[pid].tv_nsec == 0);
#endif
    return calcClockSumBetweenRebalance[pid];
  } /* getCalcClock */

  uint32_t getCalcClockCount (int pid) const
  {
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (nodeState[pid] == 1 && calcClockCountBetweenRebalance[pid] > 0
            || nodeState[pid] == 0 && calcClockCountBetweenRebalance[pid] == 0);
    return calcClockCountBetweenRebalance[pid];
  }

  void setCalcClockCount (int pid, uint32_t val)
  {
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (nodeState[pid] == 1 && val > 0
            || nodeState[pid] == 0 && val == 0);
    calcClockCountBetweenRebalance[pid] = val;
  }

  const ShareClock_t & getShareClock (int process, int pid) const
  {
    ASSERT (process >= 0 && process < totalProcCount);
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (nodeState[process] == 1 && nodeState[pid] == 1
            || (nodeState[process] == 0 || nodeState[pid] == 0) && shareClockSumBetweenRebalance[process][pid].empty ());
    return shareClockSumBetweenRebalance[process][pid];
  }

  const ShareClock_t & getShareClockCur (int pid) const
  {
    return getShareClock (processId, pid);
  }

  // const bool checkShareClockCount (int pid1, int pid2) const
  // {
  //   ASSERT (pid1 >= 0 && pid1 < totalProcCount);
  //   ASSERT (pid2 >= 0 && pid2 < totalProcCount);
  //   ASSERT (nodeState[processId] == 1 && nodeState[pid] == 1);
  //   return shareClockCountBetweenRebalance[pid] > 0;
  // }

  const uint32_t & getShareClockCountCur (int pid) const
  {
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (nodeState[processId] == 1 && nodeState[pid] == 1 && shareClockCountBetweenRebalance[pid] > 0
            || (nodeState[processId] == 0 || nodeState[pid] == 0) && shareClockCountBetweenRebalance[pid] == 0);
    return shareClockCountBetweenRebalance[pid];
  }

  const uint32_t & getShareClockIter (int process, int pid, uint32_t bufSize)
  {
    ASSERT (process >= 0 && process < totalProcCount);
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (shareClockIterBetweenRebalance[process][pid].find (bufSize) != shareClockIterBetweenRebalance[process][pid].end ());
    ASSERT (nodeState[process] == 1 && nodeState[pid] == 1 && shareClockIterBetweenRebalance[process][pid][bufSize] > 0
            || (nodeState[process] == 0 || nodeState[pid] == 0) && shareClockIterBetweenRebalance[process][pid][bufSize] == 0);
    return shareClockIterBetweenRebalance[process][pid][bufSize];
  }

  const uint32_t & getShareClockIterCur (int pid, uint32_t bufSize)
  {
    return getShareClockIter (processId, pid, bufSize);
  }

#ifdef MPI_DYNAMIC_CLOCK
  void setShareClockCur (int pid, uint32_t shareSize, FPValue val)
#else
  void setShareClockCur (int pid, uint32_t shareSize, timespec val)
#endif
  {
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (nodeState[processId] == 1 && nodeState[pid] == 1
            || (nodeState[processId] == 0 || nodeState[pid] == 0) && val == 0);
    shareClockSumBetweenRebalance[processId][pid][shareSize] = val;
  }

  void setShareClockCountCur (int pid, uint32_t val)
  {
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (nodeState[processId] == 1 && nodeState[pid] == 1 && val > 0
            || (nodeState[processId] == 0 || nodeState[pid] == 0) && val == 0);
    shareClockCountBetweenRebalance[pid] = val;
  }

  void setShareClockIterCur (int pid, uint32_t bufSize, uint32_t val)
  {
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (nodeState[processId] == 1 && nodeState[pid] == 1 && val > 0
            || (nodeState[processId] == 0 || nodeState[pid] == 0) && val == 0);
    shareClockIterBetweenRebalance[processId][pid][bufSize] = val;
  }

#endif /* DYNAMIC_GRID */

  /*
   * TODO: move out of ParallelGridCore
   */
  /**
   * Find greatest common divider of two integer numbers
   *
   * @return greatest common divider of two integer numbers
   */
  static grid_coord greatestCommonDivider (grid_coord a, /**< first integer number */
                                           grid_coord b) /**< second integer number */
  {
    if (b == 0)
    {
      return a;
    }
    else
    {
      return greatestCommonDivider (b, a % b);
    }
  } /* greatestCommonDivider */
}; /* ParallelGridCore */

#endif /* PARALLEL_GRID */

#endif /* PARALLEL_GRID_CORE_H */
