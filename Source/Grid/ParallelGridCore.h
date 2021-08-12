/*
 * Copyright (C) 2017 Gleb Balykov
 *
 * This file is part of fdtd3d.
 *
 * fdtd3d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * fdtd3d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with fdtd3d; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef PARALLEL_GRID_CORE_H
#define PARALLEL_GRID_CORE_H

#include "Grid.h"
#include "Parallel.h"
#include "DynamicGrid.h"

#ifdef PARALLEL_GRID

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
#else /* !DYNAMIC_GRID */
  /**
   * Dynamic data gather during execution
   */
public:
  DynamicGridInfo dynamicInfo;
private:
#endif /* DYNAMIC_GRID */

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
  void SetNodesForDirections (int);
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
    return dynamicInfo.nodeState;
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

#ifdef DYNAMIC_GRID
private:

  void setTotalSumPerfPointsPerProcess (int pid, DOUBLE val)
  {
    ASSERT (totalProcCount == dynamicInfo.totalSumPerfPointsPerProcess.size ());
    ASSERT (pid >= 0 && pid < totalProcCount);
    dynamicInfo.totalSumPerfPointsPerProcess[pid] = val;
  }

  void increaseTotalSumPerfPointsPerProcess (int pid, DOUBLE val)
  {
    setTotalSumPerfPointsPerProcess (pid, getTotalSumPerfPointsPerProcess (pid) + val);
  }

  void setTotalSumPerfTimePerProcess (int pid, DOUBLE val)
  {
    ASSERT (totalProcCount == dynamicInfo.totalSumPerfTimePerProcess.size ());
    ASSERT (pid >= 0 && pid < totalProcCount);
    dynamicInfo.totalSumPerfTimePerProcess[pid] = val;
  }

  void increaseTotalSumPerfTimePerProcess (int pid, DOUBLE val)
  {
    setTotalSumPerfTimePerProcess (pid, getTotalSumPerfTimePerProcess (pid) + val);
  }

  void setTotalSumLatencyPerConnection (int pid1, int pid2, DOUBLE val)
  {
    ASSERT (totalProcCount == dynamicInfo.totalSumLatencyPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == dynamicInfo.totalSumLatencyPerConnection[pid1].size ());
    dynamicInfo.totalSumLatencyPerConnection[pid1][pid2] = val;
  }

  void increaseTotalSumLatencyPerConnection (int pid1, int pid2, DOUBLE val)
  {
    setTotalSumLatencyPerConnection (pid1, pid2, getTotalSumLatencyPerConnection (pid1, pid2) + val);
  }

  void setTotalSumLatencyCountPerConnection (int pid1, int pid2, DOUBLE val)
  {
    ASSERT (totalProcCount == dynamicInfo.totalSumLatencyCountPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == dynamicInfo.totalSumLatencyCountPerConnection[pid1].size ());
    dynamicInfo.totalSumLatencyCountPerConnection[pid1][pid2] = val;
  }

  void increaseTotalSumLatencyCountPerConnection (int pid1, int pid2, DOUBLE val)
  {
    setTotalSumLatencyCountPerConnection (pid1, pid2, getTotalSumLatencyCountPerConnection (pid1, pid2) + val);
  }

  void setTotalSumBandwidthPerConnection (int pid1, int pid2, DOUBLE val)
  {
    ASSERT (totalProcCount == dynamicInfo.totalSumBandwidthPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == dynamicInfo.totalSumBandwidthPerConnection[pid1].size ());
    dynamicInfo.totalSumBandwidthPerConnection[pid1][pid2] = val;
  }

  void increaseTotalSumBandwidthPerConnection (int pid1, int pid2, DOUBLE val)
  {
    setTotalSumBandwidthPerConnection (pid1, pid2, getTotalSumBandwidthPerConnection (pid1, pid2) + val);
  }

  void setTotalSumBandwidthCountPerConnection (int pid1, int pid2, DOUBLE val)
  {
    ASSERT (totalProcCount == dynamicInfo.totalSumBandwidthCountPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == dynamicInfo.totalSumBandwidthCountPerConnection[pid1].size ());
    dynamicInfo.totalSumBandwidthCountPerConnection[pid1][pid2] = val;
  }

  void increaseTotalSumBandwidthCountPerConnection (int pid1, int pid2, DOUBLE val)
  {
    setTotalSumBandwidthCountPerConnection (pid1, pid2, getTotalSumBandwidthCountPerConnection (pid1, pid2) + val);
  }

  /**
   * Calculate overall latency of share operations between pid1 and pid2 computational nodes
   *
   * @return overall latency of share operations between pid1 and pid2 computational nodes
   */
  DOUBLE calcLatencyForConnection (int pid1, /**< id of first computational node */
                                   int pid2) /**< id of second computational node */
  {
    DOUBLE count = getTotalSumLatencyCountPerConnection (pid1, pid2);
    if (count == 0)
    {
      /*
       * Should get here only for cases when nodes have never performed share opeations
       */
      return 0;
    }

    return getTotalSumLatencyPerConnection (pid1, pid2) / count;
  }

  /**
   * Calculate overall bandwidth of share operations between pid1 and pid2 computational nodes
   *
   * @return overall bandwidth of share operations between pid1 and pid2 computational nodes
   */
  DOUBLE calcBandwidthForConnection (int pid1, /**< id of first computational node */
                                     int pid2) /**< id of second computational node */
  {
    DOUBLE count = getTotalSumBandwidthCountPerConnection (pid1, pid2);
    if (count == 0)
    {
      /*
       * Should get here only for cases when nodes have never performed share opeations
       */
      return 0;
    }

    return getTotalSumBandwidthPerConnection (pid1, pid2) / count;
  }

public:

  void StartCalcClock ();
  void StopCalcClock ();

  void StartShareClock (int, uint32_t);
  void StopShareClock (int, uint32_t);

  void ShareCalcClocks ();
  void ShareShareClocks ();

  void ClearCalcClocks ();
  void ClearShareClocks ();

  DOUBLE getPerf (int pid) const
  {
    ASSERT (totalProcCount == dynamicInfo.speed.size ());
    ASSERT (pid >= 0 && pid < totalProcCount);
    return dynamicInfo.speed[pid];
  }
  DOUBLE getLatency (int pid1, int pid2) const
  {
    ASSERT (totalProcCount == dynamicInfo.latency.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == dynamicInfo.latency[pid1].size ());
    ASSERT (dynamicInfo.latency[pid1][pid2] == dynamicInfo.latency[pid2][pid1]);
    return dynamicInfo.latency[pid1][pid2];
  }
  DOUBLE getBandwidth (int pid1, int pid2) const
  {
    ASSERT (totalProcCount == dynamicInfo.bandwidth.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == dynamicInfo.bandwidth[pid1].size ());
    ASSERT (dynamicInfo.bandwidth[pid1][pid2] == dynamicInfo.bandwidth[pid2][pid1]);
    return dynamicInfo.bandwidth[pid1][pid2];
  }

  DOUBLE getTotalSumPerfPointsPerProcess (int pid) const
  {
    ASSERT (totalProcCount == dynamicInfo.totalSumPerfPointsPerProcess.size ());
    ASSERT (pid >= 0 && pid < totalProcCount);
    return dynamicInfo.totalSumPerfPointsPerProcess[pid];
  }

  DOUBLE getTotalSumPerfTimePerProcess (int pid) const
  {
    ASSERT (totalProcCount == dynamicInfo.totalSumPerfTimePerProcess.size ());
    ASSERT (pid >= 0 && pid < totalProcCount);
    return dynamicInfo.totalSumPerfTimePerProcess[pid];
  }

  DOUBLE getTotalSumLatencyPerConnection (int pid1, int pid2)
  {
    ASSERT (totalProcCount == dynamicInfo.totalSumLatencyPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == dynamicInfo.totalSumLatencyPerConnection[pid1].size ());
    return dynamicInfo.totalSumLatencyPerConnection[pid1][pid2];
  }

  DOUBLE getTotalSumLatencyCountPerConnection (int pid1, int pid2)
  {
    ASSERT (totalProcCount == dynamicInfo.totalSumLatencyCountPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == dynamicInfo.totalSumLatencyCountPerConnection[pid1].size ());
    return dynamicInfo.totalSumLatencyCountPerConnection[pid1][pid2];
  }

  DOUBLE getTotalSumBandwidthPerConnection (int pid1, int pid2)
  {
    ASSERT (totalProcCount == dynamicInfo.totalSumBandwidthPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == dynamicInfo.totalSumBandwidthPerConnection[pid1].size ());
    return dynamicInfo.totalSumBandwidthPerConnection[pid1][pid2];
  }

  DOUBLE getTotalSumBandwidthCountPerConnection (int pid1, int pid2)
  {
    ASSERT (totalProcCount == dynamicInfo.totalSumBandwidthCountPerConnection.size ());
    ASSERT (pid1 >= 0 && pid1 < totalProcCount);
    ASSERT (pid2 >= 0 && pid2 < totalProcCount);
    ASSERT (totalProcCount == dynamicInfo.totalSumBandwidthCountPerConnection[pid1].size ());
    return dynamicInfo.totalSumBandwidthCountPerConnection[pid1][pid2];
  }

  /**
   * Getter for calculations clock
   *
   * @return calculations clock
   */
  Clock getCalcClock (int pid) const
  {
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (dynamicInfo.nodeState[pid] == 1 && dynamicInfo.calcClockSumBetweenRebalance[pid].getFP () > 0
            || dynamicInfo.nodeState[pid] == 0 && dynamicInfo.calcClockSumBetweenRebalance[pid].isZero ());

    return dynamicInfo.calcClockSumBetweenRebalance[pid];
  } /* getCalcClock */

  uint32_t getCalcClockCount (int pid) const
  {
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (dynamicInfo.nodeState[pid] == 1 && dynamicInfo.calcClockCountBetweenRebalance[pid] > 0
            || dynamicInfo.nodeState[pid] == 0 && dynamicInfo.calcClockCountBetweenRebalance[pid] == 0);
    return dynamicInfo.calcClockCountBetweenRebalance[pid];
  }

  void setCalcClockCount (int pid, uint32_t val)
  {
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (dynamicInfo.nodeState[pid] == 1 && val > 0
            || dynamicInfo.nodeState[pid] == 0 && val == 0);
    dynamicInfo.calcClockCountBetweenRebalance[pid] = val;
  }

  const ShareClock_t & getShareClock (int process, int pid) const
  {
    ASSERT (process >= 0 && process < totalProcCount);
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (dynamicInfo.nodeState[process] == 1 && dynamicInfo.nodeState[pid] == 1
            || (dynamicInfo.nodeState[process] == 0 || dynamicInfo.nodeState[pid] == 0)
               && dynamicInfo.shareClockSumBetweenRebalance[process][pid].empty ());
    return dynamicInfo.shareClockSumBetweenRebalance[process][pid];
  }

  const ShareClock_t & getShareClockCur (int pid) const
  {
    return getShareClock (processId, pid);
  }

  /**
   * Get size of buffer shared between current process and pid during computations
   *
   * @return size of buffer shared between current process and pid during computations
   */
  const uint32_t & getShareClockCountCur (int pid) const /**< id of process, with which to get buffer size */
  {
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (dynamicInfo.nodeState[processId] == 1 && dynamicInfo.nodeState[pid] == 1
            && dynamicInfo.shareClockCountBetweenRebalance[pid] > 0
            || (dynamicInfo.nodeState[processId] == 0 || dynamicInfo.nodeState[pid] == 0)
               && dynamicInfo.shareClockCountBetweenRebalance[pid] == 0);
    return dynamicInfo.shareClockCountBetweenRebalance[pid];
  }

  const uint32_t & getShareClockIter (int process, int pid, uint32_t bufSize)
  {
    ASSERT (process >= 0 && process < totalProcCount);
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (dynamicInfo.shareClockIterBetweenRebalance[process][pid].find (bufSize)
            != dynamicInfo.shareClockIterBetweenRebalance[process][pid].end ());
    ASSERT (dynamicInfo.nodeState[process] == 1 && dynamicInfo.nodeState[pid] == 1
            && dynamicInfo.shareClockIterBetweenRebalance[process][pid][bufSize] > 0
            || (dynamicInfo.nodeState[process] == 0 || dynamicInfo.nodeState[pid] == 0)
               && dynamicInfo.shareClockIterBetweenRebalance[process][pid][bufSize] == 0);
    return dynamicInfo.shareClockIterBetweenRebalance[process][pid][bufSize];
  }

  const uint32_t & getShareClockIterCur (int pid, uint32_t bufSize)
  {
    return getShareClockIter (processId, pid, bufSize);
  }

  void setShareClockCur (int pid, uint32_t shareSize, Clock val)
  {
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (dynamicInfo.nodeState[processId] == 1 && dynamicInfo.nodeState[pid] == 1
            || (dynamicInfo.nodeState[processId] == 0 || dynamicInfo.nodeState[pid] == 0) && val.isZero ());
    dynamicInfo.shareClockSumBetweenRebalance[processId][pid][shareSize] = val;
  }

  void setShareClockCountCur (int pid, uint32_t val)
  {
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (dynamicInfo.nodeState[processId] == 1 && dynamicInfo.nodeState[pid] == 1 && val > 0
            || (dynamicInfo.nodeState[processId] == 0 || dynamicInfo.nodeState[pid] == 0) && val == 0);
    dynamicInfo.shareClockCountBetweenRebalance[pid] = val;
  }

  void setShareClockIterCur (int pid, uint32_t bufSize, uint32_t val)
  {
    ASSERT (pid >= 0 && pid < totalProcCount);
    ASSERT (dynamicInfo.nodeState[processId] == 1 && dynamicInfo.nodeState[pid] == 1 && val > 0
            || (dynamicInfo.nodeState[processId] == 0 || dynamicInfo.nodeState[pid] == 0) && val == 0);
    dynamicInfo.shareClockIterBetweenRebalance[processId][pid][bufSize] = val;
  }

  int getNodeForDirectionForProcess (int, BufferPosition) const;

  void updateCurrentPerfValues (time_step);
  void approximateWithLinearRegression (DOUBLE &, DOUBLE &, const ShareClock_t &);
  void updateCurrentShareValues ();
  void doAdditionalShareMeasurements (uint32_t, uint32_t, uint32_t, uint32_t);
  void initializeIterationCounters (time_step);
  DOUBLE calcTotalPerf (time_step);
  void calcTotalLatencyAndBandwidth (time_step);
#endif /* DYNAMIC_GRID */
}; /* ParallelGridCore */

#endif /* PARALLEL_GRID */

#endif /* PARALLEL_GRID_CORE_H */
