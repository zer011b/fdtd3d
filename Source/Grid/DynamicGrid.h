/*
 * Copyright (C) 2018 Gleb Balykov
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

#ifndef DYNAMIC_GRID_H
#define DYNAMIC_GRID_H

#ifdef PARALLEL_GRID
#ifdef DYNAMIC_GRID

#include <map>

#include "PAssert.h"
#include "FieldValue.h"
#include "Settings.h"
#include "Clock.h"
#include "Parallel.h"

/**
 * Size of buffer for additional measurements
 */
#define CLOCK_BUF_SIZE 10

/**
 * Type for share clock for different buffer sizes.
 * Type of calc clock.
 */
typedef std::map<uint32_t, Clock> ShareClock_t;

/**
 * Type for number of iterations for different buffer sizes.
 */
typedef std::map<uint32_t, uint32_t> IterCount_t;


/**
 * Information required for dynamic grid, i.e. counters, time clocks, etc.
 */
struct DynamicGridInfo
{

  /**
   * States of all processes
   */
  std::vector<int> nodeState;

  /**
   * Lists of nodes for each direction (the closest come in the beggining of nodesForDirections[dir]) for each process
   */
  std::vector< std::vector< std::vector<int> > > nodesForDirections;

  /**
   * Latest clock counter for calculations of all processes
   */
  std::vector<Clock> calcClockSumBetweenRebalance;

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
  std::vector<DOUBLE> totalSumPerfPointsPerProcess;
  std::vector<DOUBLE> totalSumPerfTimePerProcess;

  /**
   * Total values for latency: latency is totalSumLatencyPerConnection/totalSumLatencyCountPerConnection then
   */
  std::vector< std::vector<DOUBLE> > totalSumLatencyPerConnection;
  std::vector< std::vector<DOUBLE> > totalSumLatencyCountPerConnection;

  /**
   * Total values of bandwidth: latency is totalSumBandwidthPerConnection/totalSumBandwidthCountPerConnection then
   */
  std::vector< std::vector<DOUBLE> > totalSumBandwidthPerConnection;
  std::vector< std::vector<DOUBLE> > totalSumBandwidthCountPerConnection;

  /**
   * Value for number of grid point between rebalances
   */
  std::vector<DOUBLE> curPoints;
  /**
   * Value for calculation time between rebalances
   */
  std::vector<DOUBLE> curTimes;

  std::vector< std::vector<int> > skipCurShareMeasurement;

  /**
   * Values of current latency and bandwidth between rebalance
   */
  std::vector< std::vector<DOUBLE> > curShareLatency;
  std::vector< std::vector<DOUBLE> > curShareBandwidth;

  std::vector<DOUBLE> speed;
  std::vector< std::vector<DOUBLE> > latency;
  std::vector< std::vector<DOUBLE> > bandwidth;

  /*
   * TODO: Use this
   */
  time_step T_balance;
  time_step T_perf;

  /**
   * Helper buffers used for sharing
   */
  /*
   * TODO: remove this buffers and add MPI data type
   */
 #ifdef MPI_CLOCK
   DOUBLE *shareClockSec_buf;
 #else /* MPI_CLOCK */
   uint64_t *shareClockSec_buf;
   uint64_t *shareClockNSec_buf;
 #endif /* !MPI_CLOCK */
  uint32_t *shareClockBufSize_buf;

  uint32_t *shareClockBufSize2_buf;
  uint32_t *shareClockIter_buf;

  /**
   * Helper clock counter for start of calculations of current process
   */
  Clock calcStart;

  /**
   * Helper clock counter for stop of calculations of current process
   */
  Clock calcStop;

  /**
   * Helper clock counter for start of share operations of current process
   */
  Clock shareStart;

  /**
   * Helper clock counter for stop of share operations of current process
   */
  Clock shareStop;
}; /* DynamicGridInfo */

#endif /* DYNAMIC_GRID */
#endif /* PARALLEL_GRID */

#endif /* !DYNAMIC_GRID_H */
