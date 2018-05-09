#include "ParallelGridCore.h"

#ifdef PARALLEL_GRID

#if PRINT_MESSAGE
extern const char* BufferPositionNames[];
#endif /* PRINT_MESSAGE */

#ifdef DYNAMIC_GRID

#include <unistd.h>

void
ParallelGridCore::SetNodesForDirections (int pid)
{
  /*
   * TODO: initialize processes considering real number of processes given, i.e. if only 2 are given for 3D-XYZ
   */
  nodesForDirections[pid].resize (BUFFER_COUNT);

#if defined (PARALLEL_BUFFER_DIMENSION_1D_X) || defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || \
    defined (PARALLEL_BUFFER_DIMENSION_2D_XZ) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  for (int i = pid - 1; i >= pid - getNodeGridX (pid); --i)
  {
    nodesForDirections[pid][LEFT].push_back (i);
  }
  for (int i = pid + 1; i < pid - getNodeGridX (pid) + getNodeGridSizeX (); ++i)
  {
    nodesForDirections[pid][RIGHT].push_back (i);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Y) || defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  for (int i = pid - 1; i >= pid - getNodeGridY (pid); --i)
  {
    nodesForDirections[pid][DOWN].push_back (i);
  }
  for (int i = pid + 1; i < pid - getNodeGridY (pid) + getNodeGridSizeY (); ++i)
  {
    nodesForDirections[pid][UP].push_back (i);
  }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  for (int i = pid - getNodeGridSizeX (); i >= pid - getNodeGridY (pid) * getNodeGridSizeX (); i -= getNodeGridSizeX ())
  {
    nodesForDirections[pid][DOWN].push_back (i);
  }
  for (int i = pid + getNodeGridSizeX (); i < pid - (getNodeGridY (pid) - getNodeGridSizeY ()) * getNodeGridSizeX (); i += getNodeGridSizeX ())
  {
    nodesForDirections[pid][UP].push_back (i);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_1D_Z)
  for (int i = pid - 1; i >= pid - getNodeGridZ (pid); --i)
  {
    nodesForDirections[pid][BACK].push_back (i);
  }
  for (int i = pid + 1; i < pid - getNodeGridZ (pid) + getNodeGridSizeZ (); ++i)
  {
    nodesForDirections[pid][FRONT].push_back (i);
  }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  for (int i = pid - getNodeGridSizeY (); i >= pid - getNodeGridZ (pid) * getNodeGridSizeY (); i -= getNodeGridSizeY ())
  {
    nodesForDirections[pid][BACK].push_back (i);
  }
  for (int i = pid + getNodeGridSizeY (); i < pid - (getNodeGridZ (pid) - getNodeGridSizeZ ()) * getNodeGridSizeY (); i += getNodeGridSizeY ())
  {
    nodesForDirections[pid][FRONT].push_back (i);
  }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  for (int i = pid - getNodeGridSizeX (); i >= pid - getNodeGridZ (pid) * getNodeGridSizeX (); i -= getNodeGridSizeX ())
  {
    nodesForDirections[pid][BACK].push_back (i);
  }
  for (int i = pid + getNodeGridSizeX (); i < pid - (getNodeGridZ (pid) - getNodeGridSizeZ ()) * getNodeGridSizeX (); i += getNodeGridSizeX ())
  {
    nodesForDirections[pid][FRONT].push_back (i);
  }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  for (int i = pid - getNodeGridSizeXY (); i >= pid - getNodeGridZ (pid) * getNodeGridSizeXY (); i -= getNodeGridSizeXY ())
  {
    nodesForDirections[pid][BACK].push_back (i);
  }
  for (int i = pid + getNodeGridSizeXY (); i < pid - (getNodeGridZ (pid) - getNodeGridSizeZ ()) * getNodeGridSizeXY (); i += getNodeGridSizeXY ())
  {
    nodesForDirections[pid][FRONT].push_back (i);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XY) || defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (getHasL (pid) && getHasD (pid))
  {
    int i = 0;
    for (i = pid - getNodeGridSizeX () - 1; getHasL (i) && getHasD (i); i -= getNodeGridSizeX () + 1)
    {
      nodesForDirections[pid][LEFT_DOWN].push_back (i);
    }
    nodesForDirections[pid][LEFT_DOWN].push_back (i);
  }
  if (getHasL (pid) && getHasU (pid))
  {
    int i = 0;
    for (i = pid + getNodeGridSizeX () - 1; getHasL (i) && getHasU (i); i += getNodeGridSizeX () - 1)
    {
      nodesForDirections[pid][LEFT_UP].push_back (i);
    }
    nodesForDirections[pid][LEFT_UP].push_back (i);
  }
  if (getHasR (pid) && getHasD (pid))
  {
    int i = 0;
    for (i = pid - getNodeGridSizeX () + 1; getHasR (i) && getHasD (i); i -= getNodeGridSizeX () - 1)
    {
      nodesForDirections[pid][RIGHT_DOWN].push_back (i);
    }
    nodesForDirections[pid][RIGHT_DOWN].push_back (i);
  }
  if (getHasR (pid) && getHasU (pid))
  {
    int i = 0;
    for (i = pid + getNodeGridSizeX () + 1; getHasR (i) && getHasU (i); i += getNodeGridSizeX () + 1)
    {
      nodesForDirections[pid][RIGHT_UP].push_back (i);
    }
    nodesForDirections[pid][RIGHT_UP].push_back (i);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_YZ)
  if (getHasD (pid) && getHasB (pid))
  {
    int i = 0;
    for (i = pid - getNodeGridSizeY () - 1; getHasD (i) && getHasB (i); i -= getNodeGridSizeY () + 1)
    {
      nodesForDirections[pid][DOWN_BACK].push_back (i);
    }
    nodesForDirections[pid][DOWN_BACK].push_back (i);
  }
  if (getHasD (pid) && getHasF (pid))
  {
    int i = 0;
    for (i = pid + getNodeGridSizeY () - 1; getHasD (i) && getHasF (i); i += getNodeGridSizeY () - 1)
    {
      nodesForDirections[pid][DOWN_FRONT].push_back (i);
    }
    nodesForDirections[pid][DOWN_FRONT].push_back (i);
  }
  if (getHasU (pid) && getHasB (pid))
  {
    int i = 0;
    for (i = pid - getNodeGridSizeY () + 1; getHasU (i) && getHasB (i); i -= getNodeGridSizeY () - 1)
    {
      nodesForDirections[pid][UP_BACK].push_back (i);
    }
    nodesForDirections[pid][UP_BACK].push_back (i);
  }
  if (getHasU (pid) && getHasF (pid))
  {
    int i = 0;
    for (i = pid + getNodeGridSizeY () + 1; getHasU (i) && getHasF (i); i += getNodeGridSizeY () + 1)
    {
      nodesForDirections[pid][UP_FRONT].push_back (i);
    }
    nodesForDirections[pid][UP_FRONT].push_back (i);
  }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (getHasD (pid) && getHasB (pid))
  {
    int i = 0;
    for (i = pid - getNodeGridSizeXY () - getNodeGridSizeX (); getHasD (i) && getHasB (i); i -= getNodeGridSizeXY () + getNodeGridSizeX ())
    {
      nodesForDirections[pid][DOWN_BACK].push_back (i);
    }
    nodesForDirections[pid][DOWN_BACK].push_back (i);
  }
  if (getHasD (pid) && getHasF (pid))
  {
    int i = 0;
    for (i = pid + getNodeGridSizeXY () - getNodeGridSizeX (); getHasD (i) && getHasF (i); i += getNodeGridSizeXY () - getNodeGridSizeX ())
    {
      nodesForDirections[pid][DOWN_FRONT].push_back (i);
    }
    nodesForDirections[pid][DOWN_FRONT].push_back (i);
  }
  if (getHasU (pid) && getHasB (pid))
  {
    int i = 0;
    for (i = pid - getNodeGridSizeXY () + getNodeGridSizeX (); getHasU (i) && getHasB (i); i -= getNodeGridSizeXY () - getNodeGridSizeX ())
    {
      nodesForDirections[pid][UP_BACK].push_back (i);
    }
    nodesForDirections[pid][UP_BACK].push_back (i);
  }
  if (getHasU (pid) && getHasF (pid))
  {
    int i = 0;
    for (i = pid + getNodeGridSizeXY () + getNodeGridSizeX (); getHasU (i) && getHasF (i); i += getNodeGridSizeXY () + getNodeGridSizeX ())
    {
      nodesForDirections[pid][UP_FRONT].push_back (i);
    }
    nodesForDirections[pid][UP_FRONT].push_back (i);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_2D_XZ)
  if (getHasL (pid) && getHasB (pid))
  {
    int i = 0;
    for (i = pid - getNodeGridSizeX () - 1; getHasL (i) && getHasB (i); i -= getNodeGridSizeX () + 1)
    {
      nodesForDirections[pid][LEFT_BACK].push_back (i);
    }
    nodesForDirections[pid][LEFT_BACK].push_back (i);
  }
  if (getHasL (pid) && getHasF (pid))
  {
    int i = 0;
    for (i = pid + getNodeGridSizeX () - 1; getHasL (i) && getHasF (i); i += getNodeGridSizeX () - 1)
    {
      nodesForDirections[pid][LEFT_FRONT].push_back (i);
    }
    nodesForDirections[pid][LEFT_FRONT].push_back (i);
  }
  if (getHasR (pid) && getHasB (pid))
  {
    int i = 0;
    for (i = pid - getNodeGridSizeX () + 1; getHasR (i) && getHasB (i); i -= getNodeGridSizeX () - 1)
    {
      nodesForDirections[pid][RIGHT_BACK].push_back (i);
    }
    nodesForDirections[pid][RIGHT_BACK].push_back (i);
  }
  if (getHasR (pid) && getHasF (pid))
  {
    int i = 0;
    for (i = pid + getNodeGridSizeX () + 1; getHasR (i) && getHasF (i); i += getNodeGridSizeX () + 1)
    {
      nodesForDirections[pid][RIGHT_FRONT].push_back (i);
    }
    nodesForDirections[pid][RIGHT_FRONT].push_back (i);
  }
#endif
#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (getHasL (pid) && getHasB (pid))
  {
    int i = 0;
    for (i = pid - getNodeGridSizeXY () - 1; getHasL (i) && getHasB (i); i -= getNodeGridSizeXY () + 1)
    {
      nodesForDirections[pid][LEFT_BACK].push_back (i);
    }
    nodesForDirections[pid][LEFT_BACK].push_back (i);
  }
  if (getHasL (pid) && getHasF (pid))
  {
    int i = 0;
    for (i = pid + getNodeGridSizeXY () - 1; getHasL (i) && getHasF (i); i += getNodeGridSizeXY () - 1)
    {
      nodesForDirections[pid][LEFT_FRONT].push_back (i);
    }
    nodesForDirections[pid][LEFT_FRONT].push_back (i);
  }
  if (getHasR (pid) && getHasB (pid))
  {
    int i = 0;
    for (i = pid - getNodeGridSizeXY () + 1; getHasR (i) && getHasB (i); i -= getNodeGridSizeXY () - 1)
    {
      nodesForDirections[pid][RIGHT_BACK].push_back (i);
    }
    nodesForDirections[pid][RIGHT_BACK].push_back (i);
  }
  if (getHasR (pid) && getHasF (pid))
  {
    int i = 0;
    for (i = pid + getNodeGridSizeXY () + 1; getHasR (i) && getHasF (i); i += getNodeGridSizeXY () + 1)
    {
      nodesForDirections[pid][RIGHT_FRONT].push_back (i);
    }
    nodesForDirections[pid][RIGHT_FRONT].push_back (i);
  }
#endif

#if defined (PARALLEL_BUFFER_DIMENSION_3D_XYZ)
  if (getHasL (pid) && getHasD (pid) && getHasB (pid))
  {
    int i = 0;
    for (i = pid - getNodeGridSizeXY () - getNodeGridSizeX () - 1;
         getHasL (i) && getHasD (i) && getHasB (i);
         i -= getNodeGridSizeXY () + getNodeGridSizeX () + 1)
    {
      nodesForDirections[pid][LEFT_DOWN_BACK].push_back (i);
    }
    nodesForDirections[pid][LEFT_DOWN_BACK].push_back (i);
  }
  if (getHasL (pid) && getHasD (pid) && getHasF (pid))
  {
    int i = 0;
    for (i = pid + getNodeGridSizeXY () - getNodeGridSizeX () - 1;
         getHasL (i) && getHasD (i) && getHasF (i);
         i += getNodeGridSizeXY () - getNodeGridSizeX () - 1)
    {
      nodesForDirections[pid][LEFT_DOWN_FRONT].push_back (i);
    }
    nodesForDirections[pid][LEFT_DOWN_FRONT].push_back (i);
  }
  if (getHasL (pid) && getHasU (pid) && getHasB (pid))
  {
    int i = 0;
    for (i = pid - getNodeGridSizeXY () + getNodeGridSizeX () - 1;
         getHasL (i) && getHasU (i) && getHasB (i);
         i -= getNodeGridSizeXY () - getNodeGridSizeX () + 1)
    {
      nodesForDirections[pid][LEFT_UP_BACK].push_back (i);
    }
    nodesForDirections[pid][LEFT_UP_BACK].push_back (i);
  }
  if (getHasL (pid) && getHasU (pid) && getHasF (pid))
  {
    int i = 0;
    for (i = pid + getNodeGridSizeXY () + getNodeGridSizeX () - 1;
         getHasL (i) && getHasU (i) && getHasF (i);
         i += getNodeGridSizeXY () + getNodeGridSizeX () - 1)
    {
      nodesForDirections[pid][LEFT_UP_FRONT].push_back (i);
    }
    nodesForDirections[pid][LEFT_UP_FRONT].push_back (i);
  }

  if (getHasR (pid) && getHasD (pid) && getHasB (pid))
  {
    int i = 0;
    for (i = pid - getNodeGridSizeXY () - getNodeGridSizeX () + 1;
         getHasR (i) && getHasD (i) && getHasB (i);
         i -= getNodeGridSizeXY () + getNodeGridSizeX () - 1)
    {
      nodesForDirections[pid][RIGHT_DOWN_BACK].push_back (i);
    }
    nodesForDirections[pid][RIGHT_DOWN_BACK].push_back (i);
  }
  if (getHasR (pid) && getHasD (pid) && getHasF (pid))
  {
    int i = 0;
    for (i = pid + getNodeGridSizeXY () - getNodeGridSizeX () + 1;
         getHasR (i) && getHasD (i) && getHasF (i);
         i += getNodeGridSizeXY () - getNodeGridSizeX () + 1)
    {
      nodesForDirections[pid][RIGHT_DOWN_FRONT].push_back (i);
    }
    nodesForDirections[pid][RIGHT_DOWN_FRONT].push_back (i);
  }
  if (getHasR (pid) && getHasU (pid) && getHasB (pid))
  {
    int i = 0;
    for (i = pid - getNodeGridSizeXY () + getNodeGridSizeX () + 1;
         getHasR (i) && getHasU (i) && getHasB (i);
         i -= getNodeGridSizeXY () - getNodeGridSizeX () - 1)
    {
      nodesForDirections[pid][RIGHT_UP_BACK].push_back (i);
    }
    nodesForDirections[pid][RIGHT_UP_BACK].push_back (i);
  }
  if (getHasR (pid) && getHasU (pid) && getHasF (pid))
  {
    int i = 0;
    for (i = pid + getNodeGridSizeXY () + getNodeGridSizeX () + 1;
         getHasR (i) && getHasU (i) && getHasF (i);
         i += getNodeGridSizeXY () + getNodeGridSizeX () + 1)
    {
      nodesForDirections[pid][RIGHT_UP_FRONT].push_back (i);
    }
    nodesForDirections[pid][RIGHT_UP_FRONT].push_back (i);
  }
#endif

#if PRINT_MESSAGE
  MPI_Barrier (MPI_COMM_WORLD);
  if (processId == 0)
  {
    DPRINTF (LOG_LEVEL_NONE, "=== Processes map #%d===\n", pid);

    DPRINTF (LOG_LEVEL_NONE, "Process #%d:\n", pid);
    for (int dir = 0; dir < BUFFER_COUNT; ++dir)
    {
      DPRINTF (LOG_LEVEL_NONE, "  Processes to %s: ", BufferPositionNames[dir]);

      for (int i = 0; i < nodesForDirections[pid][dir].size (); ++i)
      {
        DPRINTF (LOG_LEVEL_NONE, " %d, ", nodesForDirections[pid][dir][i]);
      }

      DPRINTF (LOG_LEVEL_NONE, "\n");
    }
    DPRINTF (LOG_LEVEL_NONE, "=== ===\n");
  }
#endif
}

/**
 * Start clock for calculations
 */
void
ParallelGridCore::StartCalcClock ()
{
#ifdef MPI_DYNAMIC_CLOCK
  FPValue val = -MPI_Wtime ();
  calcClockSumBetweenRebalance[processId] += val;
#else
  int status = clock_gettime (CLOCK_MONOTONIC, &calcStart);
  ASSERT (status == 0);
#endif
} /* ParallelGridCore::StartCalcClock */

/**
 * Stop clock for calculations
 */
void
ParallelGridCore::StopCalcClock ()
{
#ifdef MPI_DYNAMIC_CLOCK
  FPValue val = MPI_Wtime ();
  calcClockSumBetweenRebalance[processId] += val;
#else
  int status = clock_gettime (CLOCK_MONOTONIC, &calcStop);
  ASSERT (status == 0);

  timespec diff;
  timespec_diff (&calcStart, &calcStop, &diff);

  timespec sum;
  timespec_sum (&calcClockSumBetweenRebalance[processId], &diff, &sum);

  calcClockSumBetweenRebalance[processId] = sum;
#endif
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

void ParallelGridCore::ShareCalcClocks ()
{
  for (int process = 0; process < getTotalProcCount (); ++process)
  {
#ifdef MPI_DYNAMIC_CLOCK
    FPValue calcClockSec;
#else
    uint64_t calcClockSec;
    uint64_t calcClockNSec;
#endif
    uint32_t calcClockCount;

    if (process == getProcessId ())
    {
#ifdef MPI_DYNAMIC_CLOCK
      calcClockSec = calcClockSumBetweenRebalance[process];
#else
      calcClockSec = (uint64_t) calcClockSumBetweenRebalance[process].tv_sec;
      calcClockNSec = (uint64_t) calcClockSumBetweenRebalance[process].tv_nsec;
#endif
      calcClockCount = calcClockCountBetweenRebalance[process];
    }

#ifdef MPI_DYNAMIC_CLOCK
    MPI_Bcast (&calcClockSec, 1, MPI_FPVALUE, process, communicator);
#else
    MPI_Bcast (&calcClockSec, 1, MPI_LONG_LONG, process, communicator);
    MPI_Bcast (&calcClockNSec, 1, MPI_LONG_LONG, process, communicator);
#endif
    MPI_Bcast (&calcClockCount, 1, MPI_UNSIGNED, process, communicator);

    if (process != getProcessId ())
    {
#ifdef MPI_DYNAMIC_CLOCK
      calcClockSumBetweenRebalance[process] = calcClockSec;
#else
      calcClockSumBetweenRebalance[process].tv_sec = calcClockSec;
      calcClockSumBetweenRebalance[process].tv_nsec = calcClockNSec;
#endif
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
          for (int k = 0; k < CLOCK_BUF_SIZE; ++k)
          {
            shareClockBufSize_buf[j] = 0;
            shareClockSec_buf[j] = 0;
#ifndef MPI_DYNAMIC_CLOCK
            shareClockNSec_buf[j] = 0;
#endif
            j++;
//
//             shareClockBufSize_buf[j] = 0;
//             shareClockSec_buf[j] = 0;
// #ifndef MPI_DYNAMIC_CLOCK
//             shareClockNSec_buf[j] = 0;
// #endif
//             j++;
          }
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
          for (int k = 0; k < CLOCK_BUF_SIZE; ++k)
          {
            shareClockBufSize2_buf[jj] = 0;
            shareClockIter_buf[jj] = 0;
            jj++;
            //
            // shareClockBufSize2_buf[jj] = 0;
            // shareClockIter_buf[jj] = 0;
            // jj++;
          }
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
#ifdef MPI_DYNAMIC_CLOCK
    calcClockSumBetweenRebalance[i] = 0;
#else
    calcClockSumBetweenRebalance[i].tv_sec = 0;
    calcClockSumBetweenRebalance[i].tv_nsec = 0;
#endif

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

int
ParallelGridCore::getNodeForDirectionForProcess (int pid, BufferPosition dir) const
{
  for (int i = 0; i < nodesForDirections[pid][dir].size (); ++i)
  {
    int process = nodesForDirections[pid][dir][i];

    /*
     * Choose the first enabled node
     */
    if (nodeState[process])
    {
      return process;
    }
  }

  return PID_NONE;
}

/**
 * Update values of counters (clock and number of points) for period during rebalances.
 */
void
ParallelGridCore::updateCurrentPerfValues (time_step difft) /**< number of elapsed time steps */
{
  for (int i = 0; i < getTotalProcCount (); ++i)
  {
    curPoints[i] = difft * getCalcClockCount (i);

    CalcClock_t calcClockCur = getCalcClock (i);
#ifdef MPI_DYNAMIC_CLOCK
    FPValue timesec = calcClockCur;
#else
    FPValue timesec = (FPValue) calcClockCur.tv_sec + ((FPValue) calcClockCur.tv_nsec) / 1000000000;
#endif
    curTimes[i] = timesec;

#ifdef ENABLE_ASSERTS
    if (getNodeState ()[i] == 0)
    {
      ASSERT (curPoints[i] == 0 && curTimes[i] == 0);
    }
    else
    {
      ASSERT (curPoints[i] != 0 && curTimes[i] != 0);
    }
#endif
  }
}

/**
 * Approximate latency and bandwidth using linear regression
 */
void
ParallelGridCore::approximateWithLinearRegression (FPValue &latency, /**< out: value of latency */
                                                   FPValue &bandwidth, /**< out: value of bandwidth */
                                                   const ShareClock_t &clockMap) /**< map of times for different buffer sizes */
{
  FPValue avg_sum_x = 0;
  FPValue avg_sum_y = 0;
  FPValue avg_sum_x2 = 0;
  FPValue avg_sum_xy = 0;

  int index = 0;

  for (ShareClock_t::const_iterator it = clockMap.begin (); it != clockMap.end (); ++it)
  {
    FPValue bufSize = it->first;
#ifndef MPI_DYNAMIC_CLOCK
    FPValue clocks = (FPValue) it->second.tv_sec + ((FPValue) it->second.tv_nsec) / 1000000000;
#else
    FPValue clocks = it->second;
#endif

    avg_sum_x += bufSize;
    avg_sum_y += clocks;
    avg_sum_x2 += SQR(bufSize);
    avg_sum_xy += bufSize * clocks;

    ++index;
  }

  avg_sum_x /= index;
  avg_sum_y /= index;
  avg_sum_x2 /= index;
  avg_sum_xy /= index;

  bandwidth = (avg_sum_x2 - SQR(avg_sum_x)) / (avg_sum_xy - avg_sum_x * avg_sum_y);
  latency = avg_sum_y - avg_sum_x / bandwidth;

  // if (latency < 0)
  // {
  //   latency = 0;
  // }
  //
  // if (bandwidth < 0)
  // {
  //   bandwidth = 1000000000;
  // }
}

/**
 * Update values of latency and bandwidth for period during rebalances.
 *
 * Currently linear regression is used to identify latency and bandwidth (Hockney communication model is considered).
 */
void
ParallelGridCore::updateCurrentShareValues ()
{
  for (int i = 0; i < getTotalProcCount (); ++i)
  {
    for (int j = 0; j < getTotalProcCount (); ++j)
    {
      if (i == j
          || getNodeState ()[i] == 0
          || getNodeState ()[j] == 0)
      {
        curShareLatency[i][j] = 0;
        curShareBandwidth[i][j] = 0;

        continue;
      }

      skipCurShareMeasurement[i][j] = 0;

      ShareClock_t clockMap = getShareClock (i, j);

      ASSERT (clockMap.size () == CLOCK_BUF_SIZE
              || clockMap.empty ());

      approximateWithLinearRegression (curShareLatency[i][j], curShareBandwidth[i][j], clockMap);

      if (curShareBandwidth[i][j] <= 0 || curShareLatency[i][j] < 0)
      {
        // TODO: continue measurements if current data is not enough to produce correct latency and bandwidth
        printf ("INCORRECT: %d %d -> %f %f\n", i, j, curShareLatency[i][j], curShareBandwidth[i][j]);
        skipCurShareMeasurement[i][j] = 1;
      }

#ifdef ENABLE_ASSERTS
      if (getNodeState ()[i] == 0
          || getNodeState ()[j] == 0)
      {
        ASSERT (curShareLatency[i][j] == 0 && curShareBandwidth[i][j] == 0);
      }
#endif
    }
  }
}

/**
 * Perform additional share operations between computational nodes, which already were neighbors during time steps
 * computations. This is required, as for all time steps buffer sizes are the same, and one value is not enough to
 * perform prediction of share operations time in future. Currently, linear regression is used for all measured values
 */
void
ParallelGridCore::doAdditionalShareMeasurements (uint32_t latency_measure_count, /**< number of iterations for additional measurements */
                                                 uint32_t latency_buf_size, /**< maximum size of buffer to perform additional measurements */
                                                 uint32_t latency_buf_start, /**< start size of buffer */
                                                 uint32_t latency_buf_step) /**< step size of buffer */
{
  if (getNodeState ()[getProcessId ()] == 0)
  {
    return;
  }

#ifdef PARALLEL_BUFFER_DIMENSION_3D_XYZ
  if (getProcessId () >= getNodeGridSizeXYZ ())
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_3D_XYZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XY
  if (getProcessId () >= getNodeGridSizeXY ())
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XY */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_YZ
  if (getProcessId () >= getNodeGridSizeYZ ())
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_YZ */

#ifdef PARALLEL_BUFFER_DIMENSION_2D_XZ
  if (getProcessId () >= getNodeGridSizeXZ ())
  {
    return;
  }
#endif /* PARALLEL_BUFFER_DIMENSION_2D_XZ */

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

  std::vector<FieldValue> tmp_buffer (latency_buf_size);

  for (uint32_t buf_size = latency_buf_start; buf_size < latency_buf_size; buf_size += latency_buf_step)
  for (uint32_t count = 0; count < latency_measure_count; ++count)
  for (int buf = 0; buf < BUFFER_COUNT; ++buf)
  {
    BufferPosition bufferDirection = (BufferPosition) buf;
    BufferPosition opposite = getOppositeDirections ()[bufferDirection];

    int processTo = getNodeForDirection (bufferDirection);
    int processFrom = getNodeForDirection (opposite);

    uint32_t ssize;
    if (processTo != PID_NONE)
    {
      if (buf_size != getShareClockCountCur (processTo))
      {
        ssize = buf_size;
      }
      else
      {
        ssize = latency_buf_size;
      }
      setShareClockIterCur (processTo, ssize, latency_measure_count);
    }

    uint32_t rsize;
    if (processFrom != PID_NONE)
    {
      if (buf_size != getShareClockCountCur (processFrom))
      {
        rsize = buf_size;
      }
      else
      {
        rsize = latency_buf_size;
      }
      setShareClockIterCur (processFrom, rsize, latency_measure_count);
    }

    if (processTo != PID_NONE
        && processFrom == PID_NONE)
    {
      StartShareClock (processTo, ssize);

      int retCode = MPI_Send (tmp_buffer.data(), ssize, datatype, processTo, getProcessId (), getCommunicator ());
      ASSERT (retCode == MPI_SUCCESS);

      StopShareClock (processTo, ssize);
    }
    else if (processTo == PID_NONE
             && processFrom != PID_NONE)
    {
      StartShareClock (processFrom, rsize);

      MPI_Status status;
      int retCode = MPI_Recv (tmp_buffer.data(), rsize, datatype, processFrom, processFrom, getCommunicator (), &status);
      ASSERT (retCode == MPI_SUCCESS);

      StopShareClock (processFrom, rsize);
    }
    else if (processTo != PID_NONE
             && processFrom != PID_NONE)
    {
#ifdef COMBINED_SENDRECV
      UNREACHABLE;
#else
    // Even send first, then receive. Non-even receive first, then send
      if (getIsEvenForDirection()[bufferDirection])
      {
        StartShareClock (processTo, ssize);

        int retCode = MPI_Send (tmp_buffer.data(), ssize, datatype, processTo, getProcessId (), getCommunicator ());
        ASSERT (retCode == MPI_SUCCESS);

        StopShareClock (processTo, ssize);
        StartShareClock (processFrom, rsize);

        MPI_Status status;
        retCode = MPI_Recv (tmp_buffer.data(), rsize, datatype, processFrom, processFrom, getCommunicator (), &status);
        ASSERT (retCode == MPI_SUCCESS);

        StopShareClock (processFrom, rsize);
      }
      else
      {
        StartShareClock (processFrom, rsize);

        MPI_Status status;
        int retCode = MPI_Recv (tmp_buffer.data(), rsize, datatype, processFrom, processFrom, getCommunicator (), &status);
        ASSERT (retCode == MPI_SUCCESS);

        StopShareClock (processFrom, rsize);
        StartShareClock (processTo, ssize);

        retCode = MPI_Send (tmp_buffer.data(), ssize, datatype, processTo, getProcessId (), getCommunicator ());
        ASSERT (retCode == MPI_SUCCESS);

        StopShareClock (processTo, ssize);
      }
#endif
    }
    else
    {
      /*
       * Do nothing
       */
    }
  }
}

/**
 * Set number of performed share operations during computations between rebalances
 */
void
ParallelGridCore::initializeIterationCounters (time_step difft) /**< elapsed number of time steps */
{
  if (getNodeState ()[getProcessId ()] == 1)
  {
    for (int dir = 0; dir < BUFFER_COUNT; ++dir)
    {
      int pid = getNodeForDirection ((BufferPosition) dir);
      if (pid == PID_NONE)
      {
        continue;
      }

      setShareClockIterCur (pid, getShareClockCountCur (pid), difft);
    }
  }
}

/**
 * Calculate performance values for all computational nodes
 *
 * @return sum performance of all enabled computational nodes
 */
FPValue
ParallelGridCore::calcTotalPerf (time_step difft) /**< elapsed number of time steps */
{
  ShareCalcClocks ();

  updateCurrentPerfValues (difft);

  FPValue sumSpeedEnabled = 0;

  for (int process = 0; process < getTotalProcCount (); ++process)
  {
    if (getNodeState ()[process] == 1)
    {
      increaseTotalSumPerfPointsPerProcess (process, curPoints[process]);
      increaseTotalSumPerfTimePerProcess (process, curTimes[process]);
    }

    FPValue sumTime = getTotalSumPerfTimePerProcess (process);
    FPValue sumPoints = getTotalSumPerfPointsPerProcess (process);

    ASSERT (sumTime != 0 && sumPoints != 0);

    speed[process] = sumPoints / sumTime;

    if (getNodeState ()[process] == 1)
    {
      sumSpeedEnabled += speed[process];
    }
  }

  return sumSpeedEnabled;
}

/**
 * Calculate latency and bandwidth values for all computational nodes
 */
void
ParallelGridCore::calcTotalLatencyAndBandwidth (time_step difft) /**< elapsed number of time steps */
{
  // TODO: this should not be constants
  uint32_t latency_measure_count = 100;
  uint32_t latency_buf_size = 10000;

  initializeIterationCounters (difft);

  doAdditionalShareMeasurements (latency_measure_count,
                                 latency_buf_size,
                                 latency_buf_size / CLOCK_BUF_SIZE,
                                 latency_buf_size / CLOCK_BUF_SIZE);

  ShareShareClocks ();

  updateCurrentShareValues ();

  for (int process = 0; process < getTotalProcCount (); ++process)
  {
    for (int i = 0; i < getTotalProcCount (); ++i)
    {
      if (process == i)
      {
        continue;
      }

      /*
       * Share operations could have been performed only between enabled computational nodes.
       */
      if (getNodeState ()[process] == 1
          && getNodeState ()[i] == 1
          && skipCurShareMeasurement[process][i] == 0)
      {
        increaseTotalSumLatencyPerConnection (process, i, curShareLatency[process][i]);
        increaseTotalSumLatencyCountPerConnection (process, i, 1);

        increaseTotalSumBandwidthPerConnection (process, i, curShareBandwidth[process][i]);
        increaseTotalSumBandwidthCountPerConnection (process, i, 1);
      }

      latency[process][i] = calcLatencyForConnection (process, i);
      bandwidth[process][i] = calcBandwidthForConnection (process, i);
    }
  }
}

#endif /* DYNAMIC_GRID */

#endif /* PARALLEL_GRID */
