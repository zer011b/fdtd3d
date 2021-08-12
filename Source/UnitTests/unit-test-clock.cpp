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

/*
 * Unit test for basic operations with Clock
 */

#include <iostream>

#include "Assert.h"
#include "Clock.h"

#ifdef PARALLEL_GRID
#include <mpi.h>
#endif /* PARALLEL_GRID */

#ifndef DEBUG_INFO
#error Test requires debug info
#endif /* !DEBUG_INFO */

int main (int argc, char** argv)
{
#ifdef MPI_CLOCK
  printf ("MPI_Wtime clock\n");
#else /* MPI_CLOCK */
  printf ("clock_gettime clock\n");
#endif /* !MPI_CLOCK */

#ifdef PARALLEL_GRID
  int res = MPI_Init (&argc, &argv);
  ALWAYS_ASSERT (res == MPI_SUCCESS);

  int rank, numProcs;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &numProcs);
#endif /* PARALLEL_GRID */

  Clock clock_1;
  ASSERT (clock_1.isZero ());

  Clock clock_2;
  ASSERT ((clock_1 + clock_2).isZero ());

  Clock clock_3 = Clock::getNewClock ();
  clock_3.print ();
  clock_2.setVal (clock_3.getVal ());
  ASSERT (!clock_3.isZero ());
  ASSERT ((clock_3 + clock_1).getFP () == clock_3.getFP ());
  ASSERT ((clock_3 - clock_1).getFP () == clock_3.getFP ());
  ASSERT (clock_3 >= clock_3);
  ASSERT (clock_3 > clock_1);
  ASSERT (clock_1 < clock_3);
  ASSERT (clock_1 <= clock_3);

#ifdef MPI_CLOCK
  ASSERT (clock_3.getVal () == clock_3.getFP ());
#else /* MPI_CLOCK */
  ASSERT ((DOUBLE)((clock_3.getVal ().tv_sec + clock_3.getVal ().tv_nsec / 1000000000.0)) == clock_3.getFP ());
#endif /* !MPI_CLOCK */

  ASSERT ((clock_3 + clock_1).getFP () == clock_3.getFP ());
  ASSERT ((clock_3 - clock_1).getFP () == clock_3.getFP ());
  ASSERT (clock_2.getFP () == clock_3.getFP ());

  /*
   * Loop to sleep for a bit
   */
  for (int i = 0; i < 1000; ++i)
  for (int i = 0; i < 1000; ++i)
  {}

  Clock clock_4 = Clock::getNewClock ();
  clock_4.print ();
  ASSERT (!clock_4.isZero ());
  ASSERT (clock_4 >= clock_3);
  ASSERT (clock_4 > clock_3);
  ASSERT (clock_3 <= clock_4);
  ASSERT (clock_3 < clock_4);
  ASSERT (clock_4.getFP () >= clock_3.getFP ());

  Clock clock_5 = clock_4 - clock_3;
  Clock clock_6 = clock_4 + clock_3;

  /*
   * FP values can't be compared in the next manner due to not enough accuracy of fp value
   *
   * ASSERT (clock_5.getFP () == clock_4.getFP () - clock_3.getFP ());
   * ASSERT (clock_6.getFP () == clock_4.getFP () + clock_3.getFP ());
  */
  ASSERT (!clock_5.isZero ());
  ASSERT (!clock_6.isZero ());
  ASSERT (clock_5 > clock_1);
  ASSERT (clock_5 >= clock_1);
  ASSERT (clock_6 > clock_1);
  ASSERT (clock_6 >= clock_1);
  ASSERT (clock_5 < clock_4);
  ASSERT (clock_5 <= clock_4);
  ASSERT (clock_6 > clock_4);
  ASSERT (clock_6 >= clock_4);
  ASSERT (clock_6 > clock_3);
  ASSERT (clock_6 >= clock_3);

#ifdef MPI_CLOCK
  ASSERT (clock_5.getVal () == clock_4.getVal () - clock_3.getVal ());
  ASSERT (clock_6.getVal () == clock_4.getVal () + clock_3.getVal ());
#else /* MPI_CLOCK */
  ClockValue val3 = clock_3.getVal ();
  ClockValue val4 = clock_4.getVal ();
  ClockValue val5 = clock_5.getVal ();
  ClockValue val6 = clock_6.getVal ();

  if ((val4.tv_nsec - val3.tv_nsec) < 0)
  {
    ASSERT (val5.tv_sec == val4.tv_sec - val3.tv_sec - 1);
    ASSERT (val5.tv_nsec == val4.tv_nsec - val3.tv_nsec + 1000000000);
  }
  else
  {
    ASSERT (val5.tv_sec == val4.tv_sec - val3.tv_sec);
    ASSERT (val5.tv_nsec == val4.tv_nsec - val3.tv_nsec);
  }

  if ((val3.tv_nsec + val4.tv_nsec) > 1000000000)
  {
    ASSERT (val6.tv_sec == val3.tv_sec + val4.tv_sec + 1);
    ASSERT (val6.tv_nsec == val3.tv_nsec + val4.tv_nsec - 1000000000);
  }
  else
  {
    ASSERT (val6.tv_sec == val3.tv_sec + val4.tv_sec);
    ASSERT (val6.tv_nsec == val3.tv_nsec + val4.tv_nsec);
  }
#endif /* !MPI_CLOCK */

  Clock clock_7 = Clock::average (clock_3, clock_4);
#ifdef MPI_CLOCK
  ASSERT (clock_7.getVal () == (clock_3.getVal () + clock_4.getVal ()) / DOUBLE (2));
#else /* MPI_CLOCK */
  ClockValue val7 = clock_7.getVal ();
  ASSERT (val7.tv_sec == (val3.tv_sec + val4.tv_sec) / 2);
  ASSERT (val7.tv_nsec == (val3.tv_nsec + val4.tv_nsec) / 2);
#endif /* !MPI_CLOCK */

  /*
   * Manually creted clocks
   */
  Clock clock_8;
  Clock clock_9;
#ifdef MPI_CLOCK
  ClockValue val8 = 1234.5678;
  ClockValue val9 = 4234.5878;
#else /* MPI_CLOCK */
  ClockValue val8;
  ClockValue val9;
  val8.tv_sec = 1234;
  val8.tv_nsec = 567800000;
  val9.tv_sec = 4234;
  val9.tv_nsec = 587800000;
#endif /* !MPI_CLOCK */

  clock_8.setVal (val8);
  clock_9.setVal (val9);

#ifdef MPI_CLOCK
  ASSERT ((clock_9 - clock_8).getVal () == val9 - val8);
#else /* MPI_CLOCK */
  Clock diff89 = clock_9 - clock_8;
  ASSERT (diff89.getVal ().tv_sec == 3000);
  ASSERT (diff89.getVal ().tv_nsec == 20000000);
#endif /* !MPI_CLOCK */

#ifdef PARALLEL_GRID
  MPI_Finalize ();
#endif /* PARALLEL_GRID */

  return 0;
} /* main */
