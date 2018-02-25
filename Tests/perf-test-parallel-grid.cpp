#include <iostream>

#include "Assert.h"
#include "Settings.h"

#include <ctime>
#include <sys/time.h>

#include <unistd.h>

#ifdef PARALLEL_GRID

#include "ParallelGrid.h"
#include "ParallelYeeGridLayout.h"
#include <mpi.h>

#ifndef CXX11_ENABLED
#include "cstdlib"
#endif /* !CXX11_ENABLED */

int main (int argc, char** argv)
{
#if defined (GRID_1D)
  grid_coord gridSizeX = STOI (argv[1]);
  time_step NUM_TIME_STEPS = STOI (argv[2]);
  time_step SHARE_TIME_STEP = STOI (argv[3]);

  int bufSize = 1;

  int res = MPI_Init (&argc, &argv);
  ALWAYS_ASSERT (res == MPI_SUCCESS);

  int rank, numProcs;

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &numProcs);

  printf ("X: PID %d of %d, grid size x: " COORD_MOD "\n", rank, numProcs, gridSizeX);

  printf ("Start process %d of %d\n", rank, numProcs);

  GridCoordinate1D overallSize (gridSizeX, CoordinateType::X);
  GridCoordinate1D pmlSize (2, CoordinateType::X);
  GridCoordinate1D tfsfSize (4, CoordinateType::X);
  GridCoordinate1D bufferSize (bufSize, CoordinateType::X);
  GridCoordinate1D topologySize (0, CoordinateType::X);

#define SCHEME_TYPE SchemeType::Dim1_EzHy
#define ANGLES PhysicsConst::Pi / 2, 0, PhysicsConst::Pi / 2

  ParallelGridCore parallelGridCore (rank, numProcs, overallSize, false, topologySize);
  ParallelGrid::initializeParallelCore (&parallelGridCore);

  bool isDoubleMaterialPrecision = false;

  ParallelYeeGridLayout<SCHEME_TYPE, LayoutType::E_CENTERED> yeeLayout (overallSize, pmlSize, tfsfSize, ANGLES, isDoubleMaterialPrecision);
  yeeLayout.Initialize (&parallelGridCore);

  ParallelGrid grid (overallSize, bufferSize, 0, yeeLayout.getSizeForCurNode ());

  for (grid_coord i = 0; i < grid.getSize ().get1 (); ++i)
  {
    FieldPointValue* val = new FieldPointValue ();

#ifdef COMPLEX_FIELD_VALUES
    val->setCurValue (FieldValue (i * PhysicsConst::Eps0, i * PhysicsConst::SpeedOfLight));
#else
    val->setCurValue (FieldValue (i * PhysicsConst::Eps0));
#endif

    GridCoordinate1D pos (i, CoordinateType::X);
    //GridCoordinate1D posAbs = grid.getTotalPosition (pos);
    grid.setFieldPointValue (val, pos);
  }

  struct timeval tv1, tv2;
  gettimeofday(&tv1, NULL);

  for (time_step t = 0; t < NUM_TIME_STEPS; ++t)
  {
    int state = 1;
#ifdef DYNAMIC_GRID
    state = ParallelGrid::getParallelCore ()->getNodeState ()[ParallelGrid::getParallelCore ()->getProcessId ()];
#endif

    if (state)
    {
#ifdef DYNAMIC_GRID
      ParallelGrid::getParallelCore ()->StartCalcClock ();
#endif

      // if (ParallelGrid::getParallelCore ()->getProcessId () == ParallelGrid::getParallelCore ()->getTotalProcCount () - 1)
      // {
      //   usleep(200000);
      // }
      for (grid_coord i = 0; i < grid.getSize ().get1 (); ++i)
      {
        GridCoordinate1D pos (i, CoordinateType::X);
        //GridCoordinate1D posAbs = grid.getTotalPosition (pos);

        FieldPointValue *value = grid.getFieldPointValue (pos);

        FPValue arg = 10000 * t * 0.01 - 2 * PhysicsConst::Pi / 0.2;
#ifdef COMPLEX_FIELD_VALUES
        FieldValue val (sin (arg), cos (arg));
#else
        FieldValue val (sin (arg));
#endif

        value->setCurValue (val);
      }

#ifdef DYNAMIC_GRID
      ParallelGrid::getParallelCore ()->StopCalcClock ();
#endif
    }

    grid.nextTimeStep ();

#ifdef DYNAMIC_GRID
    if (t > 0 && t % SHARE_TIME_STEP == 0)
    {
      if (ParallelGrid::getParallelCore ()->getProcessId () == 0)
      {
        printf ("Try rebalance on step %u, steps elapsed after previous %u\n", t, SHARE_TIME_STEP);
      }

      if (yeeLayout.Rebalance (SHARE_TIME_STEP))
      {
        //printf ("Rebalancing for process %d! (size %u)\n", ParallelGrid::getParallelCore ()->getProcessId (), yeeLayout.getEpsSizeForCurNode ().get1 ());

        grid.Resize (yeeLayout.getEpsSizeForCurNode ());
      }
    }
#endif
  }

  gettimeofday(&tv2, NULL);

  MPI_Finalize();

  printf ("Total time = %f seconds\n",
       (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
       (double) (tv2.tv_sec - tv1.tv_sec));

  return 0;
#endif
} /* main */

#else /* PARALLEL_GRID */

int main (int argc, char** argv)
{
  ASSERT (0);

  return 0;
} /* main */

#endif /* !PARALLEL_GRID */
