#include <iostream>

#if defined (PARALLEL_GRID)
#include "ParallelGrid.h"
#include <mpi.h>
#else /* PARALLEL_GRID */
#include "Grid.h"
#endif /* !PARALLEL_GRID */

#include "BMPDumper.h"
#include "BMPLoader.h"
#include "DATDumper.h"
#include "DATLoader.h"

#include "SchemeTMz.h"
#include "Scheme3D.h"

int main (int argc, char** argv)
{
  int totalTimeSteps = 100;
  int gridSize = 100;
  int bufSize = 10;

  if (argc != 4)
  {
    return 1;
  }
  else
  {
    totalTimeSteps = std::stoi (argv[1]);
    gridSize = std::stoi (argv[2]);
    bufSize = std::stoi (argv[3]);
  }

#if defined (PARALLEL_GRID)
  MPI_Init(&argc, &argv);

  int rank, numProcs;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

#if PRINT_MESSAGE
  printf ("Start process %d of %d\n", rank, numProcs);
#endif /* PRINT_MESSAGE */
#endif /* PARALLEL_GRID */

#if defined (PARALLEL_GRID)
  ParallelGridCoordinate overallSize (gridSize);
  ParallelGridCoordinate bufferLeft (bufSize);
  ParallelGridCoordinate bufferRight (bufSize);

#ifdef GRID_2D
  SchemeTMz scheme (overallSize, bufferLeft, bufferRight, rank, numProcs, totalTimeSteps);
#endif
#ifdef GRID_3D
  Scheme3D scheme (overallSize, bufferLeft, bufferRight, rank, numProcs, totalTimeSteps);
#endif
#else
#ifdef GRID_2D
  GridCoordinate2D overallSize (gridSize);

  SchemeTMz scheme (overallSize, totalTimeSteps);
#endif
#ifdef GRID_3D
  GridCoordinate3D overallSize (gridSize);

  Scheme3D scheme (overallSize, totalTimeSteps);
#endif
#endif

  scheme.initScheme (0.000003, 20);

#if defined (PARALLEL_GRID)
  scheme.initProcess (rank);
#endif

  scheme.initGrids ();

  scheme.performSteps ();

#if defined (PARALLEL_GRID)
#if PRINT_MESSAGE
  printf ("Main process %d.\n", rank);
#endif

  MPI_Finalize();
#endif

  return 0;
}
